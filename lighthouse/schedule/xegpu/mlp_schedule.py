from collections import namedtuple

from mlir import ir
from mlir.dialects.transform import loop
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import xegpu
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects import transform
from mlir.dialects.transform import structured
from lighthouse.utils.mlir import (
    apply_registered_pass,
    canonicalize,
    match,
)

from lighthouse.dialects import smt_ext, transform_smt_ext as td_smt_ext
from lighthouse.dialects.transform_tune_ext import knob, KnobValue


class PipelineInterrupt(Exception):
    """Exception to signal early termination of the transform schedule."""

    pass


def match_and_split(*args, nhandles=1, **kwargs):
    """Helper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops


# hardware constraints
DPAS = namedtuple("DPAS", ["M", "N", "K", "A_TILE", "B_TILE", "C_TILE"])(
    8, 16, 16, (8, 16), (16, 16), (8, 16)
)
PREFETCH_INST_DATA = [8, 16]
NB_WORKITEMS = 16  # workitems in subgroup
LOAD_TILE_SIZES = [8, 16, 32]


@KnobValue.ast_rewrite(in_exprs=True)
def checked_params_or_knobs(
    params: dict[str, int | None], layer_id=""
) -> dict[str, int | KnobValue]:
    """Check the parameters for validity and replace `None`s with knobs with asserted ranges."""
    m, n, k = params["m"], params["n"], params["k"]
    assert isinstance(m, int) and isinstance(n, int) and isinstance(k, int)
    assert m > 0 and n > 0 and k > 0
    wg_m = params["wg_m"] or knob(layer_id + "wg_m")
    wg_n = params["wg_n"] or knob(layer_id + "wg_n")
    sg_m = params["sg_m"] or knob(layer_id + "sg_m")
    sg_n = params["sg_n"] or knob(layer_id + "sg_n")
    k_tile = params["k_tile"] or knob(layer_id + "k_tile")
    load_a_m = params["load_a_m"] or knob(layer_id + "load_a_m")
    load_a_k = params["load_a_k"] or knob(layer_id + "load_a_k")
    load_b_k = params["load_b_k"] or knob(layer_id + "load_b_k")
    load_b_n = params["load_b_n"] or knob(layer_id + "load_b_n")
    prefetch_a_m = params["prefetch_a_m"] or knob(layer_id + "prefetch_a_m")
    prefetch_a_k = params["prefetch_a_k"] or knob(layer_id + "prefetch_a_k")
    prefetch_b_k = params["prefetch_b_k"] or knob(layer_id + "prefetch_b_k")
    prefetch_b_n = params["prefetch_b_n"] or knob(layer_id + "prefetch_b_n")

    # NB: Constraints on knobs will be added as attributes on the KnobOps, while
    #     constraints on concrete values will be checked immediately.
    assert 64 <= wg_m <= 256 and m % wg_m == 0 and wg_m % DPAS.M == 0
    assert 64 <= wg_n <= 256 and n % wg_n == 0 and wg_n % DPAS.N == 0
    assert 32 <= sg_m <= 128 and m % sg_m == 0 and sg_m % DPAS.M == 0
    assert 32 <= sg_n <= 128 and n % sg_n == 0 and sg_n % DPAS.N == 0
    assert 16 <= k_tile <= 50 and k % k_tile == 0 and k_tile % DPAS.K == 0
    assert load_a_m in LOAD_TILE_SIZES and load_a_m % DPAS.M == 0
    assert load_a_k in LOAD_TILE_SIZES and load_a_k % DPAS.K == 0
    assert load_b_k in LOAD_TILE_SIZES and load_b_k % DPAS.K == 0
    assert load_b_n in LOAD_TILE_SIZES and load_b_n % DPAS.N == 0
    assert prefetch_a_m in LOAD_TILE_SIZES
    assert prefetch_a_k in LOAD_TILE_SIZES
    assert prefetch_b_k in LOAD_TILE_SIZES
    assert prefetch_b_n in LOAD_TILE_SIZES

    return {
        "wg_m": wg_m,
        "wg_n": wg_n,
        "sg_m": sg_m,
        "sg_n": sg_n,
        "k_tile": k_tile,
        "load_a_m": load_a_m,
        "load_a_k": load_a_k,
        "load_b_k": load_b_k,
        "load_b_n": load_b_n,
        "prefetch_a_m": prefetch_a_m,
        "prefetch_a_k": prefetch_a_k,
        "prefetch_b_k": prefetch_b_k,
        "prefetch_b_n": prefetch_b_n,
    }


def get_schedule_module(
    params: list[dict[str, int | None]],
    has_bias: bool = False,
    has_relu: bool = False,
    has_convert_c: bool = True,
    skip_final_layer_relu: bool = False,
    stop_at_stage: str = "",
) -> ir.Module:
    """Generate transform schedule module."""
    mod = ir.Module.create()
    mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(mod.body):
        named_sequence = transform.named_sequence(
            "__transform_main",
            [transform.AnyOpType.get()],  # input types
            [],  # output types
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            # match the payload module
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            payload_mod = transform.get_parent_op(
                anytype,
                func,
                op_name="builtin.module",
                deduplicate=True,
            )
            for i, layer_params in enumerate(params):
                layer_params |= checked_params_or_knobs(
                    layer_params, layer_id=f"layer_{i}_"
                )

            xegpu_mlp_transform_schedule(
                payload_mod,
                params=params,
                has_bias=has_bias,
                has_relu=has_relu,
                has_convert_c=has_convert_c,
                skip_final_layer_relu=skip_final_layer_relu,
                stop_at_stage=stop_at_stage,
            )

    return mod


def xegpu_mlp_transform_schedule(
    mod: ir.Value[transform.AnyOpType],
    params: list[dict[str, int | KnobValue]],
    has_bias: bool = False,
    has_relu: bool = False,
    has_convert_c: bool = True,
    skip_final_layer_relu: bool = False,
    stop_at_stage: str = "",
):
    """Transform schedule for MLP-like payload."""
    try:
        mod = bundle_xegpu_mlp_schedule(
            mod,
            params=params,
            has_bias=has_bias,
            has_relu=has_relu,
            has_convert_c=has_convert_c,
            skip_final_layer_relu=skip_final_layer_relu,
            stop_at_stage=stop_at_stage,
        )

        mod = bundle_xegpu_to_binary(
            mod,
            stop_at_stage=stop_at_stage,
        )
    except PipelineInterrupt:
        pass
    finally:
        transform.yield_()


def bundle_xegpu_mlp_schedule(
    mod: ir.Value[transform.AnyOpType],
    params: list[dict[str, int | KnobValue]],
    has_bias: bool = False,
    has_relu: bool = False,
    skip_final_layer_relu: bool = False,
    has_convert_c: bool = True,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering MLP-like payload to xegpu wg level."""
    nlayers = len(params)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # wg tiling
    if has_convert_c:
        trunc_op = match(mod, ops={"arith.truncf"})
        terminal = transform.get_parent_op(anytype, trunc_op)
        # split handle for each layer
        terminal_ops = transform.split_handle((anytype,) * nlayers, terminal)
        if nlayers == 1:
            terminal_ops = [terminal_ops]
    elif has_bias:
        terminal_ops = match_and_split(mod, ops={"linalg.add"}, nhandles=nlayers)
    else:
        terminal_ops = match_and_split(mod, ops={"linalg.matmul"}, nhandles=nlayers)
    if has_relu:
        if not skip_final_layer_relu:
            # relu on all layers
            terminal_ops = match_and_split(mod, ops={"linalg.max"}, nhandles=nlayers)
        elif nlayers > 1:
            # intermediate layers have relu activation function
            relu_ops = match_and_split(mod, ops={"linalg.max"}, nhandles=nlayers - 1)
            # the final layer does not have relu
            terminal_ops = list(relu_ops) + [terminal_ops[-1]]

    # tile each layer separately
    for terminal_op, layer_params in zip(terminal_ops, params):
        # tunable parameters: wg level tiling
        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        k_tile = layer_params["k_tile"]

        # FIXME: use structured.structured_fuse
        _, wg_loop = structured.FuseOp(
            terminal_op, tile_sizes=wg_tile, use_forall=True
        ).results
        transform.apply_cse(mod)
        canonicalize(mod)

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        # FIXME use structured.structured_tile_using_for
        wgk_matmul, k_loop = structured.TileUsingForOp(
            wg_matmul, sizes=[0, 0, k_tile]
        ).results

    func = transform.get_parent_op(
        anytype,
        k_loop,
        op_name="func.func",
        deduplicate=True,
    )
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize
    # FIXME use structured.structured_vectorize_children_and_apply_patterns
    func = structured.VectorizeChildrenAndApplyPatternsOp(
        func,
        fold_type_extensions_into_contract=True,
    ).result

    # hoist loop invariant vector read/store ops
    k_loop = match(func, ops={"scf.for"})
    loop.HoistLoopInvariantSubsetsOp(k_loop)

    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize

    # eliminate empty tensors to avoid emitting extra copy ops
    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    identity_layout = LayoutMapOption.IdentityLayoutMap
    mod = bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=True,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=identity_layout,
    ).result
    # fold memref.subviews into vector.transfer_read/write ops
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # convert forall to parallel
    wg_loops = match_and_split(mod, ops={"scf.forall"}, nhandles=nlayers)
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # convert to scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # set correct number of gpu threads
    launch_ops = match_and_split(mod, ops={"gpu.launch"}, nhandles=nlayers)
    assert len(launch_ops) == nlayers
    for launch_op, layer_params in zip(launch_ops, params):
        # tunable parameters
        wg_m, wg_n = layer_params["wg_m"], layer_params["wg_n"]
        sg_m, sg_n = layer_params["sg_m"], layer_params["sg_n"]

        @td_smt_ext.constrain_params(wg_m, wg_n, sg_m, sg_n)
        def constrain_wg_sg_and_calc_nb_threads(
            WG_M: int | smt_ext.SMTIntValue,
            WG_N: int | smt_ext.SMTIntValue,
            SG_M: int | smt_ext.SMTIntValue,
            SG_N: int | smt_ext.SMTIntValue,
        ):
            # NB: normal asserts in case of concrete values, SMT assert ops for symbolic values.
            smt_ext.assert_(WG_M % SG_M == 0)
            smt_ext.assert_(WG_N % SG_N == 0)

            # NB: normal ints in case of concrete values, SMT int values for symbolic values.
            sg_m_threads = WG_M // SG_M
            sg_n_threads = WG_N // SG_N
            sg_threads = sg_m_threads * sg_n_threads
            smt_ext.assert_(sg_threads <= 64)

            # number of threads collapsed to 1d layout
            return sg_threads * NB_WORKITEMS

        nb_threads: int | smt_ext.SMTIntValue = (
            constrain_wg_sg_and_calc_nb_threads.results
        )

        xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])

    # outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    # set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # convert vector to xegpu
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"}, nhandles=nlayers)
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    assert (
        len(gpu_mod_ops) == nlayers
    ), "Expected one gpu.module per MLP layer after outlining"
    for gpu_mod, layer_params in zip(gpu_mod_ops, params):
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        xegpu_wg_annotation_for_mlp_layer(gpu_func, **layer_params, has_bias=has_bias)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod


def xegpu_wg_annotation_for_mlp_layer(
    gpu_func: ir.Value,
    *,
    wg_m: int | KnobValue,
    wg_n: int | KnobValue,
    sg_m: int | KnobValue,
    sg_n: int | KnobValue,
    k_tile: int | KnobValue,
    load_a_m: int | KnobValue,
    load_a_k: int | KnobValue,
    load_b_k: int | KnobValue,
    load_b_n: int | KnobValue,
    prefetch_a_m: int | KnobValue,
    prefetch_a_k: int | KnobValue,
    prefetch_b_k: int | KnobValue,
    prefetch_b_n: int | KnobValue,
    nb_prefetch: int,
    has_bias: bool,
    **_catch_all,
):
    """
    Adds prefetching and XeGPU anchor layout annotations for an MLP layer.

    Should be applied after the payload has been converted to XeGPU using
    the convert-vector-to-xegpu pass.
    """

    anytype = transform.AnyOpType.get()
    anyvalue = transform.AnyValueType.get()

    # Calculate with SMT ops in case of symbolic values, normal ints in case of concrete values.
    @td_smt_ext.constrain_params(wg_m, wg_n, sg_m, sg_n)
    def calc_sg_layout(WG_M, WG_N, SG_M, SG_N):
        return WG_M // SG_M, WG_N // SG_N

    sg_layout = calc_sg_layout.results

    load_tile_a = [load_a_m, load_a_k]
    load_tile_b = [load_b_k, load_b_n]
    prefetch_tile_a = [prefetch_a_m, prefetch_a_k]
    prefetch_tile_b = [prefetch_b_k, prefetch_b_n]

    @td_smt_ext.constrain_params(
        sg_m,
        sg_n,
        k_tile,
        load_a_m,
        load_a_k,
        load_b_k,
        load_b_n,
        prefetch_a_m,
        prefetch_a_k,
        prefetch_b_k,
        prefetch_b_n,
    )
    def constrain_and_calculate_load_and_prefetch_params(
        SG_M, SG_N, K_TILE, LDA_M, LDA_K, LDB_K, LDB_N, PFA_M, PFA_K, PFB_K, PFB_N
    ):
        # NB: normal asserts in case of concrete values, SMT assert ops for symbolic values
        # TODO: Tuomas' comments explaining constraints:
        smt_ext.assert_(SG_M % PFA_M == 0)
        smt_ext.assert_(SG_M % LDA_M == 0)

        smt_ext.assert_(SG_N % PFB_N == 0)
        smt_ext.assert_(SG_N % LDB_N == 0)
        smt_ext.assert_(K_TILE % PFA_K == 0)
        smt_ext.assert_(K_TILE % PFB_K == 0)
        smt_ext.assert_(K_TILE % LDA_K == 0)
        smt_ext.assert_(K_TILE % LDB_K == 0)

        smt_ext.assert_(LDA_M * LDA_K >= 16 * 16)
        smt_ext.assert_(LDB_K * LDB_N >= 16 * 16)

        smt_ext.assert_(LDA_M <= LDA_K)
        smt_ext.assert_(LDB_K <= LDB_N)
        smt_ext.assert_(LDB_N == DPAS.N)

        PFA_M_step = SG_M // PFA_M
        PFA_K_step = K_TILE // PFA_K
        smt_ext.assert_(PFA_M_step * PFA_K_step <= 64)

        PFB_K_step = K_TILE // PFB_K
        PFB_N_step = SG_N // PFB_N
        smt_ext.assert_(PFB_K_step * PFB_N_step <= 64)

        smt_ext.assert_(PFA_M * PFA_K >= 16 * 16)
        smt_ext.assert_(PFA_M >= PFA_K)

        smt_ext.assert_(PFB_K * PFB_N >= 16 * 16)
        smt_ext.assert_(PFB_K >= PFB_N)
        smt_ext.assert_((SG_M // DPAS.M) * (SG_N // DPAS.N) * (K_TILE // DPAS.K) <= 64)

        return PFA_M_step, PFA_K_step, PFB_K_step, PFB_N_step

    prefetch_layout_a = constrain_and_calculate_load_and_prefetch_params.results[0:2]
    prefetch_layout_b = constrain_and_calculate_load_and_prefetch_params.results[2:4]

    # matmul matrix shapes
    sg_tile_a = [sg_m, k_tile]
    sg_tile_b = [k_tile, sg_n]

    # add layouts to DPAS op operands
    k_loop = match(gpu_func, ops={"scf.for"})
    dpas_op = match(k_loop, ops={"xegpu.dpas"})
    tile_a = transform.get_operand(anyvalue, dpas_op, [0])
    tile_b = transform.get_operand(anyvalue, dpas_op, [1])

    # insert prefetch ops for DPAS A and B tiles
    def add_prefetch(tile, nb_prefetch, **layout):
        desc_op = xegpu.insert_prefetch(
            tile,
            nb_prefetch=nb_prefetch,
        )
        pf_ops = transform.get_consumers_of_result(anytype, desc_op, 0)
        for pf in transform.split_handle((anytype,) * (nb_prefetch + 1), pf_ops):
            xegpu.set_op_layout_attr(pf, **layout)

    add_prefetch(
        tile_a,
        nb_prefetch,
        sg_layout=prefetch_layout_a,
        sg_data=prefetch_tile_a,
        inst_data=PREFETCH_INST_DATA,
    )
    add_prefetch(
        tile_b,
        nb_prefetch,
        sg_layout=prefetch_layout_b,
        sg_data=prefetch_tile_b,
        inst_data=PREFETCH_INST_DATA,
    )

    def annotate_ab_load(tile, layout_load, layout_dpas):
        desc_op = xegpu.get_desc_op(tile)
        load_op = transform.get_consumers_of_result(anytype, desc_op, 0)
        xegpu.set_op_layout_attr(load_op, **layout_load)
        xegpu.convert_layout(
            tile,
            input_sg_layout=layout_load["sg_layout"],
            input_sg_data=layout_load["sg_data"],
            input_inst_data=layout_load["inst_data"],
            target_sg_layout=layout_dpas["sg_layout"],
            target_sg_data=layout_dpas["sg_data"],
            target_inst_data=layout_dpas["inst_data"],
        )

    # A tile load layout
    layout_load_a = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_a,
        "inst_data": load_tile_a,
    }
    # A tile dpas layout
    layout_dpas_a = layout_load_a.copy()
    layout_dpas_a["inst_data"] = DPAS.A_TILE
    annotate_ab_load(tile_a, layout_load_a, layout_dpas_a)

    # B tile load layout
    layout_load_b = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_b,
        "inst_data": load_tile_b,
    }
    # B tile dpas layout
    layout_dpas_b = layout_load_b.copy()
    layout_dpas_b["inst_data"] = DPAS.B_TILE
    annotate_ab_load(tile_b, layout_load_b, layout_dpas_b)

    # C tile layout
    output_layout = {
        "sg_layout": sg_layout,
        "sg_data": [sg_m, sg_n],
        "inst_data": DPAS.C_TILE,
    }
    # C tile dpas anchor layout
    xegpu.set_op_layout_attr(dpas_op, index=0, **layout_dpas_a)
    xegpu.set_op_layout_attr(dpas_op, index=1, **layout_dpas_b)
    xegpu.set_op_layout_attr(dpas_op, index=2, **output_layout)
    # annotate store op
    store_op_c = match(gpu_func, ops={"xegpu.store_nd"})
    xegpu.set_op_layout_attr(store_op_c, **output_layout)

    if has_bias:
        # annotate the 1d load of the broadcast op with a slice layout
        add_op = match(gpu_func, ops={"arith.addf"})
        bcast_op = transform.get_producer_of_operand(anytype, add_op, 0)
        bcast_load = transform.get_producer_of_operand(anytype, bcast_op, 0)
        xegpu.set_op_layout_attr(
            bcast_load, result=True, index=0, **output_layout, slice_dims=[0]
        )
        raise NotImplementedError("Bias layout propagation is not supported.")

    transform.apply_cse(gpu_func)
    canonicalize(gpu_func)

    # hoist desc ops out of reduction loop
    transform.apply_licm(k_loop)

    canonicalize(gpu_func)
    transform.apply_cse(gpu_func)


def bundle_xegpu_to_binary(
    mod, stop_at_stage: str = ""
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering xegpu wg level to binary."""
    # upstream xegpu/xevm pipeline is payload independent.
    mod = apply_registered_pass(
        mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
    )

    return mod
