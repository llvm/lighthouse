from mlir import ir
from mlir.dialects.transform import xegpu
from mlir.dialects import transform
from mlir.dialects.transform import structured
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)

from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects import smt_ext
from lighthouse.dialects.transform import smt_ext as td_smt_ext
from lighthouse.dialects.transform.tune_ext import knob, KnobValue
from .xegpu_specs import XeGPUSpecs
from .xegpu_parameter_selector import XeGPUParameterSelector
from .lowering_common import (
    vectorize_bufferize_and_outline_gpu_func,
    convert_vector_to_xegpu,
)
from .matmul_constraints import (
    DPAS,
    PREFETCH_INST_DATA,
    LOAD_MAX_ROWS,
    LOAD_MAX_COLS,
    PFETCH_MIN_ROWS,
    PFETCH_MIN_COLS,
    PFETCH_MAX_ROWS,
    PFETCH_MAX_COLS,
    MIN_NB_THREADS,
)


@KnobValue.ast_rewrite(in_exprs=True)
def params_with_constraints_imposed(
    params: dict[str, int | None], knob_name_prefix=""
) -> dict[str, int | KnobValue]:
    """Check the parameters for validity and replace `None`s with knobs with asserted ranges.

    Inserts the `KnobOp`s for any knobs that are created at the active `InsertionPoint` and
    maps the parameter name to the (`Knob`)`Value` returned by the `KnobOp`."""
    m, n, k = params["m"], params["n"], params["k"]
    assert isinstance(m, int) and isinstance(n, int) and isinstance(k, int)
    assert m > 0 and n > 0 and k > 0
    wg_m = params["wg_m"] or knob(knob_name_prefix + "wg_m")
    wg_n = params["wg_n"] or knob(knob_name_prefix + "wg_n")
    sg_m = params["sg_m"] or knob(knob_name_prefix + "sg_m")
    sg_n = params["sg_n"] or knob(knob_name_prefix + "sg_n")
    k_tile = params["k_tile"] or knob(knob_name_prefix + "k_tile")
    load_a_m = params["load_a_m"] or knob(knob_name_prefix + "load_a_m")
    load_a_k = params["load_a_k"] or knob(knob_name_prefix + "load_a_k")
    load_b_k = params["load_b_k"] or knob(knob_name_prefix + "load_b_k")
    load_b_n = params["load_b_n"] or knob(knob_name_prefix + "load_b_n")
    prefetch_a_m = params["prefetch_a_m"] or knob(knob_name_prefix + "prefetch_a_m")
    prefetch_a_k = params["prefetch_a_k"] or knob(knob_name_prefix + "prefetch_a_k")
    prefetch_b_k = params["prefetch_b_k"] or knob(knob_name_prefix + "prefetch_b_k")
    prefetch_b_n = params["prefetch_b_n"] or knob(knob_name_prefix + "prefetch_b_n")
    prefetch_a_nb = params["prefetch_a_nb"] or knob(knob_name_prefix + "prefetch_a_nb")
    prefetch_b_nb = params["prefetch_b_nb"] or knob(knob_name_prefix + "prefetch_b_nb")

    # NB: Constraints on knobs will be added as attributes on the KnobOps, while
    #     constraints on concrete values will be checked immediately.
    assert min(max(m // 4, 16), 64) <= wg_m <= min(m, 256)
    assert m % wg_m == 0 and wg_m % DPAS.M == 0
    assert min(max(n // 4, 16), 64) <= wg_n <= min(n, 256)
    assert n % wg_n == 0 and wg_n % DPAS.N == 0
    assert 16 <= sg_m <= min(m, 128)
    assert m % sg_m == 0 and sg_m % DPAS.M == 0
    assert 16 <= sg_n <= min(n, 128)
    assert n % sg_n == 0 and sg_n % DPAS.N == 0
    assert 16 <= k_tile <= min(k, 256)
    assert k % k_tile == 0 and k_tile % DPAS.K == 0
    assert prefetch_a_nb > 0
    assert prefetch_b_nb > 0

    LOAD_TILE_SIZES = [8, 16, 32]
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


def mlp_schedule(
    params: list[dict[str, int | None]],
    stop_at_stage: str = "",
) -> ir.Module:
    """Generate transform schedule module for MLP payload."""
    assert params is not None and len(params) > 0, "params must be provided."
    devices = {p.get("device") for p in params if "device" in p}
    assert len(devices) <= 1, f"Multiple devices specified in params list: {devices}"
    device = devices.pop() if devices else None
    param_selector = XeGPUParameterSelector(device=device)
    gpu_specs = param_selector.gpu_specs

    with schedule_boilerplate() as (schedule, named_seq):
        # match the payload module
        anytype = transform.AnyOpType.get()
        func = match(named_seq.bodyTarget, ops={"func.func"})
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )
        # preprocess layer parameters
        for i, layer_params in enumerate(params):
            m = layer_params.get("m")
            n = layer_params.get("n")
            k = layer_params.get("k")
            assert all(d is not None for d in (m, n, k)), (
                "m, n, k must be provided in params"
            )

            required_params = [
                "wg_m",
                "wg_n",
                "sg_m",
                "sg_n",
                "k_tile",
                "load_a_m",
                "load_a_k",
                "load_b_k",
                "load_b_n",
                "prefetch_a_m",
                "prefetch_a_k",
                "prefetch_b_k",
                "prefetch_b_n",
                "prefetch_a_nb",
                "prefetch_b_nb",
            ]
            if not all(p in layer_params for p in required_params):
                # Some parameters are missing, use the parameter selector to fill
                # NOTE None values are interpreted as knobs in the constraint function
                shape = (m, n, k)
                transpose_a = layer_params.get("transpose_a", False)
                transpose_b = layer_params.get("transpose_b", False)
                generated_params = param_selector.get_parameters(
                    shape, transpose_a, transpose_b
                )
                # Overwrite original params to ensure consistent configuration
                layer_params.update(generated_params)

            layer_params |= params_with_constraints_imposed(
                layer_params, knob_name_prefix=f"layer_{i}_"
            )

        try:
            bundle_xegpu_mlp_schedule(
                payload_mod,
                gpu_specs=gpu_specs,
                params=params,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_mlp_schedule(
    mod: ir.Value[transform.AnyOpType],
    gpu_specs: XeGPUSpecs,
    params: list[dict[str, int | KnobValue]],
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering MLP-like payload to xegpu wg level."""
    nlayers = len(params)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # fuse all elementwise ops first
    mod = apply_registered_pass(mod, "linalg-fuse-elementwise-ops")

    # tile each layer separately
    matmul_ops = match_and_split(mod, ops={"linalg.matmul"}, nhandles=nlayers)
    for matmul_op, layer_params in zip(matmul_ops, params):
        # tunable parameters: wg and k tiling
        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        k_tile = layer_params["k_tile"]

        # find the last tileable consumer of the matmul
        consumers = transform_ext.get_tileable_consumers(matmul_op)
        leaf_consumer_op = transform_ext.extract_handle(consumers, -1)

        # wg tiling
        _, [wg_loop], _ = lh_transform.tile(
            leaf_consumer_op,
            tile_sizes=wg_tile,
            fuse_producers=True,
            use_forall=True,
            apply_cleanup=False,
        )

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        _, [k_loop], _ = lh_transform.tile(wg_matmul, tile_sizes=[0, 0, k_tile])
        lh_transform.cleanup(wg_loop)
        # if there's a transpose op fuse it into the k loop
        transpose_op = match(wg_loop, ops={"linalg.transpose"})
        structured.structured_fuse_into_containing_op(
            anytype, anytype, transpose_op, k_loop
        )

    func = transform.get_parent_op(
        anytype,
        k_loop,
        op_name="func.func",
        deduplicate=True,
    )
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    mod = vectorize_bufferize_and_outline_gpu_func(
        mod,
        func,
        nlayers=nlayers,
        gpu_specs=gpu_specs,
        params=params,
        stop_at_stage=stop_at_stage,
    )
    mod = convert_vector_to_xegpu(mod, nlayers=nlayers)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"}, nhandles=nlayers)
    for gpu_mod, layer_params in zip(gpu_mod_ops, params):
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        xegpu_wg_annotation_for_mlp_layer(gpu_func, gpu_specs=gpu_specs, **layer_params)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod


def xegpu_wg_annotation_for_mlp_layer(
    gpu_func: ir.Value,
    gpu_specs: XeGPUSpecs,
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
    prefetch_a_nb: int | KnobValue,
    prefetch_b_nb: int | KnobValue,
    transpose_a: bool,
    transpose_b: bool,
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
        # NB: Constraint on overall num SG threads already dealt with elsewhere.
        return WG_M // SG_M, WG_N // SG_N

    sg_layout = calc_sg_layout.results

    load_tile_a = [load_a_m, load_a_k]
    load_tile_b = [load_b_k, load_b_n]
    prefetch_tile_a = [prefetch_a_m, prefetch_a_k]
    prefetch_tile_b = [prefetch_b_k, prefetch_b_n]

    @td_smt_ext.constrain_params(
        wg_m,
        wg_n,
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
        transpose_a,
        transpose_b,
    )
    def constrain_and_calculate_load_and_prefetch_params(
        WG_M,
        WG_N,
        SG_M,
        SG_N,
        K_TILE,
        LDA_M,
        LDA_K,
        LDB_K,
        LDB_N,
        PFA_M,
        PFA_K,
        PFB_K,
        PFB_N,
        TR_A,
        TR_B,
    ):
        # NB: normal asserts in case of concrete values, SMT assert ops for symbolic values
        smt_ext.assert_(SG_M % LDA_M == 0)
        smt_ext.assert_(K_TILE % LDA_K == 0)
        smt_ext.assert_(K_TILE % LDB_K == 0)
        smt_ext.assert_(SG_N % LDB_N == 0)

        smt_ext.assert_(LDA_M <= LOAD_MAX_ROWS)
        smt_ext.assert_(LDA_K <= LOAD_MAX_COLS)
        smt_ext.assert_(LDB_K <= LOAD_MAX_ROWS)
        smt_ext.assert_(LDB_N <= LOAD_MAX_COLS)

        # prefetch tile shape depends on transpose flag
        pf_shape_a = (K_TILE, WG_M) if TR_A else (WG_M, K_TILE)
        pf_shape_b = (WG_N, K_TILE) if TR_B else (K_TILE, WG_N)

        smt_ext.assert_(pf_shape_a[0] % PFA_M == 0)
        smt_ext.assert_(pf_shape_a[1] % PFA_K == 0)
        smt_ext.assert_(pf_shape_b[0] % PFB_K == 0)
        smt_ext.assert_(pf_shape_b[1] % PFB_N == 0)

        smt_ext.assert_(PFA_M <= PFETCH_MAX_ROWS)
        smt_ext.assert_(PFA_K <= PFETCH_MAX_COLS)
        smt_ext.assert_(PFA_M >= PFETCH_MIN_ROWS)
        smt_ext.assert_(PFA_K >= PFETCH_MIN_COLS)

        smt_ext.assert_(PFB_K <= PFETCH_MAX_ROWS)
        smt_ext.assert_(PFB_N <= PFETCH_MAX_COLS)
        smt_ext.assert_(PFB_K >= PFETCH_MIN_ROWS)
        smt_ext.assert_(PFB_N >= PFETCH_MIN_COLS)

        smt_ext.assert_(LDA_M % DPAS.M == 0)
        smt_ext.assert_(LDA_K % DPAS.K == 0)
        smt_ext.assert_(LDB_K % DPAS.K == 0)
        smt_ext.assert_(LDB_N % DPAS.N == 0)

        # prefetch A thread layout
        prefetch_th_a_m = pf_shape_a[0] // PFA_M
        prefetch_th_a_k = pf_shape_a[1] // PFA_K

        prefetch_th_a = prefetch_th_a_m * prefetch_th_a_k
        smt_ext.assert_(prefetch_th_a <= gpu_specs.max_nb_threads)
        smt_ext.assert_(prefetch_th_a_m * prefetch_th_a_k >= MIN_NB_THREADS)

        # prefetch B thread layout
        prefetch_th_b_k = pf_shape_b[0] // PFB_K
        prefetch_th_b_n = pf_shape_b[1] // PFB_N
        prefetch_th_b = prefetch_th_b_k * prefetch_th_b_n
        smt_ext.assert_(prefetch_th_b <= gpu_specs.max_nb_threads)
        if isinstance(prefetch_th_b, smt_ext.SMTIntValue):
            # NB: Constraint only enabled during tuning.
            smt_ext.assert_(prefetch_th_b_k * prefetch_th_b_n >= MIN_NB_THREADS)

        return prefetch_th_a_m, prefetch_th_a_k, prefetch_th_b_k, prefetch_th_b_n

    prefetch_layout_a = constrain_and_calculate_load_and_prefetch_params.results[0:2]
    prefetch_layout_b = constrain_and_calculate_load_and_prefetch_params.results[2:4]

    # matmul matrix shapes
    sg_tile_a = [sg_m, k_tile]
    sg_tile_b = [k_tile, sg_n]

    # add layouts to DPAS op operands
    k_loop = match(gpu_func, ops={"scf.for"})
    dpas_op = match(k_loop, ops={"xegpu.dpas"})
    load_op_a = xegpu.get_load_op(transform.get_operand(anyvalue, dpas_op, [0]))
    load_op_b = xegpu.get_load_op(transform.get_operand(anyvalue, dpas_op, [1]))

    def add_prefetch(load_op, prefetch_nb, **layout):
        desc_op = xegpu.insert_prefetch(
            load_op,
            nb_prefetch=prefetch_nb,
        )
        pf_ops = transform.get_consumers_of_result(anytype, desc_op, 0)
        xegpu.set_anchor_layout(pf_ops, **layout)

    def annotate_ab_load(
        dpas_op, index, load_op, layout_load, layout_dpas, layout_prefetch, prefetch_nb
    ):
        """Annotate A/B tile load op and dpas operand and insert prefetch ops."""
        user = transform.get_consumers_of_result(anytype, load_op, 0)
        # FIXME use transform.alternatives instead of select and foreach
        # check_transpose = transform.AlternativesOp([], 2)

        # transposed case
        transpose_consumer_op = transform.select(anytype, user, "vector.transpose")
        with lh_transform.foreach(transpose_consumer_op):
            # Load op loads the transposed tile and thus sg_layout and sg_data
            # dimensions must be transposed. Keep inst_data which has been
            # validated in its current orientation.
            tr_load = layout_load.copy()
            tr_load["sg_layout"] = layout_load["sg_layout"][::-1]
            tr_load["sg_data"] = layout_load["sg_data"][::-1]
            tr_load["order"] = [0, 1]
            # annotate dpas op operand
            layout_dpas_order = layout_dpas.copy()
            layout_dpas_order["order"] = [1, 0]
            xegpu.set_anchor_layout(dpas_op, index=index, **layout_dpas_order)
            xegpu.set_anchor_layout(load_op, **tr_load)
            add_prefetch(load_op, prefetch_nb, **layout_prefetch)
            transform.yield_()

        # no transpose case
        dpas_consumer_op = transform.select(anytype, user, "xegpu.dpas")
        with lh_transform.foreach(dpas_consumer_op):
            # annotate dpas op operand
            xegpu.set_anchor_layout(dpas_op, index=index, **layout_dpas)
            xegpu.set_anchor_layout(load_op, **layout_load)
            add_prefetch(load_op, prefetch_nb, **layout_prefetch)
            transform.yield_()

    # A tile load layout
    layout_load_a = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_a,
        "inst_data": load_tile_a,
    }
    # A tile dpas layout
    layout_dpas_a = layout_load_a.copy()
    layout_dpas_a["inst_data"] = DPAS.A_TILE
    # A tile prefetch layout
    layout_prefetch_a = {
        "sg_layout": prefetch_layout_a,
        "sg_data": prefetch_tile_a,
        "inst_data": PREFETCH_INST_DATA,
    }
    annotate_ab_load(
        dpas_op,
        0,
        load_op_a,
        layout_load_a,
        layout_dpas_a,
        layout_prefetch_a,
        prefetch_a_nb,
    )

    # B tile load layout
    layout_load_b = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_b,
        "inst_data": load_tile_b,
    }
    # B tile dpas layout
    layout_dpas_b = layout_load_b.copy()
    layout_dpas_b["inst_data"] = DPAS.B_TILE
    # B tile prefetch layout
    layout_prefetch_b = {
        "sg_layout": prefetch_layout_b,
        "sg_data": prefetch_tile_b,
        "inst_data": PREFETCH_INST_DATA,
    }
    annotate_ab_load(
        dpas_op,
        1,
        load_op_b,
        layout_load_b,
        layout_dpas_b,
        layout_prefetch_b,
        prefetch_b_nb,
    )

    # C tile layout
    output_layout = {
        "sg_layout": sg_layout,
        "sg_data": [sg_m, sg_n],
        "inst_data": DPAS.C_TILE,
    }
    # C tile dpas anchor layout
    xegpu.set_anchor_layout(dpas_op, index=2, **output_layout)
    # annotate store op
    store_op_c = match(gpu_func, ops={"xegpu.store_nd"})
    xegpu.set_anchor_layout(store_op_c, **output_layout)

    # annotate the 1d load of the broadcast op with a slice layout
    # NOTE assumes that xegpu.load is followed by vector.broadcast
    maybe_bcast_load = match(gpu_func, ops={"xegpu.load"})
    load_user = transform.get_consumers_of_result(anytype, maybe_bcast_load, 0)
    bcast_ops = transform.select(anytype, load_user, "vector.broadcast")
    with lh_transform.foreach(bcast_ops) as bcast_op:
        bcast_load = xegpu.get_load_op(transform.get_operand(anyvalue, bcast_op, [0]))
        xegpu.set_anchor_layout(bcast_load, index=0, **output_layout, slice_dims=[0])
        transform.yield_()

    transform.apply_cse(gpu_func)
    canonicalize(gpu_func)

    # hoist desc ops out of reduction loop
    transform.apply_licm(k_loop)

    canonicalize(gpu_func)
    transform.apply_cse(gpu_func)
