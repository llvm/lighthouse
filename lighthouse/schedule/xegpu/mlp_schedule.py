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
from lighthouse.schedule.parameters import ScheduleParameters
from .xegpu_specs import XeGPUSpecs
from .xegpu_parameter_selector import XeGPUParameterSelector
from .lowering_common import (
    vectorize_bufferize_and_outline_gpu_func,
    convert_vector_to_xegpu,
    get_payload_func,
)
from .matmul_constraints import (
    DPAS,
    PREFETCH_INST_DATA,
)


def matmul_schedule(
    payload_func_name: str = "payload",
    device: str | None = None,
    stop_at_stage: str = "",
    **layer_params,
) -> ir.Module:
    """Generate transform schedule module for a single matmul layer."""
    return mlp_schedule(
        params=ScheduleParameters([{"layer_kind": "matmul", **layer_params}]),
        payload_func_name=payload_func_name,
        device=device,
        stop_at_stage=stop_at_stage,
    )


def mlp_schedule(
    params: ScheduleParameters,
    payload_func_name: str = "payload",
    device: str | None = None,
    stop_at_stage: str = "",
) -> ir.Module:
    """Generate transform schedule module for MLP payload."""
    assert params is not None and len(params) > 0, "params must be provided."
    param_selector = XeGPUParameterSelector(device=device)
    gpu_specs = param_selector.gpu_specs

    with schedule_boilerplate() as (schedule, named_seq):
        # match the payload module
        anytype = transform.AnyOpType.get()
        func = get_payload_func(named_seq.bodyTarget, func_name=payload_func_name)
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
                shape = (m, n, k)
                transpose_a = layer_params.get("transpose_a", False)
                transpose_b = layer_params.get("transpose_b", False)
                generated_params = param_selector.get_parameters(
                    shape, transpose_a, transpose_b
                )
                # Overwrite original params to ensure consistent configuration
                layer_params.update(generated_params[0])

        try:
            bundle_xegpu_mlp_schedule(
                payload_mod,
                payload_func_name=payload_func_name,
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
    payload_func_name: str,
    gpu_specs: XeGPUSpecs,
    params: ScheduleParameters,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering MLP-like payload to xegpu wg level."""
    nlayers = len(params)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # fuse all elementwise ops first
    func = get_payload_func(mod, func_name=payload_func_name)
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # tile each layer separately
    matmul_ops = match_and_split(func, ops={"linalg.matmul"}, nhandles=nlayers)
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

    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    mod = vectorize_bufferize_and_outline_gpu_func(
        mod,
        payload_func_name=payload_func_name,
        gpu_specs=gpu_specs,
        params=params,
        stop_at_stage=stop_at_stage,
    )
    mod = convert_vector_to_xegpu(mod)
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
    wg_m: int,
    wg_n: int,
    sg_m: int,
    sg_n: int,
    k_tile: int,
    load_a_m: int,
    load_a_k: int,
    load_b_k: int,
    load_b_n: int,
    prefetch_a_m: int,
    prefetch_a_k: int,
    prefetch_b_k: int,
    prefetch_b_n: int,
    prefetch_a_nb: int,
    prefetch_b_nb: int,
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

    sg_layout = [wg_m // sg_m, wg_n // sg_n]

    load_tile_a = [load_a_m, load_a_k]
    load_tile_b = [load_b_k, load_b_n]
    prefetch_tile_a = [prefetch_a_m, prefetch_a_k]
    prefetch_tile_b = [prefetch_b_k, prefetch_b_n]

    # prefetch tile shape depends on transpose flag
    pf_shape_a = (k_tile, wg_m) if transpose_a else (wg_m, k_tile)
    pf_shape_b = (wg_n, k_tile) if transpose_b else (k_tile, wg_n)

    prefetch_layout_a = [pf_shape_a[0] // prefetch_a_m, pf_shape_a[1] // prefetch_a_k]
    prefetch_layout_b = [pf_shape_b[0] // prefetch_b_k, pf_shape_b[1] // prefetch_b_n]

    # matmul matrix shapes
    sg_tile_a = [sg_m, k_tile]
    sg_tile_b = [k_tile, sg_n]

    # add layouts to DPAS op operands
    # Anchor on the dpas and derive the reduction loop as its nearest scf.for
    # parent. Matching scf.for directly would over-match when an outer loop
    # wraps the reduction loop (e.g. the grid-stride loop of a persistent
    # kernel), yielding a multi-op handle that the single-target ops below
    # (get_operand/get_load_op) reject.
    dpas_op = match(gpu_func, ops={"xegpu.dpas"})
    k_loop = transform.get_parent_op(
        anytype, dpas_op, op_name="scf.for", deduplicate=True
    )
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
