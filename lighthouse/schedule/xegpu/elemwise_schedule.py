from mlir import ir
from mlir.dialects.transform import xegpu
from mlir.dialects import transform
import lighthouse.transform as lh_transform
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
from lighthouse.dialects.transform.tune_ext import KnobValue
from .xegpu_specs import XeGPUSpecs
from .xegpu_parameter_selector import XeGPUParameterSelector
from .lowering_common import (
    vectorize_bufferize_and_outline_gpu_func,
    convert_vector_to_xegpu,
)
from .matmul_constraints import (
    LOAD_MAX_ROWS,
    LOAD_MAX_COLS,
)


def elemwise_schedule(
    params: list[dict[str, int | None]],
    stop_at_stage: str = "",
) -> ir.Module:
    """Generate transform schedule module for elemwise payload."""
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
        try:
            bundle_xegpu_elemwise_schedule(
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


def bundle_xegpu_elemwise_schedule(
    mod: ir.Value[transform.AnyOpType],
    gpu_specs: XeGPUSpecs,
    params: list[dict[str, int | KnobValue]],
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering elemwise-like payload to xegpu wg level."""
    nlayers = len(params)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # fuse all elementwise ops first
    mod = apply_registered_pass(mod, "linalg-fuse-elementwise-ops")

    # tile each layer separately
    generic_ops = match_and_split(mod, ops={"linalg.generic"}, nhandles=nlayers)
    for generic_op, layer_params in zip(generic_ops, params):
        # wg tiling
        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        _, [wg_loop], _ = lh_transform.tile(
            generic_op,
            tile_sizes=wg_tile,
            fuse_producers=True,
            use_forall=True,
            apply_cleanup=False,
        )

    func = transform.get_parent_op(
        anytype,
        wg_loop,
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
        xegpu_wg_annotation_for_elemwise_layer(
            gpu_func, gpu_specs=gpu_specs, **layer_params
        )

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod


def xegpu_wg_annotation_for_elemwise_layer(
    gpu_func: ir.Value,
    gpu_specs: XeGPUSpecs,
    *,
    wg_m: int | KnobValue,
    wg_n: int | KnobValue,
    sg_m: int | KnobValue,
    sg_n: int | KnobValue,
    load_m: int | KnobValue,
    load_n: int | KnobValue,
    **_catch_all,
):
    """
    Adds XeGPU anchor layout annotations for an elementwise layer.

    Should be applied after the payload has been converted to XeGPU using
    the convert-vector-to-xegpu pass.
    """
    anyvalue = transform.AnyValueType.get()

    @td_smt_ext.constrain_params(wg_m, wg_n, sg_m, sg_n, load_m, load_n)
    def calc_sg_layout(WG_M, WG_N, SG_M, SG_N, LD_M, LD_N):
        smt_ext.assert_(WG_M % SG_M == 0)
        smt_ext.assert_(WG_N % SG_N == 0)
        smt_ext.assert_(SG_M % LD_M == 0)
        smt_ext.assert_(SG_N % LD_N == 0)
        smt_ext.assert_(LD_M <= LOAD_MAX_ROWS)
        smt_ext.assert_(LD_N <= LOAD_MAX_COLS)
        return WG_M // SG_M, WG_N // SG_N

    sg_layout = calc_sg_layout.results

    sg_tile = [sg_m, sg_n]
    load_tile = [load_m, load_n]

    # add layouts to load ops
    load_ops = match(gpu_func, ops={"xegpu.load_nd"})

    def add_load_layout(load_ops, layout_load, layout_dpas):
        xegpu.set_anchor_layout(load_ops, **layout_load)
        result_tile = transform.get_result(anyvalue, load_ops, [0])
        xegpu.convert_layout(
            result_tile,
            input_sg_layout=layout_load["sg_layout"],
            input_sg_data=layout_load["sg_data"],
            input_inst_data=layout_load["inst_data"],
            target_sg_layout=layout_dpas["sg_layout"],
            target_sg_data=layout_dpas["sg_data"],
            target_inst_data=layout_dpas["inst_data"],
        )

    # load layout
    layout_load = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile,
        "inst_data": load_tile,
    }
    # inst layout
    layout_dpas = layout_load.copy()
    add_load_layout(
        load_ops,
        layout_load,
        layout_dpas,
    )
    # add layout to store ops
    store_ops = match(gpu_func, ops={"xegpu.store_nd"})
    xegpu.set_anchor_layout(store_ops, **layout_load)

    transform.apply_cse(gpu_func)
    canonicalize(gpu_func)
