from mlir import ir
from mlir.dialects.transform import loop
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import xegpu
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects import transform
from mlir.dialects.transform import structured
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
from .matmul_constraints import (
    NB_WORKITEMS,
    LOAD_MAX_ROWS,
    LOAD_MAX_COLS,
    MIN_NB_THREADS,
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
    """Schedule for lowering MLP-like payload to xegpu wg level."""
    nlayers = len(params)

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # fuse all elementwise ops
    mod = apply_registered_pass(mod, "linalg-fuse-elementwise-ops")

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

    # vectorize
    func = structured.structured_vectorize_children_and_apply_patterns(
        transform.any_op_t(),
        func,
        fold_type_extensions_into_contract=True,
    )

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
            smt_ext.assert_(
                sg_threads <= gpu_specs.max_nb_threads, "too many SG threads"
            )
            smt_ext.assert_(sg_threads >= MIN_NB_THREADS, "too few SG threads")

            # number of threads collapsed to 1d layout
            return sg_threads * NB_WORKITEMS

        nb_threads: int | transform.AnyParamType = (
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
