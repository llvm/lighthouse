from mlir.dialects import transform
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import memref
import lighthouse.transform as lh_transform

from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects import smt_ext
from lighthouse.dialects.transform import smt_ext as td_smt_ext
from lighthouse.dialects.transform.tune_ext import KnobValue

from lighthouse.pipeline.helper import (
    PipelineInterrupt,
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
)
from .xegpu_specs import XeGPUSpecs
from .matmul_constraints import NB_WORKITEMS, MIN_NB_THREADS


def vectorize_bufferize_and_outline_gpu_func(
    mod: transform.AnyOpType,
    func: transform.AnyOpType,
    *,
    nlayers: int,
    gpu_specs: XeGPUSpecs,
    params: list[dict[str, int | KnobValue]],
    stop_at_stage: str = "",
) -> transform.AnyOpType:
    """Vectorize, bufferize, emit gpu.launch, and outline to gpu.func."""

    func = vectorize(func)
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = bufferize(mod)

    # match payload function
    anytype = transform.AnyOpType.get()
    wg_loops = match(mod, ops={"scf.forall"})
    func = transform.get_parent_op(
        anytype, wg_loops, op_name="func.func", deduplicate=True
    )
    func = convert_allocs_to_gpu(func)
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    func = convert_to_gpu_launch(func, nlayers=nlayers)
    mod = outline_gpu_function(mod, func, gpu_specs=gpu_specs, params=params)

    return mod


def vectorize(
    func: transform.AnyOpType,
) -> transform.AnyOpType:
    """Vectorize and run loop-hoisting cleanup for the payload function."""

    # vectorize
    func = structured.structured_vectorize_children_and_apply_patterns(
        transform.any_op_t(),
        func,
        fold_type_extensions_into_contract=True,
    )
    # Hoist loop-invariant vector read/store ops if present.
    k_loop = match(func, ops={"scf.for"})
    lh_transform.loop_hoisting(k_loop)
    lh_transform.cleanup(func)

    return func


def bufferize(
    mod: transform.AnyOpType,
) -> transform.AnyOpType:
    """Apply one-shot bufferization."""

    # bufferize
    bufferization.bufferization_eliminate_empty_tensors(mod)
    mod = bufferization.bufferization_one_shot_bufferize(
        transform.any_op_t(),
        mod,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        bufferize_function_boundaries=True,
    )
    # fold memref.subviews into vector.transfer_read/write ops
    with ir.InsertionPoint(transform.apply_patterns(mod).patterns):
        memref.apply_patterns_memref_fold_memref_alias_ops()

    transform.apply_cse(mod)
    canonicalize(mod)

    return mod


def convert_allocs_to_gpu(
    func: transform.AnyOpType,
) -> transform.AnyOpType:
    """Insert deallocs and convert memref alloc/dealloc ops to GPU variants."""

    func = apply_registered_pass(func, "buffer-deallocation-pipeline")
    alloc_ops = match(func, ops={"memref.alloc"})
    transform_ext.replace(alloc_ops, "gpu.alloc")
    alloc_ops = match(func, ops={"memref.dealloc"})
    transform_ext.replace(alloc_ops, "gpu.dealloc")

    return func


def convert_to_gpu_launch(
    func: transform.AnyOpType,
    *,
    nlayers: int,
) -> transform.AnyOpType:
    """Convert scf.forall/scf.parallel structure to gpu.launch."""

    anytype = transform.AnyOpType.get()

    # convert forall to parallel
    wg_loops = match_and_split(func, ops={"scf.forall"}, nhandles=nlayers)
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)

    # convert scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    return func


def outline_gpu_function(
    mod: transform.AnyOpType,
    func: transform.AnyOpType,
    *,
    gpu_specs: XeGPUSpecs,
    params: list[dict[str, int | KnobValue]],
) -> transform.AnyOpType:
    """Set gpu.launch threads and outline the payload to gpu.func."""
    nlayers = len(params)

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
            smt_ext.assert_(
                sg_threads >= MIN_NB_THREADS,
                f"too few SG threads: {sg_threads} {WG_M}/{SG_M}*{WG_N}/{SG_N}",
            )

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

    return mod


def convert_vector_to_xegpu(
    mod: transform.AnyOpType, *, nlayers: int
) -> transform.AnyOpType:
    """Attach xevm target and convert vector ops to xegpu in all gpu.module ops."""

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

    return mod
