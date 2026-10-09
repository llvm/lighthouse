from mlir import ir
from mlir.dialects import transform
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import memref
from mlir.dialects.transform import vector
import lighthouse.transform as lh_transform

from lighthouse.dialects.transform import transform_ext

from lighthouse.pipeline.helper import (
    PipelineInterrupt,
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
)
from .xegpu_specs import XeGPUSpecs
from .matmul_constraints import NB_WORKITEMS


def get_payload_func(
    mod: transform.AnyOpType,
    *,
    op_name: str | list[str] | None = None,
    func_name: str | None = None,
) -> transform.AnyOpType:
    """
    Returns a handle to the payload function in the module.

    If neither `op_name` nor `func_name` is provided, returns the first `func.func` in the module. If both are provided, `func_name` takes precedence.

    Args:
        mod: The MLIR module containing the payload function.
        op_name: Optional name of the operations to match in the payload function (e.g., "linalg.generic").
        func_name: Optional name of the payload function to retrieve.

    Returns:
        A handle to the payload function.
    """
    anytype = transform.AnyOpType.get()

    neither_is_set = op_name is None and func_name is None
    if func_name is not None or neither_is_set:
        op_attrs = None
        if func_name is not None:
            op_attrs = {"sym_name": ir.StringAttr.get(func_name)}
        # Filter by function name
        func_ops = match(
            mod,
            ops={"func.func"},
            op_attrs=op_attrs,
        )
    else:
        # Filter by containing ops
        matching_children = lh_transform.match_op(mod, op_name)
        func_ops = transform.get_parent_op(
            anytype,
            matching_children,
            op_name="func.func",
            deduplicate=True,
        )
    # Return the first function
    func = transform_ext.extract_handle(func_ops, 0)
    return func


def vectorize_bufferize_and_outline_gpu_func(
    mod: transform.AnyOpType,
    *,
    payload_func_name: str | None = None,
    payload_anchor_ops: str | list[str] | None = None,
    payload_func: transform.AnyOpType | None = None,
    gpu_specs: XeGPUSpecs,
    params: list[dict[str, int]],
    stop_at_stage: str = "",
) -> transform.AnyOpType:
    """Vectorizes and bufferizes the payload function and outlines it to gpu.func."""

    def match_func():
        return get_payload_func(
            mod, func_name=payload_func_name, op_name=payload_anchor_ops
        )

    if payload_func is None:
        payload_func = match_func()
    vectorize(
        mod,
        payload_func_name=payload_func_name,
        payload_anchor_ops=payload_anchor_ops,
        payload_func=payload_func,
    )
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = bufferize(mod)
    payload_func = match_func()
    convert_allocs_to_gpu(
        mod,
        payload_func_name=payload_func_name,
        payload_anchor_ops=payload_anchor_ops,
        payload_func=payload_func,
    )
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    payload_func = match_func()
    convert_to_gpu_launch(
        mod,
        payload_func_name=payload_func_name,
        payload_anchor_ops=payload_anchor_ops,
        payload_func=payload_func,
    )
    payload_func = match_func()
    mod = outline_gpu_function(
        mod,
        payload_func_name=payload_func_name,
        payload_anchor_ops=payload_anchor_ops,
        payload_func=payload_func,
        gpu_specs=gpu_specs,
        params=params,
    )
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    return mod


def vectorize(
    mod: transform.AnyOpType,
    *,
    payload_func_name: str | None = None,
    payload_anchor_ops: str | list[str] | None = None,
    payload_func: transform.AnyOpType | None = None,
    disable_multi_reduction_to_contract_patterns: bool = False,
) -> transform.AnyOpType:
    """Vectorize and run loop-hoisting cleanup for the payload function."""
    if payload_func is None:
        payload_func = get_payload_func(
            mod, func_name=payload_func_name, op_name=payload_anchor_ops
        )
    payload_func = structured.structured_vectorize_children_and_apply_patterns(
        transform.any_op_t(),
        payload_func,
        fold_type_extensions_into_contract=True,
        disable_multi_reduction_to_contract_patterns=disable_multi_reduction_to_contract_patterns,
    )

    # Try to remove any unit dimensions that may have been introduced due to tiling (e.g. batch dim of 1)
    with ir.InsertionPoint(transform.apply_patterns(payload_func).patterns):
        vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
        vector.apply_patterns_vector_drop_unit_dims_with_shape_cast()
    lh_transform.cleanup(payload_func)

    # Hoist loop-invariant vector read/store ops if present.
    k_loop = match(payload_func, ops={"scf.for"})
    lh_transform.loop_hoisting(k_loop)

    # Narrow vector.contract operands if possible to match DPAS op spec.
    contract_ops = lh_transform.match_op(payload_func, "vector.contract")
    with lh_transform.foreach(contract_ops) as contract_op:
        transform_ext.narrow_vector_contract_operands(contract_op)
        transform.yield_()
    lh_transform.cleanup(payload_func)

    return payload_func


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
    mod: transform.AnyOpType,
    *,
    payload_func_name: str | None = None,
    payload_anchor_ops: str | list[str] | None = None,
    payload_func: transform.AnyOpType | None = None,
) -> transform.AnyOpType:
    """Insert deallocs and convert memref alloc/dealloc ops to GPU variants."""
    if payload_func is None:
        payload_func = get_payload_func(
            mod, func_name=payload_func_name, op_name=payload_anchor_ops
        )

    payload_func = apply_registered_pass(payload_func, "buffer-deallocation-pipeline")
    alloc_ops = match(payload_func, ops={"memref.alloc"})
    transform_ext.replace(alloc_ops, "gpu.alloc")
    alloc_ops = match(payload_func, ops={"memref.dealloc"})
    transform_ext.replace(alloc_ops, "gpu.dealloc")

    return payload_func


def convert_to_gpu_launch(
    mod: transform.AnyOpType,
    *,
    payload_func_name: str | None = None,
    payload_anchor_ops: str | list[str] | None = None,
    payload_func: transform.AnyOpType | None = None,
) -> transform.AnyOpType:
    """Convert scf.forall/scf.parallel structure to gpu.launch."""
    if payload_func is None:
        payload_func = get_payload_func(
            mod, func_name=payload_func_name, op_name=payload_anchor_ops
        )

    forall_loops = match(payload_func, ops={"scf.forall"})
    with lh_transform.foreach(forall_loops) as forall_op:
        loop.loop_forall_to_parallel([transform.any_op_t()], forall_op)
        transform.yield_()

    # convert scf.parallel to gpu.launch
    payload_func = apply_registered_pass(payload_func, "gpu-map-parallel-loops")
    payload_func = apply_registered_pass(payload_func, "convert-parallel-loops-to-gpu")
    payload_func = apply_registered_pass(payload_func, "lower-affine")
    transform.apply_cse(payload_func)
    canonicalize(payload_func)

    return payload_func


# TODO this should be deprecated or generalized and used in the outline schedule
def outline_gpu_function(
    mod: transform.AnyOpType,
    *,
    payload_func_name: str | None = None,
    payload_anchor_ops: str | list[str] | None = None,
    payload_func: transform.AnyOpType | None = None,
    params: list[dict[str, int]],
    gpu_specs: dict[str, int],
) -> transform.AnyOpType:
    """Set gpu.launch threads and outline the payload to gpu.func."""
    if payload_func is None:
        payload_func = get_payload_func(
            mod, func_name=payload_func_name, op_name=payload_anchor_ops
        )

    nlayers = len(params)

    # set correct number of gpu threads
    launch_ops = match_and_split(mod, ops={"gpu.launch"}, nhandles=nlayers)
    assert len(launch_ops) == nlayers
    for launch_op, layer_params in zip(launch_ops, params):
        # tunable parameters
        wg_m, wg_n = layer_params["wg_m"], layer_params["wg_n"]
        sg_m, sg_n = layer_params["sg_m"], layer_params["sg_n"]

        sg_m_threads = wg_m // sg_m
        sg_n_threads = wg_n // sg_n
        sg_threads = sg_m_threads * sg_n_threads
        nb_threads: int = sg_threads * NB_WORKITEMS

        xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])

    # outline gpu func
    payload_func = apply_registered_pass(payload_func, "lower-affine")
    canonicalize(payload_func)
    payload_func = apply_registered_pass(
        payload_func, "gpu-launch-sink-index-computations"
    )
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    return mod


def convert_vector_to_xegpu(mod: transform.AnyOpType) -> transform.AnyOpType:
    """Attach xevm target and convert vector ops to xegpu in all gpu.module ops."""

    # set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # convert vector to xegpu
    gpu_mod_ops = match(mod, ops={"gpu.module"})
    with lh_transform.foreach(gpu_mod_ops) as gpu_mod:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)
        gpu_func = apply_registered_pass(gpu_func, "loop-invariant-code-motion")
        transform.yield_()

    return mod
