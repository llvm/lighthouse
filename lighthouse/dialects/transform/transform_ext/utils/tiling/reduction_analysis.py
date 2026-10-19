from typing import NamedTuple

from mlir import ir

from lighthouse.utils.mlir import (
    dim_position,
    indexing_maps,
    is_linalg_reduction_op,
    linalg_inputs,
    linalg_loop_dim_sizes,
    linalg_outputs,
    linalg_reduction_dims,
    opview,
)


class ReductionInfo(NamedTuple):
    """Structure of a non-contraction reduction relevant to vectorization.

    Reductions are classified by their vector dim, i.e. the loop dim indexing
    the innermost (contiguous) dim of the highest-rank input:
      * inner: vector dim is reduced (e.g. row max/sum), lanes are
        combined by a horizontal reduction,
      * outer: vector dim is parallel (e.g. RMSNorm channel sum), every lane is
        an independent accumulator and no horizontal reduction is needed.
    """

    vector_dim: int
    dim_sizes: list[int | None]
    parallel_dims: list[int]
    reduction_dims: list[int]
    inner: bool
    elem_type: ir.Type


def _reduction_vector_dim(op: ir.OpView) -> int | None:
    """Loop dim indexing the innermost dim of the highest-rank input."""
    best_map = None
    for value, amap in zip(linalg_inputs(op), indexing_maps(op)):
        if not isinstance(value.type, ir.ShapedType) or not amap.results:
            continue
        if best_map is None or len(amap.results) > len(best_map.results):
            best_map = amap
    if best_map is None:
        return None
    return dim_position(best_map.results[-1])


def reduction_info(op: ir.Operation | ir.OpView) -> ReductionInfo | None:
    """Analyze a non-contraction reduction, else None."""
    ov = opview(op)
    if not is_linalg_reduction_op(ov):
        return None
    vector_dim = _reduction_vector_dim(ov)
    if vector_dim is None:
        return None
    dim_sizes = linalg_loop_dim_sizes(ov)
    reduction_dims = linalg_reduction_dims(ov)
    return ReductionInfo(
        vector_dim=vector_dim,
        dim_sizes=dim_sizes,
        parallel_dims=[d for d in range(len(dim_sizes)) if d not in reduction_dims],
        reduction_dims=reduction_dims,
        inner=vector_dim in reduction_dims,
        elem_type=ir.ShapedType(linalg_outputs(ov)[0].type).element_type,
    )
