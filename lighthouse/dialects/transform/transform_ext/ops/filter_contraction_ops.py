from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.utils.mlir import opview, indexing_maps, linalg_inputs, dim_position


def _map_dims(m: ir.AffineMap) -> set[int]:
    """Set of iteration-dim positions referenced by an affine map's results."""
    dims = set()
    for r in m.results:
        pos = dim_position(r)
        if pos is not None:
            dims.add(pos)
    return dims


def _is_structural_contraction(ov: ir.OpView) -> bool:
    """Detect a matmul-like op from its indexing maps alone (body-agnostic).

    A contraction has a reduction dimension (one that is absent from the
    output) that is shared by at least two inputs. This ignores the body, so it
    also matches contractions with extra elementwise operands (e.g. a scaled or
    dequantized input) that the strict multiply-accumulate matcher rejects.
    """
    maps = indexing_maps(ov)
    inputs = linalg_inputs(ov)
    if not maps or not inputs:
        return False
    num_inputs = len(inputs)
    input_maps = maps[:num_inputs]
    output_maps = maps[num_inputs:]

    out_dims: set[int] = set()
    for m in output_maps:
        out_dims |= _map_dims(m)

    # Reduction dims are the iteration dims not present in any output.
    reduction_dims = set(range(maps[0].n_dims)) - out_dims
    for k in reduction_dims:
        if sum(1 for m in input_maps if k in _map_dims(m)) >= 2:
            return True
    return False


def is_contraction_op(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is a linalg contraction (matmul-like) op.

    Recognizes both named contractions (e.g. linalg.batch_matmul) and their
    generic form, independent of rank. A contraction contracts a reduction
    dimension shared by two inputs, which distinguishes it from the surrounding
    elementwise or plain (single-input) reduction ops.
    """
    ov = opview(op)
    if "linalg" not in ov.operation.name:
        return False
    return linalg.isa_contraction_op(ov) or _is_structural_contraction(ov)


FilterContractionOpsOp = make_filter_handles_op(
    "filter_contraction_ops", is_contraction_op
)


def filter_contraction_ops(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterContractionOpsOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the contraction-op subset of `target`.
    """
    return FilterContractionOpsOp(target=target).ops
