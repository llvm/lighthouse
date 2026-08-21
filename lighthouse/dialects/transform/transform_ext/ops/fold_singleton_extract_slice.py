from mlir import ir
from mlir.dialects import ext, transform, tensor, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _is_dynamic_dim(dim: int) -> bool:
    return dim < 0


def _dim_compatible(a: int, b: int) -> bool:
    return a == b or _is_dynamic_dim(a) or _is_dynamic_dim(b)


def _same_ranked_tensor_type(lhs: ir.Type, rhs: ir.Type) -> bool:
    if not isinstance(lhs, ir.RankedTensorType) or not isinstance(
        rhs, ir.RankedTensorType
    ):
        return False
    if lhs.rank != rhs.rank:
        return False
    if lhs.element_type != rhs.element_type:
        return False
    return all(_dim_compatible(dl, dr) for dl, dr in zip(lhs.shape, rhs.shape))


def _is_rank_reduction_by_unit_dims(
    expanded: ir.RankedTensorType, reduced: ir.RankedTensorType
) -> bool:
    if expanded.rank <= reduced.rank:
        return False

    reduced_idx = 0
    skipped = 0
    for dim in expanded.shape:
        if reduced_idx < reduced.rank and _dim_compatible(
            dim, reduced.shape[reduced_idx]
        ):
            reduced_idx += 1
            continue
        if dim == 1:
            skipped += 1
            continue
        return False

    return reduced_idx == reduced.rank and skipped == (expanded.rank - reduced.rank)


def _as_single_result_value(op_or_val):
    if isinstance(op_or_val, ir.Value):
        return op_or_val
    if hasattr(op_or_val, "result"):
        return op_or_val.result
    if hasattr(op_or_val, "results") and len(op_or_val.results) == 1:
        return op_or_val.results[0]
    raise ValueError("Expected a value or single-result operation")


def _rewrite_extract_of_expand(
    extract_op: ir.Operation, rewriter: transform.TransformRewriter
) -> bool:
    source = extract_op.operands[0]
    producer = source.owner
    if producer is None or producer.name != "tensor.expand_shape":
        return False

    expanded_source = producer.operands[0]
    extract_result_ty = extract_op.results[0].type
    expanded_source_ty = expanded_source.type
    expanded_ty = source.type

    if not (
        isinstance(extract_result_ty, ir.RankedTensorType)
        and isinstance(expanded_source_ty, ir.RankedTensorType)
        and isinstance(expanded_ty, ir.RankedTensorType)
    ):
        return False

    if not _same_ranked_tensor_type(extract_result_ty, expanded_source_ty):
        return False

    if not _is_rank_reduction_by_unit_dims(expanded_ty, extract_result_ty):
        return False

    rewriter.replace_op(extract_op, [expanded_source])
    return True


def _rewrite_extract_of_fill(
    extract_op: ir.Operation, rewriter: transform.TransformRewriter
) -> bool:
    source = extract_op.operands[0]
    producer = source.owner
    if producer is None or producer.name != "linalg.fill":
        return False

    fill_result_ty = source.type
    extract_result_ty = extract_op.results[0].type
    if not (
        isinstance(fill_result_ty, ir.RankedTensorType)
        and isinstance(extract_result_ty, ir.RankedTensorType)
    ):
        return False

    if not _is_rank_reduction_by_unit_dims(fill_result_ty, extract_result_ty):
        return False

    if any(_is_dynamic_dim(dim) for dim in extract_result_ty.shape):
        # Keep conservative behavior for dynamic shapes.
        return False

    fill_value = producer.operands[0]
    with ir.InsertionPoint(extract_op), extract_op.location:
        empty = tensor.EmptyOp(
            tuple(extract_result_ty.shape), extract_result_ty.element_type
        )
        filled = linalg.fill(fill_value, outs=[empty.result])
        rewriter.replace_op(extract_op, [_as_single_result_value(filled)])
    return True


def _collect_extract_slice_ops(root: ir.Operation) -> list[ir.Operation]:
    extract_slice_ops = []

    def collect(op: ir.Operation) -> ir.WalkResult:
        if op.name == "tensor.extract_slice":
            extract_slice_ops.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(collect, ir.WalkOrder.PRE_ORDER)
    return extract_slice_ops


class FoldSingletonExtractSliceOp(
    TransformExtensionDialect.Operation, name="fold_singleton_extract_slice"
):
    """
    Rewrites redundant singleton-dimension tensor slice patterns.

    Rewrites:
      1) tensor.extract_slice(tensor.expand_shape(x)) -> x
      2) tensor.extract_slice(linalg.fill(... : tensor<...x1x...>))
         -> linalg.fill on rank-reduced tensor output.

    Args:
        target: Handle to root ops to rewrite within (e.g. func.func).
    Returns:
        Handle to (possibly) rewritten extract_slice ops.
    """

    target: ext.Operand[transform.AnyOpType]
    rewritten_ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FoldSingletonExtractSliceOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)

            for target in targets:
                extract_ops = _collect_extract_slice_ops(target)
                for extract_op in extract_ops:
                    did_rewrite = _rewrite_extract_of_expand(extract_op, rewriter)
                    if not did_rewrite:
                        _rewrite_extract_of_fill(extract_op, rewriter)

            # Return stable handles to the transformed target roots.
            results.set_ops(op.rewritten_ops, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FoldSingletonExtractSliceOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_singleton_extract_slice(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create FoldSingletonExtractSliceOp."""
    op = FoldSingletonExtractSliceOp(target=target)
    return op.rewritten_ops
