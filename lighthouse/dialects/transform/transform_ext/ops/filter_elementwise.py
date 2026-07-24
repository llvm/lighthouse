from mlir import ir
from mlir.dialects import ext, transform, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import indexing_maps, linalg_inputs


def _all_loops_parallel(op: ir.OpView) -> bool:
    """Check whether all iterator types of a (generic) op are parallel."""
    build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
    parallel = build(linalg.IteratorType.parallel, context=op.context)
    return all(it == parallel for it in op.iterator_types)


def is_elementwise(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is an elementwise linalg op.

    NOTE: Mimics corresponding Linalg util as it is not exposed
          in the Python bindings yet.
    """
    ov = op.opview if isinstance(op, ir.Operation) else op
    maps = indexing_maps(ov)
    if maps is None:
        return False
    if isinstance(ov, linalg.GenericOp) and not _all_loops_parallel(ov):
        return False
    if not all(m.is_projected_permutation for m in maps):
        return False
    num_inputs = len(linalg_inputs(ov))
    return all(m.is_permutation for m in maps[num_inputs:])


class FilterElementwiseOp(
    TransformExtensionDialect.Operation, name="filter_elementwise"
):
    """
    Returns the target ops that are elementwise linalg ops.

    Targets that are not elementwise linalg ops are dropped.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the elementwise subset of `target`.
    """

    target: ext.Operand[transform.AnyOpType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FilterElementwiseOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            matching_ops = [t for t in targets if is_elementwise(t)]
            results.set_ops(op.ops, matching_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FilterElementwiseOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def filter_elementwise(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterElementwiseOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the elementwise subset of `target`.
    """
    return FilterElementwiseOp(target=target).ops
