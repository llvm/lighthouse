from collections.abc import Callable

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def make_filter_handles_op(
    op_name: str,
    predicate: Callable[[ir.Operation | ir.OpView], bool],
):
    """
    Factory that creates a filter-handles transform op class.

    The returned class is a ``TransformExtensionDialect.Operation`` that filters
    a handle by applying ``predicate`` to each payload op, keeping only those
    for which the predicate returns ``True``.

    Args:
        op_name:   MLIR op name to register (e.g. ``"filter_elementwise"``).
        predicate: Callable that receives an ``ir.Operation`` and returns
                   ``True`` if the op should be kept.
    Returns:
        A new ``TransformExtensionDialect.Operation`` subclass.
    """

    class _FilterHandlesOp(TransformExtensionDialect.Operation, name=op_name):
        target: ext.Operand[transform.AnyOpType]
        ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

        @classmethod
        def attach_interface_impls(cls, ctx=None):
            cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
            cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

        class TransformOpInterfaceModel(transform.TransformOpInterface):
            @staticmethod
            def apply(
                op: "_FilterHandlesOp",
                _rewriter: transform.TransformRewriter,
                results: transform.TransformResults,
                state: transform.TransformState,
            ) -> DiagnosedSilenceableFailure:
                targets = state.get_payload_ops(op.target)
                results.set_ops(op.ops, [t for t in targets if predicate(t)])
                return DiagnosedSilenceableFailure.Success

            @staticmethod
            def allow_repeated_handle_operands(_op: "_FilterHandlesOp") -> bool:
                return False

        class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
            @staticmethod
            def get_effects(op: ir.Operation, effects):
                transform.only_reads_handle(op.op_operands, effects)
                transform.produces_handle(op.results, effects)
                transform.only_reads_payload(effects)

    _FilterHandlesOp.__name__ = op_name
    _FilterHandlesOp.__qualname__ = op_name
    return _FilterHandlesOp
