from collections.abc import Callable, Sequence
from typing import Any

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def make_filter_handles_op(
    op_name: str,
    predicate: Callable[[ir.Operation | ir.OpView], bool]
    | Callable[[ir.Operation | ir.OpView, Any], bool],
    *,
    parse_param: Callable[[Sequence[ir.Attribute]], Any | None] | None = None,
):
    """
    Factory that creates a filter-handles transform op class.

    The returned class is a ``TransformExtensionDialect.Operation`` that filters
    a handle by applying ``predicate`` to each payload op, keeping only those
    for which the predicate returns ``True``.

    If ``parse_param`` is provided, the generated op takes an additional
    ``AnyParam`` operand named ``param``. The parser receives
    ``state.get_params(op.param)`` and must return a parsed value (or ``None``
    to signal a silenceable failure). The predicate is then called as
    ``predicate(op, parsed_param)``.

    Args:
        op_name:   MLIR op name to register (e.g. ``"filter_elementwise"``).
        predicate:   Callable used to decide whether a target op is kept.
                 Signature is ``predicate(op)`` when ``parse_param`` is
                 omitted, and ``predicate(op, parsed_param)`` otherwise.
        parse_param: Optional parameter parser for a secondary ``AnyParam``
                 operand.
    Returns:
        A new ``TransformExtensionDialect.Operation`` subclass.
    """

    if parse_param is None:

        class _FilterHandlesOp(TransformExtensionDialect.Operation, name=op_name):
            target: ext.Operand[transform.AnyOpType]
            ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

            @classmethod
            def attach_interface_impls(cls, ctx=None):
                cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
                cls.MemoryEffectsOpInterfaceModel.attach(
                    cls.OPERATION_NAME,
                    context=ctx,
                )

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
                def get_effects(op: ir.Operation):
                    return (
                        transform.only_reads_handle(op.op_operands)
                        + transform.produces_handle(op.results)
                        + transform.only_reads_payload()
                    )

    else:

        class _FilterHandlesOp(TransformExtensionDialect.Operation, name=op_name):
            target: ext.Operand[transform.AnyOpType]
            param: ext.Operand[transform.AnyParamType]
            ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

            @classmethod
            def attach_interface_impls(cls, ctx=None):
                cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
                cls.MemoryEffectsOpInterfaceModel.attach(
                    cls.OPERATION_NAME,
                    context=ctx,
                )

            class TransformOpInterfaceModel(transform.TransformOpInterface):
                @staticmethod
                def apply(
                    op: "_FilterHandlesOp",
                    _rewriter: transform.TransformRewriter,
                    results: transform.TransformResults,
                    state: transform.TransformState,
                ) -> DiagnosedSilenceableFailure:
                    parsed_param = parse_param(state.get_params(op.param))
                    if parsed_param is None:
                        return DiagnosedSilenceableFailure.SilenceableFailure

                    targets = state.get_payload_ops(op.target)
                    results.set_ops(
                        op.ops,
                        [t for t in targets if predicate(t, parsed_param)],
                    )
                    return DiagnosedSilenceableFailure.Success

                @staticmethod
                def allow_repeated_handle_operands(_op: "_FilterHandlesOp") -> bool:
                    return False

            class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
                @staticmethod
                def get_effects(op: ir.Operation):
                    return (
                        transform.only_reads_handle(op.op_operands)
                        + transform.produces_handle(op.results)
                        + transform.only_reads_payload()
                    )

    _FilterHandlesOp.__name__ = op_name
    _FilterHandlesOp.__qualname__ = op_name
    return _FilterHandlesOp


def make_filter_handles_param_op(
    op_name: str,
    parse_param: Callable[[Sequence[ir.Attribute]], Any | None],
    predicate: Callable[[ir.Operation | ir.OpView, Any], bool],
):
    """Compatibility wrapper for parameterized filters."""
    return make_filter_handles_op(
        op_name,
        predicate,
        parse_param=parse_param,
    )
