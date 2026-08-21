from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure
from collections import deque

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import defining_op


class TraceProducersOp(TransformExtensionDialect.Operation, name="trace_producers"):
    """
    Collect all ops in the SSA producer chain of the target op.

    Args:
        target: Handle to a single op.
    Returns:
        Handles to all producer ops, ordered closest-first.
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
            op: "TraceProducersOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            leaf = target_ops[0]
            # Walk the SSA producer graph via operand -> owner edges.
            # Use BFS to guarantee closest-first ordering by graph distance.
            producers: list[ir.Operation] = []
            visited_ids: set[int] = set()
            worklist = deque()

            for operand in leaf.operands:
                owner_op = defining_op(operand)
                if owner_op is not None and id(owner_op) not in visited_ids:
                    visited_ids.add(id(owner_op))
                    worklist.append(owner_op)

            while worklist:
                producer = worklist.popleft()
                producers.append(producer)

                for operand in producer.operands:
                    owner_op = defining_op(operand)
                    if owner_op is not None and id(owner_op) not in visited_ids:
                        visited_ids.add(id(owner_op))
                        worklist.append(owner_op)

            results.set_ops(op.ops, producers)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "TraceProducersOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def trace_producers(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a TraceProducersOp.

    Args:
        target: Handle to a single op.
    Returns:
        Handles to all producer ops, ordered closest-first.
    """
    return TraceProducersOp(target=target).ops
