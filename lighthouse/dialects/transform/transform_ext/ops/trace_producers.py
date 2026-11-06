from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure
from collections import deque

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import defining_op
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    subgraph_id,
)


class TraceProducersOp(TransformExtensionDialect.Operation, name="trace_producers"):
    """
    Collect ops in the SSA producer chain, optionally bounded by a scope.

    With `scope`, stop at values defined outside its nested operations. An
    excluded producer is neither returned nor traversed. With `subgraph`, only
    linalg ops with that id are admitted, except untagged fills. Untagged tensor
    plumbing and constants may connect members after shape canonicalization;
    a different membership id always stops traversal.

    Args:
        target: Handle to a single op.
        scope: Optional containers bounding traversal.
        subgraph: Optional single membership id.
        order: "closest-first" (BFS), or "reverse-topological" to fuse all
            consumers before their shared producers.
    Returns:
        Handles to admissible producer ops in the requested order.
    """

    target: ext.Operand[transform.AnyOpType]
    order: ir.StringAttr = ext.attribute()
    scope: ext.Operand[transform.AnyOpType] | None = None
    subgraph: ext.Operand[transform.AnyParamType] | None = None
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
            id = None
            if op.subgraph is not None:
                ids = state.get_params(op.subgraph)
                if len(ids) != 1 or not isinstance(ids[0], ir.IntegerAttr):
                    op.operation.location.emit_error("expected one subgraph id")
                    return DiagnosedSilenceableFailure.SilenceableFailure
                id = ir.IntegerAttr(ids[0]).value
            allowed = None
            if op.scope is not None:
                allowed = set()
                for scope in state.get_payload_ops(op.scope):

                    def collect(inner):
                        allowed.add(inner)
                        return ir.WalkResult.ADVANCE

                    scope.operation.walk(collect)

            def admissible(owner):
                if owner is None or (allowed is not None and owner not in allowed):
                    return False
                if id is not None:
                    member_id = subgraph_id(owner)
                    if member_id is not None:
                        return member_id == id
                    return owner.name.startswith("tensor.") or owner.name in (
                        "arith.constant",
                        "linalg.fill",
                    )
                return True

            # Walk the SSA producer graph via operand -> owner edges.
            # Use BFS to guarantee closest-first ordering by graph distance.
            producers: list[ir.Operation] = []
            visited: set[ir.Operation] = set()
            worklist = deque()

            for operand in leaf.operands:
                owner_op = defining_op(operand)
                if admissible(owner_op) and owner_op not in visited:
                    visited.add(owner_op)
                    worklist.append(owner_op)

            while worklist:
                producer = worklist.popleft()
                producers.append(producer)

                for operand in producer.operands:
                    owner_op = defining_op(operand)
                    if admissible(owner_op) and owner_op not in visited:
                        visited.add(owner_op)
                        worklist.append(owner_op)

            if op.order.value == "reverse-topological":
                ordered, seen = [], set()

                def visit(producer):
                    if producer in seen:
                        return
                    seen.add(producer)
                    for operand in producer.operands:
                        dependency = defining_op(operand)
                        if dependency in visited:
                            visit(dependency)
                    ordered.append(producer)

                for producer in producers:
                    visit(producer)
                producers = list(reversed(ordered))
            elif op.order.value != "closest-first":
                op.operation.location.emit_error(
                    f"unknown producer order: {op.order.value}"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
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
    scope: ir.Value[transform.AnyOpType] | None = None,
    subgraph: ir.Value[transform.AnyParamType] | None = None,
    *,
    order: str = "closest-first",
) -> ir.Value:
    """
    snake_case wrapper to create a TraceProducersOp.

    Args:
        target: Handle to a single op.
        scope: Optional containers bounding traversal.
        subgraph: Optional membership id bounding linalg producers.
        order: "closest-first" or "reverse-topological".
    Returns:
        Handles to admissible producers in the requested order.
    """
    return TraceProducersOp(
        target=target, scope=scope, subgraph=subgraph, order=ir.StringAttr.get(order)
    ).ops
