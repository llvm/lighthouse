from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_EPILOGUE_ATTR_NAME,
    subgraph_id,
)


class MatchSubgraphOp(TransformExtensionDialect.Operation, name="match_subgraph"):
    """Match members of subgraphs previously tagged by tag_subgraphs.

    Uses the stable subgraph ids assigned by tag_subgraphs to find members
    within a container. This lets schedules reacquire operations after rewrites
    replace anchor/member handles, provided the subgraph tags are preserved.

    The id parameter accepts one or more ids. By default return directly tagged
    operations. With `ops`, match
    those names within members, including their scalar bodies. "*" selects
    whole members. An empty id list matches nothing.
    `role` selects all members, the scheduling body, or its layout epilogue.
    """

    target: ext.Operand[transform.AnyOpType]
    id: ext.Operand[transform.AnyParamType]
    ops: ir.ArrayAttr = ext.attribute()
    role: ir.StringAttr = ext.attribute(
        default_factory=lambda: ir.StringAttr.get("all")
    )
    members: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, results, state):
            ids = state.get_params(op.id)
            if any(not isinstance(id, ir.IntegerAttr) for id in ids):
                op.operation.location.emit_error("match_subgraph expects integer ids")
                return DiagnosedSilenceableFailure.SilenceableFailure
            expected = {ir.IntegerAttr(id).value for id in ids}
            members = []
            if op.role.value not in ("all", "body", "epilogue"):
                op.operation.location.emit_error(
                    f"unknown subgraph role: {op.role.value}"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            names = (
                None
                if "*" in [ir.StringAttr(name).value for name in op.ops]
                else {ir.StringAttr(name).value for name in op.ops}
            )

            def visit(inner):
                if subgraph_id(inner) in expected:
                    epilogue = SUBGRAPH_EPILOGUE_ATTR_NAME in inner.attributes
                    if (
                        op.role.value == "body"
                        and epilogue
                        or (op.role.value == "epilogue" and not epilogue)
                    ):
                        return ir.WalkResult.SKIP
                    if names is None:
                        members.append(inner)
                    else:

                        def collect(nested):
                            if nested.name in names:
                                members.append(nested)
                            return ir.WalkResult.ADVANCE

                        inner.walk(collect)
                    return ir.WalkResult.SKIP
                return ir.WalkResult.ADVANCE

            for target in state.get_payload_ops(op.target):
                target.operation.walk(visit)
            results.set_ops(op.members, members)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def match_subgraph(
    target: ir.Value,
    id: ir.Value,
    ops: str | list[str] | None = None,
    *,
    role: str = "all",
) -> ir.Value:
    if isinstance(ops, str):
        ops = [ops]
    names = ir.ArrayAttr.get([ir.StringAttr.get(name) for name in (ops or ["*"])])
    return MatchSubgraphOp(
        target=target, id=id, ops=names, role=ir.StringAttr.get(role)
    ).members
