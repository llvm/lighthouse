"""Explicit preparation for the scheduling policy used by tag_subgraphs."""

from mlir import ir
from mlir.dialects import ext, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import subgraph_analysis as sa
from lighthouse.dialects.transform.transform_ext.utils.ir_rewrite import (
    _ancestor_in_block,
)
from lighthouse.utils.mlir import indexing_maps


def _is_cast_generic(op):
    if op.name != "linalg.generic" or len(op.results) != 1:
        return False
    generic = op.opview
    body = generic.regions[0].blocks[0]
    operations = list(body.operations)
    return (
        len(generic.inputs) == 1
        and len(generic.outputs) == 1
        and len(operations) == 2
        and operations[0].name in ("arith.extf", "arith.truncf")
        and list(operations[0].operands) == [body.arguments[0]]
        and operations[1].name == "linalg.yield"
        and list(operations[1].operands) == list(operations[0].results)
        and all(
            mapping == ir.AffineMap.get_identity(mapping.n_dims)
            for mapping in indexing_maps(generic)
        )
    )


def _fold_gemm_cast_reshape(op, rewriter):
    """Keep pointwise output casts in a matmul's 2D iteration space."""
    if sa.is_tagged(op) or len(op.results) != 1:
        return
    source_type = op.results[0].type
    if (
        not isinstance(source_type, ir.RankedTensorType)
        or source_type.rank != 2
        or source_type.encoding is not None
        or any(size <= 0 for size in source_type.shape)
    ):
        return
    for reshape in list(sa._users(op)):
        if reshape.name != "tensor.expand_shape" or sa.is_tagged(reshape):
            continue
        if len(list(reshape.opview.reassociation)[-1]) == 1:
            continue
        for cast in list(sa._users(reshape)):
            if (
                sa.is_tagged(cast)
                or not _is_cast_generic(cast)
                or cast.operands[0] != reshape.results[0]
            ):
                continue
            result_type = ir.RankedTensorType(cast.results[0].type)
            if result_type.encoding is not None or any(
                size <= 0 for size in result_type.shape
            ):
                continue
            with ir.InsertionPoint(reshape), cast.location:
                empty = tensor.EmptyOp(source_type.shape, result_type.element_type)
                folded = cast.clone()
                folded.operands[0] = op.results[0]
                folded.operands[1] = empty.result
                folded.results[0].set_type(empty.result.type)
                identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(2))
                folded.attributes["indexing_maps"] = ir.ArrayAttr.get(
                    [identity, identity]
                )
                folded.attributes["iterator_types"] = ir.ArrayAttr.get(
                    list(cast.attributes["iterator_types"])[:2]
                )
                expanded = tensor.ExpandShapeOp(
                    result_type,
                    folded.results[0],
                    reshape.opview.reassociation,
                    [],
                    result_type.shape,
                )
            rewriter.replace_op(cast, [expanded.result])
        if not sa._users(reshape):
            rewriter.erase_op(reshape)


def _isolate_producers(members, anchor):
    """Clone externally used producers, preserving all other users."""
    order = {op.operation: i for i, op in enumerate(anchor.block.operations)}
    for producer in sorted(members, key=order.__getitem__, reverse=True):
        if producer not in sa.shared_producers(members, anchor):
            continue
        with ir.InsertionPoint(producer):
            clone = producer.clone()
        for old, new in zip(producer.results, clone.results):
            for use in [
                u
                for u in old.uses
                if _ancestor_in_block(u.owner, anchor.block) in members
            ]:
                use.owner.operands[use.operand_number] = new
        members[members.index(producer)] = clone
    return members


def _prepare(block, kinds, rewriter):
    if "gemm" in kinds:
        for op in list(block.operations):
            if op.name == "linalg.matmul":
                _fold_gemm_cast_reshape(op.operation, rewriter)
    claimed = {op.operation for op in block.operations if sa.is_tagged(op)}
    for kind in kinds:
        for op in list(block.operations):
            block_ops = [value.operation for value in block.operations]
            for members, anchor in sa.SUBGRAPH_MATCHERS[kind](
                op.operation, block_ops, claimed.__contains__
            ):
                if kind != "elementwise":
                    members = _isolate_producers(members, anchor)
                claimed.update(members)


class PrepareSubgraphsOp(TransformExtensionDialect.Operation, name="prepare_subgraphs"):
    """Clean up container IR to enable subgraph matching and tagging.

    Applies two scheduling-specific rewrites:

      * Moves pointwise arith.extf/truncf generics before a matmul result's
        tensor.expand_shape, keeping casts in the matmul's 2D iteration space
        so they can be matched as its epilogue.
      * Clones shared producers selected for recomputation in non-elementwise
        subgraphs, isolating their uses without changing external consumers.
        Shared normalization outputs and residual computations remain
        materialized.

    Run after iteration-space preparation and before tag_subgraphs, using the
    same ordered `kinds` and single-block containers. This op prepares the IR
    but adds no scheduling attributes; tag_subgraphs performs the tagging.
    Consumes container handles and invalidates nested handles. Returns the same
    containers so later transforms can reacquire their operations.
    """

    target: ext.Operand[transform.AnyOpType]
    kinds: ir.ArrayAttr = ext.attribute()
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state):
            kinds = [ir.StringAttr(kind).value for kind in op.kinds]
            if any(kind not in sa.SUBGRAPH_MATCHERS for kind in kinds):
                op.location.emit_error(f"prepare_subgraphs: unknown kind in {kinds}")
                return DiagnosedSilenceableFailure.DefiniteFailure
            targets = list(state.get_payload_ops(op.target))
            if any(
                len(target.regions) != 1 or len(target.regions[0].blocks) != 1
                for target in targets
            ):
                op.location.emit_error(
                    "prepare_subgraphs: expected targets with one single-block region"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            for target in targets:
                _prepare(target.regions[0].blocks[0], kinds, rewriter)
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def prepare_subgraphs(target: ir.Value, kinds: list[str]) -> ir.Value:
    """Prepare inputs for tag_subgraphs with the same ordered kinds."""
    return PrepareSubgraphsOp(
        target=target,
        kinds=ir.ArrayAttr.get([ir.StringAttr.get(kind) for kind in kinds]),
    ).result
