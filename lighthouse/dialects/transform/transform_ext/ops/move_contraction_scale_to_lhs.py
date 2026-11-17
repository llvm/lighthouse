from typing import NamedTuple

from mlir import ir
from mlir.dialects import arith, ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.contraction_normalization import (
    _plain_contraction,
)
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_ID_ATTR_NAME,
    SUBGRAPH_KIND_ATTR_NAME,
)
from lighthouse.utils.mlir import (
    clone_op_with_map,
    defining_op,
    indexing_maps,
    is_linalg_all_loops_parallel,
    linalg_inputs,
    linalg_outputs,
    opview,
)


class MoveContractionScaleToLhsOp(
    TransformExtensionDialect.Operation, name="move_contraction_scale_to_lhs"
):
    """
    Move a uniform scalar scale from a contraction's RHS producer to its LHS.

    Rewrites ``contract(sl * A, sr * B)`` into ``contract((sl * sr) * A, B)``,
    which is legal because a multiply-accumulate is linear in each operand.
    In flash attention the RHS (K) is loaded on every reduction iteration while
    the LHS (Q) is loaded once, so this removes the per-iteration scaling.

    A scale producer is an all-parallel ``linalg.generic`` with one input, one
    output of the same type, identity maps, and a body ``mulf(in, s)`` where ``s``
    does not depend on the block arguments. The LHS scale is optional; the RHS
    scale is required. A new LHS scale op is created and the contraction is
    rewired to the unscaled RHS. Replaced scale ops are erased once unused.
    Subgraph tags on the contraction are copied to the new ops. If the RHS is not
    scaled, or the LHS shape is dynamic or its element type differs from the
    RHS, the payload is left unchanged and `scaled_lhs` is empty. A silenceable
    failure is emitted if the handle does not hold exactly one two-input
    multiply-accumulate linalg.generic.

    Args:
        contraction: Handle to a two-input multiply-accumulate linalg.generic.
    Returns:
        scaled_lhs: The new linalg.generic scaling the LHS, or empty.
    """

    contraction: ext.Operand[transform.AnyOpType]
    scaled_lhs: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "MoveContractionScaleToLhsOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            def reject(message: str) -> DiagnosedSilenceableFailure:
                op.operation.location.emit_error(
                    f"cannot move the contraction scale to the lhs: {message}"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            targets = state.get_payload_ops(op.contraction)
            if len(targets) != 1:
                return reject(
                    f"expected exactly one payload op for 'contraction', got "
                    f"{len(targets)}"
                )
            contraction = opview(targets[0])
            if not _plain_contraction(contraction):
                return reject("expected a two-input multiply-accumulate linalg.generic")

            lhs, rhs = linalg_inputs(contraction)
            rhs_scale = _scalar_scale(rhs)
            lhs_scale = _scalar_scale(lhs)
            if (
                rhs_scale is None
                or not ir.ShapedType(lhs.type).has_static_shape
                or ir.ShapedType(lhs.type).element_type
                != ir.ShapedType(rhs.type).element_type
            ):
                # Nothing to move: leave the payload unchanged.
                results.set_ops(op.scaled_lhs, [])
                return DiagnosedSilenceableFailure.Success

            scaled = _emit_lhs_scale(contraction, lhs, lhs_scale, rhs_scale)
            contraction.operands[0] = scaled.results[0]
            contraction.operands[1] = rhs_scale.input
            erased = []
            for old in (lhs_scale, rhs_scale):
                if old is None or any(old.op.operation == e for e in erased):
                    continue
                if not list(old.op.results[0].uses):
                    erased.append(old.op.operation)
                    rewriter.erase_op(old.op)
            results.set_ops(op.scaled_lhs, [scaled.operation])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "MoveContractionScaleToLhsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def move_contraction_scale_to_lhs(
    contraction: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create MoveContractionScaleToLhsOp."""
    return MoveContractionScaleToLhsOp(contraction=contraction).scaled_lhs


class _ScalarScale(NamedTuple):
    #: The scale producer.
    op: ir.OpView
    #: The value the producer scales.
    input: ir.Value
    #: The scalar factor, possibly computed in the producer body.
    scale: ir.Value
    #: Body ops computing `scale`, in order.
    scale_ops: list[ir.Operation]


def _scalar_scale(value: ir.Value) -> _ScalarScale | None:
    """The elementwise ``value = input * s`` producer of `value`, if any."""
    producer = defining_op(value)
    if producer is None:
        return None
    producer = opview(producer)
    if not isinstance(producer, linalg.GenericOp):
        return None
    inputs, outputs = linalg_inputs(producer), linalg_outputs(producer)
    if len(inputs) != 1 or len(outputs) != 1:
        return None
    if not is_linalg_all_loops_parallel(producer):
        return None
    if not all(_is_identity(imap) for imap in indexing_maps(producer)):
        return None
    if inputs[0].type != value.type:
        return None

    body = producer.regions[0].blocks[0]
    ops = [o.operation for o in body.operations]
    yielded = ops[-1].operands[0]
    mul = defining_op(yielded)
    if mul is None or not isinstance(opview(mul), arith.MulFOp):
        return None
    arg = body.arguments[0]
    lhs, rhs = mul.operands
    if lhs == arg:
        scale = rhs
    elif rhs == arg:
        scale = lhs
    else:
        return None

    scale_ops = [o for o in ops[:-1] if o != mul]
    args = set(body.arguments)
    for scale_op in scale_ops:
        if isinstance(scale_op.opview, linalg.IndexOp) or len(scale_op.regions):
            return None
        if any(operand in args for operand in scale_op.operands):
            return None
    if scale in args:
        return None
    return _ScalarScale(producer, inputs[0], scale, scale_ops)


def _is_identity(imap: ir.AffineMap) -> bool:
    return imap == ir.AffineMap.get_identity(imap.n_dims)


def _emit_lhs_scale(contraction, lhs, lhs_scale, rhs_scale) -> ir.OpView:
    """Emit ``input * (sl * sr)`` for the contraction's lhs."""
    source = lhs_scale.input if lhs_scale is not None else lhs
    shaped = ir.ShapedType(lhs.type)
    rank = shaped.rank
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    with ir.InsertionPoint(contraction), contraction.location:
        empty = tensor.empty(shaped.shape, shaped.element_type)
        scaled = linalg.GenericOp(
            result_tensors=[lhs.type],
            inputs=[source],
            outputs=[empty],
            indexing_maps=ir.ArrayAttr.get([identity, identity]),
            iterator_types=ir.ArrayAttr.get([parallel] * rank),
        )
        block = scaled.regions[0].blocks.append(
            shaped.element_type, shaped.element_type
        )
        with ir.InsertionPoint(block):
            factor = None
            if lhs_scale is not None:
                factor = _clone_scale(lhs_scale)
            rhs_factor = _clone_scale(rhs_scale)
            factor = rhs_factor if factor is None else arith.mulf(factor, rhs_factor)
            linalg.yield_([arith.mulf(block.arguments[0], factor)])

    for name in (SUBGRAPH_ID_ATTR_NAME, SUBGRAPH_KIND_ATTR_NAME):
        if name in contraction.operation.attributes:
            attr = contraction.operation.attributes[name]
            scaled.operation.attributes[name] = attr
            defining_op(empty).attributes[name] = attr
    return scaled


def _clone_scale(scale: _ScalarScale) -> ir.Value:
    """Clone the body ops computing `scale.scale` at the insertion point."""
    value_map = {}
    for op in scale.scale_ops:
        clone_op_with_map(op, value_map)
    return value_map.get(scale.scale, scale.scale)
