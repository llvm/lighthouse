"""Temporary mirror of MLIR's linalg ``FoldScalarOrSplatConstant`` pattern."""

from mlir import ir
from mlir.dialects import arith, ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.utils.mlir import opview

_REBUILT_ATTRS = ("indexing_maps", "iterator_types", "operandSegmentSizes")


def _scalar_constant(value):
    """The scalar attribute of a scalar or splat constant input, else None.

    Slices of a splat are the same splat, so look through extract_slice.
    """
    op = opview(value.owner) if isinstance(value, ir.OpResult) else None
    while isinstance(op, tensor.ExtractSliceOp):
        source = op.source
        op = opview(source.owner) if isinstance(source, ir.OpResult) else None
    if not isinstance(op, arith.ConstantOp):
        return None
    attr = op.value
    if isinstance(value.type, ir.ShapedType):
        if isinstance(attr, ir.DenseElementsAttr) and attr.is_splat:
            return attr.get_splat_value()
        return None
    return attr if isinstance(attr, (ir.FloatAttr, ir.IntegerAttr)) else None


def _fold(generic, rewriter) -> bool:
    if not all(
        isinstance(v.type, ir.RankedTensorType) for v in generic.outputs
    ) or not all(isinstance(r.type, ir.RankedTensorType) for r in generic.results):
        return False
    inputs = list(generic.inputs)
    constants = [_scalar_constant(value) for value in inputs]
    if all(c is None for c in constants):
        return False

    maps = list(generic.indexing_maps)
    kept = [i for i, c in enumerate(constants) if c is None]
    num_inputs = len(inputs)
    src_body = generic.regions[0].blocks[0]
    src_args = list(src_body.arguments)

    with ir.InsertionPoint(generic), generic.location:
        new = linalg.GenericOp(
            result_tensors=[r.type for r in generic.results],
            inputs=[inputs[i] for i in kept],
            outputs=list(generic.outputs),
            indexing_maps=ir.ArrayAttr.get([maps[i] for i in kept] + maps[num_inputs:]),
            iterator_types=generic.iterator_types,
        )
        attrs = generic.operation.attributes
        for name in attrs:
            if name not in _REBUILT_ATTRS:
                new.operation.attributes[name] = attrs[name]
        body = new.regions[0].blocks.append(
            *[src_args[i].type for i in kept],
            *[a.type for a in src_args[num_inputs:]],
        )
        new_args = list(body.arguments)
        value_map = {}
        with ir.InsertionPoint(body):
            for i, constant in enumerate(constants):
                if constant is not None:
                    value_map[src_args[i]] = arith.constant(src_args[i].type, constant)
            for position, i in enumerate(kept):
                value_map[src_args[i]] = new_args[position]
            for position, arg in enumerate(src_args[num_inputs:]):
                value_map[arg] = new_args[len(kept) + position]
            for op in src_body.operations:
                irr.clone_op_deep_with_map(op, value_map)
    rewriter.replace_op(generic, list(new.results))
    return True


def _fold_all(root, rewriter) -> None:
    generics = []

    def visit(operation):
        if operation.name == "linalg.generic":
            generics.append(operation.opview)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    # Folding never creates new candidates, so one pass reaches a fixed point.
    for generic in generics:
        _fold(generic, rewriter)


class FoldScalarOrSplatConstantOp(
    TransformExtensionDialect.Operation, name="fold_scalar_or_splat_constant"
):
    """Inline scalar and splat-constant inputs of linalg.generic into its body.

    Temporary replacement for MLIR's ``FoldScalarOrSplatConstant``, which is
    only reachable through the broad elementwise-fusion pattern set. Each input
    defined by a scalar ``arith.constant`` or a splat dense constant (possibly
    through ``tensor.extract_slice``) becomes a scalar ``arith.constant`` in
    the body, and the operand and its indexing map are dropped. This keeps
    constant tensors from being vectorized into reads and bufferized into
    ``memref.global``. Only tensor-semantics generics are rewritten. Dead
    constants are left to DCE.

    Consumes the container handle and invalidates nested handles. Reacquire
    payload handles through the returned container after this transform.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = [target.operation for target in state.get_payload_ops(op.target)]
            if len(targets) != 1 or targets[0].name not in (
                "func.func",
                "builtin.module",
            ):
                op.operation.location.emit_error(
                    "fold_scalar_or_splat_constant expects one func.func or "
                    "builtin.module"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            _fold_all(targets[0], rewriter)
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_scalar_or_splat_constant(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Inline scalar/splat-constant generic inputs as body constants."""
    return FoldScalarOrSplatConstantOp(target=target).result
