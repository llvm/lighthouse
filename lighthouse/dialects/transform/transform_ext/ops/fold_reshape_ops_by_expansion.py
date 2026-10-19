"""Temporary static-tensor subset of MLIR's reshape-by-expansion patterns."""

from math import prod

from mlir import ir
from mlir.dialects import arith, ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import indexing_maps, opview


def _producer(value):
    return opview(value.owner) if isinstance(value, ir.OpResult) else None


def _groups(reshape):
    return [[int(dim) for dim in group] for group in reshape.reassociation]


def _plan(generic, operand_number, reshape):
    maps = indexing_maps(generic)
    if (
        not generic.outputs
        or not all(isinstance(v.type, ir.RankedTensorType) for v in generic.outputs)
        or any(not m.is_projected_permutation or m.n_symbols for m in maps)
        or not maps[operand_number].results
        or any(op.regions for op in generic.regions[0].blocks[0].operations)
    ):
        return None

    extents = [None] * len(generic.iterator_types)
    for value, m in zip(generic.operands, maps):
        ty = value.type
        if not isinstance(ty, ir.RankedTensorType):
            if isinstance(ty, ir.ShapedType) or m.results:
                return None
            continue
        if ty.encoding is not None or any(size <= 0 for size in ty.shape):
            return None
        for expr, size in zip(m.results, ty.shape):
            dim = expr.position
            if extents[dim] is not None and extents[dim] != size:
                return None
            extents[dim] = size
    if any(size is None for size in extents):
        return None

    expanded_type = (
        reshape.src.type
        if isinstance(reshape, tensor.CollapseShapeOp)
        else reshape.result.type
    )
    if expanded_type.encoding is not None or any(s <= 0 for s in expanded_type.shape):
        return None
    shapes = [[size] for size in extents]
    for expr, group in zip(maps[operand_number].results, _groups(reshape)):
        shape = [expanded_type.shape[i] for i in group]
        if prod(shape) != extents[expr.position]:
            return None
        shapes[expr.position] = shape
    if all(len(shape) == 1 for shape in shapes):
        return None

    loop_groups = []
    num_loops = 0
    for shape in shapes:
        loop_groups.append(list(range(num_loops, num_loops + len(shape))))
        num_loops += len(shape)
    return maps, shapes, loop_groups, num_loops


def _expand(value, shape, groups):
    ty = value.type
    if not isinstance(ty, ir.RankedTensorType) or list(ty.shape) == shape:
        return value
    expanded_type = ir.RankedTensorType.get(shape, ty.element_type)
    producer = _producer(value)
    if (
        isinstance(producer, tensor.CollapseShapeOp)
        and producer.src.type == expanded_type
        and _groups(producer) == groups
    ):
        return producer.src
    return tensor.ExpandShapeOp(expanded_type, value, groups, [], shape).result


def _rewrite(generic, reshape, plan, rewriter):
    maps, shapes, loop_groups, num_loops = plan
    with generic.location, ir.InsertionPoint(generic):
        operands, new_maps, reassociations = [], [], []
        for value, m in zip(generic.operands, maps):
            shape, groups, exprs = [], [], []
            for expr in m.results:
                sizes = shapes[expr.position]
                groups.append(list(range(len(shape), len(shape) + len(sizes))))
                shape.extend(sizes)
                exprs.extend(
                    ir.AffineExpr.get_dim(d) for d in loop_groups[expr.position]
                )
            operands.append(_expand(value, shape, groups))
            new_maps.append(ir.AffineMapAttr.get(ir.AffineMap.get(num_loops, 0, exprs)))
            reassociations.append(groups)

        expanded = opview(generic.operation.clone())
        for i, value in enumerate(operands):
            expanded.operands[i] = value
        expanded.attributes["indexing_maps"] = ir.ArrayAttr.get(new_maps)
        expanded.attributes["iterator_types"] = ir.ArrayAttr.get(
            [it for it, shape in zip(generic.iterator_types, shapes) for _ in shape]
        )
        for result, output in zip(expanded.results, expanded.outputs):
            result.set_type(output.type)

        # An original index denotes the linearized coordinate, not just the
        # first of the expanded loop dimensions.
        for index in list(expanded.regions[0].blocks[0].operations):
            if not isinstance(index, linalg.IndexOp):
                continue
            dim = int(index.dim)
            with ir.InsertionPoint(index):
                coord = linalg.IndexOp(loop_groups[dim][0]).result
                for new_dim, size in zip(loop_groups[dim][1:], shapes[dim][1:]):
                    stride = arith.ConstantOp(ir.IndexType.get(), size)
                    coord = arith.AddIOp(
                        arith.MulIOp(coord, stride.result).result,
                        linalg.IndexOp(new_dim).result,
                    ).result
                rewriter.replace_op(index, [coord])

        replacements = []
        for i, (old, new) in enumerate(zip(generic.results, expanded.results)):
            if old.type == new.type:
                replacements.append(new)
            else:
                replacements.append(
                    tensor.CollapseShapeOp(
                        old.type, new, reassociations[len(generic.inputs) + i]
                    ).result
                )

    if isinstance(reshape, tensor.ExpandShapeOp):
        result_number = reshape.src.result_number
        rewriter.replace_op(reshape, [expanded.results[result_number]])
    rewriter.replace_op(generic, replacements)


def _fold_one(root, rewriter):
    candidates = []

    def visit(operation):
        if operation.name in ("linalg.generic", "tensor.expand_shape"):
            candidates.append(operation.opview)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    for candidate in candidates:
        if isinstance(candidate, linalg.GenericOp):
            for i, value in enumerate(candidate.inputs):
                reshape = _producer(value)
                if not isinstance(reshape, tensor.CollapseShapeOp):
                    continue
                plan = _plan(candidate, i, reshape)
                if plan is not None:
                    _rewrite(candidate, reshape, plan, rewriter)
                    return True
        else:
            generic = _producer(candidate.src)
            # Do not rewrite a producer outside the requested container.
            if not isinstance(generic, linalg.GenericOp) or generic not in candidates:
                continue
            i = len(generic.inputs) + candidate.src.result_number
            plan = _plan(generic, i, candidate)
            if plan is not None:
                _rewrite(generic, candidate, plan, rewriter)
                return True
    return False


class FoldReshapeOpsByExpansionOp(
    TransformExtensionDialect.Operation, name="fold_reshape_ops_by_expansion"
):
    """Fold tensor reshapes into generic loop dimensions without body fusion.

    Temporary replacement for the not-yet-exposed native pattern collection.
    Within one func.func or builtin.module, greedily fold collapse_shape inputs
    and expand_shape results of linalg.generic. All legal matches are accepted,
    including shared producers; collapsed results preserve other users.

    Supports positive static, unencoded tensors, projected-permutation indexing
    maps, scalar inputs and region-free bodies (including linalg.index).
    Dynamic/encoded tensors, non-projected maps, named ops, nested scalar regions
    and the native pad patterns are not implemented and are left unchanged.
    Run canonicalization afterwards to clean up introduced reshapes.

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
                    "fold_reshape_ops_by_expansion expects one func.func or builtin.module"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            while _fold_one(targets[0], rewriter):
                pass
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


def fold_reshape_ops_by_expansion(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Expand generic iteration spaces across static tensor reshape boundaries."""
    return FoldReshapeOpsByExpansionOp(target=target).result
