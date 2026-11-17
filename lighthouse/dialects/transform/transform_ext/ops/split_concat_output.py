"""Compute concatenated output tiles piecewise to avoid vector shuffles."""

from dataclasses import dataclass
from math import prod

from mlir import ir
from mlir.dialects import affine, ext, linalg, scf, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import defining_op, indexing_maps


def _shape(value):
    ty = value.type
    if (
        not isinstance(ty, ir.RankedTensorType)
        or ty.encoding is not None
        or any(d <= 0 for d in ty.shape)
    ):
        return None
    return tuple(ty.shape)


def _generic_maps(op):
    if (
        len(op.results) != 1
        or any(str(it) != "#linalg.iterator_type<parallel>" for it in op.iterator_types)
        or any(
            nested.operation.name.split(".")[0] not in ("arith", "math", "complex")
            and nested.operation.name != "linalg.yield"
            or nested.regions
            for nested in op.regions[0].blocks[0].operations
        )
    ):
        return None
    maps = indexing_maps(op)
    if not maps[-1].is_permutation or any(
        m.n_symbols
        or not m.is_projected_permutation
        or any(not isinstance(e, ir.AffineDimExpr) for e in m.results)
        for m in maps
    ):
        return None
    return maps


def _partition(value, block, cache):
    """Find one concat partition, mapped into the output's dimensions."""
    if value not in cache:
        cache[value] = _find_partition(value, block, cache)
    return cache[value]


def _find_partition(value, block, cache):
    owner = defining_op(value)
    if owner is None or owner.block != block or _shape(value) is None:
        return None
    op = owner.opview
    if isinstance(op, tensor.ConcatOp):
        sizes = [_shape(v) for v in op.inputs]
        if any(s is None for s in sizes):
            return None
        dim = int(op.dim)
        cuts = [0]
        for shape in sizes:
            cuts.append(cuts[-1] + shape[dim])
        return dim, cuts
    if isinstance(op, linalg.TransposeOp):
        partition = _partition(op.operands[0], block, cache)
        if partition is not None:
            dim, cuts = partition
            return list(op.permutation).index(dim), cuts
    if isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)):
        partition = _partition(op.src, block, cache)
        if partition is not None:
            dim, cuts = partition
            groups = [[int(axis) for axis in group] for group in op.reassociation]
            if isinstance(op, tensor.ExpandShapeOp):
                group = groups[dim]
                if len(group) == 1:
                    return int(group[0]), cuts
            else:
                for i, group in enumerate(groups):
                    if group == [dim]:
                        return i, cuts
    if isinstance(op, linalg.GenericOp):
        maps = _generic_maps(op)
        if maps is None:
            return None
        for operand, mapping in zip(op.operands, maps):
            partition = _partition(operand, block, cache)
            if partition is not None:
                dim, cuts = partition
                loop_dim = mapping.results[dim]
                return list(maps[-1].results).index(loop_dim), cuts
    return None


@dataclass
class _Slice:
    value: ir.Value
    offsets: tuple
    sizes: tuple[int, ...]
    kind: str = "leaf"
    operands: tuple = ()


def _mixed_offsets(op):
    dynamic = iter(op.offsets)
    return tuple(
        next(dynamic) if ir.ShapedType.is_dynamic_size(int(o)) else int(o)
        for o in op.static_offsets
    )


def _build_slice_plan(value, offsets, sizes, block, cache) -> _Slice | None:
    key = value, offsets, sizes
    if key in cache:
        return cache[key]
    shape = _shape(value)
    if (
        shape is None
        or len(shape) != len(sizes)
        or any(
            o < 0 or s <= 0 or o + s > extent
            for o, s, extent in zip(offsets, sizes, shape)
        )
    ):
        return None
    owner = defining_op(value)
    plan = _Slice(value, offsets, sizes)
    if owner is None or owner.block != block:
        cache[key] = plan
        return plan
    op = owner.opview
    if isinstance(op, tensor.EmptyOp):
        plan.kind = "empty"
    elif isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)):
        source_shape = _shape(op.src)
        if source_shape is None:
            return None
        groups = [[int(dim) for dim in group] for group in op.reassociation]
        starts, extents = [], []
        if isinstance(op, tensor.ExpandShapeOp):
            for group in groups:
                if len(group) == 1:
                    starts.append(offsets[group[0]])
                    extents.append(sizes[group[0]])
                elif all(offsets[i] == 0 and sizes[i] == shape[i] for i in group):
                    starts.append(0)
                    extents.append(prod(sizes[i] for i in group))
                else:
                    return None
        else:
            for i, group in enumerate(groups):
                if len(group) == 1:
                    starts.append(offsets[i])
                    extents.append(sizes[i])
                elif offsets[i] == 0 and sizes[i] == shape[i]:
                    starts.extend([0] * len(group))
                    extents.extend(source_shape[j] for j in group)
                else:
                    return None
        child = _build_slice_plan(op.src, tuple(starts), tuple(extents), block, cache)
        if child is None:
            return None
        plan.kind, plan.operands = "reshape", (child,)
    elif isinstance(op, tensor.ExtractSliceOp):
        if (
            len(_shape(op.source) or ()) != len(shape)
            or op.sizes
            or op.strides
            or any(int(s) != 1 for s in op.static_strides)
        ):
            return None
        base = _mixed_offsets(op)
        source_owner = defining_op(op.source)
        if source_owner is None or source_owner.block != block:
            # Keep dynamic WG offsets symbolic until the rewrite is committed.
            combined = tuple(
                b + o if isinstance(b, int) else (b, o) for b, o in zip(base, offsets)
            )
            plan = _Slice(op.source, combined, sizes)
        else:
            if any(not isinstance(b, int) for b in base):
                return None
            child = _build_slice_plan(
                op.source,
                tuple(b + o for b, o in zip(base, offsets)),
                sizes,
                block,
                cache,
            )
            if child is None:
                return None
            plan.kind, plan.operands = "alias", (child,)
    elif isinstance(op, tensor.ConcatOp):
        dim, start = int(op.dim), 0
        for operand in op.inputs:
            extent = _shape(operand)
            if extent is None:
                return None
            end = start + extent[dim]
            if start <= offsets[dim] and offsets[dim] + sizes[dim] <= end:
                local = list(offsets)
                local[dim] -= start
                child = _build_slice_plan(operand, tuple(local), sizes, block, cache)
                if child is None:
                    return None
                plan.kind, plan.operands = "alias", (child,)
                break
            start = end
        else:
            return None
    elif isinstance(op, (linalg.GenericOp, linalg.TransposeOp)):
        if isinstance(op, linalg.GenericOp):
            maps = _generic_maps(op)
            if maps is None:
                return None
            dims = {
                expr.position: (offset, size)
                for expr, offset, size in zip(maps[-1].results, offsets, sizes)
            }
            tiles = [
                (
                    tuple(dims[e.position][0] for e in mapping.results),
                    tuple(dims[e.position][1] for e in mapping.results),
                )
                for mapping in maps
            ]
            unused_init = not list(op.regions[0].blocks[0].arguments[-1].uses)
        else:
            perm = list(op.permutation)
            tiles = [
                (
                    tuple(offsets[perm.index(i)] for i in range(len(perm))),
                    tuple(sizes[perm.index(i)] for i in range(len(perm))),
                ),
                (offsets, sizes),
            ]
            unused_init = True
        operands = []
        for i, (operand, (starts, extents)) in enumerate(zip(op.operands, tiles)):
            if not isinstance(operand.type, ir.ShapedType):
                operands.append(operand)
                continue
            if i == len(op.operands) - 1 and unused_init:
                child = _Slice(operand, starts, extents, "empty")
            else:
                child = _build_slice_plan(operand, starts, extents, block, cache)
            if child is None:
                return None
            operands.append(child)
        plan.kind, plan.operands = "clone", tuple(operands)
    else:
        return None
    cache[key] = plan
    return plan


def _offset(value, delta):
    if isinstance(value, int):
        return value + delta
    if delta == 0:
        return value
    return affine.AffineApplyOp(
        ir.AffineMap.get(1, 0, [ir.AffineExpr.get_dim(0) + delta]), [value]
    ).result


def _split_offsets(offsets):
    return (
        [o for o in offsets if isinstance(o, ir.Value)],
        [
            ir.ShapedType.get_dynamic_size() if isinstance(o, ir.Value) else o
            for o in offsets
        ],
    )


def _materialize(plan, cache):
    key = id(plan)
    if key in cache:
        return cache[key]
    ty = ir.RankedTensorType.get(plan.sizes, plan.value.type.element_type)
    if plan.kind == "alias":
        value = _materialize(plan.operands[0], cache)
    elif plan.kind == "empty":
        value = tensor.EmptyOp(plan.sizes, ty.element_type).result
    elif plan.kind == "clone":
        operands = [
            _materialize(operand, cache) if isinstance(operand, _Slice) else operand
            for operand in plan.operands
        ]
        cloned = defining_op(plan.value).clone()
        for i, operand in enumerate(operands):
            cloned.operands[i] = operand
        cloned.results[0].set_type(ty)
        value = cloned.results[0]
    elif plan.kind == "reshape":
        op = defining_op(plan.value).opview
        source = _materialize(plan.operands[0], cache)
        groups = [[int(dim) for dim in group] for group in op.reassociation]
        if isinstance(op, tensor.ExpandShapeOp):
            value = tensor.ExpandShapeOp(ty, source, groups, [], plan.sizes).result
        else:
            value = tensor.CollapseShapeOp(ty, source, groups).result
    else:
        offsets = [_offset(*o) if isinstance(o, tuple) else o for o in plan.offsets]
        if tuple(offsets) == (0,) * len(offsets) and _shape(plan.value) == plan.sizes:
            value = plan.value
        else:
            dynamic, static = _split_offsets(offsets)
            value = tensor.ExtractSliceOp(
                ty,
                plan.value,
                dynamic,
                [],
                [],
                static,
                plan.sizes,
                [1] * len(plan.sizes),
            ).result
    cache[key] = value
    return value


def _split(insert, rewriter):
    shape = _shape(insert.source)
    block = insert.operation.parent.block
    if (
        shape is None
        or len(insert.static_sizes) != len(shape)
        or insert.sizes
        or insert.strides
        or any(int(s) != 1 for s in insert.static_strides)
    ):
        return
    partition = _partition(insert.source, block, {})
    if partition is None:
        return
    dim, cuts = partition
    if len(cuts) < 3:
        return
    plans, cache = [], {}
    base = _mixed_offsets(insert)
    for start, end in zip(cuts, cuts[1:]):
        offsets, sizes = [0] * len(shape), list(shape)
        offsets[dim], sizes[dim] = start, end - start
        plan = _build_slice_plan(
            insert.source, tuple(offsets), tuple(sizes), block, cache
        )
        if plan is None:
            return
        plans.append((start, plan))
    with insert.location, ir.InsertionPoint(insert.operation.parent):
        materialized = {}
        values = [_materialize(plan, materialized) for _, plan in plans]
        tile = _materialize(_Slice(insert.dest, base, shape), materialized)
        for value, (start, plan) in zip(values, plans):
            offsets = [0] * len(shape)
            offsets[dim] = start
            tile = tensor.InsertSliceOp(
                value, tile, [], [], [], offsets, plan.sizes, [1] * len(shape)
            ).result
    with insert.location, ir.InsertionPoint(insert):
        dynamic, static = _split_offsets(base)
        tensor.ParallelInsertSliceOp(
            tile, insert.dest, dynamic, [], [], static, shape, [1] * len(shape)
        )
    rewriter.erase_op(insert.operation)


class SplitConcatOutputOp(
    TransformExtensionDialect.Operation, name="split_concat_output"
):
    """Split concat-dependent forall output tiles into disjoint output slices.

    Instead of assembling a concatenated tensor and computing its full output
    tile, compute each concat-dependent piece separately and insert it into
    the output tile. This lets vectorization use partial loads and stores
    rather than assembling and shuffling concatenated vectors in registers.

    The transform propagates static rectangular slices backward through
    tensor.concat, rank-preserving unit-stride tensor.extract_slice, parallel
    linalg.generic ops with projected-permutation maps, and linalg.transpose.
    Slices reach the WG input tensors instead of becoming register shuffles.
    Scalar bodies and their arithmetic ordering are preserved.

    Disjoint tensor.insert_slice ops update a WG output tile, followed by one
    parallel_insert_slice. Native insert-slice vectorization produces partial
    transfers, not concatenated vectors. Keeping one parallel insertion per
    shared output avoids spurious bufferization conflicts between insertions.

    Apply to a scf.forall or an operation containing foralls. Outputs that
    cannot be split safely are left unchanged. Run canonicalization and dead
    code elimination afterward to remove the original full-tile computations.

    An upstream implementation should use TilingInterface's result-tile
    generation for slice propagation, with concat folding and disjoint output
    insertion as separate rewrite patterns.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            inserts = {}

            def visit(nested):
                if isinstance(nested.opview, scf.ForallOp):
                    body = nested.regions[0].blocks[0]
                    terminator = list(body.operations)[-1]
                    for insert in terminator.regions[0].blocks[0].operations:
                        if isinstance(insert, tensor.ParallelInsertSliceOp):
                            inserts[insert.operation] = insert
                return ir.WalkResult.ADVANCE

            for target in targets:
                target.walk(visit)
            for insert in inserts.values():
                _split(insert, rewriter)
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


def split_concat_output(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    return SplitConcatOutputOp(target=target).result
