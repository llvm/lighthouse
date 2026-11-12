from mlir import ir
from mlir.dialects import arith, ext, tensor, transform, ub, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure
from mlir.passmanager import PassManager

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class FoldTensorVectorTransfersOp(
    TransformExtensionDialect.Operation, name="fold_tensor_vector_transfers"
):
    """Replace intermediate tensor/vector round trips with vector operations.

    After vectorization, a vector may be written into a tensor, reshaped,
    sliced or concatenated with other tensors, then read back into a vector.
    This op forwards the vector values through those tensor operations using
    vector shape casts and strided-slice extraction/insertion. It removes
    redundant tensor transfers and avoids materializing intermediate tensors
    as temporary buffers during bufferization.

    Complements native transfer forwarding with static, unmasked, minor-identity
    reads of reshapes/concats, reads contained in a preceding tensor write, and
    reshapes of fully overwritten tensors.
    No buffer aliasing or floating-point reassociation is involved. Unsupported
    transfers are left unchanged.

    Upstream's ``fold-tensor-subset-ops`` pass handles extract-slice/read and
    write/insert-slice forwarding between custom folding rounds, including
    reads introduced by reshape and concat forwarding. Targets must be isolated
    from above, as required by the pass manager.
    Consumes target handles and invalidates nested handles; reacquire them from
    the returned containers after rewriting.

    Upstreaming plan:
      1. Extend native read-after-write forwarding to handle contained reads.
      2. Add full read/write-through-tensor-reshape folding.
      3. Add concat-read vectorization as an opt-in pattern, since it changes
         representation and potentially vector sizes.
      4. Reuse upstream's existing extract-slice folding.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = list(state.get_payload_ops(op.target))
            for target in targets:
                _rewrite(target)
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


def _shape(value: ir.Value) -> list[int] | None:
    if not isinstance(value.type, ir.RankedTensorType):
        return None
    shape = list(value.type.shape)
    return shape if shape and all(dim > 0 for dim in shape) else None


def _minor_identity(rank: int, vector_rank: int) -> ir.AffineMap:
    return ir.AffineMap.get(
        rank, 0, [ir.AffineExpr.get_dim(i) for i in range(rank - vector_rank, rank)]
    )


def _constant(value: ir.Value) -> int | None:
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name != "arith.constant":
        return None
    attr = owner.attributes["value"]
    return ir.IntegerAttr(attr).value if isinstance(attr, ir.IntegerAttr) else None


def _region(
    op: vector.TransferReadOp | vector.TransferWriteOp,
) -> tuple[list[int], list[int]] | None:
    """Return a statically in-bounds rectangular transfer footprint."""
    shape = _shape(op.base)
    vector_type = ir.VectorType(
        op.result.type
        if isinstance(op, vector.TransferReadOp)
        else op.valueToStore.type
    )
    if (
        shape is None
        or vector_type.scalable
        or vector_type.element_type != ir.RankedTensorType(op.base.type).element_type
        or vector_type.rank > len(shape)
        or op.mask is not None
        or op.permutation_map.value != _minor_identity(len(shape), vector_type.rank)
    ):
        return None
    offsets = []
    for index in op.indices:
        offset = _constant(index)
        if offset is None:
            return None
        offsets.append(offset)
    sizes = [1] * (len(shape) - vector_type.rank) + list(vector_type.shape)
    if any(
        offset < 0 or offset + size > extent
        for offset, size, extent in zip(offsets, sizes, shape)
    ):
        return None
    return offsets, sizes


def _cast(value: ir.Value, shape: list[int]) -> ir.Value:
    ty = ir.VectorType.get(shape, ir.VectorType(value.type).element_type)
    return value if value.type == ty else vector.ShapeCastOp(ty, value).result


def _slice(value: ir.Value, offsets: list[int], sizes: list[int]) -> ir.Value:
    ty = ir.VectorType(value.type)
    if offsets == [0] * ty.rank and sizes == list(ty.shape):
        return value
    return vector.ExtractStridedSliceOp(
        ir.VectorType.get(sizes, ty.element_type),
        value,
        offsets,
        sizes,
        [1] * ty.rank,
    ).result


def _read(source: ir.Value, padding: ir.Value) -> ir.Value:
    ty = ir.RankedTensorType(source.type)
    zero = arith.ConstantOp(ir.IndexType.get(), 0).result
    return vector.TransferReadOp(
        ir.VectorType.get(ty.shape, ty.element_type),
        source,
        [zero] * ty.rank,
        _minor_identity(ty.rank, ty.rank),
        padding,
        [True] * ty.rank,
    ).result


def _forward(read: vector.TransferReadOp) -> ir.Value | None:
    region = _region(read)
    if region is None:
        return None
    offsets, sizes = region
    source = read.base
    owner = source.owner
    if isinstance(owner, ir.Block):
        return None
    owner = owner.operation
    if isinstance(owner.opview, vector.TransferWriteOp):
        write = owner.opview
        written = _region(write)
        if written is None:
            return None
        write_offsets, write_sizes = written
        relative = [r - w for r, w in zip(offsets, write_offsets)]
        if any(o < 0 or o + s > w for o, s, w in zip(relative, sizes, write_sizes)):
            return None
        value = _slice(_cast(write.valueToStore, write_sizes), relative, sizes)
        return _cast(value, list(ir.VectorType(read.result.type).shape))

    full_read = offsets == [0] * len(sizes) and sizes == _shape(source)
    if owner.name in ("tensor.expand_shape", "tensor.collapse_shape") and full_read:
        if _shape(owner.operands[0]) is None:
            return None
        return _cast(
            _read(owner.operands[0], read.padding),
            list(ir.VectorType(read.result.type).shape),
        )
    if isinstance(owner.opview, tensor.ConcatOp) and full_read:
        if any(_shape(operand) is None for operand in owner.operands):
            return None
        ty = ir.RankedTensorType(source.type)
        value = ub.PoisonOp(ir.VectorType.get(ty.shape, ty.element_type)).result
        dim = ir.IntegerAttr(owner.attributes["dim"]).value
        position = [0] * ty.rank
        for operand in owner.operands:
            value = vector.InsertStridedSliceOp(
                _read(operand, read.padding), value, position, [1] * ty.rank
            ).result
            position[dim] += ir.RankedTensorType(operand.type).shape[dim]
        return _cast(value, list(ir.VectorType(read.result.type).shape))
    return None


def _rewrite(target: ir.Operation) -> None:
    subset_folding = PassManager.parse("any(fold-tensor-subset-ops)")
    while True:
        subset_folding.run(target)
        reads, reshapes = [], []

        def collect(op):
            if isinstance(op.opview, vector.TransferReadOp):
                reads.append(op)
            elif op.name in ("tensor.expand_shape", "tensor.collapse_shape"):
                reshapes.append(op)
            return ir.WalkResult.ADVANCE

        target.walk(collect)
        changed = False
        for op in reads:
            parent = op.parent
            while parent is not None and parent.name != "vector.mask":
                parent = parent.parent
            if parent is not None:
                continue
            with ir.InsertionPoint(op), op.location:
                replacement = _forward(op.opview)
            if replacement is not None:
                op.results[0].replace_all_uses_with(replacement)
                op.erase()
                changed = True
        for reshape in reshapes:
            source = reshape.operands[0]
            owner = source.owner
            if (
                isinstance(owner, ir.Block)
                or owner.operation.name != "vector.transfer_write"
                or _shape(source) is None
                or _shape(reshape.results[0]) is None
                or ir.RankedTensorType(source.type).encoding is not None
                or ir.RankedTensorType(reshape.results[0].type).encoding is not None
            ):
                continue
            write = owner.operation.opview
            region = _region(write)
            if region != ([0] * len(_shape(source)), _shape(source)):
                continue
            with ir.InsertionPoint(reshape), reshape.location:
                shape = _shape(reshape.results[0])
                value = _cast(write.valueToStore, shape)
                dest = tensor.EmptyOp(
                    shape, ir.RankedTensorType(source.type).element_type
                ).result
                zero = arith.ConstantOp(ir.IndexType.get(), 0).result
                replacement = vector.TransferWriteOp(
                    dest.type,
                    value,
                    dest,
                    [zero] * len(shape),
                    _minor_identity(len(shape), len(shape)),
                    [True] * len(shape),
                ).result
            reshape.results[0].replace_all_uses_with(replacement)
            reshape.erase()
            changed = True
        if not changed:
            return


def fold_tensor_vector_transfers(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Fold tile-local tensor round-trips before bufferization."""
    return FoldTensorVectorTransfersOp(target=target).result
