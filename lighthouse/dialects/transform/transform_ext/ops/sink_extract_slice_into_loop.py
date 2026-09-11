from mlir import ir
from mlir.dialects import arith, ext, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _mixed(static_arr, dyn_ops: list[ir.Value]) -> list:
    """Interleave static entries with dynamic operands into a mixed list.

    Static positions become Python ints, dynamic positions (marked with the
    kDynamic sentinel) become the corresponding ir.Value from `dyn_ops`.
    """
    kdyn = ir.ShapedType.get_dynamic_size()
    result: list = []
    di = 0
    for s in static_arr:
        s = int(s)
        if s == kdyn:
            result.append(dyn_ops[di])
            di += 1
        else:
            result.append(s)
    return result


def _split(mixed: list) -> tuple[list[int], list[ir.Value]]:
    """Split a mixed list back into (static_array, dynamic_operands)."""
    kdyn = ir.ShapedType.get_dynamic_size()
    static: list[int] = []
    dyn: list[ir.Value] = []
    for m in mixed:
        if isinstance(m, ir.Value):
            static.append(kdyn)
            dyn.append(m)
        else:
            static.append(int(m))
    return static, dyn


def _as_val(x) -> ir.Value:
    if isinstance(x, ir.Value):
        return x
    idx = ir.IndexType.get()
    return arith.ConstantOp(ir.IntegerAttr.get(idx, int(x))).result


def _add(a, b):
    if not isinstance(a, ir.Value) and not isinstance(b, ir.Value):
        return int(a) + int(b)
    if not isinstance(a, ir.Value) and int(a) == 0:
        return b
    if not isinstance(b, ir.Value) and int(b) == 0:
        return a
    return arith.AddIOp(_as_val(a), _as_val(b)).result


def _mul(a, b):
    if not isinstance(a, ir.Value) and not isinstance(b, ir.Value):
        return int(a) * int(b)
    if not isinstance(a, ir.Value):
        if int(a) == 0:
            return 0
        if int(a) == 1:
            return b
    if not isinstance(b, ir.Value):
        if int(b) == 0:
            return 0
        if int(b) == 1:
            return a
    return arith.MulIOp(_as_val(a), _as_val(b)).result


def _op_name(owner) -> str | None:
    """Operation name for a Value.owner (OpView) or Operation, else None."""
    operation = getattr(owner, "operation", None)
    return operation.name if operation is not None else None


def _is_ancestor(ancestor: ir.Operation, descendant: ir.Operation) -> bool:
    parent = descendant.parent
    while parent is not None:
        if parent == ancestor:
            return True
        parent = parent.parent
    return False


def _same_spec(a: ir.OpView, b: ir.OpView) -> bool:
    """Whether two offset/size/stride slice ops address the same region."""
    return (
        list(a.static_offsets) == list(b.static_offsets)
        and list(a.static_sizes) == list(b.static_sizes)
        and list(a.static_strides) == list(b.static_strides)
        and list(a.offsets) == list(b.offsets)
        and list(a.sizes) == list(b.sizes)
        and list(a.strides) == list(b.strides)
    )


def _match(for_op: ir.OpView):
    """Return (for_op, ext, ins) if `for_op` matches the sink pattern, else None."""
    if len(for_op.results) != 1:
        return None

    # Single loop-carried init must be a non-rank-reducing tensor.extract_slice.
    init = for_op.operands[3]
    ext = init.owner
    if _op_name(ext) != "tensor.extract_slice":
        return None
    src_ty = ext.source.type
    res_ty = ext.result.type
    if not (
        isinstance(src_ty, ir.RankedTensorType)
        and isinstance(res_ty, ir.RankedTensorType)
    ):
        return None
    if src_ty.rank != res_ty.rank:
        return None

    # The loop result must be consumed by exactly one tensor.insert_slice that
    # writes it back into the same source slice the init was extracted from.
    loop_res = for_op.results[0]
    uses = list(loop_res.uses)
    if len(uses) != 1:
        return None
    user = uses[0].owner
    if _op_name(user) != "tensor.insert_slice":
        return None
    ins = user
    if ins.source != loop_res or ins.dest != ext.source:
        return None
    if not _same_spec(ext, ins):
        return None

    # The iter_arg may only be used as an insert_slice dest, extract_slice
    # source, or yielded; anything else cannot be safely rebased.
    old_iter = for_op.regions[0].blocks[0].arguments[1]
    for use in old_iter.uses:
        owner = use.owner
        if owner.operation.parent != for_op.operation:
            # nested-region uses of the loop iter_arg are unsupported
            return None
        name = _op_name(owner)
        if name == "scf.yield":
            continue
        if (
            name == "tensor.insert_slice"
            and owner.operands[1] == old_iter
            and owner.operands[0] != old_iter
        ):
            continue
        if name == "tensor.extract_slice" and owner.operands[0] == old_iter:
            continue
        return None

    return for_op, ext, ins


def _sink_one(
    for_op: ir.OpView,
    ext: ir.OpView,
    ins: ir.OpView,
    rewriter: transform.TransformRewriter,
) -> ir.Operation:
    src = ext.source
    full_ty = src.type
    old_block = for_op.regions[0].blocks[0]
    old_iv = old_block.arguments[0]
    old_iter = old_block.arguments[1]

    outer_off = _mixed(ext.static_offsets, list(ext.offsets))
    outer_str = _mixed(ext.static_strides, list(ext.strides))

    lb, ub, step = for_op.operands[0], for_op.operands[1], for_op.operands[2]

    with ir.InsertionPoint(for_op), for_op.location:
        new_for = ir.Operation.create(
            "scf.for",
            results=[full_ty],
            operands=[lb, ub, step, src],
            regions=1,
        )
        new_block = new_for.regions[0].blocks.append(ir.IndexType.get(), full_ty)
        new_iv = new_block.arguments[0]
        new_iter = new_block.arguments[1]

        value_map: dict = {old_iv: new_iv, old_iter: new_iter}

        def remap(v):
            return value_map.get(v, v)

        def rebase(mixed: list) -> list:
            return [remap(x) if isinstance(x, ir.Value) else x for x in mixed]

        with ir.InsertionPoint(new_block):
            for op in list(old_block.operations):
                op = op.operation
                name = op.name

                if name == "scf.yield":
                    ir.Operation.create(
                        "scf.yield", operands=[remap(o) for o in op.operands]
                    )
                    continue

                if name == "tensor.insert_slice" and op.operands[1] == old_iter:
                    opv = op.opview
                    inner_off = rebase(_mixed(opv.static_offsets, list(opv.offsets)))
                    inner_str = rebase(_mixed(opv.static_strides, list(opv.strides)))
                    abs_off = [
                        _add(oo, _mul(io, os))
                        for oo, io, os in zip(outer_off, inner_off, outer_str)
                    ]
                    abs_str = [_mul(os, i) for os, i in zip(outer_str, inner_str)]
                    stat_off, dyn_off = _split(abs_off)
                    stat_str, dyn_str = _split(abs_str)
                    new_ins = tensor.InsertSliceOp(
                        remap(op.operands[0]),
                        new_iter,
                        dyn_off,
                        [remap(s) for s in opv.sizes],
                        dyn_str,
                        stat_off,
                        list(opv.static_sizes),
                        stat_str,
                    )
                    value_map[op.results[0]] = new_ins.result
                    continue

                if name == "tensor.extract_slice" and op.operands[0] == old_iter:
                    opv = op.opview
                    inner_off = rebase(_mixed(opv.static_offsets, list(opv.offsets)))
                    inner_str = rebase(_mixed(opv.static_strides, list(opv.strides)))
                    abs_off = [
                        _add(oo, _mul(io, os))
                        for oo, io, os in zip(outer_off, inner_off, outer_str)
                    ]
                    abs_str = [_mul(os, i) for os, i in zip(outer_str, inner_str)]
                    stat_off, dyn_off = _split(abs_off)
                    stat_str, dyn_str = _split(abs_str)
                    new_ext = tensor.ExtractSliceOp(
                        opv.result.type,
                        new_iter,
                        dyn_off,
                        [remap(s) for s in opv.sizes],
                        dyn_str,
                        stat_off,
                        list(opv.static_sizes),
                        stat_str,
                    )
                    value_map[op.results[0]] = new_ext.result
                    continue

                new_op = op.clone()
                for i, operand in enumerate(new_op.operands):
                    new_op.operands[i] = remap(operand)
                value_map.update(zip(op.results, new_op.results))

    # Splice the new loop in place of the extract/loop/insert triple.
    ins.result.replace_all_uses_with(new_for.results[0])
    rewriter.erase_op(ins)
    rewriter.erase_op(for_op)
    if len(list(ext.result.uses)) == 0:
        rewriter.erase_op(ext)

    return new_for


class SinkExtractSliceIntoLoopOp(
    TransformExtensionDialect.Operation, name="sink_extract_slice_into_loop"
):
    """
    Sinks an scf.for's init tensor.extract_slice into the loop.

    Takes handles to scf.for loops directly. For each loop whose single
    loop-carried init is a non-rank-reducing tensor.extract_slice and whose
    result is written straight back into the same slice with a trailing
    tensor.insert_slice, rewrites the loop to carry the full tensor instead,
    composing the extract offsets/strides into the body's insert/extract ops,
    and removes the surrounding extract/insert pair. Loops that do not match
    are left unchanged.

    Args:
        target: Handle to one or more scf.for ops to rewrite.
    Returns:
        Handle to the resulting scf.for ops (rewritten or unchanged).
    """

    target: ext.Operand[transform.AnyOpType]
    updated_ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "SinkExtractSliceIntoLoopOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = []
            seen: set[ir.Operation] = set()
            for target in state.get_payload_ops(op.target):
                if target in seen:
                    continue
                seen.add(target)
                targets.append(target)

            candidates = [
                _match(target.opview) if _op_name(target) == "scf.for" else None
                for target in targets
            ]
            for target, cand in zip(targets, candidates):
                if cand is None:
                    continue
                if any(
                    other != target
                    and _op_name(other) == "scf.for"
                    and _is_ancestor(target, other)
                    for other in targets
                ):
                    return DiagnosedSilenceableFailure.emit_silenceable_error(
                        "rewriting an scf.for containing another target is unsupported"
                    )

            updated = []
            for target, cand in zip(targets, candidates):
                if cand is None:
                    updated.append(target)
                    continue
                with target.context, target.location:
                    updated.append(_sink_one(*cand, rewriter))

            results.set_ops(op.updated_ops, updated)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "SinkExtractSliceIntoLoopOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return [
                *transform.consumes_handle(op.op_operands),
                *transform.produces_handle(op.results),
                *transform.modifies_payload(),
            ]


def sink_extract_slice_into_loop(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a SinkExtractSliceIntoLoopOp.

    `target` is a handle to one or more scf.for ops.
    """
    op = SinkExtractSliceIntoLoopOp(target=target)
    return op.updated_ops
