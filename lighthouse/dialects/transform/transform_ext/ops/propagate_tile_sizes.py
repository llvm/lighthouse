from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure
from collections import deque
from collections.abc import Iterator, Sequence

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils import tile_propagation as tp
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa
from lighthouse.utils.mlir import op_users, defining_op


class PropagateTileSizesOp(
    TransformExtensionDialect.Operation, name="propagate_tile_sizes"
):
    """
    Propagate tile-size annotations from anchor ops to their neighbors.

    Starting from the annotated `root` ops, tile sizes are spread to surrounding
    non-barrier ops that share a tensor, translating sizes across each op's indexing
    maps so a shared tensor is tiled consistently on both sides.

    Two phases, so a barrier's epilogue wins over a downstream barrier's prologue
    for a shared op:

      1. forward (epilogue): follow consumers from the anchors, stopping at the
         next barrier. An op between two barriers is tiled to match its producer.
      2. backward (prologue): follow producers from all annotated ops to claim the
         remaining prologue ops (e.g. fills, and producers behind epilogue inputs).

    Barriers are never re-tiled. Reduction dimensions are never tiled, and already-annotated
    ops keep their sizes.

    By default, `scf.for` loops act as barriers: propagation does not cross into or
    out of a loop body. Set `propagate_through_loops` to bridge a value through a
    loop's iter args/results and continue propagating on the other side.

    Args:
        root: Handle to annotated anchor op(s).
        propagate_through_loops: Optional bool (default: false). When true, tile
            sizes are propagated through `scf.for` loop-carried values instead of
            stopping at the loop.
    Return:
        Handle to all annotated ops after propagation (roots plus newly annotated).
    """

    root: ext.Operand[transform.AnyOpType]
    propagate_through_loops: ext.Operand[transform.AnyParamType] | None = None
    annotated: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "PropagateTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            """Run the two-phase (forward then backward) tile-size propagation."""
            root_ops = list(state.get_payload_ops(op.root))

            through_loops = False
            if op.propagate_through_loops is not None:
                param_attr = state.get_params(op.propagate_through_loops)
                if len(param_attr) == 1 and isinstance(param_attr[0], ir.BoolAttr):
                    through_loops = bool(param_attr[0])

            # An op is "visited" once it carries an annotation, which prevents
            # re-processing and acts as a barrier. Track ordered, de-duplicated
            # annotated ops for the result handle.
            annotated: list[ir.Operation] = []
            seen: set[ir.Operation] = set()

            def remember(target_op: ir.Operation) -> None:
                """Record `target_op` in `annotated`, once, for the result handle."""
                if target_op not in seen:
                    seen.add(target_op)
                    annotated.append(target_op)

            def claim(
                src: ir.Operation,
                src_sizes: Sequence[int],
                src_shared: ir.Value,
                dst_shared: ir.Value,
                dst: ir.Operation,
            ) -> ir.Operation | None:
                """Try to tile `dst` from `src`'s sizes across their shared tensor.

                `src`'s sizes are projected onto the shared tensor and mapped into
                `dst`'s iteration space, so both sides tile the tensor identically.

                `dst` is left untouched when it is a barrier, when it already carries
                an annotation, or when no non-zero sizes can be derived for it. An
                already-annotated `dst` that disagrees with `src` on the shared tensor
                belongs to a different fusion group: the consumer side is marked as a
                fusion boundary so grouping can split the two cheaply later.

                Args:
                    src: Annotated op the tile sizes are propagated from.
                    src_sizes: `src`'s tile sizes, in its loop order.
                    src_shared: Value of the shared tensor as seen by `src`.
                    dst_shared: Value of the shared tensor as seen by `dst`. May
                        differ from `src_shared` when the tensor crosses a wrapper
                        op such as `scf.yield`.
                    dst: Neighboring op to annotate.
                Returns:
                    `dst` if it was newly annotated, else None.
                """
                dst_sizes = tsa.get_tile_sizes_attr(dst)
                if dst_sizes is not None:
                    # `dst` already carries a tiling. If it disagrees with `src`
                    # on how the shared tensor is tiled, they belong to different
                    # fusion groups; mark the consumer side (the op reading the
                    # shared tensor) as a boundary so grouping can split them
                    # cheaply without recomputing compatibility.
                    if not tp.compatible_on_values(
                        src,
                        src_sizes,
                        src_shared,
                        dst,
                        dst_sizes,
                        dst_shared,
                    ):
                        consumer = (
                            src if any(r == dst_shared for r in dst.results) else dst
                        )
                        fa.mark_fusion_boundary(consumer)
                    return None
                if not tp.is_propagatable(dst):
                    return None
                dst_sizes = tp.propagate_through_values(
                    src,
                    src_sizes,
                    src_shared,
                    dst_shared,
                    dst,
                )
                if dst_sizes is None or not any(dst_sizes):
                    return None
                tsa.set_tile_sizes_attr(dst, dst_sizes)
                remember(dst)
                return dst

            def forward_loop_passthrough(
                value: ir.Value, user: ir.Operation
            ) -> ir.Value | None:
                """Follow `value` forward through a loop's body to the loop's result."""
                # Bridge `%v` -> `scf.yield %v` -> `%for_result` for scf.for.
                if user.operation.name != "scf.yield":
                    return None
                parent = user.operation.parent
                if parent is None or parent.name != "scf.for":
                    return None
                operands = list(user.opview.operands)
                try:
                    idx = operands.index(value)
                except ValueError:
                    return None
                results = list(parent.opview.results)
                return results[idx] if idx < len(results) else None

            def backward_loop_passthrough(value: ir.Value) -> ir.Value | None:
                """Follow `value` backward from a loop-carried argument to its initial value."""
                # Bridge iter arg -> its initial value in the enclosing scf.for.
                block = value.owner
                if not isinstance(block, ir.Block):
                    return None
                parent_op = block.owner.operation
                if parent_op.name != "scf.for":
                    return None
                # arg 0 is the induction variable; iter args start at 1.
                args = list(block.arguments)
                try:
                    arg_num = args.index(value)
                except ValueError:
                    return None
                if arg_num == 0:
                    return None
                # scf.for operands: lb(0), ub(1), step(2), iter_init_0(3), ...
                init_index = arg_num + 2
                operands = list(parent_op.opview.operands)
                if init_index >= len(operands):
                    return None
                return operands[init_index]

            def forward_neighbors(
                src_shared: ir.Value,
            ) -> Iterator[tuple[ir.Value, ir.Value, ir.Operation]]:
                """Yield `(src_shared, dst_shared, user)` for each propagatable
                consumer of `src_shared`, transparently following it through loops.
                """
                queue = deque([src_shared])
                seen_values = {src_shared}
                while queue:
                    value = queue.popleft()
                    for user in op_users(value):
                        if tp.is_propagatable(user):
                            yield src_shared, value, user
                            continue
                        if not through_loops:
                            continue
                        next_from_loop = forward_loop_passthrough(value, user)
                        if next_from_loop is not None:
                            if next_from_loop in seen_values:
                                continue
                            seen_values.add(next_from_loop)
                            queue.append(next_from_loop)

            def backward_neighbors(
                src_shared: ir.Value,
            ) -> Iterator[tuple[ir.Value, ir.Value, ir.Operation]]:
                """Yield `(src_shared, dst_shared, producer)` for each propagatable
                producer of `src_shared`, transparently following it through loops.
                """
                queue = deque([src_shared])
                seen_values = {src_shared}
                while queue:
                    value = queue.popleft()
                    producer = defining_op(value)
                    if producer is None:
                        if not through_loops:
                            continue
                        prev_from_loop = backward_loop_passthrough(value)
                        if prev_from_loop is not None:
                            if prev_from_loop in seen_values:
                                continue
                            seen_values.add(prev_from_loop)
                            queue.append(prev_from_loop)
                        continue
                    if tp.is_propagatable(producer):
                        yield src_shared, value, producer

            seeds = [r for r in root_ops if tsa.get_tile_sizes_attr(r) is not None]
            for seed in seeds:
                remember(seed)

            # Phase 1: forward (epilogue) propagation from the anchors.
            forward: list[ir.Operation] = list(seeds)
            idx = 0
            while idx < len(forward):
                src = forward[idx]
                idx += 1
                src_sizes = tsa.get_tile_sizes_attr(src)
                for result in src.opview.results:
                    for src_shared, dst_shared, user in forward_neighbors(result):
                        dst = claim(src, src_sizes, src_shared, dst_shared, user)
                        if dst is not None:
                            forward.append(dst)

            # Phase 2: backward (prologue) propagation from all annotated ops.
            # Seeding from every annotated op to reach other epilogue producers.
            # Propagation still stops at barriers, so it never leaks into upstream
            # groups.
            backward: list[ir.Operation] = list(annotated)
            idx = 0
            while idx < len(backward):
                src = backward[idx]
                idx += 1
                src_sizes = tsa.get_tile_sizes_attr(src)
                for operand in src.opview.operands:
                    for src_shared, dst_shared, producer in backward_neighbors(operand):
                        dst = claim(src, src_sizes, src_shared, dst_shared, producer)
                        if dst is not None:
                            backward.append(dst)

            results.set_ops(op.annotated, annotated)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "PropagateTileSizesOp") -> bool:
            """Disallow the same payload op from being passed via multiple handles."""
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def propagate_tile_sizes(
    root: ir.Value[transform.AnyOpType],
    propagate_through_loops: bool | ir.Value = False,
) -> ir.Value:
    """
    snake_case wrapper to create a PropagateTileSizesOp.

    Args:
        root: Handle to annotated anchor op(s).
        propagate_through_loops: Whether to propagate through `scf.for` loop-carried
            values instead of treating loops as barriers (default: False).
    Returns:
        Handle to all ops carrying a tile-size annotation after propagation.
    """
    if isinstance(propagate_through_loops, bool):
        param_attr = ir.BoolAttr.get(propagate_through_loops)
        propagate_through_loops = transform.ParamConstantOp(
            transform.AnyParamType.get(), param_attr
        )

    return PropagateTileSizesOp(
        root=root,
        propagate_through_loops=propagate_through_loops,
    ).annotated
