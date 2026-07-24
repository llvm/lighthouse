from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

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

    Args:
        root: Handle to annotated anchor op(s).
    Return:
        Handle to all annotated ops after propagation (roots plus newly annotated).
    """

    root: ext.Operand[transform.AnyOpType]
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
            root_ops = list(state.get_payload_ops(op.root))

            # An op is "visited" once it carries an annotation, which prevents
            # re-processing and acts as a barrier. Track ordered, de-duplicated
            # annotated ops for the result handle.
            annotated: list[ir.Operation] = []
            seen: set = set()

            def remember(target_op: ir.Operation) -> None:
                key = target_op.operation.__hash__()
                if key not in seen:
                    seen.add(key)
                    annotated.append(target_op)

            def claim(
                src: ir.Operation, src_sizes, shared: ir.Value, dst: ir.Operation
            ) -> ir.Operation | None:
                dst_sizes = tsa.get_tile_sizes_attr(dst)
                if dst_sizes is not None:
                    # `dst` already carries a tiling. If it disagrees with `src`
                    # on how the shared tensor is tiled, they belong to different
                    # fusion groups; mark the consumer side (the op reading the
                    # shared tensor) as a boundary so grouping can split them
                    # cheaply without recomputing compatibility.
                    if not tp.compatible_on_value(
                        src, src_sizes, dst, dst_sizes, shared
                    ):
                        consumer = src if any(r == shared for r in dst.results) else dst
                        fa.mark_fusion_boundary(consumer)
                    return None
                if not tp.is_propagatable(dst):
                    return None
                dst_sizes = tp.propagate_through_value(src, src_sizes, shared, dst)
                if dst_sizes is None or not any(dst_sizes):
                    return None
                tsa.set_tile_sizes_attr(dst, dst_sizes)
                remember(dst)
                return dst

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
                    for user in op_users(result):
                        dst = claim(src, src_sizes, result, user)
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
                    producer = defining_op(operand)
                    if producer is None:
                        continue
                    dst = claim(src, src_sizes, operand, producer)
                    if dst is not None:
                        backward.append(dst)

            results.set_ops(op.annotated, annotated)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "PropagateTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def propagate_tile_sizes(
    root: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a PropagateTileSizesOp.

    Args:
        root: Handle to annotated anchor op(s).
    Returns:
        Handle to all ops carrying a tile-size annotation after propagation.
    """
    return PropagateTileSizesOp(root=root).annotated
