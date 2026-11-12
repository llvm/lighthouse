from mlir import ir
from mlir.dialects import ext, transform, vector, xegpu
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _collect_epilogue_load_annotations(
    store,
) -> list[tuple[ir.Operation, str, ir.Attribute]]:
    layout = store.attributes["layout"]
    shape = list(ir.VectorType(store.operands[0].type).shape)
    pending = [store.operands[0]]
    visited = set()
    annotations = []
    while pending:
        value = pending.pop()
        if value in visited or isinstance(value.owner, ir.Block):
            continue
        visited.add(value)
        op = value.owner.operation
        if isinstance(op.opview, (xegpu.LoadGatherOp, xegpu.LoadNdOp)):
            if list(ir.VectorType(value.type).shape) == shape:
                annotations.append((op, "layout", layout))
        elif isinstance(op.opview, vector.BroadcastOp):
            source = op.operands[0]
            if not isinstance(source.type, ir.VectorType):
                continue
            source_shape = list(source.type.shape)
            extra_dims = len(shape) - len(source_shape)
            if extra_dims <= 0 or shape[extra_dims:] != source_shape:
                continue
            if isinstance(source.owner, ir.Block):
                continue
            load = source.owner.operation
            if not isinstance(load.opview, (xegpu.LoadGatherOp, xegpu.LoadNdOp)):
                continue
            sliced = ir.Attribute.parse(
                f"#xegpu.slice<{layout}, dims = {list(range(extra_dims))}>"
            )
            annotations.extend(
                [
                    (op, "layout_result_0", layout),
                    (op, "layout_operand_0", sliced),
                    (load, "layout", sliced),
                ]
            )
        elif op.name.startswith(("arith.", "math.")) and not op.regions:
            pending.extend(
                operand
                for operand in op.operands
                if isinstance(operand.type, ir.VectorType)
                and list(operand.type.shape) == shape
            )
    return annotations


class AnnotateEpilogueLoadsOp(
    TransformExtensionDialect.Operation, name="annotate_epilogue_loads"
):
    """Assign store-derived layouts to ordinary and broadcast epilogue loads.

    Walks backwards through shape-preserving arithmetic/math from annotated
    store_nd ops. Same-shaped xegpu.load/load_nd ops get the store layout.
    A broadcast adding leading dimensions gets the store layout on its result
    and a slice of that layout on its directly feeding load and source operand.
    This covers elementwise inputs, 1D bias, and 2D RoPE tables without crossing
    DPAS ops or annotating their GEMM operand loads. Shape-changing paths,
    control flow, unrelated loads, and broadcasts stretching existing dims are
    not handled; these require separate layout propagation.
    Conflicting existing annotations fail rather than being overwritten.
    """

    target: ext.Operand[transform.AnyOpType]

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, _results, state) -> DiagnosedSilenceableFailure:
            planned = {}
            for store in state.get_payload_ops(op.target):
                if (
                    not isinstance(store, xegpu.StoreNdOp)
                    or "layout" not in store.attributes
                ):
                    store.location.emit_error(
                        "annotate_epilogue_loads: expected annotated store_nd"
                    )
                    return DiagnosedSilenceableFailure.DefiniteFailure
                for target, name, layout in _collect_epilogue_load_annotations(store):
                    key = (target, name)
                    existing = planned.get(key)
                    if existing is None and name in target.attributes:
                        existing = target.attributes[name]
                    if existing is not None and existing != layout:
                        target.location.emit_error(
                            "annotate_epilogue_loads: conflicting broadcast/load layout"
                        )
                        return DiagnosedSilenceableFailure.DefiniteFailure
                    planned[key] = layout
            for (target, name), layout in planned.items():
                target.attributes[name] = layout
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.modifies_payload()
            )


def annotate_epilogue_loads(target):
    return AnnotateEpilogueLoadsOp(target=target)
