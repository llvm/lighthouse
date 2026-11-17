from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from mlir import ir
from mlir.dialects import arith, ext, scf, transform, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


@dataclass(frozen=True)
class _Layout:
    sg_layout: tuple[int, ...]
    sg_data: tuple[int, ...]
    inst_data: tuple[int, ...]
    order: tuple[int, ...]

    def transpose(self, permutation):
        return _Layout(
            *(
                tuple(field[i] for i in permutation)
                for field in (self.sg_layout, self.sg_data, self.inst_data)
            ),
            tuple(permutation.index(dim) for dim in self.order),
        )


def _spread(total, shape):
    result = [1] * len(shape)
    for i in reversed(range(len(shape))):
        take = min(total, shape[i])
        if take < 1 or total % take:
            return None
        result[i] = take
        total //= take
    return tuple(result) if total == 1 else None


def _reshape(layout, source, result):
    """Split dimensions innermost-first, like XeGPU LayoutAttr::expandDim.

    Only exact subgroup tilings and contiguous dim splits/unit-dim removal are
    supported. General collapses can require nonrectangular subgroup tiles.
    """
    if any(s != n * d for s, n, d in zip(source, layout.sg_layout, layout.sg_data)):
        return None
    groups = []
    position = 0
    for extent in source:
        start = position
        size = 1
        while position < len(result) and (size < extent or result[position] == 1):
            size *= result[position]
            position += 1
        if size != extent:
            return None
        groups.append(result[start:position])
    if position != len(result):
        return None

    fields = [[], [], []]
    for n, d, inst, shape in zip(
        layout.sg_layout, layout.sg_data, layout.inst_data, groups
    ):
        data = _spread(d, shape)
        if data is None:
            return None
        subgroups = _spread(n, [s // v for s, v in zip(shape, data)])
        if subgroups is None:
            return None
        instruction = _spread(inst, data)
        if instruction is None:
            return None
        for field, expanded in zip(fields, (subgroups, data, instruction)):
            field.extend(expanded)
    result_dims = []
    position = 0
    for group in groups:
        result_dims.append(tuple(range(position, position + len(group))))
        position += len(group)
    order = tuple(dim for i in layout.order for dim in reversed(result_dims[i]))
    return _Layout(*(tuple(field) for field in fields), order)


def _infer_store_layout(source, store, layout):
    known = {source.results[0]: layout}

    def infer(value):
        if value in known:
            return known[value]
        known[value] = None
        if isinstance(value.owner, ir.Block):
            return None
        op = value.owner.operation
        result_shape = list(ir.VectorType(value.type).shape)
        inferred = None
        if isinstance(op.opview, scf.ForOp):
            inferred = infer(
                op.regions[0].blocks[0].operations[-1].operands[value.result_number]
            )
        elif isinstance(op.opview, (vector.ShapeCastOp, vector.TransposeOp)):
            operand = op.operands[0]
            parent = infer(operand)
            if parent is not None:
                if isinstance(op.opview, vector.TransposeOp):
                    inferred = parent.transpose(list(op.opview.permutation))
                else:
                    inferred = _reshape(
                        parent, list(ir.VectorType(operand.type).shape), result_shape
                    )
        elif op.name.startswith(("arith.", "math.")) and not op.regions:
            candidates = [
                candidate
                for operand in op.operands
                if isinstance(operand.type, ir.VectorType)
                and list(operand.type.shape) == result_shape
                and (candidate := infer(operand)) is not None
            ]
            if candidates and all(
                candidate == candidates[0] for candidate in candidates
            ):
                inferred = candidates[0]
        known[value] = inferred
        return inferred

    stored = store.operands[0]
    if not isinstance(stored.owner, ir.Block):
        owner = stored.owner.operation
        if isinstance(owner.opview, arith.ConstantOp):
            constant = owner.attributes["value"]
            if isinstance(constant, ir.DenseElementsAttr) and constant.is_splat:
                return _reshape(
                    layout,
                    list(ir.VectorType(source.results[0].type).shape),
                    list(ir.VectorType(stored.type).shape),
                )
    return infer(stored)


class InferGemmStoreLayoutOp(
    TransformExtensionDialect.Operation, name="infer_gemm_store_layout"
):
    """Infer the XeGPU layout of an xegpu.store_nd from a DPAS result layout.

    An epilogue can transpose or reshape the DPAS result before storing it,
    so the store cannot simply reuse the accumulator's layout. Determining
    its layout requires tracing the stored value back through the epilogue
    graph and adapting tile ownership and dimension order along the path.
    This transform performs that payload traversal and returns the inferred
    layout for the schedule to apply to the store.

    Follows scf.for result yields, elementwise arithmetic/math ops, transposes and
    contiguous shape-cast dimension splits/unit-dim removal. Unknown branches
    do not constrain the result; conflicting known branches or an unsupported
    path fail explicitly. This selects the store's desired ownership, not the
    layouts of every intermediate: subsequent XeGPU propagation may introduce
    conversions for other epilogue branches.

    Input layout fields describe the DPAS result with default row-major order,
    not the store. Returns a layout attribute param for transform.annotate;
    unlike xegpu.set_anchor_layout this permits an inferred, rank-varying order.
    Dimension renumbering preserves both tile ownership and subgroup ID order.
    Uniform constant stores (e.g. accumulator initialization before a GEMM)
    have unconstrained ownership and use the chosen accumulator layout, with
    the same supported shape adjustments. Nonuniform unrelated stores fail.
    """

    source: ext.Operand[transform.AnyOpType]
    store: ext.Operand[transform.AnyOpType]
    sg_layout: Sequence[ext.Operand[transform.AnyParamType]]
    sg_data: Sequence[ext.Operand[transform.AnyParamType]]
    inst_data: ir.ArrayAttr
    layout: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, results, state) -> DiagnosedSilenceableFailure:
            sources = list(state.get_payload_ops(op.source))
            stores = list(state.get_payload_ops(op.store))

            def flatten(handles):
                return tuple(
                    ir.IntegerAttr(attr).value
                    for handle in handles
                    for attr in state.get_params(handle)
                )

            if (
                len(sources) != 1
                or len(stores) != 1
                or sources[0].name != "xegpu.dpas"
                or stores[0].name != "xegpu.store_nd"
            ):
                op.location.emit_error(
                    "infer_gemm_store_layout: expected one DPAS and one store_nd"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            shape = list(ir.VectorType(sources[0].results[0].type).shape)
            layout = _Layout(
                flatten(op.sg_layout),
                flatten(op.sg_data),
                tuple(ir.IntegerAttr(value).value for value in op.inst_data),
                tuple(reversed(range(len(shape)))),
            )
            if (
                any(
                    len(field) != len(shape)
                    for field in (layout.sg_layout, layout.sg_data, layout.inst_data)
                )
                or any(
                    v <= 0
                    for field in (layout.sg_layout, layout.sg_data, layout.inst_data)
                    for v in field
                )
                or any(
                    n * d != s or d % i
                    for s, n, d, i in zip(
                        shape, layout.sg_layout, layout.sg_data, layout.inst_data
                    )
                )
            ):
                op.location.emit_error(
                    "infer_gemm_store_layout: invalid DPAS subgroup layout"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            inferred = _infer_store_layout(sources[0], stores[0], layout)
            if inferred is None:
                stores[0].location.emit_error(
                    "infer_gemm_store_layout: cannot infer store layout from DPAS "
                    "through supported epilogue operations"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            assert prod(inferred.sg_layout) == prod(layout.sg_layout)
            fields = ", ".join(
                f"{name} = {list(getattr(inferred, name))}"
                for name in ("sg_layout", "sg_data", "inst_data", "order")
                if name != "order"
                or inferred.order != tuple(reversed(range(len(inferred.order))))
            )
            results.set_params(
                op.layout, [ir.Attribute.parse(f"#xegpu.layout<{fields}>")]
            )
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def infer_gemm_store_layout(source, store, *, sg_layout, sg_data, inst_data):
    op = InferGemmStoreLayoutOp(
        source=source,
        store=store,
        sg_layout=sg_layout,
        sg_data=sg_data,
        inst_data=ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
                for value in inst_data
            ]
        ),
    )
    return op.layout
