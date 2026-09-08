from array import array

from mlir import ir
from mlir.dialects import arith, ext, linalg, scf, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.sfc import gilbert2d


def _static_bounds(loop: scf.ForallOp) -> tuple[list[int], list[int], list[int]] | None:
    """Return static forall bounds, or None when any bound is dynamic."""
    lower = list(loop.staticLowerBound)
    upper = list(loop.staticUpperBound)
    steps = list(loop.staticStep)
    if any(
        value == ir.ShapedType.get_dynamic_size() for value in lower + upper + steps
    ):
        return None
    return lower, upper, steps


def _uses_value(value: ir.Value, target: ir.BlockArgument, visited: set[int]) -> bool:
    """Return whether value depends on the given block argument."""
    if value == target:
        return True
    if isinstance(value, ir.BlockArgument):
        return False
    owner = value.owner
    if id(owner) in visited:
        return False
    visited.add(id(owner))
    return any(_uses_value(operand, target, visited) for operand in owner.operands)


def _used_loop_ivs(
    loop: scf.ForallOp, offsets: list[ir.Value]
) -> list[ir.BlockArgument]:
    """Return loop IVs used by at least one of the given offset values."""
    return [
        iv
        for iv in loop.induction_variables
        if any(_uses_value(offset, iv, set()) for offset in offsets)
    ]


def _relevant_forall(
    target: ir.Operation,
) -> tuple[scf.ForallOp, list[ir.BlockArgument]] | None:
    """Find the enclosing two-dimensional forall defining both tile IVs."""
    offsets = []
    for operand in target.operands:
        for extract in _find_slices(operand, set()):
            offsets.extend(extract.offsets)

    owner = target.parent
    while owner is not None:
        if isinstance(owner.opview, scf.ForallOp):
            loop = owner.opview
            if len(loop.induction_variables) == 2:
                used_ivs = _used_loop_ivs(loop, offsets)
                if len(used_ivs) == 2:
                    return loop, used_ivs
        owner = owner.parent
    return None


def _find_slices(value: ir.Value, visited: set[int]) -> list[tensor.ExtractSliceOp]:
    """Collect extract-slice operations reachable from an SSA value."""
    if isinstance(value, ir.BlockArgument):
        return []
    owner = value.owner
    if id(owner) in visited:
        return []
    visited.add(id(owner))
    slices = []
    if isinstance(owner.opview, tensor.ExtractSliceOp):
        slices.append(owner.opview)
    for operand in owner.operands:
        slices.extend(_find_slices(operand, visited))
    return slices


def _replace_operands(
    operation: ir.Operation,
    value_map: dict[ir.Value, ir.Value],
    operation_map: dict[ir.Operation, ir.Operation],
) -> ir.WalkResult:
    """Remap operands in a cloned operation and its nested operations."""
    for index, operand in enumerate(operation.operands):
        replacement = value_map.get(operand)
        if replacement is None and not isinstance(operand, ir.BlockArgument):
            producer = operation_map.get(operand.owner)
            if producer is not None:
                replacement = producer.results[operand.result_number]
        if replacement is not None:
            operation.operands[index] = replacement
    return ir.WalkResult.ADVANCE


def _clone_body(old_loop: scf.ForallOp, new_loop: scf.ForallOp, ivs: list[ir.Value]):
    """Clone a forall body while remapping IVs, outputs, and local results."""
    old_block = old_loop.region.blocks[0]
    new_block = new_loop.region.blocks[0]

    old_ivs = list(old_loop.induction_variables)
    new_ivs = list(new_loop.induction_variables)
    value_map: dict[ir.Value, ir.Value] = {
        old_iv: new_iv for old_iv, new_iv in zip(old_ivs, ivs)
    }
    value_map.update(
        {
            old_out: new_out
            for old_out, new_out in zip(
                old_block.arguments[len(old_ivs) :],
                new_block.arguments[len(new_ivs) :],
            )
        }
    )

    operation_map = {}
    with ir.InsertionPoint(new_block):
        for old_operation in old_block.operations:
            new_operation = old_operation.clone()
            operation_map[old_operation] = new_operation
            new_operation.walk(
                lambda nested: _replace_operands(nested, value_map, operation_map)
            )
            value_map.update(dict(zip(old_operation.results, new_operation.results)))


def _constant_table(values: list[int], location: ir.Location) -> ir.Value:
    """Create an i64 tensor constant containing SFC coordinates."""
    element_type = ir.IntegerType.get_signless(64)
    tensor_type = ir.RankedTensorType.get([len(values)], element_type)
    dense = ir.DenseElementsAttr.get(array("q", values), type=element_type)
    return arith.ConstantOp(tensor_type, dense, loc=location).result


def _rewrite(
    old_loop: scf.ForallOp,
    used_ivs: list[ir.BlockArgument],
    rewriter: transform.TransformRewriter,
) -> scf.ForallOp | None:
    """Replace a two-dimensional forall with its one-dimensional SFC traversal."""
    bounds = _static_bounds(old_loop)
    if bounds is None:
        return None
    lower, upper, steps = bounds
    if len(lower) != 2 or lower != [0, 0] or steps != [1, 1]:
        return None

    old_ivs = list(old_loop.induction_variables)
    if len(old_ivs) != 2 or used_ivs != old_ivs:
        return None

    axis0, axis1 = used_ivs
    height = upper[old_ivs.index(axis0)]
    width = upper[old_ivs.index(axis1)]
    points = list(gilbert2d(width, height))
    if len(points) != height * width:
        return None
    # The SFC is generated in (x, y) coordinates. Keep the row and column
    # tables explicit while rewriting the flattened loop to avoid binding the
    # indices to the wrong axes when the original 2D forall is collapsed into 1D.
    m_indices = [row for _, row in points]
    n_indices = [col for col, _ in points]
    assert all(0 <= row < height for row in m_indices)
    assert all(0 <= col < width for col in n_indices)

    with ir.InsertionPoint(old_loop.operation), old_loop.location:
        m_table = _constant_table(m_indices, old_loop.location)
        n_table = _constant_table(n_indices, old_loop.location)
        new_loop = scf.ForallOp(
            [0],
            [height * width],
            [1],
            shared_outs=list(old_loop.outputs),
            loc=old_loop.location,
        )
        loop_iv = list(new_loop.induction_variables)[0]
        with ir.InsertionPoint(new_loop.region.blocks[0]):
            m_value = tensor.ExtractOp(m_table, [loop_iv], loc=old_loop.location).result
            n_value = tensor.ExtractOp(n_table, [loop_iv], loc=old_loop.location).result
            m_index = arith.IndexCastOp(
                ir.IndexType.get(), m_value, loc=old_loop.location
            ).result
            n_index = arith.IndexCastOp(
                ir.IndexType.get(), n_value, loc=old_loop.location
            ).result
        _clone_body(old_loop, new_loop, [m_index, n_index])

    rewriter.replace_op(old_loop.operation, new_loop.operation)
    return new_loop


class SfcRemapForallOp(TransformExtensionDialect.Operation, name="sfc_remap_forall"):
    """Remap a tiled contraction's two-dimensional forall with a 2D SFC."""

    target: ext.Operand[transform.AnyOpType]
    remapped_loop: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "SfcRemapForallOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            loops = []
            targets = list(state.get_payload_ops(op.target))

            # Gather all unique loop ops as rewriting might invalidate
            # target handles and to avoid duplicate work.
            loops_by_relevant_ivs = []
            seen_loops: set[ir.Operation] = set()
            for target in targets:
                if not linalg.isa_contraction_op(target):
                    continue
                relevant = _relevant_forall(target.operation)
                if relevant is not None:
                    loop, used_ivs = relevant
                    if loop.operation in seen_loops:
                        continue
                    seen_loops.add(loop.operation)
                    loops_by_relevant_ivs.append((loop, used_ivs))

            # Try to remap the gathered loops.
            for loop, used_ivs in loops_by_relevant_ivs:
                loop = _rewrite(loop, used_ivs, _rewriter)
                if loop is not None:
                    loops.append(loop)
            results.set_ops(op.remapped_loop, loops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "SfcRemapForallOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "SfcRemapForallOp"):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def sfc_remap_forall(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Create an SFC forall-remapping transform operation."""
    return SfcRemapForallOp(target=target).remapped_loop
