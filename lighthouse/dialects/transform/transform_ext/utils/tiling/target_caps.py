from mlir import ir
from mlir.dialects import linalg

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import linalg_inputs, linalg_outputs, opview

from .common import parallel_and_reduction_dims


def _contraction_operand_types(
    op: ir.Operation | ir.OpView,
) -> tuple[ir.Type, ir.Type, ir.Type] | None:
    """(lhs, rhs, acc) element types of a single-output contraction, else None."""
    ov = opview(op)
    if not linalg.isa_contraction_op(ov):
        return None
    inputs = linalg_inputs(ov)
    outputs = linalg_outputs(ov)
    if inputs is None or outputs is None or len(inputs) < 2 or len(outputs) != 1:
        return None
    return (
        ir.ShapedType(inputs[0].type).element_type,
        ir.ShapedType(inputs[1].type).element_type,
        ir.ShapedType(outputs[0].type).element_type,
    )


def is_amx_bf16_contraction(
    op: ir.Operation | ir.OpView, target: TargetInfo | None
) -> bool:
    """True for a bf16 -> f32 contraction on an AMX-capable target."""
    if target is None or not target.is_supported("amx"):
        return False
    types = _contraction_operand_types(op)
    if types is None:
        return False
    lhs, rhs, acc = types
    return (
        isinstance(lhs, ir.BF16Type)
        and isinstance(rhs, ir.BF16Type)
        and isinstance(acc, ir.F32Type)
    )


def is_f32_contraction(op: ir.Operation | ir.OpView) -> bool:
    """True for a contraction with all-f32 operands (lhs, rhs and acc)."""
    types = _contraction_operand_types(op)
    return types is not None and all(isinstance(t, ir.F32Type) for t in types)


def _vector_lane_count(target: TargetInfo | None, elem_type: ir.Type) -> int:
    """SIMD lane count for `elem_type` on `target` (defaults assume 512-bit)."""
    vector_bits = (
        target.vector_register_width_bits
        if target is not None and target.vector_register_width_bits is not None
        else 512
    )
    if isinstance(elem_type, (ir.FloatType, ir.IntegerType)):
        return max(1, vector_bits // max(1, elem_type.width))
    return 16


def generic_parallel_tiles(
    op: ir.Operation | ir.OpView,
    out_map: ir.AffineMap,
    target: TargetInfo | None,
) -> list[int]:
    """SIMD-lane default parallel tiles for ops without a microkernel profile."""
    parallel_dims, _ = parallel_and_reduction_dims(out_map)
    if not parallel_dims:
        return []
    out_elem = ir.ShapedType(linalg_outputs(op)[0].type).element_type
    inner = _vector_lane_count(target, out_elem)
    return [inner] if len(parallel_dims) == 1 else [1, inner]


def generic_reduction_tiles() -> list[int]:
    """Default reduction tile for ops without a microkernel profile."""
    return [1]
