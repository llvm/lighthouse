from mlir import ir
from mlir.dialects import linalg

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import linalg_inputs, linalg_outputs, opview

from .common import largest_divisor, parallel_and_reduction_dims


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
    ov = opview(op)
    inputs = linalg_inputs(ov)
    outputs = linalg_outputs(ov)
    # AMX requires matrix-shaped (rank >= 2) operands; reject vector/scalar contractions.
    if any(ir.ShapedType(v.type).rank < 2 for v in (inputs[0], inputs[1], outputs[0])):
        return False
    return (
        isinstance(lhs, ir.BF16Type)
        and isinstance(rhs, ir.BF16Type)
        and isinstance(acc, ir.F32Type)
    )


def is_f32_contraction(op: ir.Operation | ir.OpView) -> bool:
    """True for a contraction with all-f32 operands (lhs, rhs and acc)."""
    types = _contraction_operand_types(op)
    return types is not None and all(isinstance(t, ir.F32Type) for t in types)


def vector_lane_count(target: TargetInfo | None, elem_type: ir.Type) -> int:
    """SIMD lane count for a single vector register; does not include the full register bank."""
    vector_bits = (
        target.vector_register_width_bits
        if target is not None and target.vector_register_width_bits is not None
        else 512
    )
    if isinstance(elem_type, (ir.FloatType, ir.IntegerType)):
        return max(1, vector_bits // max(1, elem_type.width))
    return 16


def _vector_tiles(dim_sizes: list[int], vector_width: int) -> list[int]:
    """Tiles of the trailing dims spanning one vector register.

    Inner dims narrower than the vector are taken whole and the next dims
    tiled by a divisor of the lanes left, so a narrow innermost dim does not
    shrink the tiles to a few elements.
    """
    if dim_sizes:
        innermost_size = dim_sizes[-1]
        if (
            ir.ShapedType.is_dynamic_size(innermost_size)
            or innermost_size >= vector_width
        ):
            return [vector_width]

    tiles: list[int] = []
    remaining = vector_width
    for dim_size in reversed(dim_sizes):
        dynamic = ir.ShapedType.is_dynamic_size(dim_size)
        if dynamic:
            tiles.append(1)
            break
        if dim_size >= remaining:
            tiles.append(largest_divisor(dim_size, remaining))
            break
        tiles.append(max(1, dim_size))
        remaining //= tiles[-1]
        if remaining <= 1:
            break
    return tiles[::-1]


def generic_parallel_tiles(
    op: ir.Operation | ir.OpView,
    out_map: ir.AffineMap,
    target: TargetInfo | None,
) -> list[int]:
    """SIMD-lane default parallel tiles for ops without a microkernel profile."""
    parallel_dims, _ = parallel_and_reduction_dims(out_map)
    if not parallel_dims:
        return []
    out_type = ir.ShapedType(linalg_outputs(op)[0].type)
    vector_width = vector_lane_count(target, out_type.element_type)
    return _vector_tiles(out_type.shape, vector_width)


def generic_reduction_tiles() -> list[int]:
    """Default reduction tile for ops without a microkernel profile."""
    return [1]
