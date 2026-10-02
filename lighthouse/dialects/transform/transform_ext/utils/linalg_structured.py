"""Structured-op (``LinalgOp`` interface) accessors missing from the bindings.

The bindings expose only the raw ``indexing_maps`` / ``iterator_types``
attributes, so the per-operand queries the reduction fusion relies on -- indexing
map, body block argument, loop ranges -- are rebuilt here.

Operand order for a structured linalg op is inputs first, then outputs (DPS
inits), with one output per result. That single fact is what lets every accessor
below be derived from an operand index.

`ir.OpOperand` carries only ``owner``/``operand_number`` and cannot read or write
the operand's value, so `Operand` below stands in for it.
"""

from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import indexing_maps, opview

__all__ = [
    "Operand",
    "dps_init_operands",
    "dps_input_operands",
    "indexing_map_for",
    "iterator_types",
    "matching_block_argument",
    "num_dps_inits",
    "num_loops",
    "num_reduction_loops",
    "operands_of",
    "reduction_dims",
    "region_output_args",
    "static_loop_ranges",
]


class Operand:
    """A readable/writable reference to one operand of an op.

    Holds the owning op and the operand index, and reads or writes the operand
    value through them. Two references are equal when they name the same operand
    of the same op, so they work as dict keys.
    """

    __slots__ = ("owner", "index")

    def __init__(self, owner: ir.Operation | ir.OpView, index: int):
        self.owner = opview(owner)
        self.index = index

    @property
    def value(self) -> ir.Value:
        """The value currently bound to this operand."""
        return self.owner.operands[self.index]

    def set(self, value: ir.Value) -> None:
        """Rebind this operand to `value`."""
        self.owner.operands[self.index] = value

    def _key(self):
        return (self.owner.operation.__hash__(), self.index)

    def __eq__(self, other) -> bool:
        return isinstance(other, Operand) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"Operand(#{self.index} of {self.owner.operation.name})"


def operands_of(op: ir.Operation | ir.OpView) -> list[Operand]:
    """All operands of `op` as `Operand` references, in operand order."""
    ov = opview(op)
    return [Operand(ov, i) for i in range(len(ov.operands))]


def iterator_types(op: ir.Operation | ir.OpView) -> list[str]:
    """Iterator types of a structured linalg op as ``"parallel"``/``"reduction"``.

    The attribute holds ``#linalg.iterator_type<...>`` attrs, which are compared
    against the built enum attr rather than parsed.
    """
    ov = opview(op)
    build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
    parallel = build(linalg.IteratorType.parallel, context=ov.context)
    return ["parallel" if it == parallel else "reduction" for it in ov.iterator_types]


def num_loops(op: ir.Operation | ir.OpView) -> int:
    """Number of iteration dims (loops) of a structured linalg op."""
    return len(iterator_types(op))


def reduction_dims(op: ir.Operation | ir.OpView) -> list[int]:
    """Positions of the reduction iterators, in order."""
    return [i for i, it in enumerate(iterator_types(op)) if it == "reduction"]


def num_reduction_loops(op: ir.Operation | ir.OpView) -> int:
    """Number of reduction iterators."""
    return len(reduction_dims(op))


def num_dps_inits(op: ir.Operation | ir.OpView) -> int:
    """Number of DPS init (``outs``) operands, i.e. one per result."""
    return len(list(opview(op).results))


def dps_input_operands(op: ir.Operation | ir.OpView) -> list[Operand]:
    """The ``ins`` operands as `Operand` references."""
    ov = opview(op)
    return operands_of(ov)[: len(ov.operands) - num_dps_inits(ov)]


def dps_init_operands(op: ir.Operation | ir.OpView) -> list[Operand]:
    """The ``outs`` operands as `Operand` references."""
    ov = opview(op)
    return operands_of(ov)[len(ov.operands) - num_dps_inits(ov) :]


def indexing_map_for(op: ir.Operation | ir.OpView, operand: Operand) -> ir.AffineMap:
    """The indexing map matching `operand`, i.e. ``getMatchingIndexingMap``."""
    return indexing_maps(op)[operand.index]


def matching_block_argument(
    op: ir.Operation | ir.OpView, operand: Operand
) -> ir.BlockArgument:
    """The body block argument matching `operand`.

    Body arguments come one per operand in operand order, so the operand index
    is the argument index.
    """
    return opview(op).regions[0].blocks[0].arguments[operand.index]


def region_output_args(op: ir.Operation | ir.OpView) -> list[ir.BlockArgument]:
    """The body block arguments matching the DPS init operands."""
    ov = opview(op)
    args = list(ov.regions[0].blocks[0].arguments)
    return args[len(args) - num_dps_inits(ov) :]


def static_loop_ranges(op: ir.Operation | ir.OpView) -> list[int]:
    """Static extent of every loop dim, ``ShapedType::kDynamic`` where unknown.

    Recovered by matching each operand's indexing map against its shaped type: a
    plain dim expr at map result `i` pins that loop dim to the operand's extent
    along tensor dim `i`. Composite exprs carry no single extent and are skipped.
    """
    ov = opview(op)
    ranges = [ir.ShapedType.get_dynamic_size()] * num_loops(ov)
    for operand, imap in zip(operands_of(ov), indexing_maps(ov)):
        try:
            shape = ir.ShapedType(operand.value.type).shape
        except (ValueError, TypeError):
            # A scalar operand pins no loop extent.
            continue
        for pos, expr in enumerate(imap.results):
            if isinstance(expr, ir.AffineDimExpr):
                ranges[expr.position] = shape[pos]
    return ranges
