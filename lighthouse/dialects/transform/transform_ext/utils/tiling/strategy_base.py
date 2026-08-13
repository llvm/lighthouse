from abc import ABC, abstractmethod
from dataclasses import dataclass

from mlir import ir

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import indexing_maps, linalg_outputs, opview


@dataclass
class StrategyContext:
    tile_size: int = 32  # Hint for tile size selection.
    target: TargetInfo | None = None


class TilingStrategy(ABC):
    """Base interface for tile-size selection strategies."""

    @abstractmethod
    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        """Return one tile size per loop dimension, or None if unsupported."""

    @staticmethod
    def output_map(op: ir.Operation | ir.OpView) -> ir.AffineMap | None:
        """Output indexing map of a single-output linalg op, else None."""
        ov = opview(op)
        maps = indexing_maps(ov)
        if maps is None or len(linalg_outputs(ov)) != 1:
            return None
        return maps[-1]
