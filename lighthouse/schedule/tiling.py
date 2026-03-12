from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import tile_ops
from lighthouse.transform import cleanup


def tile(
    target_op: str,
    tile_sizes: list[int],
    fuse_producers: bool = False,
    tile_interchange: list[int] | None = None,
    peel_loops: list[int] = [],
    unroll_factors: list[int] = [],
) -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        tile_ops(
            named_seq.bodyTarget,
            target_op,
            tile_sizes=tile_sizes,
            fuse_producers=fuse_producers,
            tile_interchange=tile_interchange,
            peel_loops=peel_loops,
            unroll_factors=unroll_factors,
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
