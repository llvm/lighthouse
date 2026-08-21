from mlir import ir

from lighthouse.utils.mlir import dim_position, linalg_outputs


def parallel_and_reduction_dims(out_map: ir.AffineMap) -> tuple[list[int], list[int]]:
    """Split iteration dims into (parallel, reduction) via the output indexing map.

    Dims indexed by the output map are parallel; all others are reduction.
    """
    parallel_dims = [
        pos for pos in (dim_position(e) for e in out_map.results) if pos is not None
    ]
    reduction_dims = [d for d in range(out_map.n_dims) if d not in parallel_dims]
    return parallel_dims, reduction_dims


def output_tensor_dim_of_iter_dim(out_map: ir.AffineMap) -> dict[int, int]:
    """Map each iteration dim to the output tensor dim it indexes (if any)."""
    mapping: dict[int, int] = {}
    for tensor_dim, expr in enumerate(out_map.results):
        pos = dim_position(expr)
        if pos is not None:
            mapping[pos] = tensor_dim
    return mapping


def disable_small_tiles(
    op: ir.OpView,
    out_map: ir.AffineMap,
    sizes: list[int],
    tile_size: int,
) -> None:
    """Disable tiling for parallel dims whose static extent is below tile_size."""
    out_type = ir.ShapedType(linalg_outputs(op)[0].type)
    iter_to_tensor = output_tensor_dim_of_iter_dim(out_map)
    for iter_dim, tensor_dim in iter_to_tensor.items():
        if sizes[iter_dim] <= 1:
            continue
        dim = out_type.shape[tensor_dim]
        if ir.ShapedType.is_static_size(dim) and dim < tile_size:
            sizes[iter_dim] = 0
