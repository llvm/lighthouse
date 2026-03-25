from mlir import ir
from mlir.dialects import memref, shard, tensor
from lighthouse.utils.mlir import func_cif


def split_axes_to_mlir(axes: list[list[int]]) -> ir.Attribute:
    """Convert a list of lists of axis indices to an ``#shard<axisarray …>``
    attribute."""
    inner = ", ".join("[" + ", ".join(str(x) for x in a) + "]" for a in axes)
    return ir.Attribute.parse(f"#shard<axisarray[{inner}]>")


def emit_shard_alloc(
    name: str, grid: str, ty: ir.RankedTensorType, split: list[list[int]]
):
    """Emit an ``alloc_<name>`` returning a newly allocated tensor of type *ty*,
    sharded according to *split*."""

    @func_cif(name=f"alloc_{name}")
    def _():
        e = tensor.empty(ty.shape, ty.element_type)
        sh = shard.sharding(grid, split_axes_to_mlir(split), [], [])
        s = shard.shard(e, sh)
        return shard.shard(s, sh, annotate_for_users=True)


def emit_shard_gather(
    name: str, grid: str, ty: ir.RankedTensorType, split: list[list[int]]
):
    """Emit a ``gather_<name>`` function that replicates from *split*."""

    @func_cif(ty, name=f"gather_{name}")
    def _(arg):
        sh_from = shard.sharding(grid, split_axes_to_mlir(split), [], [])
        sh_to = shard.sharding(grid, split_axes_to_mlir([[]]), [], [])
        s = shard.shard(arg, sh_from)
        return shard.shard(s, sh_to, annotate_for_users=True)


def emit_dealloc(elem_type: type, rank: int):
    """Emit a ``dealloc_<rank>d`` function which deallocates <rank>d memrefs."""
    dyn = ir.ShapedType.get_dynamic_size()
    mr_t = ir.MemRefType.get((dyn,) * rank, elem_type)

    @func_cif(mr_t, name=f"dealloc_{rank}d")
    def _(arg):
        memref.dealloc(arg)
