"""Analysis helpers for inferring matmul shape and XeGPU parameters."""

from mlir import ir
from mlir.dialects import memref, vector

from lighthouse.utils.mlir import defining_op, dim_position


def _source_shape(value: ir.Value) -> list[int] | None:
    """Global shape feeding `value`, found by walking to its transfer_read."""
    op = defining_op(value)
    seen: set[ir.Operation] = set()
    while op is not None and op not in seen:
        seen.add(op)
        if isinstance(op.opview, vector.TransferReadOp):
            return list(ir.ShapedType(op.operands[0].type).shape)
        if len(op.operands) == 0:
            break
        op = defining_op(op.operands[0])
    return None


def _matmul_shape_and_transpose(
    contract: ir.OpView,
) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) plus transpose_a/transpose_b from a vector.contract.

    The iteration space is (M, N, K); M and N are the two output dims and K is
    the remaining (reduction) dim. A transpose shows up as a reversed indexing
    map: the A/B operand lists the reduction dim before/after the output dim
    that a plain matmul would place the other way around.
    """
    maps_attr = contract.attributes["indexing_maps"]
    a_map = ir.AffineMapAttr(maps_attr[0]).value
    b_map = ir.AffineMapAttr(maps_attr[1]).value
    c_map = ir.AffineMapAttr(maps_attr[2]).value

    a_dims = [dim_position(r) for r in a_map.results]
    b_dims = [dim_position(r) for r in b_map.results]
    out_dims = [dim_position(r) for r in c_map.results]
    m_dim, n_dim = out_dims[0], out_dims[1]
    k_dim = next(d for d in range(a_map.n_dims) if d not in out_dims)

    lhs, rhs = contract.operands[0], contract.operands[1]
    a_shape = _source_shape(lhs)
    b_shape = _source_shape(rhs)
    if a_shape is None or b_shape is None:
        raise ValueError("Could not infer global shape of contract operands")

    dim_size: dict[int, int] = {}
    for dims, shape in ((a_dims, a_shape), (b_dims, b_shape)):
        for i, d in enumerate(dims):
            if d is not None:
                dim_size[d] = shape[i]

    shape = (dim_size[m_dim], dim_size[n_dim], dim_size[k_dim])
    transpose_a = a_dims == [k_dim, m_dim]
    transpose_b = b_dims == [n_dim, k_dim]
    return shape, transpose_a, transpose_b


def _root_memref_shape(value: ir.Value) -> list[int]:
    """Shape of the root memref backing `value`.

    Walks through rank-preserving view ops (memref.subview / cast) so a tile
    descriptor built on a workgroup subview still reports the global shape.
    """
    op = defining_op(value)
    seen: set[ir.Operation] = set()
    while op is not None and op not in seen:
        seen.add(op)
        if not isinstance(op.opview, (memref.SubViewOp, memref.CastOp)):
            break
        if len(op.operands) == 0:
            break
        value = op.operands[0]
        op = defining_op(value)
    return list(ir.ShapedType(value.type).shape)


def _xegpu_operand_source(value: ir.Value) -> tuple[list[int] | None, bool]:
    """Global source tile shape feeding an xegpu.dpas operand.

    Walks from the operand to the xegpu.create_nd_tdesc it is loaded from and
    reports whether a vector.transpose sits in between (i.e. the tile is stored
    transposed relative to how the dpas consumes it). The returned shape is the
    global matmul operand shape, not the small per-dpas tile.
    """
    transposed = False
    op = defining_op(value)
    seen: set[ir.Operation] = set()
    while op is not None and op not in seen:
        seen.add(op)
        if isinstance(op.opview, vector.TransposeOp):
            transposed = True
        # xegpu dialect has no Python bindings, so match by name.
        if op.name == "xegpu.create_nd_tdesc":
            return _root_memref_shape(op.operands[0]), transposed
        if len(op.operands) == 0:
            break
        op = defining_op(op.operands[0])
    return None, transposed


def _dpas_shape_and_transpose(
    dpas: ir.OpView,
) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) plus transpose_a/transpose_b from an xegpu.dpas op.

    The dpas consumes A as [M, K] and B as [K, N]. Each operand is traced back
    to the create_nd_tdesc it loads from; a vector.transpose in between means
    the source tile is stored transposed.
    """
    a_shape, transpose_a = _xegpu_operand_source(dpas.operands[0])
    b_shape, transpose_b = _xegpu_operand_source(dpas.operands[1])
    if a_shape is None or b_shape is None:
        raise ValueError("Could not infer global shape of dpas operands")

    # A source is [K, M] when transposed, else [M, K].
    m, k = (a_shape[1], a_shape[0]) if transpose_a else (a_shape[0], a_shape[1])
    # B source is [N, K] when transposed, else [K, N].
    n = b_shape[0] if transpose_b else b_shape[1]
    return (m, n, k), transpose_a, transpose_b


def _first_producer_named(value: ir.Value, op_name: str) -> ir.Operation | None:
    """First producer op with `op_name` in `value`'s producer chain, or None."""
    if value is None or isinstance(value, ir.BlockArgument):
        return None
    op = defining_op(value)
    if op is None:
        return None
    if op.name == op_name:
        return op
    for operand in op.operands:
        found = _first_producer_named(operand, op_name)
        if found is not None:
            return found
    return None


def _linalg_matmul_operand_shape(value: ir.Value) -> tuple[list[int], bool]:
    """Global operand shape in matmul orientation plus its transpose flag.

    Returns A as [M, K] and B as [K, N]: an extract_slice recovers the
    pre-tiling shape and a linalg.transpose producer means the source is
    swapped back.
    """
    transposed = _first_producer_named(value, "linalg.transpose") is not None
    slice_op = _first_producer_named(value, "tensor.extract_slice")
    source = slice_op.operands[0] if slice_op is not None else value
    shape = list(ir.ShapedType(source.type).shape)
    if transposed:
        shape.reverse()
    return shape, transposed


def _linalg_matmul_shape_and_transpose(
    matmul: ir.OpView,
) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) plus transpose_a/transpose_b from a linalg.matmul op.

    A transpose shows up as a linalg.transpose in the producer chain of the
    corresponding input operand.
    """
    a_shape, transpose_a = _linalg_matmul_operand_shape(matmul.inputs[0])
    b_shape, transpose_b = _linalg_matmul_operand_shape(matmul.inputs[1])
    m, k = a_shape
    _, n = b_shape
    return (m, n, k), transpose_a, transpose_b


# Canonical linalg.matmul operand maps over loops (m=d0, n=d1, k=d2): A reads
# [m, k], B reads [k, n]. Any other dims mean a transposed or broadcast operand.
_MATMUL_A_DIMS = [0, 2]
_MATMUL_B_DIMS = [2, 1]


def _reject_broadcast_matmul(matmul: ir.OpView) -> None:
    """Raise if `matmul` operand is produced by a linalg.broadcast op."""
    for operand in matmul.inputs:
        if _first_producer_named(operand, "linalg.broadcast") is not None:
            raise ValueError(
                "linalg.matmul has a linalg.broadcast producer, which the tile-size "
                "analysis does not support"
            )


def _reject_transposed_matmul(matmul: ir.OpView) -> None:
    """Raise if `matmul` has non-identity (transposed) input indexing maps."""
    maps = [ir.AffineMapAttr(m).value for m in matmul.attributes["indexing_maps"]]
    a_dims = [dim_position(r) for r in maps[0].results]
    b_dims = [dim_position(r) for r in maps[1].results]
    if a_dims != _MATMUL_A_DIMS or b_dims != _MATMUL_B_DIMS:
        raise ValueError(
            "linalg.matmul has non-identity (transposed) input indexing maps, which "
            "the tile-size analysis does not support"
        )


def analyze_matmul_op(op: ir.OpView) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) and transpose_a/transpose_b from a matmul-like anchor op.

    Supports linalg.matmul, vector.contract and xegpu.dpas anchor ops; other op
    kinds will be added later.
    """
    # TODO use op name as xegpu dialect python bindings are missing
    op_name = op.operation.name
    if op_name == "linalg.matmul":
        _reject_broadcast_matmul(op)
        return _linalg_matmul_shape_and_transpose(op)
    elif op_name == "vector.contract":
        return _matmul_shape_and_transpose(op)
    elif op_name == "xegpu.dpas":
        return _dpas_shape_and_transpose(op)
    raise NotImplementedError(f"unsupported anchor op '{op_name}'")


def analyze_wg_k_tile_size(op: ir.OpView) -> tuple[int, ...] | None:
    """Infer the workgroup and reduction tile size applied to a matmul anchor op.

    Returns a tuple (wg_tile, k_tile) where k_tile can be None if not determined.
    """
    op_name = op.operation.name
    wg_tile = None
    k_tile = None
    if op_name == "linalg.matmul":
        if op.parent.name != "scf.forall" and not (
            op.parent.name == "scf.for" and op.parent.parent.name == "scf.forall"
        ):
            # target is not within a scf.forall loop, so it's not workgroup tiled
            return None, None
        _reject_broadcast_matmul(op)
        _reject_transposed_matmul(op)
        # Assume we are in WG loop or WG-k loop nest
        m, k = list(ir.ShapedType(op.inputs[0].type).shape)
        _, n = list(ir.ShapedType(op.inputs[1].type).shape)
        wg_tile = (m, n)
        if op.parent.name == "scf.for":
            k_tile = k
    elif op_name == "vector.contract":
        # Assume we are in WG loop or WG-k loop nest. The accumulator carries
        # the (M, N) tile; the reduction dim gives the K tile.
        maps_attr = op.attributes["indexing_maps"]
        a_map = ir.AffineMapAttr(maps_attr[0]).value
        c_map = ir.AffineMapAttr(maps_attr[2]).value
        a_dims = [dim_position(r) for r in a_map.results]
        out_dims = [dim_position(r) for r in c_map.results]
        k_dim = next(d for d in range(a_map.n_dims) if d not in out_dims)

        acc_shape = list(ir.ShapedType(op.operands[2].type).shape)
        wg_tile = (acc_shape[0], acc_shape[1])
        if op.parent.name == "scf.for":
            # index() handles a transposed operand where K precedes M.
            a_shape = list(ir.ShapedType(op.operands[0].type).shape)
            k_tile = a_shape[a_dims.index(k_dim)]
    elif op_name == "xegpu.dpas":
        # dpas consumes A as [M, K] and acc as [M, N] regardless of how the
        # source tiles are stored, so shapes read off directly.
        acc_shape = list(ir.ShapedType(op.operands[2].type).shape)
        wg_tile = (acc_shape[0], acc_shape[1])
        if op.parent.name == "scf.for":
            k_tile = list(ir.ShapedType(op.operands[0].type).shape)[1]
    else:
        raise ValueError(f"unsupported anchor op '{op_name}'")
    return wg_tile, k_tile
