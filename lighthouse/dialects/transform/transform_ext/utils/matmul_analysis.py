"""Analysis helpers for inferring matmul shape and XeGPU parameters."""

from mlir import ir

from lighthouse.utils.mlir import defining_op, dim_position


def _source_shape(value: ir.Value) -> list[int] | None:
    """Global shape feeding `value`, found by walking to its transfer_read."""
    op = defining_op(value)
    seen: set[ir.Operation] = set()
    while op is not None and op not in seen:
        seen.add(op)
        if op.name.endswith("transfer_read"):
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
        if not (op.name.endswith("subview") or op.name.endswith("cast")):
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
        if op.name.endswith("transpose"):
            transposed = True
        if op.name.endswith("create_nd_tdesc"):
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


def _linalg_matmul_operand_shape(value: ir.Value) -> list[int]:
    """Global shape of a linalg.matmul operand.

    When the operand is a tile carved out by a tensor.extract_slice (i.e. the
    matmul has already been workgroup-tiled) the slice's source shape is the
    original global shape; otherwise the operand's own shape is global.
    """
    slice_op = _first_producer_named(value, "tensor.extract_slice")
    source = slice_op.operands[0] if slice_op is not None else value
    return list(ir.ShapedType(source.type).shape)


def _linalg_matmul_shape_and_transpose(
    matmul: ir.OpView,
) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) plus transpose_a/transpose_b from a linalg.matmul op.

    A transpose shows up as a linalg.transpose in the producer chain of the
    corresponding input operand.
    """
    inputs = matmul.inputs
    m, k = _linalg_matmul_operand_shape(inputs[0])
    _, n = _linalg_matmul_operand_shape(inputs[1])
    transpose_a = _first_producer_named(inputs[0], "linalg.transpose") is not None
    transpose_b = _first_producer_named(inputs[1], "linalg.transpose") is not None
    return (m, n, k), transpose_a, transpose_b


def analyze_matmul_op(op: ir.OpView) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer (M, N, K) and transpose_a/transpose_b from a matmul-like anchor op.

    Supports linalg.matmul, vector.contract and xegpu.dpas anchor ops; other op
    kinds will be added later.
    """
    op_name = op.operation.name
    if op_name == "linalg.matmul":
        return _linalg_matmul_shape_and_transpose(op)
    elif op_name == "vector.contract":
        return _matmul_shape_and_transpose(op)
    elif op_name == "xegpu.dpas":
        return _dpas_shape_and_transpose(op)
    raise NotImplementedError(f"unsupported anchor op '{op_name}'")
