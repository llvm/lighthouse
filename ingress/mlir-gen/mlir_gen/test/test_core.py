import pytest
import torch
from typing import Tuple

from mlir import ir
from mlir.dialects import transform, func, linalg, tensor, arith, complex, math
from mlir.dialects.transform import structured
from mlir.dialects.transform import interpreter
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import (
    get_ranked_memref_descriptor,
)
from mlir.execution_engine import ExecutionEngine


from lighthouse import utils as lh_utils


def affine_map(dim_count, exprs, *, symb_count=0):
    return ir.AffineMap.get(dim_count, symb_count, exprs)


parallel = linalg.IteratorType.parallel
reduction = linalg.IteratorType.reduction


def create_pass_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("convert-scf-to-cf")
        pm.add("expand-strided-metadata")
        pm.add("lower-affine")
        pm.add("finalize-memref-to-llvm")
        pm.add("convert-func-to-llvm")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.add("cse")
        pm.add("canonicalize")
    return pm


def create_schedule(ctx: ir.Context) -> ir.Module:
    """
    Create an MLIR module containing transformation schedule.
    The schedule provides partial lowering to scalar operations.

    Args:
        ctx: MLIR context.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )

        # Create entry point transformation sequence.
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            # For simplicity, use generic transform matchers.
            anytype = transform.AnyOpType.get()

            # Find the kernel's function op.
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )

            # Use C interface wrappers - required to make function executable after jitting.
            func = transform.apply_registered_pass(
                anytype, func, "llvm-request-c-wrappers"
            )

            # Find the kernel's module op.
            mod = transform.get_parent_op(
                anytype, func, op_name="builtin.module", deduplicate=True
            )

            # Naive lowering to loops.
            mod = transform.apply_registered_pass(
                anytype, mod, "convert-linalg-to-loops"
            )
            # Cleanup.
            transform.ApplyCommonSubexpressionEliminationOp(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.ApplyCanonicalizationPatternsOp()

            # Terminate the schedule.
            transform.YieldOp()
    return schedule


def apply_schedule(kernel: ir.Module, schedule: ir.Module) -> None:
    interpreter.apply_named_sequence(
        payload_root=kernel,
        transform_root=schedule.body.operations[0],
        transform_module=schedule,
    )


def bufferize_module(ctx: ir.Context, kernel: ir.Module) -> None:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("one-shot-bufferize{bufferize-function-boundaries}")
        pm.run(kernel.operation)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


#### IR builders #####
# TODO: Move to mlir_gen module


def get_add(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.add(a, b, outs=(out,))


def get_rsqrt(a: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.rsqrt(a, outs=(out,))


def get_powf(a: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.powf(a, outs=(out,))


def get_sqr(a: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.square(a, outs=(out,))


def get_mul(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.mul(a, b, outs=(out,))


# equvialent to torch.mean(-1, keepdim=True)
def get_mean(a: ir.Value, out: ir.Value) -> ir.Value:
    # Need to initialize the output with zeros for accumulation
    zero = arith.ConstantOp(ir.F32Type.get(), 0.0)
    out_filled = linalg.fill(zero, outs=[out])

    # Input map: (d0, d1) -> (d0, d1)
    input_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank)],
    )
    # Output map: (d0, d1) -> (d0, 0)
    output_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank - 1)]
        + [ir.AffineConstantExpr.get(0)],
    )
    iterator_types = [parallel] * (a.type.rank - 1) + [reduction]

    scale = arith.ConstantOp(ir.F32Type.get(), 1.0 / a.type.shape[-1])

    @linalg.generic(
        [a],
        [out_filled],
        [input_map, output_map],
        iterator_types,
    )
    def mean_op(a_val, acc):
        # Multiply input by scale factor and add to accumulator
        scaled = arith.mulf(a_val, scale)
        return arith.addf(scaled, acc)

    return mean_op


# repeat_kv
def get_repeat_kv(x: ir.Value, n_rep: int, out: ir.Value) -> ir.Value:
    bs, slen, n_kv_heads, head_dim = x.type.shape
    if n_rep == 1:
        return x

    b, s, h_out, d = [ir.AffineDimExpr.get(i) for i in range(4)]

    # For output head h_out, we read from input head h_out // n_rep
    # This is equivalent to: x[:, :, :, None, :].expand(...).reshape(...)
    h_in = ir.AffineExpr.get_floor_div(h_out, ir.AffineConstantExpr.get(n_rep))

    # Affine maps
    x_map = affine_map(4, [b, s, h_in, d])
    out_map = affine_map(4, [b, s, h_out, d])

    @linalg.generic(
        [x],
        [out],
        [x_map, out_map],
        [parallel] * 4,
    )
    def repeat_kv_op(a, _out):
        return a

    return repeat_kv_op


# equivalent to torch.nn.functional.silu
def get_silu(inputs: ir.Value, out: ir.Value) -> ir.Value:
    elty = inputs.type.element_type
    one = arith.constant(elty, 1.0)

    dims = [ir.AffineDimExpr.get(i) for i in range(inputs.type.rank)]
    par_affine_map = affine_map(inputs.type.rank, dims)
    par_iterator_types = [parallel] * inputs.type.rank

    @linalg.generic(
        [inputs],
        [out],
        [par_affine_map, par_affine_map],
        par_iterator_types,
    )
    def silu_op(a, _out):
        sigmoid = arith.DivFOp(
            one,
            arith.AddFOp(
                one,
                math.exp(arith.NegFOp(a).result),
            ).result,
        ).result
        return arith.MulFOp(a, sigmoid).result

    return silu_op


# equivalent to torch.softmax(a, dim=-1)
# this should be just linalg.softmax, but there's no decomposition
def get_softmax(a: ir.Value, out: ir.Value) -> ir.Value:
    elty = a.type.element_type

    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    max_uninit = tensor.EmptyOp(reduced_shape, elty)

    neg_inf = arith.ConstantOp(elty, float("-inf"))
    max_init = linalg.fill(neg_inf, outs=[max_uninit.result])

    reduce_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank - 1)]
        + [ir.AffineConstantExpr.get(0)],
    )
    identity_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank)],
    )

    iterator_types = [parallel] * (a.type.rank - 1) + [reduction]

    @linalg.generic(
        [a],
        [max_init],
        [identity_map, reduce_map],
        iterator_types,
    )
    def compute_max(val, acc):
        return arith.MaximumFOp(val, acc).result

    shifted_uninit = tensor.EmptyOp(a.type.shape, elty)

    @linalg.generic(
        [a, compute_max],
        [shifted_uninit.result],
        [identity_map, reduce_map, identity_map],
        [parallel] * a.type.rank,
    )
    def subtract_max(val, max_val, _out):
        return arith.SubFOp(val, max_val).result

    exp_uninit = tensor.EmptyOp(a.type.shape, elty)

    @linalg.generic(
        [subtract_max],
        [exp_uninit.result],
        [identity_map, identity_map],
        [parallel] * a.type.rank,
    )
    def compute_exp(val, _out):
        return math.exp(val)

    sum_uninit = tensor.EmptyOp(reduced_shape, elty)
    zero = arith.ConstantOp(elty, 0.0)
    sum_init = linalg.fill(zero, outs=[sum_uninit.result])

    @linalg.generic(
        [compute_exp],
        [sum_init],
        [identity_map, reduce_map],
        iterator_types,
    )
    def compute_sum(val, acc):
        return arith.AddFOp(val, acc).result

    @linalg.generic(
        [compute_exp, compute_sum],
        [out],
        [identity_map, reduce_map, identity_map],
        [parallel] * a.type.rank,
    )
    def divide_by_sum(exp_val, sum_val, _out):
        return arith.DivFOp(exp_val, sum_val).result

    return divide_by_sum


# torch.matmul
def get_matmul(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.matmul(a, b, outs=[out])


# torch.nn.functional.linear
def get_linear(a: ir.Value, w: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    elty = out.type.element_type
    zero = arith.constant(elty, 0.0)
    out_zeroed = linalg.fill(zero, outs=[out])

    # a[i, k] * w[j, k] -> out[i, j]
    i, j, k = [ir.AffineDimExpr.get(d) for d in range(3)]
    a_map = affine_map(3, [i, k])  # (batch, in_feat)
    w_map = affine_map(3, [j, k])  # (out_feat, in_feat)
    out_map = affine_map(3, [i, j])  # (batch, out_feat)

    @linalg.generic(
        [a, w],
        [out_zeroed],
        [a_map, w_map, out_map],
        [parallel, parallel, reduction],
    )
    def matmul_op(a_elem, w_elem, out_elem):
        prod = arith.MulFOp(a_elem, w_elem).result
        return arith.AddFOp(out_elem, prod).result

    # b[j] -> out[i, j]
    i2, j2 = [ir.AffineDimExpr.get(d) for d in range(2)]
    b_map = affine_map(2, [j2])  # (out_feat,)
    out_map2 = affine_map(2, [i2, j2])  # (batch, out_feat)

    @linalg.generic(
        [matmul_op, b],
        [out_zeroed],
        [out_map2, b_map, out_map2],
        [parallel, parallel],
    )
    def add_bias_op(matmul_elem, b_elem, _out):
        return arith.AddFOp(matmul_elem, b_elem).result

    return add_bias_op


def get_l2_norm(a: ir.Value, out: ir.Value, eps: float = 1e-5) -> ir.Value:
    """
    Compute x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)

    Args:
        a: Input tensor
        eps: Epsilon value as a tensor with reduced shape [..., 1]
        out: Output tensor
    """
    elty = a.type.element_type
    # Broadcast epsilon scalar to tensor with reduced shape
    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    eps_const = arith.ConstantOp(elty, eps)
    eps_tensor_uninit = tensor.EmptyOp(reduced_shape, elty)
    eps_tensor = linalg.fill(eps_const, outs=[eps_tensor_uninit])
    # Square the input
    squared_input = tensor.EmptyOp(a.type.shape, elty)
    sqr = get_sqr(a, squared_input)

    # Compute mean along last dimension
    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    mean_uninit = tensor.EmptyOp(reduced_shape, elty)

    mean = get_mean(sqr, mean_uninit)
    mean_plus_eps = get_add(mean, eps_tensor, mean_uninit)
    rsqrt_reduced = get_rsqrt(mean_plus_eps, mean_uninit)

    # (d0, d1) -> (d0, 0) for input, (d0, d1) -> (d0, d1) for output
    input_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank - 1)]
        + [ir.AffineConstantExpr.get(0)],
    )
    output_map = affine_map(
        a.type.rank,
        [ir.AffineDimExpr.get(i) for i in range(a.type.rank)],
    )
    iterator_types = [parallel] * a.type.rank

    @linalg.generic(
        [rsqrt_reduced],
        [out],
        [input_map, output_map],
        iterator_types,
    )
    def broadcast_rsqrt(val, _out):
        return val

    return get_mul(a, broadcast_rsqrt, out)


# equivalent to torch.polar
def get_polar(abs: ir.Value, angle: ir.Value, out: ir.Value) -> ir.Value:
    """
    Convert magnitude and angle to complex number: out = abs * (cos(angle) + i*sin(angle))
    """
    elty = abs.type.element_type
    shape = abs.type.shape
    rank = len(shape)

    # Identity map for element-wise operations
    id_map = affine_map(rank, [ir.AffineDimExpr.get(i) for i in range(rank)])

    # Compute cos(angle) and sin(angle), then multiply by abs to get real and imag parts
    @linalg.generic(
        [abs, angle],
        [out],
        [id_map, id_map, id_map],
        [parallel] * rank,
    )
    def polar_convert(abs_val, angle_val, _out):
        cos_val = math.CosOp(angle_val).result
        sin_val = math.SinOp(angle_val).result
        real_part = arith.MulFOp(abs_val, cos_val).result
        imag_part = arith.MulFOp(abs_val, sin_val).result
        return complex.CreateOp(ir.ComplexType.get(elty), real_part, imag_part).result

    return polar_convert


# equivalent to torch.outer
def get_outer(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    """
    Compute outer product: out[i,j] = a[i] * b[j]

    Assumes inputs are 1-D tensors.
    """
    # Affine maps for outer product: a[i] broadcasts to (i,j), b[j] broadcasts to (i,j)
    a_map = affine_map(2, [ir.AffineDimExpr.get(0)])
    b_map = affine_map(2, [ir.AffineDimExpr.get(1)])
    out_map = affine_map(2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])

    @linalg.generic(
        [a, b],
        [out],
        [a_map, b_map, out_map],
        [parallel, parallel],
    )
    def outer_product(a_val, b_val, _out):
        return arith.MulFOp(a_val, b_val).result

    return outer_product


# with b broadcasting, assuming it has smaller rank
def get_complex_mul(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    rank_b = b.type.rank
    rank_out = out.type.rank

    dim_exprs_a = [ir.AffineDimExpr.get(i) for i in range(rank_out)]

    if rank_b < rank_out:
        offset = rank_out - rank_b
        dim_exprs_b = [ir.AffineConstantExpr.get(0)] * offset + [
            ir.AffineDimExpr.get(i) for i in range(offset, rank_out)
        ]
    else:
        b_shape = list(b.type.shape)
        dim_exprs_b = []
        for i in range(rank_out):
            if i < len(b_shape) and b_shape[i] == 1:
                dim_exprs_b.append(ir.AffineConstantExpr.get(0))
            else:
                dim_exprs_b.append(ir.AffineDimExpr.get(i))

    dim_exprs_out = [ir.AffineDimExpr.get(i) for i in range(rank_out)]

    map_a = affine_map(rank_out, dim_exprs_a)
    map_b = affine_map(rank_out, dim_exprs_b)
    map_out = affine_map(rank_out, dim_exprs_out)

    @linalg.generic(
        [a, b],
        [out],
        [map_a, map_b, map_out],
        [parallel] * rank_out,
    )
    def complex_mul_op(a_val, b_val, _out):
        result = complex.MulOp(a_val, b_val).result
        return result

    return complex_mul_op


def get_rotary_emb(
    xq: ir.Value, xk: ir.Value, freqs_cis: ir.Value, xq_out: ir.Value, xk_out: ir.Value
):
    elty = xq.type.element_type

    xq_shape = list(xq.type.shape)
    xk_shape = list(xk.type.shape)
    batch, seq_len, n_heads, head_dim = xq_shape
    n_kv_heads = xk_shape[2]

    # Reshape xq to (batch, seq_len, n_heads, head_dim//2, 2)
    xq_reshaped_shape = [batch, seq_len, n_heads, head_dim // 2, 2]
    xq_reshaped_type = ir.RankedTensorType.get(xq_reshaped_shape, elty)
    xq_reshaped = tensor.expand_shape(
        xq_reshaped_type,
        xq,
        reassociation=[[0], [1], [2], [3, 4]],
        output_shape=[],
        static_output_shape=xq_reshaped_shape,
    )

    # View xq as complex: (batch, seq_len, n_heads, head_dim//2, 2) -> (batch, seq_len, n_heads, head_dim//2) complex
    xq_complex_shape = [batch, seq_len, n_heads, head_dim // 2]
    xq_complex_uninit = tensor.EmptyOp(
        xq_complex_shape, ir.ComplexType.get(elty)
    ).result
    xq_complex = get_view_as_complex(xq_reshaped, xq_complex_uninit)

    # same for xk
    xk_reshaped_shape = [batch, seq_len, n_kv_heads, head_dim // 2, 2]
    xk_reshaped_type = ir.RankedTensorType.get(xk_reshaped_shape, elty)
    xk_reshaped = tensor.expand_shape(
        xk_reshaped_type,
        xk,
        reassociation=[[0], [1], [2], [3, 4]],
        output_shape=[],
        static_output_shape=xk_reshaped_shape,
    )

    xk_complex_shape = [batch, seq_len, n_kv_heads, head_dim // 2]
    xk_complex_uninit = tensor.EmptyOp(
        xk_complex_shape, ir.ComplexType.get(elty)
    ).result
    xk_complex = get_view_as_complex(xk_reshaped, xk_complex_uninit)

    # Reshape freqs_cis for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_broadcast_shape = [1, seq_len, 1, head_dim // 2]
    freqs_broadcast_uninit = tensor.EmptyOp(freqs_broadcast_shape, elty).result
    freqs_broadcast = get_reshape_for_broadcast(
        freqs_cis, xq_complex, freqs_broadcast_uninit
    )

    # cast freqs_broadcast to complex
    freqs_broadcast_complex_uninit = tensor.EmptyOp(
        freqs_broadcast_shape, ir.ComplexType.get(elty)
    ).result

    d0, d1, d2, d3 = [ir.AffineDimExpr.get(i) for i in range(4)]
    indexing_maps = [
        ir.AffineMap.get(4, 0, [d0, d1, d2, d3]),
        ir.AffineMap.get(4, 0, [d0, d1, d2, d3]),
    ]

    @linalg.generic(
        inputs=[freqs_broadcast],
        outputs=[freqs_broadcast_complex_uninit],
        indexing_maps=indexing_maps,
        iterator_types=["parallel", "parallel", "parallel", "parallel"],
    )
    def real_to_complex(r, out):
        zero = arith.constant(elty, 0.0)
        return complex.CreateOp(ir.ComplexType.get(elty), r, zero).result

    freqs_broadcast_complex = real_to_complex

    # Multiply xq_complex with freqs_broadcast_complex
    xq_rotated_uninit = tensor.EmptyOp(
        xq_complex_shape, ir.ComplexType.get(elty)
    ).result
    xq_rotated = get_complex_mul(xq_complex, freqs_broadcast_complex, xq_rotated_uninit)

    xk_rotated_uninit = tensor.EmptyOp(
        xk_complex_shape, ir.ComplexType.get(elty)
    ).result
    xk_rotated = get_complex_mul(xk_complex, freqs_broadcast_complex, xk_rotated_uninit)

    # view as real
    xq_real_shape = [batch, seq_len, n_heads, head_dim // 2, 2]
    xq_real_uninit = tensor.EmptyOp(xq_real_shape, elty).result
    xq_real = get_view_as_real(xq_rotated, xq_real_uninit)

    xk_real_shape = [batch, seq_len, n_kv_heads, head_dim // 2, 2]
    xk_real_uninit = tensor.EmptyOp(xk_real_shape, elty).result
    xk_real = get_view_as_real(xk_rotated, xk_real_uninit)

    # flatten back to original shape
    xq_final = tensor.collapse_shape(
        xq.type,
        xq_real,
        reassociation=[[0], [1], [2], [3, 4]],
    )

    xk_final = tensor.collapse_shape(
        xk.type,
        xk_real,
        reassociation=[[0], [1], [2], [3, 4]],
    )

    linalg.copy(xq_final, outs=[xq_out])
    linalg.copy(xk_final, outs=[xk_out])


def get_reshape_for_broadcast(freqs_cis: ir.Value, x: ir.Value, out: ir.Value):
    # broadcast freqs_cis[seq, head] -> out[0, seq, 0, head]
    d0, d1, d2, d3 = [ir.AffineDimExpr.get(i) for i in range(4)]

    in_map = affine_map(4, [d1, d3])
    out_map = affine_map(4, [d0, d1, d2, d3])

    @linalg.generic(
        [freqs_cis],
        [out],
        [in_map, out_map],
        [parallel, parallel, parallel, parallel],
    )
    def reshape_op(val, _out):
        return val

    return reshape_op


# torch.view_as_complex
def get_view_as_complex(x: ir.Value, out: ir.Value) -> ir.Value:
    elty = x.type.element_type
    rank = x.type.rank
    shape = list(x.type.shape)
    assert shape[-1] == 2, "Last dimension must be of size 2 to form complex numbers"

    rank_out = rank - 1
    dim_exprs_out = [ir.AffineDimExpr.get(i) for i in range(rank_out)]

    # real part: access input[d0, d1, ..., d_{rank-2}, 0]
    dim_exprs_real = dim_exprs_out + [ir.AffineConstantExpr.get(0)]
    # imag part: access input[d0, d1, ..., d_{rank-2}, 1]
    dim_exprs_imag = dim_exprs_out + [ir.AffineConstantExpr.get(1)]

    input_map_real = affine_map(rank_out, dim_exprs_real)
    input_map_imag = affine_map(rank_out, dim_exprs_imag)
    output_map = affine_map(rank_out, dim_exprs_out)

    @linalg.generic(
        [x, x],  # Same input tensor accessed twice with different maps
        [out],
        [input_map_real, input_map_imag, output_map],
        [parallel] * rank_out,
    )
    def view_as_complex_op(r, i, _out):
        cplx = complex.CreateOp(ir.ComplexType.get(elty), r, i).result
        return cplx

    return view_as_complex_op


# torch.view_as_real
def get_view_as_real(x: ir.Value, out: ir.Value) -> ir.Value:
    rank = x.type.rank

    # Output has shape [..., 2]
    # extract real part to [..., 0] and imag part to [..., 1]

    dim_exprs_in = [ir.AffineDimExpr.get(i) for i in range(rank)]

    # For real part: write to output[..., 0]
    dim_exprs_real = dim_exprs_in + [ir.AffineConstantExpr.get(0)]
    # For imag part: write to output[..., 1]
    dim_exprs_imag = dim_exprs_in + [ir.AffineConstantExpr.get(1)]

    input_map = affine_map(rank, dim_exprs_in)
    output_map_real = affine_map(rank, dim_exprs_real)
    output_map_imag = affine_map(rank, dim_exprs_imag)

    @linalg.generic(
        [x],
        [out],
        [input_map, output_map_real],
        [parallel] * rank,
    )
    def write_real(cplx, _out):
        return complex.ReOp(cplx).result

    @linalg.generic(
        [x],
        [write_real],
        [input_map, output_map_imag],
        [parallel] * rank,
    )
    def write_imag(cplx, _out):
        return complex.ImOp(cplx).result

    return write_imag


#### Test cases #####


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_emb_ref(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


references = {
    get_add: torch.add,
    get_mul: torch.mul,
    get_matmul: torch.matmul,
    get_rsqrt: torch.rsqrt,
    get_sqr: torch.square,
    get_mean: lambda x: torch.mean(x, dim=-1, keepdim=True),
    get_silu: lambda x: torch.nn.functional.silu(x),
    get_softmax: lambda x: torch.softmax(x, dim=-1),
    get_polar: torch.polar,
    get_outer: torch.outer,
    get_linear: torch.nn.functional.linear,
    get_repeat_kv: repeat_kv,
    get_l2_norm: lambda x, eps: x
    * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps),
    get_rotary_emb: rotary_emb_ref,
}


# TODO: torch_dtype_to_mlir_type
def to_ir_type(type_str, ctx):
    if type_str == "f32":
        return ir.F32Type.get(context=ctx)
    elif type_str == "f64":
        return ir.F64Type.get(context=ctx)
    else:
        raise ValueError(f"Unsupported type: {type_str}")


@pytest.mark.parametrize(
    "op,shape,elem_type",
    [
        (get_add, (4, 16), "f32"),
        (get_mul, (4, 16), "f32"),
        (get_matmul, (16, 16), "f32"),
        (get_outer, (16,), "f32"),
    ],
)
def test_bin_op(op, shape, elem_type):
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tensor_type = ir.RankedTensorType.get(shape, elty)

                # Outer product produces [M, M] output for 1-D input of size M
                if op == get_outer:
                    out_shape = (shape[0], shape[0])
                    out_tensor_type = ir.RankedTensorType.get(out_shape, elty)
                else:
                    out_tensor_type = tensor_type

                @func.FuncOp.from_py_func(
                    tensor_type, tensor_type, out_tensor_type, name="bin_op"
                )
                def bin_op(a, b, out):
                    op(a, b, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type(elem_type, ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("bin_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    a = torch.randn(*shape, dtype=torch_dtype)
    b = torch.randn(*shape, dtype=torch_dtype)
    out_ref = references[op](a, b)
    out = torch.empty_like(out_ref)
    out.zero_()

    a_mem = get_ranked_memref_descriptor(a.numpy())
    b_mem = get_ranked_memref_descriptor(b.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([a_mem, b_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@pytest.mark.parametrize(
    "op,shape,elem_type",
    [
        (get_rsqrt, (4, 16), "f32"),
        (get_mean, (4, 16), "f32"),
        (get_sqr, (4, 16), "f32"),
        (get_silu, (4, 16), "f32"),
        (get_softmax, (4, 16), "f32"),
    ],
)
def test_unary_op(op, shape, elem_type):
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tensor_type = ir.RankedTensorType.get(shape, elty)

                # For mean operation, output has different shape (reduction on last dim)
                if op == get_mean:
                    out_shape = list(shape)
                    out_shape[-1] = 1
                    out_tensor_type = ir.RankedTensorType.get(out_shape, elty)
                else:
                    out_tensor_type = tensor_type

                @func.FuncOp.from_py_func(tensor_type, out_tensor_type, name="unary_op")
                def unary_op(a, out):
                    op(a, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type(elem_type, ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("unary_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    a = torch.randn(*shape, dtype=torch_dtype)
    out_ref = references[op](a)
    out = torch.empty_like(out_ref)

    a_mem = get_ranked_memref_descriptor(a.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([a_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@pytest.mark.parametrize("shape,elem_type", [((4, 16), "f32")])
def test_rms_norm(shape, elem_type):
    eps = 1e-5

    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                input_type = ir.RankedTensorType.get(shape, elty)

                @func.FuncOp.from_py_func(input_type, input_type, name="rms_norm")
                def rms_norm(a, out):
                    get_l2_norm(a, out, eps)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type(elem_type, ctx)
    module = generate_module(ctx, ir_type)
    print(module)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)
    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("rms_norm")
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    a = torch.randn(*shape, dtype=torch_dtype)
    out_ref = references[get_l2_norm](a, eps)
    out = torch.empty_like(out_ref)
    a_mem = get_ranked_memref_descriptor(a.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([a_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_linear():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                input_type = ir.RankedTensorType.get((4, 16), elty)
                weight_type = ir.RankedTensorType.get((32, 16), elty)
                bias_type = ir.RankedTensorType.get((32,), elty)
                output_type = ir.RankedTensorType.get((4, 32), elty)

                @func.FuncOp.from_py_func(
                    input_type, weight_type, bias_type, output_type, name="linear_op"
                )
                def linear_op(x, w, b, out):
                    get_linear(x, w, b, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("linear_op")
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(4, 16, dtype=torch_dtype)
    w = torch.randn(32, 16, dtype=torch_dtype)
    b = torch.randn(32, dtype=torch_dtype)
    out_ref = references[get_linear](x, w, b)
    out = torch.empty_like(out_ref)
    out.zero_()
    x_mem = get_ranked_memref_descriptor(x.numpy())
    w_mem = get_ranked_memref_descriptor(w.numpy())
    b_mem = get_ranked_memref_descriptor(b.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([x_mem, w_mem, b_mem, out_mem])
    func_ptr(args)
    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_polar():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tensor_type = ir.RankedTensorType.get((4, 16), elty)
                complex_tensor_type = ir.RankedTensorType.get(
                    (4, 16), ir.ComplexType.get(elty)
                )

                @func.FuncOp.from_py_func(
                    tensor_type, tensor_type, complex_tensor_type, name="polar_op"
                )
                def polar_op(magnitude, angle, out):
                    get_polar(magnitude, angle, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("polar_op")
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    magnitude = torch.randn(4, 16, dtype=torch_dtype)
    angle = torch.randn(4, 16, dtype=torch_dtype)
    out_ref = references[get_polar](magnitude, angle)
    out = torch.empty_like(out_ref)
    magnitude_mem = get_ranked_memref_descriptor(magnitude.numpy())
    angle_mem = get_ranked_memref_descriptor(angle.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([magnitude_mem, angle_mem, out_mem])
    func_ptr(args)
    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_repeat_kv():
    def generate_module(ctx, elty, n_rep):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                x_type = ir.RankedTensorType.get((2, 512, 8, 64), elty)
                out_type = ir.RankedTensorType.get((2, 512, 8 * n_rep, 64), elty)

                @func.FuncOp.from_py_func(x_type, out_type, name="repeat_kv_op")
                def repeat_kv_op(x, out):
                    get_repeat_kv(x, n_rep, out)

        return module

    n_rep = 4
    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type, n_rep)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("repeat_kv_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(2, 512, 8, 64, dtype=torch_dtype)
    out_ref = references[get_repeat_kv](x, n_rep)
    out = torch.empty_like(out_ref)

    x_mem = get_ranked_memref_descriptor(x.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([x_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_reshape_for_broadcast():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                freqs_cis_type = ir.RankedTensorType.get((512, 64), elty)
                x_type = ir.RankedTensorType.get((2, 512, 32, 128), elty)
                out_type = ir.RankedTensorType.get((1, 512, 1, 64), elty)

                @func.FuncOp.from_py_func(
                    freqs_cis_type, x_type, out_type, name="reshape_for_broadcast"
                )
                def reshape_for_broadcast_op(freqs_cis, x, out):
                    get_reshape_for_broadcast(freqs_cis, x, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("reshape_for_broadcast")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    freqs_cis = torch.randn(512, 64, dtype=torch_dtype)
    x = torch.randn(2, 512, 32, 128, dtype=torch_dtype)
    # Convert x to complex view as expected by reshape_for_broadcast
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    out_ref = reshape_for_broadcast(freqs_cis, x_complex)
    out = torch.empty_like(out_ref)

    freqs_cis_mem = get_ranked_memref_descriptor(freqs_cis.numpy())
    x_mem = get_ranked_memref_descriptor(x.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([freqs_cis_mem, x_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_view_as_complex():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # Input should be reshaped to have last dim = 2
                x_type = ir.RankedTensorType.get((2, 512, 32, 64, 2), elty)
                out_type = ir.RankedTensorType.get(
                    (2, 512, 32, 64), ir.ComplexType.get(elty)
                )

                @func.FuncOp.from_py_func(x_type, out_type, name="view_as_complex_op")
                def view_as_complex_op(x, out):
                    get_view_as_complex(x, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("view_as_complex_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(2, 512, 32, 128, dtype=torch_dtype)
    # Reshape to (2, 512, 32, 64, 2) before passing to the function
    x_reshaped = x.reshape(2, 512, 32, 64, 2)
    out_ref = torch.view_as_complex(x_reshaped)
    out = torch.empty_like(out_ref)

    x_mem = get_ranked_memref_descriptor(x_reshaped.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([x_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


def test_view_as_real():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                x_type = ir.RankedTensorType.get(
                    (2, 512, 32, 64), ir.ComplexType.get(elty)
                )
                out_type = ir.RankedTensorType.get((2, 512, 32, 64, 2), elty)

                @func.FuncOp.from_py_func(x_type, out_type, name="as_real_op")
                def as_real_op(x, out):
                    get_view_as_real(x, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("as_real_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(2, 512, 32, 64, 2, dtype=torch_dtype)
    x_complex = torch.view_as_complex(x)
    out_ref = torch.view_as_real(x_complex)
    out = torch.empty_like(out_ref)

    x_mem = get_ranked_memref_descriptor(x_complex.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([x_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@pytest.mark.parametrize(
    "batch_size,seq_len,n_heads,head_dim,n_kv_heads,elem_type",
    [(2, 512, 32, 128, 8, "f32")],
)
def test_rotary_emb(batch_size, seq_len, n_heads, head_dim, n_kv_heads, elem_type):
    def generate_module(ctx, elty, xq_shape, xk_shape, freqs_cis_shape):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                xq_type = ir.RankedTensorType.get(xq_shape, elty)
                xk_type = ir.RankedTensorType.get(xk_shape, elty)
                freqs_cis_type = ir.RankedTensorType.get(freqs_cis_shape, elty)

                @func.FuncOp.from_py_func(
                    xq_type,
                    xk_type,
                    freqs_cis_type,
                    xq_type,
                    xk_type,
                    name="rotary_emb",
                )
                def rotary_emb(xq, xk, freqs_cis, xq_out, xk_out):
                    get_rotary_emb(xq, xk, freqs_cis, xq_out, xk_out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type(elem_type, ctx)
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    xq_shape = (batch_size, seq_len, n_heads, head_dim)
    xk_shape = (batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_shape = (seq_len, head_dim // 2)
    xq = torch.randn(*xq_shape, dtype=torch_dtype)
    xk = torch.randn(*xk_shape, dtype=torch_dtype)
    freqs_cis = torch.randn(*freqs_cis_shape, dtype=torch_dtype)
    xq_out, xk_out = references[get_rotary_emb](xq, xk, freqs_cis)

    module = generate_module(
        ctx,
        xq_shape=xq_shape,
        xk_shape=xk_shape,
        freqs_cis_shape=freqs_cis_shape,
        elty=ir_type,
    )
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("rotary_emb")

    out1 = torch.empty_like(xq_out)
    out2 = torch.empty_like(xk_out)

    a_mem = get_ranked_memref_descriptor(xq.numpy())
    b_mem = get_ranked_memref_descriptor(xk.numpy())
    freqs_cis_mem = get_ranked_memref_descriptor(freqs_cis.numpy())
    out1_mem = get_ranked_memref_descriptor(out1.numpy())
    out2_mem = get_ranked_memref_descriptor(out2.numpy())
    args = lh_utils.memrefs_to_packed_args(
        [a_mem, b_mem, freqs_cis_mem, out1_mem, out2_mem]
    )
    func_ptr(args)

    assert torch.allclose(out1, xq_out, rtol=0.01, atol=0.01, equal_nan=True)
    assert torch.allclose(out2, xk_out, rtol=0.01, atol=0.01, equal_nan=True)


def test_feed_forward():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                input_type = ir.RankedTensorType.get((4, 16), elty)
                hidden_type = ir.RankedTensorType.get((4, 64), elty)
                output_type = ir.RankedTensorType.get((4, 16), elty)
                weight1_type = ir.RankedTensorType.get((64, 16), elty)
                bias1_type = ir.RankedTensorType.get((64,), elty)
                weight2_type = ir.RankedTensorType.get((16, 64), elty)
                bias2_type = ir.RankedTensorType.get((16,), elty)
                weight3_type = ir.RankedTensorType.get((64, 16), elty)
                bias3_type = ir.RankedTensorType.get((64,), elty)

                @func.FuncOp.from_py_func(
                    input_type,
                    weight1_type,
                    bias1_type,
                    weight2_type,
                    bias2_type,
                    weight3_type,
                    bias3_type,
                    output_type,
                    name="feed_forward",
                )
                def feed_forward(x, w1, b1, w2, b2, w3, b3, out):
                    # Compute hidden = linear(x, w1, b1)
                    hidden_uninit = tensor.EmptyOp(hidden_type.shape, elty).result
                    hidden = get_linear(x, w1, b1, hidden_uninit)

                    # Compute hidden_silu = silu(hidden)
                    hidden_silu_uninit = tensor.EmptyOp(hidden_type.shape, elty).result
                    hidden_silu = get_silu(hidden, hidden_silu_uninit)

                    # Compute gate = linear(x, w3, b3)
                    gate_uninit = tensor.EmptyOp(hidden_type.shape, elty).result
                    gate = get_linear(x, w3, b3, gate_uninit)

                    # Compute activated = hidden_silu * gate
                    activated_uninit = tensor.EmptyOp(hidden_type.shape, elty).result
                    activated = get_mul(hidden_silu, gate, activated_uninit)

                    # Compute out = linear(activated, w2, b2)
                    get_linear(activated, w2, b2, out)

        return module

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("feed_forward")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(4, 16, dtype=torch_dtype)
    w1 = torch.randn(64, 16, dtype=torch_dtype)
    b1 = torch.randn(64, dtype=torch_dtype)
    w2 = torch.randn(16, 64, dtype=torch_dtype)
    b2 = torch.randn(16, dtype=torch_dtype)
    w3 = torch.randn(64, 16, dtype=torch_dtype)
    b3 = torch.randn(64, dtype=torch_dtype)

    hidden_ref = torch.nn.functional.linear(x, w1, b1)
    activated_ref = torch.nn.functional.silu(hidden_ref)
    activated_ref *= torch.nn.functional.linear(x, w3, b3)
    out_ref = torch.nn.functional.linear(activated_ref, w2, b2)
    out = torch.empty_like(out_ref)
    out.zero_()
    x_mem = get_ranked_memref_descriptor(x.numpy())
    w1_mem = get_ranked_memref_descriptor(w1.numpy())
    b1_mem = get_ranked_memref_descriptor(b1.numpy())
    w2_mem = get_ranked_memref_descriptor(w2.numpy())
    b2_mem = get_ranked_memref_descriptor(b2.numpy())
    w3_mem = get_ranked_memref_descriptor(w3.numpy())
    b3_mem = get_ranked_memref_descriptor(b3.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args(
        [x_mem, w1_mem, b1_mem, w2_mem, b2_mem, w3_mem, b3_mem, out_mem]
    )
    func_ptr(args)
    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)
