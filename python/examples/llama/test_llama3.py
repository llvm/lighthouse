# RUN: %pytest %s

from dataclasses import dataclass
import functools
import math as pymath
import pytest
import torch
from typing import Optional, Tuple


from mlir import ir
from mlir.dialects import transform, func, linalg, tensor, arith, complex, math
from mlir.dialects.transform import structured
from mlir.dialects.transform import interpreter
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import (
    get_ranked_memref_descriptor,
)
from mlir.execution_engine import ExecutionEngine

from ref_model import (
    Attention,
    ModelArgs,
    reshape_for_broadcast,
    apply_rotary_emb,
    repeat_kv,
    TransformerBlock,
    Transformer,
)
from lighthouse import utils as lh_utils


def with_mlir_ctx_and_location(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ir.Context(), ir.Location.unknown():
            return func(*args, **kwargs)

    return wrapper


def affine_map(dim_count, exprs, *, symb_count=0):
    return ir.AffineMap.get(dim_count, symb_count, exprs)


parallel = linalg.IteratorType.parallel
reduction = linalg.IteratorType.reduction


def create_pass_pipeline(ctx: ir.Context) -> PassManager:
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


def create_schedule() -> ir.Module:
    """
    Create an MLIR module containing transformation schedule.
    The schedule provides partial lowering to scalar operations.

    Args:
        ctx: MLIR context.
    """
    # Create transform module.
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()

    # Create entry point transformation sequence.
    with ir.InsertionPoint(schedule.body):
        named_seq = transform.named_sequence(
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
        func = structured.MatchOp.match_op_names(named_seq.bodyTarget, ["func.func"])

        # Use C interface wrappers - required to make function executable after jitting.
        func = transform.apply_registered_pass(anytype, func, "llvm-request-c-wrappers")

        # Find the kernel's module op.
        mod = transform.get_parent_op(
            anytype, func, op_name="builtin.module", deduplicate=True
        )

        # Naive lowering to loops.
        mod = transform.apply_registered_pass(anytype, mod, "convert-linalg-to-loops")
        # Cleanup.
        transform.apply_cse(mod)
        with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
            transform.ApplyCanonicalizationPatternsOp()

        # Terminate the schedule.
        transform.YieldOp()
    return schedule


def bufferize_module(kernel: ir.Module) -> None:
    pm = PassManager("builtin.module")
    pm.add("one-shot-bufferize{bufferize-function-boundaries}")
    pm.run(kernel.operation)


def apply_schedule(kernel: ir.Module, schedule: ir.Module) -> None:
    bufferize_module(kernel)
    interpreter.apply_named_sequence(
        payload_root=kernel,
        transform_root=schedule.body.operations[0],
        transform_module=schedule,
    )
    pm = create_pass_pipeline(kernel.context)
    pm.run(kernel.operation)


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
        sigmoid = arith.divf(
            one,
            arith.addf(
                one,
                math.exp(arith.negf(a)),
            ),
        )
        return arith.mulf(a, sigmoid)

    return silu_op


# equivalent to torch.softmax(a, dim=-1)
# this should be just linalg.softmax, but there's no decomposition
def get_softmax(a: ir.Value, out: ir.Value) -> ir.Value:
    elty = a.type.element_type

    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    max_uninit = tensor.empty(reduced_shape, elty)

    neg_inf = arith.constant(elty, float("-inf"))
    max_init = linalg.fill(neg_inf, outs=[max_uninit])

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
        return arith.maximumf(val, acc)

    shifted_uninit = tensor.empty(a.type.shape, elty)

    @linalg.generic(
        [a, compute_max],
        [shifted_uninit],
        [identity_map, reduce_map, identity_map],
        [parallel] * a.type.rank,
    )
    def subtract_max(val, max_val, _out):
        return arith.subf(val, max_val)

    exp_uninit = tensor.empty(a.type.shape, elty)

    @linalg.generic(
        [subtract_max],
        [exp_uninit],
        [identity_map, identity_map],
        [parallel] * a.type.rank,
    )
    def compute_exp(val, _out):
        return math.exp(val)

    sum_uninit = tensor.empty(reduced_shape, elty)
    zero = arith.constant(elty, 0.0)
    sum_init = linalg.fill(zero, outs=[sum_uninit])

    @linalg.generic(
        [compute_exp],
        [sum_init],
        [identity_map, reduce_map],
        iterator_types,
    )
    def compute_sum(val, acc):
        return arith.addf(val, acc)

    @linalg.generic(
        [compute_exp, compute_sum],
        [out],
        [identity_map, reduce_map, identity_map],
        [parallel] * a.type.rank,
    )
    def divide_by_sum(exp_val, sum_val, _out):
        return arith.divf(exp_val, sum_val)

    return divide_by_sum


# torch.triu
def get_triu(a: ir.Value, out: ir.Value) -> ir.Value:
    elty = a.type.element_type
    zero = arith.constant(elty, 0.0)

    rank = a.type.rank
    dims = [ir.AffineDimExpr.get(i) for i in range(rank)]
    par_affine_map = affine_map(rank, dims)
    par_iterator_types = [parallel] * rank

    @linalg.generic(
        [a],
        [out],
        [par_affine_map, par_affine_map],
        par_iterator_types,
    )
    def triu_op(a_elem, _out):
        i = linalg.IndexOp(rank - 2).result
        j = linalg.IndexOp(rank - 1).result
        cond = arith.cmpi(arith.CmpIPredicate.sle, i, j)
        result = arith.select(cond, a_elem, zero)
        return result

    return triu_op


# torch.matmul
def get_matmul(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    return linalg.matmul(a, b, outs=[out])


# torch.nn.functional.linear
def get_linear(a: ir.Value, w: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
    elty = out.type.element_type
    zero = arith.constant(elty, 0.0)
    out_zeroed = linalg.fill(zero, outs=[out])

    a_rank = a.type.rank
    out_rank = out.type.rank

    # For matmul: a[...batch..., k] * w[j, k] -> out[...batch..., j]
    num_dims = a_rank + 1

    dims = [ir.AffineDimExpr.get(d) for d in range(num_dims)]

    # a maps to [...batch..., k] where k is the last (reduction) dimension
    batch_dims = dims[: a_rank - 1]
    k = dims[-1]  # reduction dimension
    a_map = affine_map(num_dims, batch_dims + [k])

    # w maps to [j, k] where j is the output feature dimension
    j = dims[a_rank - 1]  # after batch dims, before k
    w_map = affine_map(num_dims, [j, k])

    # out maps to [...batch..., j]
    out_map = affine_map(num_dims, batch_dims + [j])

    iterator_types = [parallel] * (a_rank - 1) + [parallel, reduction]

    @linalg.generic(
        [a, w],
        [out_zeroed],
        [a_map, w_map, out_map],
        iterator_types,
    )
    def matmul_op(a_elem, w_elem, out_elem):
        prod = arith.mulf(a_elem, w_elem)
        return arith.addf(out_elem, prod)

    out_dims = [ir.AffineDimExpr.get(d) for d in range(out_rank)]
    b_map = affine_map(out_rank, [out_dims[-1]])
    out_map2 = affine_map(out_rank, out_dims)

    bias_iterator_types = [parallel] * out_rank

    @linalg.generic(
        [matmul_op, b],
        [out_zeroed],
        [out_map2, b_map, out_map2],
        bias_iterator_types,
    )
    def add_bias_op(matmul_elem, b_elem, _out):
        return arith.addf(matmul_elem, b_elem)

    return add_bias_op


# x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
def get_l2_norm(a: ir.Value, out: ir.Value, eps: float = 1e-5) -> ir.Value:
    elty = a.type.element_type
    # Broadcast epsilon scalar to tensor with reduced shape
    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    eps_const = arith.constant(elty, eps)
    eps_tensor_uninit = tensor.empty(reduced_shape, elty)
    eps_tensor = linalg.fill(eps_const, outs=[eps_tensor_uninit])
    # Square the input
    squared_input = tensor.empty(a.type.shape, elty)
    sqr = get_sqr(a, squared_input)

    # Compute mean along last dimension
    reduced_shape = list(a.type.shape)
    reduced_shape[-1] = 1
    mean_uninit = tensor.empty(reduced_shape, elty)

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
        cos_val = math.cos(angle_val)
        sin_val = math.sin(angle_val)
        real_part = arith.mulf(abs_val, cos_val)
        imag_part = arith.mulf(abs_val, sin_val)
        return complex.create_(ir.ComplexType.get(elty), real_part, imag_part)

    return polar_convert


# equivalent to torch.outer (out[i,j] = a[i] * b[j])
def get_outer(a: ir.Value, b: ir.Value, out: ir.Value) -> ir.Value:
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
        return arith.mulf(a_val, b_val)

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
        return complex.mul(a_val, b_val)

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
    xq_complex_uninit = tensor.empty(xq_complex_shape, ir.ComplexType.get(elty))
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
    xk_complex_uninit = tensor.empty(xk_complex_shape, ir.ComplexType.get(elty))
    xk_complex = get_view_as_complex(xk_reshaped, xk_complex_uninit)

    # Reshape freqs_cis for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_broadcast_shape = [1, seq_len, 1, head_dim // 2]
    freqs_broadcast_uninit = tensor.empty(freqs_broadcast_shape, elty)
    freqs_broadcast = get_reshape_for_broadcast(
        freqs_cis, xq_complex, freqs_broadcast_uninit
    )

    # cast freqs_broadcast to complex
    freqs_broadcast_complex_uninit = tensor.empty(
        freqs_broadcast_shape, ir.ComplexType.get(elty)
    )

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
        return complex.create_(ir.ComplexType.get(elty), r, zero)

    freqs_broadcast_complex = real_to_complex

    # Multiply xq_complex with freqs_broadcast_complex
    xq_rotated_uninit = tensor.empty(xq_complex_shape, ir.ComplexType.get(elty))
    xq_rotated = get_complex_mul(xq_complex, freqs_broadcast_complex, xq_rotated_uninit)

    xk_rotated_uninit = tensor.empty(xk_complex_shape, ir.ComplexType.get(elty))
    xk_rotated = get_complex_mul(xk_complex, freqs_broadcast_complex, xk_rotated_uninit)

    # view as real
    xq_real_shape = [batch, seq_len, n_heads, head_dim // 2, 2]
    xq_real_uninit = tensor.empty(xq_real_shape, elty)
    xq_real = get_view_as_real(xq_rotated, xq_real_uninit)

    xk_real_shape = [batch, seq_len, n_kv_heads, head_dim // 2, 2]
    xk_real_uninit = tensor.empty(xk_real_shape, elty)
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
        return complex.create_(ir.ComplexType.get(elty), r, i)

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
        return complex.re(cplx)

    @linalg.generic(
        [x],
        [write_real],
        [input_map, output_map_imag],
        [parallel] * rank,
    )
    def write_imag(cplx, _out):
        return complex.im(cplx)

    return write_imag


def get_attention(
    args: ModelArgs,
    x: ir.Value,
    wq: ir.Value,
    wk: ir.Value,
    wv: ir.Value,
    wo: ir.Value,
    freqs_cis: ir.Value,
    mask: ir.Value,
    out: ir.Value,
) -> ir.Value:
    elty = x.type.element_type
    batch, seq_len, dim = x.type.shape
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    head_dim = args.dim // args.n_heads
    n_rep = n_heads // n_kv_heads

    # Q, K, V projections
    # xq = linear(x, wq) -> (batch, seq_len, n_heads * head_dim)
    xq_shape = [batch, seq_len, n_heads * head_dim]
    xq_uninit = tensor.empty(xq_shape, elty)
    bq_zeros = tensor.empty([n_heads * head_dim], elty)
    zero = arith.constant(elty, 0.0)
    bq = linalg.fill(zero, outs=[bq_zeros])
    xq_flat = get_linear(x, wq, bq, xq_uninit)

    # Reshape to (batch, seq_len, n_heads, head_dim)
    xq_reshaped_shape = [batch, seq_len, n_heads, head_dim]
    xq_reshaped_type = ir.RankedTensorType.get(xq_reshaped_shape, elty)
    xq = tensor.expand_shape(
        xq_reshaped_type,
        xq_flat,
        reassociation=[[0], [1], [2, 3]],
        output_shape=[],
        static_output_shape=xq_reshaped_shape,
    )

    # xk = linear(x, wk) -> (batch, seq_len, n_kv_heads * head_dim)
    xk_shape = [batch, seq_len, n_kv_heads * head_dim]
    xk_uninit = tensor.empty(xk_shape, elty)
    bk_zeros = tensor.empty([n_kv_heads * head_dim], elty)
    bk = linalg.fill(zero, outs=[bk_zeros])
    xk_flat = get_linear(x, wk, bk, xk_uninit)

    # Reshape to (batch, seq_len, n_kv_heads, head_dim)
    xk_reshaped_shape = [batch, seq_len, n_kv_heads, head_dim]
    xk_reshaped_type = ir.RankedTensorType.get(xk_reshaped_shape, elty)
    xk = tensor.expand_shape(
        xk_reshaped_type,
        xk_flat,
        reassociation=[[0], [1], [2, 3]],
        output_shape=[],
        static_output_shape=xk_reshaped_shape,
    )

    # xv = linear(x, wv) -> (batch, seq_len, n_kv_heads * head_dim)
    xv_shape = [batch, seq_len, n_kv_heads * head_dim]
    xv_uninit = tensor.empty(xv_shape, elty)
    bv_zeros = tensor.empty([n_kv_heads * head_dim], elty)
    bv = linalg.fill(zero, outs=[bv_zeros])
    xv_flat = get_linear(x, wv, bv, xv_uninit)

    # Reshape to (batch, seq_len, n_kv_heads, head_dim)
    xv_reshaped_shape = [batch, seq_len, n_kv_heads, head_dim]
    xv_reshaped_type = ir.RankedTensorType.get(xv_reshaped_shape, elty)
    xv = tensor.expand_shape(
        xv_reshaped_type,
        xv_flat,
        reassociation=[[0], [1], [2, 3]],
        output_shape=[],
        static_output_shape=xv_reshaped_shape,
    )

    # Apply rotary embeddings
    xq_rot_uninit = tensor.empty([batch, seq_len, n_heads, head_dim], elty)
    xk_rot_uninit = tensor.empty([batch, seq_len, n_kv_heads, head_dim], elty)
    get_rotary_emb(xq, xk, freqs_cis, xq_rot_uninit, xk_rot_uninit)
    xq_rot = xq_rot_uninit
    xk_rot = xk_rot_uninit

    # Repeat K/V if using GQA (n_kv_heads < n_heads)
    if n_rep > 1:
        keys_repeated_uninit = tensor.empty([batch, seq_len, n_heads, head_dim], elty)
        keys = get_repeat_kv(xk_rot, n_rep, keys_repeated_uninit)
        values_repeated_uninit = tensor.empty([batch, seq_len, n_heads, head_dim], elty)
        values = get_repeat_kv(xv, n_rep, values_repeated_uninit)
    else:
        keys = xk_rot
        values = xv

    # Transpose for attention: (batch, n_heads, seq_len, head_dim)
    xq_t_shape = [batch, n_heads, seq_len, head_dim]
    xq_t = tensor.empty(xq_t_shape, elty)

    # Permute [0, 2, 1, 3]
    d0, d1, d2, d3 = [ir.AffineDimExpr.get(i) for i in range(4)]
    xq_perm_map = affine_map(4, [d0, d2, d1, d3])
    xq_t_map = affine_map(4, [d0, d1, d2, d3])

    @linalg.generic(
        [xq_rot],
        [xq_t],
        [xq_perm_map, xq_t_map],
        [parallel] * 4,
    )
    def transpose_xq(val, _out):
        return val

    xq_transposed = transpose_xq

    # Transpose keys and values similarly
    keys_t = tensor.empty(xq_t_shape, elty)

    @linalg.generic(
        [keys],
        [keys_t],
        [xq_perm_map, xq_t_map],
        [parallel] * 4,
    )
    def transpose_k(val, _out):
        return val

    keys_transposed = transpose_k

    values_t = tensor.empty(xq_t_shape, elty)

    @linalg.generic(
        [values],
        [values_t],
        [xq_perm_map, xq_t_map],
        [parallel] * 4,
    )
    def transpose_v(val, _out):
        return val

    values_transposed = transpose_v

    # Compute attention scores: matmul(xq, keys.transpose(-2, -1))
    # xq_transposed: (batch, n_heads, seq_len, head_dim)
    # keys_transposed: (batch, n_heads, seq_len, head_dim) -> transpose to (batch, n_heads, head_dim, seq_len)
    # scores: (batch, n_heads, seq_len, seq_len)
    scores_shape = [batch, n_heads, seq_len, seq_len]
    scores_uninit = tensor.empty(scores_shape, elty)
    scores_zeroed = linalg.fill(zero, outs=[scores_uninit])

    # Batched matmul with transpose
    b, h, s1, s2, d = [ir.AffineDimExpr.get(i) for i in range(5)]
    xq_scores_map = affine_map(5, [b, h, s1, d])
    keys_scores_map = affine_map(5, [b, h, s2, d])  # Will read from transposed position
    scores_map = affine_map(5, [b, h, s1, s2])

    @linalg.generic(
        [xq_transposed, keys_transposed],
        [scores_zeroed],
        [xq_scores_map, keys_scores_map, scores_map],
        [parallel, parallel, parallel, parallel, reduction],
    )
    def compute_scores(q_val, k_val, score_val):
        prod = arith.mulf(q_val, k_val)
        return arith.addf(score_val, prod)

    scores_raw = compute_scores

    # Scale by 1/sqrt(head_dim)
    scale_val = 1.0 / pymath.sqrt(head_dim)
    scale_const = arith.constant(elty, scale_val)
    scores_scaled_uninit = tensor.empty(scores_shape, elty)

    d0, d1, d2, d3 = [ir.AffineDimExpr.get(i) for i in range(4)]
    identity_map = affine_map(4, [d0, d1, d2, d3])

    @linalg.generic(
        [scores_raw],
        [scores_scaled_uninit],
        [identity_map, identity_map],
        [parallel] * 4,
    )
    def scale_scores(score, _out):
        return arith.mulf(score, scale_const)

    scores_scaled = scale_scores

    # Apply mask if provided (add mask to scores)
    if mask is not None:
        scores_masked_uninit = tensor.empty(scores_shape, elty)
        scores_final = get_add(scores_scaled, mask, scores_masked_uninit)
    else:
        scores_final = scores_scaled

    # Apply softmax
    scores_softmax_uninit = tensor.empty(scores_shape, elty)
    attn_weights = get_softmax(scores_final, scores_softmax_uninit)

    # Compute output: matmul(attn_weights, values)
    # attn_weights: (batch, n_heads, seq_len, seq_len)
    # values_transposed: (batch, n_heads, seq_len, head_dim)
    # output: (batch, n_heads, seq_len, head_dim)
    attn_out_shape = [batch, n_heads, seq_len, head_dim]
    attn_out_uninit = tensor.empty(attn_out_shape, elty)
    attn_out_zeroed = linalg.fill(zero, outs=[attn_out_uninit])

    b, h, s1, s2, d = [ir.AffineDimExpr.get(i) for i in range(5)]
    attn_map = affine_map(5, [b, h, s1, s2])
    values_map = affine_map(5, [b, h, s2, d])
    out_map = affine_map(5, [b, h, s1, d])

    @linalg.generic(
        [attn_weights, values_transposed],
        [attn_out_zeroed],
        [attn_map, values_map, out_map],
        [parallel, parallel, parallel, parallel, reduction],
    )
    def compute_attn_out(attn_val, v_val, out_val):
        prod = arith.mulf(attn_val, v_val)
        return arith.addf(out_val, prod)

    attn_out = compute_attn_out

    # Transpose back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
    attn_out_perm_shape = [batch, seq_len, n_heads, head_dim]
    attn_out_perm = tensor.empty(attn_out_perm_shape, elty)

    d0, d1, d2, d3 = [ir.AffineDimExpr.get(i) for i in range(4)]
    from_map = affine_map(4, [d0, d1, d2, d3])
    to_map = affine_map(4, [d0, d2, d1, d3])

    @linalg.generic(
        [attn_out],
        [attn_out_perm],
        [from_map, to_map],
        [parallel] * 4,
    )
    def transpose_out(val, _out):
        return val

    attn_out_transposed = transpose_out

    # Reshape to (batch, seq_len, n_heads * head_dim)
    attn_out_flat_shape = [batch, seq_len, n_heads * head_dim]
    attn_out_flat_type = ir.RankedTensorType.get(attn_out_flat_shape, elty)
    attn_out_flat = tensor.collapse_shape(
        attn_out_flat_type,
        attn_out_transposed,
        reassociation=[[0], [1], [2, 3]],
    )

    # Output projection
    bo_zeros = tensor.empty([dim], elty)
    bo = linalg.fill(zero, outs=[bo_zeros])
    output_final = get_linear(attn_out_flat, wo, bo, out)

    return output_final


def get_transformer_block(
    args: ModelArgs,
    x: ir.Value,
    wq: ir.Value,
    wk: ir.Value,
    wv: ir.Value,
    wo: ir.Value,
    freqs_cis: ir.Value,
    attn_mask: ir.Value,
    w1: ir.Value,
    b1: ir.Value,
    w2: ir.Value,
    b2: ir.Value,
    w3: ir.Value,
    b3: ir.Value,
    out: ir.Value,
) -> ir.Value:
    elty = x.type.element_type

    x_norm_uninit = tensor.empty(x.type.shape, elty)
    x_norm = get_l2_norm(x, x_norm_uninit, eps=args.norm_eps)

    attn_out_uninit = tensor.empty(x.type.shape, elty)
    attn_out = get_attention(
        args,
        x_norm,
        wq,
        wk,
        wv,
        wo,
        freqs_cis,
        attn_mask,
        attn_out_uninit,
    )

    h = get_add(x, attn_out, attn_out_uninit)

    h_norm_uninit = tensor.empty(h.type.shape, elty)
    ffn_norm = get_l2_norm(h, h_norm_uninit, eps=args.norm_eps)

    hidden_dim = int(2 * (4 * x.type.shape[-1]) / 3)
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = args.multiple_of * (
        (hidden_dim + args.multiple_of - 1) // args.multiple_of
    )

    ffn_intermediate_shape = list(x.type.shape)
    ffn_intermediate_shape[-1] = hidden_dim

    ffn_w1_uninit = tensor.empty(ffn_intermediate_shape, elty)
    ffn_w1_out = get_linear(ffn_norm, w1, b1, ffn_w1_uninit)
    silu_out = get_silu(ffn_w1_out, ffn_w1_uninit)

    ffn_w3_uninit = tensor.empty(ffn_intermediate_shape, elty)
    ffn_w3_out = get_linear(ffn_norm, w3, b3, ffn_w3_uninit)
    gated = get_mul(silu_out, ffn_w3_out, ffn_w3_uninit)

    ffn_output = get_linear(gated, w2, b2, out)
    final_out = get_add(h, ffn_output, out)

    return final_out


def get_transformer(
    args: ModelArgs,
    x: ir.Value,
    freqs_cis: ir.Value,
    mask: ir.Value,
    layer_weights: list,
    out: ir.Value,
) -> ir.Value:
    elty = x.type.element_type
    h = x

    # Apply each transformer block sequentially
    for layer_id in range(args.n_layers):
        weights = layer_weights[layer_id]

        # Create output tensor for this layer
        layer_out_uninit = tensor.empty(h.type.shape, elty)

        # Apply transformer block
        h = get_transformer_block(
            args,
            h,
            weights["wq"],
            weights["wk"],
            weights["wv"],
            weights["wo"],
            freqs_cis,
            mask,
            weights["w1"],
            weights["b1"],
            weights["w2"],
            weights["b2"],
            weights["w3"],
            weights["b3"],
            layer_out_uninit,
        )

    # Apply final norm
    final_norm_uninit = tensor.empty(h.type.shape, elty)
    final_norm = get_l2_norm(h, final_norm_uninit, eps=args.norm_eps)

    # Copy to output
    rank = len(h.type.shape)
    dims = [ir.AffineDimExpr.get(i) for i in range(rank)]
    id_map = affine_map(rank, dims)

    @linalg.generic(
        [final_norm],
        [out],
        [id_map, id_map],
        [parallel] * rank,
    )
    def copy_op(val, _out):
        return val

    return copy_op


#### Test cases #####


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
    get_triu: torch.triu,
    get_outer: torch.outer,
    get_linear: torch.nn.functional.linear,
    get_repeat_kv: repeat_kv,
    get_l2_norm: lambda x, eps: x
    * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps),
    get_rotary_emb: apply_rotary_emb,
}


# TODO: torch_dtype_to_mlir_type
def to_ir_type(type_str):
    if type_str == "f32":
        return ir.F32Type.get()
    elif type_str == "f64":
        return ir.F64Type.get()
    else:
        raise ValueError(f"Unsupported type: {type_str}")


@with_mlir_ctx_and_location
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
    def generate_module(elty):
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

    ir_type = to_ir_type(elem_type)
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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
        (get_triu, (4, 4), "f32"),
    ],
)
@with_mlir_ctx_and_location
def test_unary_op(op, shape, elem_type):
    def generate_module(elty):
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

    ir_type = to_ir_type(elem_type)
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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
@with_mlir_ctx_and_location
def test_rms_norm(shape, elem_type):
    eps = 1e-5

    def generate_module(elty):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            input_type = ir.RankedTensorType.get(shape, elty)

            @func.FuncOp.from_py_func(input_type, input_type, name="rms_norm")
            def rms_norm(a, out):
                get_l2_norm(a, out, eps)

        return module

    ir_type = to_ir_type(elem_type)
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


@pytest.mark.parametrize(
    "shape,in_features,out_features",
    [
        ((4,), 16, 32),
        ((1,), 8, 16),
        ((8,), 32, 64),
        ((2,), 64, 32),
        ((2, 4), 32, 32),
        ((3, 5, 7), 16, 24),
    ],
)
@with_mlir_ctx_and_location
def test_linear(shape, in_features, out_features):
    def generate_module(elty, input_shape, in_feat, out_feat):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            input_type = ir.RankedTensorType.get(list(input_shape) + [in_feat], elty)
            weight_type = ir.RankedTensorType.get((out_feat, in_feat), elty)
            bias_type = ir.RankedTensorType.get((out_feat,), elty)
            output_type = ir.RankedTensorType.get(list(input_shape) + [out_feat], elty)

            @func.FuncOp.from_py_func(
                input_type, weight_type, bias_type, output_type, name="linear_op"
            )
            def linear_op(x, w, b, out):
                get_linear(x, w, b, out)

        return module

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type, shape, in_features, out_features)
    schedule = create_schedule()
    apply_schedule(module, schedule)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("linear_op")
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(*shape, in_features, dtype=torch_dtype)
    w = torch.randn(out_features, in_features, dtype=torch_dtype)
    b = torch.randn(out_features, dtype=torch_dtype)
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


@with_mlir_ctx_and_location
def test_polar():
    def generate_module(elty):
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

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


@with_mlir_ctx_and_location
def test_repeat_kv():
    def generate_module(elty, n_rep):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            x_type = ir.RankedTensorType.get((2, 512, 8, 64), elty)
            out_type = ir.RankedTensorType.get((2, 512, 8 * n_rep, 64), elty)

            @func.FuncOp.from_py_func(x_type, out_type, name="repeat_kv_op")
            def repeat_kv_op(x, out):
                get_repeat_kv(x, n_rep, out)

        return module

    n_rep = 4
    ir_type = to_ir_type("f32")
    module = generate_module(ir_type, n_rep)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


@with_mlir_ctx_and_location
def test_reshape_for_broadcast():
    def generate_module(elty):
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

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


@with_mlir_ctx_and_location
def test_view_as_complex():
    def generate_module(elty):
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

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("view_as_complex_op")

    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    x = torch.randn(2, 512, 32, 128, dtype=torch_dtype)
    x_reshaped = x.reshape(2, 512, 32, 64, 2)
    out_ref = torch.view_as_complex(x_reshaped)
    out = torch.empty_like(out_ref)

    x_mem = get_ranked_memref_descriptor(x_reshaped.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([x_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@with_mlir_ctx_and_location
def test_view_as_real():
    def generate_module(elty):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            x_type = ir.RankedTensorType.get((2, 512, 32, 64), ir.ComplexType.get(elty))
            out_type = ir.RankedTensorType.get((2, 512, 32, 64, 2), elty)

            @func.FuncOp.from_py_func(x_type, out_type, name="as_real_op")
            def as_real_op(x, out):
                get_view_as_real(x, out)

        return module

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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
@with_mlir_ctx_and_location
def test_rotary_emb(batch_size, seq_len, n_heads, head_dim, n_kv_heads, elem_type):
    def generate_module(elty, xq_shape, xk_shape, freqs_cis_shape):
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

    ir_type = to_ir_type(elem_type)
    torch_dtype = lh_utils.mlir_type_to_torch_dtype(ir_type)
    xq_shape = (batch_size, seq_len, n_heads, head_dim)
    xk_shape = (batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_shape = (seq_len, head_dim // 2)
    xq = torch.randn(*xq_shape, dtype=torch_dtype)
    xk = torch.randn(*xk_shape, dtype=torch_dtype)
    freqs_cis = torch.randn(*freqs_cis_shape, dtype=torch_dtype)
    xq_out, xk_out = references[get_rotary_emb](xq, xk, freqs_cis)

    module = generate_module(
        xq_shape=xq_shape,
        xk_shape=xk_shape,
        freqs_cis_shape=freqs_cis_shape,
        elty=ir_type,
    )
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


@with_mlir_ctx_and_location
def test_feed_forward():
    def generate_module(elty):
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
                hidden_uninit = tensor.empty(hidden_type.shape, elty)
                hidden = get_linear(x, w1, b1, hidden_uninit)

                # Compute hidden_silu = silu(hidden)
                hidden_silu_uninit = tensor.empty(hidden_type.shape, elty)
                hidden_silu = get_silu(hidden, hidden_silu_uninit)

                # Compute gate = linear(x, w3, b3)
                gate_uninit = tensor.empty(hidden_type.shape, elty)
                gate = get_linear(x, w3, b3, gate_uninit)

                # Compute activated = hidden_silu * gate
                activated_uninit = tensor.empty(hidden_type.shape, elty)
                activated = get_mul(hidden_silu, gate, activated_uninit)

                # Compute out = linear(activated, w2, b2)
                get_linear(activated, w2, b2, out)

        return module

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type)
    schedule = create_schedule()
    apply_schedule(module, schedule)

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


def test_smoke_standalone_attention():
    args = ModelArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=1000,
        multiple_of=8,
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=16,
    )

    attention = Attention(args)

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, args.dim)
    start_pos = 0

    freqs_cis = torch.randn(
        seq_len, args.dim // args.n_heads // 2, dtype=torch.complex64
    )

    mask = torch.full((batch_size, args.n_heads, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    output = attention(x, start_pos, freqs_cis, mask)

    assert output.shape == (
        batch_size,
        seq_len,
        args.dim,
    ), f"Expected shape {(batch_size, seq_len, args.dim)}, got {output.shape}"

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains inf"


@with_mlir_ctx_and_location
def test_attention_fwd():
    model_args = ModelArgs(
        dim=32,  # Small for testing
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,  # Test GQA
        vocab_size=1000,
        multiple_of=8,
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=8,
    )

    batch = 2
    seq_len = 4
    dim = model_args.dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    head_dim = dim // n_heads

    def generate_module(elty, args):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            x_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)
            wq_type = ir.RankedTensorType.get([n_heads * head_dim, dim], elty)
            wk_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wv_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wo_type = ir.RankedTensorType.get([dim, n_heads * head_dim], elty)
            freqs_cis_type = ir.RankedTensorType.get([seq_len, head_dim // 2], elty)
            mask_type = ir.RankedTensorType.get(
                [batch, n_heads, seq_len, seq_len], elty
            )
            out_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)

            @func.FuncOp.from_py_func(
                x_type,
                wq_type,
                wk_type,
                wv_type,
                wo_type,
                freqs_cis_type,
                mask_type,
                out_type,
                name="attention_op",
            )
            def attention_op(x, wq, wk, wv, wo, freqs_cis, mask, out):
                get_attention(args, x, wq, wk, wv, wo, freqs_cis, mask, out)

        return module

    reference = Attention(model_args)

    torch_dtype = torch.float32
    x = torch.randn(batch, seq_len, dim, dtype=torch_dtype)
    freqs_cis_real = torch.randn(seq_len, head_dim // 2, dtype=torch_dtype)
    freqs_cis_complex = torch.complex(freqs_cis_real, torch.zeros_like(freqs_cis_real))
    mask = torch.full(
        (batch, n_heads, seq_len, seq_len), float("-inf"), dtype=torch_dtype
    )
    mask = torch.triu(mask, diagonal=1)
    with torch.no_grad():
        wq = reference.wq.weight.data.clone()
        wk = reference.wk.weight.data.clone()
        wv = reference.wv.weight.data.clone()
        wo = reference.wo.weight.data.clone()

        # Run reference forward
        out_ref = reference(x, start_pos=0, freqs_cis=freqs_cis_complex, mask=mask)

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type, model_args)
    schedule = create_schedule()
    apply_schedule(module, schedule)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("attention_op")

    out = torch.empty_like(out_ref)
    x_mem = get_ranked_memref_descriptor(x.numpy())
    wq_mem = get_ranked_memref_descriptor(wq.numpy())
    wk_mem = get_ranked_memref_descriptor(wk.numpy())
    wv_mem = get_ranked_memref_descriptor(wv.numpy())
    wo_mem = get_ranked_memref_descriptor(wo.numpy())
    freqs_cis_mem = get_ranked_memref_descriptor(freqs_cis_real.numpy())
    mask_mem = get_ranked_memref_descriptor(mask.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args(
        [x_mem, wq_mem, wk_mem, wv_mem, wo_mem, freqs_cis_mem, mask_mem, out_mem]
    )
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@with_mlir_ctx_and_location
def test_transformer_block_fwd():
    model_args = ModelArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,  # Test GQA
        vocab_size=1000,
        multiple_of=8,
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=8,
    )

    batch = 2
    seq_len = 4
    dim = model_args.dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    head_dim = dim // n_heads

    hidden_dim = int(2 * (4 * dim) / 3)
    if model_args.ffn_dim_multiplier is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = model_args.multiple_of * (
        (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
    )

    def generate_module(elty, args):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            x_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)
            wq_type = ir.RankedTensorType.get([n_heads * head_dim, dim], elty)
            wk_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wv_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wo_type = ir.RankedTensorType.get([dim, n_heads * head_dim], elty)
            freqs_cis_type = ir.RankedTensorType.get([seq_len, head_dim // 2], elty)
            mask_type = ir.RankedTensorType.get(
                [batch, n_heads, seq_len, seq_len], elty
            )
            w1_type = ir.RankedTensorType.get([hidden_dim, dim], elty)
            b1_type = ir.RankedTensorType.get([hidden_dim], elty)
            w2_type = ir.RankedTensorType.get([dim, hidden_dim], elty)
            b2_type = ir.RankedTensorType.get([dim], elty)
            w3_type = ir.RankedTensorType.get([hidden_dim, dim], elty)
            b3_type = ir.RankedTensorType.get([hidden_dim], elty)
            out_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)

            @func.FuncOp.from_py_func(
                x_type,
                wq_type,
                wk_type,
                wv_type,
                wo_type,
                freqs_cis_type,
                mask_type,
                w1_type,
                b1_type,
                w2_type,
                b2_type,
                w3_type,
                b3_type,
                out_type,
                name="transformer_block_op",
            )
            def transformer_block_op(
                x, wq, wk, wv, wo, freqs_cis, mask, w1, b1, w2, b2, w3, b3, out
            ):
                get_transformer_block(
                    args,
                    x,
                    wq,
                    wk,
                    wv,
                    wo,
                    freqs_cis,
                    mask,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    out,
                )

        return module

    reference = TransformerBlock(layer_id=0, args=model_args)

    torch_dtype = torch.float32
    x = torch.randn(batch, seq_len, dim, dtype=torch_dtype)
    freqs_cis_real = torch.randn(seq_len, head_dim // 2, dtype=torch_dtype)
    freqs_cis_complex = torch.complex(freqs_cis_real, torch.zeros_like(freqs_cis_real))
    mask = torch.full(
        (batch, n_heads, seq_len, seq_len), float("-inf"), dtype=torch_dtype
    )
    mask = torch.triu(mask, diagonal=1)

    with torch.no_grad():
        # Extract weights from reference model
        wq = reference.attention.wq.weight.data.clone()
        wk = reference.attention.wk.weight.data.clone()
        wv = reference.attention.wv.weight.data.clone()
        wo = reference.attention.wo.weight.data.clone()
        w1 = reference.feed_forward.w1.weight.data.clone()
        w2 = reference.feed_forward.w2.weight.data.clone()
        w3 = reference.feed_forward.w3.weight.data.clone()
        # No bias
        b1 = torch.zeros(hidden_dim, dtype=torch_dtype)
        b2 = torch.zeros(dim, dtype=torch_dtype)
        b3 = torch.zeros(hidden_dim, dtype=torch_dtype)

        out_ref = reference(x, start_pos=0, freqs_cis=freqs_cis_complex, mask=mask)

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type, model_args)
    schedule = create_schedule()
    apply_schedule(module, schedule)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("transformer_block_op")

    out = torch.empty_like(out_ref)
    x_mem = get_ranked_memref_descriptor(x.numpy())
    wq_mem = get_ranked_memref_descriptor(wq.numpy())
    wk_mem = get_ranked_memref_descriptor(wk.numpy())
    wv_mem = get_ranked_memref_descriptor(wv.numpy())
    wo_mem = get_ranked_memref_descriptor(wo.numpy())
    freqs_cis_mem = get_ranked_memref_descriptor(freqs_cis_real.numpy())
    mask_mem = get_ranked_memref_descriptor(mask.numpy())
    w1_mem = get_ranked_memref_descriptor(w1.numpy())
    b1_mem = get_ranked_memref_descriptor(b1.numpy())
    w2_mem = get_ranked_memref_descriptor(w2.numpy())
    b2_mem = get_ranked_memref_descriptor(b2.numpy())
    w3_mem = get_ranked_memref_descriptor(w3.numpy())
    b3_mem = get_ranked_memref_descriptor(b3.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())

    args = lh_utils.memrefs_to_packed_args(
        [
            x_mem,
            wq_mem,
            wk_mem,
            wv_mem,
            wo_mem,
            freqs_cis_mem,
            mask_mem,
            w1_mem,
            b1_mem,
            w2_mem,
            b2_mem,
            w3_mem,
            b3_mem,
            out_mem,
        ]
    )
    func_ptr(args)
    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)


@with_mlir_ctx_and_location
def test_transformer_fwd():
    model_args = ModelArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # Test GQA
        vocab_size=1000,
        multiple_of=8,
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=8,
    )

    batch = 2
    seq_len = 4
    dim = model_args.dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    head_dim = dim // n_heads

    hidden_dim = int(2 * (4 * dim) / 3)
    if model_args.ffn_dim_multiplier is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = model_args.multiple_of * (
        (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
    )

    def generate_module(elty, args):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            x_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)
            freqs_cis_type = ir.RankedTensorType.get([seq_len, head_dim // 2], elty)
            mask_type = ir.RankedTensorType.get(
                [batch, n_heads, seq_len, seq_len], elty
            )
            out_type = ir.RankedTensorType.get([batch, seq_len, dim], elty)
            wq_type = ir.RankedTensorType.get([n_heads * head_dim, dim], elty)
            wk_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wv_type = ir.RankedTensorType.get([n_kv_heads * head_dim, dim], elty)
            wo_type = ir.RankedTensorType.get([dim, n_heads * head_dim], elty)
            w1_type = ir.RankedTensorType.get([hidden_dim, dim], elty)
            b1_type = ir.RankedTensorType.get([hidden_dim], elty)
            w2_type = ir.RankedTensorType.get([dim, hidden_dim], elty)
            b2_type = ir.RankedTensorType.get([dim], elty)
            w3_type = ir.RankedTensorType.get([hidden_dim, dim], elty)
            b3_type = ir.RankedTensorType.get([hidden_dim], elty)

            param_types = [x_type, freqs_cis_type, mask_type]
            for _ in range(args.n_layers):
                param_types.extend(
                    [
                        wq_type,
                        wk_type,
                        wv_type,
                        wo_type,
                        w1_type,
                        b1_type,
                        w2_type,
                        b2_type,
                        w3_type,
                        b3_type,
                    ]
                )
            param_types.append(out_type)

            @func.FuncOp.from_py_func(*param_types, name="transformer_op")
            def transformer_op(*params):
                x = params[0]
                freqs_cis = params[1]
                mask = params[2]

                # Extract weights for each layer
                layer_weights = []
                idx = 3
                for _ in range(args.n_layers):
                    layer_weights.append(
                        {
                            "wq": params[idx],
                            "wk": params[idx + 1],
                            "wv": params[idx + 2],
                            "wo": params[idx + 3],
                            "w1": params[idx + 4],
                            "b1": params[idx + 5],
                            "w2": params[idx + 6],
                            "b2": params[idx + 7],
                            "w3": params[idx + 8],
                            "b3": params[idx + 9],
                        }
                    )
                    idx += 10
                out = params[idx]

                get_transformer(args, x, freqs_cis, mask, layer_weights, out)

        return module

    ir_type = to_ir_type("f32")
    module = generate_module(ir_type, model_args)
    schedule = create_schedule()
    apply_schedule(module, schedule)
    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("transformer_op")

    reference = Transformer(model_args)

    torch_dtype = torch.float32
    x = torch.randn(batch, seq_len, dim, dtype=torch_dtype)
    freqs_cis_real = torch.randn(seq_len, head_dim // 2, dtype=torch_dtype)
    freqs_cis_complex = torch.complex(freqs_cis_real, torch.zeros_like(freqs_cis_real))
    mask = torch.full(
        (batch, n_heads, seq_len, seq_len), float("-inf"), dtype=torch_dtype
    )
    mask = torch.triu(mask, diagonal=1)

    with torch.no_grad():
        # Extract weights from all layers
        layer_weights_torch = []
        for layer_id in range(model_args.n_layers):
            layer = reference.layers[layer_id]
            layer_weights_torch.append(
                {
                    "wq": layer.attention.wq.weight.data.clone(),
                    "wk": layer.attention.wk.weight.data.clone(),
                    "wv": layer.attention.wv.weight.data.clone(),
                    "wo": layer.attention.wo.weight.data.clone(),
                    "w1": layer.feed_forward.w1.weight.data.clone(),
                    "b1": torch.zeros(hidden_dim, dtype=torch_dtype),
                    "w2": layer.feed_forward.w2.weight.data.clone(),
                    "b2": torch.zeros(dim, dtype=torch_dtype),
                    "w3": layer.feed_forward.w3.weight.data.clone(),
                    "b3": torch.zeros(hidden_dim, dtype=torch_dtype),
                }
            )

        # Supply embeddings instead of tokens
        h = x
        for layer in reference.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis_complex, mask=mask)
        out_ref = reference.norm(h)

    out = torch.empty_like(out_ref)
    x_mem = get_ranked_memref_descriptor(x.numpy())
    freqs_cis_mem = get_ranked_memref_descriptor(freqs_cis_real.numpy())
    mask_mem = get_ranked_memref_descriptor(mask.numpy())

    # Add memrefs for all layer weights
    memrefs = [x_mem, freqs_cis_mem, mask_mem]
    for layer_weights in layer_weights_torch:
        memrefs.extend(
            [
                get_ranked_memref_descriptor(layer_weights["wq"].numpy()),
                get_ranked_memref_descriptor(layer_weights["wk"].numpy()),
                get_ranked_memref_descriptor(layer_weights["wv"].numpy()),
                get_ranked_memref_descriptor(layer_weights["wo"].numpy()),
                get_ranked_memref_descriptor(layer_weights["w1"].numpy()),
                get_ranked_memref_descriptor(layer_weights["b1"].numpy()),
                get_ranked_memref_descriptor(layer_weights["w2"].numpy()),
                get_ranked_memref_descriptor(layer_weights["b2"].numpy()),
                get_ranked_memref_descriptor(layer_weights["w3"].numpy()),
                get_ranked_memref_descriptor(layer_weights["b3"].numpy()),
            ]
        )

    out_mem = get_ranked_memref_descriptor(out.numpy())
    memrefs.append(out_mem)

    args = lh_utils.memrefs_to_packed_args(memrefs)
    func_ptr(args)

    assert torch.allclose(out, out_ref, rtol=0.01, atol=0.01, equal_nan=True)
