import pytest
import torch
from typing import Tuple

from mlir import ir
from mlir.dialects import transform, func, linalg, tensor, arith, complex
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


def get_rotary_emb(
    xq: ir.Value, xk: ir.Value, freqs_cis: ir.Value, xq_out: ir.Value, xk_out: ir.Value
):
    """
    Apply rotary embeddings to query and key tensors.

    This implements the transformation:
    1. View xq, xk as complex: [B, S, H, D] -> [B, S, H, D//2] complex
    2. Broadcast freqs_cis: [S, D//2] -> [1, S, 1, D//2]
    3. Complex multiply: xq_ * freqs_cis, xk_ * freqs_cis
    4. View back as real: [B, S, H, D//2] complex -> [B, S, H, D] real

    Args:
        xq: Query tensor of shape [B, S, H, D]
        xk: Key tensor of shape [B, S, H_kv, D]
        freqs_cis: Rotary embeddings of shape [S, D//2]
                   Note: In PyTorch this is complex64, but here it's f32
                   We need to interpret as pairs or pass cos/sin separately
        xq_out: Output tensor for queries [B, S, H, D]
        xk_out: Output tensor for keys [B, S, H_kv, D]

    TODO: Properly implement rotary embeddings
    Current implementation is just a placeholder passthrough.

    For a correct implementation, we need to:
    1. Either:
       a) Pass freqs_cis as [S, D] with interleaved cos/sin values, OR
       b) Pass separate cos and sin tensors of shape [S, D//2]
    2. Extract pairs of elements from xq/xk to treat as complex (real, imag)
    3. Apply complex rotation:
       (real', imag') = (real*cos - imag*sin, real*sin + imag*cos)
    4. Interleave results back into output
    """
    elty = xq.type.element_type

    # Get shapes
    xq_shape = list(xq.type.shape)  # [B, S, H, D]
    xk_shape = list(xk.type.shape)  # [B, S, H_kv, D]

    # Placeholder implementation: just copy inputs to outputs
    # This allows the test infrastructure to work but doesn't compute correct results

    b, s, h, d = [ir.AffineDimExpr.get(i) for i in range(4)]

    @linalg.generic(
        [xq],
        [xq_out],
        [affine_map(4, [b, s, h, d]), affine_map(4, [b, s, h, d])],
        [parallel] * 4,
    )
    def copy_xq(x, _out):
        return x

    @linalg.generic(
        [xk],
        [xk_out],
        [
            affine_map(4, [b, s, ir.AffineDimExpr.get(2), d]),
            affine_map(4, [b, s, ir.AffineDimExpr.get(2), d]),
        ],
        [parallel] * 4,
    )
    def copy_xk(x, _out):
        return x

    return (copy_xq, copy_xk)


def get_as_complex(x: ir.Value, out: ir.Value) -> ir.Value:
    """
    Interpret the input tensor as complex numbers by grouping pairs of elements.

    Args:
        x: Input tensor of shape [..., 2] representing complex numbers as pairs (real, imag)
        out: Output tensor of shape [...] with complex type
    """
    elty = x.type.element_type
    rank = x.type.rank
    shape = list(x.type.shape)
    assert shape[-1] == 2, "Last dimension must be of size 2 to form complex numbers"
    complex_shape = shape[:-1]

    dim_exprs_in = [ir.AffineDimExpr.get(i) for i in range(rank)]
    dim_exprs_out = [ir.AffineDimExpr.get(i) for i in range(rank - 1)]

    input_map = affine_map(
        rank,
        dim_exprs_in,
    )
    output_map = affine_map(
        rank - 1,
        dim_exprs_out,
    )
    iterator_types = [parallel] * (rank - 1)

    @linalg.generic(
        [x],
        [out],
        [input_map, output_map],
        iterator_types,
    )
    def as_complex_op(a, _out):
        real_part = a[0]
        imag_part = a[1]
        cplx = complex.CreateOp(
            complex.ComplexType.get(elty), real_part, imag_part
        ).result
        return cplx

    return as_complex_op


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
    get_rsqrt: torch.rsqrt,
    get_sqr: torch.square,
    get_mean: lambda x: torch.mean(x, dim=-1, keepdim=True),
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
    "op,shape,elem_type", [(get_add, (4, 16), "f32"), (get_mul, (4, 16), "f32")]
)
def test_bin_op(op, shape, elem_type):
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tensor_type = ir.RankedTensorType.get(shape, elty)

                @func.FuncOp.from_py_func(
                    tensor_type, tensor_type, tensor_type, name="bin_op"
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


def test_to_complex():
    def generate_module(ctx, elty):
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                a_type = ir.RankedTensorType.get((2, 2), elty)
                b_type = ir.RankedTensorType.get((2, 2), elty)
                out_type = ir.RankedTensorType.get((2, 2), elty)

                @func.FuncOp.from_py_func(
                    a_type, b_type, out_type, name="mul_as_complex"
                )
                def mul_as_complex(a, b, out):
                    # Convert both inputs to complex
                    # (d0, d1) -> (d0, d1//2) complex<f32>
                    # multiply with linalg.mul
                    # Convert back to real
                    # (d0, d1//2) complex -> (d0, d1)

                    complex_shape = list(a.type.shape)
                    complex_shape[-1] = complex_shape[-1] // 2
                    a_complex_uninit = ir.RankedTensorType.get(
                        complex_shape, complex.ComplexType.get(elty)
                    )
                    b_complex_uninit = ir.RankedTensorType.get(
                        complex_shape, complex.ComplexType.get(elty)
                    )
                    mul_out = ir.RankedTensorType.get(
                        complex_shape, complex.ComplexType.get(elty)
                    )
                    mul = linalg.mul(
                        a_complex_uninit, b_complex_uninit, outs=(mul_out,)
                    )

        return module

    a = torch.randn(2, 2, dtype=torch.float32)
    b = torch.randn(2, 2, dtype=torch.float32)
    x_complex = torch.view_as_complex(a)
    y_complex = torch.view_as_complex(b)
    res = torch.view_as_real(x_complex * y_complex).flatten(1)

    ctx = ir.Context()
    ir_type = to_ir_type("f32", ctx)
    module = generate_module(ctx, ir_type)
    bufferize_module(ctx, module)
    schedule = create_schedule(ctx)
    apply_schedule(module, schedule)
    pm = create_pass_pipeline(ctx)
    pm.run(module.operation)

    eng = ExecutionEngine(module, opt_level=2)
    func_ptr = eng.lookup("mul_as_complex")
    out = torch.empty_like(a)
    a_mem = get_ranked_memref_descriptor(a.numpy())
    b_mem = get_ranked_memref_descriptor(b.numpy())
    out_mem = get_ranked_memref_descriptor(out.numpy())
    args = lh_utils.memrefs_to_packed_args([a_mem, b_mem, out_mem])
    func_ptr(args)

    assert torch.allclose(out, res, rtol=0.01, atol=0.01, equal_nan=True)
