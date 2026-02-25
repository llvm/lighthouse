from mlir import ir
from mlir.dialects import func, bufferization

from .gpu_utils import emit_gpu_util_funcs, emit_buf_to_tensor
from .mlp_payload import emit_mlp_layer


def generate_matmul_payload(
    func_name: str,
    M: int,
    N: int,
    K: int,
    ab_type_str: str,
    c_type_str: str,
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
) -> ir.Module:
    """Generate payload function module for a matmul kernel."""
    get_ir_dtype = {
        "f16": ir.F16Type.get(),
        "f32": ir.F32Type.get(),
    }
    ab_type = get_ir_dtype[ab_type_str]
    c_type = get_ir_dtype[c_type_str]
    memref_a_t = ir.MemRefType.get((M, K), ab_type)
    memref_b_t = ir.MemRefType.get((K, N), ab_type)
    memref_c_t = ir.MemRefType.get((M, N), c_type)
    memref_bias_t = ir.MemRefType.get((N,), c_type)
    mod = ir.Module.create()
    with ir.InsertionPoint(mod.body):
        fargs = [memref_a_t, memref_b_t]
        if has_bias:
            fargs.append(memref_bias_t)
        fargs.append(memref_c_t)

        @func.func(*fargs, name=func_name)
        def payload(*args):
            A = args[0]
            B = args[1]
            C = args[-1]
            bias = args[2] if has_bias else None
            a_tensor = emit_buf_to_tensor(A, restrict=True)
            b_tensor = emit_buf_to_tensor(B, restrict=True)
            c_tensor = emit_buf_to_tensor(C, restrict=True, writable=True)
            if has_bias:
                bias_tensor = emit_buf_to_tensor(bias, restrict=True)
            else:
                bias_tensor = None

            output = emit_mlp_layer(
                a_tensor,
                b_tensor,
                c_tensor,
                ab_type,
                c_type,
                bias_tensor,
                has_relu,
                accumulate_c=accumulate_c,
                convert_c_type=False,
            )
            bufferization.materialize_in_destination(
                None, output, C, restrict=True, writable=True
            )

        payload.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        emit_gpu_util_funcs(ab_type)
        if c_type != ab_type:
            emit_gpu_util_funcs(c_type)

    return mod
