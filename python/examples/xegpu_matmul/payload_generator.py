from mlir import ir
from mlir.dialects import func, linalg, gpu, bufferization, arith, tensor


def emit_gpu_alloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    index_t = ir.IndexType.get()
    i32_t = ir.IntegerType.get_signless(32)
    inputs = rank * (i32_t,)

    @func.func(*inputs, name="gpu_alloc_" + suffix)
    def alloc_func(*shape):
        dims = [arith.index_cast(index_t, a) for a in shape]
        alloc = gpu.alloc(memref_dyn_t, None, [], dims, [])
        return alloc

    alloc_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_gpu_dealloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func.func(memref_dyn_t, name="gpu_dealloc_" + suffix)
    def dealloc_func(memref):
        gpu.dealloc(None, [], memref)

    dealloc_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_gpu_copy(suffix: str, element_type: ir.Type, rank: int = 2):
    """Emit GPU copy function."""
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func.func(memref_dyn_t, memref_dyn_t, name="gpu_copy_" + suffix)
    def copy_func(src, dst):
        gpu.memcpy(None, [], dst, src)

    copy_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_gpu_util_funcs(element_type: ir.Type):
    """Emit GPU utility functions for allocation, deallocation and copy."""
    suffix = {
        ir.F16Type.get(): "f16",
        ir.F32Type.get(): "f32",
    }[element_type]
    emit_gpu_alloc(suffix, element_type)
    emit_gpu_dealloc(suffix, element_type)
    emit_gpu_copy(suffix, element_type)


def generate_matmul_payload(
    func_name: str,
    M: int,
    N: int,
    K: int,
    ab_type_str: str,
    c_type_str: str,
    has_bias: bool,
    has_relu: bool,
) -> ir.Module:
    """Generate payload function module."""
    get_ir_dtype = {
        "f16": ir.F16Type.get(),
        "f32": ir.F32Type.get(),
    }
    ab_type = get_ir_dtype[ab_type_str]
    c_type = get_ir_dtype[c_type_str]
    tensor_a_t = ir.RankedTensorType.get((M, K), ab_type)
    tensor_b_t = ir.RankedTensorType.get((K, N), ab_type)
    tensor_c_t = ir.RankedTensorType.get((M, N), c_type)
    tensor_bias_t = ir.RankedTensorType.get((N,), c_type)
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
            a_tensor = bufferization.to_tensor(tensor_a_t, A, restrict=True)
            b_tensor = bufferization.to_tensor(tensor_b_t, B, restrict=True)
            c_tensor = bufferization.to_tensor(
                tensor_c_t, C, restrict=True, writable=True
            )

            mmul = linalg.matmul(a_tensor, b_tensor, outs=[c_tensor])
            terminal = mmul
            if has_bias:
                bias = args[2]
                bias_tensor = bufferization.to_tensor(
                    tensor_bias_t, bias, restrict=True, writable=True
                )
                empty = tensor.empty((M, N), c_type)
                bcast = linalg.broadcast(bias_tensor, outs=[empty], dimensions=[0])
                terminal = linalg.add(bcast, terminal, outs=[empty])
            if has_relu:
                zero = arith.constant(c_type, 0.0)
                empty = tensor.empty((M, N), c_type)
                zero_tensor = linalg.fill(zero, outs=[empty])
                terminal = linalg.max(terminal, zero_tensor, outs=[empty])

            bufferization.materialize_in_destination(
                None, terminal, C, restrict=True, writable=True
            )

        payload.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        emit_gpu_util_funcs(ab_type)
        if c_type != ab_type:
            emit_gpu_util_funcs(c_type)

    return mod
