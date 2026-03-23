from mlir import ir
from mlir.dialects import gpu, bufferization, arith
from lighthouse.utils.mlir import func_cif


def emit_buf_to_tensor(memref_value: ir.Value, **kwargs) -> ir.Value:
    memref_type = memref_value.type
    shape = memref_type.shape
    element_type = memref_type.element_type
    tensor_type = ir.RankedTensorType.get(shape, element_type)
    return bufferization.to_tensor(tensor_type, memref_value, **kwargs)


def emit_gpu_alloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    index_t = ir.IndexType.get()
    i32_t = ir.IntegerType.get_signless(32)
    inputs = rank * (i32_t,)

    @func_cif(*inputs, name="gpu_alloc_" + suffix)
    def f(*shape):
        dims = [arith.index_cast(index_t, a) for a in shape]
        alloc = gpu.alloc(memref_dyn_t, None, [], dims, [])
        return alloc


def emit_gpu_dealloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func_cif(memref_dyn_t, name="gpu_dealloc_" + suffix)
    def f(memref):
        gpu.dealloc(None, [], memref)


def emit_gpu_copy(suffix: str, element_type: ir.Type, rank: int = 2):
    """Emit GPU copy function."""
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func_cif(memref_dyn_t, memref_dyn_t, name="gpu_copy_" + suffix)
    def f(src, dst):
        gpu.memcpy(None, [], dst, src)


def emit_gpu_util_funcs(element_type: ir.Type, rank: int = 2):
    """Emit GPU utility functions for allocation, deallocation and copy."""
    type_str = str(element_type)
    suffix = f"{rank}d_{type_str}"
    emit_gpu_alloc(suffix, element_type, rank)
    emit_gpu_dealloc(suffix, element_type, rank)
    emit_gpu_copy(suffix, element_type, rank)
