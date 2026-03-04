from mlir import ir
from mlir.dialects import func, gpu, bufferization, arith
from .utils import get_elem_type_str


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
    suffix = get_elem_type_str(type(element_type))
    emit_gpu_alloc(suffix, element_type)
    emit_gpu_dealloc(suffix, element_type)
    emit_gpu_copy(suffix, element_type)
