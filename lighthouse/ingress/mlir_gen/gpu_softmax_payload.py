"""Generate MLIR payload for GPU softmax operation."""

from mlir import ir
from mlir.dialects import linalg, bufferization, tensor

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor


def generate_gpu_softmax_payload(
    func_name: str,
    M: int,
    N: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for softmax payload.

    Computes softmax along the last dimension (rows):
    output[i, j] = exp(input[i, j] - max_i) / sum_i(exp(input[i, j] - max_i))

    where max_i and sum_i are computed over row i.

    Args:
        func_name: Name of the payload function
        M: Number of rows
        N: Number of columns
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the softmax payload function
    """
    mod = ir.Module.create()
    shape = (M, N)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, input)
        @func_cif(memref_t, memref_t, name=func_name)
        def payload(output, input_arg):
            # Convert memrefs to tensors
            emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input_arg, restrict=True)

            # Create output tensor and fill with zeros
            output_init = tensor.empty(shape, dtype)

            # Apply softmax along dimension 1 (last dimension)
            result = linalg.softmax(
                result=[ir.RankedTensorType.get(shape, dtype)],
                input=input_tensor,
                output=output_init,
                dimension=1,
            )

            # Materialize result back to output memref
            bufferization.materialize_in_destination(
                None, result, output, restrict=True, writable=True
            )

        # Emit utility functions for GPU memory management
        emit_gpu_util_funcs(dtype, rank=2)

    return mod
