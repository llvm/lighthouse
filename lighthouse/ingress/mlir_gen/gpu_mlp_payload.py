from mlir import ir
from mlir.dialects import linalg, gpu, bufferization, arith, tensor

from .gpu_utils import emit_buf_to_tensor
from .named import add_bias, relu, times_weights
from .generic import convert_float_type
from lighthouse.utils.mlir import func_cif


def generate_gpu_mlp_payload(
    func_name: str,
    batch_size: int,
    input_size: int,
    output_size: int,
    hidden_layer_sizes: list[int],
    ab_type: ir.Type,
    acc_type: ir.Type,
    bias_type: ir.Type,
    result_type: ir.Type,
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    relu_on_final_layer: bool = False,
) -> ir.Module:
    """Generate payload function module for an MLP kernel."""
    mod = ir.Module.create()
    memref_in_t = ir.MemRefType.get((batch_size, input_size), ab_type)
    memref_out_t = ir.MemRefType.get((batch_size, output_size), result_type)
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    feature_sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    weight_memref_types = []
    bias_memref_types = []
    for in_size, out_size in feature_sizes:
        memref_t = ir.MemRefType.get((in_size, out_size), ab_type)
        weight_memref_types.append(memref_t)
        if has_bias:
            memref_t = ir.MemRefType.get((out_size,), bias_type)
            bias_memref_types.append(memref_t)
    with ir.InsertionPoint(mod.body):
        # function argument order:
        #   output, input, weights_0, weights_1, ..., [bias_0, bias_1, ...]
        fargs = [memref_out_t, memref_in_t] + weight_memref_types
        if has_bias:
            fargs += bias_memref_types

        @func_cif(*fargs, name=func_name)
        def payload(*args):
            output = args[0]
            input = args[1]
            nlayers = len(hidden_layer_sizes) + 1
            weights = args[2 : 2 + nlayers]
            biases = args[2 + nlayers :] if has_bias else [None] * nlayers
            output_tensor = emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input, restrict=True)
            weight_tensors = [emit_buf_to_tensor(w, restrict=True) for w in weights]
            bias_tensors = [
                emit_buf_to_tensor(b, restrict=True) if has_bias else None
                for b in biases
            ]

            layer_input_tensor = input_tensor
            to_dealloc = None
            for i, (weight_tensor, bias_tensor) in enumerate(
                zip(weight_tensors, bias_tensors)
            ):
                M, K = layer_input_tensor.type.shape
                K, N = weight_tensor.type.shape
                if i == nlayers - 1:
                    c_tensor = output_tensor
                    c_memref = output
                else:
                    # allocate intermediate buffer
                    memref_type = ir.MemRefType.get((M, N), ab_type)
                    c_memref = gpu.alloc(memref_type, None, [], [], [])
                    gpu.memset(None, [], c_memref, arith.constant(ab_type, 0.0))
                    if accumulate_c:
                        c_tensor = emit_buf_to_tensor(
                            c_memref, restrict=True, writable=True
                        )
                # skip relu for final layer
                hidden_layer = i < nlayers - 1
                layer_output = emit_mlp_layer(
                    layer_input_tensor,
                    weight_tensor,
                    acc_type=acc_type,
                    result_type=ab_type if hidden_layer else result_type,
                    acc_tensor=c_tensor if accumulate_c else None,
                    bias_tensor=bias_tensor,
                    has_relu=(hidden_layer or relu_on_final_layer) and has_relu,
                )
                bufferization.materialize_in_destination(
                    None, layer_output, c_memref, restrict=True, writable=True
                )
                if to_dealloc is not None:
                    gpu.dealloc(None, [], to_dealloc)
                    to_dealloc = None
                if i != nlayers - 1:
                    # deallocate after next layer
                    to_dealloc = c_memref
                layer_input_tensor = layer_output

    return mod


def emit_mlp_layer(
    a_tensor,
    b_tensor,
    acc_type,
    result_type,
    acc_tensor=None,
    bias_tensor=None,
    has_relu=False,
) -> ir.Value:
    M, K = a_tensor.type.shape
    K, N = b_tensor.type.shape
    convert_result = acc_type != result_type
    if acc_tensor is not None:
        if acc_tensor.type.element_type != acc_type:
            empty = tensor.empty((M, N), acc_type)
            acc_tensor = convert_float_type(acc_tensor, empty)
    else:
        # use zero tensor as the accumulator
        zero = arith.constant(acc_type, 0.0)
        empty = tensor.empty((M, N), acc_type)
        zero_tensor = linalg.fill(zero, outs=[empty])
        acc_tensor = zero_tensor
    terminal = times_weights(a_tensor, b_tensor, acc_tensor)
    if bias_tensor is not None:
        if bias_tensor.type.element_type != acc_type:
            empty = tensor.empty((N,), acc_type)
            bias_tensor = convert_float_type(bias_tensor, empty)
        terminal = add_bias(terminal, bias_tensor)
    if convert_result:
        empty = tensor.empty((M, N), result_type)
        terminal = convert_float_type(terminal, empty)
    if has_relu:
        terminal = relu(terminal)

    return terminal
