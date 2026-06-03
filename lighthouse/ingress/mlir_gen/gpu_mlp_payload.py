from mlir import ir
from mlir.dialects import linalg, bufferization, arith, tensor

from .utils import emit_buf_to_tensor
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
    transpose_a: bool,
    transpose_b: bool,
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    relu_on_final_layer: bool = False,
) -> ir.Module:
    """Generate payload function module for an MLP kernel."""
    mod = ir.Module.create()
    a_shape = (batch_size, input_size) if not transpose_a else (input_size, batch_size)
    memref_in_t = ir.MemRefType.get(a_shape, ab_type)
    memref_out_t = ir.MemRefType.get((batch_size, output_size), result_type)
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    feature_sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    weight_memref_types = []
    bias_memref_types = []
    for in_size, out_size in feature_sizes:
        shape = (in_size, out_size) if not transpose_b else (out_size, in_size)
        memref_t = ir.MemRefType.get(shape, ab_type)
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
            for i, (weight_tensor, bias_tensor) in enumerate(
                zip(weight_tensors, bias_tensors)
            ):
                layer_transpose_a = (
                    transpose_a and i == 0
                )  # transpose A only for the first layer
                M, K = (
                    layer_input_tensor.type.shape[::-1]
                    if layer_transpose_a
                    else layer_input_tensor.type.shape
                )
                K, N = (
                    weight_tensor.type.shape[::-1]
                    if transpose_b
                    else weight_tensor.type.shape
                )
                c_tensor = None
                if accumulate_c:
                    if i == nlayers - 1:
                        c_tensor = output_tensor
                    else:
                        c_tensor = tensor.empty((M, N), ab_type)
                # skip relu for final layer
                hidden_layer = i < nlayers - 1
                layer_output = emit_mlp_layer(
                    layer_input_tensor,
                    weight_tensor,
                    acc_type=acc_type,
                    result_type=ab_type if hidden_layer else result_type,
                    acc_tensor=c_tensor,
                    bias_tensor=bias_tensor,
                    transpose_a=layer_transpose_a,
                    transpose_b=transpose_b,
                    has_relu=(hidden_layer or relu_on_final_layer) and has_relu,
                )
                if i == nlayers - 1:
                    bufferization.materialize_in_destination(
                        None, layer_output, output, restrict=True, writable=True
                    )
                layer_input_tensor = layer_output

    return mod


def emit_mlp_layer(
    a_tensor: ir.Value,
    b_tensor: ir.Value,
    acc_type: ir.Type,
    result_type: ir.Type,
    acc_tensor: ir.Value | None = None,
    bias_tensor: ir.Value | None = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    has_relu: bool = False,
) -> ir.Value:
    M, K = a_tensor.type.shape[::-1] if transpose_a else a_tensor.type.shape
    K, N = b_tensor.type.shape[::-1] if transpose_b else b_tensor.type.shape
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
    if transpose_a:
        empty = tensor.empty((M, K), a_tensor.type.element_type)
        a_tensor = linalg.transpose(
            a_tensor, outs=(empty,), permutation=[1, 0]
        ).results[0]
    if transpose_b:
        empty = tensor.empty((K, N), b_tensor.type.element_type)
        b_tensor = linalg.transpose(
            b_tensor, outs=(empty,), permutation=[1, 0]
        ).results[0]
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
