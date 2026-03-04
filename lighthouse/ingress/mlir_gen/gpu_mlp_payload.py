from mlir import ir
from mlir.dialects import func, linalg, gpu, bufferization, arith, tensor

from .gpu_utils import emit_gpu_util_funcs, emit_buf_to_tensor


def generate_gpu_mlp_payload(
    func_name: str,
    batch_size: int,
    input_size: int,
    output_size: int,
    hidden_layer_sizes: list[int],
    ab_type_str: str,
    c_type_str: str,
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
) -> ir.Module:
    """Generate payload function module for an MLP kernel."""
    get_ir_dtype = {
        "f16": ir.F16Type.get(),
        "f32": ir.F32Type.get(),
    }
    ab_type = get_ir_dtype[ab_type_str]
    c_type = get_ir_dtype[c_type_str]
    mod = ir.Module.create()
    memref_in_t = ir.MemRefType.get((batch_size, input_size), ab_type)
    memref_out_t = ir.MemRefType.get((batch_size, output_size), ab_type)
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    feature_sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    weight_memref_types = []
    bias_memref_types = []
    for in_size, out_size in feature_sizes:
        memref_t = ir.MemRefType.get((in_size, out_size), ab_type)
        weight_memref_types.append(memref_t)
        if has_bias:
            memref_t = ir.MemRefType.get((out_size,), c_type)
            bias_memref_types.append(memref_t)
    with ir.InsertionPoint(mod.body):
        # function argument order:
        #   input, output, weights_0, weights_1, ..., [bias_0, bias_1, ...]
        fargs = [memref_in_t, memref_out_t] + weight_memref_types
        if has_bias:
            fargs += bias_memref_types

        @func.func(*fargs, name=func_name)
        def payload(*args):
            input = args[0]
            output = args[1]
            nlayers = len(hidden_layer_sizes) + 1
            weights = args[2 : 2 + nlayers]
            biases = args[2 + nlayers :] if has_bias else [None] * nlayers
            input_tensor = emit_buf_to_tensor(input, restrict=True)
            output_tensor = emit_buf_to_tensor(output, restrict=True)
            weight_tensors = []
            for weight_memref in weights:
                weight_tensor = emit_buf_to_tensor(weight_memref, restrict=True)
                weight_tensors.append(weight_tensor)
            bias_tensors = []
            for bias_memref in biases:
                if has_bias:
                    bias_tensor = emit_buf_to_tensor(bias_memref, restrict=True)
                else:
                    bias_tensor = None
                bias_tensors.append(bias_tensor)

            layer_output = input_tensor
            to_dealloc = None
            for i, (weight, bias) in enumerate(zip(weight_tensors, bias_tensors)):
                a_tensor = layer_output
                b_tensor = weight
                M, K = a_tensor.type.shape
                _, N = b_tensor.type.shape
                if i == nlayers - 1:
                    c_tensor = output_tensor
                else:
                    # allocate intermediate buffer
                    memref_type = ir.MemRefType.get((M, N), ab_type)
                    c_memref = gpu.alloc(memref_type, None, [], [], [])
                    gpu.memset(None, [], c_memref, arith.constant(ab_type, 0.0))
                    c_tensor = emit_buf_to_tensor(
                        c_memref, restrict=True, writable=True
                    )
                bias_tensor = bias
                # skip relu for final layer
                emit_relu = has_relu if i < nlayers - 1 else False
                layer_output = emit_mlp_layer(
                    a_tensor,
                    b_tensor,
                    c_tensor,
                    ab_type,
                    c_type,
                    bias_tensor,
                    emit_relu,
                    accumulate_c=accumulate_c,
                    convert_c_type=True,
                )
                if i != nlayers - 1:
                    bufferization.materialize_in_destination(
                        None, layer_output, c_memref, restrict=True, writable=True
                    )
                if to_dealloc is not None:
                    gpu.dealloc(None, [], to_dealloc)
                    to_dealloc = None
                if i != nlayers - 1:
                    # deallocate after next layer
                    to_dealloc = c_memref

            # finalize
            bufferization.materialize_in_destination(
                None, layer_output, output, restrict=True, writable=True
            )

        payload.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        emit_gpu_util_funcs(ab_type)
        if c_type != ab_type:
            emit_gpu_util_funcs(c_type)

    return mod


def emit_mlp_layer(
    a_tensor,
    b_tensor,
    c_tensor,
    ab_type,
    c_type,
    bias_tensor=None,
    has_relu=False,
    accumulate_c=True,
    convert_c_type=False,
) -> ir.Value:
    M, N = c_tensor.type.shape
    id_map = ir.AffineMap.get_identity(2)
    par_iter = linalg.IteratorType.parallel
    if convert_c_type and accumulate_c:
        empty = tensor.empty((M, N), c_type)

        @linalg.generic(
            [c_tensor],
            [empty],
            [id_map, id_map],
            [par_iter, par_iter],
        )
        def f(a, b):
            return arith.extf(c_type, a)

        input_c_tensor = f
    else:
        if accumulate_c:
            input_c_tensor = c_tensor
        else:
            zero = arith.constant(c_type, 0.0)
            empty = tensor.empty((M, N), c_type)
            zero_tensor = linalg.fill(zero, outs=[empty])
            input_c_tensor = zero_tensor
    mmul = linalg.matmul(a_tensor, b_tensor, outs=[input_c_tensor])
    terminal = mmul
    res_type = c_type
    if convert_c_type:
        res_type = ab_type
        empty = tensor.empty((M, N), ab_type)

        @linalg.generic(
            [terminal],
            [empty],
            [id_map, id_map],
            [par_iter, par_iter],
        )
        def f(a, b):
            return arith.truncf(ab_type, a)

        terminal = f
    if bias_tensor is not None:
        empty = tensor.empty((M, N), res_type)
        bcast = linalg.broadcast(bias_tensor, outs=[empty], dimensions=[0])
        terminal = linalg.add(bcast, terminal, outs=[empty])
    if has_relu:
        zero = arith.constant(ab_type if convert_c_type else c_type, 0.0)
        empty = tensor.empty((M, N), res_type)
        zero_tensor = linalg.fill(zero, outs=[empty])
        terminal = linalg.max(terminal, zero_tensor, outs=[empty])

    return terminal
