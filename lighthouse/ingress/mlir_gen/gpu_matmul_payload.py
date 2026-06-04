from mlir import ir
from .gpu_mlp_payload import generate_gpu_mlp_payload


def generate_gpu_matmul_payload(
    func_name: str,
    M: int,
    N: int,
    K: int,
    ab_type: ir.Type,
    c_type: ir.Type,
    transpose_a: bool,
    transpose_b: bool,
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    result_type: ir.Type | None = None,
) -> ir.Module:
    """Generate payload function module for a matmul kernel."""
    return generate_gpu_mlp_payload(
        func_name,
        batch_size=M,
        input_size=K,
        output_size=N,
        hidden_layer_sizes=[],
        ab_type=ab_type,
        acc_type=c_type,
        bias_type=c_type if result_type is None else result_type,
        result_type=c_type if result_type is None else result_type,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
        has_relu=has_relu,
        accumulate_c=accumulate_c,
        relu_on_final_layer=True,
    )
