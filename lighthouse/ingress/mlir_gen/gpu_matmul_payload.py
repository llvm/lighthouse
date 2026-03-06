from mlir import ir
from .gpu_mlp_payload import generate_gpu_mlp_payload


def generate_gpu_matmul_payload(
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
    return generate_gpu_mlp_payload(
        func_name,
        batch_size=M,
        input_size=K,
        output_size=N,
        hidden_layer_sizes=[],
        ab_type_str=ab_type_str,
        c_type_str=c_type_str,
        result_type_str=c_type_str,
        has_bias=has_bias,
        has_relu=has_relu,
        accumulate_c=accumulate_c,
        relu_on_final_layer=True,
    )
