"""Query the XeGPU parameter selector for a matmul-like anchor op."""

from mlir import ir

from .matmul_analysis import analyze_matmul_op


def select_xegpu_gemm_params(op: ir.OpView, device: str | None = None) -> dict:
    """Infer the GEMM shape from an anchor op and query the parameter selector."""
    from lighthouse.schedule.xegpu.xegpu_parameter_selector import (
        XeGPUParameterSelector,
    )

    shape, transpose_a, transpose_b = analyze_matmul_op(op)
    param_selector = XeGPUParameterSelector(device=device)
    params_list = param_selector.get_parameters(shape, transpose_a, transpose_b)
    if len(params_list) == 0:
        raise ValueError(f"No XeGPU parameters found for shape {shape}")
    return params_list[0]
