"""Query the XeGPU parameter selector for a matmul-like anchor op."""

from mlir import ir

from .matmul_analysis import analyze_matmul_op


def select_xegpu_gemm_params(
    op: ir.OpView,
    wg_tile: tuple[int, ...] | None = None,
    k_tile: int | None = None,
    sg_tile: tuple[int, ...] | None = None,
    device: str | None = None,
) -> dict:
    """
    Infer the GEMM shape from an anchor op and query the parameter selector.
    Args:
        op: The anchor operation representing the matmul-like computation.
        wg_tile: The workgroup tile size applied to the operation, if known.
        k_tile: The K tile size applied to the operation, if known.
        sg_tile: The subgroup tile size applied to the operation, if known.
        device: The target device for which to query the parameters.

    Returns:
        A dictionary containing the selected XeGPU GEMM parameters.
    """
    from lighthouse.schedule.xegpu.xegpu_parameter_selector import (
        XeGPUParameterSelector,
    )

    shape, transpose_a, transpose_b = analyze_matmul_op(op)
    param_selector = XeGPUParameterSelector(device=device)
    params_list = param_selector.get_parameters(
        shape, transpose_a, transpose_b, wg_tile=wg_tile, k_tile=k_tile, sg_tile=sg_tile
    )
    if len(params_list) == 0:
        msg = f"No XeGPU parameters found for shape {shape}"
        if wg_tile is not None:
            msg += f" with wg_tile {wg_tile}"
        if k_tile is not None:
            msg += f" and k_tile {k_tile}"
        raise ValueError(msg)
    return params_list[0]
