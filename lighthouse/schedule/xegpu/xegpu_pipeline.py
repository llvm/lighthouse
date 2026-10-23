from lighthouse.schedule import xegpu


def get_xegpu_pipeline(
    parameters: dict | None = None,
    stop_at_stage: str | None = None,
    disable_multi_reduction_to_contract: bool = False,
) -> list:
    """
    Get the default XeGPU schedule pipeline.

    Args:
        parameters (dict, optional): Dictionary to override inferred tiling parameters. Defaults to None.
        stop_at_stage (str, optional): Stage at which to stop the pipeline. If None, full pipeline is returned.
        disable_multi_reduction_to_contract (bool, optional): Flag to disable vector.multi_reduction to contract conversion. Defaults to False.

    Returns:
        list: List of XeGPU schedules up to the specified stop stage.
    """
    if parameters is None:
        parameters = {}
    wg_tile = parameters.get("wg_tile", None)
    sg_tile = parameters.get("sg_tile", None)
    reduction_tile = parameters.get("reduction_tile", None)
    schedules = []
    if stop_at_stage == "initial":
        return schedules
    schedules.append(xegpu.cleanup_schedule())
    if stop_at_stage == "cleanup":
        return schedules
    schedules.append(
        xegpu.wg_tiling_schedule(
            wg_tile=wg_tile,
            reduction_tile=reduction_tile,
        )
    )
    if stop_at_stage == "tiled":
        return schedules
    schedules.append(
        xegpu.vectorize_schedule(
            disable_multi_reduction_to_contract=disable_multi_reduction_to_contract,
        )
    )
    if stop_at_stage == "vectorized":
        return schedules
    schedules.append(xegpu.bufferize_schedule())
    if stop_at_stage == "bufferized":
        return schedules
    schedules.append(xegpu.outline_gpu_func_schedule(sg_tile=sg_tile))
    if stop_at_stage == "gpu-outlining":
        return schedules
    schedules.append(xegpu.vector_to_xegpu_schedule())
    if stop_at_stage == "xegpu-initial":
        return schedules
    schedules.append(
        xegpu.annotate_layouts_schedule(
            sg_tile=sg_tile,
        )
    )
    if stop_at_stage == "xegpu-wg":
        return schedules
    schedules.append(xegpu.xegpu_to_binary())

    return schedules
