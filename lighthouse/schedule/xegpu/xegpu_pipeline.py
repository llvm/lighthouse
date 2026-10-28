from lighthouse.schedule import xegpu


def get_xegpu_pipeline(
    parameters: dict | None = None,
    stop_at_stage: str | None = None,
    disable_multi_reduction_to_contract: bool = False,
    device: str | None = None,
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
    wg_subtile = parameters.get("wg_subtile", None)
    reduction_tile = parameters.get("reduction_tile", None)
    if wg_tile is None and all(k in parameters for k in ("wg_m", "wg_n")):
        wg_tile = [parameters["wg_m"], parameters["wg_n"]]
    if sg_tile is None and all(k in parameters for k in ("sg_m", "sg_n")):
        sg_tile = [parameters["sg_m"], parameters["sg_n"]]
    load_tile_a = parameters.get("load_tile_a")
    if load_tile_a is None and all(k in parameters for k in ("load_a_m", "load_a_k")):
        load_tile_a = [parameters["load_a_m"], parameters["load_a_k"]]
    load_tile_b = parameters.get("load_tile_b")
    if load_tile_b is None and all(k in parameters for k in ("load_b_k", "load_b_n")):
        load_tile_b = [parameters["load_b_k"], parameters["load_b_n"]]
    prefetch_tile_a = parameters.get("prefetch_tile_a")
    if prefetch_tile_a is None and all(
        k in parameters for k in ("prefetch_a_m", "prefetch_a_k")
    ):
        prefetch_tile_a = [
            parameters["prefetch_a_m"],
            parameters["prefetch_a_k"],
        ]
    prefetch_tile_b = parameters.get("prefetch_tile_b")
    if prefetch_tile_b is None and all(
        k in parameters for k in ("prefetch_b_k", "prefetch_b_n")
    ):
        prefetch_tile_b = [
            parameters["prefetch_b_k"],
            parameters["prefetch_b_n"],
        ]
    k_tile = parameters.get("k_tile")
    prefetch_a_nb = parameters.get("prefetch_a_nb")
    prefetch_b_nb = parameters.get("prefetch_b_nb")
    schedules = []
    if stop_at_stage == "initial":
        return schedules
    schedules.append(xegpu.cleanup_schedule())
    if stop_at_stage == "cleanup":
        return schedules
    schedules.append(
        xegpu.wg_tiling_schedule(
            wg_tile=wg_tile,
            k_tile=k_tile,
            reduction_tile=reduction_tile,
            wg_subtile=wg_subtile,
            device=device,
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
    schedules.append(xegpu.outline_gpu_func_schedule(sg_tile=sg_tile, device=device))
    if stop_at_stage == "gpu-outlining":
        return schedules
    schedules.append(xegpu.vector_to_xegpu_schedule())
    if stop_at_stage == "xegpu-initial":
        return schedules
    schedules.append(
        xegpu.annotate_layouts_schedule(
            sg_tile=sg_tile,
            wg_tile=wg_tile,
            k_tile=k_tile,
            load_tile_a=load_tile_a,
            load_tile_b=load_tile_b,
            prefetch_tile_a=prefetch_tile_a,
            prefetch_tile_b=prefetch_tile_b,
            prefetch_a_nb=prefetch_a_nb,
            prefetch_b_nb=prefetch_b_nb,
            device=device,
        )
    )
    if stop_at_stage == "xegpu-wg":
        return schedules
    schedules.append(xegpu.xegpu_to_binary())

    return schedules
