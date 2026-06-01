"""
Utilities for matrix multiplication tile size selection and performance
estimation for XeGPU targets.
"""

from itertools import product
from typing import Callable

from .xegpu_specs import XeGPUSpecs
from .matmul_constraints import (
    check_constraints,
    check_wg_tile,
    check_sg_tile,
    check_k_tile,
    check_prefetch_tile_a,
    check_prefetch_tile_b,
)
from .matmul_constraints import (
    DPAS,
    PFETCH_MIN_ROWS,
    PFETCH_MAX_ROWS,
    PFETCH_MIN_COLS,
    PFETCH_MAX_COLS,
)


def generate_configs(
    M: int,
    N: int,
    K: int,
    gpu_specs: XeGPUSpecs,
    perf_threshold: float | None = None,
    pf_strategy: str = "first",
    max_nb_configs: int | None = None,
) -> list[tuple[float, dict[str, int]]]:
    """Generate valid tile size configurations for (M, N, K) matrix multiplication.

    gpu_specs: XeGPUSpecs object containing the target GPU specifications.

    perf_threshold: if set, only return configurations with
    estimated_perf >= perf_threshold * max_found_estimated_perf.

    pf_strategy: sets the prefetch tile selection strategy
    - "first": take the first prefetch tile for A and B
    - "all": append all valid prefetch tiles for A and B

    Load tile sizes are currently fixed to DPAS tile sizes for A and B.

    Returns:
    A list of (perf_estimate, params_dict) tuples sorted by perf_estimate (descending).
    """
    # TODO add data types as variables

    def tuple_to_param_dict(
        M: int,
        N: int,
        K: int,
        config: tuple[
            tuple[int, int],
            tuple[int, int],
            int,
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ],
    ) -> dict[str, int]:
        wg_tile, sg_tile, k_tile, ld_a, ld_b, pf_a, pf_b = config
        return {
            "m": M,
            "n": N,
            "k": K,
            "wg_m": wg_tile[0],
            "wg_n": wg_tile[1],
            "sg_m": sg_tile[0],
            "sg_n": sg_tile[1],
            "k_tile": k_tile,
            "load_a_m": ld_a[0],
            "load_a_k": ld_a[1],
            "load_b_k": ld_b[0],
            "load_b_n": ld_b[1],
            "prefetch_a_m": pf_a[0],
            "prefetch_a_k": pf_a[1],
            "prefetch_b_k": pf_b[0],
            "prefetch_b_n": pf_b[1],
            "prefetch_a_nb": 1,
            "prefetch_b_nb": 1,
        }

    # define search space
    wg_options = [64, 128, 256]
    sg_options = [32, 64, 128]
    k_tile_options = [16, 32, 64]

    wg_tiles = product(wg_options, wg_options)
    sg_tiles = product(sg_options, sg_options)

    # grid search
    valid_configs = []
    for config in product(wg_tiles, sg_tiles, k_tile_options):
        wg_tile, sg_tile, k_tile = config
        try:
            perf = estimate_performance(
                M, N, K, wg_tile, sg_tile, k_tile, gpu_specs, verbose=False
            )
            n_prefetch = 1 if pf_strategy == "first" else None
            pf_a_list, pf_b_list = generate_prefetch_tiles(
                wg_tile, k_tile, gpu_specs, n=n_prefetch
            )
            load_a_list = [DPAS.A_TILE]
            load_b_list = [DPAS.B_TILE]
            for la, lb, pa, pb in product(
                load_a_list, load_b_list, pf_a_list, pf_b_list
            ):
                c = (wg_tile, sg_tile, k_tile, la, lb, pa, pb)
                params = tuple_to_param_dict(M, N, K, c)
                if check_constraints(params, gpu_specs, verbose=False):
                    valid_configs.append((perf, params))
        except ValueError:
            pass

    # sort by performance (descending)
    valid_configs.sort(key=lambda x: x[0], reverse=True)

    if perf_threshold is not None and len(valid_configs) > 0:
        assert 0 < perf_threshold <= 1, "perf_threshold must be in (0, 1]"
        max_perf = valid_configs[0][0]
        valid_configs = [c for c in valid_configs if c[0] >= perf_threshold * max_perf]

    if max_nb_configs is not None:
        valid_configs = valid_configs[:max_nb_configs]

    return valid_configs


def generate_prefetch_tiles(
    wg_tile: tuple[int, int],
    k_tile: int,
    gpu_specs: XeGPUSpecs,
    n: int | None = None,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Generates valid prefetch tile sizes for A and B.

    Candidates are sorted by number of threads (descending) and then by how
    balanced the thread grid is (descending).
    """

    def gridsearch(
        check_fn: Callable[
            [tuple[int, int], tuple[int, int], int, XeGPUSpecs],
            tuple[int, int],
        ],
    ) -> list[tuple[int, int]]:
        tiles = []
        for rows in range(PFETCH_MIN_ROWS, PFETCH_MAX_ROWS + 1):
            for cols in range(PFETCH_MIN_COLS, PFETCH_MAX_COLS + 1):
                tile = (rows, cols)
                try:
                    grid = check_fn(tile, wg_tile, k_tile, gpu_specs)
                    nb_threads = int(grid[0] * grid[1])
                    tiles.append((tile, nb_threads, grid))
                except ValueError:
                    pass
        # sort by number of threads and then by how balanced the thread grid is
        tiles.sort(key=lambda x: (x[1], -abs(x[2][0] - x[2][1])), reverse=True)
        tiles = [t[0] for t in tiles]
        return tiles

    prefetch_tiles_a = gridsearch(check_prefetch_tile_a)
    prefetch_tiles_b = gridsearch(check_prefetch_tile_b)
    if n is not None:
        prefetch_tiles_a = prefetch_tiles_a[:n]
        prefetch_tiles_b = prefetch_tiles_b[:n]

    return prefetch_tiles_a, prefetch_tiles_b


def estimate_performance(
    M: int,
    N: int,
    K: int,
    wg_tile: tuple[int, int],
    sg_tile: tuple[int, int],
    k_tile: int,
    gpu_specs: XeGPUSpecs,
    prefetch_tile_a: tuple[int, int] | None = None,
    prefetch_tile_b: tuple[int, int] | None = None,
    verbose: bool = True,
) -> float:
    """
    Estimate the performance of the given tile size configuration for (M,N,K)
    matrix multiplication on the target GPU.

    The performance estimate is based on a simple roofline model using the
    workgroup and K tile sizes and the GPU's peak FLOPS and memory bandwidth.

    If `verbose` is True, prints a summary of the configuration.

    Returns the estimated performance in FLOPS.

    Raises ValueError if the given configuration is invalid.
    """
    if verbose:
        print("=== Global Level ===")
        print(f"Matrix sizes: M={M}, N={N}, K={K}")

    # TODO generalize
    ab_dtype_size = 2  # bytes for f16
    c_dtype_size = 4  # bytes for f32

    # WG
    if verbose:
        print("=== Workgroup Level ===")
    roofline_threshold = gpu_specs.peak_flops / gpu_specs.bw_global_mem  # in FLOPs/Byte

    wg_grid = check_wg_tile(M, N, wg_tile)
    check_k_tile(K, k_tile)
    nb_wgs = wg_grid[0] * wg_grid[1]
    if verbose:
        print(f"Workgroup tile size: {wg_tile}, grid size: {wg_grid}, nb WGs: {nb_wgs}")
        print(f"K tile size: {k_tile}")

    A_wg_shape = (wg_tile[0], k_tile)
    B_wg_shape = (k_tile, wg_tile[1])
    C_wg_shape = (wg_tile[0], wg_tile[1])

    A_footprint = A_wg_shape[0] * A_wg_shape[1] * ab_dtype_size
    B_footprint = B_wg_shape[0] * B_wg_shape[1] * ab_dtype_size
    C_footprint = C_wg_shape[0] * C_wg_shape[1] * c_dtype_size

    if verbose:
        print(f"A: shape={A_wg_shape}, footprint={A_footprint / 1024:.2f} KB")
        print(f"B: shape={B_wg_shape}, footprint={B_footprint / 1024:.2f} KB")
        print(f"C: shape={C_wg_shape}, footprint={C_footprint / 1024:.2f} KB")

    total_footprint = A_footprint + B_footprint
    if verbose:
        print(f"Total SLM footprint: {total_footprint / 1024:.1f} KB")
    # TODO check that A,B,C fit in shared local memory

    # arithmetic intensity
    f = (wg_tile[0] * wg_tile[1]) / (wg_tile[0] + wg_tile[1])
    ai = f * ab_dtype_size
    if verbose:
        print(f"Arithmetic intensity: {ai:.2f} FLOPs/Byte")
        print(f"Roofline threshold:   {roofline_threshold:.2f} FLOPs/Byte")

    if verbose:
        if ai < roofline_threshold:
            print(" => Bandwidth-bound regime")
        else:
            print(" => Compute-bound regime")

    xe_core_utilization = min(nb_wgs / gpu_specs.nb_xe_cores, 1.0)
    if verbose:
        print(f"XE core utilization: {xe_core_utilization:.2f}")

    # predict flops
    peak_flops = (
        gpu_specs.peak_flops * xe_core_utilization
    )  # possible under-utilization
    predicted_throughput = min(peak_flops, ai * gpu_specs.bw_global_mem)
    if verbose:
        print(f"Predicted throughput: {predicted_throughput / 1e12:.2f} TFLOPS")

    # SG
    if verbose:
        print("=== Subgroup Level ===")

    sg_grid = check_sg_tile(wg_tile, sg_tile, gpu_specs)
    nb_sgs = sg_grid[0] * sg_grid[1]
    if verbose:
        print(
            f"Subgroup tile size: {sg_tile}, grid size: {sg_grid}, nb SGs per WG: {nb_sgs}"
        )

    A_sg_shape = (sg_tile[0], k_tile)
    B_sg_shape = (k_tile, sg_tile[1])
    C_sg_shape = (sg_tile[0], sg_tile[1])

    A_footprint = A_sg_shape[0] * A_sg_shape[1] * ab_dtype_size
    B_footprint = B_sg_shape[0] * B_sg_shape[1] * ab_dtype_size
    C_footprint = C_sg_shape[0] * C_sg_shape[1] * c_dtype_size

    total_footprint = A_footprint + B_footprint + C_footprint
    if verbose:
        print(f"A: shape={A_sg_shape}, footprint={A_footprint / 1024:.2f} KB")
        print(f"B: shape={B_sg_shape}, footprint={B_footprint / 1024:.2f} KB")
        print(f"C: shape={C_sg_shape}, footprint={C_footprint / 1024:.2f} KB")
        print(f"Total register footprint: {total_footprint / 1024:.2f} KB")

    nb_parallel_dpas = (sg_tile[0] // DPAS.M) * (sg_tile[1] // DPAS.N)
    if verbose:
        print(f"Number of DPAS threads: {nb_parallel_dpas}")
    nb_dpas_ops = nb_parallel_dpas * (k_tile // DPAS.K)
    if verbose:
        print(f"Number of total DPAS ops: {nb_dpas_ops}")

    # FIXME move remaining checks to util funcs
    if nb_parallel_dpas > gpu_specs.dpas_exec_size:
        raise ValueError(
            f"Number of parallel DPAS ops ({nb_parallel_dpas}) exceeds hardware execution size ({gpu_specs.dpas_exec_size})."
        )

    # estimate number of used registers
    reg_size = 64  # bytes per register
    nb_reg = int((A_footprint + B_footprint + C_footprint) / reg_size)
    if verbose:
        print(f"Number of registers: {nb_reg}")

    if nb_reg > gpu_specs.nb_registers:
        raise ValueError(
            f"Number of registers ({nb_reg}) exceeds hardware register file size ({gpu_specs.nb_registers})."
        )

    if prefetch_tile_a:
        # check that prefetch tile is suitable for WG-k tile
        check_prefetch_tile_a(
            prefetch_tile_a, wg_tile, k_tile, gpu_specs, verbose=verbose
        )

    if prefetch_tile_b:
        # check that prefetch tile is suitable for WG-k tile
        check_prefetch_tile_b(
            prefetch_tile_b, wg_tile, k_tile, gpu_specs, verbose=verbose
        )

    return predicted_throughput
