from collections import namedtuple

from .xegpu_specs import XeGPUSpecs

# hardware constraints
DPAS = namedtuple("DPAS", ["M", "N", "K", "A_TILE", "B_TILE", "C_TILE"])(
    8, 16, 16, (8, 16), (16, 16), (8, 16)
)
PREFETCH_INST_DATA = [8, 16]
NB_WORKITEMS = 16  # workitems in subgroup
LOAD_MAX_ROWS = 32
LOAD_MAX_COLS = 32
PFETCH_MIN_ROWS = 8
PFETCH_MAX_ROWS = 32
PFETCH_MIN_COLS = 16
PFETCH_MAX_COLS = 32
TRANSPOSE_LOAD = [16, 16]
# heuristics: skip likely suboptimal configurations
MIN_NB_THREADS = 16


def print_header(title: str, char: str = "=", width: int = 80):
    header = f" {title} "
    print(f"{header:{char}^{width}}")


def check_wg_tile(M: int, N: int, wg_tile: tuple[int, int]) -> tuple[int, int]:
    if M % wg_tile[0] != 0:
        raise ValueError("wg_tile_m does not divide M")
    if N % wg_tile[1] != 0:
        raise ValueError("wg_tile_n does not divide N")
    wg_grid = (M // wg_tile[0], N // wg_tile[1])
    return wg_grid


def check_sg_tile(
    wg_tile: tuple[int, int],
    sg_tile: tuple[int, int],
    gpu_specs: XeGPUSpecs,
    min_nb_threads: int | None = None,
) -> tuple[int, int]:
    if wg_tile[0] % sg_tile[0] != 0:
        raise ValueError("sg_tile_m does not divide wg_tile_m")
    if wg_tile[1] % sg_tile[1] != 0:
        raise ValueError("sg_tile_n does not divide wg_tile_n")
    if sg_tile[0] % DPAS.M != 0:
        raise ValueError("sg_tile_m not multiple of dpas_m")
    if sg_tile[1] % DPAS.N != 0:
        raise ValueError("sg_tile_n not multiple of dpas_n")
    nb_sg_threads_m = wg_tile[0] // sg_tile[0]
    nb_sg_threads_n = wg_tile[1] // sg_tile[1]
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > gpu_specs.max_nb_threads:
        raise ValueError("too many sg threads")
    if min_nb_threads is not None and nb_sg_threads < min_nb_threads:
        raise ValueError("too few sg threads")
    return nb_sg_threads_m, nb_sg_threads_n


def check_k_tile(K: int, k_tile: int):
    if K % k_tile != 0:
        raise ValueError("k_tile does not divide K")
    if k_tile % DPAS.K != 0:
        raise ValueError("k_tile not multiple of dpas_k")


def check_load_tile(
    tile: tuple[int, int],
    parent_shape: tuple[int, int],
    child_shape: tuple[int, int],
    name: str = "A",
    transpose: bool = False,
):
    if parent_shape[0] % tile[0] != 0 or parent_shape[1] % tile[1] != 0:
        raise ValueError(
            f"Load tile {name} {tile} does not divide the parent shape {parent_shape}."
        )
    if tile[0] % child_shape[0] != 0 or tile[1] % child_shape[1] != 0:
        raise ValueError(
            f"Load tile {name} {tile} does not divide the child shape {child_shape}."
        )
    if tile[0] < child_shape[0]:
        raise ValueError(f"Load tile {name} {tile} has too few rows.")
    if tile[1] < child_shape[1]:
        raise ValueError(f"Load tile {name} {tile} has too few cols.")
    if tile[0] > LOAD_MAX_ROWS:
        raise ValueError(f"Load tile {name} {tile} has too many rows.")
    if tile[1] > LOAD_MAX_COLS:
        raise ValueError(f"Load tile {name} {tile} has too many cols.")
    if transpose and (tile[0] != TRANSPOSE_LOAD[1] or tile[1] != TRANSPOSE_LOAD[0]):
        raise ValueError(
            f"If {name} is transposed, load tile must be {TRANSPOSE_LOAD}."
        )


def check_load_tile_a(
    tile: tuple[int, int],
    sg_tile: tuple[int, int],
    k_tile: int,
    transpose: bool = False,
):
    data_shape = (sg_tile[0], k_tile)
    child_shape = DPAS.A_TILE
    check_load_tile(tile, data_shape, child_shape, name="A", transpose=transpose)


def check_load_tile_b(
    tile: tuple[int, int],
    sg_tile: tuple[int, int],
    k_tile: int,
    transpose: bool = False,
):
    data_shape = (k_tile, sg_tile[1])
    child_shape = DPAS.B_TILE
    check_load_tile(tile, data_shape, child_shape, name="B", transpose=transpose)


def check_prefetch_tile(
    tile: tuple[int, int],
    data_shape: tuple[int, int],
    gpu_specs: XeGPUSpecs,
    name: str = "A",
    transpose: bool = False,
    min_nb_threads: int | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    if transpose:
        data_shape = data_shape[::-1]
    if tile[0] < PFETCH_MIN_ROWS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too few rows (min {PFETCH_MIN_ROWS})."
        )
    if tile[0] > PFETCH_MAX_ROWS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too many rows (max {PFETCH_MAX_ROWS})."
        )
    if tile[1] < PFETCH_MIN_COLS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too few cols (min {PFETCH_MIN_COLS})."
        )
    if tile[1] > PFETCH_MAX_COLS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too many cols (max {PFETCH_MAX_COLS})."
        )
    if data_shape[0] % tile[0] != 0 or data_shape[1] % tile[1] != 0:
        raise ValueError(
            f"Prefetch tile {name} {tile} does not divide the parent shape {data_shape}."
        )
    rows = int(data_shape[0] / tile[0])
    cols = int(data_shape[1] / tile[1])
    nb_threads = int(rows * cols)
    if verbose:
        print_header(f"Prefetch {name}", char="-", width=50)
        print(f"data shape: {data_shape}, transpose: {transpose}")
        print(f"tile size {tile}, grid size ({rows}, {cols}), {nb_threads} threads")
    if nb_threads > gpu_specs.max_nb_threads:
        raise ValueError(
            f"Number of threads for {name} prefetch ({nb_threads}) exceeds max threads ({gpu_specs.max_nb_threads})."
        )
    if min_nb_threads is not None and nb_threads < min_nb_threads:
        raise ValueError(
            f"Number of threads for {name} prefetch ({nb_threads}) is less than minimum threads ({min_nb_threads})."
        )
    return rows, cols


def check_prefetch_tile_a(
    tile: tuple[int, int],
    wg_tile: tuple[int, int],
    k_tile: int,
    gpu_specs: XeGPUSpecs,
    transpose: bool = False,
    min_nb_threads: int | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    data_shape = (wg_tile[0], k_tile)
    return check_prefetch_tile(
        tile,
        data_shape,
        gpu_specs,
        name="A",
        transpose=transpose,
        min_nb_threads=min_nb_threads,
        verbose=verbose,
    )


def check_prefetch_tile_b(
    tile: tuple[int, int],
    wg_tile: tuple[int, int],
    k_tile: int,
    gpu_specs: XeGPUSpecs,
    transpose: bool = False,
    min_nb_threads: int | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    data_shape = (k_tile, wg_tile[1])
    return check_prefetch_tile(
        tile,
        data_shape,
        gpu_specs,
        name="B",
        transpose=transpose,
        min_nb_threads=min_nb_threads,
        verbose=verbose,
    )


def check_constraints(
    params: dict[str, int],
    gpu_specs: XeGPUSpecs,
    verbose: bool = False,
) -> bool:
    """Check that the given tile size configuration is valid."""

    M = params["m"]
    N = params["n"]
    K = params["k"]
    wg_tile_m = params["wg_m"]
    wg_tile_n = params["wg_n"]
    sg_tile_m = params["sg_m"]
    sg_tile_n = params["sg_n"]
    load_tile_a_m = params["load_a_m"]
    load_tile_a_k = params["load_a_k"]
    load_tile_b_k = params["load_b_k"]
    load_tile_b_n = params["load_b_n"]
    prefetch_tile_a_m = params["prefetch_a_m"]
    prefetch_tile_a_k = params["prefetch_a_k"]
    prefetch_tile_b_k = params["prefetch_b_k"]
    prefetch_tile_b_n = params["prefetch_b_n"]
    k_tile = params["k_tile"]
    transpose_a = params.get("transpose_a", False)
    transpose_b = params.get("transpose_b", False)

    wg_tile = (wg_tile_m, wg_tile_n)
    sg_tile = (sg_tile_m, sg_tile_n)
    load_tile_a = (load_tile_a_m, load_tile_a_k)
    load_tile_b = (load_tile_b_k, load_tile_b_n)
    prefetch_tile_a = (prefetch_tile_a_m, prefetch_tile_a_k)
    prefetch_tile_b = (prefetch_tile_b_k, prefetch_tile_b_n)

    try:
        check_wg_tile(M, N, wg_tile)
        check_sg_tile(wg_tile, sg_tile, gpu_specs, min_nb_threads=MIN_NB_THREADS)
        check_k_tile(K, k_tile)
        check_load_tile_a(load_tile_a, sg_tile, k_tile, transpose=transpose_a)
        check_load_tile_b(load_tile_b, sg_tile, k_tile, transpose=transpose_b)
        check_prefetch_tile_a(
            prefetch_tile_a,
            wg_tile,
            k_tile,
            gpu_specs,
            transpose=transpose_a,
            min_nb_threads=MIN_NB_THREADS,
            verbose=verbose,
        )
        check_prefetch_tile_b(
            prefetch_tile_b,
            wg_tile,
            k_tile,
            gpu_specs,
            transpose=transpose_b,
            min_nb_threads=MIN_NB_THREADS,
            verbose=verbose,
        )
    except ValueError as e:
        if verbose:
            print(f"Invalid configuration: {e}")
        return False
    return True
