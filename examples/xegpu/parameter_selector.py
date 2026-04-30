"""
Utility to choose matmul tile size parameters for XeGPU targets.
"""

import json
from pathlib import Path
from lighthouse.schedule.xegpu.mlp_schedule import DPAS

DEFAULT_JSON_FILE = str(Path(__file__).parent / "matmul_params.json")

DEFAULT_PARAMS = {
    "wg_m": 128,
    "wg_n": 128,
    "sg_m": 32,
    "sg_n": 32,
    "k_tile": 32,
    "load_a_m": DPAS.A_TILE[0],
    "load_a_k": DPAS.A_TILE[1],
    "load_b_k": DPAS.B_TILE[0],
    "load_b_n": DPAS.B_TILE[1],
    "prefetch_a_m": 16,
    "prefetch_a_k": 16,
    "prefetch_b_k": 16,
    "prefetch_b_n": 32,
    "prefetch_a_nb": 1,
    "prefetch_b_nb": 1,
}


def load_param_database(json_file: str = DEFAULT_JSON_FILE) -> dict:
    matmul_param_db = {}
    with open(json_file, "r") as f:
        data = json.load(f)
        for entry in data:
            M = entry["m"]
            N = entry["n"]
            K = entry["k"]
            matmul_param_db[(M, N, K)] = entry
    return matmul_param_db


matmul_param_db = load_param_database()


def get_matmul_parameters(m: int, n: int, k: int) -> list:
    shape = (m, n, k)
    if shape not in matmul_param_db:
        if m >= 128 and n >= 256 and k >= 64:
            params = DEFAULT_PARAMS.copy()
            params["m"] = m
            params["n"] = n
            params["k"] = k
            return params
        else:
            raise ValueError(
                f"Parameter selector: No parameters found for matmul shape {shape}"
            )
    return matmul_param_db[shape]


def get_parameters_for_layers(shapes: list[tuple[int, int, int]]) -> list:
    return [get_matmul_parameters(*shape) for shape in shapes]
