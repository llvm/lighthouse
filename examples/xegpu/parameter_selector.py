"""
Utility to choose matmul tile size parameters.
"""

import json
from pathlib import Path
from lighthouse.workload import Workload

DEFAULT_JSON_FILE = str(Path(__file__).parent / "matmul_params.json")


def load_param_database(json_file: str = DEFAULT_JSON_FILE) -> dict:
    matmul_param_db = {}
    with open(json_file, "r") as f:
        data = json.load(f)
        for entry in data:
            M = entry.pop("M")
            N = entry.pop("N")
            K = entry.pop("K")
            matmul_param_db[(M, N, K)] = entry
    return matmul_param_db


matmul_param_db = load_param_database()


def get_matmul_parameters(workload: Workload) -> list:
    parameters = []
    for i, shape in enumerate(workload.matmul_layers):
        if shape not in matmul_param_db:
            raise ValueError(
                f"Parameter selector: No parameters found for matmul shape {shape}"
            )
        parameters.append(matmul_param_db[shape])
    return parameters


matmul_param_db = {
    (4096, 4096, 4096): {
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "wg_m": 256,
        "wg_n": 256,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 64,
        "load_a_m": 32,
        "load_a_k": 16,
        "load_b_k": 32,
        "load_b_n": 16,
        "prefetch_a_m": 8,
        "prefetch_a_k": 32,
        "prefetch_b_k": 8,
        "prefetch_b_n": 32,
        "prefetch_nb": 1,
    },
    (128, 16384, 16384): {
        "m": 128,
        "n": 16384,
        "k": 16384,
        "wg_m": 128,
        "wg_n": 256,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 256,
        "load_a_m": 8,
        "load_a_k": 16,
        "load_b_k": 32,
        "load_b_n": 16,
        "prefetch_a_m": 8,
        "prefetch_a_k": 16,
        "prefetch_b_k": 8,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    },
    (128, 8192, 16384): {
        "m": 128,
        "n": 8192,
        "k": 16384,
        "wg_m": 64,
        "wg_n": 128,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 128,
        "load_a_m": 16,
        "load_a_k": 16,
        "load_b_k": 16,
        "load_b_n": 16,
        "prefetch_a_m": 32,
        "prefetch_a_k": 16,
        "prefetch_b_k": 16,
        "prefetch_b_n": 32,
        "prefetch_nb": 1,
    },
    (128, 32768, 16384): {
        "m": 128,
        "n": 32768,
        "k": 16384,
        "wg_m": 128,
        "wg_n": 128,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 256,
        "load_a_m": 8,
        "load_a_k": 16,
        "load_b_k": 16,
        "load_b_n": 16,
        "prefetch_a_m": 16,
        "prefetch_a_k": 32,
        "prefetch_b_k": 8,
        "prefetch_b_n": 32,
        "prefetch_nb": 1,
    },
    (128, 16384, 32768): {
        "m": 128,
        "n": 16384,
        "k": 32768,
        "wg_m": 128,
        "wg_n": 128,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 256,
        "load_a_m": 8,
        "load_a_k": 16,
        "load_b_k": 16,
        "load_b_n": 16,
        "prefetch_a_m": 32,
        "prefetch_a_k": 32,
        "prefetch_b_k": 8,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    },
    (128, 32768, 32768): {
        "m": 128,
        "n": 32768,
        "k": 32768,
        "wg_m": 128,
        "wg_n": 256,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 256,
        "load_a_m": 8,
        "load_a_k": 16,
        "load_b_k": 16,
        "load_b_n": 16,
        "prefetch_a_m": 16,
        "prefetch_a_k": 32,
        "prefetch_b_k": 32,
        "prefetch_b_n": 32,
        "prefetch_nb": 1,
    },
    (1024, 1024, 8192): {
        "m": 1024,
        "n": 1024,
        "k": 8192,
        "wg_m": 256,
        "wg_n": 128,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 32,
        "load_a_m": 8,
        "load_a_k": 16,
        "load_b_k": 32,
        "load_b_n": 16,
        "prefetch_a_m": 8,
        "prefetch_a_k": 16,
        "prefetch_b_k": 8,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    },
    (1024, 8192, 1024): {
        "m": 1024,
        "n": 8192,
        "k": 1024,
        "wg_m": 256,
        "wg_n": 128,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 32,
        "load_a_m": 16,
        "load_a_k": 16,
        "load_b_k": 32,
        "load_b_n": 16,
        "prefetch_a_m": 8,
        "prefetch_a_k": 16,
        "prefetch_b_k": 16,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    },
    (1024, 1024, 1024): {
        "m": 1024,
        "n": 1024,
        "k": 1024,
        "wg_m": 128,
        "wg_n": 64,
        "sg_m": 32,
        "sg_n": 32,
        "k_tile": 32,
        "load_a_m": 16,
        "load_a_k": 16,
        "load_b_k": 32,
        "load_b_n": 16,
        "prefetch_a_m": 8,
        "prefetch_a_k": 32,
        "prefetch_b_k": 8,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    },
}
