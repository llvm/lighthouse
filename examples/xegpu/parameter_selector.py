"""
Utility to choose matmul tile size parameters.
"""

import json
from pathlib import Path

DEFAULT_JSON_FILE = str(Path(__file__).parent / "matmul_params.json")


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
        raise ValueError(
            f"Parameter selector: No parameters found for matmul shape {shape}"
        )
    return matmul_param_db[shape]


def get_parameters_for_layers(shapes: list[tuple[int, int, int]]) -> list:
    return [get_matmul_parameters(*shape) for shape in shapes]
