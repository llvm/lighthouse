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


def get_matmul_parameters(workload: Workload) -> dict:
    parameters = {}
    for i, shape in enumerate(workload.matmul_layers):
        if shape not in matmul_param_db:
            raise ValueError(
                f"Parameter selector: No parameters found for matmul shape {shape}"
            )
        parameters[f"layer_{i}"] = matmul_param_db[shape]
    return parameters
