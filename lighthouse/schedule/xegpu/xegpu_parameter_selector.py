"""
Utility to choose matmul tile size parameters for XeGPU targets.
"""

import json
from pathlib import Path
from .matmul_costmodel import generate_configs
from .xegpu_specs import XeGPUSpecs

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


class XeGPUParameterSelector:
    def __init__(self, device: str | None = None, json_file: str | None = None):
        if json_file is None:
            json_file = DEFAULT_JSON_FILE
        self.device = device if device is not None else "B70"
        self.gpu_specs = XeGPUSpecs.get(self.device)
        self.matmul_param_db = load_param_database(json_file)

    def get_parameters(self, m: int, n: int, k: int) -> dict:
        shape = (m, n, k)
        if shape not in self.matmul_param_db:
            try:
                # Use cost model to generate tile sizes and take first config
                configs = generate_configs(m, n, k, self.gpu_specs, max_nb_configs=1)
                if not configs:
                    raise ValueError(
                        f"Cost model did not return any valid configurations for matmul {shape}."
                    )
                params = configs[0][1]
                return params
            except Exception as e:
                msg = f"Error generating parameters for shape {shape} using cost model: {e}"
                raise ValueError(msg) from e
        return self.matmul_param_db[shape]

    def get_parameters_for_layers(self, shapes: list[tuple[int, int, int]]) -> list:
        return [self.get_parameters(*shape) for shape in shapes]
