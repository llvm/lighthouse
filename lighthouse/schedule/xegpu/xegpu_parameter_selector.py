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

    def get_parameters(
        self,
        shape: tuple[int, int, int],
        transpose_a: bool = False,
        transpose_b: bool = False,
        **kwargs,
    ) -> dict:
        m, n, k = shape
        if shape not in self.matmul_param_db or transpose_a or transpose_b:
            try:
                # Use cost model to generate tile sizes and take first config
                configs = generate_configs(
                    m,
                    n,
                    k,
                    self.gpu_specs,
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    max_nb_configs=1,
                    verbose=False,
                )
                if not configs:
                    raise ValueError(
                        f"Cost model did not return any valid configurations for matmul {shape}."
                    )
                params = configs[0][1]
                return params
            except Exception as e:
                msg = f"Error generating parameters for shape {shape} using cost model: {e}"
                raise ValueError(msg) from e
        params = self.matmul_param_db[shape]
        # ensure transpose flags are set
        params.setdefault("transpose_a", False)
        params.setdefault("transpose_b", False)
        return params

    def get_parameters_for_layers(self, param_list: list[dict]) -> list:
        return [self.get_parameters(**params) for params in param_list]
