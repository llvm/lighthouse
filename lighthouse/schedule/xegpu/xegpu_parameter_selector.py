"""
Utility to choose matmul tile size parameters for XeGPU targets.
"""

import json
from pathlib import Path
from .matmul_costmodel import generate_configs
from .xegpu_specs import XeGPUSpecs
from ..parameters import ScheduleParameters

DEFAULT_JSON_FILE = str(Path(__file__).parent / "matmul_params.json")


def load_param_database(json_file: str = DEFAULT_JSON_FILE) -> dict:
    matmul_param_db = {}
    with open(json_file) as f:
        data = json.load(f)
        for entry in data:
            M = entry["m"]
            N = entry["n"]
            K = entry["k"]
            matmul_param_db[(M, N, K)] = entry
    return matmul_param_db


def get_heuristic_params(
    shape,
    transpose_a,
    transpose_b,
    gpu_specs,
    fixed_wg_tile,
    fixed_sg_tile,
    fixed_k_tile,
) -> dict:
    try:
        # Use cost model to generate tile sizes and take first config
        m, n, k = shape
        configs = generate_configs(
            m,
            n,
            k,
            gpu_specs,
            fixed_wg_tile=fixed_wg_tile,
            fixed_sg_tile=fixed_sg_tile,
            fixed_k_tile=fixed_k_tile,
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


class XeGPUParameterSelector:
    def __init__(self, device: str | None = None, json_file: str | None = None):
        if json_file is None:
            json_file = DEFAULT_JSON_FILE
        self.device = device if device is not None else "B70"
        self.gpu_specs = XeGPUSpecs.get(self.device)
        self.matmul_param_db = load_param_database(json_file)

    def get_parameters_dict(
        self,
        shape: tuple[int, int, int],
        transpose_a: bool = False,
        transpose_b: bool = False,
        **kwargs,
    ) -> dict:
        fixed_wg_tile = kwargs.get("wg_tile")
        fixed_sg_tile = kwargs.get("sg_tile")
        fixed_k_tile = kwargs.get("k_tile")
        # TODO add transposed gemms in the database
        if shape not in self.matmul_param_db or transpose_a or transpose_b:
            # not found in database or transposed, use heuristic
            return get_heuristic_params(
                shape,
                transpose_a,
                transpose_b,
                self.gpu_specs,
                fixed_wg_tile,
                fixed_sg_tile,
                fixed_k_tile,
            )
        params = self.matmul_param_db[shape]
        wg_tile = (params["wg_m"], params["wg_n"])
        if (fixed_wg_tile is not None and wg_tile != fixed_wg_tile) or (
            fixed_k_tile is not None and params["k_tile"] != fixed_k_tile
        ):
            # database entry does not match fixed tile sizes, use heuristic
            return get_heuristic_params(
                shape,
                transpose_a,
                transpose_b,
                self.gpu_specs,
                fixed_wg_tile,
                fixed_sg_tile,
                fixed_k_tile,
            )
        # ensure transpose flags are set
        params.setdefault("transpose_a", False)
        params.setdefault("transpose_b", False)
        return params

    def get_parameters(
        self,
        shape: tuple[int, int, int],
        transpose_a: bool = False,
        transpose_b: bool = False,
        **kwargs,
    ) -> ScheduleParameters:
        params_dict = self.get_parameters_dict(
            shape, transpose_a=transpose_a, transpose_b=transpose_b, **kwargs
        )
        return ScheduleParameters([params_dict])

    def get_parameters_for_layers(self, param_list: list[dict]) -> ScheduleParameters:
        return ScheduleParameters(
            [self.get_parameters_dict(**params) for params in param_list]
        )
