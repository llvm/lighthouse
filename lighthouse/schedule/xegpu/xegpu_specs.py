"""
XeGPU hardware specifications for tile size selection.
"""

from dataclasses import dataclass

__all__ = ["XeGPUSpecs"]


gpu_bmg_common = {
    "dpas_exec_size": 16,  # number of parallel dpas ops in sg
    # large register file
    "max_nb_threads_lgrf": 32,
    "nb_registers_lgrf": 256,
    # small register file
    "max_nb_threads_sgrf": 64,
    "nb_registers_sgrf": 128,
}

gpu_specs_db = {
    "B70": {
        "name": "Intel Arc B70",
        "nb_xe_cores": 32,
        "peak_flops": 155000e9,  # float16
        "bw_global_mem": 608e9,  # GB/s
        **gpu_bmg_common,
    },
    "B50": {
        "name": "Intel Arc B50",
        "nb_xe_cores": 16,
        "peak_flops": 78000e9,  # float16
        "bw_global_mem": 224e9,  # GB/s
        **gpu_bmg_common,
    },
}


@dataclass
class XeGPUSpecs:
    """XeGPU hardware specification relevant for tile size selection."""

    name: str
    nb_xe_cores: int
    peak_flops: float  # in FLOPS
    bw_global_mem: float  # in bytes/s
    dpas_exec_size: int  # number of parallel dpas ops in subgroup
    # large register file
    max_nb_threads_lgrf: int  # max number of threads per subgroup
    nb_registers_lgrf: int  # number of registers per thread
    # small register file
    max_nb_threads_sgrf: int  # max number of threads per subgroup
    nb_registers_sgrf: int  # number of registers per thread
    reg_file: str  # "large" or "small"

    @property
    def max_nb_threads(self):
        if self.reg_file == "large":
            return self.max_nb_threads_lgrf
        else:
            return self.max_nb_threads_sgrf

    @property
    def nb_registers(self):
        if self.reg_file == "large":
            return self.nb_registers_lgrf
        else:
            return self.nb_registers_sgrf

    @classmethod
    def get(cls, device_name: str, reg_file: str = "large") -> "XeGPUSpecs":
        assert reg_file in ["large", "small"], "reg_file must be 'large' or 'small'"
        if device_name not in gpu_specs_db:
            raise ValueError(
                f"Unknown device name: {device_name}. Available devices: {list(gpu_specs_db.keys())}"
            )
        specs_dict = gpu_specs_db[device_name].copy()
        specs_dict["reg_file"] = reg_file
        return cls(**specs_dict)
