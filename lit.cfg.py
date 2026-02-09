import os
import importlib.util

import lit.formats
from lit.TestingConfig import TestingConfig

# Imagine that, all your variables defined and with type information!
assert isinstance(config := eval("config"), TestingConfig)

project_root = os.path.dirname(__file__)

config.name = "Lighthouse test suite"
config.test_format = lit.formats.ShTest(True)
config.test_source_root = project_root
config.test_exec_root = project_root + "/lit.out"

config.substitutions.append(("%CACHE", project_root + "/cache"))
python = os.environ.get("PYTHON", "python")
config.substitutions.append(("%PYTHON", python))
if filecheck_path := os.environ.get("FILECHECK"):
    config.substitutions.append(("FileCheck", filecheck_path))

for pkg in ["torch", "mpi4py", "mpich", "openmpi", "impi-rt"]:
    if importlib.util.find_spec(pkg):
        config.available_features.add(pkg)

torch_kernels_dir = project_root + "/third_party/KernelBench/KernelBench"
if os.path.isdir(torch_kernels_dir):
    config.available_features.add("kernel_bench")
