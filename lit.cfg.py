import os
import importlib.util
import shutil

import lit.formats
from lit.TestingConfig import TestingConfig

# Imagine that, all your variables defined and with type information!
assert isinstance(config := eval("config"), TestingConfig)


def find_filecheck() -> str:
    """Find the full path of the newest FileCheck in the system."""
    # If FileCheck is available in path, use it.
    path = shutil.which("FileCheck")
    if path:
        return path
    # If environment variable is set, use it.
    if filecheck_path := os.environ.get("FILECHECK"):
        if os.path.isfile(filecheck_path) and os.access(filecheck_path, os.X_OK):
            return filecheck_path
    # Otherwise, search for FileCheck in the system and return the newest one.
    version = 21  # Latest version
    while version > 0:
        path = shutil.which("FileCheck-{}".format(version))
        if path:
            return path
        version -= 1
    # If not found, raise an error.
    raise FileNotFoundError(
        "FileCheck not found in the system. Please install LLVM to get FileCheck or \
         set the FILECHECK environment variable to point to the FileCheck executable."
    )


project_root = os.path.dirname(__file__)

config.name = "Lighthouse test suite"
config.test_format = lit.formats.ShTest(True)
config.test_source_root = project_root
config.test_exec_root = project_root + "/lit.out"

config.substitutions.append(("%CACHE", project_root + "/cache"))
config.substitutions.append(("%VIRTUAL_ENV", os.environ.get("VIRTUAL_ENV", "")))
python = os.environ.get("PYTHON", "python")
config.substitutions.append(("%PYTHON", python))
config.substitutions.append(("FileCheck", find_filecheck()))

for pkg in ["torch", "mpi4py", "mpich", "impi-rt"]:
    if importlib.util.find_spec(pkg):
        config.available_features.add(pkg)

torch_kernels_dir = project_root + "/third_party/KernelBench/KernelBench"
if os.path.isdir(torch_kernels_dir):
    config.available_features.add("kernel_bench")
