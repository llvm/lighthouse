import os

import lit.formats
from lit.TestingConfig import TestingConfig

# Imagine that, all your variables defined and with types!
assert isinstance(config := eval("config"), TestingConfig)

config.name = "Lighthouse test suite"
config.test_format = lit.formats.ShTest(True)
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__) + "/lit.out"

config.substitutions.append(("%PYTHON", "uv run"))
if filecheck_path := os.environ.get("FILECHECK"):
    config.substitutions.append(("FileCheck", filecheck_path))
