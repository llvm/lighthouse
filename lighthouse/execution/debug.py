import uuid
import sys

from mlir.execution_engine import ExecutionEngine


def dump_mlir_object_file(engine: ExecutionEngine) -> str:
    """
    Dump the compiled MLIR object file.

    Args:
        engine: MLIR execution engine.

    Returns:
        str: The created file name.
    """
    file = "lh-mlir-dump-" + uuid.uuid4().hex + ".o"
    engine.dump_to_object_file(file)
    print(f"MLIR JIT - object file: {file}", file=sys.stderr)
    return file
