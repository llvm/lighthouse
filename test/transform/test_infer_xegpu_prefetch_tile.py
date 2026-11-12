# RUN: %PYTHON %s | FileCheck %s

"""Tests for shape-based XeGPU prefetch tile inference."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

PAYLOAD = """
module {
  func.func @main() { return }
}
"""


def run(name, shape, min_threads=0, device=None, expect_failure=False):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        i64 = ir.IntegerType.get_signless(64)
        with schedule_boilerplate() as (schedule, named_seq):
            func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            shape_params = [
                transform.ParamConstantOp(
                    transform.AnyParamType.get(), ir.IntegerAttr.get(i64, dim)
                ).param
                for dim in shape
            ]
            rows, cols = transform_ext.infer_xegpu_prefetch_tile(
                shape_params, min_threads=min_threads, device=device
            )
            transform.annotate(func, "rows", param=rows)
            transform.annotate(func, "cols", param=cols)
            transform.yield_()
        try:
            schedule.body.operations[0].apply(payload.operation)
        except ValueError as error:
            if not expect_failure:
                raise
            print("failed:", error)
            return
        assert not expect_failure, "Expected prefetch tile inference to fail"
        print(payload)


# CHECK-LABEL: Test: attention_1b
# CHECK-DAG: rows = 8 : i64
# CHECK-DAG: cols = 16 : i64
run("attention_1b", [64, 64])

# CHECK-LABEL: Test: attention_3b
# CHECK-DAG: rows = 8 : i64
# CHECK-DAG: cols = 16 : i64
run("attention_3b", [32, 128], device="B70")

# A small tile may use fewer threads when no minimum is requested.
# CHECK-LABEL: Test: small_tile
# CHECK-DAG: rows = 8 : i64
# CHECK-DAG: cols = 16 : i64
run("small_tile", [16, 32])

# CHECK-LABEL: Test: no_candidate
# CHECK: infer_xegpu_prefetch_tile: no valid tiles for shape (16, 32)
run("no_candidate", [16, 32], min_threads=16, expect_failure=True)

# CHECK-LABEL: Test: invalid_shape
# CHECK: infer_xegpu_prefetch_tile: shape must be positive
run("invalid_shape", [0, 64], expect_failure=True)

# CHECK-LABEL: Test: invalid_device
# CHECK: infer_xegpu_prefetch_tile: Unknown device name
run("invalid_device", [64, 64], device="unknown", expect_failure=True)
