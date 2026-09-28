# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.schedule import lower_vector_multi_reductions


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = lower_vector_multi_reductions()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# Inner (row) reduction: one horizontal reduction per row, no transpose.
# CHECK-LABEL: Test: row
# CHECK-NOT: vector.transpose
# CHECK-COUNT-2: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<64xf32> into f32
# CHECK-NOT: vector.multi_reduction
run(
    "row",
    """
func.func @row(%a: vector<2x64xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.multi_reduction <add>, %a, %acc [1] : vector<2x64xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
""",
)


# Outer (column) reduction: lane-wise adds, no transpose.
# CHECK-LABEL: Test: column
# CHECK-NOT: vector.transpose
# CHECK-COUNT-2: arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
# CHECK-NOT: vector.multi_reduction
run(
    "column",
    """
func.func @column(%a: vector<2x32xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
  %0 = vector.multi_reduction <add>, %a, %acc [0] : vector<2x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}
""",
)


# 1-D reduction lowers to a single horizontal reduction.
# CHECK-LABEL: Test: rank1
# CHECK: vector.reduction <maximumf>, %{{.*}}, %{{.*}} : vector<128xf32> into f32
# CHECK-NOT: vector.multi_reduction
run(
    "rank1",
    """
func.func @rank1(%a: vector<128xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <maximumf>, %a, %acc [0] : vector<128xf32> to f32
  return %0 : f32
}
""",
)


# A reduced middle dim is not handled by either strategy directly and falls
# back to reordering it outward.
# CHECK-LABEL: Test: middle
# CHECK: vector.transpose
# CHECK-NOT: vector.multi_reduction
# CHECK: return
run(
    "middle",
    """
func.func @middle(%a: vector<2x4x16xf32>, %acc: vector<2x16xf32>) -> vector<2x16xf32> {
  %0 = vector.multi_reduction <add>, %a, %acc [1] : vector<2x4x16xf32> to vector<2x16xf32>
  return %0 : vector<2x16xf32>
}
""",
)
