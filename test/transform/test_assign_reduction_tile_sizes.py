# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.tile_and_fuse import assign_reduction_tile_sizes


def run(name: str, payload_str: str, **kwargs):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(arch="x86_64", features=["avx512f"]):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            sched = assign_reduction_tile_sizes(**kwargs)
            sched.body.operations[0].apply(payload.operation)
            print(payload)


# GEMM -> row sum -> division by the row sum.
PAYLOAD = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x32xf32>, %b: tensor<32xCOLSxf32>,
      %c: tensor<64xCOLSxf32>, %s: tensor<64xf32>) -> tensor<64xCOLSxf32> {
    %mm = linalg.matmul ins(%a, %b : tensor<64x32xf32>, tensor<32xCOLSxf32>)
        outs(%c : tensor<64xCOLSxf32>) -> tensor<64xCOLSxf32>
    %sum = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"] SUM_ATTRS}
        ins(%mm : tensor<64xCOLSxf32>) outs(%s : tensor<64xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    } -> tensor<64xf32>
    %div = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm, %sum : tensor<64xCOLSxf32>, tensor<64xf32>)
        outs(%c : tensor<64xCOLSxf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64xCOLSxf32>
    return %div : tensor<64xCOLSxf32>
  }
}
"""


def payload(cols: int = 64, sum_attrs: str = "") -> str:
    return PAYLOAD.replace("COLS", str(cols)).replace("SUM_ATTRS", sum_attrs)


# Only the reduction is annotated: 4 vector chains per 64-wide row, 2 rows.
# CHECK-LABEL: Test: register_parallel
# CHECK: linalg.matmul
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 2, 0>
# CHECK: arith.addf
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run("register_parallel", payload())

# Propagation reaches the elementwise consumer but not the GEMM (a barrier).
# CHECK-LABEL: Test: register_parallel_propagate
# CHECK: linalg.matmul
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: transform_ext.tile_sizes = array<i64: 2, 0>
# CHECK: arith.addf
# CHECK: transform_ext.tile_sizes = array<i64: 2, 0>
# CHECK: arith.divf
run("register_parallel_propagate", payload(), propagate=True)

# Long rows get a reduction tile of lanes x chains.
# CHECK-LABEL: Test: register_reduction
# CHECK: transform_ext.tile_sizes = array<i64: 0, 128>
# CHECK: arith.addf
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run("register_reduction", payload(cols=256), strategy="register_reduction")

# Existing annotations are kept.
# CHECK-LABEL: Test: keep_annotation
# CHECK: transform_ext.tile_sizes = array<i64: 4, 0>
# CHECK: arith.addf
run(
    "keep_annotation",
    payload(sum_attrs=", transform_ext.tile_sizes = array<i64: 4, 0>"),
)
