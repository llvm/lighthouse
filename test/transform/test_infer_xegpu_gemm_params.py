# RUN: %PYTHON %s | FileCheck %s

"""Test for the infer_xegpu_gemm_params transform op on a global GEMM."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

# Global (untiled) GEMM: M=512, N=8192, K=1024, no transpose.
GEMM_PAYLOAD = """
module {
  func.func @payload(%arg0: tensor<512x8192xf32>, %arg1: tensor<512x1024xf16>,
                     %arg2: tensor<1024x8192xf16>) -> tensor<512x8192xf32> {
    %3 = linalg.matmul ins(%arg1, %arg2 : tensor<512x1024xf16>, tensor<1024x8192xf16>)
        outs(%arg0 : tensor<512x8192xf32>) -> tensor<512x8192xf32>
    return %3 : tensor<512x8192xf32>
  }
}
"""

# Global (untiled) GEMM with transposed A (fed through a linalg.transpose):
# M=512, N=512, K=1024.
GEMM_TRANSPOSE_A_PAYLOAD = """
module {
  func.func @payload(%arg0: tensor<512x512xf32>, %arg1: tensor<1024x512xf16>,
                     %arg2: tensor<1024x512xf16>) -> tensor<512x512xf32> {
    %at = tensor.empty() : tensor<512x1024xf16>
    %t = linalg.transpose ins(%arg1 : tensor<1024x512xf16>)
        outs(%at : tensor<512x1024xf16>) permutation = [1, 0]
    %3 = linalg.matmul ins(%t, %arg2 : tensor<512x1024xf16>, tensor<1024x512xf16>)
        outs(%arg0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %3 : tensor<512x512xf32>
  }
}
"""


def build_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        mm = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        params = transform_ext.infer_xegpu_gemm_params(mm)
        transform.annotate(mm, "xegpu_params", param=params)
        transform.yield_()
    return sched


def run(name, payload_str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK-LABEL: Test: infer_xegpu_gemm_params
# CHECK: linalg.matmul {xegpu_params = {
# CHECK-DAG: layer_kind = "matmul"
# CHECK-DAG: m = 512 : i64
# CHECK-DAG: n = 8192 : i64
# CHECK-DAG: k = 1024 : i64
# CHECK-DAG: wg_m = 256 : i64
# CHECK-DAG: wg_n = 256 : i64
# CHECK-DAG: sg_m = 32 : i64
# CHECK-DAG: sg_n = 64 : i64
# CHECK-DAG: k_tile = 16 : i64
# CHECK-DAG: load_a_m = 8 : i64
# CHECK-DAG: load_a_k = 16 : i64
# CHECK-DAG: load_b_k = 16 : i64
# CHECK-DAG: load_b_n = 16 : i64
# CHECK-DAG: prefetch_a_m = 8 : i64
# CHECK-DAG: prefetch_a_k = 16 : i64
# CHECK-DAG: prefetch_b_k = 8 : i64
# CHECK-DAG: prefetch_b_n = 16 : i64
# CHECK-DAG: prefetch_a_nb = 1 : i64
# CHECK-DAG: prefetch_b_nb = 1 : i64
# CHECK-DAG: transpose_a = 0 : i64
# CHECK-DAG: transpose_b = 0 : i64
run("infer_xegpu_gemm_params", GEMM_PAYLOAD)

# CHECK-LABEL: Test: infer_xegpu_gemm_params_transpose_a
# CHECK: linalg.matmul {xegpu_params = {
# CHECK-DAG: layer_kind = "matmul"
# CHECK-DAG: m = 512 : i64
# CHECK-DAG: n = 512 : i64
# CHECK-DAG: k = 1024 : i64
# CHECK-DAG: wg_m = 128 : i64
# CHECK-DAG: wg_n = 128 : i64
# CHECK-DAG: sg_m = 32 : i64
# CHECK-DAG: sg_n = 32 : i64
# CHECK-DAG: k_tile = 16 : i64
# CHECK-DAG: load_a_m = 16 : i64
# CHECK-DAG: load_a_k = 16 : i64
# CHECK-DAG: load_b_k = 16 : i64
# CHECK-DAG: load_b_n = 16 : i64
# CHECK-DAG: prefetch_a_m = 8 : i64
# CHECK-DAG: prefetch_a_k = 16 : i64
# CHECK-DAG: prefetch_b_k = 8 : i64
# CHECK-DAG: prefetch_b_n = 16 : i64
# CHECK-DAG: prefetch_a_nb = 1 : i64
# CHECK-DAG: prefetch_b_nb = 1 : i64
# CHECK-DAG: transpose_a = 1 : i64
# CHECK-DAG: transpose_b = 0 : i64
run("infer_xegpu_gemm_params_transpose_a", GEMM_TRANSPOSE_A_PAYLOAD)
