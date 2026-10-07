# RUN: %PYTHON %s | FileCheck %s

"""Tests for analyze_matmul_op across the supported anchor op kinds."""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.matmul_analysis import (
    analyze_matmul_op,
)


# Plain linalg.matmul with transposed B.
MATMUL_GLOBAL = """
module {
  func.func @payload(%arg0: tensor<2048x4096xf32>, %arg1: tensor<2048x8192xf16>,
                     %arg2: tensor<4096x8192xf16>) -> tensor<2048x4096xf32> {
    %bt = tensor.empty() : tensor<8192x4096xf16>
    %t = linalg.transpose ins(%arg2 : tensor<4096x8192xf16>)
        outs(%bt : tensor<8192x4096xf16>) permutation = [1, 0]
    %mm = linalg.matmul ins(%arg1, %t : tensor<2048x8192xf16>, tensor<8192x4096xf16>)
        outs(%arg0 : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
    return %mm : tensor<2048x4096xf32>
  }
}
"""

# Workgroup-tiled linalg.matmul with transposed B.
MATMUL_WG_TILED = """
module {
  func.func @payload(%arg0: tensor<2048x4096xf32>, %arg1: tensor<2048x8192xf16>,
                     %arg2: tensor<4096x8192xf16>) -> tensor<2048x4096xf32> {
    %4 = scf.forall (%arg3, %arg4) = (0, 0) to (2048, 4096) step (128, 256)
        shared_outs(%arg5 = %arg0) -> (tensor<2048x4096xf32>) {
      %sa = tensor.extract_slice %arg1[%arg3, 0] [128, 8192] [1, 1]
          : tensor<2048x8192xf16> to tensor<128x8192xf16>
      %sb = tensor.extract_slice %arg2[%arg4, 0] [256, 8192] [1, 1]
          : tensor<4096x8192xf16> to tensor<256x8192xf16>
      %sbt = tensor.empty() : tensor<8192x256xf16>
      %sc = tensor.extract_slice %arg5[%arg3, %arg4] [128, 256] [1, 1]
          : tensor<2048x4096xf32> to tensor<128x256xf32>
      %t = linalg.transpose ins(%sb : tensor<256x8192xf16>)
          outs(%sbt : tensor<8192x256xf16>) permutation = [1, 0]
      %mm = linalg.matmul ins(%sa, %t : tensor<128x8192xf16>, tensor<8192x256xf16>)
          outs(%sc : tensor<128x256xf32>) -> tensor<128x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mm into %arg5[%arg3, %arg4] [128, 256] [1, 1]
            : tensor<128x256xf32> into tensor<2048x4096xf32>
      }
    }
    return %4 : tensor<2048x4096xf32>
  }
}
"""

# Workgroup-tiled linalg.matmul where B is transposed before being sliced
# (transpose -> extract_slice -> matmul), so the slice source is already in
# matmul [K, N] orientation.
MATMUL_WG_TILED_TRANSPOSE_FIRST = """
module {
  func.func @payload(%arg0: tensor<2048x4096xf32>, %arg1: tensor<2048x8192xf16>,
                     %arg2: tensor<4096x8192xf16>) -> tensor<2048x4096xf32> {
    %bt = tensor.empty() : tensor<8192x4096xf16>
    %t = linalg.transpose ins(%arg2 : tensor<4096x8192xf16>)
        outs(%bt : tensor<8192x4096xf16>) permutation = [1, 0]
    %4 = scf.forall (%arg3, %arg4) = (0, 0) to (2048, 4096) step (128, 256)
        shared_outs(%arg5 = %arg0) -> (tensor<2048x4096xf32>) {
      %sa = tensor.extract_slice %arg1[%arg3, 0] [128, 8192] [1, 1]
          : tensor<2048x8192xf16> to tensor<128x8192xf16>
      %sbt = tensor.extract_slice %t[0, %arg4] [8192, 256] [1, 1]
          : tensor<8192x4096xf16> to tensor<8192x256xf16>
      %sc = tensor.extract_slice %arg5[%arg3, %arg4] [128, 256] [1, 1]
          : tensor<2048x4096xf32> to tensor<128x256xf32>
      %mm = linalg.matmul ins(%sa, %sbt : tensor<128x8192xf16>, tensor<8192x256xf16>)
          outs(%sc : tensor<128x256xf32>) -> tensor<128x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mm into %arg5[%arg3, %arg4] [128, 256] [1, 1]
            : tensor<128x256xf32> into tensor<2048x4096xf32>
      }
    }
    return %4 : tensor<2048x4096xf32>
  }
}
"""

# the global memrefs; the B indexing map (d1, d2) marks it as transposed.
VECTOR_CONTRACT = """
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<4096x8192xbf16>,
                  %arg2: memref<1024x8192xbf16>) -> vector<256x256xf32> {
    %pad = arith.constant 0.0 : bf16
    %cst = arith.constant dense<0.0> : vector<256x256xf32>
    %c0 = arith.constant 0 : index
    %a = vector.transfer_read %arg2[%c0, %c0], %pad {in_bounds = [true, true]}
        : memref<1024x8192xbf16>, vector<256x16xbf16>
    %b = vector.transfer_read %arg0[%c0, %c0], %pad {in_bounds = [true, true]}
        : memref<4096x8192xbf16>, vector<256x16xbf16>
    %c = vector.contract {indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>} %a, %b, %cst
        : vector<256x16xbf16>, vector<256x16xbf16> into vector<256x256xf32>
    return %c : vector<256x256xf32>
  }
}
"""

# xegpu.dpas at the xegpu level: operands trace back to create_nd_tdesc of the
# global memrefs; the B operand goes through a vector.transpose.
XEGPU_DPAS = """
module {
  func.func @main(%arg0: memref<4096x8192xbf16>,
                  %arg1: memref<1024x8192xbf16>) -> vector<256x256xf32> {
    %cst = arith.constant dense<0.0> : vector<256x256xf32>
    %c0 = arith.constant 0 : index
    %ta = xegpu.create_nd_tdesc %arg1 : memref<1024x8192xbf16>
        -> !xegpu.tensor_desc<256x16xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    %tb = xegpu.create_nd_tdesc %arg0 : memref<4096x8192xbf16>
        -> !xegpu.tensor_desc<256x16xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    %a = xegpu.load_nd %ta[%c0, %c0]
        : !xegpu.tensor_desc<256x16xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        -> vector<256x16xbf16>
    %b = xegpu.load_nd %tb[%c0, %c0]
        : !xegpu.tensor_desc<256x16xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        -> vector<256x16xbf16>
    %bt = vector.transpose %b, [1, 0] : vector<256x16xbf16> to vector<16x256xbf16>
    %d = xegpu.dpas %a, %bt, %cst
        : vector<256x16xbf16>, vector<16x256xbf16>, vector<256x256xf32>
        -> vector<256x256xf32>
    return %d : vector<256x256xf32>
  }
}
"""


def find_op(op: ir.Operation, name: str):
    """First op named `name` in `op`'s nested regions, or None."""
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                if child.operation.name == name:
                    return child
                found = find_op(child.operation, name)
                if found is not None:
                    return found
    return None


def run(name: str, payload_text: str, anchor_name: str):
    print("Test:", name, flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload_text)
        anchor = find_op(module.operation, anchor_name)
        shape, transpose_a, transpose_b = analyze_matmul_op(anchor)
        print(
            f"shape={shape} transpose_a={transpose_a} transpose_b={transpose_b}",
            flush=True,
        )


# CHECK-LABEL: Test: linalg_matmul_global
# CHECK: shape=(2048, 4096, 8192) transpose_a=False transpose_b=True
run("linalg_matmul_global", MATMUL_GLOBAL, "linalg.matmul")

# CHECK-LABEL: Test: linalg_matmul
# CHECK: shape=(2048, 4096, 8192) transpose_a=False transpose_b=True
run("linalg_matmul", MATMUL_WG_TILED, "linalg.matmul")

# CHECK-LABEL: Test: linalg_matmul_transpose_first
# CHECK: shape=(2048, 4096, 8192) transpose_a=False transpose_b=True
run("linalg_matmul_transpose_first", MATMUL_WG_TILED_TRANSPOSE_FIRST, "linalg.matmul")

# CHECK-LABEL: Test: vector_contract
# CHECK: shape=(1024, 4096, 8192) transpose_a=False transpose_b=True
run("vector_contract", VECTOR_CONTRACT, "vector.contract")

# CHECK-LABEL: Test: xegpu_dpas
# CHECK: shape=(1024, 4096, 8192) transpose_a=False transpose_b=True
run("xegpu_dpas", XEGPU_DPAS, "xegpu.dpas")
