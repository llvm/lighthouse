# RUN: %PYTHON %s | FileCheck %s

"""Tests for analyze_matmul_op across the supported anchor op kinds."""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.matmul_analysis import (
    analyze_matmul_op,
)

# linalg.matmul after workgroup tiling: operands are tensor.extract_slice tiles
# (so the global shape must be recovered from the slice source) and B is fed
# through a linalg.transpose.
LINALG_MATMUL = """
module {
  func.func @main(%a: tensor<1024x8192xbf16>, %b: tensor<8192x8192xbf16>,
                  %bt: tensor<8192x8192xbf16>) -> tensor<256x256xf32> {
    %cst = arith.constant 0.0 : f32
    %sa = tensor.extract_slice %a[0, 0] [256, 8192] [1, 1]
        : tensor<1024x8192xbf16> to tensor<256x8192xbf16>
    %sb = tensor.extract_slice %b[0, 0] [256, 8192] [1, 1]
        : tensor<8192x8192xbf16> to tensor<256x8192xbf16>
    %sbt = tensor.extract_slice %bt[0, 0] [8192, 256] [1, 1]
        : tensor<8192x8192xbf16> to tensor<8192x256xbf16>
    %t = linalg.transpose ins(%sb : tensor<256x8192xbf16>)
        outs(%sbt : tensor<8192x256xbf16>) permutation = [1, 0]
    %e = tensor.empty() : tensor<256x256xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<256x256xf32>)
        -> tensor<256x256xf32>
    %mm = linalg.matmul ins(%sa, %t : tensor<256x8192xbf16>, tensor<8192x256xbf16>)
        outs(%f : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %mm : tensor<256x256xf32>
  }
}
"""

# vector.contract after bufferization: operands trace back to transfer_read of
# the global memrefs; the B indexing map (d1, d2) marks it as transposed.
VECTOR_CONTRACT = """
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<8192x8192xbf16>,
                  %arg2: memref<1024x8192xbf16>) -> vector<256x256xf32> {
    %pad = arith.constant 0.0 : bf16
    %cst = arith.constant dense<0.0> : vector<256x256xf32>
    %c0 = arith.constant 0 : index
    %a = vector.transfer_read %arg2[%c0, %c0], %pad {in_bounds = [true, true]}
        : memref<1024x8192xbf16>, vector<256x16xbf16>
    %b = vector.transfer_read %arg0[%c0, %c0], %pad {in_bounds = [true, true]}
        : memref<8192x8192xbf16>, vector<256x16xbf16>
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
  func.func @main(%arg0: memref<8192x8192xbf16>,
                  %arg1: memref<1024x8192xbf16>) -> vector<256x256xf32> {
    %cst = arith.constant dense<0.0> : vector<256x256xf32>
    %c0 = arith.constant 0 : index
    %ta = xegpu.create_nd_tdesc %arg1 : memref<1024x8192xbf16>
        -> !xegpu.tensor_desc<256x16xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    %tb = xegpu.create_nd_tdesc %arg0 : memref<8192x8192xbf16>
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


# CHECK-LABEL: Test: linalg_matmul
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("linalg_matmul", LINALG_MATMUL, "linalg.matmul")

# CHECK-LABEL: Test: vector_contract
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("vector_contract", VECTOR_CONTRACT, "vector.contract")

# CHECK-LABEL: Test: xegpu_dpas
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("xegpu_dpas", XEGPU_DPAS, "xegpu.dpas")
