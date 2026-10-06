# RUN: %PYTHON %s | FileCheck %s

"""Tests for analyze_matmul_op across the supported anchor op kinds."""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.matmul_analysis import (
    analyze_matmul_op,
    analyze_wg_k_tile_size,
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

# A workgroup-tiled linalg.matmul whose A operand is transposed through the op's
# own indexing_maps (A reads [k, m] instead of [m, k]); the tile-size guardrail
# must reject it since it reads M/K straight off operand 0.
MATMUL_TRANSPOSED_MAPS = """
#at = affine_map<(d0, d1, d2) -> (d2, d0)>
#b  = affine_map<(d0, d1, d2) -> (d2, d1)>
#c  = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%a: tensor<8x4xf32>, %b: tensor<8x16xf32>,
                  %c: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %r = scf.forall (%i) in (1) shared_outs(%o = %c) -> tensor<4x16xf32> {
      %mm = linalg.matmul indexing_maps = [#at, #b, #c]
          ins(%a, %b : tensor<8x4xf32>, tensor<8x16xf32>)
          outs(%o : tensor<4x16xf32>) -> tensor<4x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mm into %o[0, 0] [4, 16] [1, 1]
            : tensor<4x16xf32> into tensor<4x16xf32>
      }
    }
    return %r : tensor<4x16xf32>
  }
}
"""

# A workgroup-tiled linalg.matmul whose A operand is produced by a
# linalg.broadcast; the tile-size guardrail must reject it since the broadcast
# shape is not the real matmul operand shape.
MATMUL_BROADCAST_OPERAND = """
module {
  func.func @main(%bin: tensor<8xf32>, %b: tensor<8x16xf32>,
                  %c: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %r = scf.forall (%i) in (1) shared_outs(%o = %c) -> tensor<4x16xf32> {
      %be = tensor.empty() : tensor<4x8xf32>
      %bc = linalg.broadcast ins(%bin : tensor<8xf32>)
          outs(%be : tensor<4x8xf32>) dimensions = [0]
      %mm = linalg.matmul ins(%bc, %b : tensor<4x8xf32>, tensor<8x16xf32>)
          outs(%o : tensor<4x16xf32>) -> tensor<4x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mm into %o[0, 0] [4, 16] [1, 1]
            : tensor<4x16xf32> into tensor<4x16xf32>
      }
    }
    return %r : tensor<4x16xf32>
  }
}
"""

# Plain workgroup-tiled linalg.matmul (parent is scf.forall): wg_tile is read
# off the operand tiles, k_tile stays None (matmul carries no reduction loop).
MATMUL_WG_TILED = """
module {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>,
                  %c: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %r = scf.forall (%i) in (1) shared_outs(%o = %c) -> tensor<4x16xf32> {
      %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
          outs(%o : tensor<4x16xf32>) -> tensor<4x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mm into %o[0, 0] [4, 16] [1, 1]
            : tensor<4x16xf32> into tensor<4x16xf32>
      }
    }
    return %r : tensor<4x16xf32>
  }
}
"""

# A linalg.matmul outside any scf.forall: not workgroup tiled, so both tile
# sizes come back None.
MATMUL_NOT_TILED = """
module {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>,
                  %c: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%c : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %mm : tensor<4x16xf32>
  }
}
"""

# A vector.contract inside the reduction scf.for (WG + K loop nest): wg_tile is
# the accumulator (M, N), k_tile is the contracted dim of operand A.
VECTOR_CONTRACT_WG_K = """
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main() -> vector<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %acc0 = arith.constant dense<0.0> : vector<32x64xf32>
    %a = arith.constant dense<0.0> : vector<32x16xbf16>
    %b = arith.constant dense<0.0> : vector<16x64xbf16>
    %r = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%acc = %acc0) -> (vector<32x64xf32>) {
      %c = vector.contract {indexing_maps = [#map, #map1, #map2],
          iterator_types = ["parallel", "parallel", "reduction"],
          kind = #vector.kind<add>} %a, %b, %acc
          : vector<32x16xbf16>, vector<16x64xbf16> into vector<32x64xf32>
      scf.yield %c : vector<32x64xf32>
    }
    return %r : vector<32x64xf32>
  }
}
"""

# An xegpu.dpas inside the reduction scf.for (WG + K loop nest): wg_tile is the
# accumulator (M, N), k_tile is the contracted dim of operand A.
XEGPU_DPAS_WG_K = """
module {
  func.func @main() -> vector<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %acc0 = arith.constant dense<0.0> : vector<32x64xf32>
    %a = arith.constant dense<0.0> : vector<32x16xbf16>
    %b = arith.constant dense<0.0> : vector<16x64xbf16>
    %r = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%acc = %acc0) -> (vector<32x64xf32>) {
      %d = xegpu.dpas %a, %b, %acc
          : vector<32x16xbf16>, vector<16x64xbf16>, vector<32x64xf32>
          -> vector<32x64xf32>
      scf.yield %d : vector<32x64xf32>
    }
    return %r : vector<32x64xf32>
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


def run_tile_sizes_rejected(name: str, payload_text: str, anchor_name: str):
    """Apply analyze_wg_k_tile_size and print the guardrail error it raises."""
    print("Test:", name, flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload_text)
        anchor = find_op(module.operation, anchor_name)
        try:
            analyze_wg_k_tile_size(anchor)
        except ValueError as error:
            print("rejected:", error, flush=True)
            return
        raise AssertionError("expected analyze_wg_k_tile_size to reject this matmul")


def run_tile_sizes(name: str, payload_text: str, anchor_name: str):
    """Apply analyze_wg_k_tile_size and print the (wg_tile, k_tile) it infers."""
    print("Test:", name, flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload_text)
        anchor = find_op(module.operation, anchor_name)
        wg_tile, k_tile = analyze_wg_k_tile_size(anchor)
        print(f"wg_tile={wg_tile} k_tile={k_tile}", flush=True)


# CHECK-LABEL: Test: linalg_matmul
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("linalg_matmul", LINALG_MATMUL, "linalg.matmul")

# CHECK-LABEL: Test: vector_contract
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("vector_contract", VECTOR_CONTRACT, "vector.contract")

# CHECK-LABEL: Test: xegpu_dpas
# CHECK: shape=(1024, 8192, 8192) transpose_a=False transpose_b=True
run("xegpu_dpas", XEGPU_DPAS, "xegpu.dpas")

# CHECK-LABEL: Test: matmul_wg_tiled
# CHECK: wg_tile=(4, 16) k_tile=None
run_tile_sizes("matmul_wg_tiled", MATMUL_WG_TILED, "linalg.matmul")

# CHECK-LABEL: Test: matmul_not_tiled
# CHECK: wg_tile=None k_tile=None
run_tile_sizes("matmul_not_tiled", MATMUL_NOT_TILED, "linalg.matmul")

# CHECK-LABEL: Test: vector_contract_wg_k
# CHECK: wg_tile=(32, 64) k_tile=16
run_tile_sizes("vector_contract_wg_k", VECTOR_CONTRACT_WG_K, "vector.contract")

# CHECK-LABEL: Test: vector_contract_wg_only
# CHECK: wg_tile=(256, 256) k_tile=None
run_tile_sizes("vector_contract_wg_only", VECTOR_CONTRACT, "vector.contract")

# CHECK-LABEL: Test: xegpu_dpas_wg_k
# CHECK: wg_tile=(32, 64) k_tile=16
run_tile_sizes("xegpu_dpas_wg_k", XEGPU_DPAS_WG_K, "xegpu.dpas")

# CHECK-LABEL: Test: xegpu_dpas_wg_only
# CHECK: wg_tile=(256, 256) k_tile=None
run_tile_sizes("xegpu_dpas_wg_only", XEGPU_DPAS, "xegpu.dpas")

# CHECK-LABEL: Test: matmul_transposed_maps_rejected
# CHECK: rejected: linalg.matmul has non-identity (transposed) input indexing maps
run_tile_sizes_rejected(
    "matmul_transposed_maps_rejected", MATMUL_TRANSPOSED_MAPS, "linalg.matmul"
)

# CHECK-LABEL: Test: matmul_broadcast_operand_rejected
# CHECK: rejected: linalg.matmul has a linalg.broadcast producer
run_tile_sizes_rejected(
    "matmul_broadcast_operand_rejected", MATMUL_BROADCAST_OPERAND, "linalg.matmul"
)
