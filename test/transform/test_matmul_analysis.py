# RUN: %PYTHON %s | FileCheck %s

"""Tests for analyze_matmul_op across the supported anchor op kinds."""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.matmul_analysis import (
    analyze_matmul_op,
    analyze_wg_k_tile_size,
)

# vector.contract after bufferization: operands trace back to transfer_read of
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

# Plain workgroup-tiled linalg.matmul.
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

# A workgroup-tiled and k-tiled linalg.matmul.
MATMUL_WG_K_TILED = """
module {
  func.func @payload(%arg0: tensor<2048x4096xf32>, %arg1: tensor<2048x8192xf16>,
                     %arg2: tensor<4096x8192xf16>) -> tensor<2048x4096xf32> {
    %c16 = arith.constant 16 : index
    %c8192 = arith.constant 8192 : index
    %c0 = arith.constant 0 : index
    %4 = scf.forall (%arg3, %arg4) = (0, 0) to (2048, 4096) step (128, 256)
        shared_outs(%arg5 = %arg0) -> (tensor<2048x4096xf32>) {
      %sa = tensor.extract_slice %arg1[%arg3, 0] [128, 8192] [1, 1]
          : tensor<2048x8192xf16> to tensor<128x8192xf16>
      %sb = tensor.extract_slice %arg2[%arg4, 0] [256, 8192] [1, 1]
          : tensor<4096x8192xf16> to tensor<256x8192xf16>
      %sc = tensor.extract_slice %arg5[%arg3, %arg4] [128, 256] [1, 1]
          : tensor<2048x4096xf32> to tensor<128x256xf32>
      %5 = scf.for %arg6 = %c0 to %c8192 step %c16
          iter_args(%arg7 = %sc) -> (tensor<128x256xf32>) {
        %ka = tensor.extract_slice %sa[0, %arg6] [128, 16] [1, 1]
            : tensor<128x8192xf16> to tensor<128x16xf16>
        %kb = tensor.extract_slice %sb[0, %arg6] [256, 16] [1, 1]
            : tensor<256x8192xf16> to tensor<256x16xf16>
        %kbt = tensor.empty() : tensor<16x256xf16>
        %t = linalg.transpose ins(%kb : tensor<256x16xf16>)
            outs(%kbt : tensor<16x256xf16>) permutation = [1, 0]
        %mm = linalg.matmul ins(%ka, %t : tensor<128x16xf16>, tensor<16x256xf16>)
            outs(%arg7 : tensor<128x256xf32>) -> tensor<128x256xf32>
        scf.yield %mm : tensor<128x256xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg5[%arg3, %arg4] [128, 256] [1, 1]
            : tensor<128x256xf32> into tensor<2048x4096xf32>
      }
    }
    return %4 : tensor<2048x4096xf32>
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
# CHECK: shape=(2048, 4096, 8192) transpose_a=False transpose_b=True
run("linalg_matmul", MATMUL_WG_TILED, "linalg.matmul")

# CHECK-LABEL: Test: vector_contract
# CHECK: shape=(1024, 4096, 8192) transpose_a=False transpose_b=True
run("vector_contract", VECTOR_CONTRACT, "vector.contract")

# CHECK-LABEL: Test: xegpu_dpas
# CHECK: shape=(1024, 4096, 8192) transpose_a=False transpose_b=True
run("xegpu_dpas", XEGPU_DPAS, "xegpu.dpas")

# CHECK-LABEL: Test: matmul_wg_tiled
# CHECK: wg_tile=(128, 256) k_tile=None
run_tile_sizes("matmul_wg_tiled", MATMUL_WG_TILED, "linalg.matmul")

# CHECK-LABEL: Test: matmul_wg_k_tiled
# CHECK: wg_tile=(128, 256) k_tile=16
run_tile_sizes("matmul_wg_k_tiled", MATMUL_WG_K_TILED, "linalg.matmul")

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
