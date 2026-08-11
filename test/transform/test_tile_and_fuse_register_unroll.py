# RUN: %PYTHON %s | FileCheck %s

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule import tile_and_fuse as tf


def run(name: str, payload_str: str, *schedules):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        modules = []
        for make_schedule in schedules:
            sched = make_schedule()
            modules.append(sched)
            sched.body.operations[0].apply(payload.operation)
        print(payload)


def tile_and_unroll_matmul():
    return tf.tile_and_unroll_annotated(
        target_op="linalg.matmul",
        clear_annotations=False,
    )


def tile_and_unroll_all_annotated():
    return tf.tile_and_unroll_annotated(
        clear_annotations=False,
    )


def tile_and_unroll_default_clear():
    return tf.tile_and_unroll_annotated()


def tile_and_unroll_keep_annotations():
    return tf.tile_and_unroll_annotated(clear_annotations=False)


def assign_register_unroll():
    return tf.assign_and_propagate_tile_sizes(
        strategy="register_unroll",
    )


def assign_elementwise_register_unroll():
    return tf.assign_elementwise_tile_sizes(
        strategy="register_unroll",
        propagate=False,
    )


def assign_elementwise_register_unroll64():
    return tf.assign_elementwise_tile_sizes(
        tile_size=64,
        strategy="register_unroll",
        propagate=False,
    )


MLP = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<32x64xf32>, %w: tensor<64x128xf32>, %b: tensor<128xf32>)
        -> tensor<32x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %3 = linalg.matmul ins(%arg0, %w : tensor<32x64xf32>, tensor<64x128xf32>)
        outs(%2 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%3, %b : tensor<32x128xf32>, tensor<128xf32>)
        outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<32x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%4 : tensor<32x128xf32>)
        outs(%1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<32x128xf32>
    return %5 : tensor<32x128xf32>
  }
}
"""


# A K-split matmul in an scf.for accumulation loop, with elementwise consumers
# outside the loop.
K_LOOP_GEMM_OUTER_CHAIN = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<32x64xf32>, %b: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x128xf32>
    %acc0 = linalg.fill ins(%cst : f32) outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
    %acc = scf.for %k = %c0 to %c64 step %c32 iter_args(%iter = %acc0) -> (tensor<32x128xf32>) {
      %a_slice = tensor.extract_slice %a[0, %k] [32, 32] [1, 1]
          : tensor<32x64xf32> to tensor<32x32xf32>
      %b_slice = tensor.extract_slice %b[%k, 0] [32, 128] [1, 1]
          : tensor<64x128xf32> to tensor<32x128xf32>
      %mm = linalg.matmul ins(%a_slice, %b_slice : tensor<32x32xf32>, tensor<32x128xf32>)
          outs(%iter : tensor<32x128xf32>) -> tensor<32x128xf32>
      scf.yield %mm : tensor<32x128xf32>
    }
    %out0 = tensor.empty() : tensor<32x128xf32>
    %relu = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%acc : tensor<32x128xf32>)
        outs(%out0 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<32x128xf32>
    %out1 = tensor.empty() : tensor<32x128xf32>
    %exp = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%relu : tensor<32x128xf32>)
        outs(%out1 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<32x128xf32>
    return %exp : tensor<32x128xf32>
  }
}
"""


# Same K-loop shape but with bf16 GEMM operands accumulating into f32, so an
# AMX-capable target takes the AMX branch for the matmul while the epilogue
# still falls back to the generic SIMD tiles.
K_LOOP_GEMM_BF16_EPILOGUE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<32x64xbf16>, %b: tensor<64x128xbf16>) -> tensor<32x128xf32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x128xf32>
    %acc0 = linalg.fill ins(%cst : f32) outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
    %acc = scf.for %k = %c0 to %c64 step %c32 iter_args(%iter = %acc0) -> (tensor<32x128xf32>) {
      %a_slice = tensor.extract_slice %a[0, %k] [32, 32] [1, 1]
          : tensor<32x64xbf16> to tensor<32x32xbf16>
      %b_slice = tensor.extract_slice %b[%k, 0] [32, 128] [1, 1]
          : tensor<64x128xbf16> to tensor<32x128xbf16>
      %mm = linalg.matmul ins(%a_slice, %b_slice : tensor<32x32xbf16>, tensor<32x128xbf16>)
          outs(%iter : tensor<32x128xf32>) -> tensor<32x128xf32>
      scf.yield %mm : tensor<32x128xf32>
    }
    %out0 = tensor.empty() : tensor<32x128xf32>
    %relu = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%acc : tensor<32x128xf32>)
        outs(%out0 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<32x128xf32>
    return %relu : tensor<32x128xf32>
  }
}
"""


HIGH_DIM5_BCAST_EPILOGUE = """
#mapA = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#mapB = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#mapC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#mapBias = affine_map<(d0, d1, d2, d3) -> (d3)>
#mapOut = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%a: tensor<2x8x64xf32>, %b: tensor<2x64x16x32xf32>, %bias: tensor<32xf32>)
        -> tensor<2x8x16x32xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2x8x16x32xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<2x8x16x32xf32>) -> tensor<2x8x16x32xf32>
    %c = linalg.contract indexing_maps = [#mapA, #mapB, #mapC]
        ins(%a, %b : tensor<2x8x64xf32>, tensor<2x64x16x32xf32>)
        outs(%f : tensor<2x8x16x32xf32>) -> tensor<2x8x16x32xf32>
    %o = tensor.empty() : tensor<2x8x16x32xf32>
    %ep = linalg.generic {
        indexing_maps = [#mapOut, #mapBias, #mapOut],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%c, %bias : tensor<2x8x16x32xf32>, tensor<32xf32>)
        outs(%o : tensor<2x8x16x32xf32>) {
    ^bb0(%x: f32, %b0: f32, %y: f32):
      %s = arith.addf %x, %b0 : f32
      linalg.yield %s : f32
    } -> tensor<2x8x16x32xf32>
    return %ep : tensor<2x8x16x32xf32>
  }
}
"""


ELTWISE_CHAIN_1D_4D = """
#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @chain1d(%x: tensor<128xf32>) -> tensor<128xf32> {
    %e = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%x : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.exp %i : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>
    %b = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%a : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.sqrt %i : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>
    return %b : tensor<128xf32>
  }

  func.func @chain2d(%x: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %e = tensor.empty() : tensor<64x128xf32>
    %a = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%x : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.exp %i : f32
      linalg.yield %v : f32
    } -> tensor<64x128xf32>
    %b = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.sqrt %i : f32
      linalg.yield %v : f32
    } -> tensor<64x128xf32>
    return %b : tensor<64x128xf32>
  }

  func.func @chain3d(%x: tensor<8x32x128xf32>) -> tensor<8x32x128xf32> {
    %e = tensor.empty() : tensor<8x32x128xf32>
    %a = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x : tensor<8x32x128xf32>) outs(%e : tensor<8x32x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.exp %i : f32
      linalg.yield %v : f32
    } -> tensor<8x32x128xf32>
    %b = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<8x32x128xf32>) outs(%e : tensor<8x32x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.sqrt %i : f32
      linalg.yield %v : f32
    } -> tensor<8x32x128xf32>
    return %b : tensor<8x32x128xf32>
  }

  func.func @chain4d(%x: tensor<4x8x32x128xf32>) -> tensor<4x8x32x128xf32> {
    %e = tensor.empty() : tensor<4x8x32x128xf32>
    %a = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%x : tensor<4x8x32x128xf32>) outs(%e : tensor<4x8x32x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.exp %i : f32
      linalg.yield %v : f32
    } -> tensor<4x8x32x128xf32>
    %b = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%a : tensor<4x8x32x128xf32>) outs(%e : tensor<4x8x32x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.sqrt %i : f32
      linalg.yield %v : f32
    } -> tensor<4x8x32x128xf32>
    return %b : tensor<4x8x32x128xf32>
  }
}
"""


ELTWISE_DTYPE_WIDTHS = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @f32_case(%x: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %e = tensor.empty() : tensor<64x256xf32>
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x : tensor<64x256xf32>) outs(%e : tensor<64x256xf32>) {
    ^bb0(%i: f32, %o: f32):
      %v = math.exp %i : f32
      linalg.yield %v : f32
    } -> tensor<64x256xf32>
    return %a : tensor<64x256xf32>
  }

  func.func @bf16_case(%x: tensor<64x256xbf16>) -> tensor<64x256xbf16> {
    %c1 = arith.constant 1.0 : bf16
    %e = tensor.empty() : tensor<64x256xbf16>
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x : tensor<64x256xbf16>) outs(%e : tensor<64x256xbf16>) {
    ^bb0(%i: bf16, %o: bf16):
      %v = arith.addf %i, %c1 : bf16
      linalg.yield %v : bf16
    } -> tensor<64x256xbf16>
    return %a : tensor<64x256xbf16>
  }

  func.func @i16_case(%x: tensor<64x256xi16>) -> tensor<64x256xi16> {
    %c1 = arith.constant 1 : i16
    %e = tensor.empty() : tensor<64x256xi16>
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x : tensor<64x256xi16>) outs(%e : tensor<64x256xi16>) {
    ^bb0(%i: i16, %o: i16):
      %v = arith.addi %i, %c1 : i16
      linalg.yield %v : i16
    } -> tensor<64x256xi16>
    return %a : tensor<64x256xi16>
  }

  func.func @i8_case(%x: tensor<64x256xi8>) -> tensor<64x256xi8> {
    %c1 = arith.constant 1 : i8
    %e = tensor.empty() : tensor<64x256xi8>
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x : tensor<64x256xi8>) outs(%e : tensor<64x256xi8>) {
    ^bb0(%i: i8, %o: i8):
      %v = arith.addi %i, %c1 : i8
      linalg.yield %v : i8
    } -> tensor<64x256xi8>
    return %a : tensor<64x256xi8>
  }
}
"""


# CHECK-LABEL: Test: tile_and_unroll_annotated
# CHECK: func.func @main
# CHECK-NOT: scf.for
# CHECK-NOT: scf.forall
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 1, 16, 1>}
# CHECK: return
run(
    "tile_and_unroll_annotated",
    MLP,
    lambda: tf.assign_and_propagate_tile_sizes(strategy="register_unroll"),
    tile_and_unroll_matmul,
)


# CHECK-LABEL: Test: register_unroll_propagates_parallel_dims_only
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 1, 16>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 1, 16, 1>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
run(
    "register_unroll_propagates_parallel_dims_only",
    MLP,
    assign_register_unroll,
)


# CHECK-LABEL: Test: tile_and_unroll_all_annotated_ops
# CHECK: func.func @main
# CHECK-NOT: scf.for
# CHECK-NOT: scf.forall
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 1, 16>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 1, 16, 1>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: return
run(
    "tile_and_unroll_all_annotated_ops",
    MLP,
    assign_register_unroll,
    assign_elementwise_register_unroll,
    tile_and_unroll_all_annotated,
)


# The elementwise ops sit outside the K loop, so propagation does not reach them
# and they fall back to the SIMD-width-derived generic tiles; pin the target.
# CHECK-LABEL: Test: k_loop_matmul_unroll_all_annotated
# CHECK: func.func @main
# CHECK-COUNT-1: scf.for
# CHECK-NOT: scf.forall
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 1, 16, 1>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=["avx512f"]):
    run(
        "k_loop_matmul_unroll_all_annotated",
        K_LOOP_GEMM_OUTER_CHAIN,
        assign_register_unroll,
        assign_elementwise_register_unroll,
        tile_and_unroll_all_annotated,
    )


# The bf16 GEMM takes the AMX tiles, while the f32 fill and epilogue outside the
# K loop keep the generic SIMD tiles.
# CHECK-LABEL: Test: k_loop_bf16_amx_matmul_unroll
# CHECK: func.func @main
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 1, 16>}
# CHECK-COUNT-1: scf.for
# CHECK-NOT: scf.forall
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 16, 16, 32>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=["amx_tile", "avx512f"]):
    run(
        "k_loop_bf16_amx_matmul_unroll",
        K_LOOP_GEMM_BF16_EPILOGUE,
        assign_register_unroll,
        assign_elementwise_register_unroll,
        tile_and_unroll_all_annotated,
    )


# CHECK-LABEL: Test: tile_and_unroll_default_clears_annotations
# CHECK: func.func @main
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "tile_and_unroll_default_clears_annotations",
    MLP,
    assign_register_unroll,
    assign_elementwise_register_unroll,
    tile_and_unroll_default_clear,
)


# CHECK-LABEL: Test: tile_and_unroll_can_keep_annotations
# CHECK: func.func @main
# CHECK: transform_ext.tile_sizes
run(
    "tile_and_unroll_can_keep_annotations",
    MLP,
    assign_register_unroll,
    assign_elementwise_register_unroll,
    tile_and_unroll_keep_annotations,
)


# CHECK-LABEL: Test: high_dim5_contract_broadcast_register_unroll
# CHECK: linalg.contract
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 1, 16, 1>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 1, 16>
run(
    "high_dim5_contract_broadcast_register_unroll",
    HIGH_DIM5_BCAST_EPILOGUE,
    assign_register_unroll,
)


# The generic branch derives its inner tile from the target's SIMD width, so pin
# the target to keep the expected sizes independent of the host CPU.
# CHECK-LABEL: Test: eltwise_chain_1d_to_4d_register_unroll
# CHECK: func.func @chain1d
# CHECK: transform_ext.tile_sizes = array<i64: 16>
# CHECK: func.func @chain2d
# CHECK: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: func.func @chain3d
# CHECK: transform_ext.tile_sizes = array<i64: 1, 1, 16>
# CHECK: func.func @chain4d
# CHECK: transform_ext.tile_sizes = array<i64: 1, 1, 1, 16>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_chain_1d_to_4d_register_unroll",
        ELTWISE_CHAIN_1D_4D,
        assign_elementwise_register_unroll,
    )


# Element width adaptation at tile_size=64 on 512-bit vectors: f32->16, bf16->32,
# i16->32, i8->64.
# CHECK-LABEL: Test: eltwise_dtype_width_adaptive_register_unroll
# CHECK: func.func @f32_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: func.func @bf16_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 32>
# CHECK: func.func @i16_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 32>
# CHECK: func.func @i8_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 64>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_dtype_width_adaptive_register_unroll",
        ELTWISE_DTYPE_WIDTHS,
        assign_elementwise_register_unroll64,
    )


# Same payload on a 256-bit target: every inner tile halves, confirming the
# generic branch really tracks the target vector width.
# CHECK-LABEL: Test: eltwise_dtype_width_adaptive_register_unroll_avx2
# CHECK: func.func @f32_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 8>
# CHECK: func.func @bf16_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: func.func @i16_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 16>
# CHECK: func.func @i8_case
# CHECK: transform_ext.tile_sizes = array<i64: 1, 32>
with TargetInfo.override(features=["avx2"]):
    run(
        "eltwise_dtype_width_adaptive_register_unroll_avx2",
        ELTWISE_DTYPE_WIDTHS,
        assign_elementwise_register_unroll64,
    )
