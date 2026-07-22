# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def apply_schedule(payload_str, build_roots, name):
    """Parse a payload, build a `get_fusion_roots` schedule and print the roots."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        with schedule_boilerplate() as (sched, named_seq):
            roots = build_roots(named_seq)
            transform.print_(target=roots, name=name)
            transform.yield_()
        sched.body.operations[0].apply(payload.operation)


# Four independent elementwise ops, each its own fusion root, alternating
# linalg.generic / linalg.add in program order and each with a unique shape so
# it can be identified in the printed output.
SCRAMBLED = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<10x10xf32>, %b: tensor<11x11xf32>, %c: tensor<12x12xf32>,
        %d: tensor<13x13xf32>)
        -> (tensor<10x10xf32>, tensor<11x11xf32>, tensor<12x12xf32>, tensor<13x13xf32>) {
    %e0 = tensor.empty() : tensor<10x10xf32>
    %gA = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%a : tensor<10x10xf32>)
        outs(%e0 : tensor<10x10xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<10x10xf32>
    %e1 = tensor.empty() : tensor<11x11xf32>
    %addB = linalg.add {transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%b, %b : tensor<11x11xf32>, tensor<11x11xf32>)
        outs(%e1 : tensor<11x11xf32>) -> tensor<11x11xf32>
    %e2 = tensor.empty() : tensor<12x12xf32>
    %gC = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%c : tensor<12x12xf32>)
        outs(%e2 : tensor<12x12xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<12x12xf32>
    %e3 = tensor.empty() : tensor<13x13xf32>
    %addD = linalg.add {transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%d, %d : tensor<13x13xf32>, tensor<13x13xf32>)
        outs(%e3 : tensor<13x13xf32>) -> tensor<13x13xf32>
    return %gA, %addB, %gC, %addD
        : tensor<10x10xf32>, tensor<11x11xf32>, tensor<12x12xf32>, tensor<13x13xf32>
  }
}
"""

# A GEMM group: fill (prologue) -> matmul (barrier) -> relu (epilogue). Only the
# epilogue relu is a root: the fill is fused as a producer and the matmul is a
# barrier that still has an elementwise epilogue.
GEMM_GROUP = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<32x32xf32>, %w: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<32x32xf32>
    %f = linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%cst : f32)
        outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
        ins(%a, %w : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
    %relu = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%mm : tensor<32x32xf32>)
        outs(%e : tensor<32x32xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<32x32xf32>
    return %relu : tensor<32x32xf32>
  }
}
"""

# A GEMM barrier with no epilogue: fill (prologue) -> matmul (barrier), whose
# result is returned directly. The matmul is its own root; the fill is fused.
BARRIER_ONLY = """
module {
  func.func @main(%a: tensor<32x32xf32>, %w: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<32x32xf32>
    %f = linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%cst : f32)
        outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
        ins(%a, %w : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %mm : tensor<32x32xf32>
  }
}
"""

# Two chained elementwise ops pre-annotated with conflicting tile sizes:
#   producer 32x32 -> consumer 64x64 sharing a tensor.
# Because the tilings conflict on the shared tensor, they are different groups:
#   both are roots (each is tiled on its own).
# With compatible sizes only the downstream consumer would be a root.
INCOMPATIBLE_CHAIN = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<128x128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%a : tensor<128x128xf32>)
        outs(%e0 : tensor<128x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %e = math.exp %i : f32
      linalg.yield %e : f32
    } -> tensor<128x128xf32>
    %e1 = tensor.empty() : tensor<128x128xf32>
    %c = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 64, 64>}
        ins(%p : tensor<128x128xf32>)
        outs(%e1 : tensor<128x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %s = math.sqrt %i : f32
      linalg.yield %s : f32
    } -> tensor<128x128xf32>
    return %c : tensor<128x128xf32>
  }
}
"""

# The same chain with matching tile sizes (both 32x32): the two ops form a single
# group, so only the downstream consumer (the sqrt) is a root.
COMPATIBLE_CHAIN = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<128x128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%a : tensor<128x128xf32>)
        outs(%e0 : tensor<128x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %e = math.exp %i : f32
      linalg.yield %e : f32
    } -> tensor<128x128xf32>
    %e1 = tensor.empty() : tensor<128x128xf32>
    %c = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 32, 32>}
        ins(%p : tensor<128x128xf32>)
        outs(%e1 : tensor<128x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %s = math.sqrt %i : f32
      linalg.yield %s : f32
    } -> tensor<128x128xf32>
    return %c : tensor<128x128xf32>
  }
}
"""


def scrambled_roots(named_seq):
    # Match the two op kinds separately and merge them: the handle lists all
    # generics first, then all adds, so it is deliberately *not* in program
    # order. get_fusion_roots must still return the roots top-down.
    gens = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
    adds = lh_transform.match_op(named_seq.bodyTarget, "linalg.add")
    merged = transform.merge_handles([gens, adds])
    return transform_ext.get_fusion_roots(merged)


def all_linalg_roots(named_seq):
    candidates = lh_transform.match_op(
        named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
    )
    return transform_ext.get_fusion_roots(candidates)


# The input handle is scrambled (generics [10, 12] then adds [11, 13]); the roots
# must come back in program order regardless: 10, 11, 12, 13.
# CHECK: IR printer: PROGRAM_ORDER
# CHECK: tensor<10x10xf32>
# CHECK: tensor<11x11xf32>
# CHECK: tensor<12x12xf32>
# CHECK: tensor<13x13xf32>
apply_schedule(SCRAMBLED, scrambled_roots, "PROGRAM_ORDER")


# Only the epilogue relu (a linalg.generic) is a root; the prologue fill and the
# matmul barrier are not returned.
# CHECK: IR printer: GEMM_GROUP
# CHECK: linalg.generic
# CHECK-NOT: linalg.matmul
# CHECK-NOT: linalg.fill
apply_schedule(GEMM_GROUP, all_linalg_roots, "GEMM_GROUP")


# A barrier with no epilogue is its own root: the matmul is returned, the fill is
# not.
# CHECK: IR printer: BARRIER_ONLY
# CHECK: linalg.matmul
# CHECK-NOT: linalg.fill
apply_schedule(BARRIER_ONLY, all_linalg_roots, "BARRIER_ONLY")


# Conflicting hand-annotated tile sizes split the chain: the producer (32x32) and
# the consumer (64x64) are separate groups, so both are returned as roots, in
# program order (producer first).
# CHECK: IR printer: INCOMPATIBLE_CHAIN
# CHECK: tile_sizes = array<i64: 32, 32>
# CHECK: tile_sizes = array<i64: 64, 64>
apply_schedule(INCOMPATIBLE_CHAIN, all_linalg_roots, "INCOMPATIBLE_CHAIN")


# Matching tile sizes keep the chain in one group: only the downstream consumer
# (sqrt) is a root; the producer (exp) is fused as its producer, so it is not
# printed.
# CHECK: IR printer: COMPATIBLE_CHAIN
# CHECK-NOT: math.exp
# CHECK: math.sqrt
apply_schedule(COMPATIBLE_CHAIN, all_linalg_roots, "COMPATIBLE_CHAIN")
