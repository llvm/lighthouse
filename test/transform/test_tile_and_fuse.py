# RUN: %PYTHON %s | FileCheck %s

"""Tests for the generic tile-and-fuse transform ops and schedules.

Exercises the three-step strategy on linalg payloads:
    1. assign tile sizes to anchor ops (GEMMs / elementwise)
    2. propagate them to neighboring ops
    3. tile and fuse using the annotations
"""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.schedule import tile_and_fuse as tf


def run(name: str, payload_str: str, *schedules):
    """Parse a payload, apply the given schedules in order and print it."""
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        # Keep schedule modules alive while applying them.
        modules = []
        for make_schedule in schedules:
            sched = make_schedule()
            modules.append(sched)
            sched.body.operations[0].apply(payload.operation)
        print(payload)


def assign_gemm():
    return tf.assign_and_propagate_tile_sizes(tile_size=32)


def assign_elementwise():
    return tf.assign_elementwise_tile_sizes(tile_size=32)


def assign_elementwise64():
    return tf.assign_elementwise_tile_sizes(tile_size=64)


def tile_and_fuse():
    return tf.tile_and_fuse_annotated()


def tile_and_fuse_keep():
    # Opt out of post-fusion annotation clearing so a test can still inspect the
    # propagated tile sizes on the fused ops.
    return tf.tile_and_fuse_annotated(clear_annotations=False)


def tile_and_fuse_for():
    # Tile with sequential scf.for loops (a nested loop nest) instead of a single
    # multi-dim scf.forall.
    return tf.tile_and_fuse_annotated(use_forall=False)


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

ELTWISE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<64x256xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<64x256xf32>)
        outs(%0 : tensor<64x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<64x256xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%1 : tensor<64x256xf32>)
        outs(%0 : tensor<64x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x256xf32>
    return %2 : tensor<64x256xf32>
  }
}
"""

BMM = """
module {
  func.func @main(%a: tensor<8x64x96xf32>, %b: tensor<8x96x128xf32>) -> tensor<8x64x128xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<8x64x128xf32>
    %1 = linalg.fill ins(%cst: f32) outs(%0: tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    %2 = linalg.batch_matmul ins(%a, %b : tensor<8x64x96xf32>, tensor<8x96x128xf32>)
        outs(%1 : tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %2 : tensor<8x64x128xf32>
  }
}
"""

# A named elementwise op (linalg.add) followed by a relu generic.
# The elementwise anchor schedule must cover named variants, not just linalg.generic.
NAMED_ELTWISE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<64x256xf32>, %b: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x256xf32>
    %add = linalg.add ins(%a, %b : tensor<64x256xf32>, tensor<64x256xf32>)
        outs(%e : tensor<64x256xf32>) -> tensor<64x256xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%add : tensor<64x256xf32>)
        outs(%e : tensor<64x256xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x256xf32>
    return %relu : tensor<64x256xf32>
  }
}
"""

REDUCE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = arith.constant 0.0 : f32
    %ein = tensor.empty() : tensor<64x256xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<64x256xf32>)
        outs(%ein : tensor<64x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x256xf32>
    %e = tensor.empty() : tensor<64xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<64xf32>) -> tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #out],
        iterator_types = ["parallel", "reduction"]}
        ins(%relu : tensor<64x256xf32>)
        outs(%f : tensor<64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %s = arith.addf %in, %o : f32
      linalg.yield %s : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
"""

# Two GEMMs chained through a relu. The relu is the epilogue of the first matmul
# and the prologue of the second: it must be tiled to match its producer GEMM.
TWO_GEMM = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<64x64xf32>, %w0: tensor<64x64xf32>, %w1: tensor<64x64xf32>)
        -> tensor<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<64x64xf32>
    %f0 = linalg.fill ins(%cst: f32) outs(%e0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm0 = linalg.matmul ins(%a, %w0 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu0 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm0 : tensor<64x64xf32>)
        outs(%e0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %f1 = linalg.fill ins(%cst: f32) outs(%e0: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm1 = linalg.matmul ins(%relu0, %w1 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu1 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm1 : tensor<64x64xf32>)
        outs(%e0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %c = arith.cmpf ugt, %in, %cst : f32
      %s = arith.select %c, %in, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %relu1 : tensor<64x64xf32>
  }
}
"""

# A 3-layer MLP: each layer is matmul + bias-add + relu (last layer bias only).
MLP3 = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%x: tensor<64x64xf32>, %w0: tensor<64x64xf32>, %b0: tensor<64xf32>,
        %w1: tensor<64x64xf32>, %b1: tensor<64xf32>, %w2: tensor<64x64xf32>,
        %b2: tensor<64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %f0 = linalg.fill ins(%cst: f32) outs(%e: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm0 = linalg.matmul ins(%x, %w0 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %a0 = linalg.generic {indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm0, %b0 : tensor<64x64xf32>, tensor<64xf32>)
        outs(%e : tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      %c = arith.cmpf ugt, %s, %cst : f32
      %r = arith.select %c, %s, %cst : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %mm1 = linalg.matmul ins(%a0, %w1 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %a1 = linalg.generic {indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm1, %b1 : tensor<64x64xf32>, tensor<64xf32>)
        outs(%e : tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      %c = arith.cmpf ugt, %s, %cst : f32
      %r = arith.select %c, %s, %cst : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    %mm2 = linalg.matmul ins(%a1, %w2 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %a2 = linalg.generic {indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm2, %b2 : tensor<64x64xf32>, tensor<64xf32>)
        outs(%e : tensor<64x64xf32>) {
    ^bb0(%in: f32, %ib: f32, %o: f32):
      %s = arith.addf %in, %ib : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %a2 : tensor<64x64xf32>
  }
}
"""

# A matmul + relu with dynamic M and N dimensions.
DYN = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<?x64xf32>, %w: tensor<64x?xf32>, %init: tensor<?x?xf32>)
        -> tensor<?x?xf32> {
    %cst = arith.constant 0.0 : f32
    %f = linalg.fill ins(%cst: f32) outs(%init: tensor<?x?xf32>) -> tensor<?x?xf32>
    %mm = linalg.matmul ins(%a, %w : tensor<?x64xf32>, tensor<64x?xf32>)
        outs(%f : tensor<?x?xf32>) -> tensor<?x?xf32>
    %relu = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%mm : tensor<?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<?x?xf32>
    return %relu : tensor<?x?xf32>
  }
}
"""

# A high-dimensional contraction.
HIGH_DIM_CONTRACT = """
#mapA = affine_map<(b, m0, m1, n, k) -> (b, m0, m1, k)>
#mapB = affine_map<(b, m0, m1, n, k) -> (b, k, n)>
#mapC = affine_map<(b, m0, m1, n, k) -> (b, m0, m1, n)>
module {
  func.func @main(%a: tensor<2x4x64x64xf32>, %b: tensor<2x64x64xf32>)
        -> tensor<2x4x64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2x4x64x64xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<2x4x64x64xf32>) -> tensor<2x4x64x64xf32>
    %c = linalg.contract indexing_maps = [#mapA, #mapB, #mapC]
        ins(%a, %b : tensor<2x4x64x64xf32>, tensor<2x64x64xf32>)
        outs(%f : tensor<2x4x64x64xf32>) -> tensor<2x4x64x64xf32>
    return %c : tensor<2x4x64x64xf32>
  }
}
"""

# A matvec (C[m] = A[m, k] * B[k]) with a relu epilogue: a GEMM whose result is 1D.
# Its single parallel (M) dim must still be tiled under the default 2D tile.
MATVEC = """
#map = affine_map<(d0) -> (d0)>
module {
  func.func @main(%m: tensor<128x64xf32>, %v: tensor<64xf32>) -> tensor<128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<128xf32>) -> tensor<128xf32>
    %mv = linalg.matvec ins(%m, %v : tensor<128x64xf32>, tensor<64xf32>)
        outs(%f : tensor<128xf32>) -> tensor<128xf32>
    %relu = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel"]}
        ins(%mv : tensor<128xf32>)
        outs(%e : tensor<128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    return %relu : tensor<128xf32>
  }
}
"""

# A batch matmul feeding a transposing consumer. The batch matmul is tiled
# [1, 32, 32, 0] (batch -> 1, M/N -> 32); the consumer reads the result with a
# permuted map (D[m, b, n] = C[b, m, n]), so tile-size propagation must remap the
# per-dimension tiles through the transpose: the batch unit tile has to land on
# the consumer's transposed batch dim, giving [32, 1, 32] (not a positional copy
# [1, 32, 32]).
PERMUTED = """
#cin = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#cout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%a: tensor<2x64x96xf32>, %b: tensor<2x96x128xf32>) -> tensor<64x2x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2x64x128xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    %mm = linalg.batch_matmul ins(%a, %b : tensor<2x64x96xf32>, tensor<2x96x128xf32>)
        outs(%f : tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    %eo = tensor.empty() : tensor<64x2x128xf32>
    %t = linalg.generic {indexing_maps = [#cin, #cout],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%mm : tensor<2x64x128xf32>)
        outs(%eo : tensor<64x2x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<64x2x128xf32>
    return %t : tensor<64x2x128xf32>
  }
}
"""

# A pack -> elementwise -> unpack chain. pack / unpack are fusion barriers: they
# stay as materialization boundaries (outside the tiled loop) while the
# elementwise op in between is tiled and fused on its own.
PACK_UNPACK = """
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %d = tensor.empty() : tensor<4x4x32x32xf32>
    %packed = linalg.pack %a inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %d
        : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
    %e2 = tensor.empty() : tensor<4x4x32x32xf32>
    %r = linalg.generic {indexing_maps = [#id4, #id4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%packed : tensor<4x4x32x32xf32>)
        outs(%e2 : tensor<4x4x32x32xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<4x4x32x32xf32>
    %out = tensor.empty() : tensor<128x128xf32>
    %u = linalg.unpack %r inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %out
        : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
    return %u : tensor<128x128xf32>
  }
}
"""


# Two chained elementwise ops annotated with conflicting tile sizes:
#   producer 32x32 -> consumer 64x64.
# Their tilings disagree on the shared tensor, so they are separate groups and
# must not be fused into one loop.
INCOMPAT_SPLIT = """
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

# A chain whose downstream op is pre-annotated with a conflicting tile size (64x64).
# Assigning + propagating from the upstream op (32x32) reaches the pre-annotated op
# and records a fusion boundary on it.
PRE_ANNOTATED = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<128x128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
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

# A broadcasting generic (bias[128] -> [64x128], input map drops the leading dim)
# feeding a relu.
BROADCAST = """
#bc_in = affine_map<(d0, d1) -> (d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%bias: tensor<128xf32>) -> tensor<64x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<64x128xf32>
    %bc = linalg.generic {indexing_maps = [#bc_in, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%bias : tensor<128xf32>)
        outs(%e0 : tensor<64x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<64x128xf32>
    %e1 = tensor.empty() : tensor<64x128xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%bc : tensor<64x128xf32>)
        outs(%e1 : tensor<64x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x128xf32>
    return %relu : tensor<64x128xf32>
  }
}
"""

# Named linalg ops (broadcast, transpose) do not expose the .inputs / .outputs
# accessors, only .input / .init. Selecting / computing / propagating tile sizes
# must go through linalg_inputs / linalg_outputs so these ops do not crash the
# analysis: bias[128] --broadcast--> relu --transpose.
NAMED_BROADCAST_TRANSPOSE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%bias: tensor<128xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e0 = tensor.empty() : tensor<64x128xf32>
    %bc = linalg.broadcast ins(%bias : tensor<128xf32>) outs(%e0 : tensor<64x128xf32>)
        dimensions = [0]
    %e1 = tensor.empty() : tensor<64x128xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%bc : tensor<64x128xf32>)
        outs(%e1 : tensor<64x128xf32>) {
    ^bb0(%i: f32, %o: f32):
      %c = arith.cmpf ugt, %i, %cst : f32
      %s = arith.select %c, %i, %cst : f32
      linalg.yield %s : f32
    } -> tensor<64x128xf32>
    %e2 = tensor.empty() : tensor<128x64xf32>
    %t = linalg.transpose ins(%relu : tensor<64x128xf32>) outs(%e2 : tensor<128x64xf32>)
        permutation = [1, 0]
    return %t : tensor<128x64xf32>
  }
}
"""

# A GEMM anchor whose result feeds a named linalg.transpose: tile sizes propagate
# from the matmul onto the named op, so propagation (propagate_through_value ->
# _map_for_value) must handle a named op without crashing.
GEMM_NAMED_TRANSPOSE = """
module {
  func.func @main(%a: tensor<64x64xf32>, %w: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %f = linalg.fill ins(%cst: f32) outs(%e: tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%a, %w : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f : tensor<64x64xf32>) -> tensor<64x64xf32>
    %e2 = tensor.empty() : tensor<64x64xf32>
    %t = linalg.transpose ins(%mm : tensor<64x64xf32>) outs(%e2 : tensor<64x64xf32>)
        permutation = [1, 0]
    return %t : tensor<64x64xf32>
  }
}
"""


# A GEMM anchors tiling; its tile sizes are propagated to the fill producer and
# the elementwise (bias, relu) consumers, then the whole group is fused.
# CHECK-LABEL: Test: mlp_assign_propagate
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("mlp_assign_propagate", MLP, assign_gemm)


# Tiling the fusion root (relu) and fusing producers pulls the fill, matmul and
# both elementwise ops into a single tiled scf.forall loop.
# CHECK-LABEL: Test: mlp_tile_and_fuse
# CHECK: scf.forall
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("mlp_tile_and_fuse", MLP, assign_gemm, tile_and_fuse)


# No GEMM: an elementwise op is anchored and its sizes propagated; the chain is
# tiled into 2D tiles and fused.
# CHECK-LABEL: Test: elementwise_tile_and_fuse
# CHECK: scf.forall ({{.*}}) = (0, 0) to (64, 256) step (32, 32)
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("elementwise_tile_and_fuse", ELTWISE, assign_elementwise, tile_and_fuse)


# Batch matmul: the batch dim is tiled by 1, M/N by the tile size, K untiled.
# CHECK-LABEL: Test: batch_matmul
# CHECK: linalg.batch_matmul {transform_ext.tile_sizes = array<i64: 1, 32, 32, 0>}
run("batch_matmul", BMM, assign_gemm)


# Reduction: an elementwise op anchors the group and propagation reaches the
# reduction consumer, tiling its parallel dim and leaving the reduction dim
# untiled (0). A standalone reduction is not an elementwise anchor.
# CHECK-LABEL: Test: reduction
# CHECK: iterator_types = ["parallel", "reduction"]
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
run("reduction", REDUCE, assign_elementwise)


# The elementwise anchor schedule covers named variants: a linalg.add anchor is
# annotated and its tiles propagate to the relu generic epilogue.
# CHECK-LABEL: Test: named_elementwise_anchor
# CHECK: linalg.add {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("named_elementwise_anchor", NAMED_ELTWISE, assign_elementwise)


# Two consecutive GEMMs: the shared relu is the epilogue of the first matmul, so
# it must be tiled to match that matmul's M/N tiles (2D, not the second matmul's
# M/K prologue tiles). The first matmul is followed by a fully-tiled relu.
# CHECK-LABEL: Test: two_consecutive_gemms_assign
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("two_consecutive_gemms_assign", TWO_GEMM, assign_gemm)


# GEMMs act as fusion barriers: two consecutive GEMMs are NOT fused together.
# Each GEMM + its epilogue forms its own scf.forall; the second reads the first
# loop's result through a slice (proving they are separate).
# CHECK-LABEL: Test: two_consecutive_gemms_fuse
# CHECK: %[[F0:.+]] = scf.forall
# CHECK: linalg.matmul
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall
# CHECK: tensor.extract_slice %[[F0]]
# CHECK: linalg.matmul
# CHECK: scf.forall.in_parallel
run("two_consecutive_gemms_fuse", TWO_GEMM, assign_gemm, tile_and_fuse)


# A 3-layer MLP: each layer (GEMM + epilogue) becomes its own fused scf.forall,
# chained through the loop results. Each forall fuses its own (shared) fill.
# CHECK-LABEL: Test: mlp_three_layers
# CHECK: %[[L0:.+]] = scf.forall
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK: %[[L1:.+]] = scf.forall
# CHECK: tensor.extract_slice %[[L0]]
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall
# CHECK: tensor.extract_slice %[[L1]]
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("mlp_three_layers", MLP3, assign_gemm, tile_and_fuse)


# Dynamic shapes: the matmul + relu group is tiled and fused into a single
# scf.forall whose bounds come from the runtime extents (tensor.dim).
# CHECK-LABEL: Test: dynamic_tile_and_fuse
# CHECK: %[[D0:.+]] = tensor.dim
# CHECK: %[[D1:.+]] = tensor.dim
# CHECK: scf.forall ({{.*}}) = (0, 0) to (%[[D0]], %[[D1]]) step (32, 32)
# CHECK: linalg.fill
# CHECK: linalg.matmul
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
run("dynamic_tile_and_fuse", DYN, assign_gemm, tile_and_fuse)


# High-dimensional contraction: only the innermost two parallel (M, N) output
# dims are tiled with the tile size; the outer parallel dims (batch and the
# outer M dim) get a unit tile and the reduction (K) dim is left untiled, just
# like GetTilingSizesOp.contract_tiles.
# CHECK-LABEL: Test: high_dim_contract
# CHECK: linalg.contract
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 32, 32, 0>
run("high_dim_contract", HIGH_DIM_CONTRACT, assign_gemm)


# 1D results: the matvec's single parallel (M) dim is tiled with the full tile size
# (its reduction K stays untiled) under the default 2D tile, and the group fuses into
# a single 1D scf.forall.
# CHECK-LABEL: Test: matvec_1d_tile_and_fuse
# CHECK: scf.forall ({{.*}}) = (0) to (128) step (32)
# CHECK: linalg.fill
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32>
# CHECK: linalg.matvec
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 0>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32>
# CHECK: scf.forall.in_parallel
run("matvec_1d_tile_and_fuse", MATVEC, assign_gemm, tile_and_fuse_keep)


# Propagation across a permuted (transposed) shared tensor: the batch matmul is
# tiled [1, 32, 32, 0] and the consumer, which reads the result transposed, must
# receive the per-dimension tiles remapped through its indexing map -> the batch
# unit tile lands on the consumer's transposed batch dim, giving [32, 1, 32].
# CHECK-LABEL: Test: permuted_propagation
# CHECK: linalg.batch_matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 32, 32, 0>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 1, 32>
run("permuted_propagation", PERMUTED, assign_gemm)


# pack / unpack are fusion barriers: they are not propagation anchors and are not
# fused into the tiled loop. The elementwise op between them is tiled on its own,
# with the pack left before and the unpack after the resulting scf.forall.
# CHECK-LABEL: Test: pack_unpack_barrier
# CHECK: linalg.pack
# CHECK: scf.forall
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK: linalg.unpack
run("pack_unpack_barrier", PACK_UNPACK, assign_elementwise, tile_and_fuse)


# Conflicting pre-annotated tile sizes (32x32 producer, 64x64 consumer) are kept
# apart: each op is tiled in its own loop (32-step and 64-step).
# CHECK-LABEL: Test: incompatible_annotations_split
# CHECK: %[[L0:.+]] = scf.forall ({{.*}}) = (0, 0) to (128, 128) step (32, 32)
# CHECK: math.exp
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall ({{.*}}) = (0, 0) to (128, 128) step (64, 64)
# CHECK: tensor.extract_slice %[[L0]]
# CHECK: math.sqrt
# CHECK: scf.forall.in_parallel
run("incompatible_annotations_split", INCOMPAT_SPLIT, tile_and_fuse)


# A downstream op pre-annotated (64x64) conflicts with the propagated tiling
# (32x32). Propagation records the split by marking the conflicting op with a
# fusion attribute.
# CHECK-LABEL: Test: boundary_marked_during_propagation
# CHECK: tile_sizes = array<i64: 32, 32>
# CHECK: transform_ext.fusion_boundary
# CHECK-SAME: tile_sizes = array<i64: 64, 64>
run("boundary_marked_during_propagation", PRE_ANNOTATED, assign_elementwise)


# The pre-annotated conflict is honored end-to-end: the two ops are tiled into
# separate loops (32-step then 64-step) rather than fused.
# CHECK-LABEL: Test: boundary_split_tile_and_fuse
# CHECK: %[[B0:.+]] = scf.forall ({{.*}}) step (32, 32)
# CHECK: math.exp
# CHECK: scf.forall.in_parallel
# CHECK: scf.forall ({{.*}}) step (64, 64)
# CHECK: tensor.extract_slice %[[B0]]
# CHECK: math.sqrt
# CHECK: scf.forall.in_parallel
run("boundary_split_tile_and_fuse", PRE_ANNOTATED, assign_elementwise, tile_and_fuse)


# A broadcasting generic producer stays fused with its consumer: both land in a
# single scf.forall (the broadcast reads a 1D slice of the bias), consistent with
# FuseOp fusing broadcast producers. The untiled/broadcast dim is treated as a
# wildcard by the compatibility check, so it never forces a split.
# CHECK-LABEL: Test: broadcast_stays_fused
# CHECK: scf.forall ({{.*}}) = (0, 0) to (64, 128) step (32, 32)
# CHECK: tensor.extract_slice %arg0[%{{.*}}] [32] [1]
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK: scf.forall.in_parallel
# CHECK-NOT: scf.forall
run("broadcast_stays_fused", BROADCAST, assign_elementwise, tile_and_fuse)


# Named linalg ops annotation.
# CHECK-LABEL: Test: named_broadcast_transpose_anchor
# CHECK: linalg.broadcast
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.transpose
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("named_broadcast_transpose_anchor", NAMED_BROADCAST_TRANSPOSE, assign_elementwise)


# Propagation onto a named op.
# CHECK-LABEL: Test: gemm_named_transpose_propagate
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.transpose
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("gemm_named_transpose_propagate", GEMM_NAMED_TRANSPOSE, assign_gemm)


# A larger elementwise payload for demonstrating repeated tiling rounds.
RETILE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<128x256xf32>)
        outs(%0 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<128x256xf32>
    return %1 : tensor<128x256xf32>
  }
}
"""


# Fusion consumes the annotations, so by default they are cleared from the ops
# now inside the generated loop, leaving no stale tile and fuse annotations.
# CHECK-LABEL: Test: clears_annotations_after_fuse
# CHECK: scf.forall
# CHECK-NOT: transform_ext.tile_sizes
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK: scf.forall.in_parallel
run("clears_annotations_after_fuse", MLP, assign_gemm, tile_and_fuse)


# Clearing the annotations after each round leaves a clean slate, so a second
# assign + tile-and-fuse round tiles the already-tiled ops again: a 64-wide outer
# loop from round one, a 32-wide inner loop from round two, and no leftover
# annotations that could confuse the second assignment.
# CHECK-LABEL: Test: retile_after_clear
# CHECK: scf.forall ({{.*}}) = (0, 0) to (128, 256) step (64, 64)
# CHECK: scf.forall ({{.*}}) = (0, 0) to (64, 64) step (32, 32)
# CHECK-NOT: transform_ext.tile_sizes
# CHECK-NOT: transform_ext.fusion_boundary
# CHECK: scf.forall.in_parallel
run(
    "retile_after_clear",
    RETILE,
    assign_elementwise64,
    tile_and_fuse,
    assign_elementwise,
    tile_and_fuse,
)


# Non-forall tiling: the elementwise chain tiled with use_forall=False produces
# two NESTED scf.for loops instead of one scf.forall. Both ops are fused into the
# innermost loop, and post-fusion clearing must still reach them -- the fusion
# loop handle covers the whole nest -- so no stale annotations remain anywhere in
# the loop nest.
# CHECK-LABEL: Test: elementwise_scf_for_nested_clears
# CHECK: scf.for
# CHECK-NOT: transform_ext
# CHECK: scf.for
# CHECK-NOT: transform_ext
# CHECK: math.exp
# CHECK-NOT: transform_ext
# CHECK: arith.select
# CHECK-NOT: transform_ext
# CHECK: scf.yield
run("elementwise_scf_for_nested_clears", ELTWISE, assign_elementwise, tile_and_fuse_for)
