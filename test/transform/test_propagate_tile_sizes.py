# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import propagate_tile_sizes
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, build_schedule):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


def build_propagate_schedule(anchor_op_name: str, propagate_through_loops: bool):
    with schedule_boilerplate() as (sched, named_seq):
        anchors = lh_transform.match_op(named_seq.bodyTarget, anchor_op_name)
        propagate_tile_sizes(anchors, propagate_through_loops=propagate_through_loops)
        transform.yield_()
    return sched


# Forward propagation: matmul -> elementwise consumer
FORWARD_SIMPLE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>
  return %relu : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: forward_simple
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# Elementwise consumer picks up the M and N tile sizes from the matmul result.
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run(
    "forward_simple",
    FORWARD_SIMPLE,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Backward propagation: fill is tiled to match the matmul it initialises
BACKWARD_FILL = """
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: backward_fill
# The fill (prologue) must be annotated with [32, 32] to match the matmul output tile.
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
run(
    "backward_fill",
    BACKWARD_FILL,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Forward + backward: both epilogue and prologue get tiled
FORWARD_AND_BACKWARD = """
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>,
                %bias: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %add = linalg.add
      ins(%mm, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %add : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: forward_and_backward
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.add {transform_ext.tile_sizes = array<i64: 32, 32>}
# The add (epilogue) receives tile sizes propagated forward from the matmul.
run(
    "forward_and_backward",
    FORWARD_AND_BACKWARD,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Forward propagation stops at tensor.cast (negative test)
CAST_STOPS_FORWARD = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %casted = tensor.cast %mm : tensor<128x128xf32> to tensor<?x?xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %out_cast = tensor.cast %out : tensor<128x128xf32> to tensor<?x?xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%casted : tensor<?x?xf32>)
      outs(%out_cast : tensor<?x?xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<?x?xf32>
  %result = tensor.cast %relu : tensor<?x?xf32> to tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
"""

# CHECK-LABEL: Test: cast_stops_forward
# tensor.cast is opaque; propagation stops there, the consumer stays unannotated.
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic {indexing_maps
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
run(
    "cast_stops_forward",
    CAST_STOPS_FORWARD,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Forward propagation through an scf.for loop: annotated matmul inside the loop
# yields its result; the consumer outside is annotated via the for-result bridge.
FORWARD_THROUGH_FOR = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
                %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
    %col = tensor.extract_slice %arg0[0, %arg3] [128, 1] [1, 1] : tensor<128x128xf32> to tensor<128x1xf32>
    %row = tensor.extract_slice %arg1[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
    %update = linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
        ins(%col, %row : tensor<128x1xf32>, tensor<1x128xf32>)
        outs(%arg4 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %update : tensor<128x128xf32>
  }
  %out = tensor.empty() : tensor<128x128xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%in: f32, %o: f32):
    %cst = arith.constant 0.0 : f32
    %r = arith.maxnumf %in, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>
  return %relu : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: forward_through_for
# The forward bridge traverses scf.for: matmul yields into the loop, and the
# loop result's consumer (relu) gets annotated with the translated tile sizes.
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 2, 8>
run(
    "forward_through_for",
    FORWARD_THROUGH_FOR,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=True),
)

# CHECK-LABEL: Test: loop_is_barrier_by_default
# Without opting in, scf.for stops propagation: the consumer stays unannotated.
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
run(
    "loop_is_barrier_by_default",
    FORWARD_THROUGH_FOR,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Chained forward: two consumers in sequence both get annotated
CHAINED_FORWARD = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mid = tensor.empty() : tensor<128x128xf32>
  %act1 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<128x128xf32>)
      outs(%mid : tensor<128x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %act2 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%act1 : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    linalg.yield %i : f32
  } -> tensor<128x128xf32>
  return %act2 : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: chained_forward
# Propagation cascades: mm -> act1 -> act2 all end up annotated.
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 32, 32>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32>
run(
    "chained_forward",
    CHAINED_FORWARD,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Barrier stops propagation: a contraction op between two elementwise ops
# acts as a fusion barrier; propagation does not cross it.
BARRIER_STOPS_PROPAGATION = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>,
                %x: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %epilogue = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"],
      transform_ext.tile_sizes = array<i64: 32, 32>}
      ins(%mm : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>
  return %epilogue : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: barrier_stops_propagation
# Backward propagation from the epilogue reaches the matmul (a contraction /
# barrier), but must not annotate it, and must not continue to the fill either.
# CHECK: linalg.fill ins
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.matmul ins
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32>
run(
    "barrier_stops_propagation",
    BARRIER_STOPS_PROPAGATION,
    lambda: build_propagate_schedule("linalg.generic", propagate_through_loops=False),
)


# Pre-annotated op acts as barrier: already-annotated downstream op with
# different tile sizes prevents re-annotation and marks a fusion boundary.
PREANNOTED_BARRIER = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
      ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out = tensor.empty() : tensor<128x128xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"],
      transform_ext.tile_sizes = array<i64: 64, 64>}
      ins(%mm : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>
  return %relu : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: preannotated_barrier
# The downstream generic already carries tile sizes that disagree with matmul's [32, 32].
# Propagation must not overwrite it; the conflicting consumer is marked as a boundary.
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 64, 64>
run(
    "preannotated_barrier",
    PREANNOTED_BARRIER,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Reduction dims are never tiled during propagation
REDUCTION_UNTILED = """
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#reduce = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @main(%a: tensor<128x64x32xf32>) -> tensor<128x64xf32> {
  %cst = arith.constant 0.0 : f32
  %src_empty = tensor.empty() : tensor<128x64x32xf32>
  %src = linalg.generic {indexing_maps = [#map3, #map3],
      iterator_types = ["parallel", "parallel", "parallel"],
      transform_ext.tile_sizes = array<i64: 32, 32, 8>}
      ins(%a : tensor<128x64x32xf32>)
      outs(%src_empty : tensor<128x64x32xf32>) {
  ^bb0(%i: f32, %o: f32):
    linalg.yield %i : f32
  } -> tensor<128x64x32xf32>
  %red_empty = tensor.empty() : tensor<128x64xf32>
  %red_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
  %reduce = linalg.generic {
      indexing_maps = [#map3, #reduce],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%src : tensor<128x64x32xf32>)
      outs(%red_init : tensor<128x64xf32>) {
  ^bb0(%i: f32, %o: f32):
    %sum = arith.addf %i, %o : f32
    linalg.yield %sum : f32
  } -> tensor<128x64xf32>
  return %reduce : tensor<128x64xf32>
}
"""

# CHECK-LABEL: Test: reduction_untiled
# Propagation from the elementwise source reaches the reduction.
# The reduction dim (d2) must stay untiled (0) when annotating the source.
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32, 8>
# CHECK: linalg.fill {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32, 0>
run(
    "reduction_untiled",
    REDUCTION_UNTILED,
    lambda: build_propagate_schedule("linalg.generic", propagate_through_loops=False),
)


# Backward propagation through scf.for: the fill initialising the iter arg
# is reached by bridging the iter arg back to its initial value on the for.
BACKWARD_THROUGH_FOR = """
func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %0 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %fill) -> (tensor<128x128xf32>) {
    %col = tensor.extract_slice %arg0[0, %arg3] [128, 1] [1, 1] : tensor<128x128xf32> to tensor<128x1xf32>
    %row = tensor.extract_slice %arg1[%arg3, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
    %update = linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
        ins(%col, %row : tensor<128x1xf32>, tensor<1x128xf32>)
        outs(%arg4 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %update : tensor<128x128xf32>
  }
  return %0 : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: backward_through_for
# CHECK: linalg.fill {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32>
# CHECK: linalg.matmul {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32, 0>
run(
    "backward_through_for",
    BACKWARD_THROUGH_FOR,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=True),
)

# CHECK-LABEL: Test: backward_loop_is_barrier_by_default
# Without opting in, the iter-arg bridge is off too: the fill stays unannotated.
# CHECK: linalg.fill ins
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.matmul {{{.*}}transform_ext.tile_sizes = array<i64: 32, 32, 0>
run(
    "backward_loop_is_barrier_by_default",
    BACKWARD_THROUGH_FOR,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=False),
)


# Nested scf.for loops: forward propagation cascades through two yields
# inner matmul -> inner scf.yield -> outer scf.yield -> consumer
FORWARD_NESTED_FORS = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
                %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %outer = scf.for %i = %c0 to %c128 step %c1 iter_args(%outer_carry = %arg2) -> (tensor<128x128xf32>) {
    %col = tensor.extract_slice %arg0[0, %i] [128, 1] [1, 1] : tensor<128x128xf32> to tensor<128x1xf32>
    %inner = scf.for %j = %c0 to %c128 step %c1 iter_args(%inner_carry = %outer_carry) -> (tensor<128x128xf32>) {
      %row = tensor.extract_slice %arg1[%i, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %update = linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
          ins(%col, %row : tensor<128x1xf32>, tensor<1x128xf32>)
          outs(%inner_carry : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %update : tensor<128x128xf32>
    }
    scf.yield %inner : tensor<128x128xf32>
  }
  %out = tensor.empty() : tensor<128x128xf32>
  %consumer = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%outer : tensor<128x128xf32>)
      outs(%out : tensor<128x128xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<128x128xf32>
  return %consumer : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: forward_nested_fors
# Forward bridge cascades: mm -> inner scf.yield -> inner result -> outer scf.yield
# -> outer result -> consumer, annotating the consumer with translated tile sizes.
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 2, 8>
run(
    "forward_nested_fors",
    FORWARD_NESTED_FORS,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=True),
)


# Loop carrying two values: only the chain through the second iter arg is annotated.
# Exercises the index mapping of both bridges (yield operand -> loop result, and
# iter arg -> its init operand), which a single-iter-arg loop cannot distinguish.
MULTI_ITER_ARGS = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<128x128xf32>
  %fill0 = linalg.fill ins(%zero : f32) outs(%e0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %e1 = tensor.empty() : tensor<128x128xf32>
  %fill1 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %r:2 = scf.for %i = %c0 to %c128 step %c1 iter_args(%a0 = %fill0, %a1 = %fill1)
      -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %col = tensor.extract_slice %arg0[0, %i] [128, 1] [1, 1] : tensor<128x128xf32> to tensor<128x1xf32>
    %row = tensor.extract_slice %arg1[%i, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
    %update = linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
        ins(%col, %row : tensor<128x1xf32>, tensor<1x128xf32>)
        outs(%a1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %a0, %update : tensor<128x128xf32>, tensor<128x128xf32>
  }
  %o0 = tensor.empty() : tensor<128x128xf32>
  %use0 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%r#0 : tensor<128x128xf32>)
      outs(%o0 : tensor<128x128xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<128x128xf32>
  %o1 = tensor.empty() : tensor<128x128xf32>
  %use1 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%r#1 : tensor<128x128xf32>)
      outs(%o1 : tensor<128x128xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<128x128xf32>
  return %use0, %use1 : tensor<128x128xf32>, tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: multi_iter_args_index_mapping
# The first iter arg is only forwarded, so neither its init fill nor its consumer
# may be annotated; only the second chain (fill1 -> matmul -> %r#1) is.
# CHECK: linalg.fill ins
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.fill {transform_ext.tile_sizes = array<i64: 2, 8>}
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 2, 8>
run(
    "multi_iter_args_index_mapping",
    MULTI_ITER_ARGS,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=True),
)


# Only scf.for is bridged; scf.forall stays a barrier even when opted in.
FORALL_STAYS_BARRIER = """
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<128x64xf32>, %arg1: tensor<64x128xf32>,
                %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = scf.forall (%i) in (4) shared_outs(%o = %arg2) -> (tensor<128x128xf32>) {
    %sl = tensor.extract_slice %o[0, 0] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
    %mm = linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
        ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x128xf32>)
        outs(%sl : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %o[0, 0] [128, 128] [1, 1]
          : tensor<128x128xf32> into tensor<128x128xf32>
    }
  }
  %e = tensor.empty() : tensor<128x128xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<128x128xf32>)
      outs(%e : tensor<128x128xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<128x128xf32>
  return %relu : tensor<128x128xf32>
}
"""

# CHECK-LABEL: Test: forall_stays_barrier
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 2, 8, 0>}
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
run(
    "forall_stays_barrier",
    FORALL_STAYS_BARRIER,
    lambda: build_propagate_schedule("linalg.matmul", propagate_through_loops=True),
)


# An anchor with two propagatable producers and two consumers: every neighbor is
# claimed, not just the first one found in each direction. The second consumer
# reads the anchor transposed, so its tiles must be remapped, not copied.
FAN_IN_OUT = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#tin = affine_map<(d0, d1) -> (d1, d0)>
func.func @main(%a: tensor<128x256xf32>, %b: tensor<128x256xf32>)
    -> (tensor<128x256xf32>, tensor<256x128xf32>) {
  %cst = arith.constant 0.0 : f32
  %ep0 = tensor.empty() : tensor<128x256xf32>
  %p0 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<128x256xf32>) outs(%ep0 : tensor<128x256xf32>) {
  ^bb0(%i: f32, %o: f32):
    %e = math.exp %i : f32
    linalg.yield %e : f32
  } -> tensor<128x256xf32>
  %ep1 = tensor.empty() : tensor<128x256xf32>
  %p1 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%b : tensor<128x256xf32>) outs(%ep1 : tensor<128x256xf32>) {
  ^bb0(%i: f32, %o: f32):
    %e = math.exp %i : f32
    linalg.yield %e : f32
  } -> tensor<128x256xf32>
  %eo = tensor.empty() : tensor<128x256xf32>
  %anchor = linalg.add {transform_ext.tile_sizes = array<i64: 16, 32>}
      ins(%p0, %p1 : tensor<128x256xf32>, tensor<128x256xf32>)
      outs(%eo : tensor<128x256xf32>) -> tensor<128x256xf32>
  %ec0 = tensor.empty() : tensor<128x256xf32>
  %use0 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%anchor : tensor<128x256xf32>) outs(%ec0 : tensor<128x256xf32>) {
  ^bb0(%i: f32, %o: f32):
    %r = arith.maxnumf %i, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x256xf32>
  %ec1 = tensor.empty() : tensor<256x128xf32>
  %use1 = linalg.generic {indexing_maps = [#tin, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%anchor : tensor<128x256xf32>) outs(%ec1 : tensor<256x128xf32>) {
  ^bb0(%i: f32, %o: f32):
    %n = arith.negf %i : f32
    linalg.yield %n : f32
  } -> tensor<256x128xf32>
  return %use0, %use1 : tensor<128x256xf32>, tensor<256x128xf32>
}
"""

# CHECK-LABEL: Test: fan_in_and_fan_out
# Both producers (backward) and both consumers (forward) get the anchor's tiling.
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 16, 32>
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 16, 32>
# CHECK: linalg.add {transform_ext.tile_sizes = array<i64: 16, 32>}
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 16, 32>
# The transposing consumer tiles the same tensor dims, so its loop-order sizes swap.
# CHECK: linalg.generic {{{.*}}transform_ext.tile_sizes = array<i64: 32, 16>
run(
    "fan_in_and_fan_out",
    FAN_IN_OUT,
    lambda: build_propagate_schedule("linalg.add", propagate_through_loops=False),
)
