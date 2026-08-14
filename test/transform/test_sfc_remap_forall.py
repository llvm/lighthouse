# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.utils.sfc import gilbert2d
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.schedule.sfc import remap as remap_sfc


def run(name: str, payload_str: str, *schedules):
    """Parse a payload, apply the given schedules in order and print it."""
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        modules = []
        for make_schedule in schedules:
            sched = make_schedule()
            modules.append(sched)
            sched.body.operations[0].apply(payload.operation)
        payload.operation.verify()
        print(payload)


def remap_matmul():
    with schedule_boilerplate() as (schedule, named):
        target = lh_transform.match_op(named.bodyTarget, "linalg.matmul")
        transform_ext.sfc_remap_forall(target)
        transform.yield_()
    return schedule


def remap_fill():
    """Apply SFC remap on a non-contraction op handle."""
    with schedule_boilerplate() as (schedule, named):
        target = lh_transform.match_op(named.bodyTarget, "linalg.fill")
        transform_ext.sfc_remap_forall(target)
        transform.yield_()
    return schedule


for width, height in ((1, 1), (2, 3), (15, 12)):
    points = list(gilbert2d(width, height))
    assert len(points) == width * height
    assert len(set(points)) == len(points)
    assert set(points) == {(x, y) for x in range(width) for y in range(height)}


# An 8x2 by 2x4 GEMM tiled 1x1, so the C-tile grid is 8x4 (Mb x Nb). The A/B
# slice offsets on the two forall IVs must resolve to SFC lookup tables and a
# single flattened 1D loop over all 32 tiles.
MATMUL = """
module {
  func.func @main(%a: tensor<8x2xf32>, %b: tensor<2x4xf32>) -> tensor<8x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<8x4xf32>
    %result = scf.forall (%i, %j) in (8, 4) shared_outs(%out = %empty)
        -> (tensor<8x4xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<8x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x4xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<8x4xf32>
      }
    }
    return %result : tensor<8x4xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_zero_fill
# CHECK: arith.constant dense<[0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 6, 6, 7]> : tensor<32xi64>
# CHECK: arith.constant dense<[0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0]> : tensor<32xi64>
# CHECK: scf.forall ({{.*}}) in (32)
# CHECK: tensor.extract
# CHECK: linalg.fill
# CHECK: linalg.matmul
run("matmul_zero_fill", MATMUL, remap_matmul)


# Nested 1D IV is unrelated to the tile offsets; the outer forall is relevant.
NESTED_MATMUL = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %result = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %inner_empty = tensor.empty() : tensor<1x1xf32>
      %inner = scf.forall (%k) in (1) shared_outs(%inner_out = %inner_empty)
          -> (tensor<1x1xf32>) {
        %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
            : tensor<2x2xf32> to tensor<1x2xf32>
        %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
            : tensor<2x2xf32> to tensor<2x1xf32>
        %init_empty = tensor.empty() : tensor<1x1xf32>
        %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
            -> tensor<1x1xf32>
        %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
            outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %product into %inner_out[0, 0] [1, 1] [1, 1]
              : tensor<1x1xf32> into tensor<1x1xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %inner into %out[%i, %j] [1, 1] [1, 1]
            : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: nested_matmul
# CHECK: scf.forall ({{.*}}) in (4)
# CHECK: scf.forall ({{.*}}) in (1)
run("nested_matmul", NESTED_MATMUL, remap_matmul)


# IV declaration order sets orientation; swapping it preserves coverage.
SWAPPED_MATMUL = """
module {
  func.func @main(%a: tensor<8x2xf32>, %b: tensor<2x4xf32>) -> tensor<8x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<8x4xf32>
    %result = scf.forall (%j, %i) in (4, 8) shared_outs(%out = %empty)
        -> (tensor<8x4xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<8x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x4xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
            : tensor<1x1xf32> into tensor<8x4xf32>
      }
    }
    return %result : tensor<8x4xf32>
  }
}
"""

# CHECK-LABEL: Test: swapped_matmul
# CHECK: scf.forall ({{.*}}) in (32)
run("swapped_matmul", SWAPPED_MATMUL, remap_matmul)


# Both matmuls share one parent forall. The transform must gather both relevant
# operations before replacing that parent and must clone the loop body once.
TWO_MATMULS_ONE_FORALL = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>,
      %c: tensor<2x2xf32>, %d: tensor<2x2xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %empty0 = tensor.empty() : tensor<2x2xf32>
    %empty1 = tensor.empty() : tensor<2x2xf32>
    %result0, %result1 = scf.forall (%i, %j) in (2, 2)
        shared_outs(%out0 = %empty0, %out1 = %empty1)
        -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %as0 = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs0 = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty0 = tensor.empty() : tensor<1x1xf32>
      %init0 = linalg.fill ins(%cst : f32) outs(%init_empty0 : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product0 = linalg.matmul ins(%as0, %bs0 : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init0 : tensor<1x1xf32>) -> tensor<1x1xf32>
      %as1 = tensor.extract_slice %c[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs1 = tensor.extract_slice %d[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty1 = tensor.empty() : tensor<1x1xf32>
      %init1 = linalg.fill ins(%cst : f32) outs(%init_empty1 : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product1 = linalg.matmul ins(%as1, %bs1 : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init1 : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product0 into %out0[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
        tensor.parallel_insert_slice %product1 into %out1[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result0, %result1 : tensor<2x2xf32>, tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: two_matmuls_one_forall
# CHECK-COUNT-1: scf.forall ({{.*}}) in (4)
# CHECK-COUNT-2: linalg.matmul
run("two_matmuls_one_forall", TWO_MATMULS_ONE_FORALL, remap_matmul)

# Exercise the production schedule, which must match only contractions before
# replacing their parent foralls. Matching every linalg op would leave sibling
# fill/elementwise handles pointing into an erased forall.
# CHECK-LABEL: Test: matmul_production_schedule
# CHECK: scf.forall ({{.*}}) in (32)
# CHECK: linalg.matmul
run("matmul_production_schedule", MATMUL, remap_sfc)


# A GEMM with an elementwise prologue (scaling the A slice) and an elementwise
# epilogue (doubling the accumulator). Offsets reach the forall IVs through the
# wrapping linalg.generic ops, so SFC-remapping must still trace through them
# and flatten the loop, keeping the prologue/epilogue ops in the cloned body.
PROLOGUE_EPILOGUE = """
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %scale = arith.constant 2.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %result = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %prologue = linalg.generic {indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%as : tensor<1x2xf32>) outs(%as : tensor<1x2xf32>) {
      ^bb0(%in: f32, %o: f32):
        %scaled = arith.mulf %in, %scale : f32
        linalg.yield %scaled : f32
      } -> tensor<1x2xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%prologue, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      %epilogue = linalg.generic {indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%product : tensor<1x1xf32>) outs(%product : tensor<1x1xf32>) {
      ^bb0(%in: f32, %o: f32):
        %doubled = arith.addf %in, %in : f32
        linalg.yield %doubled : f32
      } -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %epilogue into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_prologue_epilogue
# CHECK: arith.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
# CHECK: arith.constant dense<[0, 0, 1, 1]> : tensor<4xi64>
# CHECK: scf.forall ({{.*}}) in (4)
# CHECK: arith.mulf
# CHECK: linalg.matmul
# CHECK: arith.addf
run("matmul_prologue_epilogue", PROLOGUE_EPILOGUE, remap_matmul)


# Two dependent contractions must be collected before any parent loop is
# replaced. The transform must not revisit a target handle after mutation.
MULTIPLE_DEPENDENT_MATMULS = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %first = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    %second = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %first[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %second : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: multiple_dependent_matmuls
# CHECK-COUNT-2: scf.forall ({{.*}}) in (4)
run("multiple_dependent_matmuls", MULTIPLE_DEPENDENT_MATMULS, remap_matmul)


# The matmul reads the same %a[0, 0]/%b[0, 0] tile every iteration, so its
# operands cannot be traced to the forall's induction variables. SFC-remapping
# must skip it, leaving the 2D forall untouched.
INDEPENDENT_MATMUL = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %result = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %a[0, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, 0] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_independent_of_forall
# CHECK: scf.forall ({{.*}}, {{.*}}) in (2, 2)
# CHECK-NOT: arith.constant dense
# CHECK-NOT: tensor.extract %
run("matmul_independent_of_forall", INDEPENDENT_MATMUL, remap_matmul)


# The enclosing forall is 3D. The transform requires an enclosing 2D forall,
# so it must skip rewriting.
THREED_FORALL_MATMUL = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2x2xf32>
    %result = scf.forall (%k, %i, %j) in (2, 2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2x2xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%k, %i, %j] [1, 1, 1] [1, 1, 1]
          : tensor<1x1xf32> into tensor<2x2x2xf32>
      }
    }
    return %result : tensor<2x2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_in_3d_forall
# CHECK: scf.forall ({{.*}}, {{.*}}, {{.*}}) in (2, 2, 2)
# CHECK-NOT: arith.constant dense
run("matmul_in_3d_forall", THREED_FORALL_MATMUL, remap_matmul)


# The matmul depends on only one forall IV (%i), with %j constant in B-slices.
# The transform requires both IVs to be relevant, so it must skip rewriting.
ONE_IV_MATMUL = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %result = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, 0] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_one_relevant_iv
# CHECK: scf.forall ({{.*}}, {{.*}}) in (2, 2)
# CHECK-NOT: arith.constant dense
run("matmul_one_relevant_iv", ONE_IV_MATMUL, remap_matmul)


# The transform may be invoked on non-contraction handles, but must not rewrite
# the parent forall in that case.
# CHECK-LABEL: Test: non_contraction_target
# CHECK: scf.forall ({{.*}}, {{.*}}) in (8, 4)
# CHECK-NOT: arith.constant dense
run("non_contraction_target", MATMUL, remap_fill)


# Dynamic loop bounds are unsupported by this transform, so the loop should
# remain unchanged.
DYNAMIC_BOUNDS_MATMUL = """
module {
  func.func @main(%m: index, %n: index, %a: tensor<2x2xf32>, %b: tensor<2x2xf32>)
      -> tensor<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %result = scf.forall (%i, %j) in (%m, %n) shared_outs(%out = %empty)
        -> (tensor<2x2xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<2x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x2xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<2x2xf32>
      }
    }
    return %result : tensor<2x2xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_dynamic_bounds
# CHECK: scf.forall ({{.*}}, {{.*}}) in ({{.*}}, {{.*}})
# CHECK-NOT: arith.constant dense
run("matmul_dynamic_bounds", DYNAMIC_BOUNDS_MATMUL, remap_matmul)


# Non-zero lower bounds are outside the supported shape, so the loop should
# remain unchanged.
NONZERO_LOWER_BOUNDS_MATMUL = """
module {
  func.func @main(%a: tensor<4x2xf32>, %b: tensor<2x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %result = scf.forall (%i, %j) = (1, 0) to (3, 2) step (1, 1)
        shared_outs(%out = %empty) -> (tensor<4x4xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<4x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x4xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<4x4xf32>
      }
    }
    return %result : tensor<4x4xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_nonzero_lower_bounds
# CHECK: scf.forall ({{.*}}, {{.*}}) = (1, 0) to (3, 2) step (1, 1)
# CHECK-NOT: arith.constant dense
run("matmul_nonzero_lower_bounds", NONZERO_LOWER_BOUNDS_MATMUL, remap_matmul)


# Non-unit step sizes are outside the supported shape, so the loop should
# remain unchanged.
NONUNIT_STEP_MATMUL = """
module {
  func.func @main(%a: tensor<4x2xf32>, %b: tensor<2x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %result = scf.forall (%i, %j) = (0, 0) to (4, 4) step (2, 1)
        shared_outs(%out = %empty) -> (tensor<4x4xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<4x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x4xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<4x4xf32>
      }
    }
    return %result : tensor<4x4xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_nonunit_step
# CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (4, 4) step (2, 1)
# CHECK-NOT: arith.constant dense
run("matmul_nonunit_step", NONUNIT_STEP_MATMUL, remap_matmul)


# Dynamic step operands are unsupported by this transform, so the loop should
# remain unchanged.
DYNAMIC_STEP_MATMUL = """
module {
  func.func @main(%si: index, %a: tensor<4x2xf32>, %b: tensor<2x4xf32>)
      -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %result = scf.forall (%i, %j) = (0, 0) to (4, 4) step (%si, 1)
        shared_outs(%out = %empty) -> (tensor<4x4xf32>) {
      %as = tensor.extract_slice %a[%i, 0] [1, 2] [1, 1]
          : tensor<4x2xf32> to tensor<1x2xf32>
      %bs = tensor.extract_slice %b[0, %j] [2, 1] [1, 1]
          : tensor<2x4xf32> to tensor<2x1xf32>
      %init_empty = tensor.empty() : tensor<1x1xf32>
      %init = linalg.fill ins(%cst : f32) outs(%init_empty : tensor<1x1xf32>)
          -> tensor<1x1xf32>
      %product = linalg.matmul ins(%as, %bs : tensor<1x2xf32>, tensor<2x1xf32>)
          outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %product into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<4x4xf32>
      }
    }
    return %result : tensor<4x4xf32>
  }
}
"""

# CHECK-LABEL: Test: matmul_dynamic_step
# CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (4, 4) step ({{.*}}, 1)
# CHECK-NOT: arith.constant dense
run("matmul_dynamic_step", DYNAMIC_STEP_MATMUL, remap_matmul)
