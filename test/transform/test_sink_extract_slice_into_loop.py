# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str):
    """Parse a payload, sink extract_slices into scf.for loops, and print it."""
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        with schedule_boilerplate() as (sched, named_seq):
            loops = lh_transform.match_op(named_seq.bodyTarget, "scf.for")
            transform_ext.sink_extract_slice_into_loop(loops)
            transform.yield_()
        sched.body.operations[0].apply(payload.operation)
        payload.operation.verify()
        print(payload)


# The loop carries a [8x4] slice extracted from %arg0 at column offset 4 and
# writes it straight back. Sinking makes the loop carry the full [8x16] tensor
# and composes the body offsets (column 0 -> column 4).
MATCHING = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    %ext = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x16xf32> to tensor<8x4xf32>
    %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %ext)
        -> (tensor<8x4xf32>) {
      %s = tensor.extract_slice %it[%i, 0] [2, 4] [1, 1]
          : tensor<8x4xf32> to tensor<2x4xf32>
      %f = linalg.fill ins(%cst : f32) outs(%s : tensor<2x4xf32>)
          -> tensor<2x4xf32>
      %ins = tensor.insert_slice %f into %it[%i, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> into tensor<8x4xf32>
      scf.yield %ins : tensor<8x4xf32>
    }
    %out = tensor.insert_slice %loop into %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
"""

# CHECK-LABEL: Test: matching
# The surrounding extract/insert pair is gone: no [8, 4] slice of the 8x16 tensor.
# CHECK-NOT: tensor.extract_slice %arg0[0, 4] [8, 4]
# The loop now carries the full tensor directly.
# CHECK: scf.for %[[I:.*]] = %{{.*}} iter_args(%[[IT:.*]] = %arg0) -> (tensor<8x16xf32>)
# CHECK: tensor.extract_slice %[[IT]][%[[I]], 4] [2, 4] [1, 1] : tensor<8x16xf32> to tensor<2x4xf32>
# CHECK: tensor.insert_slice %{{.*}} into %[[IT]][%[[I]], 4] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<8x16xf32>
# CHECK: scf.yield %{{.*}} : tensor<8x16xf32>
# CHECK: return %[[LOOP:.*]] : tensor<8x16xf32>
run("matching", MATCHING)


# The loop init is a tensor.empty, not an extract_slice, so the loop does not
# match the pattern and must be left unchanged.
NON_MATCHING = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<8x4xf32>
    %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %empty)
        -> (tensor<8x4xf32>) {
      %s = tensor.extract_slice %it[%i, 0] [2, 4] [1, 1]
          : tensor<8x4xf32> to tensor<2x4xf32>
      %f = linalg.fill ins(%cst : f32) outs(%s : tensor<2x4xf32>)
          -> tensor<2x4xf32>
      %ins = tensor.insert_slice %f into %it[%i, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> into tensor<8x4xf32>
      scf.yield %ins : tensor<8x4xf32>
    }
    %out = tensor.insert_slice %loop into %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
"""

# CHECK-LABEL: Test: non_matching
# CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x4xf32>
# CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %[[EMPTY]]) -> (tensor<8x4xf32>)
# CHECK: tensor.insert_slice %[[LOOP]] into %arg0[0, 4] [8, 4] [1, 1]
run("non_matching", NON_MATCHING)


# The write-back must target the same slice as the initial extract.
MISMATCHED_WRITEBACK = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    %ext = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x16xf32> to tensor<8x4xf32>
    %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %ext)
        -> (tensor<8x4xf32>) {
      %s = tensor.extract_slice %it[%i, 0] [2, 4] [1, 1]
          : tensor<8x4xf32> to tensor<2x4xf32>
      %f = linalg.fill ins(%cst : f32) outs(%s : tensor<2x4xf32>)
          -> tensor<2x4xf32>
      %ins = tensor.insert_slice %f into %it[%i, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> into tensor<8x4xf32>
      scf.yield %ins : tensor<8x4xf32>
    }
    %out = tensor.insert_slice %loop into %arg0[0, 5] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
"""

# CHECK-LABEL: Test: mismatched_writeback
# CHECK: %[[EXT:.*]] = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
# CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %[[EXT]]) -> (tensor<8x4xf32>)
# CHECK: tensor.insert_slice %[[LOOP]] into %arg0[0, 5] [8, 4] [1, 1]
run("mismatched_writeback", MISMATCHED_WRITEBACK)


# Rewriting is unsafe when the loop result has users besides its write-back.
MULTIPLE_RESULT_USERS = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    %ext = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x16xf32> to tensor<8x4xf32>
    %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %ext)
        -> (tensor<8x4xf32>) {
      %s = tensor.extract_slice %it[%i, 0] [2, 4] [1, 1]
          : tensor<8x4xf32> to tensor<2x4xf32>
      %f = linalg.fill ins(%cst : f32) outs(%s : tensor<2x4xf32>)
          -> tensor<2x4xf32>
      %ins = tensor.insert_slice %f into %it[%i, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> into tensor<8x4xf32>
      scf.yield %ins : tensor<8x4xf32>
    }
    %out0 = tensor.insert_slice %loop into %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    %out1 = tensor.insert_slice %loop into %out0[0, 8] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    return %out1 : tensor<8x16xf32>
  }
}
"""

# CHECK-LABEL: Test: multiple_result_users
# CHECK: %[[EXT:.*]] = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
# CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %[[EXT]]) -> (tensor<8x4xf32>)
# CHECK: tensor.insert_slice %[[LOOP]] into %arg0[0, 4] [8, 4] [1, 1]
# CHECK: tensor.insert_slice %[[LOOP]] into %{{.*}}[0, 8] [8, 4] [1, 1]
run("multiple_result_users", MULTIPLE_RESULT_USERS)


# Direct uses of the iter_arg cannot be rebased unless they are supported
# extract_slice/insert_slice operations.
UNSUPPORTED_ITER_ARG_USE = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    %ext = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x16xf32> to tensor<8x4xf32>
    %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %ext)
        -> (tensor<8x4xf32>) {
      %filled = linalg.fill ins(%cst : f32) outs(%it : tensor<8x4xf32>)
          -> tensor<8x4xf32>
      %s = tensor.extract_slice %filled[%i, 0] [2, 4] [1, 1]
          : tensor<8x4xf32> to tensor<2x4xf32>
      %ins = tensor.insert_slice %s into %it[%i, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> into tensor<8x4xf32>
      scf.yield %ins : tensor<8x4xf32>
    }
    %out = tensor.insert_slice %loop into %arg0[0, 4] [8, 4] [1, 1]
        : tensor<8x4xf32> into tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
"""

# CHECK-LABEL: Test: unsupported_iter_arg_use
# CHECK: %[[EXT:.*]] = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
# CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%[[IT:.*]] = %[[EXT]]) -> (tensor<8x4xf32>)
# CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[IT]] : tensor<8x4xf32>)
# CHECK: tensor.insert_slice %[[LOOP]] into %arg0[0, 4] [8, 4] [1, 1]
run("unsupported_iter_arg_use", UNSUPPORTED_ITER_ARG_USE)


# Matching all scf.for ops may include nested loops. A nonmatching outer loop
# must not prevent an independently matching inner loop from being rewritten.
NESTED_MATCHING_INNER = """
module {
  func.func @main(%arg0: tensor<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f32
    scf.for %outer = %c0 to %c1 step %c1 {
      %ext = tensor.extract_slice %arg0[0, 4] [8, 4] [1, 1]
          : tensor<8x16xf32> to tensor<8x4xf32>
      %loop = scf.for %i = %c0 to %c8 step %c2 iter_args(%it = %ext)
          -> (tensor<8x4xf32>) {
        %s = tensor.extract_slice %it[%i, 0] [2, 4] [1, 1]
            : tensor<8x4xf32> to tensor<2x4xf32>
        %f = linalg.fill ins(%cst : f32) outs(%s : tensor<2x4xf32>)
            -> tensor<2x4xf32>
        %ins = tensor.insert_slice %f into %it[%i, 0] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<8x4xf32>
        scf.yield %ins : tensor<8x4xf32>
      }
      %out = tensor.insert_slice %loop into %arg0[0, 4] [8, 4] [1, 1]
          : tensor<8x4xf32> into tensor<8x16xf32>
    }
    return
  }
}
"""

# CHECK-LABEL: Test: nested_matching_inner
# CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
# CHECK-NOT: tensor.extract_slice %arg0[0, 4] [8, 4]
# CHECK: scf.for %[[I:.*]] = %{{.*}} iter_args(%[[IT:.*]] = %arg0) -> (tensor<8x16xf32>)
# CHECK: tensor.extract_slice %[[IT]][%[[I]], 4] [2, 4] [1, 1]
# CHECK: tensor.insert_slice %{{.*}} into %[[IT]][%[[I]], 4] [2, 4] [1, 1]
run("nested_matching_inner", NESTED_MATCHING_INNER)
