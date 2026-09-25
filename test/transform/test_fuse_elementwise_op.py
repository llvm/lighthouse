# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(
    name: str,
    payload_str: str,
    producer_index=0,
    consumer_index=1,
    expect_failure=False,
):
    """Parse a payload, fuse the selected pair, and print the verified result."""
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        before = str(payload)
        with schedule_boilerplate() as (schedule, named_seq):
            matches = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
            producer = (
                matches
                if producer_index is None
                else transform_ext.extract_handle(matches, producer_index)
            )
            consumer = (
                lh_transform.match_op(named_seq.bodyTarget, "linalg.fill")
                if consumer_index is None
                else transform_ext.extract_handle(matches, consumer_index)
            )
            fused = transform_ext.fuse_elementwise_op(producer, consumer)
            transform.annotate(fused, "test.fused")
            transform.annotate(producer, "test.producer")
            transform.yield_()
        schedule.operation.verify()
        ir.Module.parse(str(schedule)).operation.verify()
        try:
            schedule.body.operations[0].apply(payload.operation)
        except ValueError as error:
            assert expect_failure, error
            assert "Failed to apply named transform sequence" in str(error), error
            assert str(payload) == before
            print("Fusion rejected; payload unchanged")
        else:
            assert not expect_failure, payload
        payload.operation.verify()
        print(payload)


# The consumer computes exp(-arg0) directly; the original producer is preserved.
SIMPLE = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %consumer : tensor<4xf32>
  }
}
"""

# CHECK-LABEL: Test: simple
# CHECK: linalg.generic
# CHECK-SAME: test.producer
# CHECK: %[[FUSED:.*]] = linalg.generic
# CHECK-SAME: ins(%arg0 : tensor<4xf32>)
# CHECK-SAME: test.fused
# CHECK: %[[NEG:.*]] = arith.negf
# CHECK-NEXT: %[[EXP:.*]] = math.exp %[[NEG]]
# CHECK-NEXT: linalg.yield %[[EXP]]
# CHECK-NOT: linalg.generic
# CHECK: return %[[FUSED]]
run("simple", SIMPLE)


# Other consumers keep their original input and body.
SHARED_PRODUCER = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %other = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %other, %consumer : tensor<4xf32>, tensor<4xf32>
  }
}
"""

# CHECK-LABEL: Test: shared_producer
# CHECK: %[[PRODUCER:.*]] = linalg.generic
# CHECK: %[[OTHER:.*]] = linalg.generic
# CHECK-SAME: ins(%[[PRODUCER]]
# CHECK-NEXT: ^bb0(%[[OTHER_INPUT:.*]]: f32, %{{.*}}: f32):
# CHECK-NEXT: %[[OTHER_EXP:.*]] = math.exp %[[OTHER_INPUT]]
# CHECK-NEXT: linalg.yield %[[OTHER_EXP]]
# CHECK: %[[SHARED_FUSED:.*]] = linalg.generic
# CHECK-SAME: ins(%arg0 : tensor<4xf32>)
# CHECK-SAME: test.fused
# CHECK: %[[SHARED_NEG:.*]] = arith.negf
# CHECK-NEXT: math.exp %[[SHARED_NEG]]
# CHECK: return %[[OTHER]], %[[SHARED_FUSED]]
run("shared_producer", SHARED_PRODUCER, consumer_index=2)


# A scalar captured from outside the producer must remain available after fusion.
CAPTURED_SCALAR = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %scale = arith.constant 2.0 : f32
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      %scaled = arith.mulf %neg, %scale : f32
      linalg.yield %scaled : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %consumer : tensor<4xf32>
  }
}
"""

# CHECK-LABEL: Test: captured_scalar
# CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: test.fused
# CHECK: %[[CAPTURED_NEG:.*]] = arith.negf
# CHECK-NEXT: %[[SCALED:.*]] = arith.mulf %[[CAPTURED_NEG]], %[[SCALE]]
# CHECK-NEXT: math.exp %[[SCALED]]
run("captured_scalar", CAPTURED_SCALAR)


# Producer init values read by its body become inputs to the fused reduction.
REDUCTION_CONSUMER = """
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>, %sum: tensor<f32>)
      -> tensor<f32> {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      %add = arith.addf %neg, %output : f32
      linalg.yield %add : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #scalar], iterator_types = ["reduction"]}
        ins(%producer : tensor<4xf32>) outs(%sum : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<f32>
    return %consumer : tensor<f32>
  }
}
"""

# CHECK-LABEL: Test: reduction_consumer
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["reduction"]
# CHECK-SAME: ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%arg2 : tensor<f32>)
# CHECK-SAME: test.fused
# CHECK: arith.negf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: linalg.yield
run("reduction_consumer", REDUCTION_CONSUMER)


# Transposing the producer result also changes its input map and index values.
PERMUTED_MAPS_AND_INDEX = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @main(%arg0: tensor<2x3xf32>, %init: tensor<3x2xf32>, %out: tensor<3x2xf32>)
      -> tensor<3x2xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #transpose], iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
    ^bb0(%input: f32, %output: f32):
      %index = linalg.index 0 : index
      %integer = arith.index_cast %index : index to i64
      %value = arith.sitofp %integer : i64 to f32
      linalg.yield %value : f32
    } -> tensor<3x2xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%producer : tensor<3x2xf32>) outs(%out : tensor<3x2xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<3x2xf32>
    return %consumer : tensor<3x2xf32>
  }
}
"""

# CHECK-LABEL: Test: permuted_maps_and_index
# CHECK-DAG: #[[TRANSPOSE:.*]] = affine_map<(d0, d1) -> (d1, d0)>
# CHECK-DAG: #[[INDEX_MAP:.*]] = affine_map<(d0, d1) -> (d1)>
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: indexing_maps = [#[[TRANSPOSE]],
# CHECK-SAME: test.fused
# CHECK: %[[ROW:.*]] = linalg.index 0
# CHECK-NEXT: %[[COL:.*]] = linalg.index 1
# CHECK-NEXT: %[[INDEX:.*]] = affine.apply #[[INDEX_MAP]](%[[ROW]], %[[COL]])
# CHECK-NEXT: %[[INTEGER:.*]] = arith.index_cast %[[INDEX]]
# CHECK-NEXT: arith.sitofp %[[INTEGER]]
run("permuted_maps_and_index", PERMUTED_MAPS_AND_INDEX)


# Producer input maps may contain arithmetic; only its result map must permute dims.
AFFINE_PRODUCER_INPUT = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#diagonal = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @main(%arg0: tensor<7xf32>, %init: tensor<4x4xf32>, %out: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %producer = linalg.generic {indexing_maps = [#diagonal, #id], iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<7xf32>) outs(%init : tensor<4x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4x4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%producer : tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<4x4xf32>
    return %consumer : tensor<4x4xf32>
  }
}
"""

# CHECK-LABEL: Test: affine_producer_input
# CHECK: #[[DIAGONAL:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: indexing_maps = [#[[DIAGONAL]],
# CHECK-SAME: test.fused
run("affine_producer_input", AFFINE_PRODUCER_INPUT)


# Removing the broadcast result would lose the consumer's reduction bound.
MISSING_REDUCTION_BOUND = """
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
module {
  func.func @main(%arg0: tensor<f32>, %init: tensor<4xf32>, %sum: tensor<f32>)
      -> tensor<f32> {
    %producer = linalg.generic {indexing_maps = [#scalar, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<f32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #scalar], iterator_types = ["reduction"]}
        ins(%producer : tensor<4xf32>) outs(%sum : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<f32>
    return %consumer : tensor<f32>
  }
}
"""

# CHECK-LABEL: Test: missing_reduction_bound
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("missing_reduction_bound", MISSING_REDUCTION_BOUND, expect_failure=True)


# The producer must have only parallel loops.
REDUCTION_PRODUCER = """
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
#scalar_id = affine_map<() -> ()>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<f32>, %out: tensor<f32>)
      -> tensor<f32> {
    %producer = linalg.generic {indexing_maps = [#id, #scalar], iterator_types = ["reduction"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      %add = arith.addf %neg, %output : f32
      linalg.yield %add : f32
    } -> tensor<f32>
    %consumer = linalg.generic {indexing_maps = [#scalar_id, #scalar_id], iterator_types = []}
        ins(%producer : tensor<f32>) outs(%out : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<f32>
    return %consumer : tensor<f32>
  }
}
"""

# CHECK-LABEL: Test: reduction_producer
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("reduction_producer", REDUCTION_PRODUCER, expect_failure=True)


# The accessed tensor rank must match the producer's number of loops.
RANK_MISMATCH = """
#id = affine_map<(d0) -> (d0)>
#expand = affine_map<(d0) -> (d0, 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4x1xf32>, %out: tensor<4x1xf32>)
      -> tensor<4x1xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #expand], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4x1xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4x1xf32>
    %consumer = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%producer : tensor<4x1xf32>) outs(%out : tensor<4x1xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<4x1xf32>
    return %consumer : tensor<4x1xf32>
  }
}
"""

# CHECK-LABEL: Test: rank_mismatch
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("rank_mismatch", RANK_MISMATCH, expect_failure=True)


# Invalid handle selections use the same simple payload.
# CHECK-LABEL: Test: multiple_producers
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("multiple_producers", SIMPLE, producer_index=None, expect_failure=True)

# CHECK-LABEL: Test: empty_consumer
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("empty_consumer", SIMPLE, consumer_index=None, expect_failure=True)

# CHECK-LABEL: Test: reversed_pair
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("reversed_pair", SIMPLE, producer_index=1, consumer_index=0, expect_failure=True)

# CHECK-LABEL: Test: same_operation
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("same_operation", SIMPLE, consumer_index=0, expect_failure=True)


# Independent operations have no producer-to-consumer input edge.
UNRELATED_PAIR = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %consumer : tensor<4xf32>
  }
}
"""

# CHECK-LABEL: Test: unrelated_pair
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("unrelated_pair", UNRELATED_PAIR, expect_failure=True)


# Fusion through the consumer's destination/init operand is unsupported.
PRODUCER_AS_INIT = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<4xf32>) outs(%producer : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %consumer : tensor<4xf32>
  }
}
"""

# CHECK-LABEL: Test: producer_as_init
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("producer_as_init", PRODUCER_AS_INIT, expect_failure=True)


# Buffer operations have no tensor-result edge to fuse.
BUFFER_SEMANTICS = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: memref<4xf32>, %init: memref<4xf32>, %out: memref<4xf32>) {
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : memref<4xf32>) outs(%init : memref<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%init : memref<4xf32>) outs(%out : memref<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    }
    return
  }
}
"""

# CHECK-LABEL: Test: buffer_semantics
# CHECK: Fusion rejected; payload unchanged
# CHECK: memref<4xf32>
# CHECK-NOT: test.fused
run("buffer_semantics", BUFFER_SEMANTICS, expect_failure=True)


# A same-rank result map containing a constant is not a permutation.
NON_PERMUTATION_RESULT_MAP = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#project = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @main(%arg0: tensor<4x1xf32>, %init: tensor<4x1xf32>, %out: tensor<4x1xf32>)
      -> tensor<4x1xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #project], iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4x1xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%producer : tensor<4x1xf32>) outs(%out : tensor<4x1xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<4x1xf32>
    return %consumer : tensor<4x1xf32>
  }
}
"""

# CHECK-LABEL: Test: non_permutation_result_map
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
run("non_permutation_result_map", NON_PERMUTATION_RESULT_MAP, expect_failure=True)


# Rank-zero tensors need no loop dimensions or index remapping.
ZERO_RANK = """
#scalar = affine_map<() -> ()>
module {
  func.func @main(%arg0: tensor<f32>, %init: tensor<f32>, %out: tensor<f32>)
      -> tensor<f32> {
    %producer = linalg.generic {indexing_maps = [#scalar, #scalar], iterator_types = []}
        ins(%arg0 : tensor<f32>) outs(%init : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<f32>
    %consumer = linalg.generic {indexing_maps = [#scalar, #scalar], iterator_types = []}
        ins(%producer : tensor<f32>) outs(%out : tensor<f32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<f32>
    return %consumer : tensor<f32>
  }
}
"""

# CHECK-LABEL: Test: zero_rank
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = []
# CHECK-SAME: ins(%arg0 : tensor<f32>)
# CHECK-SAME: test.fused
# CHECK: arith.negf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: linalg.yield
run("zero_rank", ZERO_RANK)


# Dynamic extents remain recoverable from the fused operands.
DYNAMIC_SHAPES = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<?xf32>, %init: tensor<?xf32>, %out: tensor<?xf32>)
      -> tensor<?xf32> {
    %producer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%arg0 : tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<?xf32>
    %consumer = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%producer : tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%input: f32, %output: f32):
      %add = arith.addf %input, %output : f32
      linalg.yield %add : f32
    } -> tensor<?xf32>
    return %consumer : tensor<?xf32>
  }
}
"""

# CHECK-LABEL: Test: dynamic_shapes
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: ins(%arg0 : tensor<?xf32>)
# CHECK-SAME: test.fused
# CHECK: arith.negf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: linalg.yield
run("dynamic_shapes", DYNAMIC_SHAPES)
