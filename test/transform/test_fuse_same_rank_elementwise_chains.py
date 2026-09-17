# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.fuse_same_rank_elementwise_chains`.

The op merges an elementwise producer into an elementwise consumer of the same
rank and stops there: it never fuses across a reduction.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


#: A softmax whose term is split across two elementwise ops, as torch-mlir emits
#: it: `%d = x - m` then `%p = exp(%d)`, between the max and the sum.
SPLIT_TERM_SOFTMAX = """
#rc = affine_map<(d0, d1) -> (d0, d1)>
#r  = affine_map<(d0, d1) -> (d0)>

func.func @softmax(%x: tensor<64x512xf32>) -> tensor<64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %row_init = tensor.empty() : tensor<64xf32>
  %full_init = tensor.empty() : tensor<64x512xf32>

  %m_init = linalg.fill ins(%ninf : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %m = linalg.generic {indexing_maps = [#rc, #r],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<64x512xf32>) outs(%m_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.maximumf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>

  %d = linalg.generic {indexing_maps = [#rc, #r, #rc],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<64x512xf32>, tensor<64xf32>)
      outs(%full_init : tensor<64x512xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.subf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<64x512xf32>

  %p = linalg.generic {indexing_maps = [#rc, #rc],
                       iterator_types = ["parallel", "parallel"]}
      ins(%d : tensor<64x512xf32>) outs(%full_init : tensor<64x512xf32>) {
  ^bb0(%a: f32, %o: f32):
    %e = math.exp %a : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  %s_init = linalg.fill ins(%zero : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %s = linalg.generic {indexing_maps = [#rc, #r],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<64x512xf32>) outs(%s_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>
  return %s : tensor<64xf32>
}
"""

#: A broadcasting consumer: the producer's result is read under a non-identity map,
#: so it is not a same-rank chain and must be left alone.
BROADCAST_CONSUMER = """
#rc  = affine_map<(d0, d1) -> (d0, d1)>
#r   = affine_map<(d0, d1) -> (d0)>
#one = affine_map<(d0) -> (d0)>

func.func @broadcast(%x: tensor<64xf32>) -> tensor<64x512xf32> {
  %row_init = tensor.empty() : tensor<64xf32>
  %full_init = tensor.empty() : tensor<64x512xf32>
  %a = linalg.generic {indexing_maps = [#one, #one], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%row_init : tensor<64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %b = linalg.generic {indexing_maps = [#r, #rc], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64xf32>) outs(%full_init : tensor<64x512xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.absf %in : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>
  return %b : tensor<64x512xf32>
}
"""


def fuse_schedule() -> ir.Module:
    """Run the op over the whole function."""
    with schedule_boilerplate() as (sched, seq):
        func = transform.structured.MatchOp(
            transform.AnyOpType.get(), seq.bodyTarget, ops=["func.func"]
        ).results[0]
        transform_ext.fuse_same_rank_elementwise_chains(func)
        transform.yield_([])
    return sched


def apply(payload_str: str) -> ir.Module:
    payload = ir.Module.parse(payload_str)
    # Bound to a local: the schedule module must outlive `apply`.
    schedule = fuse_schedule()
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


def test_split_term_softmax() -> None:
    """The two elementwise ops merge; both reductions stay put."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(SPLIT_TERM_SOFTMAX))


# The max reduction is untouched.
# CHECK-LABEL: func.func @softmax
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "reduction"]
# CHECK:           arith.maximumf
#
# The term is now a single all-parallel op computing both the subtract and the exp.
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel"]
# CHECK:           %[[D:.+]] = arith.subf
# CHECK:           math.exp %[[D]]
# CHECK:           linalg.yield
#
# The sum reduction is untouched -- nothing was fused into it.
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "reduction"]
# CHECK:           arith.addf


def test_broadcast_consumer_is_left_alone() -> None:
    """A rank-changing read is not a same-rank chain, so nothing fuses."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(BROADCAST_CONSUMER))


# Both ops survive, in order.
# CHECK-LABEL: func.func @broadcast
# CHECK:         math.exp
# CHECK:         linalg.generic
# CHECK:         math.absf


if __name__ == "__main__":
    test_split_term_softmax()
    test_broadcast_consumer_is_left_alone()
