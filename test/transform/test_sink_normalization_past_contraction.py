# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.sink_normalization_past_contraction`.

The op rewrites ``contract(A / N, B)`` into ``contract(A, B) / N``, which is legal
because ``N`` does not vary along the reduction axis. Two input shapes are covered
-- the scale as its own op, and the scale already fused into the contraction's body
by ``linalg-fuse-elementwise-ops`` -- plus one case that must be rejected.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


_MAPS = """
#rc  = affine_map<(d0, d1) -> (d0, d1)>
#r   = affine_map<(d0, d1) -> (d0)>
#ik  = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj  = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij  = affine_map<(d0, d1, d2) -> (d0, d1)>
#i   = affine_map<(d0, d1, d2) -> (d0)>
"""

#: `(P / l) @ V`, the divide as its own elementwise op feeding the contraction.
SEPARATE_SCALE = (
    _MAPS
    + """
func.func @pv(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
              %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %pn_init = tensor.empty() : tensor<64x512xf32>
  %pn = linalg.generic {indexing_maps = [#rc, #r, #rc],
                        iterator_types = ["parallel", "parallel"]}
      ins(%p, %l : tensor<64x512xf32>, tensor<64xf32>)
      outs(%pn_init : tensor<64x512xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %d = arith.divf %a, %b : f32
    linalg.yield %d : f32
  } -> tensor<64x512xf32>
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%pn, %v : tensor<64x512xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)

#: The same computation after elementwise fusion has sunk the divide into the
#: contraction's body, which is the form the attention schedule actually sees.
FUSED_SCALE = (
    _MAPS
    + """
func.func @pv_fused(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
                    %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #i, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %l, %v : tensor<64x512xf32>, tensor<64xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %n: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %n : f32
    %m = arith.mulf %d, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)

#: The divisor varies along the reduction axis, so it does not factor out of the
#: sum and the op must decline.
REDUCTION_VARYING_SCALE = (
    _MAPS
    + """
func.func @pv_varying(%p: tensor<64x512xf32>, %n: tensor<512xf32>,
                      %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, affine_map<(d0, d1, d2) -> (d2)>, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %n, %v : tensor<64x512xf32>, tensor<512xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %nv: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %nv : f32
    %m = arith.mulf %d, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)


def sink_schedule() -> ir.Module:
    with schedule_boilerplate() as (sched, seq):
        func = transform.structured.MatchOp(
            transform.AnyOpType.get(), seq.bodyTarget, ops=["func.func"]
        ).results[0]
        transform_ext.sink_normalization_past_contraction(func)
        transform.yield_([])
    return sched


def apply(payload_str: str) -> ir.Module:
    payload = ir.Module.parse(payload_str)
    # Bound to a local: the schedule module must outlive `apply`.
    schedule = sink_schedule()
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


def test_separate_scale() -> None:
    """A standalone divide moves to after the contraction."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(SEPARATE_SCALE))


# The contraction now reads the unscaled numerator directly, and only multiplies.
# CHECK-LABEL: func.func @pv
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
# CHECK-SAME:      ins(%arg0, %arg2
# CHECK:           arith.mulf
# CHECK:           arith.addf
# CHECK-NOT:       arith.divf
# CHECK:           linalg.yield
#
# The divide follows it, once per output element, reading the row scale.
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel"]
# CHECK:           arith.divf
# CHECK:         return %[[N]]


def test_fused_scale() -> None:
    """A divide already inside the contraction's body is lifted back out."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(FUSED_SCALE))


# The scale operand is dropped from the contraction, leaving two inputs.
# CHECK-LABEL: func.func @pv_fused
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
# CHECK-SAME:      ins(%arg0, %arg2
# CHECK:           arith.mulf
# CHECK:           arith.addf
# CHECK-NOT:       arith.divf
# CHECK:           linalg.yield
#
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel"]
# CHECK:           arith.divf
# CHECK:         return %[[N]]


def test_reduction_varying_scale_is_rejected() -> None:
    """A divisor indexed by the reduction dim does not factor out."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(REDUCTION_VARYING_SCALE))


# The divide stays inside the contraction.
# CHECK-LABEL: func.func @pv_varying
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
# CHECK:           arith.divf
# CHECK:           arith.mulf
# CHECK:           arith.addf


if __name__ == "__main__":
    test_separate_scale()
    test_fused_scale()
    test_reduction_varying_scale_is_rejected()
