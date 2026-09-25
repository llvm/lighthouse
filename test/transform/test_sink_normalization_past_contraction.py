# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.sink_normalization_past_contraction`.

The op rewrites ``contract(A / S, B)`` into ``contract(A, B) / S``, taking the scale
out of the contraction's body. Legal because ``S`` does not vary along the reduction
axis.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


#: `(P / l) @ V` with the divide in the contraction's body.
SIMPLE = """
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#i  = affine_map<(d0, d1, d2) -> (d0)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pv(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
              %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #i, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %l, %v : tensor<64x512xf32>, tensor<64xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %n: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %n : f32
    %m = arith.mulf %d, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""

#: The shape the kernel-bench attention payload reaches the schedule as: batched over
#: two leading dims, reduction innermost of five loops, and a bf16 scale on an f32
#: accumulator, so the scale has to be widened when the divide moves.
BATCHED_MIXED = """
#p4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#r3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#v4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#o4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @pv_batched(%p: tensor<4x4x64x512xbf16>, %l: tensor<4x4x64xbf16>,
                      %v: tensor<4x4x512x128xbf16>) -> tensor<4x4x64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4x64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<4x4x64x128xf32>) -> tensor<4x4x64x128xf32>
  %o = linalg.generic {indexing_maps = [#p4, #r3, #v4, #o4],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%p, %l, %v : tensor<4x4x64x512xbf16>, tensor<4x4x64xbf16>, tensor<4x4x512x128xbf16>)
      outs(%fill : tensor<4x4x64x128xf32>) {
  ^bb0(%a: bf16, %n: bf16, %b: bf16, %acc: f32):
    %d = arith.divf %a, %n : bf16
    %ae = arith.extf %d : bf16 to f32
    %be = arith.extf %b : bf16 to f32
    %m = arith.mulf %ae, %be : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x4x64x128xf32>
  return %o : tensor<4x4x64x128xf32>
}
"""

#: The scale is indexed by the reduction dim, so it does not factor out of the sum.
REDUCTION_VARYING = """
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#k  = affine_map<(d0, d1, d2) -> (d2)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pv_varying(%p: tensor<64x512xf32>, %n: tensor<512xf32>,
                      %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #k, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %n, %v : tensor<64x512xf32>, tensor<512xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %nv: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %nv : f32
    %m = arith.mulf %d, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""

#: Two divides on input arguments, so which one normalizes is ambiguous.
TWO_SCALES = """
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#i  = affine_map<(d0, d1, d2) -> (d0)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pv_two_scales(%p: tensor<64x512xf32>, %l: tensor<64xf32>, %g: tensor<64xf32>,
                         %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #i, #i, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %l, %g, %v : tensor<64x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %n: f32, %n2: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %n : f32
    %d2 = arith.divf %a, %n2 : f32
    %e = arith.addf %d, %d2 : f32
    %m = arith.mulf %e, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""

#: The scale feeds a transcendental before the multiply, so it is not what the
#: contraction sums: ``exp(a / n) != exp(a) / n``, and sinking would change the value.
SCALE_NOT_MULTIPLIED = """
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#i  = affine_map<(d0, d1, d2) -> (d0)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pv_not_multiplied(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
                             %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #i, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %l, %v : tensor<64x512xf32>, tensor<64xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %n: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %n : f32
    %e = math.exp %d : f32
    %m = arith.mulf %e, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""

#: The scale's map is a composite expression holding the reduction dim, so it cannot
#: be re-expressed over the output space.
COMPOSITE_SCALE_MAP = """
#ik  = affine_map<(d0, d1, d2) -> (d0, d2)>
#sum = affine_map<(d0, d1, d2) -> (d0 + d2)>
#kj  = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij  = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pv_composite(%p: tensor<64x512xf32>, %n: tensor<576xf32>,
                        %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #sum, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %n, %v : tensor<64x512xf32>, tensor<576xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) {
  ^bb0(%a: f32, %nv: f32, %b: f32, %acc: f32):
    %d = arith.divf %a, %nv : f32
    %m = arith.mulf %d, %b : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""

#: A named contraction has a fixed body, so it never carries a scale.
NAMED_MATMUL = """
func.func @pv_matmul(%p: tensor<64x512xf32>, %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.matmul ins(%p, %v : tensor<64x512xf32>, tensor<512x128xf32>)
      outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""


def _match(root, *names):
    """Handle to the payload ops named `names`, in program order."""
    return transform.structured.MatchOp(
        transform.AnyOpType.get(), root, ops=list(names)
    ).results[0]


def reduction_generic(root):
    """The contraction when it is the payload's only reduction `linalg.generic`."""
    return transform_ext.filter_reduction_ops(_match(root, "linalg.generic"))


def named(name):
    """The contraction when it is the named op `name`."""

    def matcher(root):
        return _match(root, name)

    return matcher


def sink_schedule(match_contraction) -> ir.Module:
    """Schedule applying the op to the contraction `match_contraction` picks out."""
    with schedule_boilerplate() as (sched, seq):
        transform_ext.sink_normalization_past_contraction(
            match_contraction(seq.bodyTarget)
        )
        transform.yield_([])
    return sched


def apply(payload_str: str, match_contraction=reduction_generic) -> ir.Module:
    payload = ir.Module.parse(payload_str)
    # Bound to a local: the schedule module must outlive `apply`.
    schedule = sink_schedule(match_contraction)
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


def expect_rejected(payload_str: str, match_contraction=reduction_generic) -> None:
    """Apply and print the diagnostic, which the op is expected to emit.

    The interpreter turns the silenceable failure into a `ValueError` carrying it.
    """
    try:
        apply(payload_str, match_contraction)
    except ValueError as error:
        print(error)
        return
    raise AssertionError("expected the op to reject this contraction")


def test_simple() -> None:
    """The divide moves to after the contraction, which loses the scale operand."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(SIMPLE))


# The rebuilt contraction has two inputs and only multiplies.
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


def test_batched_mixed() -> None:
    """A batched contraction keeps its parallel dims; a bf16 scale is widened."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(BATCHED_MIXED))


# CHECK-LABEL: func.func @pv_batched
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
# CHECK-SAME:      ins(%arg0, %arg2
# CHECK-NOT:       arith.divf
# CHECK:           linalg.yield
#
# The moved divide runs in the accumulator's type, so the bf16 scale is extended.
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
# CHECK:         ^bb0(%[[ACC:.+]]: f32, %[[S:.+]]: bf16, %{{.+}}: f32):
# CHECK:           %[[W:.+]] = arith.extf %[[S]] : bf16 to f32
# CHECK:           arith.divf %[[ACC]], %[[W]] : f32
# CHECK:         return %[[N]]


def test_reduction_varying_is_rejected() -> None:
    """A scale indexed by the reduction dim does not factor out."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(REDUCTION_VARYING)


# CHECK-LABEL: rejected: reduction-varying scale
# CHECK: the scale varies along the contraction's reduction dim d2


def test_two_scales_is_rejected() -> None:
    """With two candidate scales, which one normalizes is ambiguous."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(TWO_SCALES)


# CHECK-LABEL: rejected: two scales
# CHECK: has 2 arith.divf/arith.mulf ops on two input arguments


def test_scale_not_multiplied_is_rejected() -> None:
    """A scale that is not what the contraction multiplies does not factor out."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(SCALE_NOT_MULTIPLIED)


# CHECK-LABEL: rejected: scale not multiplied
# CHECK: not consumed by the contraction's multiply-accumulate


def test_composite_scale_map_is_rejected() -> None:
    """A scale map that is not a plain dim projection cannot be re-expressed."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(COMPOSITE_SCALE_MAP)


# CHECK-LABEL: rejected: composite scale map
# CHECK: cannot re-express the scale's map


def test_named_contraction_is_rejected() -> None:
    """A named op has a fixed body, so there is no scale to find."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(NAMED_MATMUL, named("linalg.matmul"))


# CHECK-LABEL: rejected: named contraction
# CHECK: expected a linalg.generic, got 'linalg.matmul'


if __name__ == "__main__":
    test_simple()
    test_batched_mixed()
    print("rejected: reduction-varying scale")
    test_reduction_varying_is_rejected()
    print("rejected: two scales")
    test_two_scales_is_rejected()
    print("rejected: scale not multiplied")
    test_scale_not_multiplied_is_rejected()
    print("rejected: composite scale map")
    test_composite_scale_map_is_rejected()
    print("rejected: named contraction")
    test_named_contraction_is_rejected()
