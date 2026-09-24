# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.sink_normalization_past_contraction`.

The op rewrites ``contract(A / S, B)`` into ``contract(A, B) / S``, which is legal
because ``S`` does not vary along the reduction axis. It takes the two payload ops
explicitly. Covered here: a `linalg.generic` contraction, the named
``linalg.matmul`` and ``linalg.batch_matmul``, a mixed-precision accumulator, and
two cases the op must reject.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


_MAPS = """
#rc  = affine_map<(d0, d1) -> (d0, d1)>
#r   = affine_map<(d0, d1) -> (d0)>
#c   = affine_map<(d0, d1) -> (d1)>
#brc = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#br  = affine_map<(d0, d1, d2) -> (d0, d1)>
#ik  = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj  = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij  = affine_map<(d0, d1, d2) -> (d0, d1)>
"""

#: `(P / l) @ V` with the contraction spelled as a `linalg.generic`.
GENERIC_CONTRACTION = (
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

#: The same computation with a named `linalg.matmul` as the contraction. Its loop
#: kinds come from the indexing maps -- it carries no `iterator_types` attribute.
NAMED_MATMUL = (
    _MAPS
    + """
func.func @pv_matmul(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
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
  %o = linalg.matmul ins(%pn, %v : tensor<64x512xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)

#: A batched variant, so the rewrite has to keep two parallel dims and project the
#: trailing reduction out of a two-result scale map.
NAMED_BATCH_MATMUL = (
    _MAPS
    + """
func.func @pv_batch_matmul(%p: tensor<4x64x512xf32>, %l: tensor<4x64xf32>,
                           %v: tensor<4x512x128xf32>) -> tensor<4x64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %pn_init = tensor.empty() : tensor<4x64x512xf32>
  %pn = linalg.generic {indexing_maps = [#brc, #br, #brc],
                        iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%p, %l : tensor<4x64x512xf32>, tensor<4x64xf32>)
      outs(%pn_init : tensor<4x64x512xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %d = arith.divf %a, %b : f32
    linalg.yield %d : f32
  } -> tensor<4x64x512xf32>
  %o_init = tensor.empty() : tensor<4x64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<4x64x128xf32>) -> tensor<4x64x128xf32>
  %o = linalg.batch_matmul ins(%pn, %v : tensor<4x64x512xf32>, tensor<4x512x128xf32>)
      outs(%o_fill : tensor<4x64x128xf32>) -> tensor<4x64x128xf32>
  return %o : tensor<4x64x128xf32>
}
"""
)

#: The contraction accumulates in f32 while the numerator and the row scale are
#: bf16, as torch-mlir emits attention. The scale has to be widened to the
#: accumulator type when the divide moves after the contraction.
MIXED_PRECISION_SCALE = (
    _MAPS
    + """
func.func @pv_mixed(%p: tensor<64x512xbf16>, %l: tensor<64xbf16>,
                    %v: tensor<512x128xbf16>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %pn_init = tensor.empty() : tensor<64x512xbf16>
  %pn = linalg.generic {indexing_maps = [#rc, #r, #rc],
                        iterator_types = ["parallel", "parallel"]}
      ins(%p, %l : tensor<64x512xbf16>, tensor<64xbf16>)
      outs(%pn_init : tensor<64x512xbf16>) {
  ^bb0(%a: bf16, %b: bf16, %o: bf16):
    %d = arith.divf %a, %b : bf16
    linalg.yield %d : bf16
  } -> tensor<64x512xbf16>
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%pn, %v : tensor<64x512xbf16>, tensor<512x128xbf16>)
      outs(%o_fill : tensor<64x128xf32>) {
  ^bb0(%a: bf16, %b: bf16, %acc: f32):
    %ae = arith.extf %a : bf16 to f32
    %be = arith.extf %b : bf16 to f32
    %m = arith.mulf %ae, %be : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)

#: The scale varies along the reduction axis, so it does not factor out of the sum
#: and the op must reject the pair.
REDUCTION_VARYING_SCALE = (
    _MAPS
    + """
func.func @pv_varying(%p: tensor<64x512xf32>, %n: tensor<512xf32>,
                      %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %pn_init = tensor.empty() : tensor<64x512xf32>
  %pn = linalg.generic {indexing_maps = [#rc, #c, #rc],
                        iterator_types = ["parallel", "parallel"]}
      ins(%p, %n : tensor<64x512xf32>, tensor<512xf32>)
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

#: The normalization is all-parallel but its result also feeds the return, so the
#: contraction is not its only user and sinking would change the other use.
EXTRA_USER_SCALE = (
    _MAPS
    + """
func.func @pv_extra_user(%p: tensor<64x512xf32>, %l: tensor<64xf32>,
                         %v: tensor<512x128xf32>)
    -> (tensor<64x128xf32>, tensor<64x512xf32>) {
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
  return %o, %pn : tensor<64x128xf32>, tensor<64x512xf32>
}
"""
)


#: The normalization reads a transposed numerator, so the contraction would have to
#: read it under a different indexing map. A named op's own verifier constrains its
#: maps, so the op must decline rather than retype it.
TRANSPOSED_NUMERATOR = (
    _MAPS
    + """
func.func @pv_transposed(%pt: tensor<512x64xf32>, %l: tensor<64xf32>,
                         %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %pn_init = tensor.empty() : tensor<64x512xf32>
  %pn = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, #r, #rc],
                        iterator_types = ["parallel", "parallel"]}
      ins(%pt, %l : tensor<512x64xf32>, tensor<64xf32>)
      outs(%pn_init : tensor<64x512xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %d = arith.divf %a, %b : f32
    linalg.yield %d : f32
  } -> tensor<64x512xf32>
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.matmul ins(%pn, %v : tensor<64x512xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}
"""
)


def _match(root, *names):
    """Handle to the payload ops named `names`, in program order."""
    return transform.structured.MatchOp(
        transform.AnyOpType.get(), root, ops=list(names)
    ).results[0]


def two_generics(root):
    """The pair when both ops are `linalg.generic`: normalization then contraction."""
    return transform.split_handle(
        [transform.AnyOpType.get()] * 2, _match(root, "linalg.generic")
    )


def generic_and(name):
    """The pair when the contraction is the named op `name`."""

    def matcher(root):
        return _match(root, "linalg.generic"), _match(root, name)

    return matcher


def sink_schedule(match_pair) -> ir.Module:
    """Schedule applying the op to the pair `match_pair` picks out of the payload."""
    with schedule_boilerplate() as (sched, seq):
        normalization, contraction = match_pair(seq.bodyTarget)
        transform_ext.sink_normalization_past_contraction(normalization, contraction)
        transform.yield_([])
    return sched


def apply(payload_str: str, match_pair=two_generics) -> ir.Module:
    payload = ir.Module.parse(payload_str)
    # Bound to a local: the schedule module must outlive `apply`.
    schedule = sink_schedule(match_pair)
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


def expect_rejected(payload_str: str, match_pair=two_generics) -> None:
    """Apply and print the diagnostic, which the op is expected to emit.

    The interpreter turns the silenceable failure into a `ValueError` carrying the
    emitted diagnostic.
    """
    try:
        apply(payload_str, match_pair)
    except ValueError as error:
        print(error)
        return
    raise AssertionError("expected the op to reject this pair")


def test_generic_contraction() -> None:
    """The divide moves to after a linalg.generic contraction."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(GENERIC_CONTRACTION))


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


def test_named_matmul() -> None:
    """A named linalg.matmul contraction, whose loop kinds come from its maps."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(NAMED_MATMUL, generic_and("linalg.matmul")))


# The matmul stays a matmul and reads the numerator; the divide follows it.
# CHECK-LABEL: func.func @pv_matmul
# CHECK:         linalg.matmul ins(%arg0, %arg2
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel"]
# CHECK:           arith.divf
# CHECK:         return %[[N]]


def test_named_batch_matmul() -> None:
    """A batched named contraction keeps its two leading parallel dims."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(NAMED_BATCH_MATMUL, generic_and("linalg.batch_matmul")))


# CHECK-LABEL: func.func @pv_batch_matmul
# CHECK:         linalg.batch_matmul ins(%arg0, %arg2
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
# CHECK:           arith.divf
# CHECK:         return %[[N]]


def test_mixed_precision_scale() -> None:
    """A bf16 scale is widened to the f32 accumulator it now divides."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(MIXED_PRECISION_SCALE))


# CHECK-LABEL: func.func @pv_mixed
# CHECK:         linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
# CHECK-NOT:       arith.divf
# CHECK:           linalg.yield
#
# The moved divide runs in the accumulator's type, so the bf16 scale is extended.
# CHECK:         %[[N:.+]] = linalg.generic
# CHECK-SAME:      iterator_types = ["parallel", "parallel"]
# CHECK:         ^bb0(%[[ACC:.+]]: f32, %[[S:.+]]: bf16, %{{.+}}: f32):
# CHECK:           %[[W:.+]] = arith.extf %[[S]] : bf16 to f32
# CHECK:           arith.divf %[[ACC]], %[[W]] : f32
# CHECK:         return %[[N]]


def test_reduction_varying_scale_is_rejected() -> None:
    """A scale indexed by the reduction dim does not factor out."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(REDUCTION_VARYING_SCALE)


# CHECK-LABEL: rejected: reduction-varying scale
# CHECK: the scale varies along the contraction's reduction dim d2


def test_extra_user_is_rejected() -> None:
    """A normalization with a second use cannot be sunk."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(EXTRA_USER_SCALE)


# CHECK-LABEL: rejected: extra user
# CHECK: only user


def test_named_op_map_change_is_rejected() -> None:
    """A named contraction cannot be retyped to read a transposed numerator."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        expect_rejected(TRANSPOSED_NUMERATOR, generic_and("linalg.matmul"))


# CHECK-LABEL: rejected: named op needing a new map
# CHECK: 'linalg.matmul' cannot read the numerator under


if __name__ == "__main__":
    test_generic_contraction()
    test_named_matmul()
    test_named_batch_matmul()
    test_mixed_precision_scale()
    print("rejected: reduction-varying scale")
    test_reduction_varying_scale_is_rejected()
    print("rejected: extra user")
    test_extra_user_is_rejected()
    print("rejected: named op needing a new map")
    test_named_op_map_change_is_rejected()
