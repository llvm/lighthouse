# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.move_contraction_scale_to_lhs`.

The op rewrites ``contract(sl * A, sr * B)`` into ``contract((sl * sr) * A, B)``.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

CONTRACTION = """
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<128x32xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  %s = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%q, %kt : tensor<128x64xf32>, tensor<64x32xf32>)
      outs(%fill : tensor<128x32xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %m = arith.mulf %a, %b : f32
    %r = arith.addf %acc, %m : f32
    linalg.yield %r : f32
  } -> tensor<128x32xf32>
"""

HEADER = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
"""


def scale(name, source, shape, factor="%c", extra=""):
    return f"""
  %{name}_init = tensor.empty() : tensor<{shape}xf32>
  %{name} = linalg.generic {{indexing_maps = [#id, #id],
                            iterator_types = ["parallel", "parallel"]}}
      ins({source} : tensor<{shape}xf32>) outs(%{name}_init : tensor<{shape}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    {extra}
    %v = arith.mulf %in, {factor} : f32
    linalg.yield %v : f32
  }} -> tensor<{shape}xf32>
"""


#: Q and K are both scaled by a constant computed in the body, as torch-mlir emits.
BOTH_SCALED = (
    HEADER
    + """
func.func @both(%q0: tensor<128x64xf32>, %k0: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant 0.35355339059327379 : f64
"""
    + scale("q", "%q0", "128x64", "%c", "%c = arith.truncf %cst : f64 to f32")
    + scale("kt", "%k0", "64x32", "%c", "%c = arith.truncf %cst : f64 to f32")
    + CONTRACTION
    + """
  return %s : tensor<128x32xf32>
}
"""
)

#: Only K is scaled, by a function argument.
RHS_SCALED = (
    HEADER
    + """
func.func @rhs(%q: tensor<128x64xf32>, %k0: tensor<64x32xf32>, %c: f32) -> tensor<128x32xf32> {
"""
    + scale("kt", "%k0", "64x32")
    + CONTRACTION
    + """
  return %s : tensor<128x32xf32>
}
"""
)

#: The scaled K has another user, so its scale op must survive.
SHARED_RHS = (
    HEADER
    + """
func.func @shared(%q: tensor<128x64xf32>, %k0: tensor<64x32xf32>, %c: f32)
    -> (tensor<128x32xf32>, tensor<64x32xf32>) {
"""
    + scale("kt", "%k0", "64x32")
    + CONTRACTION
    + """
  return %s, %kt : tensor<128x32xf32>, tensor<64x32xf32>
}
"""
)

#: The factor is the output block argument, not a uniform scalar.
NOT_UNIFORM = (
    HEADER
    + """
func.func @not_uniform(%q: tensor<128x64xf32>, %k0: tensor<64x32xf32>) -> tensor<128x32xf32> {
"""
    + scale("kt", "%k0", "64x32", "%out")
    + CONTRACTION
    + """
  return %s : tensor<128x32xf32>
}
"""
)


def apply(payload_str: str) -> ir.Module:
    payload = ir.Module.parse(payload_str)
    with schedule_boilerplate() as (schedule, seq):
        contraction = transform_ext.filter_contraction_ops(
            transform.structured.MatchOp(
                transform.AnyOpType.get(), seq.bodyTarget, ops=["linalg.generic"]
            ).results[0]
        )
        transform_ext.move_contraction_scale_to_lhs(contraction)
        transform.yield_([])
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


def run(payload_str: str) -> None:
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply(payload_str))


run(BOTH_SCALED)
# The K scale is erased and the Q scale is replaced by one combined scale.
# CHECK-LABEL: func.func @both(
# CHECK-SAME:      %[[Q0:[a-z0-9]+]]: tensor<128x64xf32>, %[[K0:[a-z0-9]+]]: tensor<64x32xf32>
# CHECK:         %[[Q:.+]] = linalg.generic
# CHECK-SAME:      ins(%[[Q0]] : tensor<128x64xf32>)
# CHECK:           %[[CQ:.+]] = arith.truncf
# CHECK:           %[[CK:.+]] = arith.truncf
# CHECK:           %[[C:.+]] = arith.mulf %[[CQ]], %[[CK]]
# CHECK:           arith.mulf %{{.+}}, %[[C]]
# CHECK:         linalg.generic
# CHECK-SAME:      ins(%[[Q]], %[[K0]] : tensor<128x64xf32>, tensor<64x32xf32>)
# CHECK-NOT:     linalg.generic
# CHECK:         return

run(RHS_SCALED)
# CHECK-LABEL: func.func @rhs(
# CHECK-SAME:      %[[Q0:[a-z0-9]+]]: tensor<128x64xf32>, %[[K0:[a-z0-9]+]]: tensor<64x32xf32>, %[[C:[a-z0-9]+]]: f32
# CHECK:         %[[Q:.+]] = linalg.generic
# CHECK-SAME:      ins(%[[Q0]] : tensor<128x64xf32>)
# CHECK:           arith.mulf %{{.+}}, %[[C]]
# CHECK:         linalg.generic
# CHECK-SAME:      ins(%[[Q]], %[[K0]] : tensor<128x64xf32>, tensor<64x32xf32>)
# CHECK-NOT:     linalg.generic
# CHECK:         return

run(SHARED_RHS)
# CHECK-LABEL: func.func @shared(
# CHECK-SAME:      %[[Q0:[a-z0-9]+]]: tensor<128x64xf32>, %[[K0:[a-z0-9]+]]: tensor<64x32xf32>
# CHECK:         %[[KT:.+]] = linalg.generic
# CHECK-SAME:      ins(%[[K0]] : tensor<64x32xf32>)
# CHECK:         %[[Q:.+]] = linalg.generic
# CHECK-SAME:      ins(%[[Q0]] : tensor<128x64xf32>)
# CHECK:         %[[S:.+]] = linalg.generic
# CHECK-SAME:      ins(%[[Q]], %[[K0]] : tensor<128x64xf32>, tensor<64x32xf32>)
# CHECK:         return %[[S]], %[[KT]]

run(NOT_UNIFORM)
# The payload is left unchanged.
# CHECK-LABEL: func.func @not_uniform(
# CHECK-SAME:      %[[Q0:[a-z0-9]+]]: tensor<128x64xf32>
# CHECK:         %[[KT:.+]] = linalg.generic
# CHECK:           arith.mulf %{{.+}}, %out
# CHECK:         linalg.generic
# CHECK-SAME:      ins(%[[Q0]], %[[KT]] : tensor<128x64xf32>, tensor<64x32xf32>)
