# RUN: %PYTHON %s | FileCheck %s

"""Tests for the narrow_vector_contract_operands transform op.

The op brings a `vector.contract`'s multiplicands to the narrow (DPAS) operand
type while the accumulator stays wide: `arith.extf` producers are dropped, and a
wider operand (e.g. f32 softmax weights) gets an `arith.truncf`.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        with schedule_boilerplate() as (sched, named_seq):
            contracts = lh_transform.match_op(named_seq.bodyTarget, "vector.contract")
            transform_ext.narrow_vector_contract_operands(contracts)
            transform.yield_()
        sched.body.operations[0].apply(payload.operation)
        payload.operation.verify()
        print(payload)


# Both multiplicands are extf'd bf16 -> f32 (the Q@K^T case); the accumulator
# stays f32. Both widens are dropped, no truncf is needed.
BOTH = """
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
module {
  func.func @main(%a: vector<128x64xbf16>, %b: vector<64x64xbf16>,
                  %acc: vector<128x64xf32>) -> vector<128x64xf32> {
    %ea = arith.extf %a : vector<128x64xbf16> to vector<128x64xf32>
    %eb = arith.extf %b : vector<64x64xbf16> to vector<64x64xf32>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #acc],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %ea, %eb, %acc : vector<128x64xf32>, vector<64x64xf32> into vector<128x64xf32>
    return %c : vector<128x64xf32>
  }
}
"""

# CHECK-LABEL: Test: both_extf
# Both extf ops are gone and the contract takes the bf16 operands directly.
# CHECK-NOT: arith.extf
# CHECK-NOT: arith.truncf
# CHECK: vector.contract
# CHECK-SAME: vector<128x64xbf16>, vector<64x64xbf16> into vector<128x64xf32>
run("both_extf", BOTH)


# The P@V case: lhs P is already f32 (softmax weights) and used again by a
# reduction, rhs V is extf'd bf16. The widen on V is dropped and P is truncf'd to
# the narrow type only at the contract, leaving the other P use untouched.
MIXED = """
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
module {
  func.func @main(%p: vector<128x64xf32>, %v: vector<64x64xbf16>,
                  %acc: vector<128x64xf32>) -> (vector<128x64xf32>, vector<128x64xf32>) {
    %ev = arith.extf %v : vector<64x64xbf16> to vector<64x64xf32>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #acc],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %p, %ev, %acc : vector<128x64xf32>, vector<64x64xf32> into vector<128x64xf32>
    %other = arith.mulf %p, %p : vector<128x64xf32>
    return %c, %other : vector<128x64xf32>, vector<128x64xf32>
  }
}
"""

# CHECK-LABEL: Test: mixed_extf_and_truncf
# V's extf is dropped; P is narrowed with a truncf and the contract is bf16 x bf16.
# CHECK-NOT: arith.extf
# CHECK: arith.truncf %arg0 : vector<128x64xf32> to vector<128x64xbf16>
# CHECK: vector.contract
# CHECK-SAME: vector<128x64xbf16>, vector<64x64xbf16> into vector<128x64xf32>
# The wide P is still available to its other consumer.
# CHECK: arith.mulf %arg0
run("mixed_extf_and_truncf", MIXED)


# The scaled Q@K^T case: both multiplicands are f32 scaled copies of extf'd bf16
# values, K also transposed. The narrow type is traced through the scaling and
# both operands are truncf'd at the contract.
SCALED = """
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
module {
  func.func @main(%q: vector<128x64xbf16>, %k: vector<64x64xbf16>, %s: f32,
                  %acc: vector<128x64xf32>) -> vector<128x64xf32> {
    %eq = arith.extf %q : vector<128x64xbf16> to vector<128x64xf32>
    %sq = vector.broadcast %s : f32 to vector<128x64xf32>
    %qs = arith.mulf %eq, %sq : vector<128x64xf32>
    %ek = arith.extf %k : vector<64x64xbf16> to vector<64x64xf32>
    %kt = vector.transpose %ek, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    %sk = vector.broadcast %s : f32 to vector<64x64xf32>
    %ks = arith.mulf %kt, %sk : vector<64x64xf32>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #acc],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %qs, %ks, %acc : vector<128x64xf32>, vector<64x64xf32> into vector<128x64xf32>
    return %c : vector<128x64xf32>
  }
}
"""

# CHECK-LABEL: Test: scaled_extf
# CHECK: %[[QS:.+]] = arith.mulf {{.*}} : vector<128x64xf32>
# CHECK: %[[KS:.+]] = arith.mulf {{.*}} : vector<64x64xf32>
# CHECK-DAG: %[[QT:.+]] = arith.truncf %[[QS]] : vector<128x64xf32> to vector<128x64xbf16>
# CHECK-DAG: %[[KT:.+]] = arith.truncf %[[KS]] : vector<64x64xf32> to vector<64x64xbf16>
# CHECK: vector.contract
# CHECK-SAME: %[[QT]], %[[KT]], %arg3 : vector<128x64xbf16>, vector<64x64xbf16> into vector<128x64xf32>
run("scaled_extf", SCALED)


# Genuinely f32 operands with no extf origin are left unchanged.
F32 = """
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
module {
  func.func @main(%a: vector<128x64xf32>, %b: vector<64x64xf32>,
                  %acc: vector<128x64xf32>) -> vector<128x64xf32> {
    %sa = arith.mulf %a, %a : vector<128x64xf32>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #acc],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %sa, %b, %acc : vector<128x64xf32>, vector<64x64xf32> into vector<128x64xf32>
    return %c : vector<128x64xf32>
  }
}
"""

# CHECK-LABEL: Test: no_extf_origin
# CHECK-NOT: arith.truncf
# CHECK: vector.contract
# CHECK-SAME: vector<128x64xf32>, vector<64x64xf32> into vector<128x64xf32>
run("no_extf_origin", F32)
