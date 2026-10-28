# RUN: %PYTHON %s | FileCheck %s

"""Test priority-ordered subgraph tagging."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


PAYLOAD = """
module {
  func.func @payload(%a: tensor<128x64xf32>, %b: tensor<64x32xf32>,
                     %out: tensor<128x32xf32>) -> tensor<128x32xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x32xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<128x32xf32>) -> tensor<128x32xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x32xf32>)
        outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
    %result = linalg.generic {
        indexing_maps=[affine_map<(m,n)->(m,n)>,
                       affine_map<(m,n)->(m,n)>],
        iterator_types=["parallel","parallel"]}
      ins(%mm : tensor<128x32xf32>)
      outs(%out : tensor<128x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %relu = arith.maximumf %x, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x32xf32>
    return %result : tensor<128x32xf32>
  }
}
"""


def test():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        with schedule_boilerplate() as (schedule, named_sequence):
            func = lh_transform.match_op(named_sequence.bodyTarget, "func.func")
            assert (
                transform_ext.tag_subgraphs(func, kinds=["gemm", "elementwise"]) is None
            )
            tag_op = list(named_sequence.regions[0].blocks[0].operations)[-1]
            assert len(tag_op.results) == 0
            transform.yield_()
        schedule.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK: linalg.fill {transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}
# CHECK: linalg.matmul {transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}
# CHECK: linalg.generic
# CHECK-SAME: attrs = {transform_ext.subgraph_anchor, transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}


test()


def test_handle_preservation():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        with schedule_boilerplate() as (schedule, sequence):
            func = lh_transform.match_op(sequence.bodyTarget, "func.func")
            members = lh_transform.match_op(func, "linalg.generic")
            transform_ext.tag_subgraphs(func, kinds=["gemm", "elementwise"])
            transform.annotate(members, "test.preserved")
            transform.yield_()
        schedule.body.operations[0].apply(payload.operation)
        assert "test.preserved" in str(payload)
    print("Annotation-only tagging preserves existing operation handles")


# CHECK: Annotation-only tagging preserves existing operation handles

test_handle_preservation()


def test_preparation_invalidates_handles():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        with schedule_boilerplate() as (schedule, sequence):
            func = lh_transform.match_op(sequence.bodyTarget, "func.func")
            members = lh_transform.match_op(func, "linalg.generic")
            transform_ext.prepare_subgraphs(func, ["gemm", "elementwise"])
            transform.annotate(members, "test.stale")
            transform.yield_()
        try:
            schedule.body.operations[0].apply(payload.operation)
        except ValueError as error:
            assert "invalidated" in str(error), str(error)
        else:
            raise AssertionError("Preparation must invalidate nested handles")
    print("Only structural preparation invalidates nested operation handles")


# CHECK: Only structural preparation invalidates nested operation handles

test_preparation_invalidates_handles()


def test_scope():
    computations = (
        """%result = linalg.matmul
          ins(%a, %b : tensor<128x64xf32>, tensor<64x32xf32>)
          outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>""",
        """%result = linalg.generic {
          indexing_maps=[affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
          iterator_types=["parallel","parallel"]}
          ins(%init : tensor<128x32xf32>) outs(%iter : tensor<128x32xf32>) {
        ^bb0(%x: f32, %o: f32):
          %neg = arith.negf %x : f32
          linalg.yield %neg : f32
        } -> tensor<128x32xf32>""",
    )
    for computation in computations:
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(
                """
func.func @payload(%a: tensor<128x64xf32>, %b: tensor<64x32xf32>,
                   %out: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x32xf32>)
      -> tensor<128x32xf32>
  %loop = scf.for %i = %c0 to %c2 step %c1
      iter_args(%iter = %out) -> tensor<128x32xf32> {
"""
                + computation
                + """
    scf.yield %result : tensor<128x32xf32>
  }
  return %loop : tensor<128x32xf32>
}
"""
            )
            with schedule_boilerplate() as (schedule, sequence):
                loop = lh_transform.match_op(sequence.bodyTarget, "scf.for")
                transform_ext.tag_subgraphs(loop, kinds=["gemm", "elementwise"])
                transform.yield_()
            schedule.body.operations[0].apply(payload.operation)
            assert payload.operation.verify()
            fills = []

            def visit(op):
                if op.name == "linalg.fill":
                    fills.append(op)
                return ir.WalkResult.ADVANCE

            payload.operation.walk(visit)
            assert len(fills) == 1
            assert "transform_ext.subgraph_id" not in fills[0].attributes
    print("Subgraph matching does not traverse or clone outside the target block")


# CHECK: Subgraph matching does not traverse or clone outside the target block

test_scope()


def test_multiblock_rejection():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse("""
func.func @payload(%condition: i1) {
  cf.cond_br %condition, ^then, ^else
^then:
  return
^else:
  return
}
""")
        before = str(payload)
        with schedule_boilerplate() as (schedule, sequence):
            func = lh_transform.match_op(sequence.bodyTarget, "func.func")
            transform_ext.tag_subgraphs(func, kinds=["elementwise"])
            transform.yield_()
        try:
            schedule.body.operations[0].apply(payload.operation)
        except ValueError as error:
            assert "single-block" in str(error), str(error)
        else:
            raise AssertionError("Multiple blocks must not share subgraph ids")
        assert str(payload) == before
    print("Multiblock targets fail before changing the payload")


# CHECK: Multiblock targets fail before changing the payload

test_multiblock_rejection()
