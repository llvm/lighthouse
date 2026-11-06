# RUN: %PYTHON %s | FileCheck %s

"""Fold the boolean-log mask workaround without touching general logarithms."""

import numpy as np
from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from test_attention_mask_numerics import compile_payload


def schedule():
    with schedule_boilerplate() as (module, sequence):
        transform_ext.fold_log_of_bool(sequence.bodyTarget)
        transform.yield_()
    return module


def apply(module):
    sched = schedule()
    sched.body.operations[0].apply(module.operation)


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    for dtype in ("f16", "bf16", "f32", "f64"):
        for shape in ("{}", "vector<8x{}>", "vector<2x4x{}>"):
            src, dst = shape.format("i1"), shape.format(dtype)
            module = ir.Module.parse(f"""
func.func @mask(%p: {src}) -> ({dst}, {dst}) {{
  %cast = arith.uitofp %p : {src} to {dst}
  %log = math.log %cast : {dst}
  return %log, %cast : {dst}, {dst}
}}
""")
            apply(module)
            assert module.operation.verify()
            text = str(module)
            assert "math.log" not in text
            assert "arith.select" in text
            assert "arith.uitofp" in text, "Shared casts must survive"
            selects = []

            def visit(op):
                if op.name == "arith.select":
                    selects.append(op)
                return ir.WalkResult.ADVANCE

            module.operation.walk(visit)
            assert len(selects) == 1
            for operand, expected in zip(selects[0].operands[1:], (0.0, -np.inf)):
                attr = operand.owner.attributes["value"]
                if isinstance(attr, ir.DenseElementsAttr):
                    attr = attr.get_splat_value()
                value = ir.FloatAttr(attr).value
                assert value == expected
                if expected == 0:
                    assert not np.signbit(value)
            apply(module)
            assert str(module) == text
    print("Scalar/vector boolean logs fold for f16, bf16, f32 and f64")

    module = ir.Module.parse("""
func.func @unchanged(%i: i32, %b: i1, %f: f32, %t: tensor<8xi1>)
    -> (f32, f32, f32, tensor<8xf32>) {
  %u = arith.uitofp %i : i32 to f32
  %s = arith.sitofp %b : i1 to f32
  %v = arith.uitofp %t : tensor<8xi1> to tensor<8xf32>
  %l1 = math.log %u : f32
  %l2 = math.log %s : f32
  %l3 = math.log %f : f32
  %l4 = math.log %v : tensor<8xf32>
  return %l1, %l2, %l3, %l4 : f32, f32, f32, tensor<8xf32>
}
""")
    before = str(module)
    apply(module)
    assert str(module) == before
    print("Signed, wider-integer, arbitrary-float and tensor logs stay unchanged")

    payload = """
func.func @mask(%p: tensor<8xi1>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %out = linalg.generic {
      indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}
      ins(%p : tensor<8xi1>) outs(%empty : tensor<8xf32>) {
  ^bb0(%b: i1, %o: f32):
    %v = arith.uitofp %b : i1 to f32
    %l = math.log %v : f32
    linalg.yield %l : f32
  } -> tensor<8xf32>
  return %out : tensor<8xf32>
}
"""
    predicates = np.array([False, True] * 4)
    for sched in (None, schedule()):
        runner, name = compile_payload(payload, sched)
        output = np.empty(8, dtype=np.float32)
        runner.execute(name, [predicates, output])
        assert np.isneginf(output[~predicates]).all()
        assert (output[predicates] == 0).all()
        assert not np.signbit(output[predicates]).any()
    print("CPU execution preserves exact +zero/-infinity mask values")

# CHECK: Scalar/vector boolean logs fold for f16, bf16, f32 and f64
# CHECK: Signed, wider-integer, arbitrary-float and tensor logs stay unchanged
# CHECK: CPU execution preserves exact +zero/-infinity mask values
