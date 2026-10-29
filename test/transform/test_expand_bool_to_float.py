# RUN: %PYTHON %s | FileCheck %s

"""Boolean casts become floating selects before SPIR-V code generation."""

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from test_attention_mask_numerics import compile_payload


def schedule():
    with schedule_boilerplate() as (module, sequence):
        transform_ext.expand_bool_to_float(sequence.bodyTarget)
        transform.yield_([])
    return module


def apply(module):
    sched = schedule()
    sched.body.operations[0].apply(module.operation)


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    for dtype in ("f16", "bf16", "f32", "f64"):
        for shape in ("", "vector<8x{}>", "vector<2x4x{}>"):
            src = shape.format("i1") if shape else "i1"
            dst = shape.format(dtype) if shape else dtype
            module = ir.Module.parse(f"""
                func.func @cast(%p: {src}) -> {dst} {{
                  %v = arith.uitofp %p : {src} to {dst}
                  return %v : {dst}
                }}
            """)
            apply(module)
            assert module.operation.verify()
            PassManager.parse("builtin.module(canonicalize,cse)").run(module.operation)
            text = str(module)
            assert "arith.uitofp" not in text
            assert "arith.select" in text
            assert f" : {dst}" in text
            apply(module)
            assert str(module) == text
    print("Scalar and vector i1 casts expand for f16, bf16, f32 and f64")

    module = ir.Module.parse("""
        func.func @unchanged(%i: vector<8xi32>, %b: vector<8xi1>,
                             %t: tensor<8xi1>)
            -> (vector<8xf32>, vector<8xf32>, tensor<8xf32>) {
          %u = arith.uitofp %i : vector<8xi32> to vector<8xf32>
          %s = arith.sitofp %b : vector<8xi1> to vector<8xf32>
          %v = arith.uitofp %t : tensor<8xi1> to tensor<8xf32>
          return %u, %s, %v : vector<8xf32>, vector<8xf32>, tensor<8xf32>
        }
    """)
    before = str(module)
    apply(module)
    assert str(module) == before
    print("Signed, non-boolean and tensor casts remain unchanged")

    payload = """
        func.func @cast(%p: tensor<8xi1>) -> tensor<8xf32> {
          %empty = tensor.empty() : tensor<8xf32>
          %out = linalg.generic {
              indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
              iterator_types = ["parallel"]}
              ins(%p : tensor<8xi1>) outs(%empty : tensor<8xf32>) {
          ^bb0(%b: i1, %o: f32):
            %v = arith.uitofp %b : i1 to f32
            linalg.yield %v : f32
          } -> tensor<8xf32>
          return %out : tensor<8xf32>
        }
    """
    predicates = np.array([False, True] * 4)
    for sched in (None, schedule()):
        runner, name = compile_payload(payload, sched)
        output = np.empty(8, dtype=np.float32)
        runner.execute(name, [predicates, output])
        np.testing.assert_array_equal(output, predicates.astype(np.float32))
    print("Expanded casts preserve exact zero/one CPU results")

# CHECK: Scalar and vector i1 casts expand for f16, bf16, f32 and f64
# CHECK: Signed, non-boolean and tensor casts remain unchanged
# CHECK: Expanded casts preserve exact zero/one CPU results
