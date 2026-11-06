# RUN: %PYTHON %s | FileCheck %s

"""Store layout inference preserves reshaped/transposed subgroup ownership."""

import numpy as np
from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.ops.infer_gemm_store_layout import (
    _Layout,
    _infer_store_layout,
    _reshape,
)
from lighthouse.schedule import schedule_boilerplate


def source(m, n, epilogue, shape):
    output = "x".join(map(str, shape))
    return f"""
func.func @store(%a: vector<{m}x16xbf16>, %b: vector<16x{n}xbf16>,
                 %c: vector<{m}x{n}xf32>,
                 %other: vector<{output}xbf16>,
                 %desc: !xegpu.tensor_desc<{output}xbf16>) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iv = %c0 to %c1 step %c1
      iter_args(%acc = %c) -> vector<{m}x{n}xf32> {{
    %p = xegpu.dpas %a, %b, %acc
      : vector<{m}x16xbf16>, vector<16x{n}xbf16>, vector<{m}x{n}xf32>
        -> vector<{m}x{n}xf32>
    scf.yield %p : vector<{m}x{n}xf32>
  }}
  %v = arith.truncf %loop : vector<{m}x{n}xf32> to vector<{m}x{n}xbf16>
  {epilogue}
  xegpu.store_nd %result, %desc[{", ".join(["0"] * len(shape))}] : vector<{output}xbf16>,
    !xegpu.tensor_desc<{output}xbf16>
  return
}}
"""


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def run(m, sg_layout, sg_data, epilogue, shape):
    module = ir.Module.parse(source(m, 256, epilogue, shape))
    assert module.operation.verify()
    with schedule_boilerplate() as (schedule, sequence):
        dpas = lh_transform.match_op(sequence.bodyTarget, "xegpu.dpas")
        store = lh_transform.match_op(sequence.bodyTarget, "xegpu.store_nd")

        def params(values):
            return [
                transform.ParamConstantOp(
                    transform.AnyParamType.get(),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value),
                ).param
                for value in values
            ]

        layout = transform_ext.infer_gemm_store_layout(
            dpas,
            store,
            sg_layout=params(sg_layout),
            sg_data=params(sg_data),
            inst_data=[8, 16],
        )
        transform.annotate(store, "layout", param=layout)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    (store,) = operations(module, "xegpu.store_nd")
    return store.attributes["layout"]


def owners(shape, layout):
    indices = np.indices(shape)
    result = np.zeros(shape, dtype=np.int32)
    stride = 1
    for dim in layout.order:
        result += (indices[dim] // layout.sg_data[dim]) * stride
        stride *= layout.sg_layout[dim]
    return result


def test_rope_layouts():
    for m, counts, data in (
        (256, (8, 4), (32, 64)),
        (128, (4, 8), (32, 32)),
    ):
        epilogue = f"""
  %s = vector.shape_cast %v : vector<{m}x256xbf16> to vector<{m}x4x64xbf16>
  %t = vector.transpose %s, [1,0,2]
    : vector<{m}x4x64xbf16> to vector<4x{m}x64xbf16>
  %result = arith.addf %other, %t : vector<4x{m}x64xbf16>
"""
        initial = _Layout(counts, data, (8, 16), (1, 0))
        reshaped = _reshape(initial, [m, 256], [m, 4, 64])
        assert reshaped is not None
        final = reshaped.transpose([1, 0, 2])
        np.testing.assert_array_equal(
            owners((m, 256), initial).reshape(m, 4, 64).transpose(1, 0, 2),
            owners((4, m, 64), final),
        )
        fields = ", ".join(
            f"{name} = {list(getattr(final, name))}"
            for name in ("sg_layout", "sg_data", "inst_data", "order")
        )
        expected = ir.Attribute.parse(f"#xegpu.layout<{fields}>")
        assert run(m, counts, data, epilogue, [4, m, 64]) == expected
    print("Q/K stores preserve subgroup IDs through reshape and transpose")


# CHECK: Q/K stores preserve subgroup IDs through reshape and transpose


def test_elementwise_and_units():
    layout = run(
        128,
        [4, 8],
        [32, 32],
        "%result = arith.mulf %v, %other : vector<128x256xbf16>",
        [128, 256],
    )
    assert layout == ir.Attribute.parse(
        "#xegpu.layout<sg_layout=[4,8],sg_data=[32,32],inst_data=[8,16]>"
    )
    assert (
        run(
            128,
            [4, 8],
            [32, 32],
            """
  %s = vector.shape_cast %v : vector<128x256xbf16> to vector<1x128x256xbf16>
  %result = vector.shape_cast %s : vector<1x128x256xbf16> to vector<128x256xbf16>
""",
            [128, 256],
        )
        == layout
    )
    initial = _Layout((1, 4, 8), (1, 32, 32), (1, 8, 16), (2, 1, 0))
    assert _reshape(initial, [1, 128, 256], [128, 256]) == _Layout(
        (4, 8), (32, 32), (8, 16), (1, 0)
    )
    assert _reshape(initial, [1, 128, 256], [1, 128, 4, 64]) is not None
    assert _reshape(initial, [1, 128, 256], [32768]) is None
    print("Shape-preserving epilogues and unit dims work; general collapse is rejected")


# CHECK: Shape-preserving epilogues and unit dims work; general collapse is rejected


def test_unknown_and_conflicting_paths():
    initial = _Layout((4, 8), (32, 32), (8, 16), (1, 0))
    module = ir.Module.parse(
        source(
            128,
            256,
            "%result = vector.shape_cast %v : vector<128x256xbf16> to vector<32768xbf16>",
            [32768],
        )
    )
    (dpas,) = operations(module, "xegpu.dpas")
    (store,) = operations(module, "xegpu.store_nd")
    assert _infer_store_layout(dpas, store, initial) is None
    module = ir.Module.parse(
        source(
            128,
            256,
            "%result = arith.addf %v, %other : vector<128x256xbf16>",
            [128, 256],
        )
    )
    (dpas,) = operations(module, "xegpu.dpas")
    (store,) = operations(module, "xegpu.store_nd")
    # No route from this source to an unrelated store value.
    store.operands[0] = module.body.operations[0].regions[0].blocks[0].arguments[3]
    assert _infer_store_layout(dpas, store, initial) is None
    module = ir.Module.parse(
        source(
            256,
            256,
            """
  %t = vector.transpose %v, [1,0] : vector<256x256xbf16> to vector<256x256xbf16>
  %result = arith.addf %v, %t : vector<256x256xbf16>
""",
            [256, 256],
        )
    )
    (dpas,) = operations(module, "xegpu.dpas")
    (store,) = operations(module, "xegpu.store_nd")
    assert (
        _infer_store_layout(dpas, store, _Layout((8, 4), (32, 64), (8, 16), (1, 0)))
        is None
    )
    print(
        "Unsupported, unrelated and conflicting paths do not acquire a guessed layout"
    )


# CHECK: Unsupported, unrelated and conflicting paths do not acquire a guessed layout


def test_uniform_initialization():
    for value in ("0.0", "1.0"):
        result = run(
            128,
            [4, 8],
            [32, 32],
            f"%result = arith.constant dense<{value}> : vector<128x256xbf16>",
            [128, 256],
        )
        assert result == ir.Attribute.parse(
            "#xegpu.layout<sg_layout=[4,8],sg_data=[32,32],inst_data=[8,16]>"
        )
    transposed = run(
        256,
        [8, 4],
        [32, 64],
        """
  %zero = arith.constant dense<0.0> : vector<256x256xbf16>
  %t = vector.transpose %v, [1,0] : vector<256x256xbf16> to vector<256x256xbf16>
  %result = arith.addf %zero, %t : vector<256x256xbf16>
""",
        [256, 256],
    )
    assert transposed == ir.Attribute.parse(
        "#xegpu.layout<sg_layout=[4,8],sg_data=[64,32],inst_data=[16,8],order=[0,1]>"
    )
    module = ir.Module.parse(
        source(
            128,
            256,
            "%result = arith.constant dense<[0.0, 1.0]> : vector<2xbf16>",
            [2],
        )
    )
    (dpas,) = operations(module, "xegpu.dpas")
    (store,) = operations(module, "xegpu.store_nd")
    initial = _Layout((4, 8), (32, 32), (8, 16), (1, 0))
    assert _infer_store_layout(dpas, store, initial) is None
    print(
        "Uniform initialization stores accept the accumulator layout; nonuniform stores do not"
    )


# CHECK: Uniform initialization stores accept the accumulator layout; nonuniform stores do not


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    test_rope_layouts()
    test_elementwise_and_units()
    test_unknown_and_conflicting_paths()
    test_uniform_initialization()
