# RUN: %PYTHON %s | FileCheck %s

"""Separate normalization producers stay separate from the contraction body."""

import numpy as np
from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.schedule.xegpu.lowering_common import vectorize
from lighthouse.utils.mlir import indexing_maps, linalg_inputs
from test_attention_mask_numerics import compile_payload
from test_sink_normalization_past_contraction import reduction_generic, sink_schedule


PRODUCER = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#ik = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @producer(%p: tensor<4x8xf32>, %scale: tensor<4xf32>,
                    %mask: tensor<4xi1>, %v: tensor<8x3xf32>) -> tensor<4x3xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %normalized = linalg.generic {
      indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%p, %scale : tensor<4x8xf32>, tensor<4xf32>)
      outs(%empty : tensor<4x8xf32>) {
  ^bb0(%pv: f32, %sv: f32, %out: f32):
    %n = arith.divf %pv, %sv : f32
    linalg.yield %n : f32
  } -> tensor<4x8xf32>
  %masked = linalg.generic {
      indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%normalized, %mask : tensor<4x8xf32>, tensor<4xi1>)
      outs(%empty : tensor<4x8xf32>) {
  ^bb0(%pv: f32, %mv: i1, %out: f32):
    %safe = arith.select %mv, %zero, %pv : f32
    linalg.yield %safe : f32
  } -> tensor<4x8xf32>
  %out_empty = tensor.empty() : tensor<4x3xf32>
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<4x3xf32>) -> tensor<4x3xf32>
  %o = linalg.generic {
      indexing_maps = [#ik, #kj, #ij],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%masked, %v : tensor<4x8xf32>, tensor<8x3xf32>)
      outs(%init : tensor<4x3xf32>) {
  ^bb0(%pv: f32, %vv: f32, %acc: f32):
    %product = arith.mulf %pv, %vv : f32
    %sum = arith.addf %acc, %product : f32
    linalg.yield %sum : f32
  } -> tensor<4x3xf32>
  return %o : tensor<4x3xf32>
}
"""


def reshaped_source(expand=False):
    source = PRODUCER
    if not expand:
        # The producers are batched; PV collapses those batch axes into its row axis.
        prefix, contraction = source.split("  %out_empty =", 1)
        prefix = prefix.replace("tensor<4x8xf32>", "tensor<2x2x8xf32>")
        prefix = prefix.replace("tensor<4xf32>", "tensor<2x2xf32>")
        prefix = prefix.replace("tensor<4xi1>", "tensor<2x2xi1>")
        prefix = (
            prefix.replace(
                "#id = affine_map<(d0, d1) -> (d0, d1)>",
                "#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
            )
            .replace(
                "#row = affine_map<(d0, d1) -> (d0)>",
                "#row = affine_map<(d0, d1, d2) -> (d0, d1)>",
            )
            .replace(
                'iterator_types = ["parallel", "parallel"]',
                'iterator_types = ["parallel", "parallel", "parallel"]',
            )
        )
        reshape = (
            "  %flat = tensor.collapse_shape %masked [[0, 1], [2]] "
            ": tensor<2x2x8xf32> into tensor<4x8xf32>\n"
        )
        contraction = contraction.replace("ins(%masked, %v", "ins(%flat, %v")
        return prefix + reshape + "  %out_empty =" + contraction
    source = source.replace(
        "  %out_empty =",
        "  %expanded = tensor.expand_shape %masked [[0, 1], [2]] "
        "output_shape [2, 2, 8] : tensor<4x8xf32> into tensor<2x2x8xf32>\n"
        "  %out_empty =",
    ).replace("tensor<4x3xf32>", "tensor<2x2x3xf32>")
    source = (
        source.replace(
            "#ik = affine_map<(d0, d1, d2) -> (d0, d2)>",
            "#ik = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
        )
        .replace(
            "#kj = affine_map<(d0, d1, d2) -> (d2, d1)>",
            "#kj = affine_map<(d0, d1, d2, d3) -> (d3, d2)>",
        )
        .replace(
            "#ij = affine_map<(d0, d1, d2) -> (d0, d1)>",
            "#ij = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>",
        )
        .replace(
            'iterator_types = ["parallel", "parallel", "reduction"]',
            'iterator_types = ["parallel", "parallel", "parallel", "reduction"]',
        )
        .replace(
            "ins(%masked, %v : tensor<4x8xf32>",
            "ins(%expanded, %v : tensor<2x2x8xf32>",
        )
    )
    return source


def collect(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(source):
    module = ir.Module.parse(source)
    schedule = sink_schedule(reduction_generic)
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure_and_shared_users():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (PRODUCER, reshaped_source(), reshaped_source(True)):
            shared = source.replace(
                "  %out_empty =",
                "  func.call @observe(%normalized, %masked) : "
                + (
                    "(tensor<2x2x8xf32>, tensor<2x2x8xf32>)"
                    if source == reshaped_source()
                    else "(tensor<4x8xf32>, tensor<4x8xf32>)"
                )
                + " -> ()\n  %out_empty =",
            )
            argument_type = (
                "tensor<2x2x8xf32>"
                if source == reshaped_source()
                else "tensor<4x8xf32>"
            )
            shared += (
                f"\nfunc.func private @observe({argument_type}, {argument_type})\n"
            )
            module = apply(shared)
            (call,) = collect(module, "func.call")
            original_normalization = call.operands[0].owner
            original_mask = call.operands[1].owner
            assert linalg_inputs(original_mask)[0] == original_normalization.results[0]
            assert any(
                op.operation.name == "arith.divf"
                for op in original_normalization.regions[0].blocks[0].operations
            )
            reductions = [
                op
                for op in collect(module, "linalg.generic")
                if "reduction" in str(op.iterator_types)
            ]
            (contraction,) = reductions
            assert [
                op.operation.name for op in contraction.regions[0].blocks[0].operations
            ] == ["arith.mulf", "arith.addf", "linalg.yield"]
            assert linalg_inputs(contraction)[0] != original_mask.results[0]
        module = apply(reshaped_source())
        epilogue = collect(module, "linalg.generic")[-1]
        expected = ir.AffineMapAttr.parse(
            "affine_map<(d0, d1) -> (d0 floordiv 2, d0 mod 2)>"
        ).value
        assert indexing_maps(epilogue)[1] == expected
    print(
        "Producer sinking preserves plain contractions, reshape indexing and shared users"
    )


# CHECK: Producer sinking preserves plain contractions, reshape indexing and shared users


def test_numerics():
    p = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    v = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for reshape in ("none", "collapse", "expand"):
            base = (
                PRODUCER if reshape == "none" else reshaped_source(reshape == "expand")
            )
            for masked in (False, True):
                for multiply in (False, True):
                    for zero_on_true in (False, True):
                        source = base
                        data = p.copy()
                        scale = np.array([1, 2, 3, 4], dtype=np.float32)
                        predicate = np.array([True, True, True, False])
                        if masked:
                            data[:3] = np.nan
                            scale[:3] = [np.nan, 0, np.inf]
                        else:
                            source = source.replace(
                                "ins(%masked, %v", "ins(%normalized, %v"
                            )
                            source = source.replace(
                                "shape %masked", "shape %normalized"
                            )
                        if not zero_on_true:
                            source = source.replace(
                                "arith.select %mv, %zero, %pv",
                                "arith.select %mv, %pv, %zero",
                            )
                            predicate = ~predicate
                        if multiply:
                            source = source.replace("arith.divf", "arith.mulf")
                        expected = np.zeros((4, 3), dtype=np.float32)
                        active = (
                            np.array([False, False, False, True])
                            if masked
                            else np.ones(4, dtype=bool)
                        )
                        weights = (
                            data[active] * scale[active, None]
                            if multiply
                            else data[active] / scale[active, None]
                        )
                        expected[active] = weights @ v
                        args = [data, scale, predicate, v]
                        if reshape == "collapse":
                            args = [
                                data.reshape(2, 2, 8),
                                scale.reshape(2, 2),
                                predicate.reshape(2, 2),
                                v,
                            ]
                        shape = (2, 2, 3) if reshape == "expand" else (4, 3)
                        for transformed in (False, True):
                            runner, name = compile_payload(
                                source,
                                sink_schedule(reduction_generic)
                                if transformed
                                else None,
                            )
                            output = np.empty(shape, dtype=np.float32)
                            runner.execute(name, [*args, output])
                            np.testing.assert_allclose(
                                output.reshape(4, 3), expected, rtol=2e-6, atol=1e-7
                            )
    print(
        "Producer sinking agrees on CPU with masks, divide/multiply and both reshape directions"
    )


# CHECK: Producer sinking agrees on CPU with masks, divide/multiply and both reshape directions


def test_rejections():
    crossing_reduction = (
        reshaped_source()
        .replace("shape %masked", "shape %normalized")
        .replace("[[0, 1], [2]]", "[[0], [1, 2]]")
        .replace("tensor<4x8xf32>", "tensor<2x16xf32>")
        .replace("tensor<4x3xf32>", "tensor<2x3xf32>")
        .replace("tensor<8x3xf32>", "tensor<16x3xf32>")
    )
    two_scales = (
        PRODUCER.replace(
            "%v: tensor<8x3xf32>)",
            "%v: tensor<8x3xf32>, %columns: tensor<3xf32>)",
        )
        .replace(
            "  %out_empty =",
            """  %v_empty = tensor.empty() : tensor<8x3xf32>
  %v_scaled = linalg.generic {
      indexing_maps = [#id, affine_map<(d0, d1) -> (d1)>, #id],
      iterator_types = ["parallel", "parallel"]}
      ins(%v, %columns : tensor<8x3xf32>, tensor<3xf32>)
      outs(%v_empty : tensor<8x3xf32>) {
  ^bb0(%vv: f32, %sv: f32, %out: f32):
    %scaled = arith.divf %vv, %sv : f32
    linalg.yield %scaled : f32
  } -> tensor<8x3xf32>
  %out_empty =""",
        )
        .replace("ins(%masked, %v :", "ins(%masked, %v_scaled :")
    )
    cases = [
        (
            PRODUCER.replace("ins(%zero : f32)", "ins(%one : f32)").replace(
                "  %zero =", "  %one = arith.constant 1.0 : f32\n  %zero ="
            ),
            "zero contraction init",
        ),
        (
            PRODUCER.replace("%zero, %pv", "%one, %pv").replace(
                "  %zero =", "  %one = arith.constant 1.0 : f32\n  %zero ="
            ),
            "must select zero",
        ),
        (
            PRODUCER.replace("tensor<4xf32>", "tensor<8xf32>").replace(
                "[#id, #row, #id]", "[#id, affine_map<(d0, d1) -> (d1)>, #id]", 1
            ),
            "scale varies",
        ),
        (
            PRODUCER.replace("tensor<4xi1>", "tensor<4x8xi1>")
            .replace(
                "indexing_maps = [#id, #row, #id], iterator_types",
                "indexing_maps = [#id, #id, #id], iterator_types",
            )
            .replace("tensor<4xf32>", "tensor<4x8xf32>"),
            "mask varies",
        ),
        (
            PRODUCER.replace(
                "%n = arith.divf %pv, %sv : f32",
                "%d = arith.divf %pv, %sv : f32\n    %n = math.exp %d : f32",
            ),
            "found 0",
        ),
        (crossing_reduction, "scale varies"),
        (two_scales, "found 2"),
    ]
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source, diagnostic in cases:
            module = ir.Module.parse(source)
            before = str(module)
            schedule = sink_schedule(reduction_generic)
            try:
                schedule.body.operations[0].apply(module.operation)
            except ValueError as error:
                assert diagnostic in str(error), str(error)
            else:
                raise AssertionError(f"expected rejection: {diagnostic}")
            assert str(module) == before
    print("Invalid producer chains fail without changing the payload")


# CHECK: Invalid producer chains fail without changing the payload


def test_rhs_and_mixed_precision():
    rhs = PRODUCER.replace("[#ik, #kj, #ij]", "[#kj, #ik, #ij]").replace(
        "ins(%masked, %v : tensor<4x8xf32>, tensor<8x3xf32>)",
        "ins(%v, %masked : tensor<8x3xf32>, tensor<4x8xf32>)",
    )
    mixed = (
        PRODUCER.replace("tensor<4x8xf32>", "tensor<4x8xf16>")
        .replace("tensor<4xf32>", "tensor<4xf16>")
        .replace("arith.constant 0.0 : f32", "arith.constant 0.0 : f16")
        .replace("%pv: f32", "%pv: f16")
        .replace("%sv: f32, %out: f32", "%sv: f16, %out: f16")
        .replace("%mv: i1, %out: f32", "%mv: i1, %out: f16")
        .replace("%pv, %sv : f32", "%pv, %sv : f16")
        .replace("%zero, %pv : f32", "%zero, %pv : f16")
        .replace("linalg.yield %n : f32", "linalg.yield %n : f16")
        .replace("linalg.yield %safe : f32", "linalg.yield %safe : f16")
        .replace(
            "  %init =",
            "  %zero32 = arith.constant 0.0 : f32\n  %init =",
        )
        .replace("linalg.fill ins(%zero : f32)", "linalg.fill ins(%zero32 : f32)")
        .replace(
            "    %product = arith.mulf %pv, %vv : f32",
            "    %wide = arith.extf %pv : f16 to f32\n"
            "    %product = arith.mulf %wide, %vv : f32",
        )
    )
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (rhs, mixed):
            module = apply(source)
            epilogue = collect(module, "linalg.generic")[-1]
            if source == mixed:
                (cast,) = [
                    op
                    for op in epilogue.regions[0].blocks[0].operations
                    if op.operation.name == "arith.extf"
                ]
                assert str(cast.operands[0].type) == "f16"
                assert str(cast.results[0].type) == "f32"
            p = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
            if source == mixed:
                p = p.astype(np.float16)
            scales = np.array([1, 2, 4, 8], dtype=p.dtype)
            mask = np.array([False, True, False, False])
            v = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
            expected = np.where(mask[:, None], 0, p / scales[:, None]) @ v
            runner, name = compile_payload(source, sink_schedule(reduction_generic))
            output = np.empty((4, 3), dtype=np.float32)
            runner.execute(name, [p, scales, mask, v, output])
            np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-7)
    print("Either contraction operand and mixed-precision scales are supported")


# CHECK: Either contraction operand and mixed-precision scales are supported


def test_vectorization():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(PRODUCER)
        with schedule_boilerplate() as (schedule, sequence):
            root = sequence.bodyTarget
            transform_ext.sink_normalization_past_contraction(reduction_generic(root))
            func = lh_transform.match_op(root, "func.func")
            transform.apply_dce(func)
            vectorize(root, payload_func=func)
            transform.yield_([])
        schedule.body.operations[0].apply(module.operation)
        assert module.operation.verify()
        (contract,) = collect(module, "vector.contract")
        assert [str(v.type) for v in contract.operands] == [
            "vector<4x8xf32>",
            "vector<8x3xf32>",
            "vector<4x3xf32>",
        ]
        selects = collect(module, "arith.select")
        assert all(ir.VectorType(op.results[0].type).rank <= 2 for op in selects)
    print("Separate producer masking vectorizes without 3-D contraction operands")


# CHECK: Separate producer masking vectorizes without 3-D contraction operands


if __name__ == "__main__":
    test_structure_and_shared_users()
    test_numerics()
    test_rejections()
    test_rhs_and_mixed_precision()
    test_vectorization()
