# RUN: %PYTHON %s | FileCheck %s

"""Execute masked normalization and finite-V online attention on the CPU."""

import re

import numpy as np
from mlir import ir
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
from lighthouse.execution.runner import Runner
from test_dependent_reduction_legality import ATTENTION
from test_fuse_dependent_reduction_ops import flash_attention_schedule
from test_sink_normalization_past_contraction import MASK_TO_ZERO, sink_schedule
from test_sink_normalization_past_contraction import reduction_generic


def compile_payload(source, schedule=None):
    """Give a tensor-returning fixture an output buffer and lower it to the CPU."""
    result = re.search(r"return (%\w+) : tensor<([^>]+)>", source)
    assert result is not None
    value, shape = result.groups()
    source, count = re.subn(
        r"\)\s*-> tensor<" + shape + r"> \{",
        f", %dest: memref<{shape}>) {{",
        source,
        count=1,
    )
    assert count == 1
    source = source.replace(
        result.group(),
        f"bufferization.materialize_in_destination {value} in restrict writable "
        f"%dest : (tensor<{shape}>, memref<{shape}>) -> ()\n  return",
    )
    module = ir.Module.parse(source)
    if schedule is not None:
        schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    PassManager.parse(
        "builtin.module(empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "allow-return-allocs-from-loops "
        "function-boundary-type-conversion=identity-layout-map},"
        "buffer-deallocation-pipeline,convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf,convert-math-to-llvm)"
    ).run(module.operation)
    name = re.search(r"func.func @(\w+)", source).group(1)
    Runner.make_function_callable(module, name)
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    return Runner(module), name


def test_masked_normalization():
    source = MASK_TO_ZERO.replace("64", "4").replace("512", "8").replace("128", "3")
    p = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    p[0] = np.nan
    scales = np.array([np.nan, 0, np.inf, 2], dtype=np.float32)
    mask = np.array([True, True, True, False])
    v = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
    expected = np.zeros((4, 3), dtype=np.float32)
    expected[3] = (p[3] / scales[3]) @ v

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for zero_on_true in (True, False):
            payload = source
            predicates = mask
            if not zero_on_true:
                payload = payload.replace("%mv, %c0, %d", "%mv, %d, %c0")
                predicates = ~mask
            for multiply in (False, True):
                scaled_payload = (
                    payload.replace("arith.divf", "arith.mulf") if multiply else payload
                )
                expected[3] = (p[3] * scales[3] if multiply else p[3] / scales[3]) @ v
                for transformed in (False, True):
                    schedule = sink_schedule(reduction_generic) if transformed else None
                    runner, name = compile_payload(scaled_payload, schedule)
                    output = np.empty_like(expected)
                    runner.execute(name, [p, scales, predicates, v, output])
                    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-7)
    print("Masked normalization preserves zero rows and ordinary rows")


# CHECK: Masked normalization preserves zero rows and ordinary rows


def masked_contraction(source, *, separate):
    source = source.replace(
        "%mask: tensor<6xi1>)",
        "%mask: tensor<6xi1>, %key_mask: tensor<6x8xi1>)",
    )
    if separate:
        masks = """
  %row_p = linalg.generic {
      indexing_maps = [#rowcol, #row, #rowcol],
      iterator_types = ["parallel", "parallel"]}
      ins(%p, %mask : tensor<6x8xf32>, tensor<6xi1>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%pv: f32, %masked: i1, %out: f32):
    %safe = arith.select %masked, %zero, %pv : f32
    linalg.yield %safe : f32
  } -> tensor<6x8xf32>
  %masked_p = linalg.generic {
      indexing_maps = [#rowcol, #rowcol, #rowcol],
      iterator_types = ["parallel", "parallel"]}
      ins(%row_p, %key_mask : tensor<6x8xf32>, tensor<6x8xi1>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%pv: f32, %keep: i1, %out: f32):
    %i = linalg.index 0 : index
    %k = linalg.index 1 : index
    %causal = arith.cmpi sle, %k, %i : index
    %valid = arith.andi %keep, %causal : i1
    %safe = arith.select %valid, %pv, %zero : f32
    linalg.yield %safe : f32
  } -> tensor<6x8xf32>
"""
        return source.replace("  // R2b:", masks + "\n  // R2b:").replace(
            "ins(%p, %v :", "ins(%masked_p, %v :"
        )
    return (
        source.replace("[#ik, #kj, #ij]", "[#row, #ik, #kj, #ik, #ij]")
        .replace(
            "ins(%p, %v : tensor<6x8xf32>, tensor<8x3xf32>)",
            "ins(%mask, %p, %v, %key_mask : tensor<6xi1>, tensor<6x8xf32>, "
            "tensor<8x3xf32>, tensor<6x8xi1>)",
        )
        .replace(
            "^bb0(%pv: f32, %vv: f32, %out: f32):",
            "^bb0(%masked: i1, %pv: f32, %vv: f32, %keep: i1, %out: f32):",
        )
        .replace(
            "    %mul = arith.mulf %pv, %vv : f32",
            "    %i = linalg.index 0 : index\n"
            "    %k = linalg.index 2 : index\n"
            "    %causal = arith.cmpi sle, %k, %i : index\n"
            "    %valid = arith.andi %keep, %causal : i1\n"
            "    %row_safe = arith.select %masked, %zero, %pv : f32\n"
            "    %safe = arith.select %valid, %row_safe, %zero : f32\n"
            "    %mul = arith.mulf %safe, %vv : f32",
        )
        .replace(
            "[#row, #ik, #kj, #ik, #ij]",
            "[affine_map<(d0, d1, d2) -> (d0)>, #ik, #kj, #ik, #ij]",
        )
    )


def test_online_attention():
    source = ATTENTION.replace("64", "6").replace("512", "8").replace("128", "3")
    source = (
        source.replace(
            "%v: tensor<8x3xf32>)",
            "%v: tensor<8x3xf32>, %mask: tensor<6xi1>)",
        )
        .replace(
            "ins(%o, %l : tensor<6x3xf32>, tensor<6xf32>)",
            "ins(%o, %l, %mask : tensor<6x3xf32>, tensor<6xf32>, tensor<6xi1>)",
        )
        .replace("%lv: f32, %o2: f32", "%lv: f32, %masked: i1, %o2: f32")
        .replace(
            "indexing_maps = [#rowcol, #row, #rowcol],\n"
            "                         iterator_types",
            "indexing_maps = [#rowcol, #row, #row, #rowcol],\n"
            "                         iterator_types",
        )
        .replace(
            "    linalg.yield %d : f32",
            "    %safe = arith.select %masked, %zero, %d : f32\n"
            "    linalg.yield %safe : f32",
        )
    )
    x = np.array(
        [
            [0, 1, 2, 3, -1, -2, 1, 0],
            [-np.inf, -np.inf, 0, 1, 2, -np.inf, -1, 0],
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 1],
            [-np.inf] * 8,
            [1000, 1001, 999, 1002, 1000, 998, 1001, 999],
            [-1000, -999, -1001, -998, -1000, -1002, -999, -1001],
        ],
        dtype=np.float32,
    )
    v = np.arange(24, dtype=np.float32).reshape(8, 3) / 12 - 1
    mask = np.all(np.isneginf(x), axis=1)
    expected = np.zeros((6, 3), dtype=np.float32)
    scores = x[~mask]
    p = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    expected[~mask] = (p / np.sum(p, axis=1, keepdims=True)) @ v
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for tile_size in (None, 1, 2, 4):
            schedule = (
                flash_attention_schedule(tile_size) if tile_size is not None else None
            )
            runner, name = compile_payload(source, schedule)
            output = np.empty_like(expected)
            runner.execute(name, [x, v, mask, output])
            np.testing.assert_allclose(output, expected, rtol=3e-6, atol=1e-7)
        sums = np.zeros(6, dtype=np.float32)
        sums[~mask] = np.sum(p, axis=1)
        numerators = np.zeros_like(expected)
        numerators[~mask] = p @ v
        for result, reference, shape in (
            ("%l", sums, "6xf32"),
            ("%o", numerators, "6x3xf32"),
        ):
            payload = source.replace(
                "-> tensor<6x3xf32> {", f"-> tensor<{shape}> {{", 1
            ).replace(
                "return %out : tensor<6x3xf32>",
                f"return {result} : tensor<{shape}>",
            )
            runner, name = compile_payload(payload, flash_attention_schedule(2))
            output = np.empty_like(reference)
            runner.execute(name, [x, v, mask, output])
            np.testing.assert_allclose(output, reference, rtol=3e-6, atol=1e-7)
        key_mask = np.ones((6, 8), dtype=np.bool_)
        key_mask[:, 1::3] = False
        causal = np.arange(8)[None, :] <= np.arange(6)[:, None]
        weights = np.zeros((6, 8), dtype=np.float32)
        weights[~mask] = p / np.sum(p, axis=1, keepdims=True)
        masked_expected = np.where(key_mask & causal, weights, 0) @ v
        for separate in (False, True):
            payload = masked_contraction(source, separate=separate)
            for tile_size in (None, 1, 2, 4):
                schedule = flash_attention_schedule(tile_size) if tile_size else None
                runner, name = compile_payload(payload, schedule)
                output = np.empty_like(expected)
                runner.execute(name, [x, v, mask, key_mask, output])
                np.testing.assert_allclose(
                    output, masked_expected, rtol=3e-6, atol=1e-7
                )
        shared = (
            masked_contraction(source, separate=True)
            .replace("-> tensor<6x3xf32> {", "-> tensor<6x8xf32> {", 1)
            .replace(
                "return %out : tensor<6x3xf32>", "return %masked_p : tensor<6x8xf32>"
            )
        )
        expected_shared = np.zeros((6, 8), dtype=np.float32)
        expected_shared[~mask] = p
        expected_shared = np.where(key_mask & causal, expected_shared, 0)
        runner, name = compile_payload(shared, flash_attention_schedule(2))
        output = np.empty_like(expected_shared)
        runner.execute(name, [x, v, mask, key_mask, output])
        np.testing.assert_allclose(output, expected_shared, rtol=3e-6, atol=1e-7)
    print("Online attention handles masked prefixes, masked rows and large scores")
    print("Same schedule handles fused and separate row, tensor and causal masks")


# CHECK: Online attention handles masked prefixes, masked rows and large scores
# CHECK: Same schedule handles fused and separate row, tensor and causal masks


if __name__ == "__main__":
    test_masked_normalization()
    test_online_attention()
