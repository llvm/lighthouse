# RUN: %PYTHON %s | FileCheck %s

"""Tests for the VNNI layout utilities."""

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.utils.mlir import is_vnni_layout, vnni_reduction_dims


def contract(name: str, maps: list[str], a: str, b: str, out: str) -> str:
    return f"""
#mA = affine_map<{maps[0]}>
#mB = affine_map<{maps[1]}>
#mC = affine_map<{maps[2]}>
func.func @{name}(%a: tensor<{a}>, %b: tensor<{b}>, %o: tensor<{out}>) -> tensor<{out}> {{
  %c = linalg.contract indexing_maps = [#mA, #mB, #mC]
      ins(%a, %b : tensor<{a}>, tensor<{b}>) outs(%o : tensor<{out}>) -> tensor<{out}>
  return %c : tensor<{out}>
}}
"""


VNNI_MAPS = [
    "(m, n, k, v) -> (m, k, v)",
    "(m, n, k, v) -> (k, n, v)",
    "(m, n, k, v) -> (m, n)",
]
PAYLOADS = {
    # A [M, K/2, 2], B [K/2, N, 2].
    "vnni": contract("vnni", VNNI_MAPS, "64x16x2xbf16", "16x64x2xbf16", "64x64xf32"),
    # An outer K-block reduction dim before the VNNI dims.
    "blocked_vnni": contract(
        "blocked_vnni",
        [
            "(m, n, kb, k, v) -> (kb, m, k, v)",
            "(m, n, kb, k, v) -> (kb, k, n, v)",
            "(m, n, kb, k, v) -> (m, n)",
        ],
        "4x64x8x4xi8",
        "4x8x64x4xi8",
        "64x64xi32",
    ),
    # B is [N, K/2, 2].
    "transposed_b": contract(
        "transposed_b",
        [
            "(m, n, k, v) -> (m, k, v)",
            "(m, n, k, v) -> (n, k, v)",
            "(m, n, k, v) -> (m, n)",
        ],
        "64x16x2xbf16",
        "64x16x2xbf16",
        "64x64xf32",
    ),
    # The VNNI factor must be even.
    "odd_factor": contract(
        "odd_factor", VNNI_MAPS, "64x16x3xf32", "16x64x3xf32", "64x64xf32"
    ),
    # The VNNI factor must be static.
    "dynamic_factor": contract(
        "dynamic_factor", VNNI_MAPS, "64x16x?xf32", "16x64x?xf32", "64x64xf32"
    ),
    # Plain 2-D matmul.
    "matmul": contract(
        "matmul",
        ["(m, n, k) -> (m, k)", "(m, n, k) -> (k, n)", "(m, n, k) -> (m, n)"],
        "64x32xbf16",
        "32x64xbf16",
        "64x64xf32",
    ),
}

with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    for name, payload in PAYLOADS.items():
        module = ir.Module.parse(payload)
        op = module.body.operations[0].regions[0].blocks[0].operations[0]
        print(
            f"{name}: vnni {is_vnni_layout(op)}"
            f" factor_2 {is_vnni_layout(op, blocking_factor=2)}"
            f" dims {vnni_reduction_dims(op)}"
        )
    # Not a linalg op.
    print("non_linalg: vnni", is_vnni_layout(module.body.operations[0]))

# CHECK: vnni: vnni True factor_2 True dims VnniReductionDims(k_dim=2, vnni_dim=3, factor=2)
# CHECK: blocked_vnni: vnni True factor_2 False dims VnniReductionDims(k_dim=3, vnni_dim=4, factor=4)
# CHECK: transposed_b: vnni False factor_2 False dims None
# CHECK: odd_factor: vnni False factor_2 False dims None
# CHECK: dynamic_factor: vnni False factor_2 False dims None
# CHECK: matmul: vnni False factor_2 False dims None
# CHECK: non_linalg: vnni False
