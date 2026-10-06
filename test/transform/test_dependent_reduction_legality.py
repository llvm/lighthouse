# RUN: %PYTHON %s | FileCheck %s

"""Exercise fusion legality without applying the fusion rewrite."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.utils import (
    dependent_reduction_legality as legality,
)
from lighthouse.schedule.builders import schedule_boilerplate


SOFTMAX = """
#rowcol = affine_map<(d0, d1) -> (d0, d1)>
#row    = affine_map<(d0, d1) -> (d0)>

func.func @softmax(%x: tensor<64x512xf32>) -> tensor<64x512xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %row_init = tensor.empty() : tensor<64xf32>
  %full_init = tensor.empty() : tensor<64x512xf32>

  // R1: m = max_j x
  %m_init = linalg.fill ins(%ninf : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %m = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<64x512xf32>) outs(%m_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mx = arith.maximumf %in, %out : f32
    linalg.yield %mx : f32
  } -> tensor<64xf32>

  // E: p = exp(x - m)
  %p = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<64x512xf32>, tensor<64xf32>)
      outs(%full_init : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2: s = sum_j p
  %s_init = linalg.fill ins(%zero : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %s = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<64x512xf32>) outs(%s_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>

  // The normalizing divide, downstream of the chain. This is the extra consumer
  // of `E` that forces the fusion to work on a clone.
  %out = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                         iterator_types = ["parallel", "parallel"]}
      ins(%p, %s : tensor<64x512xf32>, tensor<64xf32>)
      outs(%full_init : tensor<64x512xf32>) {
  ^bb0(%in: f32, %sv: f32, %o: f32):
    %d = arith.divf %in, %sv : f32
    linalg.yield %d : f32
  } -> tensor<64x512xf32>
  return %out : tensor<64x512xf32>
}
"""

MIXED_SOFTMAX = """
#rowcol = affine_map<(d0, d1) -> (d0, d1)>
#row    = affine_map<(d0, d1) -> (d0)>

func.func @mixed_softmax(%x: tensor<64x512xf16>) -> tensor<64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFC00 : f16
  %row_init_f16 = tensor.empty() : tensor<64xf16>
  %row_init_f32 = tensor.empty() : tensor<64xf32>
  %full_init = tensor.empty() : tensor<64x512xf16>

  // R1: m = max_j x, in f16.
  %m_init = linalg.fill ins(%ninf : f16) outs(%row_init_f16 : tensor<64xf16>) -> tensor<64xf16>
  %m = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<64x512xf16>) outs(%m_init : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %mx = arith.maximumf %in, %out : f16
    linalg.yield %mx : f16
  } -> tensor<64xf16>

  // E: p = exp(x - m), also f16.
  %p = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<64x512xf16>, tensor<64xf16>)
      outs(%full_init : tensor<64x512xf16>) {
  ^bb0(%in: f16, %mv: f16, %out: f16):
    %d = arith.subf %in, %mv : f16
    %e = math.exp %d : f16
    linalg.yield %e : f16
  } -> tensor<64x512xf16>

  // R2: s = sum_j p, widened into an f32 accumulator.
  %s_init = linalg.fill ins(%zero : f32) outs(%row_init_f32 : tensor<64xf32>) -> tensor<64xf32>
  %s = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<64x512xf16>) outs(%s_init : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %w = arith.extf %in : f16 to f32
    %a = arith.addf %w, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>
  return %s : tensor<64xf32>
}
"""

ATTENTION = """
#rowcol = affine_map<(d0, d1) -> (d0, d1)>
#row    = affine_map<(d0, d1) -> (d0)>
#ik     = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj     = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij     = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @attention(%x: tensor<64x512xf32>, %v: tensor<512x128xf32>)
    -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %row_init = tensor.empty() : tensor<64xf32>
  %p_init = tensor.empty() : tensor<64x512xf32>

  // R1: m = max_k x
  %m_init = linalg.fill ins(%ninf : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %m = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<64x512xf32>) outs(%m_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mx = arith.maximumf %in, %out : f32
    linalg.yield %mx : f32
  } -> tensor<64xf32>

  // E: p = exp(x - m), read by BOTH reductions below.
  %p = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<64x512xf32>, tensor<64xf32>)
      outs(%p_init : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2a: l = sum_k p
  %l_init = linalg.fill ins(%zero : f32) outs(%row_init : tensor<64xf32>) -> tensor<64xf32>
  %l = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<64x512xf32>) outs(%l_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>

  // R2b: o = p @ v, a contraction over the same axis.
  %o_init = tensor.empty() : tensor<64x128xf32>
  %o_fill = linalg.fill ins(%zero : f32) outs(%o_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %v : tensor<64x512xf32>, tensor<512x128xf32>)
      outs(%o_fill : tensor<64x128xf32>) {
  ^bb0(%pv: f32, %vv: f32, %out: f32):
    %mul = arith.mulf %pv, %vv : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<64x128xf32>

  // The deferred normalization, downstream of both reductions.
  %out_init = tensor.empty() : tensor<64x128xf32>
  %out = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                         iterator_types = ["parallel", "parallel"]}
      ins(%o, %l : tensor<64x128xf32>, tensor<64xf32>)
      outs(%out_init : tensor<64x128xf32>) {
  ^bb0(%in: f32, %lv: f32, %o2: f32):
    %d = arith.divf %in, %lv : f32
    linalg.yield %d : f32
  } -> tensor<64x128xf32>
  return %out : tensor<64x128xf32>
}
"""


def check_case(name: str, source: str, *, r2_index: int = 1, tile_size: int = 32):
    """Tile R1, then report only the legality check's result."""
    print(f"Case: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, sequence):
            generics = lh_transform.match_op(sequence.bodyTarget, "linalg.generic")
            r1 = transform_ext.extract_handle(generics, 0)
            _, loop = structured.TileUsingForOp(r1, sizes=[0, tile_size]).results
            transform.annotate(loop, legality.REDUCTION_LOOP_ATTR_NAME)
            transform.yield_([])
        schedule.operation.verify()
        schedule.body.operations[0].apply(payload.operation)
        payload.operation.verify()

        func = next(
            op for op in payload.body.operations if op.operation.name == "func.func"
        )
        body = list(func.operation.regions[0].blocks[0].operations)
        r1_loop = next(op for op in body if op.operation.name == "scf.for")
        top_level_generics = [
            op for op in body if op.operation.name == "linalg.generic"
        ]
        e, r2 = top_level_generics[0], top_level_generics[r2_index]
        inner = legality.map_loop_results_to_inner_reductions(r1_loop)
        try:
            dim, tile = legality.check_legal_fusion_triple(r1_loop, inner, e, r2)
        except legality.FusionRejected as error:
            print(f"Not legal to fuse: {error}")
        else:
            print(f"Legal to fuse: E dim {dim}, tile {tile}")


def replace_once(source: str, before: str, after: str) -> str:
    """Create one targeted invalid variant of a valid payload."""
    assert source.count(before) == 1
    return source.replace(before, after, 1)


def softmax_with_term(body: str) -> str:
    """Replace the scalar body of softmax's elementwise term."""
    return replace_once(
        SOFTMAX,
        "    %e = math.exp %d : f32\n    linalg.yield %e : f32",
        body,
    )


def main() -> None:
    # CHECK-LABEL: Case: softmax
    # CHECK-NEXT: Legal to fuse: E dim 1, tile 32
    check_case("softmax", SOFTMAX)

    # CHECK-LABEL: Case: earlier R1 user
    # CHECK-NEXT: Not legal to fuse: user of an R1 result does not post-dominate E
    check_case(
        "earlier R1 user",
        replace_once(
            SOFTMAX,
            "  // E: p = exp(x - m)",
            "  %early = tensor.cast %m : tensor<64xf32> to tensor<64xf32>\n\n"
            "  // E: p = exp(x - m)",
        ),
    )

    # CHECK-LABEL: Case: attention row sum
    # CHECK-NEXT: Legal to fuse: E dim 1, tile 32
    check_case("attention row sum", ATTENTION)

    # CHECK-LABEL: Case: attention contraction
    # CHECK-NEXT: Legal to fuse: E dim 1, tile 32
    check_case("attention contraction", ATTENTION, r2_index=2)

    # CHECK-LABEL: Case: mixed precision
    # CHECK-NEXT: Legal to fuse: E dim 1, tile 32
    check_case("mixed precision", MIXED_SOFTMAX)

    # CHECK-LABEL: Case: nonseparable term
    # CHECK-NEXT: Not legal to fuse: E is not multiplicatively separable
    check_case(
        "nonseparable term",
        replace_once(SOFTMAX, "%e = math.exp %d : f32", "%e = arith.mulf %d, %d : f32"),
    )

    # CHECK-LABEL: Case: absf remains supported
    # CHECK-NEXT: Legal to fuse: E dim 1, tile 32
    check_case(
        "absf remains supported",
        softmax_with_term(
            "    %e = math.exp %d : f32\n"
            "    %absolute = math.absf %e : f32\n"
            "    linalg.yield %absolute : f32"
        ),
    )

    # CHECK-LABEL: Case: wrong reduction input
    # CHECK-NEXT: Not legal to fuse: no R2 input is E's result
    check_case(
        "wrong reduction input",
        replace_once(
            SOFTMAX,
            "ins(%p : tensor<64x512xf32>) outs(%s_init",
            "ins(%x : tensor<64x512xf32>) outs(%s_init",
        ),
    )

    # CHECK-LABEL: Case: nonzero R2 init
    # CHECK-NEXT: Not legal to fuse: R2's init is not the additive identity
    check_case(
        "nonzero R2 init",
        replace_once(
            SOFTMAX,
            "%s_init = linalg.fill ins(%zero : f32)",
            "%s_init = linalg.fill ins(%ninf : f32)",
        ),
    )

    # CHECK-LABEL: Case: nondividing tile
    # CHECK-NEXT: Not legal to fuse: tile size 30 does not evenly divide
    check_case("nondividing tile", SOFTMAX, tile_size=30)


if __name__ == "__main__":
    main()
