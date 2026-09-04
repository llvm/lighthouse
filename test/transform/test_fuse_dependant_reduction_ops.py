# RUN: %PYTHON %s | FileCheck %s

"""Tests for `transform_ext.fuse_dependant_reduction_ops`.

The op fuses a dependency chain ``R1 -> E -> R2`` into ``R1``'s already-tiled
reduction loop, turning a two-pass reduction into an online (one-pass) one. Three
scenarios are covered:

  1. **softmax** -- ``max`` then ``sum(exp)``, with the normalizing divide reading
     ``E``'s full extent. Because ``E`` has a consumer besides ``R2``, the op fuses
     a *clone* and leaves the original outside for the divide.
  2. **flash attention** -- one ``exp`` term feeding both a row sum and a `P @ V`
     contraction. Applying the op once per consumer reduction folds both into a
     single loop, leaving only the normalization outside.
  3. **mixed-precision softmax** -- the same chain with an f16 term feeding an f32
     sum accumulator, checking the correction factor is evaluated in the wider of
     the two element types.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

#: Softmax over the trailing (reduction) axis, untiled:
#:   m = max_j x, p = exp(x - m), s = sum_j p, out = p / s
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

#: Softmax whose term is f16 while the sum accumulates in f32, the usual attention
#: mix: `m = max_j x`, `p = exp(x - m)` in f16, `s = sum_j p` in f32.
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

#: `softmax(x) @ v`, with the softmax written as an unfused two-pass reduction.
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


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def online_softmax_schedule(tile_size: int = 32) -> ir.Module:
    """Tile `R1` along its reduction axis, then fuse the `E -> R2` chain into it.

    The fusion op does not tile `R1` itself -- it reads the tile size off the
    loop's step -- so the schedule tiles first and annotates the resulting loop.
    """
    with schedule_boilerplate() as (sched, seq):
        generics = lh_transform.match_op(seq.bodyTarget, "linalg.generic")
        anyop = transform.AnyOpType.get()
        # In program order: the max (R1), the exp term (E), the sum (R2) and the
        # normalizing divide.
        r1, e, r2, _divide = transform.split_handle([anyop] * 4, generics)

        # Tile R1 along the reduction dim only, giving the `scf.for` the fusion
        # needs, and mark it as a reduction loop.
        _tiled_r1, r1_loop = structured.TileUsingForOp(r1, sizes=[0, tile_size]).results
        transform.annotate(r1_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

        fused = transform_ext.fuse_dependant_reduction_ops(e, r2, r1_loop)
        transform.annotate(fused, "online_softmax_loop")
        transform.yield_([])
    return sched


def mixed_softmax_schedule(tile_size: int = 32) -> ir.Module:
    """Same as `online_softmax_schedule`, for a payload with no trailing divide."""
    with schedule_boilerplate() as (sched, seq):
        generics = lh_transform.match_op(seq.bodyTarget, "linalg.generic")
        anyop = transform.AnyOpType.get()
        # In program order: the max (R1), the exp term (E) and the sum (R2).
        r1, e, r2 = transform.split_handle([anyop] * 3, generics)

        _tiled_r1, r1_loop = structured.TileUsingForOp(r1, sizes=[0, tile_size]).results
        transform.annotate(r1_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

        fused = transform_ext.fuse_dependant_reduction_ops(e, r2, r1_loop)
        transform.annotate(fused, "mixed_softmax_loop")
        transform.yield_([])
    return sched


def flash_attention_schedule(tile_size: int = 32) -> ir.Module:
    """Tile `R1`, then fuse both consumer reductions into its loop in turn."""
    with schedule_boilerplate() as (sched, seq):
        generics = lh_transform.match_op(seq.bodyTarget, "linalg.generic")
        anyop = transform.AnyOpType.get()
        # In program order: the max (R1), the exp term (E), the row sum (R2a), the
        # contraction (R2b) and the normalizing divide.
        r1, e, r2a, r2b, _divide = transform.split_handle([anyop] * 5, generics)

        _tiled_r1, r1_loop = structured.TileUsingForOp(r1, sizes=[0, tile_size]).results
        transform.annotate(r1_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

        # First chain: the row sum. This fuses a *clone* of E, since E still feeds
        # the contraction, and leaves the original E in place for it.
        loop = transform_ext.fuse_dependant_reduction_ops(e, r2a, r1_loop)
        # The first fusion consumed the handle to E; the original E is still the
        # contraction's operand, so re-derive it from there.
        e_again = transform.get_producer_of_operand(anyop, r2b, 0)
        # Second chain: the contraction, into the same (now replaced) loop.
        loop = transform_ext.fuse_dependant_reduction_ops(e_again, r2b, loop)
        transform.annotate(loop, "flash_attention_loop")
        transform.yield_([])
    return sched


def apply_schedule(payload_str: str, build_schedule, tile_size: int) -> ir.Module:
    """Parse `payload_str` and apply `build_schedule` at `tile_size`."""
    payload = ir.Module.parse(payload_str)
    # Bound to a local: the schedule module must outlive `apply`.
    schedule = build_schedule(tile_size)
    schedule.body.operations[0].apply(payload.operation)
    assert payload.operation.verify()
    return payload


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_softmax_structure() -> None:
    """Fuse softmax and print the resulting online loop."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply_schedule(SOFTMAX, online_softmax_schedule, 32))


# The fused loop carries three accumulators: the running max, the (stale,
# write-only) full-extent E result and the running sum.
# CHECK-LABEL: func.func @softmax
# CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
# CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
# CHECK-SAME:        iter_args(%[[MARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[EARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[SARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}})
# CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)

# R1 over this tile. %[[MOLD]] is its DPS init, i.e. the previous running max.
# CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]]
# CHECK:           %[[MNEW:.+]] = linalg.generic
# CHECK-SAME:          outs(%[[MOLD]] :
# CHECK:             arith.maximumf

# The fused clone of E, cut from the full 512 extent to the 32-wide tile and
# reading the *new* max.
# CHECK:           %[[XT:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
# CHECK:           %[[ET:.+]] = tensor.extract_slice %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
# CHECK:           %[[P:.+]] = linalg.generic
# CHECK-SAME:          ins(%[[XT]], %[[MNEW]] : tensor<64x32xf32>, tensor<64xf32>)
# CHECK-SAME:          outs(%[[ET]] : tensor<64x32xf32>)
# CHECK:             arith.subf
# CHECK:             math.exp

# The online correction: E isolated on the max, evaluated at the new and the old
# max, their ratio rescaling the running sum.
# CHECK:           %[[SOLD:.+]] = tensor.extract_slice %[[SARG]]
# CHECK:           %[[TNEW:.+]] = linalg.generic {{.*}}ins(%[[MNEW]] :
# CHECK:             arith.subf %{{.+}}, %in
# CHECK:             math.exp
# CHECK:           %[[TOLD:.+]] = linalg.generic {{.*}}ins(%[[MOLD]] :
# CHECK:             arith.subf %{{.+}}, %in
# CHECK:             math.exp
# CHECK:           %[[F:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TNEW]], %[[TOLD]]
# CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%[[SOLD]], %[[F]]

# The fused R2 accumulates this tile's terms into the rescaled running sum.
# CHECK:           %[[SNEW:.+]] = linalg.generic
# CHECK-SAME:          ins(%[[P]] : tensor<64x32xf32>)
# CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
# CHECK:             arith.addf
# CHECK:           tensor.insert_slice %[[P]] into %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
# CHECK:           tensor.insert_slice %[[SNEW]] into %[[SARG]]
# CHECK:         } {__reduction_loop__, online_softmax_loop}

# The original E survives outside the loop, reading the *final* max off it, so the
# normalizing divide sees a correctly recomputed numerator.
# CHECK:         %[[PFINAL:.+]] = linalg.generic
# CHECK-SAME:        ins(%[[X]], %[[LOOP]]#0 : tensor<64x512xf32>, tensor<64xf32>)
# CHECK:           arith.subf
# CHECK:           math.exp
# CHECK:         linalg.generic
# CHECK-SAME:        ins(%[[PFINAL]], %[[LOOP]]#2 : tensor<64x512xf32>, tensor<64xf32>)
# CHECK:           arith.divf


def test_attention_structure() -> None:
    """Fuse both consumer reductions of an attention chain into one loop."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply_schedule(ATTENTION, flash_attention_schedule, 32))


# Both consumer reductions end up in one loop, which now carries five accumulators,
# in this order: the running max, the first chain's full-extent (stale, write-only) E
# result, the running row sum, the second chain's E result, and the running
# contraction accumulator.
# CHECK-LABEL: func.func @attention
# CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
# CHECK-SAME:      %[[V:[a-zA-Z0-9_]+]]: tensor<512x128xf32>
# CHECK:         %[[LOOP:.+]]:5 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
# CHECK-SAME:        iter_args(%[[MARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}},
# CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>, tensor<64x512xf32>, tensor<64x128xf32>)

# R1 over this tile, then the first chain: E's clone, its correction and the row
# sum accumulating into the rescaled running sum.
# CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]]
# CHECK:           %[[MNEW:.+]] = linalg.generic
# CHECK-SAME:          outs(%[[MOLD]] :
# CHECK:             arith.maximumf
# CHECK:           %[[P1:.+]] = linalg.generic
# CHECK-SAME:          ins(%{{.+}}, %[[MNEW]] : tensor<64x32xf32>, tensor<64xf32>)
# CHECK:             math.exp
# CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<div>
# CHECK:           %[[LSCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>
# CHECK:           linalg.generic
# CHECK-SAME:          ins(%[[P1]] : tensor<64x32xf32>)
# CHECK-SAME:          outs(%[[LSCALED]] : tensor<64xf32>)
# CHECK:             arith.addf

# The second chain. No clone this time: the row sum is gone, so the contraction is
# E's only remaining user and E itself is fused. Its correction runs over the
# contraction's *wider* 64x128 accumulator -- the per-row factor broadcast over N.
# CHECK:           %[[P2:.+]] = linalg.generic
# CHECK-SAME:          ins(%{{.+}}, %[[MNEW]] : tensor<64x32xf32>, tensor<64xf32>)
# CHECK:             math.exp
# CHECK:           linalg.generic {{.*}}ins(%[[MNEW]] : tensor<64xf32>) outs(%{{.+}} : tensor<64x128xf32>)
# CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<div>
# CHECK:           %[[OSCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>

# The contraction reads V sliced along the shared reduction axis.
# CHECK:           %[[VT:.+]] = tensor.extract_slice %[[V]][%[[IV]], 0] [32, 128] [1, 1]
# CHECK:           linalg.generic
# CHECK-SAME:          ins(%[[P2]], %[[VT]] : tensor<64x32xf32>, tensor<32x128xf32>)
# CHECK-SAME:          outs(%[[OSCALED]] : tensor<64x128xf32>)
# CHECK:         } {__reduction_loop__, flash_attention_loop}

# Only the normalization is left outside, reading the two running results.
# CHECK:         linalg.generic
# CHECK-SAME:        ins(%[[LOOP]]#4, %[[LOOP]]#2 : tensor<64x128xf32>, tensor<64xf32>)
# CHECK:           arith.divf


def test_mixed_precision_softmax() -> None:
    """Fuse a softmax whose f16 term feeds an f32 sum accumulator."""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        print(apply_schedule(MIXED_SOFTMAX, mixed_softmax_schedule, 32))


# The running max and the term stay f16; only the sum accumulator is f32.
# CHECK-LABEL: func.func @mixed_softmax
# CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf16>
# CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
# CHECK-SAME:        iter_args(%[[MARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}} = %{{[a-zA-Z0-9_]+}}, %[[SARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}})
# CHECK-SAME:        -> (tensor<64xf16>, tensor<64x512xf16>, tensor<64xf32>)
# CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]]
# CHECK:           %[[MNEW:.+]] = linalg.generic
# CHECK:             arith.maximumf %{{.+}} : f16
# CHECK:           %[[P:.+]] = linalg.generic
# CHECK:             math.exp %{{.+}} : f16

# The correction is evaluated in f32 -- the wider of E's f16 and R2's f32
# accumulator -- so the f16 running max enters the body and is widened there.
# CHECK:           linalg.generic {{.*}}ins(%[[MNEW]] : tensor<64xf16>) outs(%{{.+}} : tensor<64xf32>)
# CHECK:             %[[WNEW:.+]] = arith.extf %in : f16 to f32
# CHECK:             arith.subf %{{.+}}, %[[WNEW]] : f32
# CHECK:             math.exp %{{.+}} : f32
# CHECK:           linalg.generic {{.*}}ins(%[[MOLD]] : tensor<64xf16>) outs(%{{.+}} : tensor<64xf32>)
# CHECK:             %[[WOLD:.+]] = arith.extf %in : f16 to f32
# CHECK:             arith.subf %{{.+}}, %[[WOLD]] : f32
# CHECK:             math.exp %{{.+}} : f32

# The factor already has R2's element type, so it rescales the running sum directly.
# CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%{{.+}} : tensor<64xf32>, tensor<64xf32>)
# CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>
# CHECK-SAME:          tensor<64xf32>, tensor<64xf32>
# CHECK:           linalg.generic
# CHECK-SAME:          ins(%[[P]] : tensor<64x32xf16>)
# CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
# CHECK:             arith.addf
# CHECK:         } {__reduction_loop__, mixed_softmax_loop}


if __name__ == "__main__":
    test_softmax_structure()
    test_attention_structure()
    test_mixed_precision_softmax()
