"""Payloads shared by dependent-reduction legality and fusion tests."""

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
