# RUN: %PYTHON %s | FileCheck %s

"""Structural and CPU regressions for online attention's optional row mask."""

import numpy as np
from mlir import ir
from mlir.dialects import arith, linalg, scf, transform
from mlir.dialects.transform import structured
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.schedule.xegpu.attention_fusion import transform_to_flash_attention
from lighthouse.utils.mlir import indexing_maps, linalg_inputs, reduction_dims
from test_attention_mask_numerics import compile_payload
from test_dependent_reduction_legality import ATTENTION


UNMASKED = ATTENTION.replace("64", "6").replace("512", "8").replace("128", "3")
MASK = """
  %true = arith.constant true
  %ninf64 = arith.constant 0xFFF0000000000000 : f64
  %mask_empty = tensor.empty() : tensor<6xi1>
  %mask_init = linalg.fill ins(%true : i1)
      outs(%mask_empty : tensor<6xi1>) -> tensor<6xi1>
  %mask = linalg.generic {
      indexing_maps = [#rowcol, #row],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<6x8xf32>) outs(%mask_init : tensor<6xi1>) {
  ^bb0(%score: f32, %acc: i1):
    %wide = arith.extf %score : f32 to f64
    %empty = arith.cmpf oeq, %wide, %ninf64 : f64
    %all = arith.andi %empty, %acc : i1
    linalg.yield %all : i1
  } -> tensor<6xi1>
"""
MASKED = (
    UNMASKED.replace("  // R1:", MASK + "\n  // R1:")
    .replace(
        "[#ik, #kj, #ij]",
        "[affine_map<(d0, d1, d2) -> (d0)>, #ik, #kj, #ij]",
    )
    .replace(
        "ins(%p, %v : tensor<6x8xf32>, tensor<8x3xf32>)",
        "ins(%mask, %p, %v : tensor<6xi1>, tensor<6x8xf32>, tensor<8x3xf32>)",
    )
    .replace(
        "^bb0(%pv: f32, %vv: f32, %out: f32):",
        "^bb0(%masked: i1, %pv: f32, %vv: f32, %out: f32):",
    )
    .replace(
        "    %mul = arith.mulf %pv, %vv : f32",
        "    %safe = arith.select %masked, %zero, %pv : f32\n"
        "    %mul = arith.mulf %safe, %vv : f32",
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
        "    %guarded = arith.select %masked, %in, %d : f32\n"
        "    linalg.yield %guarded : f32",
    )
)
# The mask select as a separate elementwise producer of a plain PV contraction.
SEPARATE = (
    MASKED.replace(
        "[affine_map<(d0, d1, d2) -> (d0)>, #ik, #kj, #ij]", "[#ik, #kj, #ij]"
    )
    .replace(
        "ins(%mask, %p, %v : tensor<6xi1>, tensor<6x8xf32>, tensor<8x3xf32>)",
        "ins(%pm, %v : tensor<6x8xf32>, tensor<8x3xf32>)",
    )
    .replace(
        "^bb0(%masked: i1, %pv: f32, %vv: f32, %out: f32):",
        "^bb0(%pv: f32, %vv: f32, %out: f32):",
    )
    .replace(
        "    %safe = arith.select %masked, %zero, %pv : f32\n"
        "    %mul = arith.mulf %safe, %vv : f32",
        "    %mul = arith.mulf %pv, %vv : f32",
    )
    .replace(
        "  // R2b:",
        """  %pm = linalg.generic {indexing_maps = [#row, #rowcol, #rowcol],
                        iterator_types = ["parallel", "parallel"]}
      ins(%mask, %p : tensor<6xi1>, tensor<6x8xf32>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%masked: i1, %pv: f32, %o: f32):
    %safe = arith.select %masked, %zero, %pv : f32
    linalg.yield %safe : f32
  } -> tensor<6x8xf32>

  // R2b:""",
    )
)
assert "%pm" in SEPARATE and "%safe = arith.select %masked, %zero, %pv" in SEPARATE
# Without broad elementwise fusion: the -inf test is its own generic and the
# select's zero comes from a splat tensor input.
SPLIT = (
    SEPARATE.replace(
        "  %mask = linalg.generic {",
        """  %is_ninf_init = tensor.empty() : tensor<6x8xi1>
  %is_ninf = linalg.generic {
      indexing_maps = [#rowcol, #rowcol],
      iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<6x8xf32>) outs(%is_ninf_init : tensor<6x8xi1>) {
  ^bb0(%score: f32, %o: i1):
    %wide = arith.extf %score : f32 to f64
    %empty = arith.cmpf oeq, %wide, %ninf64 : f64
    linalg.yield %empty : i1
  } -> tensor<6x8xi1>
  %mask = linalg.generic {""",
    )
    .replace(
        "      ins(%x : tensor<6x8xf32>) outs(%mask_init : tensor<6xi1>) {\n"
        "  ^bb0(%score: f32, %acc: i1):\n"
        "    %wide = arith.extf %score : f32 to f64\n"
        "    %empty = arith.cmpf oeq, %wide, %ninf64 : f64\n"
        "    %all = arith.andi %empty, %acc : i1",
        "      ins(%is_ninf : tensor<6x8xi1>) outs(%mask_init : tensor<6xi1>) {\n"
        "  ^bb0(%empty: i1, %acc: i1):\n"
        "    %all = arith.andi %empty, %acc : i1",
    )
    .replace(
        "  %pm = linalg.generic {indexing_maps = [#row, #rowcol, #rowcol],",
        "  %zeros = arith.constant dense<0.0> : tensor<6x8xf32>\n"
        "  %pm = linalg.generic {indexing_maps = [#row, #rowcol, #rowcol, #rowcol],",
    )
    .replace(
        "      ins(%mask, %p : tensor<6xi1>, tensor<6x8xf32>)",
        "      ins(%mask, %zeros, %p : tensor<6xi1>, tensor<6x8xf32>, tensor<6x8xf32>)",
    )
    .replace(
        "  ^bb0(%masked: i1, %pv: f32, %o: f32):\n"
        "    %safe = arith.select %masked, %zero, %pv : f32",
        "  ^bb0(%masked: i1, %z: f32, %pv: f32, %o: f32):\n"
        "    %safe = arith.select %masked, %z, %pv : f32",
    )
)
assert "%is_ninf :" in SPLIT and "%z, %pv" in SPLIT


def rewrite_schedule():
    with schedule_boilerplate() as (schedule, sequence):
        loop = lh_transform.match_op(sequence.bodyTarget, "scf.for")
        transform_ext.rewrite_attention_row_mask(loop)
        transform.yield_([])
    return schedule


def online_schedule(tile_size, *, rewrite=True):
    with schedule_boilerplate() as (schedule, sequence):
        root = sequence.bodyTarget
        anyop = transform.AnyOpType.get()
        maximum = transform.get_parent_op(
            anyop,
            lh_transform.match_op(root, "arith.maximumf"),
            op_name="linalg.generic",
        )
        exp = transform.get_parent_op(
            anyop,
            lh_transform.match_op(root, "math.exp"),
            op_name="linalg.generic",
        )
        addition = transform_ext.extract_handle(
            lh_transform.match_op(root, "arith.addf"), 0
        )
        row_sum = transform.get_parent_op(anyop, addition, op_name="linalg.generic")
        pv = transform_ext.filter_contraction_ops(
            lh_transform.match_op(root, "linalg.generic")
        )
        _, loop = structured.TileUsingForOp(maximum, sizes=[0, tile_size]).results
        transform.annotate(loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)
        loop = transform_ext.fuse_dependent_reduction_ops(exp, row_sum, loop)
        exp = transform.get_parent_op(
            anyop,
            transform_ext.extract_handle(lh_transform.match_op(root, "math.exp"), -1),
            op_name="linalg.generic",
        )
        loop = transform_ext.fuse_dependent_reduction_ops(exp, pv, loop)
        if rewrite:
            transform_ext.rewrite_attention_row_mask(loop)
        transform.yield_([])
    return schedule


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(source, *, rewrite=True):
    module = ir.Module.parse(source)
    schedule = online_schedule(2, rewrite=rewrite)
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (
            MASKED,
            MASKED.replace("    %wide = arith.extf %score : f32 to f64\n", "").replace(
                "%wide, %ninf64 : f64", "%score, %ninf : f32"
            ),
        ):
            module = apply(source)
            PassManager.parse("builtin.module(canonicalize,cse)").run(module.operation)
            assert not operations(module, "arith.andi")
            (loop,) = operations(module, "scf.for")
            (maximum,) = operations(module, "arith.maximumf")
            max_generic = maximum.operation.parent.opview
            predicates = [
                op
                for op in operations(module, "linalg.generic")
                if str(ir.ShapedType(op.results[0].type).element_type) == "i1"
            ]
            assert len(predicates) == 2
            prefix = next(
                op for op in predicates if op.operation.parent == loop.operation
            )
            final = next(op for op in predicates if op != prefix)
            assert linalg_inputs(prefix)[0] == max_generic.results[0]
            assert linalg_inputs(final)[0] == loop.results[0]
            assert not reduction_dims(prefix) and not reduction_dims(final)
            pv_uses = list(prefix.results[0].uses)
            assert len(pv_uses) == 1
            assert reduction_dims(pv_uses[0].owner.opview)
            # The normalization still selects the numerator on masked rows.
            normalization = list(final.results[0].uses)[0].owner.opview
            select = next(
                op
                for op in normalization.regions[0].blocks[0].operations
                if isinstance(op, arith.SelectOp)
            )
            assert select.operands[1] == normalization.regions[0].blocks[0].arguments[0]
    print("Row mask uses running and final maxima without a boolean reduction")


# CHECK: Row mask uses running and final maxima without a boolean reduction


def test_preservation():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        variants = [
            UNMASKED,
            MASKED.replace("%ninf64 : f64", "%other_ninf : f64").replace(
                "  %true =", "  %other_ninf = arith.constant -1.0 : f64\n  %true ="
            ),
            MASKED.replace("arith.constant true", "arith.constant false"),
            MASKED.replace("arith.andi %empty", "arith.ori %empty"),
            MASKED.replace("arith.cmpf oeq, %wide", "arith.cmpf one, %wide"),
            MASKED.replace(
                "arith.select %masked, %zero, %pv",
                "arith.select %masked, %pv, %zero",
            ),
            MASKED.replace(
                "ins(%x : tensor<6x8xf32>) outs(%mask_init",
                "ins(%other : tensor<6x8xf32>) outs(%mask_init",
            ).replace(
                "%v: tensor<8x3xf32>)",
                "%v: tensor<8x3xf32>, %other: tensor<6x8xf32>)",
            ),
        ]
        for source in variants:
            module = apply(source, rewrite=False)
            before = str(module)
            schedule = rewrite_schedule()
            schedule.body.operations[0].apply(module.operation)
            assert str(module) == before
            assert module.operation.verify()

        module = apply(MASKED, rewrite=False)
        # Without the guarded exponential the prefix substitution is not legal.
        for op in operations(module, "arith.select"):
            if isinstance(op.operands[0].owner, arith.CmpFOp):
                if op.operands[2].type == ir.F32Type.get():
                    op.operands[1] = op.operands[2]
        before = str(module)
        schedule = rewrite_schedule()
        schedule.body.operations[0].apply(module.operation)
        assert str(module) == before
        assert module.operation.verify()

        shared = (
            MASKED.replace(
                "  // R1:",
                "  func.call @observe(%mask) : (tensor<6xi1>) -> ()\n  // R1:",
            )
            + "\nfunc.func private @observe(tensor<6xi1>)\n"
        )
        module = apply(shared)
        PassManager.parse("builtin.module(canonicalize,cse)").run(module.operation)
        assert len(operations(module, "arith.andi")) == 1
        (call,) = operations(module, "func.call")
        assert isinstance(call.operands[0].owner, linalg.GenericOp)
        assert reduction_dims(call.operands[0].owner)
        (loop,) = operations(module, "scf.for")
        assert all(
            operand != call.operands[0]
            for op in loop.body.operations
            for operand in op.operands
        )
    print("Unmasked, unmatched and shared mask users are preserved")


# CHECK: Unmasked, unmatched and shared mask users are preserved


def test_numerics():
    scores = np.array(
        [
            [0, 1, 2, 3, -1, -2, 1, 0],
            [-np.inf, -np.inf, 0, 1, 2, -np.inf, -1, 0],
            [-np.inf] * 6 + [0, 1],
            [-np.inf] * 8,
            [1000, 1001, 999, 1002, 1000, 998, 1001, 999],
            [-1000, -999, -1001, -998, -1000, -1002, -999, -1001],
        ],
        dtype=np.float32,
    )
    values = np.arange(24, dtype=np.float32).reshape(8, 3) / 12 - 1
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (MASKED, SEPARATE, SPLIT, UNMASKED):
            x = scores.copy()
            if source == UNMASKED:
                x[3] = 0
            valid = ~np.all(np.isneginf(x), axis=1)
            p = np.exp(x[valid] - np.max(x[valid], axis=1, keepdims=True))
            expected = np.zeros((6, 3), dtype=np.float32)
            expected[valid] = (p / np.sum(p, axis=1, keepdims=True)) @ values
            for tile in (None, 1, 2, 4, 8):
                schedule = online_schedule(tile) if tile is not None else None
                runner, name = compile_payload(source, schedule)
                output = np.empty_like(expected)
                runner.execute(name, [x, values, output])
                np.testing.assert_allclose(output, expected, rtol=3e-6, atol=1e-7)
                if source != UNMASKED:
                    np.testing.assert_array_equal(output[3], 0)
    print(
        "CPU attention agrees for masked prefixes, rows and no mask at four tile sizes"
    )


# CHECK: CPU attention agrees for masked prefixes, rows and no mask at four tile sizes


def check_separate_mask_producer(source):
    """PV keeps a plain multiply-add body; the mask select is tiled before it."""
    module = apply(source)
    PassManager.parse("builtin.module(canonicalize,cse)").run(module.operation)
    assert not operations(module, "arith.andi")
    (loop,) = operations(module, "scf.for")
    in_loop = [
        op
        for op in operations(module, "linalg.generic")
        if op.operation.parent == loop.operation
    ]
    (pv,) = [op for op in in_loop if len(op.iterator_types) == 3]
    assert sorted(op.name for op in pv.regions[0].blocks[0].operations) == [
        "arith.addf",
        "arith.mulf",
        "linalg.yield",
    ]
    mask_select = linalg_inputs(pv)[0].owner.opview
    assert mask_select.operation.parent == loop.operation
    assert not reduction_dims(mask_select)
    assert any(
        isinstance(op, arith.SelectOp)
        for op in mask_select.regions[0].blocks[0].operations
    )
    prefix = linalg_inputs(mask_select)[0].owner.opview
    assert prefix.operation.parent == loop.operation
    assert str(ir.ShapedType(prefix.results[0].type).element_type) == "i1"
    # No full-extent predicate is computed ahead of the loop.
    assert not any(
        str(ir.ShapedType(op.results[0].type).element_type) == "i1"
        and op.operation.block == loop.operation.block
        and op.operation.is_before_in_block(loop.operation)
        for op in operations(module, "linalg.generic")
    )


def test_separate_mask_producer():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (SEPARATE, SPLIT):
            check_separate_mask_producer(source)
    print("Separate mask producer is tiled in the loop ahead of a plain PV contraction")


# CHECK: Separate mask producer is tiled in the loop ahead of a plain PV contraction


# WG-tiled attention with the QK contraction as the score producer. Q, K and V
# are workgroup tiles, i.e. slices of the full operands. The row sum is the last
# reduction and P@V the second contraction, as the attention schedule expects.
SCORE_UNMASKED = """
#rowcol = affine_map<(d0, d1) -> (d0, d1)>
#row    = affine_map<(d0, d1) -> (d0)>
#ik     = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj     = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij     = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @attention(%q_full: tensor<12x4xf32>, %k_full: tensor<4x16xf32>,
                     %v_full: tensor<16x3xf32>) -> tensor<6x3xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %q = tensor.extract_slice %q_full[6, 0] [6, 4] [1, 1]
      : tensor<12x4xf32> to tensor<6x4xf32>
  %k = tensor.extract_slice %k_full[0, 8] [4, 8] [1, 1]
      : tensor<4x16xf32> to tensor<4x8xf32>
  %v = tensor.extract_slice %v_full[8, 0] [8, 3] [1, 1]
      : tensor<16x3xf32> to tensor<8x3xf32>

  // S: x = q @ k
  %qk_empty = tensor.empty() : tensor<6x8xf32>
  %qk_init = linalg.fill ins(%zero : f32)
      outs(%qk_empty : tensor<6x8xf32>) -> tensor<6x8xf32>
  %x = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%q, %k : tensor<6x4xf32>, tensor<4x8xf32>)
      outs(%qk_init : tensor<6x8xf32>) {
  ^bb0(%qv: f32, %kv: f32, %acc: f32):
    %product = arith.mulf %qv, %kv : f32
    %sum = arith.addf %acc, %product : f32
    linalg.yield %sum : f32
  } -> tensor<6x8xf32>

  // R1: m = max_k x
  %row_init = tensor.empty() : tensor<6xf32>
  %m_init = linalg.fill ins(%ninf : f32)
      outs(%row_init : tensor<6xf32>) -> tensor<6xf32>
  %m = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<6x8xf32>) outs(%m_init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mx = arith.maximumf %in, %out : f32
    linalg.yield %mx : f32
  } -> tensor<6xf32>

  // E: p = exp(x - m)
  %p_init = tensor.empty() : tensor<6x8xf32>
  %p = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<6x8xf32>, tensor<6xf32>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<6x8xf32>

  // R2b: o = p @ v
  %o_init = tensor.empty() : tensor<6x3xf32>
  %o_fill = linalg.fill ins(%zero : f32)
      outs(%o_init : tensor<6x3xf32>) -> tensor<6x3xf32>
  %o = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p, %v : tensor<6x8xf32>, tensor<8x3xf32>)
      outs(%o_fill : tensor<6x3xf32>) {
  ^bb0(%pv: f32, %vv: f32, %out: f32):
    %mul = arith.mulf %pv, %vv : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<6x3xf32>

  // R2a: l = sum_k p
  %l_init = linalg.fill ins(%zero : f32)
      outs(%row_init : tensor<6xf32>) -> tensor<6xf32>
  %l = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<6x8xf32>) outs(%l_init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<6xf32>

  // out = o / l
  %out_init = tensor.empty() : tensor<6x3xf32>
  %out = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                         iterator_types = ["parallel", "parallel"]}
      ins(%o, %l : tensor<6x3xf32>, tensor<6xf32>)
      outs(%out_init : tensor<6x3xf32>) {
  ^bb0(%in: f32, %lv: f32, %o2: f32):
    %d = arith.divf %in, %lv : f32
    linalg.yield %d : f32
  } -> tensor<6x3xf32>
  return %out : tensor<6x3xf32>
}
"""

# As above with an all--inf row mask, in the form left without broad elementwise
# fusion: a separate -inf test, an andi row reduction, a separate P select whose
# zero is a splat tensor, and a guarded normalization. PV stays a plain
# multiply-add contraction.
SCORE_MASKED = """
#rowcol = affine_map<(d0, d1) -> (d0, d1)>
#row    = affine_map<(d0, d1) -> (d0)>
#ik     = affine_map<(d0, d1, d2) -> (d0, d2)>
#kj     = affine_map<(d0, d1, d2) -> (d2, d1)>
#ij     = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @attention(%q_full: tensor<12x4xf32>, %k_full: tensor<4x16xf32>,
                     %v_full: tensor<16x3xf32>) -> tensor<6x3xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %true = arith.constant true
  %ninf64 = arith.constant 0xFFF0000000000000 : f64
  %zeros = arith.constant dense<0.0> : tensor<6x8xf32>
  %q = tensor.extract_slice %q_full[6, 0] [6, 4] [1, 1]
      : tensor<12x4xf32> to tensor<6x4xf32>
  %k = tensor.extract_slice %k_full[0, 8] [4, 8] [1, 1]
      : tensor<4x16xf32> to tensor<4x8xf32>
  %v = tensor.extract_slice %v_full[8, 0] [8, 3] [1, 1]
      : tensor<16x3xf32> to tensor<8x3xf32>

  // S: x = q @ k
  %qk_empty = tensor.empty() : tensor<6x8xf32>
  %qk_init = linalg.fill ins(%zero : f32)
      outs(%qk_empty : tensor<6x8xf32>) -> tensor<6x8xf32>
  %x = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%q, %k : tensor<6x4xf32>, tensor<4x8xf32>)
      outs(%qk_init : tensor<6x8xf32>) {
  ^bb0(%qv: f32, %kv: f32, %acc: f32):
    %product = arith.mulf %qv, %kv : f32
    %sum = arith.addf %acc, %product : f32
    linalg.yield %sum : f32
  } -> tensor<6x8xf32>

  // mask = all_k (x == -inf)
  %is_ninf_init = tensor.empty() : tensor<6x8xi1>
  %is_ninf = linalg.generic {indexing_maps = [#rowcol, #rowcol],
                             iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<6x8xf32>) outs(%is_ninf_init : tensor<6x8xi1>) {
  ^bb0(%score: f32, %o: i1):
    %wide = arith.extf %score : f32 to f64
    %empty = arith.cmpf oeq, %wide, %ninf64 : f64
    linalg.yield %empty : i1
  } -> tensor<6x8xi1>
  %mask_empty = tensor.empty() : tensor<6xi1>
  %mask_init = linalg.fill ins(%true : i1)
      outs(%mask_empty : tensor<6xi1>) -> tensor<6xi1>
  %mask = linalg.generic {indexing_maps = [#rowcol, #row],
                          iterator_types = ["parallel", "reduction"]}
      ins(%is_ninf : tensor<6x8xi1>) outs(%mask_init : tensor<6xi1>) {
  ^bb0(%empty: i1, %acc: i1):
    %all = arith.andi %empty, %acc : i1
    linalg.yield %all : i1
  } -> tensor<6xi1>

  // R1: m = max_k x
  %row_init = tensor.empty() : tensor<6xf32>
  %m_init = linalg.fill ins(%ninf : f32)
      outs(%row_init : tensor<6xf32>) -> tensor<6xf32>
  %m = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<6x8xf32>) outs(%m_init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mx = arith.maximumf %in, %out : f32
    linalg.yield %mx : f32
  } -> tensor<6xf32>

  // E: p = exp(x - m)
  %p_init = tensor.empty() : tensor<6x8xf32>
  %p = linalg.generic {indexing_maps = [#rowcol, #row, #rowcol],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<6x8xf32>, tensor<6xf32>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<6x8xf32>

  // pm = mask ? 0 : p
  %pm = linalg.generic {indexing_maps = [#row, #rowcol, #rowcol, #rowcol],
                        iterator_types = ["parallel", "parallel"]}
      ins(%mask, %zeros, %p : tensor<6xi1>, tensor<6x8xf32>, tensor<6x8xf32>)
      outs(%p_init : tensor<6x8xf32>) {
  ^bb0(%masked: i1, %z: f32, %pv: f32, %o: f32):
    %safe = arith.select %masked, %z, %pv : f32
    linalg.yield %safe : f32
  } -> tensor<6x8xf32>

  // R2b: o = pm @ v
  %o_init = tensor.empty() : tensor<6x3xf32>
  %o_fill = linalg.fill ins(%zero : f32)
      outs(%o_init : tensor<6x3xf32>) -> tensor<6x3xf32>
  %o = linalg.generic {indexing_maps = [#ik, #kj, #ij],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%pm, %v : tensor<6x8xf32>, tensor<8x3xf32>)
      outs(%o_fill : tensor<6x3xf32>) {
  ^bb0(%pv: f32, %vv: f32, %out: f32):
    %mul = arith.mulf %pv, %vv : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<6x3xf32>

  // R2a: l = sum_k p
  %l_init = linalg.fill ins(%zero : f32)
      outs(%row_init : tensor<6xf32>) -> tensor<6xf32>
  %l = linalg.generic {indexing_maps = [#rowcol, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<6x8xf32>) outs(%l_init : tensor<6xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<6xf32>

  // out = mask ? o : o / l
  %out_init = tensor.empty() : tensor<6x3xf32>
  %out = linalg.generic {indexing_maps = [#rowcol, #row, #row, #rowcol],
                         iterator_types = ["parallel", "parallel"]}
      ins(%o, %l, %mask : tensor<6x3xf32>, tensor<6xf32>, tensor<6xi1>)
      outs(%out_init : tensor<6x3xf32>) {
  ^bb0(%in: f32, %lv: f32, %masked: i1, %o2: f32):
    %d = arith.divf %in, %lv : f32
    %guarded = arith.select %masked, %in, %d : f32
    linalg.yield %guarded : f32
  } -> tensor<6x3xf32>
  return %out : tensor<6x3xf32>
}
"""


def test_score_producer_fusion():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source in (SCORE_MASKED, SCORE_UNMASKED):
            module = ir.Module.parse(source)
            with schedule_boilerplate() as (schedule, sequence):
                func = lh_transform.match_op(sequence.bodyTarget, "func.func")
                transform_to_flash_attention(
                    transform.AnyOpType.get(), func, 2, n_parallel=1
                )
                transform.yield_([])
            schedule.body.operations[0].apply(module.operation)
            assert module.operation.verify()
            assert not operations(module, "arith.andi")
            (loop,) = operations(module, "scf.for")
            assert isinstance(loop, scf.ForOp)
            qk_ops = [
                op
                for op in operations(module, "linalg.generic")
                if len(linalg_inputs(op)) == 2
                and [str(value.type) for value in linalg_inputs(op)]
                == ["tensor<6x4xf32>", "tensor<4x2xf32>"]
            ]
            assert len(qk_ops) == 1
            assert qk_ops[0].operation.parent == loop.operation
            assert str(qk_ops[0].results[0].type) == "tensor<6x2xf32>"
            assert not any(
                reduction_dims(op) and op.operation.parent != loop.operation
                for op in operations(module, "linalg.generic")
            )
            exps = operations(module, "math.exp")
            assert all(exp.operation.parent.parent == loop.operation for exp in exps)
            factors = [
                exp.operation.parent.opview
                for exp in exps
                if str(exp.operation.parent.results[0].type) == "tensor<6xf32>"
            ]
            assert len(factors) == 1, "expected one shared correction exp"
            assert len(list(factors[0].results[0].uses)) == 2
            # The row sum and PV share one probability tile.
            probabilities = [
                exp.operation.parent.opview
                for exp in exps
                if str(exp.operation.parent.results[0].type) == "tensor<6x2xf32>"
            ]
            assert len(probabilities) == 1, "expected one shared probability exp"
            assert len(list(probabilities[0].results[0].uses)) == 2
            assert len(exps) == 2
    print("The same attention schedule sinks QK tiles with and without the mask")


# CHECK: The same attention schedule sinks QK tiles with and without the mask


def test_compact_correction_layouts():
    """Project non-leading factor dimensions without changing correction precision."""
    transposed = UNMASKED.split("  // The deferred normalization")[0]
    transposed += "  return %o : tensor<6x3xf32>\n}\n"
    transposed = (
        transposed.replace("(d0, d2)>", "(d1, d2)>")
        .replace("(d2, d1)>", "(d2, d0)>")
        .replace("tensor<6x3xf32>", "tensor<3x6xf32>")
    )
    x = np.arange(48, dtype=np.float32).reshape(6, 8) / 16 - 1
    x[:, 1::2] *= -1
    v = np.arange(24, dtype=np.float32).reshape(8, 3) / 12 - 1
    p = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = (p @ v).T
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for generic in (False, True):
            for wide_accumulator in (False, True):
                source = transposed
                if generic:
                    source = source.replace(
                        "    linalg.yield %e : f32",
                        "    %absolute = math.absf %e : f32\n"
                        "    linalg.yield %absolute : f32",
                    )
                dtype = np.float64 if wide_accumulator else np.float32
                element = "f64" if wide_accumulator else "f32"
                if wide_accumulator:
                    source = (
                        source.replace("tensor<3x6xf32>", "tensor<3x6xf64>")
                        .replace(
                            "  %o_init =",
                            "  %zero64 = arith.constant 0.0 : f64\n  %o_init =",
                        )
                        .replace(
                            "linalg.fill ins(%zero : f32) outs(%o_init",
                            "linalg.fill ins(%zero64 : f64) outs(%o_init",
                        )
                        .replace("%vv: f32, %out: f32", "%vv: f32, %out: f64")
                        .replace(
                            "    %add = arith.addf %out, %mul : f32\n"
                            "    linalg.yield %add : f32",
                            "    %wide = arith.extf %mul : f32 to f64\n"
                            "    %add = arith.addf %out, %wide : f64\n"
                            "    linalg.yield %add : f64",
                        )
                    )
                module = apply(source)
                scalers = [
                    op
                    for op in operations(module, "linalg.generic")
                    if [str(value.type) for value in linalg_inputs(op)]
                    == [f"tensor<3x6x{element}>", f"tensor<6x{element}>"]
                ]
                assert len(scalers) == 1
                assert indexing_maps(scalers[0])[1] == ir.AffineMap.get(
                    2, 0, [ir.AffineDimExpr.get(1)]
                )
                for exp in operations(module, "math.exp"):
                    assert str(exp.operation.parent.results[0].type) in (
                        "tensor<6x2xf32>",
                        "tensor<6xf32>",
                        "tensor<6xf64>",
                    )
                for tile in (None, 1, 2, 4):
                    schedule = online_schedule(tile) if tile is not None else None
                    runner, name = compile_payload(source, schedule)
                    output = np.empty((3, 6), dtype=dtype)
                    runner.execute(name, [x, v, output])
                    np.testing.assert_allclose(output, expected, rtol=3e-6, atol=1e-7)
    print(
        "Compact corrections preserve transposed, generic and mixed-precision results"
    )


# CHECK: Compact corrections preserve transposed, generic and mixed-precision results


if __name__ == "__main__":
    test_structure()
    test_preservation()
    test_numerics()
    test_separate_mask_producer()
    test_score_producer_fusion()
    test_compact_correction_layouts()
