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

from dependent_reduction_payloads import ATTENTION, MIXED_SOFTMAX, SOFTMAX


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


# CHECK-LABEL: Case: softmax
# CHECK-NEXT: Legal to fuse: E dim 1, tile 32
check_case("softmax", SOFTMAX)

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
