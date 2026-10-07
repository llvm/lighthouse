"""Generate MLIR transform schedule that normalizes input IR to a canonical form."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import apply_registered_pass
from lighthouse.schedule import schedule_boilerplate
from .lowering_common import get_payload_func


def cleanup_schedule(
    payload_func_name: str | None = None,
) -> ir.Module:
    """Normalize singleton dimensions and fuse elementwise ops in the payload."""

    with schedule_boilerplate() as (schedule, named_seq):
        op_names = [
            "linalg.generic",
            "linalg.matmul",
            "linalg.batch_matmul",
            "linalg.elementwise",
            "linalg.softmax",
        ]
        func = get_payload_func(
            named_seq.bodyTarget,
            op_name=op_names,
            func_name=payload_func_name,
        )

        # Match linalg.softmax operation if any and decompose it into generic ops
        anytype = transform.AnyOpType.get()
        softmax_ops = structured.structured_match(anytype, func, ops=["linalg.softmax"])
        structured.structured_decompose_interface(anytype, softmax_ops)

        # Convert linalg.elementwise and linalg.batch_matmul to linalg.generic
        structured.structured_generalize(
            anytype,
            structured.structured_match(
                anytype,
                func,
                ops=["linalg.elementwise", "linalg.batch_matmul"],
            ),
        )

        # Normalize possible singleton dimensions so tile+fuse logic works.
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            # fold unit dims in linalg.generic op inputs
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            # fold tensor.extract_slice(tensor.expand_shape(x)) into x
            tensor.apply_patterns_tensor_reassociative_reshape_folding()
            # swap tensor.extract_slice(linalg.fill(...)) ops
            structured.apply_patterns_linalg_swap_extract_slice_with_fill()
            # fold tensor.extract_slice(tensor.empty(...)) into tensor.tensor_empty(...)
            tensor.apply_patterns_tensor_fold_tensor_empty(fold_single_use_only=True)
        lh_transform.cleanup(func)

        # Fuse elementwise ops, also removes unused linalg op results (if any).
        func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")
        lh_transform.cleanup(func)

        transform.yield_()

    return schedule
