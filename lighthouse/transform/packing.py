from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def pack_propagation(target):
    """
    Apply pack propagation patterns to the target.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        structured.apply_patterns_linalg_data_layout_propagation(poison_padding=True)
        structured.apply_patterns_linalg_extract_slice_sinking()
        structured.apply_patterns_linalg_fold_pack_unpack_into_empty()
        structured.apply_patterns_tensor_fold_into_pack_and_unpack()
        transform.apply_patterns_canonicalization()
