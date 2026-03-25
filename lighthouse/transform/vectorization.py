from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import vector
from mlir.dialects.transform import x86


def vector_contract_to_fma(target):
    """
    Apply vector contract to FMA lowering patterns.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        vector.apply_patterns_vector_lower_contraction(
            lowering_strategy=vector.VectorContractLowering.OuterProduct
        )
        vector.apply_patterns_vector_lower_outerproduct()


def x86_vector_patterns(target):
    """
    Apply x86-specific vector patterns.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        x86.apply_patterns_x86_vector_contract_to_packed_type_dot_product()
        x86.apply_patterns_x86_vector_contract_to_fma()
        x86.apply_patterns_x86_sink_vector_producer_ops()
