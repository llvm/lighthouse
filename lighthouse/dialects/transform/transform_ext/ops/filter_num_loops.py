from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class FilterNumLoopsOp(TransformExtensionDialect.Operation, name="filter_num_loops"):
    """
    Returns ops that have at least `num_loops` loops.
    Supports Linalg operations only.

    Args:
        target: Handle to target op
        num_loops: Number of loops to filter by
    Returns:
        Matching ops
    """

    target: ext.Operand[transform.AnyOpType]
    num_loops: ext.Operand[transform.AnyParamType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FilterNumLoopsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            num_loops_attr = state.get_params(op.num_loops)
            if len(num_loops_attr) == 1 and isinstance(
                num_loops_attr[0], ir.IntegerAttr
            ):
                num_loops = num_loops_attr[0].value
            else:
                return DiagnosedSilenceableFailure.SilenceableFailure

            matching_ops = []
            for target in targets:
                if "linalg" not in target.name:
                    continue
                if hasattr(target, "indexing_maps"):
                    map: ir.AffineMap = target.indexing_maps[0].value
                    if map.n_dims >= num_loops:
                        matching_ops.append(target)
                elif hasattr(target, "iterator_types"):
                    if len(target.iterator_types) >= num_loops:
                        matching_ops.append(target)
                else:
                    continue

            results.set_ops(op.ops, matching_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FilterNumLoopsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def filter_num_loops(
    target: ir.Value[transform.AnyOpType],
    num_loops: int | ir.Value[transform.AnyParamType],
) -> ir.Value:
    """
    snake_case wrapper to create a FilterNumLoopsOp.

    Args:
        target: Handle to target op
        num_loops: Number of loops to filter by
    Returns:
        List of matching ops that have at least `num_loops` loops.
    """
    if isinstance(num_loops, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), num_loops)
        num_loops = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return FilterNumLoopsOp(target=target, num_loops=num_loops).ops
