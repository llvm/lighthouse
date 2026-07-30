from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _normalize_op_names(op_names: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a list of operation names into a canonical tuple."""
    if isinstance(op_names, str):
        op_names = [op_names]
    names = tuple(dict.fromkeys(op_names))
    if not names:
        raise ValueError("op_names must contain at least one operation name")
    return names


class FilterByNameOp(TransformExtensionDialect.Operation, name="filter_by_name"):
    """Return only payload ops whose operation name is in `op_names`."""

    target: ext.Operand[transform.AnyOpType]
    op_names: ext.Operand[transform.AnyParamType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FilterByNameOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            params = state.get_params(op.op_names)

            allowed_names = set()
            if len(params) == 1 and isinstance(params[0], ir.ArrayAttr):
                array_attr = params[0]
                for name_attr in array_attr:
                    if not isinstance(name_attr, ir.StringAttr):
                        return DiagnosedSilenceableFailure.SilenceableFailure
                    allowed_names.add(name_attr.value)
            else:
                for name_attr in params:
                    if not isinstance(name_attr, ir.StringAttr):
                        return DiagnosedSilenceableFailure.SilenceableFailure
                    allowed_names.add(name_attr.value)

            filtered = [target for target in target_ops if target.name in allowed_names]
            results.set_ops(op.ops, filtered)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FilterByNameOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def filter_by_name(
    target: ir.Value[transform.AnyOpType],
    op_names: str | list[str] | tuple[str, ...],
) -> ir.Value:
    """Return the subset of `target` whose payload operation names match `op_names`."""
    names = _normalize_op_names(op_names)
    names_attr = ir.ArrayAttr.get([ir.StringAttr.get(name) for name in names])
    op_names_param = transform.ParamConstantOp(transform.AnyParamType.get(), names_attr)
    return FilterByNameOp(target=target, op_names=op_names_param).ops
