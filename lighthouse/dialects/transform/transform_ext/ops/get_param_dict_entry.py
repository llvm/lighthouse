from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GetParamDictEntryOp(
    TransformExtensionDialect.Operation, name="get_param_dict_entry"
):
    """
    Extract one named entry from a dictionary-valued param as its own param.

    Args:
        dict_param: Param holding a single DictionaryAttr.
        key: Name of the entry to extract.
    Return:
        Param holding the value associated with `key`.
    """

    dict_param: ext.Operand[transform.AnyParamType]
    key: ir.StringAttr
    value: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetParamDictEntryOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            dict_params = state.get_params(op.dict_param)
            if len(dict_params) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            dict_attr = dict_params[0]
            if not isinstance(dict_attr, ir.DictAttr):
                return DiagnosedSilenceableFailure.SilenceableFailure

            key = op.key.value
            if key not in dict_attr:
                return DiagnosedSilenceableFailure.SilenceableFailure

            results.set_params(op.value, [dict_attr[key]])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetParamDictEntryOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def get_param_dict_entry(
    dict_param: ir.Value[transform.AnyParamType],
    key: str | ir.StringAttr,
) -> ir.Value[transform.AnyParamType]:
    """
    snake_case wrapper to create a GetParamDictEntryOp.

    Args:
        dict_param: Param holding a single DictionaryAttr.
        key: Name of the entry to extract.
    Return:
        Param holding the value associated with `key`.
    """
    if not isinstance(key, ir.StringAttr):
        key = ir.StringAttr.get(key)
    return GetParamDictEntryOp(dict_param=dict_param, key=key).value
