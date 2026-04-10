from mlir import ir
from mlir.dialects import ext, transform

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ParamCmpEqOp(TransformExtensionDialect.Operation, name="param_cmp_eq"):
    """
    Compare the values of the `lhs` and `rhs` parameters for equality.

    The operation succeeds if the values are equal, and fails otherwise. If the
    parameters resolve to multiple values, the operation succeeds if all values
    are pairwise equal, and fails otherwise.
    """

    lhs: ext.Operand[transform.AnyParamType]
    rhs: ext.Operand[transform.AnyParamType]

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ParamCmpEqOp",
            _rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
            state: transform.TransformState,
        ) -> transform.DiagnosedSilenceableFailure:
            lhs_params = state.get_params(op.lhs)
            rhs_params = state.get_params(op.rhs)
            if len(lhs_params) != len(rhs_params):
                return transform.DiagnosedSilenceableFailure.SilenceableFailure
            for lhs_param, rhs_param in zip(lhs_params, rhs_params):
                if lhs_param != rhs_param:
                    return transform.DiagnosedSilenceableFailure.SilenceableFailure
            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ParamCmpEqOp") -> bool:
            return True

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ParamCmpEqOp", effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.only_reads_payload(effects)


def param_cmp_eq(lhs: ir.Value, rhs: ir.Value):
    return ParamCmpEqOp(lhs=lhs, rhs=rhs)
