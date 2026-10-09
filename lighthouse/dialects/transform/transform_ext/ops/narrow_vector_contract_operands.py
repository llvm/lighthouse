from mlir import ir
from mlir.dialects import ext, transform, arith
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class NarrowVectorContractOperandsOp(
    TransformExtensionDialect.Operation, name="narrow_vector_contract_operands"
):
    """Make a ``vector.contract``'s multiplicands the narrow (DPAS) operand type.

    For each targeted ``vector.contract``, both multiplicand operands (0 and 1)
    are brought to the narrow operand type while the accumulator (operand 2) stays
    wide, turning the op into a mixed-precision contract that lowers to a DPAS:

      * an operand produced by ``arith.extf`` is replaced by the extf's narrow
        source (the widen is dropped);
      * a wider operand gets an ``arith.truncf`` to the narrow type, rewired
        only at the contract so its other consumers keep the wide value.

    The narrow type is taken from an ``arith.extf``-produced operand of the same
    contract; contracts without one are left unchanged.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, results, state) -> DiagnosedSilenceableFailure:
            contracts = list(state.get_payload_ops(op.target))
            for contract in contracts:
                _narrow_contract(contract)
            results.set_ops(op.result, contracts)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def _extf_source(value: ir.Value) -> ir.Value | None:
    """The source of an ``arith.extf`` producing ``value``, else None."""
    producer = value.owner
    if isinstance(producer, ir.Block):
        return None
    if producer.operation.name != "arith.extf":
        return None
    return producer.operands[0]


def _narrow_contract(contract: ir.OpView) -> None:
    # The narrow operand type is whatever an extf multiplicand widens from.
    narrow_type = None
    for i in (0, 1):
        source = _extf_source(contract.operands[i])
        if source is not None:
            narrow_type = ir.ShapedType(source.type).element_type
            break
    if narrow_type is None:
        return

    for i in (0, 1):
        operand = contract.operands[i]
        source = _extf_source(operand)
        if source is not None:
            # Drop the widen: feed the narrow source straight to the contract.
            extf = operand.owner
            extf.results[0].replace_all_uses_with(source)
            extf.operation.erase()
            continue
        shaped = ir.ShapedType(operand.type)
        if shaped.element_type == narrow_type:
            continue
        # Operand is wider than the DPAS type: narrow it only at this contract.
        with ir.InsertionPoint(contract):
            narrowed = arith.truncf(
                ir.VectorType.get(shaped.shape, narrow_type), operand
            )
        contract.operands[i] = narrowed


def narrow_vector_contract_operands(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Narrow the targeted contracts' multiplicands to the DPAS operand type."""
    return NarrowVectorContractOperandsOp(target=target).result
