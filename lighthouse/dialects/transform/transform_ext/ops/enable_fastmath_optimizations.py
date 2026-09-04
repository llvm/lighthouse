from mlir import ir
from mlir.dialects import arith, ext, math, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect

#: The `fastmath` attribute's default. It is present in the attribute dictionary
#: but elided when printed, and carries no intent, so it counts as absent.
NO_FASTMATH = "#arith.fastmath<none>"

#: Ops that get the `fastmath` attribute. Deliberately narrow -- only what the
#: current usecases need.
#:
#:   - `math.exp`, for the `exp(a)/exp(b)` fold below.
#:   - `arith.subf`, because the correction's terms come out as `0.0 - m` (the
#:     fusion substitutes the additive neutral for `E`'s data operand). LLVM only
#:     narrows that to a negation under `nsz` -- `0.0 - x` and `-x` disagree at
#:     `x = 0.0` -- so without the flags the subtraction survives to the ISA.
FASTMATH_OP_NAMES = frozenset({"math.exp", "arith.subf"})


def _collect(root: ir.Operation, names) -> list[ir.Operation]:
    """Every op under `root` whose name is in `names`, in pre-order."""
    ops: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in names:
            ops.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit, ir.WalkOrder.PRE_ORDER)
    return ops


def _annotate_fastmath(root: ir.Operation) -> None:
    """Set `fastmath<fast>` on every `FASTMATH_OP_NAMES` op under `root`.

    Ops that already carry a non-default `fastmath` attribute are left alone, so an
    explicitly-chosen weaker flag set is never widened.
    """
    with root.context:
        fast = ir.Attribute.parse("#arith.fastmath<fast>")
    for op in _collect(root, FASTMATH_OP_NAMES):
        existing = op.attributes.get("fastmath")
        if existing is not None and str(existing) != NO_FASTMATH:
            continue
        op.attributes["fastmath"] = fast


def _fastmath_allows_transform(op: ir.Operation) -> bool:
    """True if `op`'s `fastmath` flags permit `exp(a)/exp(b)` -> `exp(a-b)`.

    The rewrite is not IEEE-preserving and relies on three flags:

      - `reassoc`, which licenses regrouping the expression at all.
      - `nnan` and `ninf`, because the two forms disagree once an exponential
        overflows or underflows. For `a = b = 1000`, `exp(a)/exp(b)` is `inf/inf`,
        i.e. NaN, while `exp(a-b)` is 1.

    Only the full `fastmath<fast>` set is accepted. That is stricter than the
    three flags above require -- `<reassoc,nnan,ninf>` would also be sound -- but
    it is the only set the payloads here produce, so the check stays a plain
    comparison.
    """
    if "fastmath" not in op.attributes:
        return False
    return str(op.attributes["fastmath"]) == "#arith.fastmath<fast>"


def _exp_operand(value: ir.Value) -> ir.Value | None:
    """The argument of `value`'s defining `math.exp`, if it is a fast-math one."""
    producer = value.owner
    if not isinstance(producer, ir.Operation):
        producer = getattr(producer, "operation", None)
    if producer is None or producer.name != "math.exp":
        return None
    if not _fastmath_allows_transform(producer):
        return None
    return producer.operands[0]


def _fold_exp_div(div_op: ir.Operation, rewriter: transform.TransformRewriter) -> bool:
    """Rewrite one `exp(a) / exp(b)` into `exp(a - b)`. True if it applied.

    Gated on the two `math.exp` producers alone, not on the `arith.divf`: the flags
    the rewrite needs describe the exponentials overflowing, and the divide is not in
    `FASTMATH_OP_NAMES`.
    """
    numerator = _exp_operand(div_op.operands[0])
    denominator = _exp_operand(div_op.operands[1])
    if numerator is None or denominator is None:
        return False

    # exp(a)/exp(b) and exp(a-b) only agree when a and b -- hence a-b -- share the
    # result type; a mixed-precision chain is left alone.
    if numerator.type != denominator.type:
        return False
    if numerator.type != div_op.results[0].type:
        return False

    # Both replacements are flagged directly: step 1 has already walked the payload,
    # so ops created here never pass through `_annotate_fastmath`. The subtract wants
    # the flags for the same reason every other one does -- `a - b` is
    # `(0.0 - m_new) - (0.0 - m_old)`, which only collapses to `m_old - m_new` once
    # reassociation is licensed.
    fast = arith.FastMathFlags.fast
    with ir.InsertionPoint(div_op), div_op.location:
        difference = arith.SubFOp(numerator, denominator, fastmath=fast)
        replacement = math.ExpOp(difference.result, fastmath=fast)
    # The two original exps are left in place; they are dead only if nothing else
    # reads them, which DCE/CSE afterwards decides.
    rewriter.replace_op(div_op, [replacement.result])
    return True


class EnableFastmathOptimizationsOp(
    TransformExtensionDialect.Operation, name="enable_fastmath_optimizations"
):
    """
    Enable fast-math and apply the rewrites it licenses, under `target`.

    Two steps, in this order:

      1. Set `fastmath<fast>` on every op in `FASTMATH_OP_NAMES` -- a deliberately
         narrow set, see there for what is in it and why. Ops that already carry a
         non-default flag set keep it, so an explicitly-chosen weaker one is never
         widened.
      2. Rewrite `exp(a) / exp(b)` into `exp(a - b)`, trading a transcendental and
         a divide for a subtract. Only applied where both `math.exp` producers carry
         `fastmath<fast>`; see `_fastmath_allows_transform` for which flags the
         rewrite relies on.

    The order matters and is why the two are one op: the fold keys off the
    annotation. Ops without the flag are skipped rather than rejected, so this is
    safe to run over a whole function.

    Ideally must be run after vectorization.

    Args:
        target: Handle to root ops to work within (e.g. func.func).
    Returns:
        Handle to the same target roots.
    """

    target: ext.Operand[transform.AnyOpType]
    optimized_ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "EnableFastmathOptimizationsOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)

            for target in targets:
                _annotate_fastmath(target)
                for div_op in _collect(target, {"arith.divf"}):
                    _fold_exp_div(div_op, rewriter)

            results.set_ops(op.optimized_ops, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "EnableFastmathOptimizationsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def enable_fastmath_optimizations(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create EnableFastmathOptimizationsOp."""
    op = EnableFastmathOptimizationsOp(target=target)
    return op.optimized_ops
