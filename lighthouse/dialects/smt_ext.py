from typing import Callable

from mlir import ir
from mlir.dialects import smt

__all__ = ["SMTIntValue", "assert_", "register_and_load"]


def register_and_load(**kwargs):
    """Register and load the SMTIntValue caster."""

    SMTIntValue.register_value_caster()


def assert_(predicate: ir.Value[smt.BoolType] | bool, error_message: str = ""):
    """Assert normally if a bool else produce an SMT assertion op."""

    if isinstance(predicate, bool):
        assert predicate, error_message
    else:
        assert_ = smt.assert_(predicate)
        if error_message:
            assert_.attributes["error"] = ir.StringAttr.get(error_message)


def int_to_smt(operand: "int | SMTIntValue") -> "SMTIntValue":
    if isinstance(operand, int):
        int_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), operand)
        return SMTIntValue(smt.int_constant(int_attr))
    return operand


def swapped(
    f: Callable[["int | SMTIntValue", "int | SMTIntValue"], "int | SMTIntValue"],
) -> Callable[["int | SMTIntValue", "int | SMTIntValue"], "int | SMTIntValue"]:
    return lambda a, b: f(b, a)


class SMTIntValue(ir.Value[smt.IntType]):
    """A Value caster for `!smt.int` that supports Pythonic arithmetic and comparison operations."""

    def __init__(self, v):
        super().__init__(v)

    def __hash__(self):
        return super().__hash__()

    @staticmethod
    def register_value_caster():
        if not hasattr(SMTIntValue, "_is_registered"):
            ir.register_value_caster(smt.IntType.static_typeid)(SMTIntValue)
            setattr(SMTIntValue, "_is_registered", True)

    def __add__(self, rhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_add([self, int_to_smt(rhs)]))

    def __radd__(self, lhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_add([int_to_smt(lhs), self]))

    def __sub__(self, rhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_sub(self, int_to_smt(rhs)))

    def __rsub__(self, lhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_sub(int_to_smt(lhs), self))

    def __mul__(self, rhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_mul([self, int_to_smt(rhs)]))

    def __rmul__(self, lhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_mul([int_to_smt(lhs), self]))

    def __floordiv__(self, rhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_div(self, int_to_smt(rhs)))

    def __rfloordiv__(self, lhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_div(int_to_smt(lhs), self))

    def __mod__(self, rhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_mod(self, int_to_smt(rhs)))

    def __rmod__(self, lhs: "int | SMTIntValue") -> "SMTIntValue":
        return SMTIntValue(smt.int_mod(int_to_smt(lhs), self))

    def __eq__(self, rhs: "int | SMTIntValue") -> ir.Value[smt.BoolType]:
        return smt.eq([self, int_to_smt(rhs)])

    def __le__(self, rhs: "int | SMTIntValue") -> ir.Value[smt.BoolType]:
        return smt.int_cmp(smt.IntPredicate.le, self, int_to_smt(rhs))

    def __lt__(self, rhs: "int | SMTIntValue") -> ir.Value[smt.BoolType]:
        return smt.int_cmp(smt.IntPredicate.lt, self, int_to_smt(rhs))

    def __ge__(self, rhs: "int | SMTIntValue") -> ir.Value[smt.BoolType]:
        return smt.int_cmp(smt.IntPredicate.ge, self, int_to_smt(rhs))

    def __gt__(self, rhs: "int | SMTIntValue") -> ir.Value[smt.BoolType]:
        return smt.int_cmp(smt.IntPredicate.gt, self, int_to_smt(rhs))

    def __str__(self):
        return super().__str__().replace(ir.Value.__name__, SMTIntValue.__name__)
