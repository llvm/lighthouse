import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_args
from functools import wraps

from mlir import ir

NonDet = object()  # Sentinel value for non-determinized parameters.


def check_annotated_constraints(f: Callable):
    """Wrapper that runs __metadata__ constraints from Annotated types on function arguments."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for name, value in bound_args.arguments.items():
            if value is NonDet:
                continue

            param = sig.parameters[name]

            if param.kind != param.KEYWORD_ONLY or not hasattr(
                param.annotation, "__metadata__"
            ):
                continue

            wrapped_type, constraint = get_args(param.annotation)

            if not isinstance(value, wrapped_type):
                raise TypeError(
                    f"Argument {name} must be of type {param.annotation}, got {type(value)}"
                )

            if not constraint(value):
                raise ValueError(
                    f"Constraint {constraint} not satisfied for argument {name} with value {value}"
                )

        return f(*args, **kwargs)

    return wrapper


@dataclass
class ConstraintCollector:
    """Used on annotated constraint functions to reify the constraints as data."""

    children: list[Any] = field(default_factory=list)
    lb: int | None = None
    ub: int | None = None

    def __mod__(self, other):
        self.children.append(mod := ModConstraintCollector(modulus=other))
        return mod

    def __le__(self, other):
        self.ub = other
        return self

    def __ge__(self, other):
        self.lb = other
        return self

    def to_mlir(self) -> ir.Attribute:
        dict_attrs = {}
        i64 = ir.IntegerType.get_signless(64)
        if self.lb is not None:
            dict_attrs["lb"] = ir.IntegerAttr.get(i64, self.lb)
        if self.ub is not None:
            dict_attrs["ub"] = ir.IntegerAttr.get(i64, self.ub)
        if self.children:
            assert len(self.children) == 1 and isinstance(
                self.children[0], ModConstraintCollector
            )
            dict_attrs["step"] = ir.IntegerAttr.get(i64, self.children[0].modulus)
        return ir.DictAttr.get(dict_attrs)


@dataclass
class ModConstraintCollector:
    modulus: int
    remainder: int | None = None

    def __eq__(self, other):
        assert other == 0, "Only equality with zero is currently supported"
        return self
