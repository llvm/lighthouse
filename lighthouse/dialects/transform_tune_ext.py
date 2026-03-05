import inspect
import re
import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional
from functools import wraps
from operator import mod

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tune as transform_tune

__all__ = ["KnobValue", "knob"]


def register_and_load(context=None):
    pass  # NB: currently nothing to register or load.


def knob(
    *args,
    result: Optional[ir.Type] = None,
    **kwargs,
) -> "KnobValue":
    options = ir.DictAttr.get()
    result = result or transform.AnyParamType.get()
    return KnobValue(
        transform_tune.KnobOp(result, *args, options=options, **kwargs).result
    )


def update_knob_options(knob: transform_tune.KnobOp, key, value):
    items = list((namedattr.name, namedattr.attr) for namedattr in knob.options)
    if isinstance(value, int):
        value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
    items.append((key, value))
    knob.options = ir.DictAttr.get(dict(items))


class KnobValue(ir.Value):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def in_(self, options):
        i64 = ir.IntegerType.get_signless(64)
        options_attr = ir.ArrayAttr.get([ir.IntegerAttr.get(i64, v) for v in options])

        assert (
            isinstance(self.owner.options, ir.DictAttr) and len(self.owner.options) == 0
        )  # Only one constraint supported for now.
        self.owner.options = ir.DictAttr.get({"options": options_attr})
        return True

    @staticmethod
    def ast_rewrite(in_exprs: bool = False):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_source = inspect.getsource(func)
                indent = math.inf
                for line in func_source.splitlines():
                    indent = min(indent, len(re.match(" *", line).group(0)))
                func_source = "\n".join(line[indent:] for line in func_source.splitlines())
                func_ast = ast.parse(func_source)
                func_def_ast = func_ast.body[0]

                # TODO: carefully remove just the @KnobValue.ast_rewrite decorator in case of multiple decorators.
                func_def_ast.decorator_list.clear()  # Remove the decorator to avoid infinite recursion.
                if in_exprs:
                    func_def_ast.body = [
                        InTransformer().visit(stmt) for stmt in func_def_ast.body
                    ]
                    ast.fix_missing_locations(func_def_ast)
                mod = compile(ast.unparse(func_ast), filename="<ast>", mode="exec")
                frame = inspect.currentframe()
                assert frame and frame.f_back
                temp_globals = frame.f_back.f_globals.copy()
                temp_globals |= frame.f_back.f_locals.copy()
                temp_locals = frame.f_back.f_locals.copy()
                temp_globals["In"] = In
                exec(mod, temp_globals, temp_locals)
                return temp_locals[func.__name__](*args, **kwargs)

            return wrapper

        return decorator

    def _set_bound(self, key, combine, value):
        assert isinstance(self.owner.options, ir.DictAttr)
        existing = self.owner.options[key] if key in self.owner.options else None
        update_knob_options(
            self.owner.opview,
            key,
            value if existing is None else combine(existing.value, value),
        )
        return True

    def __mod__(self, other):
        assert isinstance(other, int)
        return KnobExpression(lhs=self, rhs=other, operator=mod)

    def __rmod__(self, other):
        assert isinstance(other, int)
        return KnobExpression(lhs=other, rhs=self, operator=mod)

    def __lt__(self, other):
        assert isinstance(other, int)
        return self._set_bound("upper_bound", min, other + 1)

    def __le__(self, other):
        assert isinstance(other, int)
        return self._set_bound("upper_bound", min, other)

    def __ge__(self, other):
        assert isinstance(other, int)
        return self._set_bound("lower_bound", max, other)

    def __gt__(self, other):
        assert isinstance(other, int)
        return self._set_bound("lower_bound", max, other + 1)

    def __eq__(self, other):
        assert isinstance(other, int)
        assert isinstance(self.owner.options, ir.DictAttr)
        assert len(self.owner.options) == 0, "Only one constraint supported for now."
        i64 = ir.IntegerType.get_signless(64)
        update_knob_options(
            self.owner.opview,
            "options",
            ir.ArrayAttr.get([ir.IntegerAttr.get(i64, other)]),
        )
        return True


@dataclass
class KnobExpression:
    lhs: KnobValue | int
    rhs: KnobValue | int
    operator: Literal[mod]

    def __eq__(self, other):
        assert other == 0, "Only equality to zero supported for now."
        assert self.operator is mod
        i64 = ir.IntegerType.get_signless(64)
        if isinstance(self.lhs, KnobValue):
            assert isinstance(self.lhs.owner.options, ir.DictAttr)
            assert isinstance(self.rhs, int)
            assert "divisible_by" not in self.lhs.owner.options
            update_knob_options(self.lhs.owner.opview, "divisible_by", self.rhs)
        elif isinstance(self.rhs, KnobValue):
            assert isinstance(self.lhs, int)
            assert isinstance(self.rhs.owner.options, ir.DictAttr)
            assert "divides" not in self.rhs.owner.options
            update_knob_options(self.rhs.owner.opview, "divides", self.lhs)
        else:
            assert False, "At least one operand must be a KnobValue."

        return True


@dataclass
class In:
    lhs: Any
    rhs: Any

    def __bool__(self):
        if isinstance(self.lhs, KnobValue):
            return self.lhs.in_(self.rhs)
        return self.lhs in self.rhs


class InTransformer(ast.NodeTransformer):
    def visit_Compare(self, node: ast.Compare) -> Any:
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            return ast.Call(
                func=ast.Name(id="In", ctx=ast.Load()),
                args=[node.left, node.comparators[0]],
                keywords=[],
            )
        return node
