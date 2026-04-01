from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

from operator import eq, ge, gt, le, lt, ne, mul, mod, floordiv, add

from typing import Callable, Generator, Sequence, Optional

from mlir import ir
from mlir.dialects import transform, smt
from mlir.dialects.transform import tune as transform_tune

from lighthouse.dialects import transform_smt_ext


class Node(ABC):
    """Base class for `Node`s which can be evaluated w.r.t. an `environment`.

    When tracing the IR to construct a DAG of values dependent on tuneables,
    the DAG's nodes are all of (sub-)type (of) `Node`. In the context of
    evaluating nodes of the DAG, such as the roots, `environment` means the
    tuneable |-> value map.
    """

    @abstractmethod
    def evaluate(self, environment: dict["Node", int]) -> int | bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Node):
    """Leaf `Node` which evaluates to a constant irrespective of the environment.

    Intended to represent `Value`s which are constants in IR."""

    value: int

    def evaluate(self, environment: dict[Node, int]) -> int:
        return self.value


@dataclass(frozen=True)
class Tuneable(Node, ABC):
    """Abstract `Node` which evaluates via looking up its name in the environment."""

    name: str

    @abstractmethod
    def possibilities(self) -> Generator[int, None, None]:
        raise NotImplementedError

    def evaluate(self, environment: dict[Node, int]) -> int | bool:
        return environment[self]


@dataclass(frozen=True)
class Knob(Tuneable):
    """Leaf `Node` which evals per its name in the environment and knows its possible values.

    Intended to represent the `Value` associated with a tuneable knob in IR."""

    options: Optional[Sequence[int]] = None
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    divisible_by: Optional[int] = None
    divides: Optional[int] = None

    def __post_init__(self):
        assert self.options or (None not in (self.lower_bound, self.upper_bound)), (
            "Options attribute not finitely specified"
        )
        assert self.divisible_by is None or self.divisible_by > 0, (
            "divisible_by must be positive"
        )
        assert self.divides is None or self.divides > 0, "divides must be positive"

    def __repr__(self):
        return (
            f"{self.name}<".upper()
            + f"{list(self.possibilities())}>".replace(", ", "|")[1:-2]
            + ">"
        )

    def possibilities(self) -> Generator[int, None, None]:
        if self.options is not None:
            if self.divides is not None or self.divisible_by is not None:
                yield from filter(
                    lambda val: (self.divides is None or (self.divides % val == 0))
                    and (self.divisible_by is None or (val % self.divisible_by == 0)),
                    self.options,
                )
            else:
                yield from self.options
        else:
            low = self.lower_bound
            step = 1
            if self.divisible_by is not None:
                low = self.lower_bound + (-self.lower_bound % self.divisible_by)
                step = self.divisible_by
            for val in range(low, self.upper_bound + 1, step):
                if self.divides is None or self.divides % val == 0:
                    yield val


@dataclass(frozen=True)
class Apply(Node):
    """Recursive case `Node` which calculates a function from the evaluation of other nodes.

    Intended to represent `Value`s in IR dependent on other `Value`s."""

    operator: Callable[..., int]
    args: Sequence[Node]

    def evaluate(self, environment: dict[Node, int]) -> int:
        return self.operator(*[arg.evaluate(environment) for arg in self.args])


@dataclass(frozen=True)
class Predicate(Node):
    """Recursive case `Node` which applies a predicate to the evaluation of other nodes.

    Intended to represent the condition that must hold for execution to be able
    to succesfully proceed beyond a specified op."""

    operator: Callable[..., bool]
    args: Sequence[Node]

    def evaluate(self, environment: dict[Node, int]) -> bool:
        return self.operator(*[arg.evaluate(environment) for arg in self.args])


@dataclass(frozen=True)
class Alternatives(Tuneable):
    """Recursive case `Node` which acts as its selected child predicate, per its env.

    Intended to represent selection among a fixed number of regions in IR."""

    alt_idx_to_pred: Sequence[Predicate]

    def evaluate(self, environment: dict[Node, int]) -> bool:
        selected_region_idx = super().evaluate(environment)  # evals name to int
        return self.alt_idx_to_pred[selected_region_idx].evaluate(environment)

    def possibilities(self) -> Generator[int, None, None]:
        yield from range(len(self.alt_idx_to_pred))


@dataclass
class AlternativesResult(Node):
    """Recursive case `Node` which selects among child nodes per the env-chosen alternative.

    Intended to represent results associated with a particular region being selected."""

    alternatives: Alternatives
    region_idx_to_result: dict[int, Node]

    def evaluate(self, environment: dict[Node, int]) -> int:
        alt_idx = environment[self.alternatives]
        return self.region_idx_to_result[alt_idx].evaluate(environment)


def trace_smt_op(op: ir.Operation, env: dict) -> dict:
    """Add mapping from SMT op to a new Node that represents it to the env."""

    op = op.opview
    match op:
        case smt.IntConstantOp():
            env[op.result] = Constant(op.value.value)

        case smt.EqOp():
            assert len(op.operands) == 2
            env[op.result] = Predicate(eq, [env[value] for value in op.operands])

        case smt.IntAddOp():
            assert len(op.operands) == 2
            env[op.result] = Apply(add, [env[value] for value in op.operands])

        case smt.IntMulOp():
            assert len(op.operands) == 2
            env[op.result] = Apply(mul, [env[value] for value in op.operands])

        case smt.IntModOp():
            env[op.result] = Apply(mod, [env[op.lhs], env[op.rhs]])

        case smt.IntDivOp():
            env[op.result] = Apply(floordiv, [env[op.lhs], env[op.rhs]])

        case smt.IntCmpOp():
            operator = [lt, le, gt, ge][op.pred.value]
            env[op.result] = Predicate(operator, [env[op.lhs], env[op.rhs]])

        case smt.AssertOp():
            pred = env[op.input]
            assert isinstance(pred, Predicate), (
                "SMT assert expected argument to map to a Predicate node"
            )
            env[op] = pred

        case _:
            assert False, f"Unknown SMT operation: {op}"

    return env


def trace_tune_and_smt_ops(op: ir.Operation, env: Optional[dict] = None) -> dict:
    """Recursively add mapping from transform(.tune) and SMT ops to representative Nodes to env."""

    env = env if env is not None else {}  # TODO: nested scopes

    op = op.opview
    match op:
        case transform.ParamConstantOp():
            env[op.result] = Constant(op.value.value)

        case transform.MatchParamCmpIOp():
            operator = [eq, ne, le, lt, ge, gt][op.predicate.value]
            env[op] = Predicate(operator, [env[op.param], env[op.reference]])

        case transform_tune.KnobOp():
            kwargs = {}

            # Inspect attrs on KnobOp and convert to args to pass to Knob Node.
            if op.selected is not None:
                kwargs["options"] = (op.selected.value,)
            elif isinstance(op.options, ir.ArrayAttr):
                kwargs["options"] = tuple(opt.value for opt in op.options)
            elif isinstance(op.options, ir.DictAttr):
                if "options" in op.options:
                    kwargs["options"] = tuple(
                        opt.value for opt in op.options["options"]
                    )
                if "lower_bound" in op.options:
                    kwargs["lower_bound"] = op.options["lower_bound"].value
                if "upper_bound" in op.options:
                    kwargs["upper_bound"] = op.options["upper_bound"].value
                if "divisible_by" in op.options:
                    kwargs["divisible_by"] = op.options["divisible_by"].value
                if "divides" in op.options:
                    kwargs["divides"] = op.options["divides"].value
            else:
                assert False, "Unknown options attribute type"

            env[op.result] = Knob(name=op.name.value, **kwargs)

        case transform_tune.AlternativesOp():
            # Recursively visit each "alternative" child region, deriving a predicate
            # for each region and track which results are associated to that region.
            region_idx_to_pred = []
            result_idx_region_idx_to_node = defaultdict(lambda: dict())
            for reg_idx, region in enumerate(op.regions):
                region_preds = []
                for child in region.blocks[0]:
                    trace_tune_and_smt_ops(child.operation, env)
                    if (child_pred := env.get(child)) is not None:
                        assert isinstance(child_pred, Predicate)
                        region_preds.append(child_pred)
                assert isinstance(child, transform.YieldOp)

                region_idx_to_pred[reg_idx] = Predicate(
                    lambda *args: all(args), region_preds
                )
                for res_idx, yield_operand in enumerate(child.operands):
                    result_idx_region_idx_to_node[res_idx][reg_idx] = env[yield_operand]

            # Construct the node, that acts as a predicate, which represents
            # selecting among the "alternative" regions.
            env[op] = Alternatives(name=op.name, alt_idx_to_pred=region_idx_to_pred)

            # Construct the nodes, which act as a functions, that represent the
            # results corresponding to one of the "alternative" regions being selected.
            for res_idx, result in enumerate(op.results):
                env[result] = AlternativesResult(
                    alternatives=env[op],
                    region_idx_to_result=result_idx_region_idx_to_node[res_idx],
                )

        case transform_smt_ext.ConstrainParamsOp():
            # Map the block args in the op's region to the nodes already
            # associated to the corresponding arguments on the op itself.
            for operand, block_arg in zip(op.operands, op.body.arguments):
                env[block_arg] = env[operand]

            # Recursively trace the child (SMT) ops and construct an overall
            # predicate representing the block/region successfully terminating.
            child_predicates = []
            for child in islice(op.body.operations, len(op.body.operations) - 1):
                trace_smt_op(child, env)
                if (child_pred := env.get(child)) is not None:
                    assert isinstance(child_pred, Predicate)
                    child_predicates.append(child_pred)

            env[op] = Predicate(lambda *args: all(args), child_predicates)

            assert isinstance(smt_yield := op.body.operations[-1], smt.YieldOp)

            # Map the op's results to the nodes already associated to the
            # corresponding values yielded by the region/block's terminator.
            for yield_operand, op_res in zip(smt_yield.operands, op.results):
                env[op_res] = env[yield_operand]

        case transform.NamedSequenceOp() | transform.ForeachOp():
            # Recursively trace the child ops and construct an overall
            # predicate representing the block/region successfully terminating.
            child_predicates = []
            for child in op.body.operations:
                trace_tune_and_smt_ops(child.operation, env)
                if (child_pred := env.get(child)) is not None:
                    assert isinstance(child_pred, Predicate)
                    child_predicates.append(child_pred)

            env[op] = Predicate(lambda *args: all(args), child_predicates)

        case transform.ApplyPatternsOp():
            # A transform op with child ops we do skip over.
            pass

        case _:
            assert len(op.regions) == 0, f"Unhandled operation with regions: {op}"

    return env
