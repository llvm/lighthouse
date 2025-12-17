import operator

from functools import reduce

from mlir import ir
from mlir.dialects import transform, smt
from mlir.dialects.transform import smt as transform_smt, tune as transform_tune

import z3


# From: http://theory.stanford.edu/%7Enikolaj/programmingz3.html#sec-blocking-evaluations
def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t))

    def fix_term(s, m, t):
        s.add(t == m.eval(t))

    def all_smt_rec(terms):
        if z3.sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()

    yield from all_smt_rec(list(initial_terms))


def model_to_mlir(x):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), x.py_value())


name_counter = 0


def fresh_name(prefix="fresh"):
    global name_counter
    name_counter += 1
    return f"{prefix}{name_counter}"


def transform_tune_and_smt_ops_to_z3_constraints(
    op: ir.Operation | ir.OpView, env=None, path=None, constraints=None
):
    env = env if env is not None else {}  # TODO: nested scopes
    path = path if path is not None else []
    constraints = constraints if constraints is not None else []

    match type(op):
        case transform.ParamConstantOp:
            name = fresh_name("cst")
            var = env[op.result] = z3.Int(name)
            C = op.value.value
            constraints += [z3.Implies(z3.And(*path), var == C)]

        case transform.MatchParamCmpIOp:
            lvar = env[op.operands[0]]
            rvar = env[op.operands[1]]
            predicate_to_operator = {
                transform.MatchCmpIPredicate.eq: operator.eq,
                transform.MatchCmpIPredicate.le: operator.le,
                transform.MatchCmpIPredicate.lt: operator.lt,
                transform.MatchCmpIPredicate.ge: operator.ge,
                transform.MatchCmpIPredicate.gt: operator.gt,
                transform.MatchCmpIPredicate.ne: operator.ne,
            }
            constraints += [
                z3.Implies(
                    z3.And(*path),
                    predicate_to_operator[
                        transform.MatchCmpIPredicate(op.predicate.value)
                    ](lvar, rvar),
                )
            ]

        case transform_tune.KnobOp:
            var = env[op.result] = z3.Int(op.name.value)
            if isinstance(op.options, ir.ArrayAttr):
                constraints += [
                    z3.Implies(
                        z3.And(*path), z3.Or(*[var == opt.value for opt in op.options])
                    )
                ]
            elif isinstance(op.options, ir.DictAttr):
                assert "lb" in op.options and "ub" in op.options
                atoms = [op.options["lb"].value <= var, var <= op.options["ub"].value]
                if "step" in op.options:
                    atoms += [var % op.options["step"].value == 0]

                constraints += [z3.Implies(z3.And(*path), z3.And(atoms))]
            else:
                assert False, "Unknown options attribute type"

        case transform_tune.AlternativesOp:
            var = env[op] = z3.Int(op.name.value)
            constraints += [
                z3.Implies(z3.And(path), z3.And(0 <= var, var < len(op.regions)))
            ]
            for idx, region in enumerate(op.regions):
                for child in region.blocks[0]:
                    transform_tune_and_smt_ops_to_z3_constraints(
                        child, env, path + [var == idx], constraints
                    )
                for yield_operand, result in zip(
                    region.blocks[0].operations[-1].operands, op.results
                ):
                    if isinstance(yield_operand.type, transform.ParamType):
                        env[result] = z3.Int(fresh_name("alt_res"))
                        constraints += [
                            z3.Implies(
                                z3.And(*(path + [var == idx])),
                                env[result] == env[yield_operand],
                            )
                        ]

        case transform_smt.ConstrainParamsOp:
            mapping_constraints = []
            for operand, block_arg in zip(op.operands, op.body.arguments):
                var = env[block_arg] = z3.Int(fresh_name("cp_bbarg"))
                mapping_constraints += [var == env[operand]]
            constraints += [z3.Implies(z3.And(*path), z3.And(*mapping_constraints))]
            for idx in range(len(op.body.operations) - 1):
                transform_tune_and_smt_ops_to_z3_constraints(
                    op.body.operations[idx], env, path, constraints
                )
            assert isinstance(op.body.operations[-1], smt.YieldOp)
            mapping_constraints = []
            for result, yield_arg in zip(op.results, op.body.operations[-1].operands):
                var = env[result] = z3.Int(fresh_name("cp_bbres"))
                mapping_constraints += [var == env[yield_arg]]
            constraints += [z3.Implies(z3.And(*path), z3.And(*mapping_constraints))]

        case smt.IntAddOp:
            var = env[op.result] = z3.Int(fresh_name("add"))
            constraints += [
                z3.Implies(
                    z3.And(*path), var == sum(env[value] for value in op.operands)
                )
            ]

        case smt.IntMulOp:
            var = env[op.result] = z3.Int(fresh_name("mul"))
            constraints += [
                z3.Implies(
                    z3.And(*path),
                    var == reduce(operator.mul, (env[value] for value in op.operands)),
                )
            ]

        case smt.IntConstantOp:
            var = env[op.result] = z3.Int(fresh_name("cst"))
            constraints += [z3.Implies(z3.And(*path), var == op.value.value)]

        case smt.EqOp:
            assert len(op.operands) == 2
            lhs, rhs = env[op.operands[0]], env[op.operands[1]]
            var = env[op.result] = z3.Bool(fresh_name("eq"))
            constraints += [z3.Implies(z3.And(*path), var == (lhs == rhs))]

        case smt.IntModOp:
            lhs, rhs = env[op.lhs], env[op.rhs]
            var = env[op.result] = z3.Int(fresh_name("int.mod"))
            constraints += [z3.Implies(z3.And(*path), var == (lhs % rhs))]

        case smt.IntDivOp:
            lhs, rhs = env[op.lhs], env[op.rhs]
            var = env[op.result] = z3.Int(fresh_name("int.div"))
            constraints += [z3.Implies(z3.And(*path), var == (lhs / rhs))]

        case smt.AssertOp:
            constraints += [z3.Implies(z3.And(*path), env[op.input])]

        case _:
            for region in op.regions:
                for block in region.blocks:
                    for child in block:
                        transform_tune_and_smt_ops_to_z3_constraints(
                            child, env, path, constraints
                        )

    return [z3.simplify(c) for c in constraints], env
