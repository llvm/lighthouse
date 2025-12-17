import sys
import argparse
from pprint import pprint
from typing import Mapping

import z3

from mlir import ir
import lighthouse.tune as lh_tune
from lighthouse.utils.types import LazyChainMap

HEADER = "//" * 40 + "\n// {}\n" + "//" * 40


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the MLIR file to process")
    parser.add_argument(
        "-n", type=int, help="Number of determinized schedules to find", default=1
    )
    parser.add_argument(
        "--print-smtlib",
        action="store_true",
        help="Print the constraints in SMT-LIB format",
    )
    parser.add_argument(
        "--print-model", action="store_true", help="Print the model from the SMT solver"
    )
    parser.add_argument(
        "--print-knobs-set",
        action="store_true",
        help="Print the schedule with knobs set",
    )
    args = parser.parse_args()

    file = sys.stdin if args.file == "-" else open(args.file, "r")
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(file.read())

        z3_constraints, values_to_z3_vars = (
            lh_tune.smt.z3.transform_tune_and_smt_ops_to_z3_constraints(
                module.operation
            )
        )

        solver = z3.Solver()
        solver.add(z3_constraints)

        if args.print_smtlib:
            print(HEADER.format("SMT-LIB constraints"))
            print(solver.sexpr())

        all_models = lh_tune.smt.z3.all_smt(solver, values_to_z3_vars.values())
        for i in range(args.n):
            model = next(all_models)
            if args.print_model:
                print(HEADER.format(f"SMT Model #{i + 1}"))
                pprint(model)

            env: Mapping[ir.Value | ir.Operation, ir.Attribute] = LazyChainMap(
                values_to_z3_vars, lambda var: lh_tune.smt.z3.model_to_mlir(model[var])
            )

            mod_op = lh_tune.rewrite.set_selected(module.operation, env)

            mod_op, undo = lh_tune.rewrite.constraint_results_to_constants(mod_op, env)

            if args.print_knobs_set:
                print(HEADER.format(f"Schedule #{i + 1} with knobs set"))
                print(mod_op)

            print(HEADER.format(f"Determinized schedule #{i + 1}"))
            print(lh_tune.rewrite.nondet_to_det(mod_op.clone()))

            undo()  # Undo the introduction of constants for the results of constraints.
