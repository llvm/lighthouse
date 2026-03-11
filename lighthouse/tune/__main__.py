import sys
import argparse
from typing import Mapping

from mlir import ir
from mlir.dialects import transform
from lighthouse.tune import (
    rewrite as lh_tune_rewrite,
    trace as lh_tune_trace,
    enumerate as lh_tune_enumerate,
)
from lighthouse import dialects as lh_dialects
from lighthouse.utils.types import LazyChainMap

HEADER = "//" * 40 + "\n// {}\n" + "//" * 40


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=str, help="Path to the MLIR file to process")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["enumerate"],
        default="enumerate",
        help="Mode of operation",
    )
    parser.add_argument(
        "-n", type=int, help="Number of concrete schedules to output", default=1
    )
    args = parser.parse_args()

    file = sys.stdin if args.file == "-" else open(args.file, "r")
    with ir.Context() as ctx, ir.Location.unknown():
        lh_dialects.register_and_load()

        module = ir.Module.parse(file.read())

        if args.mode == "enumerate":
            # Trace the named_seq, obtaining a DAG of tunable nodes and nodes
            # which are functions and predicates dependent on the tunable nodes.
            named_seq = module.body.operations[0].opview
            assert isinstance(named_seq, transform.NamedSequenceOp)
            op_or_value_to_node = lh_tune_trace.trace_tune_and_smt_ops(
                named_seq.operation
            )

            # The predicate associated to the overall named_seq is the conjunct
            # -ion of each of the predicates for each operation in seq's body.
            overall_predicate = op_or_value_to_node[named_seq]
            assert isinstance(overall_predicate, lh_tune_trace.Predicate)
            tunables = list(
                set(
                    node
                    for node in op_or_value_to_node.values()
                    if isinstance(node, lh_tune_trace.NonDeterministic)
                )
            )

            # Start enumerating assignments for the tune.knob and tune.alternatives ops.
            count = 0
            for count, node_to_int in zip(
                range(1, args.n + 1),
                lh_tune_enumerate.all_satisfying_assignments(
                    tunables, [overall_predicate]
                ),
            ):
                if args.count_only:
                    if count >= args.n:
                        break
                    continue

                print(HEADER.format(f"Config {count}:"))

                i64 = ir.IntegerType.get_signless(64)

                # Map the tuneable ops to the attributes that should assigned to them.
                mapping: Mapping[ir.Value | ir.Operation, ir.Attribute] = LazyChainMap(
                    op_or_value_to_node,
                    lambda node: ir.IntegerAttr.get(i64, node_to_int[node]),
                )

                # Walk the IR, obtaining and setting the corresponding attr for each tuneable op.
                mod_op = lh_tune_rewrite.set_selected(module.operation, mapping)
                print(mod_op)

                if count >= args.n:
                    break
            print("// count:", count)
        else:
            assert False, "Other modes are not yet implemented"
