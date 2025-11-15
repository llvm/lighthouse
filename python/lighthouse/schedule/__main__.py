import argparse
import sys

from mlir import ir
from mlir.dialects import transform


if __name__ == "__main__":
    ArgParser = argparse.ArgumentParser(prog="lighthouse.transform")
    ArgParser.add_argument("schedule", help="file which contains the MLIR schedule module")
    ArgParser.add_argument("payload", help="file which contains the MLIR payload module")
    args = ArgParser.parse_args(sys.argv[1:])

    with ir.Context(), ir.Location.unknown():
        with open(args.schedule) as f:
            schedule_module = ir.Module.parse(f.read())
        with open(args.payload) as f:
            payload_module = ir.Module.parse(f.read())

        schedule = schedule_module.body.operations[0]
        assert(isinstance(schedule, transform.NamedSequenceOp))
        schedule.apply(payload_module)

        print(payload_module)
