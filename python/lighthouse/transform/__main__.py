import argparse
import sys

from mlir import ir

from .. import transform as lh_transform


if __name__ == "__main__":
    ArgParser = argparse.ArgumentParser()
    ArgParser.add_argument("schedule")
    ArgParser.add_argument("payload")
    args = ArgParser.parse_args(sys.argv[1:])

    with ir.Context(), ir.Location.unknown():
        with open(args.schedule) as f:
            schedule_module = ir.Module.parse(f.read())
        with open(args.payload) as f:
            payload_module = ir.Module.parse(f.read())

        schedule = schedule_module.body.operations[0]
        lh_transform.apply(schedule, payload_module)

        print(payload_module)
