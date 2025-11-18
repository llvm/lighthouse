#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys

from mlir import ir
from mlir.dialects import transform


if __name__ == "__main__":
    ArgParser = argparse.ArgumentParser(prog="lighthouse.transform")
    ArgParser.add_argument(
        "schedule", help="MLIR schedule module (path)"
    )
    ArgParser.add_argument(
        "payload", help="MLIR payload module (path)"
    )
    args = ArgParser.parse_args(sys.argv[1:])

    with ir.Context(), ir.Location.unknown():
        with open(args.schedule) as f:
            schedule_module = ir.Module.parse(f.read())
        with open(args.payload) as f:
            payload_module = ir.Module.parse(f.read())

        schedule = schedule_module.body.operations[0]
        if not isinstance(schedule, transform.NamedSequenceOp):
            sys.exit(
                "The following op was expected to be a `transform.named_sequence`, instead got:\n"
                + str(schedule)
            )
        schedule.apply(payload_module)

        print(payload_module)
