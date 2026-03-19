#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys

from mlir import ir
from mlir.dialects import transform

from lighthouse import dialects as lh_dialects


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="lighthouse.transform")
    arg_parser.add_argument("schedule", help="MLIR schedule module (path)")
    arg_parser.add_argument("payload", help="MLIR payload module (path)")
    args = arg_parser.parse_args(sys.argv[1:])

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        with open(args.schedule) as sched_file, open(args.payload) as payload_file:
            schedule_module = ir.Module.parse(sched_file.read())
            payload_module = ir.Module.parse(payload_file.read())

        schedule = schedule_module.body.operations[0]
        if not isinstance(schedule, transform.NamedSequenceOp):
            sys.exit(
                "The following op was expected to be a `transform.named_sequence`, instead got:\n"
                + str(schedule)
            )
        schedule.apply(payload_module)

        print(payload_module)
