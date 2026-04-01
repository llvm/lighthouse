import os
import re

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def convert_string(value: str) -> str | int | float | bool:
    if value == "True":
        return True
    elif value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_csv(line: str, separator: str = ",") -> dict:
    result = {}
    arg_tuples = line.split(separator)
    for arg in arg_tuples:
        if not arg:
            continue
        if "=" in arg:
            key, value = arg.split("=")
            result[key] = convert_string(value)
        else:
            result[arg] = True
    return result


def remove_args_and_opts(line: str) -> str:
    if m := re.search(r"^([^[{]*)", line):
        line = m.group(1)
    return line


def update_filename(line: str, filename: str) -> str:
    if m := re.search(r"^([^[{]+)(.*)$", line):
        line = filename + m.group(2)
    return line


def parse_args_and_opts(line: str) -> tuple[str, dict, dict]:
    args = {}
    options = {}

    # Args: [arg1=val1,args2]
    if m := re.search(r"\[([^]]*)\]", line):
        args_str = m.group(1)
        args = parse_csv(args_str, ",")

    # Opts: {arg1=val1 args2}
    if m := re.search(r"\{([^}]+)\}", line):
        opts_str = m.group(1)
        options = parse_csv(opts_str, " ")

    # Cleanup the original string
    line = remove_args_and_opts(line)

    return [line, args, options]


def import_mlir_module(path: str, context: ir.Context) -> ir.Module:
    """Import an MLIR text file into an MLIR module"""
    if path is None:
        raise ValueError("Path to the module must be provided.")
    if not os.path.exists(path):
        raise ValueError(f"Path to the module does not exist: {path}")
    with open(path, "r") as f:
        return ir.Module.parse(f.read(), context=context)


def apply_registered_pass(*args, **kwargs):
    """Utility function to add a bundle of passes to a Transform Schedule"""
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    """Matches a pattern to AnyType"""
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    """Runs canonicalization patterns on the given operation"""
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()


def cleanup_func(target):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_cse(func)
    canonicalize(func)


class PipelineInterrupt(Exception):
    """Exception to signal early termination of the transform schedule."""

    pass


def match_and_split(*args, nhandles=1, **kwargs):
    """Helper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops
