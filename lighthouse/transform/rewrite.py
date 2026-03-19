from mlir import ir

def move_block(
    source_block: ir.Block,
    target_block: ir.Block,
    arg_mapping: dict[ir.Value, ir.Value],
):
    # Create a mapping from source block arguments to target block arguments, making sure to cast to ir.Value for dict keys to ensure correct hashing and equality checks.
    env = dict((ir.Value(arg), ir.Value(arg_mapping[arg])) for arg in arg_mapping)
    with ir.InsertionPoint(target_block):
        for op in source_block.operations:
            target_block.append(op)
            for idx, operand in enumerate(op.operands):
                if (mapped_to_val := env.get(ir.Value(operand))) is not None:
                    op.operands[idx] = mapped_to_val
                else:
                    assert False, f"Operand {operand} not found in env"
            for result, clone_result in zip(op.results, op.results):
                env[ir.Value(result)] = ir.Value(clone_result)
            del op


def erase_block(block: ir.Block):
    for op in reversed(block.operations):
        op.erase()
