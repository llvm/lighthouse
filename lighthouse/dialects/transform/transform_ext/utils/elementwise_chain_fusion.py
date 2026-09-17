"""Fusion of same-rank elementwise `linalg.generic` chains.

Merges an elementwise producer into an elementwise consumer of the same rank,
inlining the producer's body. Unlike ``linalg-fuse-elementwise-ops`` this never
fuses across a reduction, so a chain like

    %m = max_k %s                      (reduction, left alone)
    %d = %s - %m                       (elementwise)
    %p = exp(%d)                       (elementwise)
    %l = sum_k %p                      (reduction, left alone)

collapses to a single ``%p = exp(%s - %m)`` term and nothing else moves.
"""

from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import opview, op_users
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.dialects.transform.transform_ext.utils import linalg_structured as ls

__all__ = ["fuse_same_rank_elementwise_chains"]


def _is_identity_map(imap: ir.AffineMap) -> bool:
    """True if `imap` maps every loop dim to the tensor dim of the same position.

    `ir.AffineMap` has no `isIdentity`, so it is spelled out here.
    """
    if imap.n_dims != len(imap.results) or imap.n_symbols != 0:
        return False
    return all(
        isinstance(r, ir.AffineDimExpr) and r.position == i
        for i, r in enumerate(imap.results)
    )


def _is_all_parallel(op: ir.OpView) -> bool:
    return all(it == "parallel" for it in ls.iterator_types(op))


def _is_fusable_elementwise(op) -> bool:
    """An all-parallel single-result `linalg.generic` whose output map is identity.

    The identity output map is what makes the op's loop space the same thing as its
    result's dims, so a consumer reading the result under an identity map can adopt
    the producer's input maps unchanged.
    """
    ov = opview(op)
    if not isinstance(ov, linalg.GenericOp):
        return False
    if len(ov.results) != 1 or ls.num_dps_inits(ov) != 1:
        return False
    if not _is_all_parallel(ov):
        return False
    out_map = ls.indexing_map_for(ov, ls.dps_init_operands(ov)[0])
    return _is_identity_map(out_map)


def _producer_body_ignores_init(producer: ir.OpView) -> bool:
    """True if the producer's body never reads its output block argument.

    A body that accumulates into `outs` is not a pure function of its inputs, so
    inlining it into a consumer with a different destination would change what it
    computes.
    """
    init_arg = ls.region_output_args(producer)[0]
    return len(list(init_arg.uses)) == 0


def _find_fusable_operand(consumer: ir.OpView):
    """A (operand, producer) pair of `consumer` that can absorb its producer.

    The operand must be read under an identity map -- same rank, no broadcast and
    no transpose -- which is what "same rank chain" means here, and the producer
    must have this consumer as its only user so fusing does not duplicate work.
    """
    for operand in ls.dps_input_operands(consumer):
        producer = opview(operand.value.owner) if operand.value.owner else None
        if producer is None or not _is_fusable_elementwise(producer):
            continue
        if not _is_identity_map(ls.indexing_map_for(consumer, operand)):
            continue
        if ls.num_loops(producer) != ls.num_loops(consumer):
            continue
        if not _producer_body_ignores_init(producer):
            continue
        if len(op_users(producer.results[0])) != 1:
            continue
        return operand, producer
    return None, None


def _fuse_one(
    consumer: ir.OpView, operand: ls.Operand, producer: ir.OpView
) -> ir.OpView:
    """Replace `consumer` with a copy that computes `producer`'s body inline.

    ``%c = f(..., g(x, y), ...)`` for a producer ``%p = g(x, y)`` read by the
    consumer at `operand`: the producer's inputs take that operand's place, keeping
    their own indexing maps (legal because both output maps are the identity), and
    its body is cloned into the consumer's at the point the operand was read.
    """
    fused_index = ls.dps_input_operands(consumer).index(operand)
    consumer_ins = [o.value for o in ls.dps_input_operands(consumer)]
    consumer_in_maps = [
        ls.indexing_map_for(consumer, o) for o in ls.dps_input_operands(consumer)
    ]
    producer_ins = [o.value for o in ls.dps_input_operands(producer)]
    producer_in_maps = [
        ls.indexing_map_for(producer, o) for o in ls.dps_input_operands(producer)
    ]

    new_ins = (
        consumer_ins[:fused_index] + producer_ins + consumer_ins[fused_index + 1 :]
    )
    new_in_maps = (
        consumer_in_maps[:fused_index]
        + producer_in_maps
        + consumer_in_maps[fused_index + 1 :]
    )
    init = ls.dps_init_operands(consumer)[0]
    new_maps = new_in_maps + [ls.indexing_map_for(consumer, init)]

    with ir.InsertionPoint(consumer), consumer.location:
        fused = linalg.GenericOp(
            result_tensors=[r.type for r in consumer.results],
            inputs=new_ins,
            outputs=[init.value],
            indexing_maps=ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in new_maps]),
            iterator_types=consumer.iterator_types,
        )
        for name, attr in irr.op_attributes(consumer).items():
            if name not in ("indexing_maps", "iterator_types", "operandSegmentSizes"):
                fused.operation.attributes[name] = attr

        arg_types = [ir.ShapedType(v.type).element_type for v in new_ins]
        arg_types.append(ir.ShapedType(init.value.type).element_type)
        block = fused.regions[0].blocks.append(*arg_types)
        with ir.InsertionPoint(block):
            n_prod = len(producer_ins)
            producer_args = list(block.arguments[fused_index : fused_index + n_prod])
            # The producer's own init arg is unused (checked above), so it is left
            # unbound; its yielded value is what the consumer reads.
            pmap = irr.clone_block_body(
                producer.regions[0].blocks[0], producer_args + [None]
            )
            producer_terminator = list(producer.regions[0].blocks[0].operations)[-1]
            fused_value = pmap[producer_terminator.operands[0]]

            # The consumer's arguments in its own order: the inputs before the fused
            # operand, the inlined value in its place, the inputs after it, the init.
            consumer_args = list(block.arguments[:fused_index])
            consumer_args.append(fused_value)
            consumer_args += list(block.arguments[fused_index + n_prod : len(new_ins)])
            consumer_args.append(block.arguments[len(new_ins)])
            cmap = irr.clone_block_body(consumer.regions[0].blocks[0], consumer_args)
            consumer_terminator = list(consumer.regions[0].blocks[0].operations)[-1]
            linalg.yield_([cmap[consumer_terminator.operands[0]]])

    return fused


def fuse_same_rank_elementwise_chains(root: ir.Operation, rewriter) -> int:
    """Fuse every same-rank elementwise producer/consumer pair under `root`.

    Repeats to a fixed point so a chain of any length collapses into one op.
    Returns the number of fusions performed.
    """
    fused_count = 0
    changed = True
    while changed:
        changed = False
        candidates = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "linalg.generic":
                candidates.append(op)
            return ir.WalkResult.ADVANCE

        root.walk(visit, ir.WalkOrder.PRE_ORDER)
        for op in candidates:
            consumer = opview(op)
            if not _is_fusable_elementwise(consumer):
                continue
            operand, producer = _find_fusable_operand(consumer)
            if operand is None:
                continue
            fused = _fuse_one(consumer, operand, producer)
            rewriter.replace_op(consumer, list(fused.results))
            rewriter.erase_op(producer)
            fused_count += 1
            changed = True
            break
    return fused_count
