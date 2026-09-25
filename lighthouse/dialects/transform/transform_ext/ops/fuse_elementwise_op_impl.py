"""Direct elementwise fusion using LLVM's areElementwiseOpsFusable checks."""

from mlir import ir
from mlir.dialects import affine, linalg

from lighthouse.utils.mlir import indexing_maps, is_linalg_all_loops_parallel, opview


def _compose(outer: ir.AffineMap, inner: ir.AffineMap) -> ir.AffineMap:
    """Compose symbol-free ``outer`` with ``inner`` using expression bindings."""
    return ir.AffineMap.get(
        inner.n_dims,
        inner.n_symbols,
        [expression.compose(inner) for expression in outer.results],
    )


def _covers_loops(maps: list[ir.AffineMap], num_loops: int) -> bool:
    """Check that every loop dimension appears directly in an indexing-map result.

    Only plain dimension expressions count; constants and expressions such as
    ``d0 + d1`` do not identify an individual loop bound.
    """
    covered = {
        expression.position
        for indexing_map in maps
        for expression in indexing_map.results
        if isinstance(expression, ir.AffineDimExpr)
    }
    return covered == set(range(num_loops))


def _fusion_maps(producer, consumer, operand_index):
    """Return original and composed fusion maps, or None if fusion is illegal."""
    # Match LLVM's tensor-semantics, parallel-loop, and input-edge requirements.
    if (
        not isinstance(producer, linalg.GenericOp)
        or not isinstance(consumer, linalg.GenericOp)
        or not producer.outputs
        or any(
            not isinstance(value.type, ir.RankedTensorType)
            for value in producer.outputs
        )
        or any(
            isinstance(value.type, (ir.MemRefType, ir.UnrankedMemRefType))
            for value in producer.inputs
        )
        or not is_linalg_all_loops_parallel(producer)
        or operand_index >= len(consumer.inputs)
    ):
        return None
    operand = consumer.inputs[operand_index]
    if (
        not isinstance(operand, ir.OpResult)
        or operand.owner != producer
        or not isinstance(operand.type, ir.RankedTensorType)
    ):
        return None
    original_producer_maps = indexing_maps(producer)
    original_consumer_maps = indexing_maps(consumer)
    original_producer_result_map = original_producer_maps[
        len(producer.inputs) + operand.result_number
    ]
    original_consumer_input_map = original_consumer_maps[operand_index]
    # All producer loop dims must be accessed by consumer and must be accessed only once.
    if (
        len(original_consumer_input_map.results) != len(producer.iterator_types)
        or not original_producer_result_map.is_permutation
    ):
        return None
    # Map consumer loops to producer loops through the inverse result map.
    inverse_results = [None] * original_producer_result_map.n_dims
    for index, expression in enumerate(original_producer_result_map.results):
        inverse_results[expression.position] = ir.AffineExpr.get_dim(index)
    inverse_producer_result_map = ir.AffineMap.get(
        original_producer_result_map.n_dims, 0, inverse_results
    )
    consumer_to_producer_map = _compose(
        inverse_producer_result_map, original_consumer_input_map
    )
    fused_producer_input_maps = [
        _compose(indexing_map, consumer_to_producer_map)
        for indexing_map in original_producer_maps[: len(producer.inputs)]
    ]
    # Removing the intermediate tensor (i.e. producer output) must not lose any reduction loop bounds.
    fused_retained_consumer_maps = [
        indexing_map
        for value, indexing_map in zip(consumer.operands, original_consumer_maps)
        if value != operand
    ]
    reduction = ir.AttrBuilder.get("linalg.IteratorTypeEnum")(
        linalg.IteratorType.reduction, context=consumer.context
    )
    if reduction in consumer.iterator_types and not _covers_loops(
        fused_retained_consumer_maps + fused_producer_input_maps,
        len(consumer.iterator_types),
    ):
        return None
    return (
        original_producer_maps,
        original_consumer_maps,
        consumer_to_producer_map,
        fused_producer_input_maps,
    )


def _clone(operation, mapping):
    """Clone an op, remap nested uses, and record its replacement results."""
    cloned = operation.clone()

    def remap(nested):
        """Update references to mapped values while preserving cloned locals."""
        for index, operand in enumerate(nested.operands):
            if operand in mapping:
                nested.operands[index] = mapping[operand]
        return ir.WalkResult.ADVANCE

    cloned.walk(remap)
    mapping.update(zip(operation.results, cloned.results))


def fuse_elementwise(producer, consumer, rewriter) -> ir.Operation | None:
    """Fuse one producer input edge, preserving the original producer and users.

    Only the selected consumer is replaced. Other users continue to use the
    original producer. Producer outs tensors become fused inputs when their
    values are read by the body or their shapes are needed for loop bounds.
    Return the fused op on success, or None with the payload unchanged on failure.
    """
    # Require one producer-to-consumer input edge and validate fusion legality.
    producer, consumer = opview(producer), opview(consumer)
    if (
        not isinstance(producer, linalg.GenericOp)
        or not isinstance(consumer, linalg.GenericOp)
        or producer == consumer
    ):
        return None
    producer_results = set(producer.results)
    edges = [
        index
        for index, value in enumerate(consumer.inputs)
        if value in producer_results
    ]
    # Only one producer-to-consumer input edge and, producer must be used as an input (i.e., used in DPS ins(...) of consumer).
    if len(edges) != 1 or any(value in producer_results for value in consumer.outputs):
        return None
    operand_index = edges[0]
    maps = _fusion_maps(producer, consumer, operand_index)
    if maps is None:
        return None
    (
        original_producer_maps,
        original_consumer_maps,
        consumer_to_producer_map,
        fused_producer_input_maps,
    ) = maps
    producer_body = producer.regions[0].blocks[0]
    consumer_body = consumer.regions[0].blocks[0]
    # Replace the intermediate input with producer inputs, maps, and block args.
    inputs = (
        list(consumer.inputs[:operand_index])
        + list(producer.inputs)
        + list(consumer.inputs[operand_index + 1 :])
    )
    fused_maps = (
        original_consumer_maps[:operand_index]
        + fused_producer_input_maps
        + original_consumer_maps[operand_index + 1 : len(consumer.inputs)]
    )
    arguments = (
        list(consumer_body.arguments[:operand_index])
        + list(producer_body.arguments[: len(producer.inputs)])
        + list(consumer_body.arguments[operand_index + 1 : len(consumer.inputs)])
    )
    # Keep producer inits needed for scalar computation or loop-bound recovery.
    original_consumer_output_maps = original_consumer_maps[len(consumer.inputs) :]
    for index, value in enumerate(producer.outputs):
        argument = producer_body.arguments[len(producer.inputs) + index]
        if list(argument.uses) or not _covers_loops(
            fused_maps + original_consumer_output_maps,
            consumer_to_producer_map.n_dims,
        ):
            inputs.append(value)
            arguments.append(argument)
            fused_maps.append(
                _compose(
                    original_producer_maps[len(producer.inputs) + index],
                    consumer_to_producer_map,
                )
            )
    fused_maps.extend(original_consumer_output_maps)
    if not _covers_loops(fused_maps, consumer_to_producer_map.n_dims):
        return None
    arguments.extend(consumer_body.arguments[len(consumer.inputs) :])
    # Build a detached candidate with the consumer's outputs and iteration space.
    fused = linalg.GenericOp(
        [value.type for value in consumer.results],
        inputs,
        list(consumer.outputs),
        ir.ArrayAttr.get(
            [ir.AffineMapAttr.get(indexing_map) for indexing_map in fused_maps]
        ),
        consumer.iterator_types,
        loc=consumer.location,
        ip=False,
    )
    body = fused.regions[0].blocks.append(*[argument.type for argument in arguments])
    mapping = dict(zip(arguments, body.arguments))
    producer_ops = list(producer_body.operations)
    consumer_ops = list(consumer_body.operations)
    with ir.InsertionPoint(body):
        # Producer linalg.index values must use producer, not consumer, coordinates.
        indices = []
        if any(isinstance(operation, linalg.IndexOp) for operation in producer_ops):
            indices = [
                linalg.IndexOp(index).result
                for index in range(consumer_to_producer_map.n_dims)
            ]
        for operation in producer_ops[:-1]:
            if isinstance(operation, linalg.IndexOp):
                fused_producer_index_map = consumer_to_producer_map.get_submap(
                    [operation.dim.value]
                )
                mapping[operation.result] = affine.AffineApplyOp(
                    fused_producer_index_map, indices
                ).result
            else:
                _clone(operation.operation, mapping)
        # Feed the producer's scalar yield directly into the cloned consumer body.
        result_number = consumer.inputs[operand_index].result_number
        yielded = producer_ops[-1].operands[result_number]
        mapping[consumer_body.arguments[operand_index]] = mapping.get(yielded, yielded)
        for operation in consumer_ops[:-1]:
            _clone(operation.operation, mapping)
        linalg.YieldOp(
            [mapping.get(value, value) for value in consumer_ops[-1].operands]
        )
    # Verify in place for captured-value dominance; discard only the candidate on failure.
    ir.InsertionPoint(consumer).insert(fused.operation)
    try:
        fused.operation.verify()
    except ir.MLIRError:
        fused.operation.erase()
        return None
    rewriter.replace_op(consumer, fused)
    return fused.operation
