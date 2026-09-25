# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import affine, arith, func, linalg, transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(test):
    print(f"Test: {test.__name__}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        test()


PAYLOAD = """
module {
  func.func @pair(%arg: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %producer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%arg : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %neg = arith.negf %input : f32
      linalg.yield %neg : f32
    } -> tensor<4xf32>
    %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %exp = math.exp %input : f32
      linalg.yield %exp : f32
    } -> tensor<4xf32>
    return %consumer : tensor<4xf32>
  }
}
"""


def apply_fusion(payload, producer_index=0, consumer_index=1, expect_failure=False):
    before = str(payload)
    with schedule_boilerplate() as (schedule, sequence):
        matches = lh_transform.match_op(sequence.bodyTarget, "linalg.generic")
        producer = (
            matches
            if producer_index is None
            else transform_ext.extract_handle(matches, producer_index)
        )
        consumer = (
            lh_transform.match_op(sequence.bodyTarget, "linalg.fill")
            if consumer_index is None
            else transform_ext.extract_handle(matches, consumer_index)
        )
        fused = transform_ext.fuse_elementwise_op(producer, consumer)
        transform.annotate(fused, "test.fused")
        transform.annotate(producer, "test.producer")
        transform.yield_()
    assert schedule.operation.verify()
    assert ir.Module.parse(str(schedule)).operation.verify()
    try:
        sequence.apply(payload)
    except ValueError as error:
        assert expect_failure, error
        assert "Failed to apply named transform sequence" in str(error), error
        assert str(payload) == before
        print("Fusion rejected; payload unchanged")
    else:
        assert not expect_failure, payload
    assert payload.operation.verify()
    print(payload)


def get_generics(payload):
    return [
        operation
        for operation in payload.body.operations[0].regions[0].blocks[0].operations
        if isinstance(operation, linalg.GenericOp)
    ]


# CHECK-LABEL: Test: test_simple
# CHECK: linalg.generic
# CHECK-SAME: test.producer
# CHECK: linalg.generic
# CHECK-SAME: test.fused
# CHECK: %[[NEG:.*]] = arith.negf
# CHECK-NEXT: %[[EXP:.*]] = math.exp %[[NEG]]
# CHECK-NEXT: linalg.yield %[[EXP]]
@run
def test_simple():
    payload = ir.Module.parse(PAYLOAD)
    apply_fusion(payload)
    generics = get_generics(payload)
    assert len(generics) == 2, payload
    assert all(operand not in generics[0].results for operand in generics[1].operands)
    assert "test.fused" in generics[1].attributes
    assert "test.producer" in generics[0].attributes


# CHECK-LABEL: Test: test_shared_producer
# CHECK: %[[PRODUCER:.*]] = linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: ins(%[[PRODUCER]]
# CHECK: math.exp
# CHECK: linalg.generic
# CHECK-SAME: test.fused
# CHECK: %[[SHARED_NEG:.*]] = arith.negf
# CHECK-NEXT: math.exp %[[SHARED_NEG]]
@run
def test_shared_producer():
    payload = ir.Module.parse(PAYLOAD)
    producer, consumer = get_generics(payload)
    with ir.InsertionPoint(consumer):
        other_consumer = consumer.operation.clone()
    other_operands = list(other_consumer.operands)
    other_body = list(other_consumer.regions[0].blocks[0].operations)
    apply_fusion(payload, consumer_index=2)
    assert list(other_consumer.operands) == other_operands
    assert list(other_consumer.regions[0].blocks[0].operations) == other_body
    assert producer.result in other_consumer.operands
    fused = get_generics(payload)[-1]
    assert producer.result not in fused.operands


# CHECK-LABEL: Test: test_captured_scalar
# CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: test.fused
# CHECK: %[[CAPTURED_NEG:.*]] = arith.negf
# CHECK-NEXT: %[[SCALED:.*]] = arith.mulf %[[CAPTURED_NEG]], %[[SCALE]]
# CHECK-NEXT: math.exp %[[SCALED]]
@run
def test_captured_scalar():
    payload = ir.Module.parse(PAYLOAD)
    producer, consumer = get_generics(payload)
    with ir.InsertionPoint(producer):
        scale = arith.ConstantOp(ir.F32Type.get(), 2.0).result
    terminator = list(producer.regions[0].blocks[0].operations)[-1]
    with ir.InsertionPoint(terminator):
        scaled = arith.MulFOp(terminator.operands[0], scale).result
        terminator.operands[0] = scaled
    assert payload.operation.verify()
    apply_fusion(payload)
    fused = get_generics(payload)[-1]
    multiplies = [
        operation
        for operation in fused.regions[0].blocks[0].operations
        if isinstance(operation, arith.MulFOp)
    ]
    assert len(multiplies) == 1 and scale in multiplies[0].operands


def parse_map(text):
    return ir.AffineMapAttr.parse(text).value


def make_pair(
    input_type,
    producer_type,
    consumer_type,
    producer_maps,
    consumer_maps,
    producer_iterators,
    consumer_iterators,
    index_dim=None,
    use_init=False,
):
    types = [
        ir.Type.parse(type_name)
        for type_name in (input_type, producer_type, consumer_type)
    ]
    producer_result_types = (
        [types[1]] if isinstance(types[1], ir.RankedTensorType) else []
    )
    result_types = [types[-1]] if isinstance(types[-1], ir.RankedTensorType) else []
    payload = ir.Module.create()
    with ir.InsertionPoint(payload.body):
        function = func.FuncOp("pair", (types, result_types))
    body = function.add_entry_block()
    with ir.InsertionPoint(body):
        producer = linalg.GenericOp(
            producer_result_types,
            [body.arguments[0]],
            [body.arguments[1]],
            [parse_map(indexing_map) for indexing_map in producer_maps],
            producer_iterators,
        )
        producer_body = producer.regions[0].blocks.append(
            ir.F32Type.get(), ir.F32Type.get()
        )
        with ir.InsertionPoint(producer_body):
            value = arith.NegFOp(producer_body.arguments[0]).result
            if index_dim is not None:
                index = linalg.IndexOp(index_dim).result
                integer = arith.IndexCastOp(
                    ir.IntegerType.get_signless(64), index
                ).result
                value = arith.SIToFPOp(ir.F32Type.get(), integer).result
            if use_init:
                value = arith.AddFOp(value, producer_body.arguments[1]).result
            linalg.YieldOp([value])
        consumer_input = producer.result if producer.results else body.arguments[1]
        consumer = linalg.GenericOp(
            result_types,
            [consumer_input],
            [body.arguments[2]],
            [parse_map(indexing_map) for indexing_map in consumer_maps],
            consumer_iterators,
        )
        consumer_body = consumer.regions[0].blocks.append(
            ir.F32Type.get(), ir.F32Type.get()
        )
        with ir.InsertionPoint(consumer_body):
            value = arith.AddFOp(*consumer_body.arguments).result
            linalg.YieldOp([value])
        func.ReturnOp(list(consumer.results))
    assert payload.operation.verify()
    return payload


# CHECK-LABEL: Test: test_reduction_consumer
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["reduction"]
# CHECK-SAME: test.fused
# CHECK: arith.negf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: arith.addf
# CHECK-NEXT: linalg.yield
@run
def test_reduction_consumer():
    identity = "affine_map<(d0) -> (d0)>"
    scalar = "affine_map<(d0) -> ()>"
    payload = make_pair(
        "tensor<4xf32>",
        "tensor<4xf32>",
        "tensor<f32>",
        [identity, identity],
        [identity, scalar],
        ["parallel"],
        ["reduction"],
        use_init=True,
    )
    apply_fusion(payload)
    fused = get_generics(payload)[-1]
    assert len(fused.inputs) == 2
    assert len(fused.outputs) == 1


# CHECK-LABEL: Test: test_permuted_maps_and_index
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: test.fused
# CHECK: %[[ROW:.*]] = linalg.index 0
# CHECK-NEXT: %[[COL:.*]] = linalg.index 1
# CHECK: affine.apply {{.*}}(%[[ROW]], %[[COL]])
# CHECK: arith.index_cast
# CHECK-NEXT: arith.sitofp
@run
def test_permuted_maps_and_index():
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    transpose = "affine_map<(d0, d1) -> (d1, d0)>"
    payload = make_pair(
        "tensor<2x3xf32>",
        "tensor<3x2xf32>",
        "tensor<3x2xf32>",
        [identity, transpose],
        [identity, identity],
        ["parallel"] * 2,
        ["parallel"] * 2,
        index_dim=0,
    )
    apply_fusion(payload)
    fused = get_generics(payload)[-1]
    assert fused.indexing_maps[0].value == parse_map(transpose)
    applications = [
        operation
        for operation in fused.regions[0].blocks[0].operations
        if isinstance(operation, affine.AffineApplyOp)
    ]
    assert len(applications) == 1
    assert applications[0].map.value == parse_map("affine_map<(d0, d1) -> (d1)>")
    assert [operand.owner.dim.value for operand in applications[0].operands] == [0, 1]


# CHECK-LABEL: Test: test_affine_producer_input
# CHECK: #[[DIAGONAL:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: indexing_maps = [#[[DIAGONAL]],
# CHECK-SAME: test.fused
@run
def test_affine_producer_input():
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    diagonal = "affine_map<(d0, d1) -> (d0 + d1)>"
    payload = make_pair(
        "tensor<7xf32>",
        "tensor<4x4xf32>",
        "tensor<4x4xf32>",
        [diagonal, identity],
        [identity, identity],
        ["parallel"] * 2,
        ["parallel"] * 2,
    )
    apply_fusion(payload)
    assert get_generics(payload)[-1].indexing_maps[0].value == parse_map(diagonal)


# CHECK-LABEL: Test: test_legality_checks
# CHECK-COUNT-3: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
@run
def test_legality_checks():
    identity = "affine_map<(d0) -> (d0)>"
    scalar = "affine_map<(d0) -> ()>"
    payload = make_pair(
        "tensor<f32>",
        "tensor<4xf32>",
        "tensor<f32>",
        [scalar, identity],
        [identity, scalar],
        ["parallel"],
        ["reduction"],
    )
    apply_fusion(payload, expect_failure=True)
    payload = make_pair(
        "tensor<4xf32>",
        "tensor<f32>",
        "tensor<f32>",
        [identity, scalar],
        ["affine_map<() -> ()>"] * 2,
        ["reduction"],
        [],
        use_init=True,
    )
    apply_fusion(payload, expect_failure=True)
    payload = make_pair(
        "tensor<4xf32>",
        "tensor<4x1xf32>",
        "tensor<4x1xf32>",
        [identity, "affine_map<(d0) -> (d0, 0)>"],
        ["affine_map<(d0, d1) -> (d0, d1)>"] * 2,
        ["parallel"],
        ["parallel"] * 2,
    )
    apply_fusion(payload, expect_failure=True)


# CHECK-LABEL: Test: test_rejections
# CHECK-COUNT-6: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
@run
def test_rejections():
    for producer_index, consumer_index in [(None, 1), (0, None), (1, 0), (0, 0)]:
        apply_fusion(
            ir.Module.parse(PAYLOAD),
            producer_index,
            consumer_index,
            expect_failure=True,
        )
    payload = ir.Module.parse(PAYLOAD)
    producer, consumer = get_generics(payload)
    consumer.operands[0] = producer.inputs[0]
    apply_fusion(payload, expect_failure=True)
    payload = ir.Module.parse(PAYLOAD)
    producer, consumer = get_generics(payload)
    consumer.operands[0] = producer.inputs[0]
    consumer.operands[1] = producer.result
    apply_fusion(payload, expect_failure=True)


# CHECK-LABEL: Test: test_tensor_semantics
# CHECK: Fusion rejected; payload unchanged
# CHECK: memref<4xf32>
# CHECK-NOT: test.fused
@run
def test_tensor_semantics():
    identity = "affine_map<(d0) -> (d0)>"
    payload = make_pair(
        "memref<4xf32>",
        "memref<4xf32>",
        "memref<4xf32>",
        [identity] * 2,
        [identity] * 2,
        ["parallel"],
        ["parallel"],
    )
    apply_fusion(payload, expect_failure=True)


# CHECK-LABEL: Test: test_non_permutation_result_map
# CHECK: Fusion rejected; payload unchanged
# CHECK-NOT: test.fused
@run
def test_non_permutation_result_map():
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    payload = make_pair(
        "tensor<4x1xf32>",
        "tensor<4x1xf32>",
        "tensor<4x1xf32>",
        [identity, "affine_map<(d0, d1) -> (d0, 0)>"],
        [identity] * 2,
        ["parallel"] * 2,
        ["parallel"] * 2,
    )
    apply_fusion(payload, expect_failure=True)


# CHECK-LABEL: Test: test_zero_rank_and_dynamic_shapes
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: tensor<f32>
# CHECK-SAME: test.fused
# CHECK: linalg.generic
# CHECK: linalg.generic
# CHECK-SAME: tensor<?xf32>
# CHECK-SAME: test.fused
@run
def test_zero_rank_and_dynamic_shapes():
    for tensor_type, identity, iterators in (
        ("tensor<f32>", "affine_map<() -> ()>", []),
        ("tensor<?xf32>", "affine_map<(d0) -> (d0)>", ["parallel"]),
    ):
        payload = make_pair(
            tensor_type,
            tensor_type,
            tensor_type,
            [identity] * 2,
            [identity] * 2,
            iterators,
            iterators,
        )
        apply_fusion(payload)
