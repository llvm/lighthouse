# RUN: %PYTHON %s | FileCheck %s

"""Tests for the expand_output_destination transform op.

The op rewrites `materialize_in_destination(collapse_shape(X), D_memref)` into
`materialize_in_destination(X, memref.expand_shape(D_memref))`, moving the
reshape off the result write path so the producer can bufferize in place.
"""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        with schedule_boilerplate() as (sched, named_seq):
            func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            transform_ext.expand_output_destination(func)
            transform.yield_()
        sched.body.operations[0].apply(payload.operation)
        payload.operation.verify()
        print(payload)


# A 5-D result collapsed to the 4-D output memref: the op expands the destination
# back to 5-D so the materialize writes the un-reshaped result in place.
COLLAPSE = """
module {
  func.func @main(%out: memref<4x32x512x64xbf16>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<4x8x4x512x64xbf16>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]]
        : tensor<4x8x4x512x64xbf16> into tensor<4x32x512x64xbf16>
    bufferization.materialize_in_destination %collapsed in restrict writable %out
        : (tensor<4x32x512x64xbf16>, memref<4x32x512x64xbf16>) -> ()
    return
  }
}
"""

# CHECK-LABEL: Test: collapse_to_expand
# The destination memref is expanded to the result's 5-D layout ...
# CHECK: memref.expand_shape
# CHECK-SAME: memref<4x32x512x64xbf16> into memref<4x8x4x512x64xbf16>
# ... and the result is materialized into it without any reshape on the source.
# CHECK: bufferization.materialize_in_destination %0 in restrict writable
# CHECK-SAME: (tensor<4x8x4x512x64xbf16>, memref<4x8x4x512x64xbf16>)
run("collapse_to_expand", COLLAPSE)


# No collapse_shape on the write path: the op leaves the materialize unchanged.
NOOP = """
module {
  func.func @main(%out: memref<4x32x512x64xbf16>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<4x32x512x64xbf16>
    bufferization.materialize_in_destination %0 in restrict writable %out
        : (tensor<4x32x512x64xbf16>, memref<4x32x512x64xbf16>) -> ()
    return
  }
}
"""

# CHECK-LABEL: Test: noop_no_collapse
# CHECK: bufferization.materialize_in_destination %0 in restrict writable
# CHECK-SAME: (tensor<4x32x512x64xbf16>, memref<4x32x512x64xbf16>)
# CHECK-NOT: memref.expand_shape
run("noop_no_collapse", NOOP)
