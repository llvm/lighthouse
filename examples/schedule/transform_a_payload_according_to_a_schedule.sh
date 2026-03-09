# RUN: %PYTHON %S/transform_a_payload_according_to_a_schedule.py payload > %S/payload.mlir
# RUN: %PYTHON %S/transform_a_payload_according_to_a_schedule.py schedule > %S/schedule.mlir
# RUN: lh-transform %S/schedule.mlir %S/payload.mlir | FileCheck %S/transform_a_payload_according_to_a_schedule.py
# RUN: rm -f %S/payload.mlir %S/schedule.mlir
