module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {deduplicate, op_name = "builtin.module"} : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "one-shot-bufferize" with options = {"bufferize-function-boundaries" = true, "function-boundary-type-conversion" = "identity-layout-map"} to %1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.apply_registered_pass "drop-equivalent-buffer-results" to %2 : (!transform.any_op) -> !transform.any_op
    %4 = transform.apply_registered_pass "convert-linalg-to-loops" to %3 : (!transform.any_op) -> !transform.any_op
    %5 = transform.apply_registered_pass "cse" to %4 : (!transform.any_op) -> !transform.any_op
    %6 = transform.apply_registered_pass "canonicalize" to %5 : (!transform.any_op) -> !transform.any_op
    %7 = transform.apply_registered_pass "convert-scf-to-cf" to %6 : (!transform.any_op) -> !transform.any_op
    %8 = transform.apply_registered_pass "convert-to-llvm" to %7 : (!transform.any_op) -> !transform.any_op
    %9 = transform.apply_registered_pass "reconcile-unrealized-casts" to %8 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}
