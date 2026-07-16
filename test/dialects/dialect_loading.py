# RUN: %PYTHON %s | FileCheck %s

"""Tests for context-aware, idempotent loading of Lighthouse dialect extensions.

A Python-defined dialect is bound to the MLIR context it is loaded into, yet
MLIR forbids loading the same dialect into a context twice. These tests exercise
`register_and_load` across fresh, repeated, and interleaved contexts to ensure
loading never raises (or aborts on a C++ assertion) and that the extension ends
up registered in every context it was requested for.
"""

import gc
import weakref

from mlir import ir
import lighthouse.dialects as lh_dialects


# An operation contributed by one of the IRDL-based extensions. Its presence in a
# context's operation registry is proof that the extension was loaded there.
PROBE_OP = "transform_ext.replace"


def run(f):
    print("Test:", f.__name__, flush=True)
    f()
    return f


# CHECK-LABEL: Test: test_fresh_load_registers_ops
@run
def test_fresh_load_registers_ops():
    """A fresh context lacks the extension ops until it is explicitly loaded."""
    ctx = ir.Context()
    # CHECK: before load: False
    print("before load:", ctx.is_registered_operation(PROBE_OP), flush=True)
    with ctx:
        lh_dialects.register_and_load()
    # CHECK: after load: True
    print("after load:", ctx.is_registered_operation(PROBE_OP), flush=True)


# CHECK-LABEL: Test: test_redundant_same_context_load_is_noop
@run
def test_redundant_same_context_load_is_noop():
    """Loading repeatedly into the same context must be a no-op, not an assert."""
    ctx = ir.Context()
    with ctx:
        lh_dialects.register_and_load()
        lh_dialects.register_and_load()
        lh_dialects.register_and_load()
    # CHECK: same-context reload ok: True
    print("same-context reload ok:", ctx.is_registered_operation(PROBE_OP), flush=True)


# CHECK-LABEL: Test: test_load_into_multiple_contexts
@run
def test_load_into_multiple_contexts():
    """The extensions can be loaded into several independent contexts.

    With naive `reload=False` this would raise "loaded in a different context";
    the automatic reload detection must handle it instead.
    """
    ctx1 = ir.Context()
    ctx2 = ir.Context()
    with ctx1:
        lh_dialects.register_and_load()
    with ctx2:
        lh_dialects.register_and_load()
    # CHECK: ctx1 loaded: True
    print("ctx1 loaded:", ctx1.is_registered_operation(PROBE_OP), flush=True)
    # CHECK: ctx2 loaded: True
    print("ctx2 loaded:", ctx2.is_registered_operation(PROBE_OP), flush=True)


# CHECK-LABEL: Test: test_return_to_previous_context_is_noop
@run
def test_return_to_previous_context_is_noop():
    """Returning to an earlier context must not attempt a second load.

    A dialect stays loaded in every context it was loaded into, so a naive
    "last context" tracker would try to reload here and abort on a duplicate
    operation-name registration assertion.
    """
    ctx1 = ir.Context()
    ctx2 = ir.Context()
    with ctx1:
        lh_dialects.register_and_load()
    with ctx2:
        lh_dialects.register_and_load()
    # Coming back to ctx1 (which already has the dialect) must be a no-op.
    with ctx1:
        lh_dialects.register_and_load()
    with ctx2:
        lh_dialects.register_and_load()
    # CHECK: interleaved reload ok: True True
    print(
        "interleaved reload ok:",
        ctx1.is_registered_operation(PROBE_OP),
        ctx2.is_registered_operation(PROBE_OP),
        flush=True,
    )


# CHECK-LABEL: Test: test_registry_holds_contexts_weakly
@run
def test_registry_holds_contexts_weakly():
    """The loaded-context registry must hold contexts weakly (no leak).

    Without weak references the registry would pin every context ever loaded and
    accumulate them across repeated compilations.
    """

    def load_fresh_context():
        ctx = ir.Context()
        with ctx:
            lh_dialects.register_and_load()
        return ctx

    ctx_a = load_fresh_context()
    ref_a = weakref.ref(ctx_a)

    # A later load re-emits the extensions into ctx_b, releasing the loader's
    # strong reference to ctx_a's module (its only non-weak referrer).
    ctx_b = load_fresh_context()

    # Drop the last strong reference to ctx_a; only the weak registry remains.
    del ctx_a
    gc.collect()
    # CHECK: dead context collected: True
    print("dead context collected:", ref_a() is None, flush=True)
    # Keep ctx_b referenced past the collection check above.
    # CHECK: alive context preserved: True
    print("alive context preserved:", ctx_b is not None, flush=True)

    # Loading into a new context still works after an earlier one was collected.
    ctx_c = load_fresh_context()
    # CHECK: reload after gc ok: True
    print("reload after gc ok:", ctx_c.is_registered_operation(PROBE_OP), flush=True)


# CHECK-LABEL: Test: test_explicit_reload_flag_is_safe
@run
def test_explicit_reload_flag_is_safe():
    """Passing reload=True must never abort, even in an already-loaded context."""
    ctx = ir.Context()
    with ctx:
        lh_dialects.register_and_load()
        # Explicit reload in the same context cannot re-load into MLIR and must
        # be handled gracefully rather than asserting.
        lh_dialects.register_and_load(reload=True)
    # CHECK: explicit reload ok: True
    print("explicit reload ok:", ctx.is_registered_operation(PROBE_OP), flush=True)


# CHECK: All dialect loading tests passed
print("All dialect loading tests passed", flush=True)
