# RUN: %PYTHON %s | FileCheck %s

from lighthouse.execution.target import TargetInfo


TargetInfo.reset_host_cache()
host_before_overrides = TargetInfo.host()

source_features = ["amx_tile"]
with TargetInfo.override(features=source_features, arch="x86_64"):
    # Mutating the caller list after entering the context should not affect
    # the active override (it must snapshot/copy features on entry).
    source_features.append("unexpected")
    overridden_first = TargetInfo.host()
    overridden_second = TargetInfo.host()
    print(
        "override_values="
        f"{overridden_first.arch} "
        f"{','.join(overridden_first.features)} "
        f"same_obj={overridden_first is overridden_second}"
    )
    # Override should apply explicit arch/features and return fresh host objects.
    # CHECK: override_values=x86_64 amx_tile same_obj=False

    with TargetInfo.override(features=["avx2"], arch="aarch64"):
        nested = TargetInfo.host()
        print(f"nested_values={nested.arch} {','.join(nested.features)}")
        # Inner override should take precedence over outer override.
        # CHECK: nested_values=aarch64 avx2

    restored = TargetInfo.host()
    print(f"outer_restored={restored.arch} {','.join(restored.features)}")
    # After inner override exits, outer override values should be restored.
    # CHECK: outer_restored=x86_64 amx_tile

with TargetInfo.override(features=["sse4_2"], arch="x86_64"):
    fresh = TargetInfo.host()
    print(f"fresh_override_values={fresh.arch} {','.join(fresh.features)}")
    # A new override scope should be independent and use its own explicit values.
    # CHECK: fresh_override_values=x86_64 sse4_2

# Partial override with features only should keep host arch.
with TargetInfo.override(features=["avx512f"]):
    features_only = TargetInfo.host()
    print(
        "features_only="
        f"arch_same={features_only.arch == host_before_overrides.arch} "
        f"features={','.join(features_only.features)}"
    )
    # CHECK: features_only=arch_same=True features=avx512f

# Partial override with arch only should keep host features.
with TargetInfo.override(arch="x86_64"):
    arch_only = TargetInfo.host()
    print(
        "arch_only="
        f"arch={arch_only.arch} "
        f"features_same={arch_only.features == host_before_overrides.features}"
    )
    # CHECK: arch_only=arch=x86_64 features_same=True

host_after_overrides = TargetInfo.host()
same_info = (
    host_before_overrides.arch == host_after_overrides.arch
    and host_before_overrides.features == host_after_overrides.features
)
print(f"before_after_same_info={same_info}")
# Override scopes should not permanently change host arch/features.
# CHECK: before_after_same_info=True
