from .strategy_base import TilingStrategy
from .strategy_cache import CacheTilingStrategy
from .strategy_register_parallel import RegisterParallelTilingStrategy
from .strategy_register_reduction import RegisterReductionTilingStrategy
from .strategy_register_unroll import RegisterUnrollTilingStrategy


# Maps each canonical strategy name to its implementation class.
_STRATEGY_REGISTRY: dict[str, type[TilingStrategy]] = {
    "cache": CacheTilingStrategy,
    "register_parallel": RegisterParallelTilingStrategy,
    "register_reduction": RegisterReductionTilingStrategy,
    "register_unroll": RegisterUnrollTilingStrategy,
}


def normalize_strategy_name(name: str | None) -> str:
    """Canonicalize a strategy name to its registry key."""
    return (name or "cache").strip().lower().replace("-", "_")


def get_tiling_strategy(name: str) -> TilingStrategy:
    """Instantiate the tiling strategy registered under `name`.

    Raises ValueError if `name` (after normalization) is not a known strategy.
    """
    key = normalize_strategy_name(name)
    if key not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unknown tiling strategy: {name}")
    return _STRATEGY_REGISTRY[key]()
