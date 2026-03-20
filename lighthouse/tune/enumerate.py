from itertools import product
from typing import Sequence

from .trace import Tuneable, Predicate


def all_satisfying_assignments(
    tuneables: Sequence[Tuneable], predicates: Sequence[Predicate]
):
    """Generate all assignments of values to the tuneables that satisfy the predicates."""

    for tuneable_values in product(
        *(tuneable.possibilities() for tuneable in tuneables)
    ):
        environment = dict(zip(tuneables, tuneable_values))
        for pred in predicates:
            if not pred.evaluate(environment):
                break
        else:
            yield environment
