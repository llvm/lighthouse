"""
Genetic algorithm-based optimization of kernel parameters.
"""

import numpy as np
import random
import time
from types import FunctionType


class Variable:
    """Represents a single tunable parameter with list of valid choices."""

    def __init__(self, name: str, choices: list):
        self.name = name
        self.choices = choices

    def random_sample(self) -> int:
        return random.choice(self.choices)


class VariableSet:
    """A tunable variable set forming the search space."""

    def __init__(self, variables: list[Variable], is_valid_fn: FunctionType = None):
        self.variables = variables
        self.is_valid_fn = is_valid_fn

    def random_sample(self) -> list:
        return [var.random_sample() for var in self.variables]

    def names(self) -> list[str]:
        return [var.name for var in self.variables]

    def complexity(self) -> int:
        """Return total number of unconstrained combinations."""
        total = 1
        for var in self.variables:
            total *= len(var.choices)
        return total

    def is_valid(self, sample: list) -> bool:
        return self.is_valid_fn(self.sample_to_dict(sample))

    def sample_to_dict(self, sample: list) -> dict:
        assert len(sample) == len(self.variables)
        return dict(zip(self.names(), sample))

    def iterables(self) -> list:
        return [v.choices for v in self.variables]

    def print(self):
        print("Variable set:")
        for v in self.variables:
            print(f"{v.name}={v.choices}")
        print(f"Total complexity: {self.complexity()} configurations")


class Population:
    """A population of individuals drawn from the variable set."""

    def __init__(self, variable_set: VariableSet, individuals: list = None):
        self.variable_set = variable_set
        self.individuals = individuals if individuals is not None else []
        self.fitness_scores = []
        self.generation = 0

    def increment_generation(self):
        self.generation += 1

    def size(self) -> int:
        return len(self.individuals)

    def sort(self):
        scores = np.array(self.fitness_scores)
        i_sorted = np.argsort(scores)[::-1]
        self.individuals = [self.individuals[i] for i in i_sorted]
        self.fitness_scores = [self.fitness_scores[i] for i in i_sorted]

    def extend(self, new_individuals: list, new_fitness: list):
        assert len(new_individuals) == len(new_fitness)
        for ind, fit in zip(new_individuals, new_fitness):
            if ind not in self.individuals:
                self.individuals.append(ind)
                self.fitness_scores.append(fit)

    def shrink(self, nbest: int):
        if nbest >= len(self.individuals):
            return
        self.sort()
        self.individuals = self.individuals[:nbest]
        self.fitness_scores = self.fitness_scores[:nbest]

    def print(self):
        print(
            f"\nPopulation of size {len(self.individuals)}, generation {self.generation}:"
        )
        if not self.fitness_scores:
            for individual in self.individuals:
                print(f" {individual}")
        else:
            for individual, fitness in zip(self.individuals, self.fitness_scores):
                print(f" {fitness:.2f}: {individual}")
        print("\n")


def init_random_population(pop_size: int, variable_set: VariableSet) -> Population:
    population = Population(variable_set=variable_set)
    population.individuals = []
    i = 0
    while len(population.individuals) < pop_size:
        sample = variable_set.random_sample()
        if sample not in population.individuals and variable_set.is_valid(sample):
            population.individuals.append(sample)
        i += 1
        if i > pop_size * 10000 or i > 0.2 * variable_set.complexity():
            raise RuntimeError(
                "Unable to initialize population with given constraints."
            )
    return population


class GeneticAlgorithm:
    def __init__(
        self,
        population: Population,
        recombination_rate: float = 0.5,
        mutation_rate: float = 0.001,
        fertility_rate: float = 1.0,
        evaluate_fitness: FunctionType = None,
    ):
        self.fixed_population_size = population.size()
        self.population = population
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.fertility_rate = fertility_rate
        self.evaluate_fitness = evaluate_fitness
        self.ntrials = 50
        self.population_history = []
        self.fitness_history = []

    def recombine_and_mutate(self, individuals: list) -> list:
        variable_set = self.population.variable_set
        # every individual gets an update from another donor
        new_individuals = []
        npopulation = len(individuals)
        for i in range(npopulation):
            parent = individuals[i]
            donor_idx = random.choice([j for j in range(npopulation) if j != i])
            donor = individuals[donor_idx]
            for _ in range(self.ntrials):
                child = parent.copy()
                # perform recombination
                # one gene is always copied from donor
                force_idx = random.randint(0, len(child) - 1)
                # a gene is copied from donor with probability recombination_rate
                for j in range(len(child)):
                    if random.random() < self.recombination_rate or j == force_idx:
                        child[j] = donor[j]
                    # mutate
                    if random.random() < self.mutation_rate:
                        child[j] = variable_set.variables[j].random_sample()
                if (
                    child not in individuals
                    and child not in new_individuals
                    and variable_set.is_valid(child)
                ):
                    new_individuals.append(child)
                    break
        return new_individuals

    def initialize(self):
        if not self.population.fitness_scores:
            # evaluate fitness for the initial population
            self.population.fitness_scores = [
                self.evaluate_fitness(*ind) for ind in self.population.individuals
            ]
            self.population.sort()

    def next_generation(self):
        # select parents probabilistically based on fitness
        nb_parents = int(self.population.size() * self.fertility_rate)
        scores = np.array(self.population.fitness_scores)
        default = scores.min() / 20
        scores[scores == 0] = default
        parents = random.choices(
            population=self.population.individuals,
            k=nb_parents,
            weights=scores,
        )
        # get new set of individuals and extend population
        new_individuals = self.recombine_and_mutate(parents)
        new_fitness = [self.evaluate_fitness(*ind) for ind in new_individuals]
        self.population.extend(new_individuals, new_fitness)
        # keep only the best individuals
        self.population.shrink(self.fixed_population_size)
        self.population.increment_generation()

    def optimize(self, ngen: int, verbose: int = 0):
        self.initialize()
        tic = time.perf_counter()
        for gen in range(ngen):
            self.population_history.append(self.population.individuals.copy())
            self.fitness_history.append(self.population.fitness_scores.copy())
            self.next_generation()
            if verbose:
                best_individual = self.population.individuals[0]
                best_fitness = self.population.fitness_scores[0]
                scores = np.array(self.population.fitness_scores)
                avg_fitness = scores[scores > 0].mean()
                print(
                    f"Generation {self.population.generation:4d}: "
                    f" best: {best_fitness:.2f}, avg: {avg_fitness:.2f},"
                    f" best config: {best_individual}"
                )
        toc = time.perf_counter()
        if verbose:
            print(f"\nTime spent in optimization: {toc - tic:.2f} s\n")
