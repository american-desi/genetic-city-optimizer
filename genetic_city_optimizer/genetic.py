from __future__ import annotations
from typing import Optional
"""Genetic algorithm engine with NSGA-II style Pareto ranking.

Supports tournament selection, multiple crossover operators,
mutation, and elitism.
"""

import numpy as np
from dataclasses import dataclass, field
from .city import CityGrid, Zone, ZONE_PROPERTIES
from .fitness import evaluate_all, weighted_fitness, OBJECTIVES


@dataclass
class Individual:
    """A single individual in the GA population."""
    city: CityGrid
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0
    weighted_score: float = 0.0

    def evaluate(self) -> None:
        self.scores = evaluate_all(self.city)
        self.weighted_score = weighted_fitness(self.scores)


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""
    grid_width: int = 30
    grid_height: int = 30
    population_size: int = 200
    generations: int = 500
    tournament_size: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    mutation_cell_rate: float = 0.03
    elitism_count: int = 10
    crossover_type: str = "uniform"  # "uniform" or "single_point"


def _fast_nondominated_sort(population: list[Individual]) -> list[list[int]]:
    """NSGA-II fast non-dominated sorting.

    Returns list of fronts, where each front is a list of indices.
    """
    n = len(population)
    obj_names = [name for name, _, _ in OBJECTIVES]

    domination_count = np.zeros(n, dtype=np.int32)
    dominated_set: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    # Build score matrix for vectorized comparison
    score_matrix = np.array(
        [[population[i].scores.get(obj, 0.0) for obj in obj_names] for i in range(n)],
        dtype=np.float64,
    )

    for i in range(n):
        for j in range(i + 1, n):
            diff = score_matrix[i] - score_matrix[j]
            if np.all(diff >= 0) and np.any(diff > 0):
                # i dominates j
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif np.all(diff <= 0) and np.any(diff < 0):
                # j dominates i
                dominated_set[j].append(i)
                domination_count[i] += 1

    # First front
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = front_idx + 1
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    # Remove empty last front
    if not fronts[-1]:
        fronts.pop()

    return fronts


def _crowding_distance(population: list[Individual], front: list[int]) -> None:
    """Compute crowding distance for individuals in a front."""
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = float("inf")
        return

    obj_names = [name for name, _, _ in OBJECTIVES]

    for idx in front:
        population[idx].crowding_distance = 0.0

    for obj in obj_names:
        # Sort front by this objective
        sorted_front = sorted(front, key=lambda i: population[i].scores.get(obj, 0.0))

        # Boundary individuals get infinite distance
        population[sorted_front[0]].crowding_distance = float("inf")
        population[sorted_front[-1]].crowding_distance = float("inf")

        obj_min = population[sorted_front[0]].scores.get(obj, 0.0)
        obj_max = population[sorted_front[-1]].scores.get(obj, 0.0)
        obj_range = obj_max - obj_min

        if obj_range <= 0:
            continue

        for k in range(1, len(sorted_front) - 1):
            prev_val = population[sorted_front[k - 1]].scores.get(obj, 0.0)
            next_val = population[sorted_front[k + 1]].scores.get(obj, 0.0)
            population[sorted_front[k]].crowding_distance += (next_val - prev_val) / obj_range


def tournament_selection(
    population: list[Individual], tournament_size: int, rng: np.random.Generator
) -> Individual:
    """Select an individual via tournament selection using Pareto rank and crowding distance."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    candidates = [population[i] for i in indices]

    # Sort by rank (lower is better), then by crowding distance (higher is better)
    candidates.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
    return candidates[0]


def uniform_crossover(
    parent1: CityGrid, parent2: CityGrid, rng: np.random.Generator
) -> tuple[CityGrid, CityGrid]:
    """Uniform crossover: each cell independently chosen from either parent."""
    mask = rng.random((parent1.height, parent1.width)) < 0.5
    child1_grid = np.where(mask, parent1.grid, parent2.grid)
    child2_grid = np.where(mask, parent2.grid, parent1.grid)
    return (
        CityGrid(parent1.width, parent1.height, child1_grid),
        CityGrid(parent1.width, parent1.height, child2_grid),
    )


def single_point_crossover(
    parent1: CityGrid, parent2: CityGrid, rng: np.random.Generator
) -> tuple[CityGrid, CityGrid]:
    """Single-point crossover on flattened grid."""
    flat1 = parent1.grid.flatten()
    flat2 = parent2.grid.flatten()
    point = rng.integers(1, len(flat1))

    child1_flat = np.concatenate([flat1[:point], flat2[point:]])
    child2_flat = np.concatenate([flat2[:point], flat1[point:]])

    return (
        CityGrid(parent1.width, parent1.height, child1_flat.reshape(parent1.height, parent1.width)),
        CityGrid(parent1.width, parent1.height, child2_flat.reshape(parent1.height, parent1.width)),
    )


def mutate(city: CityGrid, cell_rate: float, rng: np.random.Generator) -> CityGrid:
    """Mutate a city grid by randomly changing zone types of some cells."""
    new_grid = city.grid.copy()
    mutation_mask = rng.random((city.height, city.width)) < cell_rate
    num_mutations = np.sum(mutation_mask)
    if num_mutations > 0:
        new_zones = rng.integers(0, len(Zone), size=num_mutations, dtype=np.int8)
        new_grid[mutation_mask] = new_zones
    return CityGrid(city.width, city.height, new_grid)


def _repair_zone_fractions(city: CityGrid, rng: np.random.Generator) -> CityGrid:
    """Light repair: if any zone is far outside its min/max fraction, nudge it back."""
    grid = city.grid.copy()
    total = city.width * city.height
    fractions = city.zone_fractions()

    for zone in Zone:
        props = ZONE_PROPERTIES[zone]
        frac = fractions.get(zone, 0.0)

        if frac < props.min_fraction * 0.5:
            # Add some cells of this zone by replacing random residential cells
            deficit = int((props.min_fraction - frac) * total * 0.5)
            if deficit > 0:
                # Find cells that are residential (most common, safe to replace)
                candidates = np.argwhere(grid == Zone.RESIDENTIAL.value)
                if len(candidates) > deficit:
                    chosen = rng.choice(len(candidates), size=deficit, replace=False)
                    for idx in chosen:
                        r, c = candidates[idx]
                        grid[r, c] = zone.value

    return CityGrid(city.width, city.height, grid)


class GeneticOptimizer:
    """Main genetic algorithm optimizer."""

    def __init__(self, config: Optional[GAConfig] = None, seed: Optional[int] = None):
        self.config = config or GAConfig()
        self.rng = np.random.default_rng(seed)
        self.population: list[Individual] = []
        self.history: list[dict] = []
        self.best_individual: Individual | None = None
        self.initial_best: Individual | None = None

    def initialize_population(self) -> None:
        """Create the initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            city = CityGrid(self.config.grid_width, self.config.grid_height)
            ind = Individual(city=city)
            ind.evaluate()
            self.population.append(ind)

        self._update_best()
        self.initial_best = Individual(
            city=self.best_individual.city.copy(),
            scores=dict(self.best_individual.scores),
            weighted_score=self.best_individual.weighted_score,
        )

    def _update_best(self) -> None:
        """Track the best individual by weighted fitness."""
        for ind in self.population:
            if self.best_individual is None or ind.weighted_score > self.best_individual.weighted_score:
                self.best_individual = Individual(
                    city=ind.city.copy(),
                    scores=dict(ind.scores),
                    weighted_score=ind.weighted_score,
                    rank=ind.rank,
                    crowding_distance=ind.crowding_distance,
                )

    def _record_history(self, generation: int) -> None:
        """Record statistics for this generation."""
        scores = [ind.weighted_score for ind in self.population]
        obj_names = [name for name, _, _ in OBJECTIVES]
        obj_avgs = {}
        for obj in obj_names:
            vals = [ind.scores.get(obj, 0.0) for ind in self.population]
            obj_avgs[obj] = float(np.mean(vals))

        self.history.append({
            "generation": generation,
            "best_fitness": float(np.max(scores)),
            "avg_fitness": float(np.mean(scores)),
            "worst_fitness": float(np.min(scores)),
            "std_fitness": float(np.std(scores)),
            "objective_avgs": obj_avgs,
        })

    def run(self, verbose: bool = True) -> Individual:
        """Run the genetic algorithm for the configured number of generations."""
        if not self.population:
            self.initialize_population()

        cfg = self.config

        # Choose crossover function
        crossover_fn = uniform_crossover if cfg.crossover_type == "uniform" else single_point_crossover

        self._rank_population()
        self._record_history(0)

        if verbose:
            print(f"Generation   0 | Best: {self.best_individual.weighted_score:.4f} | "
                  f"Avg: {self.history[-1]['avg_fitness']:.4f}")

        for gen in range(1, cfg.generations + 1):
            new_population: list[Individual] = []

            # Elitism: carry forward top individuals
            sorted_pop = sorted(self.population, key=lambda x: x.weighted_score, reverse=True)
            for i in range(cfg.elitism_count):
                elite = Individual(city=sorted_pop[i].city.copy())
                elite.evaluate()
                new_population.append(elite)

            # Generate offspring
            while len(new_population) < cfg.population_size:
                parent1 = tournament_selection(self.population, cfg.tournament_size, self.rng)
                parent2 = tournament_selection(self.population, cfg.tournament_size, self.rng)

                if self.rng.random() < cfg.crossover_rate:
                    child1, child2 = crossover_fn(parent1.city, parent2.city, self.rng)
                else:
                    child1 = parent1.city.copy()
                    child2 = parent2.city.copy()

                # Mutation
                if self.rng.random() < cfg.mutation_rate:
                    child1 = mutate(child1, cfg.mutation_cell_rate, self.rng)
                if self.rng.random() < cfg.mutation_rate:
                    child2 = mutate(child2, cfg.mutation_cell_rate, self.rng)

                # Light repair every 50 generations
                if gen % 50 == 0:
                    child1 = _repair_zone_fractions(child1, self.rng)
                    child2 = _repair_zone_fractions(child2, self.rng)

                ind1 = Individual(city=child1)
                ind1.evaluate()
                new_population.append(ind1)

                if len(new_population) < cfg.population_size:
                    ind2 = Individual(city=child2)
                    ind2.evaluate()
                    new_population.append(ind2)

            self.population = new_population[:cfg.population_size]
            self._rank_population()
            self._update_best()
            self._record_history(gen)

            if verbose and (gen % 50 == 0 or gen == cfg.generations):
                print(f"Generation {gen:3d} | Best: {self.best_individual.weighted_score:.4f} | "
                      f"Avg: {self.history[-1]['avg_fitness']:.4f}")

        return self.best_individual

    def _rank_population(self) -> None:
        """Apply NSGA-II ranking and crowding distance to the population.

        For large populations, use a fast approximate approach:
        sort by weighted score and assign rank tiers.
        """
        n = len(self.population)

        if n <= 50:
            # Full NSGA-II for small populations
            fronts = _fast_nondominated_sort(self.population)
            for front in fronts:
                _crowding_distance(self.population, front)
        else:
            # Approximate Pareto ranking for performance
            # Sort by weighted score and assign ranks in tiers
            sorted_indices = sorted(range(n), key=lambda i: self.population[i].weighted_score, reverse=True)
            tier_size = max(1, n // 10)
            for rank, start in enumerate(range(0, n, tier_size)):
                tier = sorted_indices[start:start + tier_size]
                for idx in tier:
                    self.population[idx].rank = rank

                # Compute crowding distance within tier using a subset of objectives
                if len(tier) > 2:
                    _crowding_distance(self.population, tier)
                else:
                    for idx in tier:
                        self.population[idx].crowding_distance = float("inf")

    def get_pareto_front(self) -> list[Individual]:
        """Get the Pareto-optimal front from the current population."""
        if len(self.population) <= 50:
            fronts = _fast_nondominated_sort(self.population)
            return [self.population[i] for i in fronts[0]] if fronts else []

        # For large populations, do NSGA-II on top 50
        sorted_pop = sorted(self.population, key=lambda x: x.weighted_score, reverse=True)
        top = sorted_pop[:50]
        fronts = _fast_nondominated_sort(top)
        return [top[i] for i in fronts[0]] if fronts else []

    def get_statistics(self) -> dict:
        """Compute final statistics about the optimization run."""
        if not self.history:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        return {
            "initial_best_fitness": initial["best_fitness"],
            "final_best_fitness": final["best_fitness"],
            "improvement": final["best_fitness"] - initial["best_fitness"],
            "improvement_pct": (
                (final["best_fitness"] - initial["best_fitness"]) / max(initial["best_fitness"], 1e-9) * 100
            ),
            "initial_avg_fitness": initial["avg_fitness"],
            "final_avg_fitness": final["avg_fitness"],
            "final_std": final["std_fitness"],
            "generations_run": len(self.history) - 1,
            "best_scores": dict(self.best_individual.scores) if self.best_individual else {},
            "initial_scores": dict(self.initial_best.scores) if self.initial_best else {},
        }
