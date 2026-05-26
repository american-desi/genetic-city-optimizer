"""Main entry point: run the genetic algorithm city layout optimizer."""


from __future__ import annotations
import time
import sys
import os
import numpy as np

# Allow running as `python main.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_city_optimizer.city import CityGrid, Zone, ZONE_PROPERTIES
from genetic_city_optimizer.genetic import GeneticOptimizer, GAConfig
from genetic_city_optimizer.fitness import OBJECTIVES, evaluate_all, weighted_fitness
from genetic_city_optimizer.visualization import (
    plot_before_after,
    plot_fitness_evolution,
    plot_pareto_front,
    plot_heatmaps,
)


def print_header():
    print("=" * 65)
    print("   GENETIC ALGORITHM CITY LAYOUT OPTIMIZER")
    print("=" * 65)
    print()


def print_config(config: GAConfig):
    print("Configuration:")
    print(f"  Grid size:        {config.grid_width} x {config.grid_height} ({config.grid_width * config.grid_height} cells)")
    print(f"  Population:       {config.population_size}")
    print(f"  Generations:      {config.generations}")
    print(f"  Tournament size:  {config.tournament_size}")
    print(f"  Crossover rate:   {config.crossover_rate}")
    print(f"  Crossover type:   {config.crossover_type}")
    print(f"  Mutation rate:    {config.mutation_rate}")
    print(f"  Cell mutation:    {config.mutation_cell_rate}")
    print(f"  Elitism count:    {config.elitism_count}")
    print(f"  Objectives:       {len(OBJECTIVES)}")
    for name, _, weight in OBJECTIVES:
        print(f"    - {name} (weight: {weight})")
    print()


def print_zone_distribution(city: CityGrid, label: str):
    fracs = city.zone_fractions()
    print(f"\n  Zone Distribution ({label}):")
    for zone in Zone:
        props = ZONE_PROPERTIES[zone]
        frac = fracs.get(zone, 0.0)
        bar_len = int(frac * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        status = ""
        if frac < props.min_fraction:
            status = " [LOW]"
        elif frac > props.max_fraction:
            status = " [HIGH]"
        print(f"    {props.name:14s} {frac:5.1%} [{bar}] target: {props.min_fraction:.0%}-{props.max_fraction:.0%}{status}")


def print_scores(scores: dict[str, float], label: str):
    print(f"\n  Objective Scores ({label}):")
    for name, _, _ in OBJECTIVES:
        val = scores.get(name, 0.0)
        bar_len = int(val * 30)
        bar = "=" * bar_len + "-" * (30 - bar_len)
        print(f"    {name:24s} {val:.4f} [{bar}]")
    wf = weighted_fitness(scores)
    print(f"    {'WEIGHTED TOTAL':24s} {wf:.4f}")


def print_statistics(stats: dict):
    print("\n" + "=" * 65)
    print("   OPTIMIZATION RESULTS")
    print("=" * 65)
    print(f"\n  Generations run:      {stats['generations_run']}")
    print(f"  Initial best fitness: {stats['initial_best_fitness']:.4f}")
    print(f"  Final best fitness:   {stats['final_best_fitness']:.4f}")
    print(f"  Improvement:          {stats['improvement']:.4f} ({stats['improvement_pct']:+.1f}%)")
    print(f"  Final avg fitness:    {stats['final_avg_fitness']:.4f}")
    print(f"  Final std deviation:  {stats['final_std']:.4f}")


def main():
    print_header()

    config = GAConfig(
        grid_width=30,
        grid_height=30,
        population_size=200,
        generations=500,
        tournament_size=5,
        crossover_rate=0.85,
        mutation_rate=0.15,
        mutation_cell_rate=0.03,
        elitism_count=10,
        crossover_type="uniform",
    )

    print_config(config)

    optimizer = GeneticOptimizer(config=config, seed=42)

    print("Initializing population...")
    optimizer.initialize_population()

    initial_scores = dict(optimizer.initial_best.scores)
    print_scores(initial_scores, "Initial Best")
    print_zone_distribution(optimizer.initial_best.city, "Initial Best")

    print("\n" + "-" * 65)
    print("Running optimization...\n")

    start_time = time.time()
    best = optimizer.run(verbose=True)
    elapsed = time.time() - start_time

    print(f"\nOptimization completed in {elapsed:.1f} seconds")

    # Final results
    stats = optimizer.get_statistics()
    print_statistics(stats)

    print_scores(best.scores, "Final Best")
    print_zone_distribution(best.city, "Final Best")

    # Per-objective improvement
    print("\n  Per-Objective Improvement:")
    for name, _, _ in OBJECTIVES:
        initial_val = stats["initial_scores"].get(name, 0.0)
        final_val = stats["best_scores"].get(name, 0.0)
        delta = final_val - initial_val
        pct = (delta / max(initial_val, 1e-9)) * 100
        arrow = "+" if delta >= 0 else ""
        print(f"    {name:24s} {initial_val:.4f} -> {final_val:.4f} ({arrow}{pct:.1f}%)")

    # Generate visualizations
    print("\n" + "-" * 65)
    print("Generating visualizations...\n")

    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    plot_before_after(
        optimizer.initial_best.city,
        best.city,
        initial_scores,
        best.scores,
        save_path=os.path.join(output_dir, "city_before_after.png"),
    )

    plot_fitness_evolution(
        optimizer.history,
        save_path=os.path.join(output_dir, "fitness_evolution.png"),
    )

    pareto_front = optimizer.get_pareto_front()
    if pareto_front:
        plot_pareto_front(
            pareto_front,
            obj_x="Commute Distance",
            obj_y="Green Space Access",
            save_path=os.path.join(output_dir, "pareto_front.png"),
        )

    plot_heatmaps(
        best.city,
        save_path=os.path.join(output_dir, "zone_heatmaps.png"),
    )

    print("\n" + "=" * 65)
    print("   DONE")
    print("=" * 65)
    print(f"\nOutput files:")
    print(f"  - city_before_after.png")
    print(f"  - fitness_evolution.png")
    print(f"  - pareto_front.png")
    print(f"  - zone_heatmaps.png")


if __name__ == "__main__":
    main()
