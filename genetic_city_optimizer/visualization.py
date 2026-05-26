"""Visualization utilities for city grids, fitness evolution, and Pareto fronts."""


from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from .city import CityGrid, Zone, ZONE_PROPERTIES
from .fitness import OBJECTIVES


def _build_zone_colormap():
    """Build a colormap and norm for zone types."""
    colors = [ZONE_PROPERTIES[zone].color for zone in Zone]
    cmap = ListedColormap(colors)
    boundaries = [z.value - 0.5 for z in Zone] + [max(z.value for z in Zone) + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def plot_city_grid(city: CityGrid, ax: plt.Axes, title: str = "City Layout") -> None:
    """Plot a city grid on the given axes."""
    cmap, norm = _build_zone_colormap()
    ax.imshow(city.grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(0, city.width, 5))
    ax.set_yticks(np.arange(0, city.height, 5))
    ax.grid(True, linewidth=0.3, alpha=0.5)


def _zone_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """Add a zone color legend."""
    patches = [
        mpatches.Patch(color=ZONE_PROPERTIES[zone].color, label=ZONE_PROPERTIES[zone].name)
        for zone in Zone
    ]
    ax.legend(
        handles=patches, loc="center", ncol=2, fontsize=9,
        frameon=False, title="Zone Types", title_fontsize=10,
    )
    ax.axis("off")


def plot_before_after(
    initial: CityGrid,
    final: CityGrid,
    initial_scores: dict[str, float],
    final_scores: dict[str, float],
    save_path: str = "city_before_after.png",
) -> None:
    """Plot before and after city grids side by side with score comparison."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], hspace=0.35, wspace=0.3)

    # Before grid
    ax1 = fig.add_subplot(gs[0, 0])
    plot_city_grid(initial, ax1, "Initial (Random) Layout")

    # After grid
    ax2 = fig.add_subplot(gs[0, 1])
    plot_city_grid(final, ax2, "Optimized Layout")

    # Legend
    ax_legend = fig.add_subplot(gs[0, 2])
    _zone_legend(fig, ax_legend)

    # Score comparison bar chart
    ax3 = fig.add_subplot(gs[1, :])
    obj_names = [name for name, _, _ in OBJECTIVES]
    x = np.arange(len(obj_names))
    width = 0.35

    initial_vals = [initial_scores.get(name, 0.0) for name in obj_names]
    final_vals = [final_scores.get(name, 0.0) for name in obj_names]

    bars1 = ax3.bar(x - width / 2, initial_vals, width, label="Initial", color="#EF5350", alpha=0.8)
    bars2 = ax3.bar(x + width / 2, final_vals, width, label="Optimized", color="#66BB6A", alpha=0.8)

    ax3.set_ylabel("Score")
    ax3.set_title("Objective Scores: Before vs After", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(obj_names, rotation=25, ha="right", fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=7)

    fig.suptitle("Genetic Algorithm City Layout Optimization", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved before/after comparison to {save_path}")


def plot_fitness_evolution(
    history: list[dict],
    save_path: str = "fitness_evolution.png",
) -> None:
    """Plot fitness evolution over generations."""
    generations = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    avg = [h["avg_fitness"] for h in history]
    worst = [h["worst_fitness"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall fitness evolution
    ax = axes[0]
    ax.plot(generations, best, label="Best", color="#2E7D32", linewidth=2)
    ax.plot(generations, avg, label="Average", color="#1976D2", linewidth=1.5)
    ax.fill_between(generations, worst, best, alpha=0.1, color="#66BB6A")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted Fitness")
    ax.set_title("Fitness Evolution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-objective evolution
    ax2 = axes[1]
    obj_names = [name for name, _, _ in OBJECTIVES]
    colors = plt.cm.tab10(np.linspace(0, 1, len(obj_names)))

    for i, obj_name in enumerate(obj_names):
        vals = [h["objective_avgs"].get(obj_name, 0.0) for h in history]
        ax2.plot(generations, vals, label=obj_name, color=colors[i], linewidth=1.2)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Average Score")
    ax2.set_title("Per-Objective Evolution", fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimization Progress", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved fitness evolution to {save_path}")


def plot_pareto_front(
    pareto_individuals: list,
    obj_x: str = "Commute Distance",
    obj_y: str = "Green Space Access",
    save_path: str = "pareto_front.png",
) -> None:
    """Plot the Pareto front for two selected objectives."""
    if not pareto_individuals:
        print("No Pareto front individuals to plot.")
        return

    x_vals = [ind.scores.get(obj_x, 0.0) for ind in pareto_individuals]
    y_vals = [ind.scores.get(obj_y, 0.0) for ind in pareto_individuals]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by weighted score
    weighted = [ind.weighted_score for ind in pareto_individuals]
    scatter = ax.scatter(x_vals, y_vals, c=weighted, cmap="viridis", s=60,
                         edgecolors="black", linewidth=0.5, alpha=0.8)

    # Sort and draw Pareto front line
    sorted_pairs = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    front_x = [p[0] for p in sorted_pairs]
    front_y = [p[1] for p in sorted_pairs]
    ax.plot(front_x, front_y, "r--", alpha=0.5, linewidth=1)

    ax.set_xlabel(obj_x, fontsize=11)
    ax.set_ylabel(obj_y, fontsize=11)
    ax.set_title(f"Pareto Front: {obj_x} vs {obj_y}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Weighted Fitness")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Pareto front to {save_path}")


def plot_heatmaps(city: CityGrid, save_path: str = "zone_heatmaps.png") -> None:
    """Plot heatmaps showing the distribution of key zone types."""
    from .fitness import _min_distance_to_zone

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("City Analysis Heatmaps", fontsize=14, fontweight="bold")

    heatmap_configs = [
        (Zone.PARK, "Distance to Nearest Park", "YlGn_r"),
        (Zone.SCHOOL, "Distance to Nearest School", "YlOrRd_r"),
        (Zone.HOSPITAL, "Distance to Nearest Hospital", "RdPu_r"),
        (Zone.FIRE_STATION, "Distance to Nearest Fire Station", "OrRd_r"),
        (Zone.COMMERCIAL, "Distance to Nearest Jobs", "Blues_r"),
        (Zone.ROAD, "Distance to Nearest Road", "Greys_r"),
    ]

    for ax, (zone, title, cmap_name) in zip(axes.flat, heatmap_configs):
        dist = _min_distance_to_zone(city.grid, zone, city.height, city.width)
        im = ax.imshow(dist, cmap=cmap_name, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(0, city.width, 10))
        ax.set_yticks(np.arange(0, city.height, 10))
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved heatmaps to {save_path}")
