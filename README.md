# Genetic Algorithm City Layout Optimizer

A multi-objective genetic algorithm that optimizes city layouts on a 30x30 grid, balancing competing urban planning goals such as minimizing commute distances, maximizing green space access, ensuring service coverage, and managing noise and pollution.

## Zone Types

- **Residential** - Housing for citizens
- **Commercial** - Shops, offices, and businesses
- **Industrial** - Factories and warehouses
- **Park** - Green spaces and recreation
- **School** - Educational facilities
- **Hospital** - Healthcare services
- **Fire Station** - Emergency services
- **Road** - Transportation network

## Fitness Objectives

1. **Commute Distance** - Minimizes average distance from residential zones to jobs
2. **Green Space Access** - Ensures residents are close to parks
3. **Service Coverage** - Schools, hospitals, and fire stations cover all residential areas
4. **Noise/Pollution** - Reduces impact of noisy/polluting zones on residential areas
5. **Land Use Efficiency** - Maintains balanced zone distribution
6. **Traffic Flow** - Road connectivity and access to key areas
7. **Constraint Compliance** - Enforces rules like no industrial zones adjacent to schools

## GA Features

- NSGA-II style Pareto ranking with crowding distance
- Tournament selection
- Uniform and single-point crossover
- Cell-level mutation with configurable rate
- Elitism to preserve top solutions
- Zone fraction repair mechanism
- Population size: 200, Generations: 500

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Output

The optimizer produces four visualization files:

- `city_before_after.png` - Side-by-side comparison of initial random layout vs optimized layout
- `fitness_evolution.png` - Fitness scores over generations (best, average, per-objective)
- `pareto_front.png` - Pareto front visualization for two selected objectives
- `zone_heatmaps.png` - Distance heatmaps to key services in the optimized city
