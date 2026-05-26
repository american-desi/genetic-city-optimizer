"""City grid representation and zone type definitions."""


from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
import numpy as np


class Zone(IntEnum):
    """Zone types for the city grid."""
    RESIDENTIAL = 0
    COMMERCIAL = 1
    INDUSTRIAL = 2
    PARK = 3
    SCHOOL = 4
    HOSPITAL = 5
    FIRE_STATION = 6
    ROAD = 7


@dataclass(frozen=True)
class ZoneProperties:
    """Properties associated with each zone type."""
    name: str
    color: str
    noise_level: float       # 0.0 (silent) to 1.0 (very loud)
    pollution_level: float   # 0.0 (clean) to 1.0 (very polluted)
    population_density: float  # relative density of residents
    job_capacity: float      # relative number of jobs provided
    min_fraction: float      # minimum fraction of total grid
    max_fraction: float      # maximum fraction of total grid


ZONE_PROPERTIES = {
    Zone.RESIDENTIAL: ZoneProperties(
        name="Residential", color="#4CAF50", noise_level=0.1,
        pollution_level=0.05, population_density=1.0, job_capacity=0.0,
        min_fraction=0.25, max_fraction=0.45,
    ),
    Zone.COMMERCIAL: ZoneProperties(
        name="Commercial", color="#2196F3", noise_level=0.4,
        pollution_level=0.15, population_density=0.1, job_capacity=0.8,
        min_fraction=0.10, max_fraction=0.25,
    ),
    Zone.INDUSTRIAL: ZoneProperties(
        name="Industrial", color="#9E9E9E", noise_level=0.8,
        pollution_level=0.7, population_density=0.0, job_capacity=1.0,
        min_fraction=0.05, max_fraction=0.15,
    ),
    Zone.PARK: ZoneProperties(
        name="Park", color="#8BC34A", noise_level=0.0,
        pollution_level=0.0, population_density=0.0, job_capacity=0.05,
        min_fraction=0.08, max_fraction=0.20,
    ),
    Zone.SCHOOL: ZoneProperties(
        name="School", color="#FF9800", noise_level=0.3,
        pollution_level=0.02, population_density=0.0, job_capacity=0.3,
        min_fraction=0.02, max_fraction=0.06,
    ),
    Zone.HOSPITAL: ZoneProperties(
        name="Hospital", color="#F44336", noise_level=0.3,
        pollution_level=0.05, population_density=0.0, job_capacity=0.5,
        min_fraction=0.01, max_fraction=0.04,
    ),
    Zone.FIRE_STATION: ZoneProperties(
        name="Fire Station", color="#FF5722", noise_level=0.5,
        pollution_level=0.1, population_density=0.0, job_capacity=0.2,
        min_fraction=0.01, max_fraction=0.03,
    ),
    Zone.ROAD: ZoneProperties(
        name="Road", color="#795548", noise_level=0.5,
        pollution_level=0.3, population_density=0.0, job_capacity=0.0,
        min_fraction=0.08, max_fraction=0.18,
    ),
}

# Constraints: pairs of zone types that must NOT be adjacent
ADJACENCY_CONSTRAINTS = [
    (Zone.INDUSTRIAL, Zone.SCHOOL),
    (Zone.INDUSTRIAL, Zone.HOSPITAL),
    (Zone.INDUSTRIAL, Zone.RESIDENTIAL),
]


class CityGrid:
    """Represents a city as a 2D grid of zones."""

    def __init__(self, width: int = 30, height: int = 30, grid: np.ndarray | None = None):
        self.width = width
        self.height = height
        if grid is not None:
            self.grid = grid.copy()
        else:
            self.grid = self._random_grid()

    def _random_grid(self) -> np.ndarray:
        """Generate a random city grid respecting approximate zone fractions."""
        total = self.width * self.height
        cells = []
        for zone in Zone:
            props = ZONE_PROPERTIES[zone]
            target = int(total * (props.min_fraction + props.max_fraction) / 2)
            cells.extend([zone.value] * target)
        # Fill remaining cells with residential
        while len(cells) < total:
            cells.append(Zone.RESIDENTIAL.value)
        cells = cells[:total]
        rng = np.random.default_rng()
        rng.shuffle(cells)
        return np.array(cells, dtype=np.int8).reshape(self.height, self.width)

    def copy(self) -> "CityGrid":
        return CityGrid(self.width, self.height, self.grid)

    def zone_counts(self) -> dict[Zone, int]:
        """Count how many cells of each zone type exist."""
        unique, counts = np.unique(self.grid, return_counts=True)
        result = {zone: 0 for zone in Zone}
        for val, cnt in zip(unique, counts):
            result[Zone(val)] = int(cnt)
        return result

    def zone_fractions(self) -> dict[Zone, float]:
        """Fraction of total grid occupied by each zone."""
        total = self.width * self.height
        counts = self.zone_counts()
        return {zone: cnt / total for zone, cnt in counts.items()}

    def get_neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Get valid 4-connected neighbor coordinates."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append((nr, nc))
        return neighbors

    def get_positions(self, zone: Zone) -> list[tuple[int, int]]:
        """Get all (row, col) positions of a given zone type."""
        positions = np.argwhere(self.grid == zone.value)
        return [(int(r), int(c)) for r, c in positions]
