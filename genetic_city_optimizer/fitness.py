"""Multi-objective fitness functions for city layout evaluation.

Each fitness function returns a score where HIGHER is BETTER.
All scores are normalized to roughly [0, 1].
"""


from __future__ import annotations
import numpy as np
from .city import CityGrid, Zone, ZoneProperties, ZONE_PROPERTIES, ADJACENCY_CONSTRAINTS


def _distance_matrix(height: int, width: int) -> np.ndarray:
    """Precompute Manhattan distances between all cell pairs.

    Returns a (H*W, H*W) matrix of Manhattan distances.
    For performance, we compute distances on demand using coordinate math.
    """
    rows = np.arange(height * width) // width
    cols = np.arange(height * width) % width
    return rows, cols


def _min_distance_to_zone(grid: np.ndarray, zone: Zone, height: int, width: int) -> np.ndarray:
    """Compute minimum Manhattan distance from every cell to the nearest cell of given zone.

    Returns an (H, W) array of distances. Cells of the target zone have distance 0.
    Uses BFS for efficiency.
    """
    from collections import deque

    dist = np.full((height, width), fill_value=height + width, dtype=np.float32)
    queue = deque()

    targets = np.argwhere(grid == zone.value)
    for r, c in targets:
        dist[r, c] = 0
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                new_dist = dist[r, c] + 1
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    queue.append((nr, nc))

    return dist


def commute_distance_score(city: CityGrid) -> float:
    """Evaluate average commute distance from residential to commercial/industrial zones.

    Lower average commute -> higher score.
    """
    residential_mask = city.grid == Zone.RESIDENTIAL.value
    if not np.any(residential_mask):
        return 0.0

    # Distance to nearest commercial zone
    dist_commercial = _min_distance_to_zone(
        city.grid, Zone.COMMERCIAL, city.height, city.width
    )
    # Distance to nearest industrial zone
    dist_industrial = _min_distance_to_zone(
        city.grid, Zone.INDUSTRIAL, city.height, city.width
    )

    # Minimum of distance to commercial or industrial (either provides jobs)
    dist_jobs = np.minimum(dist_commercial, dist_industrial)

    avg_commute = np.mean(dist_jobs[residential_mask])
    max_possible = city.width + city.height
    # Normalize: 0 distance -> score 1.0, max distance -> score ~0
    return max(0.0, 1.0 - avg_commute / max_possible)


def green_space_access_score(city: CityGrid) -> float:
    """Evaluate how close residential zones are to parks.

    Every resident should be near a park.
    """
    residential_mask = city.grid == Zone.RESIDENTIAL.value
    if not np.any(residential_mask):
        return 0.0

    park_positions = np.argwhere(city.grid == Zone.PARK.value)
    if len(park_positions) == 0:
        return 0.0

    dist_park = _min_distance_to_zone(city.grid, Zone.PARK, city.height, city.width)
    avg_dist = np.mean(dist_park[residential_mask])

    # Ideal: parks within 3-5 blocks
    ideal_distance = 4.0
    max_dist = city.width + city.height
    score = max(0.0, 1.0 - avg_dist / max_dist)
    # Bonus for being close to ideal
    if avg_dist <= ideal_distance:
        score = min(1.0, score * 1.2)
    return min(1.0, score)


def service_coverage_score(city: CityGrid) -> float:
    """Evaluate coverage of essential services (school, hospital, fire station).

    All residential areas should be within reasonable distance of each service.
    """
    residential_mask = city.grid == Zone.RESIDENTIAL.value
    if not np.any(residential_mask):
        return 0.0

    service_zones = [Zone.SCHOOL, Zone.HOSPITAL, Zone.FIRE_STATION]
    coverage_radii = {
        Zone.SCHOOL: 8,
        Zone.HOSPITAL: 12,
        Zone.FIRE_STATION: 10,
    }

    total_score = 0.0
    for service_zone in service_zones:
        if not np.any(city.grid == service_zone.value):
            continue
        dist = _min_distance_to_zone(
            city.grid, service_zone, city.height, city.width
        )
        radius = coverage_radii[service_zone]
        # Fraction of residential cells within coverage radius
        covered = np.sum((dist[residential_mask] <= radius))
        total_residential = np.sum(residential_mask)
        coverage = covered / total_residential if total_residential > 0 else 0.0
        total_score += coverage

    return total_score / len(service_zones)


def noise_pollution_score(city: CityGrid) -> float:
    """Evaluate noise/pollution impact on residential areas.

    Residential zones near noisy/polluting zones should be penalized.
    Higher score = less noise pollution affecting residents.
    Uses direct neighbor counting for speed instead of full BFS.
    """
    grid = city.grid
    height, width = city.height, city.width

    residential_mask = grid == Zone.RESIDENTIAL.value
    if not np.any(residential_mask):
        return 1.0

    # Fast approach: for each residential cell, check neighbors within radius 2
    # and accumulate noise based on zone properties
    noise_values = np.zeros(len(Zone), dtype=np.float32)
    for zone in Zone:
        noise_values[zone.value] = ZONE_PROPERTIES[zone].noise_level

    # Build a per-cell noise level map (noise emitted by each cell)
    emitted_noise = noise_values[grid.astype(np.intp)]

    # Approximate noise exposure via shifted-grid averaging (fast convolution proxy)
    noise_map = np.zeros((height, width), dtype=np.float32)
    # Direct cell noise
    noise_map += emitted_noise
    # Neighbors at distance 1 (decay factor 0.5)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(emitted_noise, dr, axis=0), dc, axis=1)
        noise_map += shifted * 0.5
    # Neighbors at distance 2 (decay factor 0.25)
    for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.roll(np.roll(emitted_noise, dr, axis=0), dc, axis=1)
        noise_map += shifted * 0.25

    # Normalize
    max_noise = noise_map.max() if noise_map.max() > 0 else 1.0
    noise_map /= max_noise

    avg_noise_on_residential = np.mean(noise_map[residential_mask])
    return max(0.0, 1.0 - avg_noise_on_residential)


def land_use_efficiency_score(city: CityGrid) -> float:
    """Evaluate how well zone fractions match ideal distribution.

    Penalizes deviations from target zone proportions.
    """
    fractions = city.zone_fractions()
    total_penalty = 0.0

    for zone in Zone:
        props = ZONE_PROPERTIES[zone]
        frac = fractions.get(zone, 0.0)
        target = (props.min_fraction + props.max_fraction) / 2.0
        # Penalty for being outside bounds
        if frac < props.min_fraction:
            total_penalty += (props.min_fraction - frac) * 3.0
        elif frac > props.max_fraction:
            total_penalty += (frac - props.max_fraction) * 3.0
        # Small penalty for deviation from target even within bounds
        total_penalty += abs(frac - target) * 0.5

    return max(0.0, 1.0 - total_penalty)


def traffic_flow_score(city: CityGrid) -> float:
    """Evaluate road connectivity and traffic flow.

    Good traffic flow requires roads to form connected networks
    linking residential to commercial/industrial areas.
    """
    grid = city.grid
    height, width = city.height, city.width

    road_mask = grid == Zone.ROAD.value
    if not np.any(road_mask):
        return 0.0

    # Check road connectivity: what fraction of roads are in the largest connected component
    visited = np.zeros((height, width), dtype=bool)
    road_positions = np.argwhere(road_mask)

    if len(road_positions) == 0:
        return 0.0

    # BFS to find largest connected component of roads
    from collections import deque

    components = []
    for r, c in road_positions:
        if visited[r, c]:
            continue
        queue = deque([(r, c)])
        visited[r, c] = True
        size = 0
        while queue:
            cr, cc = queue.popleft()
            size += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and road_mask[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        components.append(size)

    total_road = np.sum(road_mask)
    largest_component = max(components)
    connectivity = largest_component / total_road

    # Check road adjacency to residential/commercial using numpy shifts
    res_mask = grid == Zone.RESIDENTIAL.value
    com_mask = (grid == Zone.COMMERCIAL.value) | (grid == Zone.INDUSTRIAL.value)

    # For each direction, check if shifting the residential/commercial mask overlaps with roads
    near_res = np.zeros((height, width), dtype=bool)
    near_com = np.zeros((height, width), dtype=bool)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        near_res |= np.roll(np.roll(res_mask, dr, axis=0), dc, axis=1)
        near_com |= np.roll(np.roll(com_mask, dr, axis=0), dc, axis=1)

    road_near_residential = np.sum(road_mask & near_res)
    road_near_commercial = np.sum(road_mask & near_com)

    if total_road > 0:
        access_score = (road_near_residential + road_near_commercial) / (2 * total_road)
    else:
        access_score = 0.0

    return 0.6 * connectivity + 0.4 * access_score


def constraint_violation_score(city: CityGrid) -> float:
    """Evaluate constraint violations (e.g., industrial not adjacent to schools).

    Returns 1.0 if no violations, decreasing toward 0 with more violations.
    Uses numpy shifts for fast adjacency checks.
    """
    grid = city.grid
    height, width = city.height, city.width
    violations = 0
    total_checks = 0

    for zone_a, zone_b in ADJACENCY_CONSTRAINTS:
        mask_a = grid == zone_a.value
        mask_b = grid == zone_b.value
        count_a = int(np.sum(mask_a))
        if count_a == 0:
            continue

        # For each direction, check if zone_a is adjacent to zone_b
        total_checks += count_a * 4  # approximate: 4 neighbors per cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_b = np.roll(np.roll(mask_b, dr, axis=0), dc, axis=1)
            violations += int(np.sum(mask_a & shifted_b))

    if total_checks == 0:
        return 1.0
    return max(0.0, 1.0 - violations / max(total_checks, 1))


# All fitness objectives: name, function, weight for weighted-sum fallback
OBJECTIVES = [
    ("Commute Distance", commute_distance_score, 1.5),
    ("Green Space Access", green_space_access_score, 1.2),
    ("Service Coverage", service_coverage_score, 1.3),
    ("Noise/Pollution", noise_pollution_score, 1.0),
    ("Land Use Efficiency", land_use_efficiency_score, 0.8),
    ("Traffic Flow", traffic_flow_score, 1.0),
    ("Constraint Compliance", constraint_violation_score, 2.0),
]


def evaluate_all(city: CityGrid) -> dict[str, float]:
    """Evaluate all fitness objectives for a city layout."""
    return {name: func(city) for name, func, _ in OBJECTIVES}


def weighted_fitness(scores: dict[str, float]) -> float:
    """Compute a single weighted fitness score from all objectives."""
    total = 0.0
    weight_sum = 0.0
    for name, _, weight in OBJECTIVES:
        total += scores.get(name, 0.0) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 0.0
