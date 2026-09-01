"""Configuration for the simulation-driven MO-ACO reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Physical shop-floor parameters (Sections 3.1 and 5.1; Table 2).
GRID_WIDTH: Final = 40
GRID_HEIGHT: Final = 28
CELL_SIZE_M: Final = 0.5
FREE_SPEED_MPS: Final = 1.0
SHARED_SPEED_MPS: Final = 0.6

# REFERENCE_IMPLEMENTATION_ASSUMPTION
# The manuscript does not publish a complete coordinate table. These deterministic
# coordinates form a synthetic layout with the same stated resources and qualitative
# aisle structure as Figure 2; they are not the original experimental coordinates.
MACHINE_NODES: Final = {
    "M1": (9, 21),
    "M2": (9, 14),
    "M3": (20, 6),
    "M4": (30, 21),
    "M5": (30, 14),
    "M6": (23, 6),
}
BUFFER_NODES: Final = {
    "BUF1": (2, 24),
    "BUF2": (37, 24),
    "BUF3": (2, 3),
    "BUF4": (37, 3),
}
CHARGING_STATION: Final = (20, 24)


def _rectangle(x0: int, x1: int, y0: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


OBSTACLE_CELLS: Final = frozenset(
    _rectangle(6, 8, 20, 22)
    | _rectangle(6, 8, 13, 15)
    | _rectangle(17, 19, 4, 6)
    | _rectangle(31, 33, 20, 22)
    | _rectangle(31, 33, 13, 15)
    | _rectangle(24, 26, 4, 6)
    | _rectangle(14, 14, 8, 12)
    | _rectangle(14, 14, 16, 20)
    | _rectangle(26, 26, 9, 12)
    | _rectangle(26, 26, 16, 19)
)
SHARED_ZONE_CELLS: Final = frozenset(
    _rectangle(4, 11, 11, 24)
    | _rectangle(28, 35, 11, 24)
    | _rectangle(15, 27, 8, 11)
)

# Temporary S1 blockages are sampled from these central corridor edges.
DISTURBANCE_SENSITIVE_EDGES: Final = tuple(
    ((x, 12), (x + 1, 12)) for x in range(15, 25)
) + tuple(((20, y), (20, y + 1)) for y in range(12, 23))


# Production system and fleet (Section 5.1; Table 2).
N_JOBS: Final = 24
OPERATIONS_MIN: Final = 3
OPERATIONS_MAX: Final = 5
PROCESSING_TIME_RANGE: Final = (18, 45)
SETUP_TIME_RANGE: Final = (6, 15)
N_AMRS: Final = 3
BATTERY_MAX: Final = 100.0
INITIAL_BATTERY: Final = 0.8 * BATTERY_MAX
BATTERY_CONSUMPTION_PER_M: Final = 1.0
BATTERY_SAFE: Final = 10.0
CHARGING_RATE: Final = 0.08
CHARGING_DURATIONS: Final = (5.0, 10.0, 15.0, 20.0, 30.0)
CHARGING_PORTS: Final = 2

# MO-ACO settings from Section 5.1 and Table 2.
PAPER_ANTS: Final = 30
PAPER_ITERATIONS: Final = 100
QUICK_ANTS: Final = 5
QUICK_ITERATIONS: Final = 5
ALPHA: Final = 1.0
BETA: Final = 2.0
RHO: Final = 0.1
ARCHIVE_MAX_SIZE: Final = 40
NO_IMPROVEMENT_LIMIT: Final = 20
HEURISTIC_WEIGHTS: Final = {
    "distance": 0.30,
    "queue": 0.15,
    "urgency": 0.25,
    "charging": 0.20,
    "risk": 0.10,
}
K_ROUTE_CANDIDATES: Final = 3

# REFERENCE_IMPLEMENTATION_ASSUMPTION
# These conventional internal constants are not numerically specified in the paper.
TAU0: Final = 1.0
PHEROMONE_Q: Final = 1.0
EPSILON: Final = 1e-9

SCENARIOS: Final = ("S1", "S2", "S3", "S4", "S5")
SCENARIO_WEIGHTS: Final = {name: 0.2 for name in SCENARIOS}
REFERENCE_IMPLEMENTATION_SEEDS: Final = tuple(range(1001, 1031))
JOB_ROUTE_SEED: Final = 314159

# Disturbances from Table 1.
BLOCKAGE_CHECK_INTERVAL_MIN: Final = 60.0
BLOCKAGE_PROBABILITY: Final = 0.15
BLOCKAGE_DURATION_RANGE: Final = (8, 15)
MACHINE_FAILURE_PROBABILITY_PER_HOUR: Final = 0.08
MACHINE_REPAIR_RANGE: Final = (10, 25)
CHARGING_OUTAGE_DURATION_MIN: Final = 12.0
PROCESSING_INFLATION_RANGE: Final = (1.0, 1.3)


@dataclass(frozen=True)
class RunSettings:
    """Computational scale; quick mode changes scale, not algorithm structure."""

    ants: int
    iterations: int
    scenarios: tuple[str, ...] = SCENARIOS
    no_improvement_limit: int = NO_IMPROVEMENT_LIMIT
    archive_max_size: int = ARCHIVE_MAX_SIZE
    base_seed: int = REFERENCE_IMPLEMENTATION_SEEDS[0]


QUICK_SETTINGS: Final = RunSettings(ants=QUICK_ANTS, iterations=QUICK_ITERATIONS)
PAPER_SETTINGS: Final = RunSettings(ants=PAPER_ANTS, iterations=PAPER_ITERATIONS)
