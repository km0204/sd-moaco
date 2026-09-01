"""SimPy discrete-event evaluation for complete MO-ACO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Iterator

import networkx as nx
import numpy as np
import simpy

import config


Node = tuple[int, int]
ActionKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class Operation:
    machine: str
    setup_time: float
    processing_time: float


@dataclass(frozen=True)
class JobDefinition:
    job_id: int
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class TransportationRequest:
    request_id: int
    job_id: int
    operation_index: int
    origin: Node
    destination: Node
    estimated_release_time: float
    operations_remaining: int


@dataclass(frozen=True)
class PolicyDecision:
    request_id: int
    assigned_amr: int
    service_priority: int
    route_index: int
    route_to_pickup: tuple[Node, ...]
    loaded_route: tuple[Node, ...]
    charging_inserted: bool
    charging_duration: float
    route_to_charger: tuple[Node, ...] = ()
    route_from_charger: tuple[Node, ...] = ()
    action_key: ActionKey = (0, 0, 0, 0)


@dataclass
class SimulationResult:
    scenario: str
    makespan: float
    travel_distance: float
    charging_actions: int
    average_charging_waiting_time: float
    emergency_charging_interruptions: int
    minimum_battery: float
    maximum_charging_users: int
    blocked_edge_violations: int
    obstacle_violations: int
    downstream_precedence_violations: int
    completed_job_route_violations: int


@dataclass
class CandidatePolicy:
    solution_id: str
    decisions: list[PolicyDecision] = field(default_factory=list)
    scenario_results: dict[str, SimulationResult] = field(default_factory=dict)
    f1_makespan: float | None = None
    f2_travel_distance: float | None = None
    crowding_distance: float = 0.0

    def action_keys(self) -> set[ActionKey]:
        return {decision.action_key for decision in self.decisions}


def build_shop_floor_graph() -> nx.Graph:
    """Build the weighted undirected four-neighbor grid from Sections 3.1/5.1."""

    graph = nx.Graph()
    for x in range(config.GRID_WIDTH):
        for y in range(config.GRID_HEIGHT):
            node = (x, y)
            if node not in config.OBSTACLE_CELLS:
                graph.add_node(node, shared=node in config.SHARED_ZONE_CELLS)

    for x, y in list(graph.nodes):
        for neighbor in ((x + 1, y), (x, y + 1)):
            if neighbor not in graph:
                continue
            shared = (x, y) in config.SHARED_ZONE_CELLS or neighbor in config.SHARED_ZONE_CELLS
            speed = config.SHARED_SPEED_MPS if shared else config.FREE_SPEED_MPS
            distance_m = config.CELL_SIZE_M
            # Simulation time is minutes; speed is meters per second.
            travel_time_min = distance_m / speed / 60.0
            edge = _canonical_edge((x, y), neighbor)
            graph.add_edge(
                (x, y),
                neighbor,
                distance_m=distance_m,
                time_min=travel_time_min,
                shared=shared,
                disturbance_sensitive=edge in _sensitive_edge_set(),
            )

    required_nodes = set(config.MACHINE_NODES.values()) | set(config.BUFFER_NODES.values()) | {
        config.CHARGING_STATION
    }
    if not required_nodes <= set(graph.nodes):
        raise ValueError("A machine, buffer, or charging coordinate lies on an obstacle")
    if not nx.is_connected(graph):
        raise ValueError("Synthetic reference layout must remain connected")
    return graph


def generate_reference_instance() -> tuple[list[JobDefinition], list[TransportationRequest]]:
    """Create deterministic synthetic job routes and their known request templates."""

    # REFERENCE_IMPLEMENTATION_ASSUMPTION
    # Exact job routes/times are unpublished. A fixed Generator creates transparent
    # synthetic routes with no immediately repeated machine.
    rng = np.random.default_rng(config.JOB_ROUTE_SEED)
    machine_names = tuple(config.MACHINE_NODES)
    jobs: list[JobDefinition] = []
    requests: list[TransportationRequest] = []
    request_id = 0

    for job_id in range(config.N_JOBS):
        operation_count = int(rng.integers(config.OPERATIONS_MIN, config.OPERATIONS_MAX + 1))
        route: list[str] = []
        for _ in range(operation_count):
            choices = [name for name in machine_names if not route or name != route[-1]]
            route.append(str(rng.choice(choices)))
        operations = tuple(
            Operation(
                machine=machine,
                setup_time=float(rng.integers(config.SETUP_TIME_RANGE[0], config.SETUP_TIME_RANGE[1] + 1)),
                processing_time=float(
                    rng.integers(config.PROCESSING_TIME_RANGE[0], config.PROCESSING_TIME_RANGE[1] + 1)
                ),
            )
            for machine in route
        )
        jobs.append(JobDefinition(job_id=job_id, operations=operations))

        estimated_release = 0.0
        for operation_index, operation in enumerate(operations[:-1]):
            estimated_release += operation.setup_time + operation.processing_time
            next_operation = operations[operation_index + 1]
            requests.append(
                TransportationRequest(
                    request_id=request_id,
                    job_id=job_id,
                    operation_index=operation_index,
                    origin=config.MACHINE_NODES[operation.machine],
                    destination=config.MACHINE_NODES[next_operation.machine],
                    estimated_release_time=estimated_release,
                    operations_remaining=len(operations) - operation_index - 1,
                )
            )
            request_id += 1
    return jobs, requests


def candidate_routes(graph: nx.Graph, source: Node, target: Node, count: int) -> list[tuple[Node, ...]]:
    """Return a small set of weighted feasible paths for route selection."""

    # REFERENCE_IMPLEMENTATION_ASSUMPTION
    # Alternative-route enumeration is not specified. We use the first K weighted
    # shortest simple paths only as an internal route generator, never as a benchmark.
    if source == target:
        return [(source,)]
    try:
        paths: Iterator[list[Node]] = nx.shortest_simple_paths(graph, source, target, weight="time_min")
        return [tuple(path) for _, path in zip(range(count), paths)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def path_distance(graph: nx.Graph, path: Iterable[Node]) -> float:
    nodes = tuple(path)
    return sum(graph.edges[u, v]["distance_m"] for u, v in zip(nodes, nodes[1:]))


def path_time(graph: nx.Graph, path: Iterable[Node]) -> float:
    nodes = tuple(path)
    return sum(graph.edges[u, v]["time_min"] for u, v in zip(nodes, nodes[1:]))


def nonlinear_charge(battery_in: float, duration_min: float) -> float:
    """Equations 5 and 22: nonlinear exponential charging recovery."""

    return config.BATTERY_MAX - (config.BATTERY_MAX - battery_in) * math.exp(
        -config.CHARGING_RATE * duration_min
    )


def _canonical_edge(u: Node, v: Node) -> tuple[Node, Node]:
    return (u, v) if u <= v else (v, u)


def _sensitive_edge_set() -> set[tuple[Node, Node]]:
    return {_canonical_edge(u, v) for u, v in config.DISTURBANCE_SENSITIVE_EDGES}


@dataclass
class _AMRState:
    amr_id: int
    position: Node
    battery: float
    distance: float = 0.0


class _SimulationModel:
    """One isolated policy/scenario replication; it knows no MO-ACO learning state."""

    def __init__(
        self,
        policy: CandidatePolicy,
        jobs: list[JobDefinition],
        requests: list[TransportationRequest],
        scenario: str,
        seed: int,
        reference_graph: nx.Graph,
    ) -> None:
        self.env = simpy.Environment()
        self.policy = policy
        self.jobs = jobs
        self.requests = requests
        self.request_by_id = {request.request_id: request for request in requests}
        self.decision_by_request = {decision.request_id: decision for decision in policy.decisions}
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.graph = reference_graph.copy()
        self.blocked_edges: set[tuple[Node, Node]] = set()
        self.graph_changed = self.env.event()
        self.station_operational = True
        self.station_changed = self.env.event()
        self.machine_down_until = {name: 0.0 for name in config.MACHINE_NODES}
        self.machines = {name: simpy.Resource(self.env, capacity=1) for name in config.MACHINE_NODES}
        self.charger = simpy.Resource(self.env, capacity=config.CHARGING_PORTS)
        self.amrs = [
            _AMRState(amr_id=i, position=config.CHARGING_STATION, battery=config.INITIAL_BATTERY)
            for i in range(config.N_AMRS)
        ]
        self.amr_queues = [simpy.PriorityStore(self.env) for _ in range(config.N_AMRS)]
        self.delivery_events = {request.request_id: self.env.event() for request in requests}
        self.delivery_times: dict[int, float] = {}
        self.operation_starts: dict[tuple[int, int], float] = {}
        self.completed_machines: dict[int, list[str]] = {job.job_id: [] for job in jobs}
        self.finished_jobs = 0
        self.finished_event = self.env.event()
        self.job_completion_times: dict[int, float] = {}
        self.charging_actions = 0
        self.charging_waits: list[float] = []
        self.emergency_interruptions = 0
        self.minimum_battery = config.INITIAL_BATTERY
        self.maximum_charging_users = 0
        self.blocked_edge_violations = 0
        self.obstacle_violations = 0
        self.downstream_precedence_violations = 0

    @property
    def has_blockages(self) -> bool:
        return self.scenario in {"S1", "S5"}

    @property
    def has_machine_downtime(self) -> bool:
        return self.scenario in {"S2", "S5"}

    @property
    def has_station_outage(self) -> bool:
        return self.scenario in {"S3", "S5"}

    @property
    def has_processing_inflation(self) -> bool:
        return self.scenario in {"S4", "S5"}

    def run(self) -> SimulationResult:
        if set(self.decision_by_request) != set(self.request_by_id):
            raise ValueError("DES requires one complete policy decision for every request")
        if len(self.policy.decisions) != len(self.requests):
            raise ValueError("DES received a policy with duplicate or missing request decisions")

        for amr in self.amrs:
            self.env.process(self._amr_worker(amr))
        for job in self.jobs:
            self.env.process(self._job_process(job))
        if self.has_blockages:
            self.env.process(self._blockage_process())
        if self.has_machine_downtime:
            self.env.process(self._machine_failure_process())
        if self.has_station_outage:
            self.env.process(self._station_outage_process())

        self.env.run(until=self.finished_event)
        makespan = max(self.job_completion_times.values())
        travel_distance = sum(amr.distance for amr in self.amrs)
        route_violations = sum(
            self.completed_machines[job.job_id] != [operation.machine for operation in job.operations]
            for job in self.jobs
        )
        return SimulationResult(
            scenario=self.scenario,
            makespan=float(makespan),
            travel_distance=float(travel_distance),
            charging_actions=self.charging_actions,
            average_charging_waiting_time=float(np.mean(self.charging_waits)) if self.charging_waits else 0.0,
            emergency_charging_interruptions=self.emergency_interruptions,
            minimum_battery=self.minimum_battery,
            maximum_charging_users=self.maximum_charging_users,
            blocked_edge_violations=self.blocked_edge_violations,
            obstacle_violations=self.obstacle_violations,
            downstream_precedence_violations=self.downstream_precedence_violations,
            completed_job_route_violations=int(route_violations),
        )

    def _job_process(self, job: JobDefinition):
        for operation_index, operation in enumerate(job.operations):
            if operation_index > 0:
                request_id = next(
                    request.request_id
                    for request in self.requests
                    if request.job_id == job.job_id and request.operation_index == operation_index - 1
                )
                yield self.delivery_events[request_id]
                if self.env.now + config.EPSILON < self.delivery_times[request_id]:
                    self.downstream_precedence_violations += 1

            machine = self.machines[operation.machine]
            with machine.request() as machine_request:
                yield machine_request
                self.operation_starts[(job.job_id, operation_index)] = self.env.now
                yield self.env.timeout(operation.setup_time)
                inflation = (
                    float(self.rng.uniform(*config.PROCESSING_INFLATION_RANGE))
                    if self.has_processing_inflation
                    else 1.0
                )
                remaining = operation.processing_time * inflation
                # REFERENCE_IMPLEMENTATION_ASSUMPTION
                # Machine failures use preempt-and-resume in one-minute event slices.
                while remaining > config.EPSILON:
                    down_for = self.machine_down_until[operation.machine] - self.env.now
                    if down_for > config.EPSILON:
                        yield self.env.timeout(down_for)
                        continue
                    work = min(1.0, remaining)
                    yield self.env.timeout(work)
                    remaining -= work

            self.completed_machines[job.job_id].append(operation.machine)
            if operation_index < len(job.operations) - 1:
                request = next(
                    request
                    for request in self.requests
                    if request.job_id == job.job_id and request.operation_index == operation_index
                )
                decision = self.decision_by_request[request.request_id]
                yield self.amr_queues[decision.assigned_amr].put(
                    (decision.service_priority, request.request_id)
                )
            else:
                self.job_completion_times[job.job_id] = self.env.now
                self.finished_jobs += 1
                if self.finished_jobs == len(self.jobs) and not self.finished_event.triggered:
                    self.finished_event.succeed()

    def _amr_worker(self, amr: _AMRState):
        queue = self.amr_queues[amr.amr_id]
        while True:
            _, request_id = yield queue.get()
            request = self.request_by_id[request_id]
            decision = self.decision_by_request[request_id]

            if decision.charging_inserted:
                yield from self._charge(amr, decision.charging_duration, planned=True, decision=decision)

            if not self._task_is_feasible(amr, request, decision):
                # REFERENCE_IMPLEMENTATION_ASSUMPTION
                # This is a feasibility-only safety intervention, not policy optimization.
                self.emergency_interruptions += 1
                yield from self._charge(amr, max(config.CHARGING_DURATIONS), planned=False, decision=decision)
                while not self._task_is_feasible(amr, request, decision):
                    yield from self._charge(
                        amr, max(config.CHARGING_DURATIONS), planned=False, decision=decision
                    )

            planned_empty = decision.route_to_pickup if not decision.charging_inserted else decision.route_from_charger
            yield from self._move(amr, request.origin, planned_empty)
            yield self.env.timeout(0.01)  # pickup-completion event
            yield from self._move(amr, request.destination, decision.loaded_route)
            yield self.env.timeout(0.01)  # delivery-completion event
            self.delivery_times[request_id] = self.env.now
            if not self.delivery_events[request_id].triggered:
                self.delivery_events[request_id].succeed()

    def _task_is_feasible(
        self, amr: _AMRState, request: TransportationRequest, decision: PolicyDecision
    ) -> bool:
        try:
            d1 = self._shortest_distance(amr.position, request.origin)
            if self._route_is_currently_usable(
                decision.loaded_route, request.origin, request.destination
            ):
                d2 = path_distance(self.graph, decision.loaded_route)
            else:
                d2 = self._shortest_distance(request.origin, request.destination)
            d3 = self._shortest_distance(request.destination, config.CHARGING_STATION)
        except nx.NetworkXNoPath:
            return False
        required = config.BATTERY_CONSUMPTION_PER_M * (d1 + d2 + d3) + config.BATTERY_SAFE
        return amr.battery + config.EPSILON >= required

    def _charge(
        self,
        amr: _AMRState,
        duration: float,
        planned: bool,
        decision: PolicyDecision,
    ):
        while True:
            try:
                distance = self._shortest_distance(amr.position, config.CHARGING_STATION)
                if amr.battery + config.EPSILON >= (
                    config.BATTERY_CONSUMPTION_PER_M * distance + config.BATTERY_SAFE
                ):
                    break
            except nx.NetworkXNoPath:
                pass
            yield self.graph_changed

        planned_route = decision.route_to_charger if planned else ()
        yield from self._move(amr, config.CHARGING_STATION, planned_route)
        yield from self._wait_for_station()
        arrival = self.env.now
        with self.charger.request() as port_request:
            yield port_request
            yield from self._wait_for_station()
            self.charging_waits.append(self.env.now - arrival)
            self.maximum_charging_users = max(self.maximum_charging_users, self.charger.count)
            remaining = duration
            while remaining > config.EPSILON:
                if not self.station_operational:
                    yield from self._wait_for_station()
                    continue
                increment = min(0.25, remaining)
                yield self.env.timeout(increment)
                remaining -= increment
            amr.battery = nonlinear_charge(amr.battery, duration)
            self.minimum_battery = min(self.minimum_battery, amr.battery)
            self.charging_actions += 1

    def _wait_for_station(self):
        while not self.station_operational:
            event = self.station_changed
            yield event

    def _move(self, amr: _AMRState, target: Node, planned_route: Iterable[Node]):
        route = tuple(planned_route)
        while amr.position != target:
            if not self._route_is_currently_usable(route, amr.position, target):
                try:
                    route = tuple(nx.shortest_path(self.graph, amr.position, target, weight="time_min"))
                except nx.NetworkXNoPath:
                    event = self.graph_changed
                    yield event
                    route = ()
                    continue

            next_node = route[1]
            edge = _canonical_edge(amr.position, next_node)
            if edge in self.blocked_edges or not self.graph.has_edge(amr.position, next_node):
                route = ()
                continue
            if amr.position in config.OBSTACLE_CELLS or next_node in config.OBSTACLE_CELLS:
                self.obstacle_violations += 1
                raise AssertionError("AMR attempted to enter an obstacle")

            edge_data = self.graph.edges[amr.position, next_node]
            distance = float(edge_data["distance_m"])
            consumption = config.BATTERY_CONSUMPTION_PER_M * distance
            if amr.battery + config.EPSILON < consumption:
                raise RuntimeError("Battery would become negative during movement")
            yield self.env.timeout(float(edge_data["time_min"]))
            amr.position = next_node
            amr.battery = max(0.0, amr.battery - consumption)
            amr.distance += distance
            self.minimum_battery = min(self.minimum_battery, amr.battery)
            route = route[1:]

    def _route_is_currently_usable(self, route: tuple[Node, ...], source: Node, target: Node) -> bool:
        if not route or route[0] != source or route[-1] != target:
            return False
        return all(self.graph.has_edge(u, v) for u, v in zip(route, route[1:]))

    def _shortest_distance(self, source: Node, target: Node) -> float:
        return float(nx.shortest_path_length(self.graph, source, target, weight="distance_m"))

    def _signal_graph_change(self) -> None:
        if not self.graph_changed.triggered:
            self.graph_changed.succeed()
        self.graph_changed = self.env.event()

    def _blockage_process(self):
        sensitive = [
            _canonical_edge(u, v)
            for u, v in config.DISTURBANCE_SENSITIVE_EDGES
            if self.graph.has_edge(u, v)
        ]
        while True:
            yield self.env.timeout(config.BLOCKAGE_CHECK_INTERVAL_MIN)
            if self.rng.random() >= config.BLOCKAGE_PROBABILITY or not sensitive:
                continue
            edge = sensitive[int(self.rng.integers(0, len(sensitive)))]
            if not self.graph.has_edge(*edge):
                continue
            attributes = dict(self.graph.edges[edge])
            self.graph.remove_edge(*edge)
            self.blocked_edges.add(edge)
            self._signal_graph_change()
            duration = float(
                self.rng.integers(config.BLOCKAGE_DURATION_RANGE[0], config.BLOCKAGE_DURATION_RANGE[1] + 1)
            )
            self.env.process(self._restore_blocked_edge(edge, attributes, duration))

    def _restore_blocked_edge(
        self, edge: tuple[Node, Node], attributes: dict[str, object], duration: float
    ):
        yield self.env.timeout(duration)
        self.graph.add_edge(*edge, **attributes)
        self.blocked_edges.discard(edge)
        self._signal_graph_change()

    def _machine_failure_process(self):
        while True:
            yield self.env.timeout(60.0)
            for machine in config.MACHINE_NODES:
                if self.rng.random() < config.MACHINE_FAILURE_PROBABILITY_PER_HOUR:
                    repair = float(
                        self.rng.integers(config.MACHINE_REPAIR_RANGE[0], config.MACHINE_REPAIR_RANGE[1] + 1)
                    )
                    self.machine_down_until[machine] = max(
                        self.machine_down_until[machine], self.env.now + repair
                    )

    def _station_outage_process(self):
        # REFERENCE_IMPLEMENTATION_ASSUMPTION
        # The endogenous horizon is approximated by expected workload divided by six
        # machines; outage start is sampled from 20%-80% of that deterministic estimate.
        total_nominal_work = sum(
            operation.setup_time + operation.processing_time
            for job in self.jobs
            for operation in job.operations
        )
        nominal_horizon = total_nominal_work / len(config.MACHINE_NODES)
        outage_start = float(self.rng.uniform(0.2 * nominal_horizon, 0.8 * nominal_horizon))
        yield self.env.timeout(outage_start)
        self.station_operational = False
        if not self.station_changed.triggered:
            self.station_changed.succeed()
        self.station_changed = self.env.event()
        yield self.env.timeout(config.CHARGING_OUTAGE_DURATION_MIN)
        self.station_operational = True
        if not self.station_changed.triggered:
            self.station_changed.succeed()
        self.station_changed = self.env.event()


# Figure 3, Stage 2: DES event execution and policy evaluation
def execute_policy_in_des(
    policy: CandidatePolicy,
    requests: list[TransportationRequest],
    jobs: list[JobDefinition],
    scenario: str,
    seed: int,
    reference_graph: nx.Graph,
) -> SimulationResult:
    """Execute one complete policy under one scenario and return measurements only."""

    if scenario not in config.SCENARIOS and scenario != "nominal":
        raise ValueError(f"Unknown scenario: {scenario}")
    model = _SimulationModel(policy, jobs, requests, scenario, seed, reference_graph)
    return model.run()
