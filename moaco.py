"""Transparent MO-ACO implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import networkx as nx
import numpy as np

import config
from simulation import (
    ActionKey,
    CandidatePolicy,
    JobDefinition,
    Node,
    PolicyDecision,
    SimulationResult,
    TransportationRequest,
    candidate_routes,
    execute_policy_in_des,
    nonlinear_charge,
    path_distance,
    path_time,
)


@dataclass
class ConstructionState:
    """Deterministic local state used only to construct a complete policy."""

    unserved_request_ids: set[int]
    amr_locations: list[Node]
    amr_batteries: list[float]
    amr_available_times: list[float]
    charging_station_operational: bool
    charging_port_available_times: list[float]
    graph: nx.Graph
    request_by_id: dict[int, TransportationRequest]
    route_cache: dict[tuple[Node, Node, int], list[tuple[Node, ...]]]
    path_metric_cache: dict[tuple[Node, ...], tuple[float, float, float, float]]
    current_time: float = 0.0


@dataclass(frozen=True)
class FeasibleAction:
    request_id: int
    amr_id: int
    route_index: int
    path_to_pickup: tuple[Node, ...]
    loaded_path: tuple[Node, ...]
    charging_inserted: bool
    charging_duration: float
    path_to_charger: tuple[Node, ...]
    path_from_charger: tuple[Node, ...]
    estimated_distance: float
    estimated_duration: float
    estimated_end_battery: float
    heuristic: float
    action_key: ActionKey


def _get_routes(state: ConstructionState, source: Node, target: Node) -> list[tuple[Node, ...]]:
    key = (source, target, config.K_ROUTE_CANDIDATES)
    if key not in state.route_cache:
        state.route_cache[key] = candidate_routes(
            state.graph, source, target, config.K_ROUTE_CANDIDATES
        )
    return state.route_cache[key]


def _shortest_path(state: ConstructionState, source: Node, target: Node) -> tuple[Node, ...] | None:
    routes = _get_routes(state, source, target)
    return routes[0] if routes else None


def _path_metrics(
    state: ConstructionState, path: tuple[Node, ...]
) -> tuple[float, float, float, float]:
    """Cache distance, time, shared fraction, and route-risk fraction."""

    if path not in state.path_metric_cache:
        edge_count = max(1, len(path) - 1)
        state.path_metric_cache[path] = (
            path_distance(state.graph, path),
            path_time(state.graph, path),
            sum(bool(state.graph.edges[u, v]["shared"]) for u, v in zip(path, path[1:]))
            / edge_count,
            sum(
                bool(state.graph.edges[u, v]["disturbance_sensitive"])
                for u, v in zip(path, path[1:])
            )
            / edge_count,
        )
    return state.path_metric_cache[path]


def _heuristic_desirability(
    state: ConstructionState,
    request: TransportationRequest,
    paths: tuple[tuple[Node, ...], ...],
    estimated_distance: float,
    estimated_end_battery: float,
    charging_inserted: bool,
    charging_duration: float,
    estimated_queue_wait: float,
) -> float:
    """using explicit normalized reference proxies."""

    # REFERENCE_IMPLEMENTATION_ASSUMPTION
    # The paper defines meanings but not formulas. These bounded proxies make all five
    # components inspectable without changing Equation 19's weighted denominator.
    d_hat = float(np.clip(estimated_distance / 40.0, 0.0, 1.0))
    shared_fraction = float(np.mean([_path_metrics(state, path)[2] for path in paths]))
    availability_delay = max(0.0, min(state.amr_available_times) - state.current_time)
    q_hat = float(
        np.clip((shared_fraction + estimated_queue_wait / 30.0 + availability_delay / 60.0) / 3.0, 0, 1)
    )

    waiting_age = max(0.0, state.current_time - request.estimated_release_time)
    age_score = float(np.clip(waiting_age / 120.0, 0.0, 1.0))
    criticality_score = 1.0 / max(1, request.operations_remaining)
    downstream_idle_proxy = float(state.current_time >= request.estimated_release_time)
    urgency_score = float(np.clip(0.4 * age_score + 0.3 * criticality_score + 0.3 * downstream_idle_proxy, 0, 1))
    # Larger production urgency must reduce the cost in Eq. 19.
    u_hat = 1.0 - urgency_score

    low_residual_cost = float(
        np.clip((config.BATTERY_MAX - estimated_end_battery) / config.BATTERY_MAX, 0, 1)
    )
    duration_cost = charging_duration / max(config.CHARGING_DURATIONS) if charging_inserted else 0.0
    insertion_cost = 1.0 if charging_inserted else 0.0
    c_hat = float(np.clip((low_residual_cost + duration_cost + insertion_cost) / 3.0, 0, 1))
    g_hat = float(np.mean([_path_metrics(state, path)[3] for path in paths]))

    weights = config.HEURISTIC_WEIGHTS
    denominator = (
        weights["distance"] * d_hat
        + weights["queue"] * q_hat
        + weights["urgency"] * u_hat
        + weights["charging"] * c_hat
        + weights["risk"] * g_hat
        + config.EPSILON
    )
    return 1.0 / denominator


def initialize_pheromones(requests: list[TransportationRequest]) -> dict[ActionKey, float]:
    """tau_a(0)=tau_0 for every admissible action label."""

    pheromones: dict[ActionKey, float] = {}
    durations = (0,) + tuple(int(duration) for duration in config.CHARGING_DURATIONS)
    for request in requests:
        for amr_id in range(config.N_AMRS):
            for route_index in range(config.K_ROUTE_CANDIDATES):
                for duration in durations:
                    pheromones[(request.request_id, amr_id, route_index, duration)] = config.TAU0
    return pheromones


def initialize_archive() -> list[CandidatePolicy]:

    return []


def initialize_construction_state(
    requests: list[TransportationRequest],
    graph: nx.Graph,
    route_cache: dict[tuple[Node, Node, int], list[tuple[Node, ...]]],
    path_metric_cache: dict[tuple[Node, ...], tuple[float, float, float, float]],
) -> ConstructionState:
    """create local estimated state, not a SimPy run."""

    return ConstructionState(
        unserved_request_ids={request.request_id for request in requests},
        amr_locations=[config.CHARGING_STATION for _ in range(config.N_AMRS)],
        amr_batteries=[config.INITIAL_BATTERY for _ in range(config.N_AMRS)],
        amr_available_times=[0.0 for _ in range(config.N_AMRS)],
        charging_station_operational=True,
        charging_port_available_times=[0.0 for _ in range(config.CHARGING_PORTS)],
        graph=graph,
        request_by_id={request.request_id: request for request in requests},
        route_cache=route_cache,
        path_metric_cache=path_metric_cache,
    )


def identify_feasible_actions(state: ConstructionState) -> list[FeasibleAction]:

    actions: list[FeasibleAction] = []
    charger = config.CHARGING_STATION
    eta = config.BATTERY_CONSUMPTION_PER_M

    for request_id in sorted(state.unserved_request_ids):
        request = state.request_by_id[request_id]
        loaded_routes = _get_routes(state, request.origin, request.destination)
        destination_to_charger = _shortest_path(state, request.destination, charger)
        if not loaded_routes or destination_to_charger is None:
            continue
        reserve_distance = _path_metrics(state, destination_to_charger)[0]

        for amr_id in range(config.N_AMRS):
            location = state.amr_locations[amr_id]
            battery = state.amr_batteries[amr_id]
            to_pickup = _shortest_path(state, location, request.origin)
            to_charger = _shortest_path(state, location, charger)
            charger_to_pickup = _shortest_path(state, charger, request.origin)
            if to_pickup is None:
                continue

            for route_index, loaded_path in enumerate(loaded_routes):
                empty_distance = _path_metrics(state, to_pickup)[0]
                loaded_distance = _path_metrics(state, loaded_path)[0]
                direct_distance = empty_distance + loaded_distance
                direct_required = eta * (direct_distance + reserve_distance) + config.BATTERY_SAFE
                direct_end_battery = battery - eta * direct_distance

                if battery + config.EPSILON >= direct_required:
                    paths = (to_pickup, loaded_path)
                    duration = _path_metrics(state, to_pickup)[1] + _path_metrics(state, loaded_path)[1]
                    heuristic = _heuristic_desirability(
                        state,
                        request,
                        paths,
                        direct_distance,
                        direct_end_battery,
                        False,
                        0.0,
                        0.0,
                    )
                    key = (request_id, amr_id, route_index, 0)
                    actions.append(
                        FeasibleAction(
                            request_id=request_id,
                            amr_id=amr_id,
                            route_index=route_index,
                            path_to_pickup=to_pickup,
                            loaded_path=loaded_path,
                            charging_inserted=False,
                            charging_duration=0.0,
                            path_to_charger=(),
                            path_from_charger=(),
                            estimated_distance=direct_distance,
                            estimated_duration=duration,
                            estimated_end_battery=direct_end_battery,
                            heuristic=heuristic,
                            action_key=key,
                        )
                    )

                # REFERENCE_IMPLEMENTATION_ASSUMPTION
                # Charging alternatives are enumerated when direct service is infeasible
                # or state of charge is below 55%. This retains proactive charging while
                # avoiding redundant high-SoC duration variants in the reference search.
                charging_needed_or_prudent = (
                    battery + config.EPSILON < direct_required
                    or battery < 0.55 * config.BATTERY_MAX
                )
                if (
                    charging_needed_or_prudent
                    and
                    state.charging_station_operational
                    and to_charger is not None
                    and charger_to_pickup is not None
                ):
                    to_charger_distance = _path_metrics(state, to_charger)[0]
                    if battery + config.EPSILON < eta * to_charger_distance + config.BATTERY_SAFE:
                        continue
                    arrival_battery = battery - eta * to_charger_distance
                    service_distance = _path_metrics(state, charger_to_pickup)[0] + loaded_distance
                    port_time = min(state.charging_port_available_times)
                    arrival_time = max(
                        state.current_time, state.amr_available_times[amr_id]
                    ) + _path_metrics(state, to_charger)[1]
                    queue_wait = max(0.0, port_time - arrival_time)
                    for charging_duration in config.CHARGING_DURATIONS:
                        charged_battery = nonlinear_charge(arrival_battery, charging_duration)
                        required_after_charge = eta * (service_distance + reserve_distance) + config.BATTERY_SAFE
                        if charged_battery + config.EPSILON < required_after_charge:
                            continue
                        end_battery = charged_battery - eta * service_distance
                        total_distance = to_charger_distance + service_distance
                        paths = (to_charger, charger_to_pickup, loaded_path)
                        duration = (
                            _path_metrics(state, to_charger)[1]
                            + queue_wait
                            + charging_duration
                            + _path_metrics(state, charger_to_pickup)[1]
                            + _path_metrics(state, loaded_path)[1]
                        )
                        heuristic = _heuristic_desirability(
                            state,
                            request,
                            paths,
                            total_distance,
                            end_battery,
                            True,
                            charging_duration,
                            queue_wait,
                        )
                        key = (request_id, amr_id, route_index, int(charging_duration))
                        actions.append(
                            FeasibleAction(
                                request_id=request_id,
                                amr_id=amr_id,
                                route_index=route_index,
                                path_to_pickup=charger_to_pickup,
                                loaded_path=loaded_path,
                                charging_inserted=True,
                                charging_duration=charging_duration,
                                path_to_charger=to_charger,
                                path_from_charger=charger_to_pickup,
                                estimated_distance=total_distance,
                                estimated_duration=duration,
                                estimated_end_battery=end_battery,
                                heuristic=heuristic,
                                action_key=key,
                            )
                        )
    return actions


def compute_action_probabilities(
    feasible_actions: list[FeasibleAction],
    pheromones: dict[ActionKey, float],
) -> np.ndarray:

    if not feasible_actions:
        raise ValueError("Cannot compute probabilities for an empty feasible action set")
    log_values = np.array(
        [
            config.ALPHA * math.log(max(pheromones[action.action_key], config.EPSILON))
            + config.BETA * math.log(max(action.heuristic, config.EPSILON))
            for action in feasible_actions
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(log_values)):
        warnings.warn("Invalid action desirabilities; using uniform probabilities", RuntimeWarning)
        return np.full(len(feasible_actions), 1.0 / len(feasible_actions))
    desirabilities = np.exp(log_values - np.max(log_values))
    total = float(np.sum(desirabilities))
    if not math.isfinite(total) or total <= 0.0:
        warnings.warn("Zero action desirability sum; using uniform probabilities", RuntimeWarning)
        return np.full(len(feasible_actions), 1.0 / len(feasible_actions))
    return desirabilities / total


def select_action(
    feasible_actions: list[FeasibleAction], probabilities: np.ndarray, rng: np.random.Generator
) -> FeasibleAction:

    return feasible_actions[int(rng.choice(len(feasible_actions), p=probabilities))]


def update_candidate_policy(policy: CandidatePolicy, action: FeasibleAction) -> None:
    """add assignment, sequence, route, and charging data."""

    policy.decisions.append(
        PolicyDecision(
            request_id=action.request_id,
            assigned_amr=action.amr_id,
            service_priority=len(policy.decisions),
            route_index=action.route_index,
            route_to_pickup=action.path_to_pickup,
            loaded_route=action.loaded_path,
            charging_inserted=action.charging_inserted,
            charging_duration=action.charging_duration,
            route_to_charger=action.path_to_charger,
            route_from_charger=action.path_from_charger,
            action_key=action.action_key,
        )
    )


def update_local_construction_state(state: ConstructionState, action: FeasibleAction) -> None:
    """deterministic estimate only, never full DES evaluation."""

    amr_id = action.amr_id
    start_time = max(state.current_time, state.amr_available_times[amr_id])
    if action.charging_inserted:
        travel_to_charger = _path_metrics(state, action.path_to_charger)[1]
        arrival = start_time + travel_to_charger
        port_index = int(np.argmin(state.charging_port_available_times))
        charge_start = max(arrival, state.charging_port_available_times[port_index])
        state.charging_port_available_times[port_index] = charge_start + action.charging_duration
    state.amr_available_times[amr_id] = start_time + action.estimated_duration
    state.amr_locations[amr_id] = state.request_by_id[action.request_id].destination
    state.amr_batteries[amr_id] = action.estimated_end_battery
    state.unserved_request_ids.remove(action.request_id)
    state.current_time = min(state.amr_available_times)


def compute_objectives(
    scenario_results: dict[str, SimulationResult], scenario_weights: dict[str, float]
) -> tuple[float, float]:
    """scenario-weighted makespan and physical distance."""

    if set(scenario_results) != set(scenario_weights):
        raise ValueError("Objectives require one result for every weighted scenario")
    f1 = sum(scenario_weights[name] * scenario_results[name].makespan for name in scenario_weights)
    f2 = sum(
        scenario_weights[name] * scenario_results[name].travel_distance for name in scenario_weights
    )
    return float(f1), float(f2)


def _dominates(left: CandidatePolicy, right: CandidatePolicy) -> bool:
    assert left.f1_makespan is not None and left.f2_travel_distance is not None
    assert right.f1_makespan is not None and right.f2_travel_distance is not None
    return (
        left.f1_makespan <= right.f1_makespan
        and left.f2_travel_distance <= right.f2_travel_distance
        and (
            left.f1_makespan < right.f1_makespan
            or left.f2_travel_distance < right.f2_travel_distance
        )
    )


def update_non_dominated_archive(
    archive: list[CandidatePolicy], candidates: list[CandidatePolicy]
) -> list[CandidatePolicy]:

    combined = archive + candidates
    unique: list[CandidatePolicy] = []
    seen_pairs: set[tuple[float, float]] = set()
    for policy in combined:
        assert policy.f1_makespan is not None and policy.f2_travel_distance is not None
        pair = (round(policy.f1_makespan, 9), round(policy.f2_travel_distance, 9))
        if pair not in seen_pairs:
            unique.append(policy)
            seen_pairs.add(pair)
    return [policy for policy in unique if not any(_dominates(other, policy) for other in unique if other is not policy)]


def _assign_crowding_distances(archive: list[CandidatePolicy]) -> None:
    """Conventional two-objective crowding distance with preserved extremes."""

    for policy in archive:
        policy.crowding_distance = 0.0
    if len(archive) <= 2:
        for policy in archive:
            policy.crowding_distance = math.inf
        return
    for attribute in ("f1_makespan", "f2_travel_distance"):
        ordered = sorted(archive, key=lambda policy: float(getattr(policy, attribute)))
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        minimum = float(getattr(ordered[0], attribute))
        maximum = float(getattr(ordered[-1], attribute))
        span = maximum - minimum
        if span <= config.EPSILON:
            continue
        for index in range(1, len(ordered) - 1):
            if math.isinf(ordered[index].crowding_distance):
                continue
            previous_value = float(getattr(ordered[index - 1], attribute))
            next_value = float(getattr(ordered[index + 1], attribute))
            ordered[index].crowding_distance += (next_value - previous_value) / span


def apply_diversity_control(
    archive: list[CandidatePolicy], maximum_size: int
) -> list[CandidatePolicy]:
    """retain high-crowding solutions and both extremes."""

    _assign_crowding_distances(archive)
    if len(archive) <= maximum_size:
        return archive
    return sorted(archive, key=lambda policy: policy.crowding_distance, reverse=True)[:maximum_size]


def update_pheromones(
    pheromones: dict[ActionKey, float], archive: list[CandidatePolicy]
) -> dict[ActionKey, float]:
    """evaporation followed by archive-based reinforcement."""

    updated = {key: (1.0 - config.RHO) * value for key, value in pheromones.items()}
    if not archive:
        return updated
    _assign_crowding_distances(archive)
    f1_values = np.array([float(policy.f1_makespan) for policy in archive])
    f2_values = np.array([float(policy.f2_travel_distance) for policy in archive])

    def normalized(value: float, values: np.ndarray) -> float:
        span = float(np.max(values) - np.min(values))
        return 0.5 if span <= config.EPSILON else (value - float(np.min(values))) / span

    finite_crowding = [
        policy.crowding_distance for policy in archive if math.isfinite(policy.crowding_distance)
    ]
    crowding_scale = max(finite_crowding, default=1.0)
    for policy in archive:
        # REFERENCE_IMPLEMENTATION_ASSUMPTION
        # f_bar is the mean min-max-normalized cost. psi ranges from 1 to 2,
        # assigning larger reinforcement to isolated/crowding-boundary solutions.
        aggregate_quality_cost = 0.5 * (
            normalized(float(policy.f1_makespan), f1_values)
            + normalized(float(policy.f2_travel_distance), f2_values)
        )
        diversity_factor = (
            2.0
            if math.isinf(policy.crowding_distance)
            else 1.0 + policy.crowding_distance / max(crowding_scale, config.EPSILON)
        )
        reinforcement = (
            config.PHEROMONE_Q
            * diversity_factor
            / (config.EPSILON + aggregate_quality_cost)
        )
        for key in policy.action_keys():
            updated[key] += reinforcement
    return updated


def archive_is_mutually_non_dominated(archive: list[CandidatePolicy]) -> bool:
    return all(
        not _dominates(left, right)
        for index, left in enumerate(archive)
        for other_index, right in enumerate(archive)
        if index != other_index
    )


# Stage 1: Offline MO ACO solution construction
# Stage 3: Non dominated archive and pheromone update
def run_moaco(
    requests: list[TransportationRequest],
    jobs: list[JobDefinition],
    graph: nx.Graph,
    settings: config.RunSettings,
    audit: dict[str, object] | None = None,
) -> list[CandidatePolicy]:
    """Run Algorithm visibly in its published order and return A_ND."""

    audit = audit if audit is not None else {}
    rng = np.random.default_rng(settings.base_seed)
    route_cache: dict[tuple[Node, Node, int], list[tuple[Node, ...]]] = {}
    path_metric_cache: dict[tuple[Node, ...], tuple[float, float, float, float]] = {}

    # Initialize pheromone values
    pheromones = initialize_pheromones(requests)

    # Initialize external non dominated archive
    archive = initialize_archive()

    # Initialize iteration counter
    t = 0
    no_improvement_iterations = 0
    archive_update_ant_counts: list[int] = []
    pheromone_update_ant_counts: list[int] = []
    complete_before_des = True
    scenarios_per_policy: list[int] = []
    all_simulation_results: list[SimulationResult] = []

    # Main MO ACO iteration loop
    while t < settings.iterations and no_improvement_iterations < settings.no_improvement_limit:
        iteration_candidates: list[CandidatePolicy] = []

        # Ant loop
        for ant_index in range(settings.ants):
            # Initialize construction state and candidate policy
            state = initialize_construction_state(requests, graph, route_cache, path_metric_cache)
            policy = CandidatePolicy(solution_id=f"t{t:03d}_k{ant_index:03d}")

            # Candidate policy construction
            while state.unserved_request_ids:
                # Observe state and identify feasible actions
                feasible_actions = identify_feasible_actions(state)
                if not feasible_actions:
                    raise RuntimeError("No feasible action exists in local construction state")

                # Compute probabilities and select action
                probabilities = compute_action_probabilities(feasible_actions, pheromones)
                selected_action = select_action(feasible_actions, probabilities, rng)

                # Update policy and local construction state
                update_candidate_policy(policy, selected_action)
                update_local_construction_state(state, selected_action)

            complete_before_des &= len(policy.decisions) == len(requests)

            # Scenario based DES evaluation
            for scenario_index, scenario in enumerate(settings.scenarios):
                # Common random numbers: seed depends on scenario, never policy order.
                scenario_seed = settings.base_seed + 10_000 * (scenario_index + 1)
                scenario_result = execute_policy_in_des(
                    policy=policy,
                    requests=requests,
                    jobs=jobs,
                    scenario=scenario,
                    seed=scenario_seed,
                    reference_graph=graph,
                )
                policy.scenario_results[scenario] = scenario_result
                all_simulation_results.append(scenario_result)
            scenarios_per_policy.append(len(policy.scenario_results))

            # Scenario weighted objective calculation
            weights = {scenario: config.SCENARIO_WEIGHTS[scenario] for scenario in settings.scenarios}
            policy.f1_makespan, policy.f2_travel_distance = compute_objectives(
                policy.scenario_results, weights
            )
            iteration_candidates.append(policy)

        previous_pairs = {
            (round(float(policy.f1_makespan), 9), round(float(policy.f2_travel_distance), 9))
            for policy in archive
        }

        # Non dominated archive update
        archive_update_ant_counts.append(len(iteration_candidates))
        archive = update_non_dominated_archive(archive, iteration_candidates)

        # Diversity control
        archive = apply_diversity_control(archive, settings.archive_max_size)

        # Pheromone update
        pheromone_update_ant_counts.append(len(iteration_candidates))
        pheromones = update_pheromones(pheromones, archive)

        current_pairs = {
            (round(float(policy.f1_makespan), 9), round(float(policy.f2_travel_distance), 9))
            for policy in archive
        }
        # REFERENCE_IMPLEMENTATION_ASSUMPTION
        # Archive improvement means that at least one new objective pair enters A_ND.
        if current_pairs - previous_pairs:
            no_improvement_iterations = 0
        else:
            no_improvement_iterations += 1

        # Increment iteration
        t += 1

    audit.update(
        {
            "iterations_completed": t,
            "complete_before_des": complete_before_des,
            "scenarios_per_policy": scenarios_per_policy,
            "archive_update_ant_counts": archive_update_ant_counts,
            "pheromone_update_ant_counts": pheromone_update_ant_counts,
            "archive_mutually_non_dominated": archive_is_mutually_non_dominated(archive),
            "des_signature_excludes_learning_state": True,
            "all_simulation_results": all_simulation_results,
        }
    )

    # Return approximated Pareto set
    return archive
