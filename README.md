# Simulation Driven MO ACO for Charging Aware AMR Transportation

## Purpose

This repository provides a reference implementation of the MO ACO and discrete event simulation framework described in the associated manuscript. The implementation is intended to improve computational transparency and demonstrate the principal simulation optimization procedure.

The implementation preserves the three stage structure shown in Figure 3 and the execution order specified in Algorithm 1. It implements only MO ACO plus SimPy-based DES. It does not implement or evaluate benchmark algorithms, and it is not intended to reproduce every numerical table or figure in the manuscript.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Synthetic reference layout, physical units, experiment parameters, disturbance settings, and deterministic seeds. |
| `simulation.py` | Shop-floor graph, reference production instance, policy data structures, and Figure 3 Stage 2 SimPy execution. |
| `moaco.py` | Figure 3 Stages 1 and 3, including Algorithm 1 construction, objectives, archive, diversity, and pheromone updates. |
| `run_demo.py` | Command-line modes, output generation, and executable validation checks. |
| `requirements.txt` | Minimal third-party Python dependencies. |

The `output/` directory is generated automatically.

## Installation

Use Python 3 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No proprietary solver or GPU is required.

## Quick mode

```bash
python run_demo.py --quick
```

Quick mode uses five ants and five iterations, while retaining all five weighted scenarios and exactly the same Algorithm 1 order. Its console results are labeled `DEMONSTRATION OUTPUT` and are only a correctness check.

## Paper settings mode

```bash
python run_demo.py --paper-settings
```

This mode uses 30 ants, at most 100 iterations, archive size 40, scenarios S1-S5 with equal weights, and the parameters in Section 5.1/Table 2. It is substantially more computationally expensive. It does not imply exact reproduction of the reported results.

## Main shop floor and simulation parameters

| Parameter | Value |
|---|---:|
| Grid | 40 x 28 cells |
| Cell size | 0.5 m |
| Resources | 6 machines, 4 buffers, 1 charging station |
| Ordinary/shared-zone speed | 1.0 / 0.6 m/s |
| Jobs | 24, each with 3-5 operations |
| Setup / processing times | discrete uniform 6-15 / 18-45 min |
| AMRs | 3 |
| Battery | `Bmax=100`, initial 80, `eta=1.0` unit/m, `Bsafe=10` |
| Charging | nonlinear `lambda=0.08`; durations 5, 10, 15, 20, 30 min |
| Charging capacity | 2 ports, enforced by `simpy.Resource(capacity=2)` |
| Scenarios | S1 blockage, S2 downtime, S3 charging outage, S4 time inflation, S5 combined |

Distance is always physical meters. Edge travel time is `distance_m / speed_m_per_s / 60.0` minutes. Weighted routing time and physical objective distance remain separate.

## MO ACO parameters

| Parameter | Value |
|---|---:|
| Ants / maximum iterations | 30 / 100 |
| `alpha`, `beta`, `rho` | 1.0, 2.0, 0.1 |
| Archive limit | 40 |
| Heuristic weights `(wd,wq,wu,wc,wg)` | `(0.30,0.15,0.25,0.20,0.10)` |
| No-improvement limit | 20 iterations |
| Weighted scenarios | S1-S5, each 0.2 |

The two objectives are scenario-weighted production makespan and actual total AMR travel distance. The algorithm returns the external non-dominated archive, not one scalarized solution.

## Output files

- `output/results_summary.csv`: run, policy, scenario, makespan, physical distance, charging actions, charging waiting, and emergency interventions.
- `output/pareto_archive.csv`: final non-dominated objectives and concise policy summaries.

## Implementation assumptions

Several low level implementation details, including exact grid coordinates, job routes, heuristic component definitions, some pheromone reinforcement details, and the original random seeds, are not fully specified in the manuscript. The reference implementation therefore uses transparent deterministic assumptions for these elements. These assumptions are documented in the source code and in this README.

Every corresponding code location is marked `REFERENCE_IMPLEMENTATION_ASSUMPTION`:

1. **Coordinates and obstacles.** `config.py` defines a deterministic synthetic layout qualitatively consistent with Figure 2. It is not the original coordinate-level experimental layout.
2. **Job routes and operation samples.** A fixed NumPy `Generator` seed creates 24 routes of three to five non-repeating machines plus fixed setup/processing samples.
3. **Policy completeness.** All inter-operation request templates are known offline. A complete policy orders and assigns them before DES; their decisions remain inactive until production releases each request.
4. **Alternative paths.** Up to three weighted shortest simple paths are internal candidate routes. This is route generation/feasibility support, not a benchmark method.
5. **Construction-time charging availability.** An operational, reachable station makes charging actions admissible when direct service is infeasible or estimated state of charge is below 55%; this keeps proactive charging available without enumerating redundant high-SoC variants. Estimated port times contribute to the heuristic. Actual two-port queueing and outages are enforced only in DES.
6. **Heuristic proxies.** Normalized distance, shared-zone/queue exposure, urgency cost, charging cost, and sensitive-corridor fraction implement the five Equation 19 meanings. Urgency combines estimated waiting age, operations remaining, and a downstream-idle proxy, then uses `1 - urgency_score` as its denominator cost.
7. **Unspecified ACO constants.** `tau0=1`, `Q=1`, and `epsilon=1e-9` are conventional reference values.
8. **Pheromone reinforcement.** `f_bar` is the mean of min-max-normalized objective costs; the crowding-based diversity factor ranges from 1 to 2, with 2 for boundary solutions.
9. **Archive improvement.** Improvement means at least one new objective pair enters the bounded non-dominated archive.
10. **Machine downtime.** Failures use preempt-and-resume processing checked in one-minute event slices; completed work is not discarded.
11. **Charging outage horizon.** A deterministic nominal horizon is total nominal machine workload divided by six; the single 12-minute outage begins uniformly within 20%-80% of that estimate.
12. **Emergency intervention.** If the next planned task fails dynamic task-level battery feasibility, an unplanned safety charge is counted. It never updates the archive or pheromones.
13. **Dynamic rerouting.** Planned routes are followed when currently usable; a feasibility-only weighted reroute occurs at node boundaries after an edge blockage or construction/execution state mismatch.
14. **Random seeds.** Seeds 1001-1030 are reference implementation seeds, not manuscript seeds. Policy comparisons use a scenario-specific common seed independent of evaluation order.

## Reproducibility limitations

The manuscript does not fully disclose the original coordinate layout, job routes, low-level heuristic normalization, all reinforcement details, or the original 30 seeds. Battery parameters are normalized and are not a calibrated electrochemical model. Shared zones use deterministic speed penalties; time-dependent AMR collision reservations and deadlock handling are intentionally outside the manuscript model. Accordingly, numerical equality with manuscript tables is neither claimed nor targeted.

## Correspondence with Figure 3 and Algorithm 1

| Manuscript element | Algorithm 1 line | Figure 3 component | Python function | Python file | Brief description |
|---|---:|---|---|---|---|
| Pheromone initialization | 1 | MO ACO input/state | `initialize_pheromones` | `moaco.py` | Sets every admissible action label to `tau0`. |
| Archive initialization | 2 | Non-dominated archive | `initialize_archive` | `moaco.py` | Creates empty `A_ND`. |
| Iteration counter | 3 | Iterative feedback | `run_moaco` | `moaco.py` | Sets `t=0`. |
| Outer loop | 4-26 | Offline optimization loop | `run_moaco` | `moaco.py` | Applies stopping condition without reordering stages. |
| Ant loop | 5-21 | Stage 1 then Stage 2 | `run_moaco` | `moaco.py` | Constructs and evaluates all K policies. |
| Local initialization | 6-7 | Stage 1: state observation | `initialize_construction_state` | `moaco.py` | Creates estimated local state and empty `X^k`. |
| Complete construction | 8-15 | Stage 1: solution construction | `run_moaco` | `moaco.py` | Ends only after every request template has a decision. |
| State and feasible set | 9-10 | State observation / feasible action set | `identify_feasible_actions` | `moaco.py` | Applies graph, battery, charging, and task feasibility. |
| Probability and choice | 11-12 | Probabilistic selection | `compute_action_probabilities`, `select_action` | `moaco.py` | Implements Equation 18 safely. |
| Policy/local update | 13-14 | Candidate policy output | `update_candidate_policy`, `update_local_construction_state` | `moaco.py` | Records all Equation 16 decisions; no DES occurs here. |
| Scenario evaluation | 16-19 | Stage 2: DES execution | `execute_policy_in_des` | `simulation.py` | Executes a complete policy under each scenario and returns measures only. |
| Weighted objectives | 20 | DES objective output | `compute_objectives` | `moaco.py` | Implements Equations 24 and 25. |
| Archive update | 22 | Stage 3: Pareto archive | `update_non_dominated_archive` | `moaco.py` | Applies Equation 26 after all K ants. |
| Diversity control | 23 | Stage 3: diversity control | `apply_diversity_control` | `moaco.py` | Uses conventional two-objective crowding distance. |
| Pheromone update | 24 | Stage 3: pheromone update | `update_pheromones` | `moaco.py` | Applies Equations 27 and 28 after archive/diversity. |
| Increment | 25 | Iterative feedback | `run_moaco` | `moaco.py` | Increments only after Stage 3. |
| Pareto result | 27 | Approximated Pareto set | `run_moaco` | `moaco.py` | Returns external archive `A_ND`. |

The dependency direction is strictly: MO ACO constructs complete `X^k` -> DES executes it -> DES returns performance results -> MO ACO computes objectives and updates `A_ND` and pheromones. The DES module receives no pheromone or archive object and performs no learning.
