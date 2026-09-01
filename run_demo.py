"""Command-line entry point for quick and paper-scale MO-ACO runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

import config
from moaco import archive_is_mutually_non_dominated, run_moaco
from simulation import build_shop_floor_graph, generate_reference_instance


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="5 ants x 5 iterations demonstration")
    mode.add_argument(
        "--paper-settings", action="store_true", help="30 ants x 100 iterations, Table 2 settings"
    )
    return parser.parse_args()


def _save_outputs(archive, mode_name: str, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    for policy in archive:
        charging_decisions = sum(decision.charging_inserted for decision in policy.decisions)
        assignment_counts = {
            amr_id: sum(decision.assigned_amr == amr_id for decision in policy.decisions)
            for amr_id in range(config.N_AMRS)
        }
        archive_rows.append(
            {
                "solution_id": policy.solution_id,
                "f1_makespan": policy.f1_makespan,
                "f2_travel_distance": policy.f2_travel_distance,
                "policy_decisions": len(policy.decisions),
                "planned_charging_insertions": charging_decisions,
                "amr_assignment_counts": str(assignment_counts),
            }
        )
        for scenario, result in policy.scenario_results.items():
            result_rows.append(
                {
                    "run_id": mode_name,
                    "solution_id": policy.solution_id,
                    "scenario": scenario,
                    "makespan": result.makespan,
                    "travel_distance": result.travel_distance,
                    "charging_actions": result.charging_actions,
                    "average_charging_waiting_time": result.average_charging_waiting_time,
                    "emergency_charging_interruptions": result.emergency_charging_interruptions,
                }
            )
    pd.DataFrame(result_rows).to_csv(output_directory / "results_summary.csv", index=False)
    pd.DataFrame(archive_rows).to_csv(output_directory / "pareto_archive.csv", index=False)


def _validate(archive, audit: dict[str, object], settings: config.RunSettings) -> dict[str, bool]:
    # Validate every ant/scenario evaluation, not only policies surviving in A_ND.
    results = audit["all_simulation_results"]
    checks = {
        "complete_policy_before_DES": bool(audit["complete_before_des"]),
        "every_policy_evaluated_under_every_scenario": all(
            count == len(settings.scenarios) for count in audit["scenarios_per_policy"]
        ),
        "scenario_measures_recorded": all(
            result.makespan > 0.0 and result.travel_distance > 0.0 for result in results
        ),
        "battery_nonnegative": all(result.minimum_battery >= -config.EPSILON for result in results),
        "obstacles_never_traversed": all(result.obstacle_violations == 0 for result in results),
        "blocked_edges_never_entered": all(result.blocked_edge_violations == 0 for result in results),
        "charging_capacity_respected": all(
            result.maximum_charging_users <= config.CHARGING_PORTS for result in results
        ),
        "production_transport_precedence": all(
            result.downstream_precedence_violations == 0 for result in results
        ),
        "job_machine_routes_completed": all(
            result.completed_job_route_violations == 0 for result in results
        ),
        "positive_makespan": bool(results) and all(result.makespan > 0.0 for result in results),
        "positive_travel_distance": bool(results)
        and all(result.travel_distance > 0.0 for result in results),
        "archive_mutually_non_dominated": archive_is_mutually_non_dominated(archive),
        "DES_has_no_learning_state": bool(audit["des_signature_excludes_learning_state"]),
        "archive_updated_only_after_all_ants": all(
            count == settings.ants for count in audit["archive_update_ant_counts"]
        ),
        "pheromones_updated_only_after_all_ants": all(
            count == settings.ants for count in audit["pheromone_update_ant_counts"]
        ),
    }
    return checks


def main() -> int:
    args = _parse_arguments()
    settings = config.QUICK_SETTINGS if args.quick else config.PAPER_SETTINGS
    mode_name = "quick" if args.quick else "paper-settings"
    if args.quick:
        print("DEMONSTRATION OUTPUT")
    else:
        print("PAPER SETTINGS OUTPUT (reference assumptions; not exact numerical reproduction)")

    graph = build_shop_floor_graph()
    jobs, requests = generate_reference_instance()
    audit: dict[str, object] = {}
    archive = run_moaco(requests, jobs, graph, settings, audit)
    checks = _validate(archive, audit, settings)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Validation failed: {', '.join(failed)}")

    output_directory = Path(__file__).resolve().parent / "output"
    _save_outputs(archive, mode_name, output_directory)
    print(f"Jobs: {len(jobs)}; transportation requests: {len(requests)}")
    print(f"Iterations completed: {audit['iterations_completed']}; Pareto archive size: {len(archive)}")
    for policy in archive:
        print(
            f"  {policy.solution_id}: f1={policy.f1_makespan:.3f} min, "
            f"f2={policy.f2_travel_distance:.3f} m"
        )
    print("Validation: all 15 executable checks passed")
    print(f"Outputs: {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
