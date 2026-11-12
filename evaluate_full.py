import csv
import statistics
from pathlib import Path
from collections import defaultdict

from src.routing import compute_route, get_graph
from src.simulator import run_simulation, load_config, load_signal_nodes
from src.predict import delay_fn

N_SEEDS = 30
MODES = ["normal", "priority"]

cfg = load_config()
G = get_graph()

paths = cfg["paths"]

signal_nodes = load_signal_nodes(
    paths["nodes"],
    paths["edges"],
    cfg["signals"]["extra_simulated_signals"],
)

runs = []

for trip_index, trip in enumerate(cfg["trips"], start=1):
    for scenario in cfg["scenarios"]:

        route = compute_route(
            trip["source"],
            trip["hospital"],
            scenario=scenario,
            delay_fn=delay_fn,
            G=G,
        )

        print(
            f"\nTrip {trip_index} | Scenario: {scenario['name']} | "
            f"Route cost: {route['total_cost_s']:.2f}s"
        )

        for seed in range(N_SEEDS):
            for mode in MODES:

                result = run_simulation(
                    route,
                    mode,
                    seed,
                    signal_nodes,
                    cfg,
                )

                runs.append({
                    "trip": f"Trip {trip_index}",
                    "source": trip["source"],
                    "hospital": trip["hospital"],
                    "scenario": scenario["name"],
                    "mode": mode,
                    "seed": seed,
                    "route_cost_s": route["total_cost_s"],
                    "journey_time_s": result["journey_time_s"],
                    "wait_time_s": result["wait_time_s"],
                    "priority_time_s": result["priority_time_s"],
                    "red_stops": result["red_stops"],
                })

# Calculate paired time savings using the same trip, scenario and seed.
normal_times = {
    (r["trip"], r["scenario"], r["seed"]): r["journey_time_s"]
    for r in runs
    if r["mode"] == "normal"
}

for r in runs:
    key = (r["trip"], r["scenario"], r["seed"])
    saved = normal_times[key] - r["journey_time_s"]
    r["time_saved_vs_normal_s"] = round(saved, 3)

# Save all individual simulation runs.
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

run_file = output_dir / "full_evaluation_runs.csv"

with open(run_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(runs[0]))
    writer.writeheader()
    writer.writerows(runs)

# Aggregate by trip, scenario and mode.
groups = defaultdict(list)

for r in runs:
    groups[(r["trip"], r["scenario"], r["mode"])].append(r)

summary = []

for (trip_name, scenario_name, mode), records in groups.items():

    savings = [r["time_saved_vs_normal_s"] for r in records]

    summary.append({
        "trip": trip_name,
        "scenario": scenario_name,
        "mode": mode,
        "avg_route_cost_s": round(
            statistics.mean(r["route_cost_s"] for r in records), 3
        ),
        "avg_journey_time_s": round(
            statistics.mean(r["journey_time_s"] for r in records), 3
        ),
        "avg_wait_time_s": round(
            statistics.mean(r["wait_time_s"] for r in records), 3
        ),
        "avg_priority_time_s": round(
            statistics.mean(r["priority_time_s"] for r in records), 3
        ),
        "avg_saved_vs_normal_s": round(
            statistics.mean(savings), 3
        ),
        "saved_pct": round(
    100 * statistics.mean(savings) /
    statistics.mean(
        normal_times[(trip_name, scenario_name, r["seed"])]
        for r in records
    ),
    2
) if mode == "priority" else 0.0,
        "priority_slower_seeds": sum(x < 0 for x in savings)
            if mode == "priority" else 0,
        "seeds_evaluated": len(records),
    })
# Validate expected evaluation counts.
assert len(runs) == 720, f"Expected 720 runs, got {len(runs)}"
assert len(summary) == 24, f"Expected 24 summary rows, got {len(summary)}"

summary_file = output_dir / "full_evaluation_summary.csv"

with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("\n========== FINAL EVALUATION ==========")
print("Total simulation runs:", len(runs))
print("Expected simulation runs:", 720)
print("Summary rows:", len(summary))
print("Expected summary rows:", 24)
print("Runs CSV:", run_file)
print("Summary CSV:", summary_file)

print("\n========== SUMMARY ==========")

for row in summary:
    print(
        row["trip"],
        "|", row["scenario"],
        "|", row["mode"],
        "| Journey:", row["avg_journey_time_s"],
        "| Wait:", row["avg_wait_time_s"],
        "| Priority overhead:", row["avg_priority_time_s"],
        "| Saved:", row["avg_saved_vs_normal_s"],
        "| Slower seeds:", row["priority_slower_seeds"],
    )

print("\nEvaluation complete.")
