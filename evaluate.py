"""Evaluate Normal vs Priority ambulance journeys over multiple seeds."""

import csv
import statistics
from pathlib import Path

from src.routing import compute_route, get_graph
from src.simulator import run_simulation, load_config, load_signal_nodes

N_SEEDS = 30

cfg = load_config()
paths = cfg["paths"]

signal_nodes = load_signal_nodes(
    paths["nodes"],
    paths["edges"],
    cfg["signals"]["extra_simulated_signals"],
)

G = get_graph()
rows = []

for trip in cfg["trips"]:
    route = compute_route(trip["source"], trip["hospital"], G=G)

    normal_t = []
    prior_t = []
    normal_w = []
    priority_overhead = []
    saved = []

    for seed in range(N_SEEDS):
        n = run_simulation(route, "normal", seed, signal_nodes, cfg)
        p = run_simulation(route, "priority", seed, signal_nodes, cfg)

        normal_t.append(n["journey_time_s"])
        prior_t.append(p["journey_time_s"])
        normal_w.append(n["wait_time_s"])
        priority_overhead.append(p["priority_time_s"])

        # Positive = time saved; negative = time lost.
        saved.append(n["journey_time_s"] - p["journey_time_s"])

    avg_normal = statistics.mean(normal_t)
    avg_priority = statistics.mean(prior_t)
    avg_saved = statistics.mean(saved)

    row = {
        "source": trip["source"],
        "hospital": trip["hospital"],
        "avg_normal_s": round(avg_normal, 1),
        "avg_priority_s": round(avg_priority, 1),
        "avg_normal_wait_s": round(statistics.mean(normal_w), 1),
        "avg_priority_overhead_s": round(
            statistics.mean(priority_overhead), 1
        ),
        "avg_saved_s": round(avg_saved, 1),
        "saved_pct": round(100 * avg_saved / avg_normal, 1)
            if avg_normal else 0,
        "min_saved_s": round(min(saved), 1),
        "max_saved_s": round(max(saved), 1),
        "priority_slower_seeds": sum(x < 0 for x in saved),
    }

    rows.append(row)

    print(
        f"{row['source']} -> {row['hospital']} | "
        f"Normal {row['avg_normal_s']} s | "
        f"Priority {row['avg_priority_s']} s | "
        f"Saved {row['avg_saved_s']} s ({row['saved_pct']}%) | "
        f"Slower seeds: {row['priority_slower_seeds']}/{N_SEEDS}"
    )

out = Path("outputs")
out.mkdir(exist_ok=True)

with open(out / "evaluation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(
    f"\nAverage calculated over {N_SEEDS} seeds. "
    "Results: outputs/evaluation_results.csv"
)
