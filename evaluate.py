"""evaluate.py - har demo trip ko kai seeds par chalao, Normal vs Priority ka average."""
import csv
import statistics
from pathlib import Path

from src.routing import compute_route, get_graph
from src.simulator import run_simulation, load_config, load_signal_nodes

N_SEEDS = 30

cfg = load_config()
paths = cfg["paths"]
signal_nodes = load_signal_nodes(paths["nodes"], paths["edges"],
                                 cfg["signals"]["extra_simulated_signals"])
G = get_graph()

rows = []
for trip in cfg["trips"]:
    route = compute_route(trip["source"], trip["hospital"], G=G)
    normal_t, prior_t, normal_w, saved = [], [], [], []

    for seed in range(N_SEEDS):
        n = run_simulation(route, "normal", seed, signal_nodes, cfg)
        p = run_simulation(route, "priority", seed, signal_nodes, cfg)
        assert p["journey_time_s"] <= n["journey_time_s"] + 1e-9, f"seed {seed}: priority slower"
        normal_t.append(n["journey_time_s"])
        prior_t.append(p["journey_time_s"])
        normal_w.append(n["wait_time_s"])
        saved.append(n["journey_time_s"] - p["journey_time_s"])

    row = {
        "source": trip["source"],
        "hospital": trip["hospital"],
        "avg_normal_s": round(statistics.mean(normal_t), 1),
        "avg_priority_s": round(statistics.mean(prior_t), 1),
        "avg_normal_wait_s": round(statistics.mean(normal_w), 1),
        "avg_saved_s": round(statistics.mean(saved), 1),
        "saved_pct": round(100 * statistics.mean(saved) / statistics.mean(normal_t), 1),
        "min_saved_s": round(min(saved), 1),
        "max_saved_s": round(max(saved), 1),
    }
    rows.append(row)
    print(f"{row['source']} -> {row['hospital']} | "
          f"Normal {row['avg_normal_s']} s | Priority {row['avg_priority_s']} s | "
          f"saved {row['avg_saved_s']} s ({row['saved_pct']}%) "
          f"[min {row['min_saved_s']}, max {row['max_saved_s']}]")

out = Path("outputs")
out.mkdir(exist_ok=True)
with open(out / "evaluation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n{N_SEEDS} seeds par average. Results: outputs/evaluation_results.csv")
