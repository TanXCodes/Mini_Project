"""Owner: Tanay. Move an ambulance along a route and log signal events.

Signal waiting is added here, not by routing.
Priority transition, crossing and recovery durations are also
included in priority journey time.
"""

import random

import pandas as pd
import yaml

from src.signals import SignalController


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_signal_nodes(nodes_csv, edges_csv, extra=15):
    """Return real signal junctions and busiest simulated junctions."""
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    real = set(nodes.loc[nodes["has_signal"], "junction_id"])
    degree = pd.concat(
        [edges["from_node"], edges["to_node"]]
    ).value_counts()

    simulated = set(degree.index[:extra])
    return real | simulated


def run_simulation(route, mode, seed, signal_nodes=None, config=None):
    """Run one ambulance trip.

    mode:
        normal   -> signals follow their normal cycle
        priority -> emergency priority phases are simulated

    Returns:
        journey_time_s, wait_time_s, priority_time_s,
        red_stops, events
    """
    full_config = config or load_config()
    cfg = full_config["signals"]

    if signal_nodes is None:
        paths = full_config["paths"]
        signal_nodes = load_signal_nodes(
            paths["nodes"],
            paths["edges"],
            cfg["extra_simulated_signals"],
        )

    path = route["path"]
    seg_times = route["segment_times_s"]

    rng = random.Random(seed)
    priority = mode == "priority"

    signals = {}
    t = 0.0
    total_wait = 0.0
    total_priority_time = 0.0
    red_stops = 0
    events = []

    for i, node in enumerate(path):

        # Travel along the previous road segment.
        if i > 0:
            t += seg_times[i - 1]

        # Start and destination are not treated as junctions.
        if 0 < i < len(path) - 1 and node in signal_nodes:

            if node not in signals:
                offset = rng.uniform(0, cfg["cycle_s"])

                signals[node] = SignalController(
                    junction_id=node,
                    cycle_s=cfg["cycle_s"],
                    green_s=cfg["green_s"],
                    priority_timeout_s=cfg["priority_timeout_s"],
                    offset_s=offset,
                    transition_s=cfg.get("transition_s", 5),
                    crossing_s=cfg.get("crossing_s", 4),
                    recovery_s=cfg.get("recovery_s", 2),
                )

            sig = signals[node]
            arrival_time = t

            wait = sig.wait_time(arrival_time, priority)

            if wait > 0:
                red_stops += 1

            # Record normal-mode event.
            if not priority:
                events.append({
                    "t": round(arrival_time, 1),
                    "junction": node,
                    "state": "NORMAL",
                    "wait_s": round(wait, 1),
                    "duration_s": 0.0,
                })

            # Record every priority phase separately.
            else:
                for event in sig.last_events:
                    events.append({
                        "t": round(event["time"], 1),
                        "junction": node,
                        "state": event["state"],
                        "wait_s": 0.0,
                        "duration_s": round(event["duration_s"], 1),
                    })

                priority_time = sig.priority_duration_s
                total_priority_time += priority_time
                t += priority_time

            t += wait
            total_wait += wait

    return {
        "journey_time_s": round(t, 1),
        "wait_time_s": round(total_wait, 1),
        "priority_time_s": round(total_priority_time, 1),
        "red_stops": red_stops,
        "events": events,
    }
