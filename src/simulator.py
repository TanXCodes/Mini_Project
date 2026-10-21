"""Owner: Tanay. Move an ambulance along a route and log what happens at signals.

Input route (from compute_route, Muhurt's module):
    {"path": [node ids], "segment_times_s": [seconds for each road segment], ...}
    segment_times_s = travel time + ML predicted delay for that segment
    (signal waiting time is added HERE, not by routing).
"""
import random

import pandas as pd
import yaml

from src.signals import SignalController


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_signal_nodes(nodes_csv, edges_csv, extra=15):
    """Junctions that have a signal.

    Only a few junctions have has_signal=True in the data, so we also mark the
    `extra` busiest junctions (most connecting roads) as simulated signals.
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    real = set(nodes.loc[nodes["has_signal"], "junction_id"])
    degree = pd.concat([edges["from_node"], edges["to_node"]]).value_counts()
    simulated = set(degree.index[:extra])
    return real | simulated


def run_simulation(route, mode, seed, signal_nodes=None, config=None):
    """Run one ambulance trip.

    mode: "normal" (signals follow cycle) or "priority" (green for ambulance)
    returns dict: journey_time_s, wait_time_s, red_stops, events
    """
    cfg = (config or load_config())["signals"]
    if signal_nodes is None:
        paths = (config or load_config())["paths"]
        signal_nodes = load_signal_nodes(paths["nodes"], paths["edges"],
                                         cfg["extra_simulated_signals"])

    path = route["path"]
    seg_times = route["segment_times_s"]
    rng = random.Random(seed)       # same seed => same signal offsets in both modes
    priority = (mode == "priority")

    signals = {}                    # created lazily, one controller per junction
    t = 0.0
    total_wait = 0.0
    red_stops = 0
    events = []

    for i, node in enumerate(path):
        if i > 0:
            t += seg_times[i - 1]   # drive the previous road segment
        # skip start and destination, they are not crossed as junctions
        if 0 < i < len(path) - 1 and node in signal_nodes:
            if node not in signals:
                offset = rng.uniform(0, cfg["cycle_s"])
                signals[node] = SignalController(
                    node, cfg["cycle_s"], cfg["green_s"],
                    cfg["priority_timeout_s"], offset)
            sig = signals[node]
            wait = sig.wait_time(t, priority)
            if wait > 0:
                red_stops += 1
            events.append({"t": round(t, 1), "junction": node,
                           "state": "PRIORITY_GREEN" if priority else "NORMAL",
                           "wait_s": round(wait, 1)})
            t += wait
            total_wait += wait

    return {"journey_time_s": round(t, 1), "wait_time_s": round(total_wait, 1),
            "red_stops": red_stops, "events": events}
