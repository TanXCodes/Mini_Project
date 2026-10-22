"""Owner: Muhurt. Dijkstra on the road graph.

Edge cost = free-flow travel time + ML delay (delay_fn, added in the ML step).
Signal waiting time is NOT added here, the simulator adds it.
"""
import networkx as nx
import yaml

from src.graph import build_graph


class NoPath(Exception):
    pass


_GRAPH = None


def get_graph(config_path="config.yaml"):
    """Build the graph once and reuse it."""
    global _GRAPH
    if _GRAPH is None:
        with open(config_path) as f:
            p = yaml.safe_load(f)["paths"]
        _GRAPH = build_graph(p["nodes"], p["edges"], p["traffic"])
    return _GRAPH


def compute_route(source, hospital, scenario=None, delay_fn=None, G=None):
    """Fastest route from source to hospital.

    scenario : dict from config.yaml (time_slot, accident), used by delay_fn
    delay_fn : function(segment_id, scenario) -> delay in seconds.
               None = no delay (basic Dijkstra, free-flow only).
    returns  : {"path": [node ids], "segments": [segment ids],
                "segment_times_s": [seconds per segment],
                "total_cost_s": float}
    """
    G = G or get_graph()
    if source == hospital:
        raise ValueError("source and hospital must be different")
    if source not in G or hospital not in G:
        raise ValueError("unknown junction id")

    def seg_time(a, b):
        d = G[a][b]
        extra = delay_fn(d["segment_id"], scenario) if delay_fn else 0.0
        return d["free_flow_s"] + extra

    def weight(a, b, d):
        return seg_time(a, b)

    try:
        path = nx.dijkstra_path(G, source, hospital, weight=weight)
    except nx.NetworkXNoPath:
        raise NoPath(f"no route from {source} to {hospital}")

    pairs = list(zip(path, path[1:]))
    times = [seg_time(a, b) for a, b in pairs]
    return {"path": path,
            "segments": [G[a][b]["segment_id"] for a, b in pairs],
            "segment_times_s": times,
            "total_cost_s": sum(times)}
