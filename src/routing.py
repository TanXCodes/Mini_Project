
"""Owner: Muhurt. Dijkstra on the road graph.

Edge cost = free-flow travel time + ML delay (delay_fn).
Signal waiting time is NOT added here; the simulator adds it.
"""

import networkx as nx
import math
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

        _GRAPH = build_graph(
            p["nodes"], p["edges"], p["traffic"]
        )

    return _GRAPH


def compute_route(source, hospital, scenario=None, delay_fn=None, G=None):
    """Return the minimum predicted road-cost route."""

    if G is None:
        G = get_graph()

    if source == hospital:
        raise ValueError("source and hospital must be different")

    if source not in G or hospital not in G:
        raise ValueError("unknown junction id")

    def seg_time(a, b):
        d = G[a][b]
        free_flow = float(d["free_flow_s"])

        extra = float(
            delay_fn(d["segment_id"], scenario)
            if delay_fn else 0.0
        )

        if not math.isfinite(free_flow) or free_flow < 0:
            raise ValueError(
                "free-flow time must be finite and non-negative"
            )

        if not math.isfinite(extra) or extra < 0:
            raise ValueError(
                "ML delay must be finite and non-negative"
            )

        return free_flow + extra

    def weight(a, b, d):
        return seg_time(a, b)

    try:
        path = nx.dijkstra_path(
            G, source, hospital, weight=weight
        )
    except nx.NetworkXNoPath:
        raise NoPath(f"no route from {source} to {hospital}")

    pairs = list(zip(path, path[1:]))
    times = [seg_time(a, b) for a, b in pairs]

    return {
        "path": path,
        "segments": [
            G[a][b]["segment_id"] for a, b in pairs
        ],
        "segment_times_s": times,
        "total_cost_s": sum(times)
    }


def compute_alternative_routes(
    source, hospital, scenario=None, delay_fn=None,
    G=None, max_routes=3
):
    """Return up to max_routes candidate paths ranked by road cost."""

    if G is None:
        G = get_graph()

    if source == hospital:
        raise ValueError("source and hospital must be different")

    if source not in G or hospital not in G:
        raise ValueError("unknown junction id")

    if (
        isinstance(max_routes, bool)
        or not isinstance(max_routes, int)
        or max_routes < 1
    ):
        raise ValueError("max_routes must be a positive integer")

    def seg_time(a, b):
        d = G[a][b]
        free_flow = float(d["free_flow_s"])

        extra = float(
            delay_fn(d["segment_id"], scenario)
            if delay_fn else 0.0
        )

        if not math.isfinite(free_flow) or free_flow < 0:
            raise ValueError(
                "free-flow time must be finite and non-negative"
            )

        if not math.isfinite(extra) or extra < 0:
            raise ValueError(
                "ML delay must be finite and non-negative"
            )

        return free_flow + extra

    def weight(a, b, d):
        return seg_time(a, b)

    try:
        paths = nx.shortest_simple_paths(
            G, source, hospital, weight=weight
        )

        routes = []

        for path in paths:
            pairs = list(zip(path, path[1:]))
            times = [seg_time(a, b) for a, b in pairs]

            routes.append({
                "path": path,
                "segments": [
                    G[a][b]["segment_id"] for a, b in pairs
                ],
                "segment_times_s": times,
                "total_cost_s": sum(times)
            })

            if len(routes) >= max_routes:
                break

        return routes

    except nx.NetworkXNoPath:
        raise NoPath(f"no route from {source} to {hospital}")
