"""Owner: Muhurt (Routing). Build the NetworkX road graph.

Step 1 of the synopsis plan: graph + basic Dijkstra.
Edge weight here is free-flow travel time (seconds). ML delay is added later.
"""
import networkx as nx
import pandas as pd

# fallback speeds (km/h) if a segment has no speed in the traffic data
DEFAULT_SPEED = {"primary": 40, "secondary": 35, "tertiary": 30,
                 "residential": 25, "living_street": 15,
                 "primary_link": 30, "busway": 30}


def build_graph(nodes_csv, edges_csv, traffic_csv=None):
    """Return nx.DiGraph.

    - oneway road: one direction only, otherwise both directions
    - node attributes: lat, lon, has_signal
    - edge attributes: segment_id, length_km, road_class, free_flow_s
    - if two roads connect the same pair of junctions, keep the faster one
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    speed = {}
    if traffic_csv:
        t = pd.read_csv(traffic_csv, usecols=["segment_id", "free_flow_speed_kmph"])
        speed = t.groupby("segment_id")["free_flow_speed_kmph"].mean().to_dict()

    G = nx.DiGraph()
    for r in nodes.itertuples():
        G.add_node(r.junction_id, lat=r.lat, lon=r.lon, has_signal=bool(r.has_signal))

    for r in edges.itertuples():
        kmph = speed.get(r.segment_id) or DEFAULT_SPEED.get(r.road_class, 25)
        free_flow_s = r.length_km / kmph * 3600
        data = dict(segment_id=r.segment_id, length_km=r.length_km,
                    road_class=r.road_class, free_flow_s=free_flow_s)
        pairs = [(r.from_node, r.to_node)]
        if not r.oneway:
            pairs.append((r.to_node, r.from_node))
        for a, b in pairs:
            if G.has_edge(a, b) and G[a][b]["free_flow_s"] <= free_flow_s:
                continue                      # keep the faster duplicate
            G.add_edge(a, b, **data)
    return G


def largest_component(G):
    """Junctions from which every other junction is reachable (safe to use
    as source / hospital)."""
    return max(nx.strongly_connected_components(G), key=len)
