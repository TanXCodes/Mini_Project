import networkx as nx

def test_shortest_path_small_graph():
    G = nx.Graph()
    G.add_weighted_edges_from([
        ("A", "B", 4),
        ("A", "C", 2),
        ("C", "B", 1),
        ("B", "D", 5),
        ("C", "D", 8),
    ])

    path = nx.dijkstra_path(G, "A", "D", weight="weight")
    cost = nx.dijkstra_path_length(G, "A", "D", weight="weight")

    assert path == ["A", "C", "B", "D"]
    assert cost == 8
