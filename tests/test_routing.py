import networkx as nx
import pytest

from src.routing import compute_route, NoPath


def tiny_graph():
    """A -> B -> D (slow) and A -> C -> D (fast)."""
    G = nx.DiGraph()
    G.add_edge("A", "B", segment_id="AB", free_flow_s=10)
    G.add_edge("B", "D", segment_id="BD", free_flow_s=50)
    G.add_edge("A", "C", segment_id="AC", free_flow_s=20)
    G.add_edge("C", "D", segment_id="CD", free_flow_s=20)
    G.add_node("Z")
    return G


def test_picks_fastest_route():
    r = compute_route("A", "D", G=tiny_graph())
    assert r["path"] == ["A", "C", "D"]
    assert r["total_cost_s"] == 40


def test_delay_can_change_route():
    delay = lambda seg, sc: 100 if seg == "CD" else 0   # CD becomes jammed
    r = compute_route("A", "D", delay_fn=delay, G=tiny_graph())
    assert r["path"] == ["A", "B", "D"]


def test_same_source_and_hospital():
    with pytest.raises(ValueError):
        compute_route("A", "A", G=tiny_graph())


def test_unreachable():
    with pytest.raises(NoPath):
        compute_route("A", "Z", G=tiny_graph())
