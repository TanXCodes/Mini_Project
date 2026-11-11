import yaml
import pytest

from src.graph import build_graph, largest_component
from src.routing import compute_route


@pytest.fixture(scope="module")
def road_graph():
    with open("config.yaml") as f:
        paths = yaml.safe_load(f)["paths"]

    return build_graph(
        paths["nodes"],
        paths["edges"],
        paths["traffic"]
    )


@pytest.mark.parametrize("source, hospital", [
    (363775666, 3716332298),
    (309593064, 7148693828),
    (305091341, 2228225821),
])
def test_real_route(source, hospital, road_graph):
    component = largest_component(road_graph)

    assert source in component
    assert hospital in component

    result = compute_route(source, hospital, G=road_graph)

    assert result["path"][0] == source
    assert result["path"][-1] == hospital
    assert result["total_cost_s"] > 0
    assert len(result["segments"]) == len(result["path"]) - 1