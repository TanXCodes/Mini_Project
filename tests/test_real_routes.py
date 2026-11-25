import yaml
import pytest

from src.graph import build_graph, largest_component
from src.routing import compute_route
from src.predict import delay_fn

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
@pytest.mark.parametrize("source, hospital", [
    (363775666, 3716332298),
    (309593064, 7148693828),
    (305091341, 2228225821),
])
def test_real_route_with_ml_delay(source, hospital, road_graph):
    component = largest_component(road_graph)

    assert source in component
    assert hospital in component

    scenario = {
        "time_slot": "evening_peak",
        "accident": False,
        "affected_segments": []
    }

    result = compute_route(
        source,
        hospital,
        scenario=scenario,
        delay_fn=delay_fn,
        G=road_graph
    )

    assert result["path"][0] == source
    assert result["path"][-1] == hospital
    assert result["total_cost_s"] > 0
    assert len(result["segments"]) == len(result["path"]) - 1
    assert all(t >= 0 for t in result["segment_times_s"])
