"""Tests for ML delay prediction."""

from src.predict import predict_delay, delay_fn


def get_features():
    return {
        "day": 3,
        "time_slot": "evening_peak",
        "free_flow_speed_kmph": 40.0,
        "volume": 1500.0,
        "capacity": 1800.0,
        "incident_flag": False,
        "from_node": 17327139,
        "to_node": 443224605,
        "road_class": "primary",
        "length_km": 0.044631,
        "maxspeed": 50.0,
        "oneway": True,
        "from_lat": 12.934965,
        "from_lon": 77.624072,
        "from_has_signal": 0,
        "to_lat": 12.934000,
        "to_lon": 77.624000,
        "to_has_signal": 0,
        "signal_near_segment": 0,
        "vc_ratio": 1500 / 1800,
    }


def test_predict_delay_returns_number():
    """Prediction should return a numeric delay."""
    delay = predict_delay(get_features())

    assert isinstance(delay, (int, float))


def test_predict_delay_is_not_negative():
    """Predicted delay should not be negative."""
    delay = predict_delay(get_features())

    assert delay >= 0


def test_accident_delay_is_higher_than_normal():
    """Affected accident segment should have higher delay."""
    segment = "17327139_309593057_0"

    normal = {
        "name": "midday_normal",
        "time_slot": "midday",
        "accident": False,
        "affected_segments": [],
    }

    accident = {
        "name": "accident",
        "time_slot": "evening_peak",
        "accident": True,
        "affected_segments": [segment],
    }

    normal_delay = delay_fn(segment, normal)
    accident_delay = delay_fn(segment, accident)

    assert accident_delay >= normal_delay