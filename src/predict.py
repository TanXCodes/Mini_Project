"""Owner: Tamanna (Dataset + ML).

Load the trained delay model and provide:
- predict_delay(features): delay in minutes
- delay_fn(segment_id, scenario): delay in seconds for routing
"""

import os

import joblib
import pandas as pd


MODEL_PATH = "models/delay_model.joblib"


# Cache the model so it is loaded only once.
_MODEL = None


def _load_model():
    global _MODEL

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


def predict_delay(features):
    """Predict delay in minutes.

    Parameters
    ----------
    features : dict or pandas.DataFrame
        Features required by the trained model.

    Returns
    -------
    float or array-like
        Predicted delay in minutes.
    """

    model = _load_model()

    # Convert a single dictionary into a one-row DataFrame.
    if isinstance(features, dict):
        features = pd.DataFrame([features])

    elif not isinstance(features, pd.DataFrame):
        raise TypeError(
            "features must be a dictionary or pandas DataFrame"
        )

    # The model was trained without these leakage columns.
    features = features.drop(
        columns=[
            "delay_min",
            "speed_kmph",
            "travel_time_min",
        ],
        errors="ignore",
    )

    predictions = model.predict(features)

    # Return a float for a single row.
    if len(predictions) == 1:
        return float(predictions[0])

    return predictions


def delay_fn(segment_id, scenario):
    """Return predicted delay for a segment in seconds.

    Parameters
    ----------
    segment_id : str
        Road segment identifier.

    scenario : dict
        Scenario containing:
        - time_slot
        - accident

    Returns
    -------
    float
        Predicted delay in seconds.
    """

    if scenario is None:
        scenario = {}

    time_slot = scenario.get(
        "time_slot",
        "night"
    )

    accident = scenario.get(
        "accident",
        scenario.get(
            "incident_flag",
            False
        )
    )

    # Load traffic and edge information.
    traffic = pd.read_csv(
        "data/traffic_raw.csv"
    )

    edges = pd.read_csv(
        "data/edges.csv"
    )

    nodes = pd.read_csv(
        "data/nodes.csv"
    )

    # Find traffic records for this segment and time slot.
    segment_traffic = traffic[
        (traffic["segment_id"] == segment_id)
        & (traffic["time_slot"] == time_slot)
    ]

    # If the requested time slot does not exist,
    # use the segment's first available traffic record.
    if segment_traffic.empty:
        segment_traffic = traffic[
            traffic["segment_id"] == segment_id
        ]

    if segment_traffic.empty:
        return 0.0

    traffic_row = segment_traffic.iloc[0].copy()

    # Get road information.
    edge_rows = edges[
        edges["segment_id"] == segment_id
    ]

    if edge_rows.empty:
        return 0.0

    edge_row = edge_rows.iloc[0]

    # Get from/to junction information.
    from_node = nodes[
        nodes["junction_id"] == edge_row["from_node"]
    ]

    to_node = nodes[
        nodes["junction_id"] == edge_row["to_node"]
    ]

    # Build the same features used during training.
    from_row = (
        from_node.iloc[0]
        if not from_node.empty
        else None
    )

    to_row = (
        to_node.iloc[0]
        if not to_node.empty
        else None
    )

    from_has_signal = (
        int(bool(from_row["has_signal"]))
        if from_row is not None
        else 0
    )

    to_has_signal = (
        int(bool(to_row["has_signal"]))
        if to_row is not None
        else 0
    )

    feature_data = {
        "day": traffic_row["day"],
        "time_slot": time_slot,
        "free_flow_speed_kmph": traffic_row[
            "free_flow_speed_kmph"
        ],
        "volume": traffic_row["volume"],
        "capacity": traffic_row["capacity"],
        "incident_flag": bool(accident),
        "from_node": edge_row["from_node"],
        "to_node": edge_row["to_node"],
        "road_class": edge_row["road_class"],
        "length_km": edge_row["length_km"],
        "maxspeed": edge_row["maxspeed"],
        "oneway": edge_row["oneway"],
        "from_lat": (
            from_row["lat"]
            if from_row is not None
            else None
        ),
        "from_lon": (
            from_row["lon"]
            if from_row is not None
            else None
        ),
        "from_has_signal": from_has_signal,
        "to_lat": (
            to_row["lat"]
            if to_row is not None
            else None
        ),
        "to_lon": (
            to_row["lon"]
            if to_row is not None
            else None
        ),
        "to_has_signal": to_has_signal,
        "signal_near_segment": int(
            from_has_signal or to_has_signal
        ),
        "vc_ratio": (
            traffic_row["volume"]
            / traffic_row["capacity"]
            if traffic_row["capacity"] != 0
            else 0.0
        ),
    }

    delay_minutes = predict_delay(
        feature_data
    )

    # Convert minutes -> seconds for routing.
    delay_seconds = delay_minutes * 60.0

    return float(delay_seconds)