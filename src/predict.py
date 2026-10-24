"""Owner: Tamanna (Dataset + ML).

Load the trained delay model and provide:
- predict_delay(features): delay in minutes
- delay_fn(segment_id, scenario): delay in seconds for routing

Optimized for routing:
- CSV data is loaded only once.
- Model is loaded only once.
- Scenario predictions are cached.
- delay_fn() performs a fast dictionary lookup.
"""

import os

import joblib
import pandas as pd


MODEL_PATH = "models/delay_model.joblib"

TRAFFIC_PATH = "data/traffic_raw.csv"
EDGES_PATH = "data/edges.csv"
NODES_PATH = "data/nodes.csv"


# -------------------------------------------------------------------
# Global caches
# -------------------------------------------------------------------

_MODEL = None

_TRAFFIC = None
_EDGES = None
_NODES = None

# Cache:
# {
#     ("evening_peak", False): {
#         "segment_id_1": delay_seconds,
#         "segment_id_2": delay_seconds,
#         ...
#     }
# }
_DELAY_CACHE = {}


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def _load_model():
    """Load the trained model only once."""

    global _MODEL

    if _MODEL is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def _load_data():
    """Load CSV files only once."""

    global _TRAFFIC
    global _EDGES
    global _NODES

    if _TRAFFIC is None:
        _TRAFFIC = pd.read_csv(TRAFFIC_PATH)

    if _EDGES is None:
        _EDGES = pd.read_csv(EDGES_PATH)

    if _NODES is None:
        _NODES = pd.read_csv(NODES_PATH)

    return _TRAFFIC, _EDGES, _NODES


# -------------------------------------------------------------------
# Normal prediction
# -------------------------------------------------------------------

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

    # Convert dictionary into one-row DataFrame.
    if isinstance(features, dict):

        features = pd.DataFrame([features])

    elif not isinstance(features, pd.DataFrame):

        raise TypeError(
            "features must be a dictionary or pandas DataFrame"
        )

    # Remove target and leakage columns.
    features = features.drop(
        columns=[
            "delay_min",
            "speed_kmph",
            "travel_time_min",
        ],
        errors="ignore",
    )

    predictions = model.predict(features)

    # Single prediction -> float
    if len(predictions) == 1:

        return float(predictions[0])

    return predictions


# -------------------------------------------------------------------
# Feature creation
# -------------------------------------------------------------------

def _build_features(traffic_row, edge_row, from_row, to_row, accident):
    """Build exactly the features expected by the trained model."""

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

    capacity = float(traffic_row["capacity"])
    volume = float(traffic_row["volume"])

    if capacity != 0:
        vc_ratio = volume / capacity
    else:
        vc_ratio = 0.0

    return {
        "day": traffic_row["day"],

        "time_slot": traffic_row["time_slot"],

        "free_flow_speed_kmph":
            traffic_row["free_flow_speed_kmph"],

        "volume":
            volume,

        "capacity":
            capacity,

        "incident_flag":
            bool(accident),

        "from_node":
            edge_row["from_node"],

        "to_node":
            edge_row["to_node"],

        "road_class":
            edge_row["road_class"],

        "length_km":
            edge_row["length_km"],

        "maxspeed":
            edge_row["maxspeed"],

        "oneway":
            edge_row["oneway"],

        "from_lat":
            from_row["lat"]
            if from_row is not None
            else None,

        "from_lon":
            from_row["lon"]
            if from_row is not None
            else None,

        "from_has_signal":
            from_has_signal,

        "to_lat":
            to_row["lat"]
            if to_row is not None
            else None,

        "to_lon":
            to_row["lon"]
            if to_row is not None
            else None,

        "to_has_signal":
            to_has_signal,

        "signal_near_segment":
            int(
                from_has_signal or to_has_signal
            ),

        "vc_ratio":
            vc_ratio,
    }


# -------------------------------------------------------------------
# Scenario batch prediction
# -------------------------------------------------------------------

def _build_scenario_cache(time_slot, accident):
    """Predict all segment delays for one scenario at once.

    The result is stored as:

        {
            segment_id: delay_seconds
        }

    This means delay_fn() only needs a dictionary lookup.
    """

    traffic, edges, nodes = _load_data()

    cache_key = (
        time_slot,
        bool(accident),
    )

    # Return existing cache.
    if cache_key in _DELAY_CACHE:

        return _DELAY_CACHE[cache_key]

    # ---------------------------------------------------------------
    # Select traffic rows for requested time slot
    # ---------------------------------------------------------------

    scenario_traffic = traffic[
        traffic["time_slot"] == time_slot
    ].copy()

    # If the time slot doesn't exist, use all traffic rows.
    if scenario_traffic.empty:

        scenario_traffic = traffic.copy()

    # One traffic row per segment.
    scenario_traffic = (
        scenario_traffic
        .sort_values(["segment_id", "day"])
        .drop_duplicates(
            subset=["segment_id"],
            keep="first"
        )
    )

    # ---------------------------------------------------------------
    # Merge traffic + edge data
    # ---------------------------------------------------------------

    df = scenario_traffic.merge(
        edges,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Merge starting node information
    # ---------------------------------------------------------------

    from_nodes = nodes.rename(
        columns={
            "junction_id": "from_node",
            "lat": "from_lat",
            "lon": "from_lon",
            "has_signal": "from_has_signal",
        }
    )

    df = df.merge(
        from_nodes[
            [
                "from_node",
                "from_lat",
                "from_lon",
                "from_has_signal",
            ]
        ],
        on="from_node",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Merge ending node information
    # ---------------------------------------------------------------

    to_nodes = nodes.rename(
        columns={
            "junction_id": "to_node",
            "lat": "to_lat",
            "lon": "to_lon",
            "has_signal": "to_has_signal",
        }
    )

    df = df.merge(
        to_nodes[
            [
                "to_node",
                "to_lat",
                "to_lon",
                "to_has_signal",
            ]
        ],
        on="to_node",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Build ML features
    # ---------------------------------------------------------------

    df["from_has_signal"] = (
        df["from_has_signal"]
        .fillna(False)
        .astype(int)
    )

    df["to_has_signal"] = (
        df["to_has_signal"]
        .fillna(False)
        .astype(int)
    )

    df["signal_near_segment"] = (
        (
            df["from_has_signal"] == 1
        )
        |
        (
            df["to_has_signal"] == 1
        )
    ).astype(int)

    df["vc_ratio"] = (
        df["volume"]
        /
        df["capacity"].replace(0, 1)
    )

    # Accident scenario.
    df["incident_flag"] = bool(accident)

    # ---------------------------------------------------------------
    # Columns used by the trained model
    # ---------------------------------------------------------------

    feature_columns = [
        "day",
        "time_slot",
        "free_flow_speed_kmph",
        "volume",
        "capacity",
        "incident_flag",
        "from_node",
        "to_node",
        "road_class",
        "length_km",
        "maxspeed",
        "oneway",
        "from_lat",
        "from_lon",
        "from_has_signal",
        "to_lat",
        "to_lon",
        "to_has_signal",
        "signal_near_segment",
        "vc_ratio",
    ]

    X = df[feature_columns].copy()

    # ---------------------------------------------------------------
    # Predict ALL segments in one model call
    # ---------------------------------------------------------------

    predictions_minutes = predict_delay(X)

    df["predicted_delay_seconds"] = (
        predictions_minutes * 60.0
    )

    # ---------------------------------------------------------------
    # Ensure accident scenario gives higher delay
    # ---------------------------------------------------------------

    if accident:

        # Accident multiplier.
        #
        # The model's incident_flag is already set to True.
        # This additional factor guarantees that an accident
        # scenario increases routing delay even if the trained
        # model predicts very little difference.
        df["predicted_delay_seconds"] *= 2.0

    # ---------------------------------------------------------------
    # Create fast lookup dictionary
    # ---------------------------------------------------------------

    delay_dict = dict(
        zip(
            df["segment_id"],
            df["predicted_delay_seconds"],
        )
    )

    # Save scenario cache.
    _DELAY_CACHE[cache_key] = delay_dict

    return delay_dict


# -------------------------------------------------------------------
# Routing function
# -------------------------------------------------------------------

def delay_fn(segment_id, scenario):
    """Return predicted delay for a segment in seconds.

    This function is intentionally lightweight because routing
    may call it many times.

    Parameters
    ----------
    segment_id : str
        Road segment identifier.

    scenario : dict
        Scenario containing:
            time_slot
            accident

    Returns
    -------
    float
        Predicted delay in seconds.
    """

    if scenario is None:

        scenario = {}

    time_slot = scenario.get(
        "time_slot",
        "night",
    )

    accident = scenario.get(
        "accident",
        scenario.get(
            "incident_flag",
            False,
        ),
    )

    # Build/load the scenario cache.
    delay_dict = _build_scenario_cache(
        time_slot=time_slot,
        accident=bool(accident),
    )

    # Fast dictionary lookup.
    return float(
        delay_dict.get(
            segment_id,
            0.0,
        )
    )