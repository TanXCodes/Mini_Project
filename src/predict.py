"""Owner: Tamanna (Dataset + ML).

Load the trained delay model and provide:
- predict_delay(features): delay in minutes
- delay_fn(segment_id, scenario): delay in seconds for routing

Optimizations:
- Model is loaded only once.
- CSV data is loaded only once.
- A scenario is batch-predicted once and cached.
- delay_fn() performs only a dictionary lookup.
- Accident delay is applied only to affected_segments.
"""

import os

import joblib
import pandas as pd


MODEL_PATH = "models/delay_model.joblib"

TRAFFIC_PATH = "data/traffic_raw.csv"
EDGES_PATH = "data/edges.csv"
NODES_PATH = "data/nodes.csv"


# -------------------------------------------------------------------
# Caches
# -------------------------------------------------------------------

_MODEL = None
_STATIC_DATA = None
_SCENARIO_CACHE = {}


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def _load_model():
    """Load the trained model once and reuse it."""

    global _MODEL

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


# -------------------------------------------------------------------
# Static data loading
# -------------------------------------------------------------------

def _load_static_data():
    """Load traffic, edges and nodes only once."""

    global _STATIC_DATA

    if _STATIC_DATA is None:

        traffic = pd.read_csv(TRAFFIC_PATH)
        edges = pd.read_csv(EDGES_PATH)
        nodes = pd.read_csv(NODES_PATH)

        # Keep segment IDs consistent for dictionary lookups.
        traffic["segment_id"] = (
            traffic["segment_id"].astype(str)
        )

        edges["segment_id"] = (
            edges["segment_id"].astype(str)
        )

        nodes["junction_id"] = (
            nodes["junction_id"].astype(str)
        )

        _STATIC_DATA = {
            "traffic": traffic,
            "edges": edges,
            "nodes": nodes,
        }

    return _STATIC_DATA


# -------------------------------------------------------------------
# General prediction function
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

    # Convert a single dictionary into a one-row DataFrame.
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

    # Return a float for one row.
    if len(predictions) == 1:
        return float(predictions[0])

    return predictions


# -------------------------------------------------------------------
# Scenario batch prediction
# -------------------------------------------------------------------

def _build_scenario_cache(
    time_slot,
    accident,
    affected_segments=None,
):
    """Predict delays for all road segments for one scenario.

    The expensive work happens only once per unique scenario.
    """

    if affected_segments is None:
        affected_segments = []

    # Normalize segment IDs.
    affected_segments = tuple(
        sorted(
            str(segment)
            for segment in affected_segments
        )
    )

    # Cache key.
    cache_key = (
        str(time_slot),
        bool(accident),
        affected_segments,
    )

    if cache_key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[cache_key]

    data = _load_static_data()

    traffic = data["traffic"]
    edges = data["edges"]
    nodes = data["nodes"]

    # ---------------------------------------------------------------
    # Select traffic records for the requested time slot.
    # ---------------------------------------------------------------

    segment_traffic = traffic[
        traffic["time_slot"] == time_slot
    ].copy()

    # If the requested time slot does not exist,
    # use the first available record for each segment.
    if segment_traffic.empty:

        segment_traffic = (
            traffic
            .sort_values(
                ["segment_id", "day"]
            )
            .groupby(
                "segment_id",
                as_index=False
            )
            .first()
        )

    else:

        # Keep one traffic row per segment.
        segment_traffic = (
            segment_traffic
            .sort_values(
                ["segment_id", "day"]
            )
            .groupby(
                "segment_id",
                as_index=False
            )
            .first()
        )

    # ---------------------------------------------------------------
    # Merge traffic with edge information.
    # ---------------------------------------------------------------

    df = segment_traffic.merge(
        edges,
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Merge starting node information.
    # ---------------------------------------------------------------

    from_nodes = nodes.rename(
        columns={
            "junction_id": "from_node",
            "lat": "from_lat",
            "lon": "from_lon",
            "has_signal": "from_has_signal",
        }
    )

    df["from_node"] = df["from_node"].astype(str)

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
    # Merge ending node information.
    # ---------------------------------------------------------------

    to_nodes = nodes.rename(
        columns={
            "junction_id": "to_node",
            "lat": "to_lat",
            "lon": "to_lon",
            "has_signal": "to_has_signal",
        }
    )

    df["to_node"] = df["to_node"].astype(str)

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
    # Signal features.
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
            (df["from_has_signal"] == 1)
            |
            (df["to_has_signal"] == 1)
        )
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Accident handling.
    #
    # If affected_segments is supplied, ONLY those segments receive
    # incident_flag=True.
    #
    # If accident=True but affected_segments is empty, we keep
    # backward compatibility by treating all segments as affected.
    # ---------------------------------------------------------------

    if accident:

        if affected_segments:

            df["incident_flag"] = (
                df["segment_id"]
                .astype(str)
                .isin(affected_segments)
            )

        else:

            df["incident_flag"] = True

    else:

        df["incident_flag"] = False

    # ---------------------------------------------------------------
    # VC ratio.
    # ---------------------------------------------------------------

    df["vc_ratio"] = (
        df["volume"]
        / df["capacity"].replace(0, pd.NA)
    )

    df["vc_ratio"] = (
        df["vc_ratio"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------------
    # Make sure categorical/bool columns have usable values.
    # ---------------------------------------------------------------

    df["time_slot"] = time_slot

    df["road_class"] = (
        df["road_class"]
        .fillna("unknown")
    )

    df["oneway"] = (
        df["oneway"]
        .fillna(False)
        .astype(bool)
    )

    df["maxspeed"] = (
        df["maxspeed"]
        .fillna(0)
    )

    df["length_km"] = (
        df["length_km"]
        .fillna(0)
    )

    df["free_flow_speed_kmph"] = (
        df["free_flow_speed_kmph"]
        .fillna(0)
    )

    df["volume"] = (
        df["volume"]
        .fillna(0)
    )

    df["capacity"] = (
        df["capacity"]
        .fillna(0)
    )

    df["from_lat"] = (
        df["from_lat"]
        .fillna(0)
    )

    df["from_lon"] = (
        df["from_lon"]
        .fillna(0)
    )

    df["to_lat"] = (
        df["to_lat"]
        .fillna(0)
    )

    df["to_lon"] = (
        df["to_lon"]
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Exact features used by the trained model.
    #
    # IMPORTANT:
    # speed_kmph and travel_time_min are NOT included.
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
    # Batch prediction.
    # ---------------------------------------------------------------

    predicted_minutes = predict_delay(X)

    df["predicted_delay_seconds"] = (
        predicted_minutes * 60.0
    )

    # ---------------------------------------------------------------
    # Accident multiplier.
    #
    # Only affected segments get the additional increase when
    # affected_segments is explicitly supplied.
    # ---------------------------------------------------------------

    if accident and affected_segments:

        affected_mask = (
            df["segment_id"]
            .astype(str)
            .isin(affected_segments)
        )

        df.loc[
            affected_mask,
            "predicted_delay_seconds"
        ] *= 2.0

    elif accident:

        # Backward-compatible behavior when no specific
        # affected segments were supplied.
        df["predicted_delay_seconds"] *= 2.0

    # ---------------------------------------------------------------
    # Build the final dictionary.
    # ---------------------------------------------------------------

    delay_dict = dict(
        zip(
            df["segment_id"].astype(str),
            df["predicted_delay_seconds"].astype(float),
        )
    )

    _SCENARIO_CACHE[cache_key] = delay_dict

    return delay_dict


# -------------------------------------------------------------------
# Routing function
# -------------------------------------------------------------------

def delay_fn(segment_id, scenario):
    """Return predicted delay for one segment in seconds.

    Parameters
    ----------
    segment_id : str
        Road segment identifier.

    scenario : dict
        Scenario containing:
        - time_slot
        - accident
        - affected_segments

    Returns
    -------
    float
        Predicted delay in seconds.

    Notes
    -----
    Expensive CSV loading and model prediction do NOT happen here.

    The scenario is batch-predicted once and cached.
    This function then performs a simple dictionary lookup.
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

    affected_segments = scenario.get(
        "affected_segments",
        [],
    )

    # Build/retrieve the complete scenario prediction.
    delay_dict = _build_scenario_cache(
        time_slot=time_slot,
        accident=bool(accident),
        affected_segments=affected_segments,
    )

    # Fast dictionary lookup.
    return float(
        delay_dict.get(
            str(segment_id),
            0.0,
        )
    )