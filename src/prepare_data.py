"""Owner: Tamanna. Merge nodes/edges/traffic, build features, split by day."""

import pandas as pd


def build_dataset():
    # Load datasets
    traffic = pd.read_csv("data/traffic_raw.csv")
    edges = pd.read_csv("data/edges.csv")
    nodes = pd.read_csv("data/nodes.csv")

    # Merge traffic data with road/edge information
    df = traffic.merge(
        edges,
        on="segment_id",
        how="left",
        validate="many_to_one"
    )

    # Add information about the starting junction
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
            ["from_node", "from_lat", "from_lon", "from_has_signal"]
        ],
        on="from_node",
        how="left",
        validate="many_to_one"
    )

    # Add information about the ending junction
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
            ["to_node", "to_lat", "to_lon", "to_has_signal"]
        ],
        on="to_node",
        how="left",
        validate="many_to_one"
    )

    # Create useful ML features
    df["from_has_signal"] = df["from_has_signal"].fillna(False).astype(int)
    df["to_has_signal"] = df["to_has_signal"].fillna(False).astype(int)

    df["signal_near_segment"] = (
        (df["from_has_signal"] == 1)
        | (df["to_has_signal"] == 1)
    ).astype(int)
    # Volume-to-capacity ratio
    df["vc_ratio"] = df["volume"] / df["capacity"]

    # Handle missing speed limits using road-class median
    df["maxspeed"] = df["maxspeed"].fillna(
        df.groupby("road_class")["maxspeed"].transform("median")
    )

    # Final fallback in case a road class has no speed value
    df["maxspeed"] = df["maxspeed"].fillna(
        df["maxspeed"].median()
    )

    # Split by day
    train = df[df["day"] <= 3].copy()
    validation = df[df["day"] == 4].copy()
    test = df[df["day"] == 5].copy()

    return train, validation, test