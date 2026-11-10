import pandas as pd


# Load the three datasets
edges = pd.read_csv("data/edges.csv")
nodes = pd.read_csv("data/nodes.csv")
traffic = pd.read_csv("data/traffic_raw.csv")


# =========================
# EDGES DATASET
# =========================
print("\n========== EDGES ==========")
print("Shape:", edges.shape)
print("\nHead:")
print(edges.head())
print("\nDescription:")
print(edges.describe(include="all"))
print("\nColumns:")
print(edges.columns.tolist())


# =========================
# NODES DATASET
# =========================
print("\n========== NODES ==========")
print("Shape:", nodes.shape)
print("\nHead:")
print(nodes.head())
print("\nDescription:")
print(nodes.describe(include="all"))
print("\nColumns:")
print(nodes.columns.tolist())


# =========================
# TRAFFIC DATASET
# =========================
print("\n========== TRAFFIC ==========")
print("Shape:", traffic.shape)
print("\nHead:")
print(traffic.head())
print("\nDescription:")
print(traffic.describe(include="all"))
print("\nColumns:")
print(traffic.columns.tolist())


# =========================
# MISSING VALUES
# =========================
print("\n========== MISSING VALUES ==========")

print("\nEdges:")
print(edges.isnull().sum())

print("\nNodes:")
print(nodes.isnull().sum())

print("\nTraffic:")
print(traffic.isnull().sum())


# =========================
# DELAY ANALYSIS
# =========================
if "delay_min" in traffic.columns:
    print("\n========== DELAY ANALYSIS ==========")

    print("\nDelay distribution:")
    print(traffic["delay_min"].value_counts().sort_index())

    print("\nDelay statistics:")
    print(traffic["delay_min"].describe())

    zero_delay_percentage = (
        (traffic["delay_min"] == 0).mean() * 100
    )

    print(
        f"\nPercentage of records with delay_min = 0: "
        f"{zero_delay_percentage:.2f}%"
    )
else:
    print("\n'delay_min' column was not found in traffic_raw.csv")

print("\n========== DATASET SUMMARY ==========")

print("Edges shape:", edges.shape)
print("Nodes shape:", nodes.shape)
print("Traffic shape:", traffic.shape)

print(
    f"Zero-delay records: "
    f"{(traffic['delay_min'] == 0).sum()} "
    f"({(traffic['delay_min'] == 0).mean() * 100:.2f}%)"
)

print(
    "Missing maxspeed values:",
    edges["maxspeed"].isnull().sum()
)

print(
    "Signalized junctions:",
    nodes["has_signal"].sum()
)    