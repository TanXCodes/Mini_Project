"""Owner: Tamanna. EDA for traffic delay analysis."""

import matplotlib.pyplot as plt
from prepare_data import build_dataset


def create_eda_plots():
    train, validation, test = build_dataset()

    # Use training data for EDA
    df = train.copy()

    # 1. Delay distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["delay_min"], bins=50)
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Number of Records")
    plt.title("Delay Distribution")
    plt.tight_layout()
    plt.savefig("outputs/eda_delay_distribution.png")
    plt.close()

    # 2. Average delay by time slot
    avg_delay = df.groupby("time_slot")["delay_min"].mean()

    plt.figure(figsize=(9, 5))
    avg_delay.plot(kind="bar")
    plt.xlabel("Time Slot")
    plt.ylabel("Average Delay (minutes)")
    plt.title("Average Delay by Time Slot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/eda_delay_by_timeslot.png")
    plt.close()

    # 3. Delay vs VC ratio
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["vc_ratio"],
        df["delay_min"],
        alpha=0.3
    )
    plt.xlabel("Volume-to-Capacity Ratio")
    plt.ylabel("Delay (minutes)")
    plt.title("Delay vs VC Ratio")
    plt.tight_layout()
    plt.savefig("outputs/eda_delay_vs_vc_ratio.png")
    plt.close()


if __name__ == "__main__":
    create_eda_plots()