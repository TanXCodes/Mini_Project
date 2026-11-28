
import csv
from pathlib import Path

from src.routing import compute_route, get_graph
from src.predict import delay_fn, _build_scenario_cache
from src.simulator import load_config


def main():
    cfg = load_config()
    G = get_graph()

    affected_segments = [
        "12451306645_3399067454_0",
        "4761188837_11964724285_0",
        "1620344798_254920166_0",
    ]

    normal_scenario = {
        "name": "evening_peak",
        "time_slot": "evening_peak",
        "accident": False,
    }

    accident_scenario = {
        "name": "localized_accident",
        "time_slot": "evening_peak",
        "accident": True,
        "affected_segments": affected_segments,
    }

    # Build both scenario caches.
    normal_delays = _build_scenario_cache(
        time_slot="evening_peak",
        accident=False,
    )

    accident_delays = _build_scenario_cache(
        time_slot="evening_peak",
        accident=True,
        affected_segments=affected_segments,
    )

    # Verify localized accident calculations.
    for segment in affected_segments:
        assert segment in normal_delays, f"Missing segment: {segment}"
        assert abs(
            accident_delays[segment] - 2 * normal_delays[segment]
        ) < 1e-8, f"2x validation failed: {segment}"

    affected_set = set(affected_segments)

    for segment, normal_delay in normal_delays.items():
        if segment not in affected_set:
            assert abs(
                accident_delays[segment] - normal_delay
            ) < 1e-8, f"Unaffected segment changed: {segment}"

    print("Affected and unaffected segment validation passed.")

    results = []

    for trip_index, trip in enumerate(cfg["trips"], start=1):
        normal_route = compute_route(
            trip["source"],
            trip["hospital"],
            scenario=normal_scenario,
            delay_fn=delay_fn,
            G=G,
        )

        accident_route = compute_route(
            trip["source"],
            trip["hospital"],
            scenario=accident_scenario,
            delay_fn=delay_fn,
            G=G,
        )

        route_changed = (
            normal_route["path"] != accident_route["path"]
        )

        normal_time = normal_route["total_cost_s"]
        accident_time = accident_route["total_cost_s"]
        additional_delay = accident_time - normal_time

        results.append({
            "trip": f"Trip {trip_index}",
            "source": trip["source"],
            "hospital": trip["hospital"],
            "normal_evening_s": round(normal_time, 3),
            "localized_accident_s": round(accident_time, 3),
            "additional_delay_s": round(additional_delay, 3),
            "route_changed": route_changed,
        })

        print(
            f"Trip {trip_index}: "
            f"Normal={normal_time:.3f}s | "
            f"Accident={accident_time:.3f}s | "
            f"Additional={additional_delay:.3f}s | "
            f"Route changed={route_changed}"
        )

    assert len(results) == 3, "Expected exactly 3 trip results"

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = (
        output_dir / "review_localized_accident_comparison.csv"
    )

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nLocalized accident evaluation completed.")
    print("Results saved to:", output_file)


if __name__ == "__main__":
    main()
