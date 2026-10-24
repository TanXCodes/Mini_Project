"""Owner: Vinay (UI + Integration). Run: streamlit run app/app.py (from project root)"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.routing import compute_route, get_graph
from src.simulator import load_config, load_signal_nodes, run_simulation


USE_ML_DELAY = False
N_RUNS = 30   

TRIPS = {
    "Trip 1": (363775666, 3716332298),
    "Trip 2": (309593064, 7148693828),
    "Trip 3": (305091341, 2228225821),
}

st.set_page_config(page_title="AI Emergency Corridor System", layout="wide")
st.title("AI-Based Emergency Corridor System")
st.caption("Team Vortex | Normal vs Priority mode comparison (simulation)")


@st.cache_resource
def load_everything():
    cfg = load_config()
    G = get_graph()
    p = cfg["paths"]
    signal_nodes = load_signal_nodes(p["nodes"], p["edges"],
                                     cfg["signals"]["extra_simulated_signals"])
    return cfg, G, signal_nodes


cfg, G, signal_nodes = load_everything()
scenarios = {s["name"]: s for s in cfg["scenarios"]}

col1, col2 = st.columns(2)
scenario_name = col1.selectbox("Scenario", list(scenarios))
trip_name = col2.selectbox("Trip", list(TRIPS))
source, hospital = TRIPS[trip_name]

if st.button("Calculate route", type="primary"):
    delay_fn = None
    if USE_ML_DELAY:
        from src.predict import delay_fn
    try:
        route = compute_route(source, hospital, scenarios[scenario_name],
                              delay_fn=delay_fn, G=G)
    except Exception as e:                      # NoPath, ValueError ...
        st.error(f"Could not find a route: {e}")
        st.stop()

    runs = {"normal": [], "priority": []}
    for seed in range(N_RUNS):
        for mode in runs:
            runs[mode].append(run_simulation(route, mode, seed, signal_nodes, cfg))

    def avg(mode, key):
        return round(sum(r[key] for r in runs[mode]) / N_RUNS, 1)

    n_time, p_time = avg("normal", "journey_time_s"), avg("priority", "journey_time_s")
    n_wait, p_wait = avg("normal", "wait_time_s"), avg("priority", "wait_time_s")
    n_red, p_red = avg("normal", "red_stops"), avg("priority", "red_stops")

    st.subheader(f"{scenario_name} | {source} -> {hospital}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Normal journey (s)", n_time)
    m2.metric("Priority journey (s)", p_time)
    m3.metric("Time saved (s)", round(n_time - p_time, 1))
    m4.metric("Red stops avoided", round(n_red - p_red, 1))
    st.caption(f"Averages over {N_RUNS} runs with random signal timings. "
               f"Route: {len(route['path']) - 1} road segments.")

    result = pd.DataFrame({
        "Mode": ["Normal", "Priority"],
        "Journey time (s)": [n_time, p_time],
        "Wait time (s)": [n_wait, p_wait],
        "Red stops": [n_red, p_red],
    })
    left, right = st.columns(2)
    left.table(result.style.format({"Journey time (s)": "{:.1f}", "Wait time (s)": "{:.1f}", "Red stops": "{:.1f}"}))
    right.bar_chart(result.set_index("Mode")["Journey time (s)"])

    st.subheader("Route on map")
    st.map(pd.DataFrame({"lat": [G.nodes[n]["lat"] for n in route["path"]],
                         "lon": [G.nodes[n]["lon"] for n in route["path"]]}))

    st.subheader("Signal events (run 1, Normal mode)")
    events = runs["normal"][0]["events"]
    st.dataframe(pd.DataFrame(events) if events else
                 pd.DataFrame({"info": ["No signals on this route in this run"]}))

    st.download_button("Download results (CSV)", result.to_csv(index=False),
                       file_name=f"{scenario_name}_{trip_name}.csv", mime="text/csv")