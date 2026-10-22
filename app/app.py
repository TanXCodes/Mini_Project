import streamlit as st

SCENARIOS = ["midday_normal", "morning_peak", "evening_peak", "evening_peak_accident"]
TRIPS = {
    "Trip 1": (363775666, 3716332298),
    "Trip 2": (309593064, 7148693828),
    "Trip 3": (305091341, 2228225821),
}

st.set_page_config(page_title="AI Emergency Corridor System", layout="wide")
st.title("AI-Based Emergency Corridor System")
st.caption("Team Vortex | Normal vs Priority mode comparison (simulation)")


col1, col2 = st.columns(2)
scenario = col1.selectbox("Scenario", SCENARIOS)
trip_name = col2.selectbox("Trip", list(TRIPS))
source, hospital = TRIPS[trip_name]


if st.button("Calculate route", type="primary"):
    
    normal = {"journey_time_s": 330.3, "wait_time_s": 82.9, "red_stops": 3}
    priority = {"journey_time_s": 247.4, "wait_time_s": 0.0, "red_stops": 0}

    st.subheader(f"{scenario} | {source} -> {hospital}")
    st.table({
        "Mode": ["Normal", "Priority"],
        "Journey time (s)": [normal["journey_time_s"], priority["journey_time_s"]],
        "Wait time (s)": [normal["wait_time_s"], priority["wait_time_s"]],
        "Red stops": [normal["red_stops"], priority["red_stops"]],
    })