from src.signals import (
    SignalController,
    NORMAL,
    PRIORITY_TRANSITION,
    PRIORITY_GREEN,
    CROSSED,
    RECOVERY,
)
from src.simulator import run_simulation


def test_green_means_no_wait():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)
    assert s.wait_time(10, priority=False) == 0


def test_red_means_wait_until_cycle_ends():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)
    assert s.wait_time(50, priority=False) == 40


def test_priority_has_zero_signal_wait_and_returns_to_normal():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)

    assert s.wait_time(50, priority=True) == 0
    assert s.state == NORMAL


def test_priority_overhead_is_11_seconds():
    s = SignalController(
        "J1",
        transition_s=5,
        crossing_s=4,
        recovery_s=2,
    )

    s.wait_time(50, priority=True)

    assert s.priority_duration_s == 11


def test_priority_event_sequence():
    s = SignalController("J1")

    s.wait_time(50, priority=True)

    states = [event["state"] for event in s.last_events]

    assert states == [
        PRIORITY_TRANSITION,
        PRIORITY_GREEN,
        CROSSED,
        RECOVERY,
        NORMAL,
    ]


def test_priority_event_durations():
    s = SignalController("J1")

    s.wait_time(50, priority=True)

    durations = [
        event["duration_s"] for event in s.last_events
    ]

    assert durations == [5, 4, 0, 2, 0]


def test_simulator_adds_priority_overhead():
    cfg = {
        "signals": {
            "cycle_s": 90,
            "green_s": 40,
            "priority_timeout_s": 30,
            "transition_s": 5,
            "crossing_s": 4,
            "recovery_s": 2,
            "extra_simulated_signals": 0,
        }
    }

    route = {
        "path": [1, 2, 3, 4],
        "segment_times_s": [60, 60, 60],
    }

    result = run_simulation(
        route, "priority", 7, {2, 3}, cfg
    )

    assert result["wait_time_s"] == 0
    assert result["priority_time_s"] == 22
    assert result["journey_time_s"] == 202


def test_simulator_logs_priority_phases():
    cfg = {
        "signals": {
            "cycle_s": 90,
            "green_s": 40,
            "priority_timeout_s": 30,
            "transition_s": 5,
            "crossing_s": 4,
            "recovery_s": 2,
            "extra_simulated_signals": 0,
        }
    }

    route = {
        "path": [1, 2, 3],
        "segment_times_s": [60, 60],
    }

    result = run_simulation(
        route, "priority", 7, {2}, cfg
    )

    states = [event["state"] for event in result["events"]]

    assert states == [
        PRIORITY_TRANSITION,
        PRIORITY_GREEN,
        CROSSED,
        RECOVERY,
        NORMAL,
    ]
