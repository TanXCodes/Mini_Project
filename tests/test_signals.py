from src.signals import SignalController, NORMAL
from src.simulator import run_simulation


def test_green_means_no_wait():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)
    assert s.wait_time(10, priority=False) == 0          # t=10 is inside green


def test_red_means_wait_until_cycle_ends():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)
    assert s.wait_time(50, priority=False) == 40         # red until t=90


def test_priority_never_waits_and_returns_to_normal():
    s = SignalController("J1", cycle_s=90, green_s=40, offset_s=0)
    assert s.wait_time(50, priority=True) == 0
    assert s.state == NORMAL


def test_priority_is_never_slower_than_normal():
    cfg = {"signals": {"cycle_s": 90, "green_s": 40, "priority_timeout_s": 30,
                       "extra_simulated_signals": 0}}
    route = {"path": [1, 2, 3, 4], "segment_times_s": [60, 60, 60]}
    sig = {2, 3}
    a = run_simulation(route, "normal", 7, sig, cfg)
    b = run_simulation(route, "priority", 7, sig, cfg)
    assert b["journey_time_s"] <= a["journey_time_s"]
    assert b["wait_time_s"] == 0
