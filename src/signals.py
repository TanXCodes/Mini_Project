"""Owner: Tanay (Signal + Simulation).

Traffic signal with a fixed cycle (green then red) and an emergency
priority mode.

States (state machine):
    NORMAL          -> signal follows its normal green/red cycle
    PRIORITY_GREEN  -> ambulance is near, signal is forced green
    CROSSED         -> ambulance has passed, signal goes back to NORMAL
"""

NORMAL = "NORMAL"
PRIORITY_GREEN = "PRIORITY_GREEN"
CROSSED = "CROSSED"


class SignalController:
    def __init__(self, junction_id, cycle_s=90, green_s=40,
                 priority_timeout_s=30, offset_s=0):
        self.junction_id = junction_id
        self.cycle_s = cycle_s                  # full cycle length (green + red)
        self.green_s = green_s                  # green part of the cycle
        self.priority_timeout_s = priority_timeout_s
        self.offset_s = offset_s                # shifts the cycle so signals differ
        self.state = NORMAL
        self.priority_until = None              # time till which green is forced

    # ---- normal cycle -------------------------------------------------
    def is_green(self, t):
        """True if the signal is green at time t (seconds) in normal mode."""
        return (t + self.offset_s) % self.cycle_s < self.green_s

    def normal_wait(self, t):
        """Seconds an ambulance arriving at time t must wait in normal mode."""
        if self.is_green(t):
            return 0.0
        phase = (t + self.offset_s) % self.cycle_s
        return self.cycle_s - phase             # time left until red ends

    # ---- priority mode ------------------------------------------------
    def request_priority(self, t):
        """Ambulance is approaching: force green for a limited time."""
        self.state = PRIORITY_GREEN
        self.priority_until = t + self.priority_timeout_s

    def ambulance_crossed(self):
        """Ambulance passed the junction: go CROSSED, then back to NORMAL."""
        self.state = CROSSED
        self.priority_until = None
        self.state = NORMAL

    def wait_time(self, arrival_t, priority):
        """Seconds the ambulance waits at this junction.

        priority=False -> normal cycle, may have to wait for red to end.
        priority=True  -> priority green is requested, so no waiting.
        """
        if priority:
            self.request_priority(arrival_t)
            wait = 0.0
            self.ambulance_crossed()
            return wait
        return self.normal_wait(arrival_t)
