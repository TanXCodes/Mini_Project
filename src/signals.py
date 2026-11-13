"""Owner: Tanay (Signal + Simulation).

Traffic signal with a fixed normal cycle and an emergency priority mode.

Priority state flow:
NORMAL -> PRIORITY_TRANSITION -> PRIORITY_GREEN
       -> CROSSED -> RECOVERY -> NORMAL
"""

NORMAL = "NORMAL"
PRIORITY_TRANSITION = "PRIORITY_TRANSITION"
PRIORITY_GREEN = "PRIORITY_GREEN"
CROSSED = "CROSSED"
RECOVERY = "RECOVERY"
PRIORITY_TIMEOUT = "PRIORITY_TIMEOUT"


class SignalController:
    def __init__(
        self,
        junction_id,
        cycle_s=90,
        green_s=40,
        priority_timeout_s=30,
        offset_s=0,
        transition_s=5,
        crossing_s=4,
        recovery_s=2,
    ):
        self.junction_id = junction_id

        # Normal signal cycle
        self.cycle_s = cycle_s
        self.green_s = green_s
        self.offset_s = offset_s

        # Emergency priority timings
        self.priority_timeout_s = priority_timeout_s
        self.transition_s = transition_s
        self.crossing_s = crossing_s
        self.recovery_s = recovery_s

        self.state = NORMAL
        self.priority_until = None

        # Priority timing and event tracking
        self.priority_duration_s = 0.0
        self.last_events = []

    # ---- normal cycle -------------------------------------------------

    def is_green(self, t):
        """Check whether the normal signal is green at time t."""
        return (t + self.offset_s) % self.cycle_s < self.green_s

    def normal_wait(self, t):
        """Calculate waiting time during the normal signal cycle."""
        if self.is_green(t):
            return 0.0

        phase = (t + self.offset_s) % self.cycle_s
        return self.cycle_s - phase

    # ---- priority mode ------------------------------------------------

    def request_priority(self, t):
        """Process an ambulance priority request through timed phases."""
        self.last_events = []
        self.priority_duration_s = 0.0

        # Phase 1: transition before priority green
        self.state = PRIORITY_TRANSITION
        self.last_events.append({
            "time": round(t, 1),
            "state": self.state,
            "duration_s": self.transition_s,
        })

        green_start = t + self.transition_s

        # Phase 2: activate priority green
        self.state = PRIORITY_GREEN
        self.priority_until = green_start + self.priority_timeout_s

        self.last_events.append({
            "time": round(green_start, 1),
            "state": self.state,
            "duration_s": min(
                self.crossing_s, self.priority_timeout_s
            ),
        })

        # Phase 3: ambulance crossing or priority timeout
        if self.crossing_s <= self.priority_timeout_s:
            crossing_duration = self.crossing_s
            crossing_end = green_start + crossing_duration

            self.state = CROSSED
            self.last_events.append({
                "time": round(crossing_end, 1),
                "state": self.state,
                "duration_s": 0.0,
            })
        else:
            crossing_duration = self.priority_timeout_s
            crossing_end = green_start + crossing_duration

            self.state = PRIORITY_TIMEOUT
            self.last_events.append({
                "time": round(crossing_end, 1),
                "state": self.state,
                "duration_s": 0.0,
            })

        # Phase 4: recovery after priority mode
        self.state = RECOVERY
        self.last_events.append({
            "time": round(crossing_end, 1),
            "state": self.state,
            "duration_s": self.recovery_s,
        })

        self.priority_duration_s = (
            self.transition_s
            + crossing_duration
            + self.recovery_s
        )

        # Return to normal operation
        self.state = NORMAL
        self.priority_until = None

        self.last_events.append({
            "time": round(crossing_end + self.recovery_s, 1),
            "state": NORMAL,
            "duration_s": 0.0,
        })

        return self.priority_duration_s

    def ambulance_crossed(self):
        """Compatibility method to reset the signal after crossing."""
        self.state = CROSSED
        self.priority_until = None
        self.state = NORMAL

    def wait_time(self, arrival_t, priority):
        """Return signal waiting time.

        Normal mode: may wait for the next green phase.
        Priority mode: requests emergency priority; normal signal waiting
        is zero. Priority transition/crossing/recovery time is tracked
        separately in priority_duration_s.
        """
        self.priority_duration_s = 0.0

        if priority:
            self.request_priority(arrival_t)
            return 0.0

        return self.normal_wait(arrival_t)
