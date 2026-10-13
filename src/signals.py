"""Owner: Tanay (Signal + Simulation). Signal state machine."""

NORMAL = "NORMAL"
PRIORITY_GREEN = "PRIORITY_GREEN"
CROSSED = "CROSSED"


class SignalController:
    def __init__(self, junction_id, cycle_s, green_s, priority_timeout_s):
        self.junction_id = junction_id
        self.state = NORMAL

    def request_priority(self, t):
        raise NotImplementedError

    def wait_time(self, arrival_t, priority):
        """seconds the ambulance waits at this junction"""
        raise NotImplementedError
