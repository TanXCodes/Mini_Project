"""Quick check: run each demo trip in Normal and Priority mode."""
from src.routing import compute_route, get_graph
from src.simulator import run_simulation, load_config, load_signal_nodes

cfg = load_config()
paths = cfg["paths"]
signal_nodes = load_signal_nodes(paths["nodes"], paths["edges"],
                                 cfg["signals"]["extra_simulated_signals"])
G = get_graph()

for trip in cfg["trips"]:
    route = compute_route(trip["source"], trip["hospital"], G=G)
    normal = run_simulation(route, "normal", cfg["seed"], signal_nodes, cfg)
    prior = run_simulation(route, "priority", cfg["seed"], signal_nodes, cfg)
    print(f"Trip {trip['source']} -> {trip['hospital']}")
    print(f"  Normal  : {normal['journey_time_s']} s, wait {normal['wait_time_s']} s, red stops {normal['red_stops']}")
    print(f"  Priority: {prior['journey_time_s']} s, wait {prior['wait_time_s']} s, red stops {prior['red_stops']}")
    saved = round(normal['journey_time_s'] - prior['journey_time_s'], 1)
    print(f"  Time saved: {saved} s")
    assert saved >= 0, "Priority must never be slower than Normal"
