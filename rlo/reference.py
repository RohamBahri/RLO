"""Reference solutions: exact small DP, or optional feasible OR-Tools routes."""

import json
from pathlib import Path
import numpy as np
from .data import fingerprint
from .problem import distances, validate


def load(path, cases):
    """Load frozen references after checking every instance matches its seeded split."""
    from .data import instances as make_instances
    records = json.loads(Path(path).read_text())
    if len(records["lengths"]) < len(cases):
        raise ValueError("not enough reference entries")
    expected = make_instances(records["n"], records["split"], len(cases))
    if any(fingerprint(a) != fingerprint(b) for a, b in zip(cases, expected)):
        raise ValueError("reference does not match these instances")
    return records["lengths"][:len(cases)], records["method"]


def ortools(instance, seconds=10):
    """Feasible heuristic reference (never labeled exact). Requires pip install ortools."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    d = np.rint(distances(instance) * 1_000_000).astype(np.int64)
    demands = instance["demands"].astype(int)
    n = len(d)
    manager = pywrapcp.RoutingIndexManager(n, n - 1, 0)
    model = pywrapcp.RoutingModel(manager)
    cost_id = model.RegisterTransitCallback(lambda i, j: int(d[manager.IndexToNode(i), manager.IndexToNode(j)]))
    demand_id = model.RegisterUnaryTransitCallback(lambda i: int(demands[manager.IndexToNode(i)]))
    model.SetArcCostEvaluatorOfAllVehicles(cost_id)
    model.AddDimensionWithVehicleCapacity(demand_id, 0, [int(instance["capacity"])] * (n - 1), True, "load")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(seconds)
    solution = model.SolveWithParameters(params)
    if solution is None:
        raise ValueError("OR-Tools found no feasible route")
    tour = [0]
    for vehicle in range(n - 1):
        idx = model.Start(vehicle)
        route = []
        while not model.IsEnd(idx):
            idx = solution.Value(model.NextVar(idx))
            node = manager.IndexToNode(idx)
            if node:
                route.append(node)
        if route:
            tour.extend(route + [0])
    validate(instance, tour)
    return tour


def save(path, cases, tours, method, n, split):
    """Freeze references for a complete deterministic split (one tour per case)."""
    from .data import instances as make_instances
    if len(tours) != len(cases) or any(
        fingerprint(a) != fingerprint(b)
        for a, b in zip(cases, make_instances(n, split, len(cases)))
    ):
        raise ValueError("routes and cases must match the requested seeded split")
    costs = [validate(inst, route) for inst, route in zip(cases, tours)]
    Path(path).write_text(json.dumps({"method": method, "n": n, "split": split,
                                      "lengths": costs}, indent=2) + "\n")
