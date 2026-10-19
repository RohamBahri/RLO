"""Auditable route-based reference solutions; every cost is independently checked."""

import json
from pathlib import Path
import numpy as np
from .data import fingerprint
from .problem import distances, validate


def load(path, cases):
    """Load frozen routes; reject archived cost-only files."""
    record = json.loads(Path(path).read_text())
    if "routes" not in record:
        raise ValueError("cost-only archive cannot be validated: regenerate and freeze routes")
    routes, ids, costs = (record[k] for k in ("routes", "fingerprints", "costs"))
    if not len(routes) == len(ids) == len(costs) or len(routes) < len(cases):
        raise ValueError("reference entry count mismatch")
    if any(fingerprint(inst) != ids[i] for i, inst in enumerate(cases)):
        raise ValueError("reference does not match these instances")
    measured = [validate(inst, routes[i]) for i, inst in enumerate(cases)]
    if not np.allclose(measured, costs[:len(cases)], atol=1e-9, rtol=0):
        raise ValueError("stored reference costs disagree with routes")
    return measured, record["method"]


def save(path, cases, routes, method, n, split, *, overwrite=False):
    """Freeze Euclidean reference costs together with actual feasible tours."""
    from .data import instances
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use overwrite=True to replace it")
    if len(routes) != len(cases) or any(
        fingerprint(a) != fingerprint(b)
        for a, b in zip(cases, instances(n, split, len(cases)))
    ):
        raise ValueError("routes and cases must match the seeded split")
    record = {"n": n, "split": split, "method": method,
              "fingerprints": [fingerprint(x) for x in cases],
              "routes": [list(map(int, tour)) for tour in routes],
              "costs": [validate(inst, tour) for inst, tour in zip(cases, routes)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def pyvrp_solver(instance, seconds=3, seed=0):
    """Optional PyVRP 0.14.x solver; integer costs for search, Euclidean for scoring."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n = len(instance["demands"]) - 1
    model = Model()
    model.add_depot(location=model.add_location(*map(float, instance["coords"][0])))
    model.add_vehicle_type(num_available=n, capacity=int(instance["capacity"]))
    for xy, demand in zip(instance["coords"][1:], instance["demands"][1:]):
        model.add_client(location=model.add_location(*map(float, xy)), delivery=int(demand))
    d = np.rint(distances(instance) * 1_000_000).astype(int)
    for i, frm in enumerate(model.locations):
        for j, to in enumerate(model.locations):
            model.add_edge(frm, to, distance=int(d[i, j]))
    result = model.solve(stop=MaxRuntime(seconds), seed=seed, display=False)
    if not result.is_feasible():
        raise ValueError("PyVRP did not find a feasible route")
    tour = [0]
    for route in result.best.routes():
        tour.extend(int(act.idx) + 1 for act in route if act.is_client())
        tour.append(0)
    validate(instance, tour)
    return tour


def ortools(instance, seconds=10):
    """Optional heuristic OR-Tools reference, with recomputed Euclidean cost."""
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
