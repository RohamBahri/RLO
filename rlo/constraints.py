"""Capacity is the sole constraint in the initial, unlimited-fleet CVRP."""

import numpy as np


class Capacity:
    def __init__(self, instance):
        self.demands = instance["demands"]
        self.limit = instance["capacity"]

    def legal(self, visited, load):
        return (~visited) & (load + self.demands <= self.limit)

    def update(self, load, node):
        return 0 if node == 0 else load + float(self.demands[node])

    def features(self, load):
        return np.array([load / self.limit], dtype=np.float32)
