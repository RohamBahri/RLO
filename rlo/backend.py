"""Tiny optional backend selector for future model code; core CVRP is NumPy-only."""

import sys


def choose(name="auto"):
    if name == "auto":
        if sys.platform == "darwin":
            try:
                import mlx.core  # noqa: F401
                return "mlx"
            except ImportError:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                return "torch"
        except ImportError:
            pass
        return "numpy"
    if name == "mlx":
        import mlx.core  # noqa: F401
    elif name == "torch":
        import torch  # noqa: F401
    elif name != "numpy":
        raise ValueError("backend must be auto, mlx, torch, or numpy")
    return name
