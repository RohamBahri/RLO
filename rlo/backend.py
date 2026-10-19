"""Current training uses one PyTorch implementation (CUDA, MPS, or CPU).

MLX availability is not treated as MLX *training* support. Add a separate
verified backend only if the research benefit justifies the duplication.
"""


def choose():
    try:
        from .learning import device_auto
        return "torch:" + device_auto()
    except ImportError:
        return "numpy"
