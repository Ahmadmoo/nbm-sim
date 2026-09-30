"""Camera-only simulation environment for Next Best Motion in event-based 3D reconstruction.

Isaac-dependent modules (``camera``, ``environment`` with the default backend) import Isaac
lazily, so the pure modules work in any Python environment.
"""
from .config import SimConfig

__all__ = ["SimConfig", "CameraNBMEnv"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "CameraNBMEnv":
        from .environment import CameraNBMEnv
        return CameraNBMEnv
    raise AttributeError(name)
