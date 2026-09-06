"""PEKT-R package."""

from .config import PEKTRConfig

__all__ = ["PEKTRConfig", "PEKTRModel"]


def __getattr__(name: str):
    if name == "PEKTRModel":
        from .model import PEKTRModel

        return PEKTRModel
    raise AttributeError(name)

