"""Implementations that ship with phost: small, readable laws that exercise each stage.

Each has a numpy backend and a Warp backend (chosen by ``host.backend``) and a float64 reference of
its own math. They are the examples a user-written law is compared against.
"""

from __future__ import annotations

from ..registry import Registry
from . import controllers, forces, probes


def register_builtin(registry: Registry) -> None:
    for module in (forces, controllers, probes):
        module.register(registry)
