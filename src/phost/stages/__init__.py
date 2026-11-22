"""Stage contracts that ship with phost, and the one place that lists them."""

from __future__ import annotations

from ..registry import Registry
from . import control_law, dof_passive_force, effective_solve, newton_linear, probe_state

MODULES = (newton_linear, effective_solve, dof_passive_force, control_law, probe_state)


def register_all(registry: Registry) -> None:
    """Register every shipped contract and its native (leave-it-alone) implementation."""
    for module in MODULES:
        module.register(registry)
