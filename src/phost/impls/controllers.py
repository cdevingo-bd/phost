"""Built-in implementations of ``control.law``."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind, ParamSpec
from ..registry import Registry
from ..stages import control_law as stage
from ..stages.control_law import advisory as _advisory, applicable as _applicable

PD_ID = f"{stage.NAME}/pd"


def _reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """ctrl = kp (target - length) - kd velocity, in float64."""
    import numpy as np
    length, velocity = np.asarray(inputs["length"], np.float64), np.asarray(inputs["velocity"], np.float64)
    return dict(ctrl=params["kp"] * (params["target"] - length) - params["kd"] * velocity)


class _Numpy:
    def __init__(self, params):
        self.p = params

    def apply(self, system):
        import numpy as np
        p = self.p
        law = p["kp"] * (p["target"] - system.length.astype(np.float64)) - p["kd"] * system.velocity.astype(np.float64)
        system.ctrl[:] = law.astype(system.ctrl.dtype)

    def metrics(self):
        return {}


class _Jax(_Numpy):
    def apply(self, system):
        from ..ports import unwrap
        p = self.p
        system.ctrl[...] = p["kp"] * (p["target"] - unwrap(system.length)) - p["kd"] * unwrap(system.velocity)


class _Warp(_Numpy):
    def apply(self, system):
        import warp as wp
        from . import _warp
        p = self.p
        wp.launch(_warp.pd, dim=system.ctrl.shape, inputs=[system.length, system.velocity, p["kp"], p["kd"], p["target"]],
                  outputs=[system.ctrl], device=system.ctrl.device)


def _factory(host, params):
    return {"warp": _Warp, "jax": _Jax}.get(host.backend, _Numpy)(params)


def register(registry: Registry) -> None:
    registry.add_impl(Impl(
        id=PD_ID, stage=stage.NAME, stage_version=1, version="1", title="PD to a constant target",
        kind=ImplKind.BUILTIN, doc="ctrl = kp (target - length) - kd velocity on every actuator.",
        params=(ParamSpec("kp", "float", 1.0, "proportional gain"), ParamSpec("kd", "float", 0.0, "derivative gain"),
                ParamSpec("target", "float", 0.0, "target actuator length")),
        fit="An example law: one target for every actuator. A real controller reads a per-actuator target.",
        applicable=_applicable, advisory=_advisory, factory=_factory, reference=_reference))
