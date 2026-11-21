"""Built-in implementations of ``control.law``."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind, ParamSpec
from ..registry import Registry
from ..stages import control_law as stage

PD_ID = f"{stage.NAME}/pd"


def _reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """ctrl = kp (target - length) - kd velocity, in float64."""
    import numpy as np
    length, velocity = np.asarray(inputs["length"], np.float64), np.asarray(inputs["velocity"], np.float64)
    return dict(ctrl=params["kp"] * (params["target"] - length) - params["kd"] * velocity)


class _Numpy:
    def __init__(self, params):
        self.p, self.calls = params, 0

    def apply(self, system):
        import numpy as np
        p = self.p
        law = p["kp"] * (p["target"] - system.length.astype(np.float64)) - p["kd"] * system.velocity.astype(np.float64)
        system.ctrl[:] = law.astype(system.ctrl.dtype)
        self.calls += 1

    def metrics(self):
        return dict(calls=self.calls)


class _Warp(_Numpy):
    def apply(self, system):
        import warp as wp
        from . import _warp
        p = self.p
        wp.launch(_warp.pd, dim=system.ctrl.shape, inputs=[system.length, system.velocity, p["kp"], p["kd"], p["target"]],
                  outputs=[system.ctrl], device=system.ctrl.device)
        self.calls += 1


def _advisory(facts):
    if facts.integrator in ("implicit", "implicitfast"):
        return f"{what} is not part of the implicit integrator's velocity Jacobian; a stiff law may need a smaller timestep"
    return None


def _applicable(facts):
    if facts.integrator == "discrete":
        return "the discrete integrator builds its own effective-force solve, which bypasses this stage"
    if facts.sleep:
        return "sleeping islands change the actuation path"
    if facts.extra.get("nu", 0) == 0:
        return "the model has no actuators"
    if facts.extra.get("nhistory", 0) > 0:
        return "the model delays ctrl through a history buffer, which this stage would bypass"
    return None


def _factory(host, params):
    return (_Warp if host.backend == "warp" else _Numpy)(params)


def register(registry: Registry) -> None:
    registry.add_impl(Impl(
        id=PD_ID, stage=stage.NAME, stage_version=1, version="1", title="PD to a constant target",
        kind=ImplKind.BUILTIN, doc="ctrl = kp (target - length) - kd velocity on every actuator.",
        params=(ParamSpec("kp", "float", 1.0, "proportional gain"), ParamSpec("kd", "float", 0.0, "derivative gain"),
                ParamSpec("target", "float", 0.0, "target actuator length")),
        fit="An example law: one target for every actuator. A real controller reads a per-actuator target.",
        applicable=_applicable, advisory=_advisory, factory=_factory, reference=_reference))
