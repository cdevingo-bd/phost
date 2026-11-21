"""Built-in implementations of ``dof.passive_force``."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind, ParamSpec
from ..registry import Registry
from ..stages import dof_passive_force as stage

SPRING_DAMPER_ID = f"{stage.NAME}/spring-damper"


def _reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """f = -k (q - q_ref) - c v on scalar DOFs, in float64."""
    import numpy as np
    q, v = np.asarray(inputs["q"], np.float64), np.asarray(inputs["v"], np.float64)
    force = -params["k"] * (q - params["q_ref"]) - params["c"] * v
    return dict(force=np.where(np.asarray(inputs["scalar"]).astype(bool), force, 0.0))


class _Numpy:
    def __init__(self, params):
        self.p, self.calls = params, 0

    def apply(self, system):
        import numpy as np
        p = self.p
        law = -p["k"] * (system.q.astype(np.float64) - p["q_ref"]) - p["c"] * system.v.astype(np.float64)
        system.force[:] = np.where(system.scalar.astype(bool), law, 0.0).astype(system.force.dtype)
        self.calls += 1

    def metrics(self):
        return dict(calls=self.calls)


class _Warp(_Numpy):
    def apply(self, system):
        import warp as wp
        from . import _warp
        p = self.p
        wp.launch(_warp.spring_damper, dim=system.force.shape,
                  inputs=[system.q, system.v, system.scalar, p["k"], p["c"], p["q_ref"]], outputs=[system.force],
                  device=system.force.device)
        self.calls += 1


def _advisory(facts):
    if facts.integrator in ("implicit", "implicitfast"):
        return f"{what} is not part of the implicit integrator's velocity Jacobian; a stiff law may need a smaller timestep"
    if facts.integrator == "euler":
        return ("the law is applied explicitly; the host's own joint damping is integrated implicitly, so damping written "
                "here differs from the same damping in the model by O(timestep)")
    return None


def _applicable(facts):
    if facts.integrator == "discrete":
        return "the discrete integrator builds its own effective-force solve, which bypasses this stage"
    if facts.sleep:
        return "sleeping islands change the passive-force path"
    if facts.extra.get("scalar_dofs", 0) == 0:
        return "the model has no hinge or slide DOF"
    return None


def _factory(host, params):
    return (_Warp if host.backend == "warp" else _Numpy)(params)


def register(registry: Registry) -> None:
    registry.add_impl(Impl(
        id=SPRING_DAMPER_ID, stage=stage.NAME, stage_version=1, version="1", title="Linear spring-damper",
        kind=ImplKind.BUILTIN,
        doc="f = -k (q - q_ref) - c v on every scalar DOF, added to the host's own passive force.",
        params=(ParamSpec("k", "float", 0.0, "stiffness, N/m or N m/rad", min=0.0),
                ParamSpec("c", "float", 0.0, "damping, N s/m or N m s/rad", min=0.0),
                ParamSpec("q_ref", "float", 0.0, "rest position")),
        fit="An example law. A host whose joints already have springs and dampers gets these on top.",
        applicable=_applicable, advisory=_advisory, factory=_factory, reference=_reference))
