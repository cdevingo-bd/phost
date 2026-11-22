"""Built-in implementations of ``dof.passive_force``."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind, ParamSpec
from ..registry import Registry
from ..stages import dof_passive_force as stage

SPRING_DAMPER_ID = f"{stage.NAME}/spring-damper"
IMPLICIT = "implicit_damping"


def _reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """f = -k (q - q_ref) - c v on scalar DOFs, and the damping c it reports when asked to, in float64."""
    import numpy as np
    q, v = np.asarray(inputs["q"], np.float64), np.asarray(inputs["v"], np.float64)
    on = np.asarray(inputs["scalar"]).astype(bool)
    force = -params["k"] * (q - params["q_ref"]) - params["c"] * v
    damping = np.where(on, params["c"] if params["implicit"] else 0.0, 0.0) * np.ones_like(v)
    return dict(force=np.where(on, force, 0.0), damping=damping)


class _Numpy:
    def __init__(self, params):
        self.p = params

    def apply(self, system):
        import numpy as np
        p, on = self.p, system.scalar.astype(bool)
        law = -p["k"] * (system.q.astype(np.float64) - p["q_ref"]) - p["c"] * system.v.astype(np.float64)
        system.force[:] = np.where(on, law, 0.0).astype(system.force.dtype)
        if system.damping is not None and p["implicit"]:
            system.damping[:] = np.where(on, p["c"], 0.0).astype(system.damping.dtype)

    def metrics(self):
        return {}


class _Warp(_Numpy):
    def apply(self, system):
        import warp as wp
        from . import _warp
        p = self.p
        arguments = [system.q, system.v, system.scalar, p["k"], p["c"], p["q_ref"]]
        if system.damping is not None and p["implicit"]:
            wp.launch(_warp.spring_damper_implicit, dim=system.force.shape, inputs=arguments,
                      outputs=[system.force, system.damping], device=system.force.device)
        else:
            wp.launch(_warp.spring_damper, dim=system.force.shape, inputs=arguments, outputs=[system.force],
                      device=system.force.device)


def _advisory(facts):
    if facts.integrator in ("implicit", "implicitfast"):
        return ("the law is applied explicitly: the integrator's velocity Jacobian comes from the model's own damping, "
                "so a stiff velocity-dependent law may need a smaller timestep")
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
        id=SPRING_DAMPER_ID, stage=stage.NAME, stage_version=1, version="2", title="Linear spring-damper",
        kind=ImplKind.BUILTIN,
        doc=("f = -k (q - q_ref) - c v on every scalar DOF, added to the host's own passive force. By default it also "
             "reports the damping c so a host whose integrator treats joint damping implicitly does the same for this one."),
        params=(ParamSpec("k", "float", 0.0, "stiffness, N/m or N m/rad", min=0.0),
                ParamSpec("c", "float", 0.0, "damping, N s/m or N m s/rad", min=0.0),
                ParamSpec("q_ref", "float", 0.0, "rest position"),
                ParamSpec("implicit", "bool", True,
                          "report c so the host integrates it implicitly where it can; False applies it explicitly")),
        fit="An example law. A host whose joints already have springs and dampers gets these on top.",
        extensions_optional=(IMPLICIT,),
        applicable=_applicable, advisory=_advisory, factory=_factory, reference=_reference))
