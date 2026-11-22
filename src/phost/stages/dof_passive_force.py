"""Stage ``dof.passive_force``: an additional passive force law on the scalar joints.

The host computes its own passive forces (springs, dampers, gravity compensation, fluid) and then
this stage adds ``force`` to the passive generalized force of every scalar-joint DOF (hinge and
slide). The law is the implementation's: a spring that stiffens, friction that depends on velocity,
a damper tied to time. It changes the physics, so it is judged against its own float64 law, not
against the host.

A force that depends on velocity changes how the host integrates. MuJoCo's Euler integrator treats the model's own
joint damping implicitly, which is what keeps a stiff damper stable and is part of what a damper means in that host. A
force added to the passive force is explicit, so a damping law differs from the same damping written into the model by
O(timestep): measured on a small arm at a 2 ms timestep, c/m = 0.75 gave a 0.1% difference in displacement over 150
steps. An implementation closes the gap by also reporting ``damping``, the diagonal of -d force / d v. A host that
offers the ``implicit_damping`` extension adds it to the damping its integrator treats implicitly, so the law then
behaves as the model's own damper does. Position-dependent laws (springs) have no such gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..judge import Judge, StageSpec
from ..registry import Registry
from ..stage import CALLS, Extension, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import native_impl as _native, relative_error

NAME = "dof.passive_force"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Add a force law to the host's passive generalized force. force[world, dof] is added where scalar[dof] is 1 "
         "(a hinge or slide DOF, whose position is one number). Evaluated once per host passive-force computation. "
         "The force is explicit; the optional damping term lets the host integrate the velocity dependence implicitly "
         "where its integrator does so for its own damping."),
    swap_class=SwapClass.MODEL_CHANGING,
    dims=("world", "nv"),
    ports=(
        Port("q", Role.IN, "float32", ("world", "nv"),
             doc="Joint position of each scalar DOF (radians or metres); 0 for DOFs that are not scalar."),
        Port("v", Role.IN, "float32", ("world", "nv"), doc="Joint velocity (qvel)."),
        Port("time", Role.IN, "float32", ("world",), doc="Simulation time of each world."),
        Port("scalar", Role.IN, "int32", ("nv",), doc="1 where the DOF is a scalar joint coordinate (hinge, slide)."),
        Port("force", Role.OUT, "float32", ("world", "nv"),
             doc="Force to add. The host zeroes it before each call; write only where scalar is 1."),
        Port("damping", Role.OUT, "float32", ("world", "nv"),
             doc="Optional. -d force / d v of the same call, on scalar DOFs (>= 0 for a dissipative law). The host zeroes "
                 "it before each call and passes it only when the implementation uses the implicit_damping extension "
                 "(None otherwise). Leaving it zero is always valid: the law is then applied explicitly."),
    ),
    invariants=(
        "force is a function of the same world's q, v and time and of the implementation's parameters only.",
        "force is finite and is 0 where scalar is 0.",
        "No input is modified.",
        "The host's passive force after the call equals its own passive force plus force, on scalar DOFs, "
        "and equals its own passive force elsewhere.",
        "Each damping entry is 0 (that DOF is applied explicitly) or equals -d force / d v of the same call for that DOF "
        "on its own (the diagonal of the velocity Jacobian), and is 0 where scalar is 0.",
        "The same inputs give bitwise the same outputs.",
    ),
    metrics=(
        MetricSpec("max_law_error", "1", "lower", "gauge",
                   "max |force - reference force| / max |reference force|, reference in float64 from the implementation"),
        MetricSpec("max_effect_error", "1", "lower", "gauge",
                   "relative error of the host's passive-force change against force (float64)"),
        MetricSpec("max_damping_error", "1", "lower", "gauge",
                   "max |damping + d force / d v| / max |d force / d v| over the nonzero entries, the derivative taken by "
                   "central differences of the implementation's float64 reference, DOF by DOF; 0 when none is reported"),
        CALLS,
    ),
    tolerances=(
        Tolerance("max_law_error", 1e-5, "float32 evaluation of a float64 law"),
        Tolerance("max_effect_error", 1e-5, "the host added what the law returned"),
        Tolerance("max_damping_error", 1e-4, "float32 damping against a float64 central difference"),
    ),
    extensions=(
        Extension("implicit_damping",
                  "The host adds the reported damping to the damping its integrator treats implicitly, for the "
                  "step in which it was reported. A host whose integrator has no implicit damping (an explicit one) "
                  "needs no such change and does not offer it."),
    ),
    reference="impl",
    entry="apply",
)


@dataclass
class DofForceSystem:
    q: Any
    v: Any
    time: Any
    scalar: Any
    force: Any
    damping: Any = None          # None when the host does not take it (the implicit_damping extension was not used)


class DofForceLaw(Protocol):
    def apply(self, system: DofForceSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl() -> Any:
    return _native(NAME, CONTRACT.version, "No extra passive force",
                   "Leave the host's passive forces alone. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_stage(SPEC)
    registry.add_impl(native_impl())


def _damping_error(damping, scalar, inputs, law) -> float:
    """Error of the reported damping against -d force / d v from central differences of the float64 law.

    An entry is either 0 (that DOF is left explicit, always valid) or the derivative.

    Each DOF is perturbed alone, so a law that couples DOFs is judged on the diagonal of its Jacobian, which is what
    the host needs. At most 16 DOFs are checked, spread over the range: a sample, not a proof for the rest.
    """
    import numpy as np
    v, reported = np.asarray(inputs["v"], np.float64), np.asarray(damping, np.float64)
    on = np.asarray(scalar).astype(bool)
    scalar_dofs = np.nonzero(on)[0]
    sample = scalar_dofs[np.unique(np.linspace(0, len(scalar_dofs) - 1, min(16, len(scalar_dofs))).astype(int))]
    derivative = np.zeros_like(v)
    for j in sample:
        step = 1e-6 * (1.0 + np.abs(v[:, j]))
        ahead, behind = v.copy(), v.copy()
        ahead[:, j] += step
        behind[:, j] -= step
        derivative[:, j] = -(law(dict(inputs, v=ahead))["force"][:, j] - law(dict(inputs, v=behind))["force"][:, j]) / (2 * step)
    checked = np.zeros_like(on)
    checked[sample] = True
    claimed = np.where(reported == 0.0, derivative, reported)
    return max(relative_error(claimed, derivative, checked), relative_error(reported[..., ~on], 0.0 * reported[..., ~on]))


def evaluate(call: Mapping[str, Any], reference: Mapping[str, Any], effect: Mapping[str, Any] | None, ctx=None) -> dict:
    """Metrics of one recorded call: the law against its reference, the host's effect against the law, and the reported
    damping against the law's own velocity derivative. ``ctx.law(inputs)`` runs the float64 reference."""
    import numpy as np
    force, scalar = call["outputs"]["force"], call["inputs"]["scalar"]
    out = dict(max_law_error=max(relative_error(force, reference["force"], scalar),
                                 relative_error(force[..., scalar == 0], 0.0 * force[..., scalar == 0])))
    if effect is not None:
        delta = np.asarray(effect["delta"], np.float64)
        expected = np.where(np.asarray(scalar).astype(bool), np.asarray(force, np.float64), 0.0)
        out["max_effect_error"] = relative_error(delta, expected)
    damping = call["outputs"].get("damping")
    law = getattr(ctx, "law", None)
    out["max_damping_error"] = 0.0 if damping is None or law is None else _damping_error(damping, scalar, call["inputs"], law)
    return out


SPEC = StageSpec(CONTRACT, Judge(evaluate=evaluate, repeatable=True))
