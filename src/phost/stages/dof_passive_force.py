"""Stage ``dof.passive_force``: an additional passive force law on the scalar joints.

The host computes its own passive forces (springs, dampers, gravity compensation, fluid) and then
this stage adds ``force`` to the passive generalized force of every scalar-joint DOF (hinge and
slide). The law is the implementation's: a spring that stiffens, friction that depends on velocity,
a damper tied to time. It changes the physics, so it is judged against its own float64 law, not
against the host.

The force is applied explicitly. MuJoCo's Euler integrator treats the model's own joint damping
implicitly, so a velocity-dependent law is not equivalent to the same damping written into the
model: measured on a small arm at a 2 ms timestep, damping c/m = 0.75 gave a 0.1% difference in
displacement over 150 steps. Position-dependent laws (springs) match the native ones to float32
rounding. Compare a velocity-dependent law against a reference that is also explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..registry import Registry
from ..stage import MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import native_impl as _native, relative_error

NAME = "dof.passive_force"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Add a force law to the host's passive generalized force. force[world, dof] is added where scalar[dof] is 1 "
         "(a hinge or slide DOF, whose position is one number). Evaluated once per host passive-force computation, "
         "and applied explicitly (the host's own joint damping may be integrated implicitly)."),
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
    ),
    invariants=(
        "force is a function of the same world's q, v and time and of the implementation's parameters only.",
        "force is finite and is 0 where scalar is 0.",
        "No input is modified.",
        "The host's passive force after the call equals its own passive force plus force, on scalar DOFs, "
        "and equals its own passive force elsewhere.",
        "The same inputs give bitwise the same outputs.",
    ),
    metrics=(
        MetricSpec("max_law_error", "1", "lower", "gauge",
                   "max |force - reference force| / max |reference force|, reference in float64 from the implementation"),
        MetricSpec("max_effect_error", "1", "lower", "gauge",
                   "relative error of the host's passive-force change against force (float64)"),
        MetricSpec("calls", "count", "none", "counter", "times the law was evaluated"),
    ),
    tolerances=(
        Tolerance("max_law_error", 1e-5, "float32 evaluation of a float64 law"),
        Tolerance("max_effect_error", 1e-5, "the host added what the law returned"),
    ),
    reference="impl",
)


@dataclass
class DofForceSystem:
    q: Any
    v: Any
    time: Any
    scalar: Any
    force: Any


class DofForceLaw(Protocol):
    def apply(self, system: DofForceSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl() -> Any:
    return _native(NAME, CONTRACT.version, "No extra passive force",
                   "Leave the host's passive forces alone. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_contract(CONTRACT)
    registry.add_impl(native_impl())


def evaluate(call: Mapping[str, Any], reference: Mapping[str, Any], effect: Mapping[str, Any] | None) -> dict:
    """Metrics of one recorded call: the law against its reference, and the host's effect against the law."""
    force, scalar = call["outputs"]["force"], call["inputs"]["scalar"]
    out = dict(max_law_error=max(relative_error(force, reference["force"], scalar),
                                 relative_error(force[..., scalar == 0], 0.0 * force[..., scalar == 0])))
    if effect is not None:
        import numpy as np
        delta = np.asarray(effect["delta"], np.float64)
        expected = np.where(np.asarray(scalar).astype(bool), np.asarray(force, np.float64), 0.0)
        out["max_effect_error"] = relative_error(delta, expected)
    return out
