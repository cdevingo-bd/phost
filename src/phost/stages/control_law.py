"""Stage ``control.law``: a controller that sets the actuator commands.

Before the host turns commands into actuator forces, this stage writes ``ctrl`` from the actuators'
lengths and velocities and the time. A PD law, a feed-forward trajectory, a saturating command: the
law is the implementation's. It replaces whatever was in ``ctrl``, so it is a controller, not a
filter, and it changes the trajectory: it is judged against its own float64 law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..registry import Registry
from ..stage import CALLS, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import native_impl as _native, relative_error

NAME = "control.law"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Write ctrl[world, actuator] from the actuators' length and velocity and the time, once per host "
         "actuation computation. Whatever ctrl held before is discarded."),
    swap_class=SwapClass.MODEL_CHANGING,
    dims=("world", "nu"),
    ports=(
        Port("length", Role.IN, "float32", ("world", "nu"), doc="Actuator length (joint position for a joint actuator)."),
        Port("velocity", Role.IN, "float32", ("world", "nu"), doc="Actuator velocity."),
        Port("time", Role.IN, "float32", ("world",), doc="Simulation time of each world."),
        Port("ctrl", Role.OUT, "float32", ("world", "nu"), doc="The command the host uses. Write every entry."),
    ),
    invariants=(
        "ctrl is a function of the same world's length, velocity and time and of the implementation's parameters only.",
        "ctrl is finite.",
        "No input is modified.",
        "The host's actuator force equals what its gain and bias give for this ctrl.",
        "The same inputs give bitwise the same outputs.",
    ),
    metrics=(
        MetricSpec("max_law_error", "1", "lower", "gauge",
                   "max |ctrl - reference ctrl| / max |reference ctrl|, reference in float64 from the implementation"),
        MetricSpec("max_effect_error", "1", "lower", "gauge",
                   "relative error of the host's actuator force against its gain/bias applied to ctrl (float64)"),
        CALLS,
    ),
    tolerances=(
        Tolerance("max_law_error", 1e-5, "float32 evaluation of a float64 law"),
        Tolerance("max_effect_error", 1e-4, "the host used the command it was given"),
    ),
    reference="impl",
    entry="apply",
)


@dataclass
class ControlSystem:
    length: Any
    velocity: Any
    time: Any
    ctrl: Any


class ControlLaw(Protocol):
    def apply(self, system: ControlSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl() -> Any:
    return _native(NAME, CONTRACT.version, "Host's own ctrl", "Leave ctrl as the caller set it. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_contract(CONTRACT)
    registry.add_impl(native_impl())


def evaluate(call: Mapping[str, Any], reference: Mapping[str, Any], effect: Mapping[str, Any] | None, law=None) -> dict:
    out = dict(max_law_error=relative_error(call["outputs"]["ctrl"], reference["ctrl"]))
    if effect is not None:
        out["max_effect_error"] = relative_error(effect["actuator_force"], effect["expected_actuator_force"])
    return out
