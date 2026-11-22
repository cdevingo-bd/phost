"""Stage ``body.wrench``: an external force and torque on each body.

Before the host folds applied Cartesian forces into the generalized force, this stage writes the wrench on every body from
the bodies' pose and velocity and the time: wind, drag, thrust, a spring to a point. The law is the implementation's. It
replaces whatever the wrench held, so it changes the trajectory and is judged against its own float64 law, and against the
effect it has on the host: the host's smooth force must equal what MuJoCo (C) computes for the same wrench.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..judge import Judge, StageSpec
from ..registry import Registry
from ..stage import CALLS, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import native_impl as _native, relative_error

NAME = "body.wrench"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Write wrench[world, body] (force then torque, in the world frame, at the body's centre of mass) from the bodies' "
         "position and velocity and the time, once per host computation of the smooth force. Whatever the wrench held is "
         "discarded."),
    swap_class=SwapClass.MODEL_CHANGING,
    dims=("world", "body"),
    ports=(
        Port("xpos", Role.IN, "float32", ("world", "body", 3), doc="Body frame position in the world."),
        Port("cvel", Role.IN, "float32", ("world", "body", 6), doc="Body velocity about its centre of mass, rotation first."),
        Port("time", Role.IN, "float32", ("world",), doc="Simulation time of each world."),
        Port("wrench", Role.OUT, "float32", ("world", "body", 6),
             doc="The applied wrench, force then torque. Write every entry; the world body is ignored by the host."),
    ),
    invariants=(
        "wrench is a function of the same world's xpos, cvel and time and of the implementation's parameters only.",
        "wrench is finite.",
        "No input is modified.",
        "The host's smooth force equals what MuJoCo (C) computes for the same state and wrench.",
        "The same inputs give bitwise the same outputs.",
    ),
    metrics=(
        MetricSpec("max_law_error", "1", "lower", "gauge",
                   "max |wrench - reference wrench| / max |reference wrench|, reference in float64 from the implementation"),
        MetricSpec("max_effect_error", "1", "lower", "gauge",
                   "relative error of the host's smooth force against MuJoCo (C, float64) for the same state and wrench"),
        CALLS,
    ),
    tolerances=(
        Tolerance("max_law_error", 1e-5, "float32 evaluation of a float64 law"),
        Tolerance("max_effect_error", 1e-4, "the host applied the wrench it was given, with MuJoCo's conventions"),
    ),
    reference="impl",
    entry="apply",
)


@dataclass
class WrenchSystem:
    xpos: Any
    cvel: Any
    time: Any
    wrench: Any


class BodyWrench(Protocol):
    def apply(self, system: WrenchSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl() -> Any:
    return _native(NAME, CONTRACT.version, "No external wrench", "Apply no extra force. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_stage(SPEC)
    registry.add_impl(native_impl())


def evaluate(call: Mapping[str, Any], reference: Mapping[str, Any], effect: Mapping[str, Any] | None, ctx=None) -> dict:
    out = dict(max_law_error=relative_error(call["outputs"]["wrench"], reference["wrench"]))
    if effect is not None:
        out["max_effect_error"] = relative_error(effect["qfrc_smooth"], effect["expected_qfrc_smooth"])
    return out


SPEC = StageSpec(CONTRACT, Judge(evaluate=evaluate, repeatable=True))
