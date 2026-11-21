"""Stage ``probe.state``: read the state after forward dynamics and compute a measurement.

An observer. It reads positions, velocities and the accelerations the host just solved for, and
writes a vector of numbers per world into a buffer the host owns. It cannot change the simulation, so
a swap is judged on whether its numbers match its own float64 definition, and on what it costs. This
is how a metric typed by a user gets a live value.

Under integrators that call forward dynamics more than once per step (Runge-Kutta), the probe runs
at each call and the buffer holds the last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..registry import Registry
from ..stage import MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import native_impl as _native, relative_error

NAME = "probe.state"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("After the host's forward dynamics, compute value[world, width] from qpos, qvel, qacc and time. The runtime "
         "declares its width. Read-only: the simulation is not affected."),
    swap_class=SwapClass.OBSERVER,
    dims=("world", "nq", "nv", "width"),
    ports=(
        Port("qpos", Role.IN, "float32", ("world", "nq"), doc="Generalized positions."),
        Port("qvel", Role.IN, "float32", ("world", "nv"), doc="Generalized velocities."),
        Port("qacc", Role.IN, "float32", ("world", "nv"), doc="Accelerations the host just solved for."),
        Port("time", Role.IN, "float32", ("world",), doc="Simulation time of each world."),
        Port("value", Role.OUT, "float32", ("world", "width"), doc="The measurement. Host-owned; write every entry."),
    ),
    invariants=(
        "value is a function of the same world's inputs and the implementation's parameters only.",
        "value is finite.",
        "No input is modified.",
        "The same inputs give bitwise the same outputs.",
    ),
    metrics=(
        MetricSpec("max_law_error", "1", "lower", "gauge",
                   "max |value - reference value| / max |reference value|, reference in float64 from the implementation"),
        MetricSpec("traces", "count", "none", "counter", "times the runtime ran in Python: once per eager call, once per warmup or graph capture, not once per replay"),
    ),
    tolerances=(Tolerance("max_law_error", 1e-5, "float32 evaluation of a float64 definition"),),
    reference="impl",
)


@dataclass
class ProbeSystem:
    qpos: Any
    qvel: Any
    qacc: Any
    time: Any
    value: Any


class Probe(Protocol):
    width: int                                  # columns of ``value``; read once when the probe is installed

    def observe(self, system: ProbeSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl() -> Any:
    return _native(NAME, CONTRACT.version, "No probe", "Observe nothing. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_contract(CONTRACT)
    registry.add_impl(native_impl())


def evaluate(call: Mapping[str, Any], reference: Mapping[str, Any], effect: Mapping[str, Any] | None) -> dict:
    return dict(max_law_error=relative_error(call["outputs"]["value"], reference["value"]))
