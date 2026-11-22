"""How an implementation of a stage is judged: data that ships with the contract, read by ``check_impl``.

A contract says what a stage is (ports, metrics, tolerances). A judge says how to measure a run of it: which entries of a
port are defined, how to turn what the runtime was handed and returned into the contract's metrics, what a repeat of the
run must reproduce. Keeping this beside the contract, instead of inside ``check_impl``, is what lets a new stage be added
without editing the checker: ``check_impl`` has no stage names in it.

The judge holds callables, so it is not serialized (``Registry.describe`` shows contracts, which are data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .equation import Channels, Iteration, Pointwise, validate as validate_equation
from .stage import Port, StageContract

Call = Mapping[str, Any]          # one recorded runtime call: inputs (before), after (inputs again), outputs


@dataclass(frozen=True)
class Judge:
    """How ``check_impl`` judges one stage.

    ``evaluate(call, reference, effect, ctx)`` returns this call's metrics by name (the check keeps the worst of each).
    ``call`` is what the runtime was handed and returned, ``reference`` the float64 outputs for those inputs (when the
    stage has a reference to run), ``effect`` the host-side effect the driver measured (given for the last call of a
    drive only), and ``ctx`` has ``nv`` (the drive's size), ``run`` (the drive's whole record) and ``law`` (the reference
    run on other inputs, for stages that probe it).

    ``reference`` runs the float64 reference on a call's inputs. Leave it unset when ``evaluate`` measures against the
    contract directly (a residual) and when the contract says every implementation brings its own (``reference="impl"``:
    the check uses the implementation's).

    ``defined(port, array, run)`` returns the entries of a port the contract defines (the rest may be anything), so the
    "input unchanged" and "output finite" rules ignore what is undefined. Default: every entry.

    ``repeatable``: drive the same call twice and require identical outputs.

    ``calls``: the drives a check makes by default, as keyword dicts for ``host.drive`` (or a function of the host that
    returns them, when they depend on its size).

    ``driven``: the driver observes the host's hook itself and returns the whole record instead of the check recording the
    runtime's calls. Only the Newton solve needs it (its inputs are built per host call, with inactive worlds).
    """

    evaluate: Callable[[Call, Mapping[str, Any] | None, Mapping[str, Any] | None, Any], Mapping[str, float]]
    reference: Callable[[Mapping[str, Any], Any], Mapping[str, Any]] | None = None
    defined: Callable[[Port, Any, Mapping[str, Any]], Any] | None = None
    repeatable: bool = False
    driven: bool = False
    calls: "tuple[Mapping[str, Any], ...] | Callable[[Any], list]" = ({"seed": 0}, {"seed": 1}, {"seed": 2})

    def entries(self, port: Port, array, run: Mapping[str, Any]):
        return array if self.defined is None else self.defined(port, array, run)

    def default_calls(self, host) -> list:
        """The keyword dicts ``host.drive`` is called with when the caller gives none (a callable gets the host)."""
        return list(self.calls(host) if callable(self.calls) else self.calls)


def context(run: Mapping[str, Any], law=None) -> SimpleNamespace:
    """The ``ctx`` handed to ``Judge.evaluate``."""
    return SimpleNamespace(nv=run.get("nv"), run=run, law=law)


@dataclass(frozen=True)
class StageSpec:
    """Everything that defines a stage, host-neutral: its contract, how it is judged and, when a law for it can be typed, the
    form the text takes (``equation``)."""

    contract: StageContract
    judge: Judge
    doc: str = field(default="", compare=False)
    equation: "Pointwise | Channels | Iteration | None" = None   # how a law for this stage can be typed (phost.equation)

    def __post_init__(self):
        if self.equation is not None:
            validate_equation(self.equation, self.contract)

    @property
    def name(self) -> str:
        return self.contract.name
