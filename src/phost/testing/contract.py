"""Judge an implementation against its stage contract. The same check gates a leaderboard entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..registry import Registry, default
from ..session import Selection, Session
from ..stage import Role, StageContract
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import newton_linear as nl
from ..stages import probe_state as ps

EVALUATORS: dict[str, Callable] = {nl.NAME: nl.evaluate}

# Stages judged by recording what the runtime was handed and what it returned (everything but the Newton solve,
# whose driver observes the host hook directly). stage name -> (runtime method, evaluate)
RECORDED: dict[str, tuple[str, Callable]] = {
    pf.NAME: ("apply", pf.evaluate),
    cl.NAME: ("apply", cl.evaluate),
    ps.NAME: ("observe", ps.evaluate),
}


@dataclass
class Verdict:
    stage: str
    impl: str
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


def _defined(name, array, nv):
    """The entries of an input the contract defines: upper triangle of h's leading block, first nv of grad."""
    array = np.asarray(array)
    if name == "h":
        rows, cols = np.triu_indices(nv)
        return array[:, rows, cols]
    return array[:, :nv]


def check_impl(host, selection: Selection, registry: Registry | None = None, *, calls=None) -> Verdict:
    """Install ``selection`` on ``host``, drive calls, and check every contract rule.

    ``calls`` is a list of keyword dicts for ``host.drive`` (default: a few skip/active patterns for the Newton
    solve, a few seeds for the other stages).
    Checks: tolerances, inputs unchanged, outputs finite, repeatable, plus the stage's own rules (inactive
    outputs untouched for the Newton solve; the host-side effect and the implementation's float64 law for the rest).
    """
    registry = registry or default
    contract: StageContract = registry.contract(selection.stage, registry.impl(selection.impl).stage_version)
    if contract.name in RECORDED:
        return _check_recorded(host, selection, registry, contract, calls)
    evaluate = EVALUATORS[contract.name]
    calls = calls or [dict(), dict(active=[1, 0] + [1] * (host.data.nworld - 2)), dict(skip_unchanged=True, skip_noflip=True)]
    verdict = Verdict(contract.name, selection.impl, True)
    worst: dict[str, float] = {}
    with Session(host, [selection], registry):
        for kwargs in calls:
            run = host.drive(contract.name, **kwargs)
            for name in ("h", "grad"):
                if not np.array_equal(_defined(name, run["unchanged"][name], run["nv"]),
                                      _defined(name, run["after"][name], run["nv"]), equal_nan=True):
                    verdict.failures.append(f"input {name!r} was modified ({kwargs})")
            out = run["outputs"]
            for w in range(run["active"].shape[0]):
                if run["active"][w]:
                    for name, value in out.items():
                        if not np.all(np.isfinite(np.asarray(value)[w])):
                            verdict.failures.append(f"output {name!r} not finite in world {w} ({kwargs})")
                else:
                    for name, value in out.items():
                        if not np.all(np.asarray(value)[w] == 123.0):
                            verdict.failures.append(f"output {name!r} written for inactive world {w} ({kwargs})")
            for name, value in evaluate(run["inputs"], out, run["nv"]).items():
                worst[name] = max(worst.get(name, 0.0), value)
    verdict.metrics = worst
    for gate in contract.tolerances:
        value = worst.get(gate.metric)
        if value is None or not value <= gate.max:
            verdict.failures.append(f"{gate.metric} = {value} exceeds {gate.max}")
    verdict.ok = not verdict.failures
    return verdict


# ---- stages judged by recording the runtime's calls -------------------------------------------------

def _array(value):
    """A numpy copy of a numpy or Warp array."""
    if hasattr(value, "numpy"):
        import warp as wp
        wp.synchronize()
        return value.numpy().copy()
    return np.array(value, copy=True)


class _Recorder:
    """Wraps one method of a runtime and keeps, for every call, its inputs before and after and its outputs."""

    def __init__(self, runtime, method: str, contract: StageContract):
        self.calls: list[dict] = []
        original = getattr(runtime, method)

        def wrapper(system):
            ins = [p.name for p in contract.ports if p.role is Role.IN]
            outs = [p.name for p in contract.ports if p.role is not Role.IN]
            before = {name: _array(getattr(system, name)) for name in ins}
            original(system)
            # an optional output the host did not hand over is None: there is nothing to record for it
            self.calls.append(dict(inputs=before, after={name: _array(getattr(system, name)) for name in ins},
                                   outputs={name: _array(getattr(system, name)) for name in outs
                                            if getattr(system, name, None) is not None}))

        setattr(runtime, method, wrapper)


def _check_recorded(host, selection, registry, contract, calls) -> Verdict:
    method, evaluate = RECORDED[contract.name]
    impl = registry.impl(selection.impl)
    params = impl.resolve_params(selection.params)
    calls = calls or [dict(seed=0), dict(seed=1), dict(seed=2)]
    verdict = Verdict(contract.name, selection.impl, True)
    if impl.reference is None:
        verdict.failures.append("the implementation supplies no float64 reference")
        verdict.ok = False
        return verdict
    worst: dict[str, float] = {}

    def law(inputs):
        """The implementation's float64 reference on ``inputs`` (floating point entries as float64)."""
        return impl.reference({k: np.asarray(v, np.float64) if np.asarray(v).dtype.kind == "f" else v
                               for k, v in inputs.items()}, params)

    with Session(host, [selection], registry) as session:
        runtime = session.runtimes[contract.name]
        recorder = _Recorder(runtime, method, contract)
        for kwargs in calls:
            recorder.calls.clear()
            run = host.drive(contract.name, **kwargs)
            first = list(recorder.calls)
            recorder.calls.clear()
            host.drive(contract.name, **kwargs)
            second = list(recorder.calls)
            if not first:
                verdict.failures.append(f"the runtime was never called ({kwargs})")
                continue
            if len(first) != len(second):
                verdict.failures.append(f"{len(first)} calls the first time, {len(second)} the second ({kwargs})")
            for index, call in enumerate(first):
                for name, value in call["inputs"].items():
                    if not np.array_equal(value, call["after"][name], equal_nan=True):
                        verdict.failures.append(f"input {name!r} was modified ({kwargs})")
                for name, value in call["outputs"].items():
                    if not np.all(np.isfinite(value)):
                        verdict.failures.append(f"output {name!r} is not finite ({kwargs})")
                    if index < len(second) and not np.array_equal(value, second[index]["outputs"][name], equal_nan=True):
                        verdict.failures.append(f"output {name!r} differs between identical calls ({kwargs})")
                is_last = index == len(first) - 1
                for name, value in evaluate(call, law(call["inputs"]), run.get("effect") if is_last else None, law).items():
                    worst[name] = max(worst.get(name, 0.0), value)
    verdict.metrics = worst
    for gate in contract.tolerances:
        value = worst.get(gate.metric)
        if value is None:
            verdict.failures.append(f"{gate.metric} was not measured")
        elif not value <= gate.max:
            verdict.failures.append(f"{gate.metric} = {value} exceeds {gate.max}")
    verdict.ok = not verdict.failures
    return verdict
