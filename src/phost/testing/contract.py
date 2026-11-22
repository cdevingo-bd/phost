"""Judge an implementation against its stage contract. The same check gates a leaderboard entry.

This module names no stage. What a stage's ports mean, which entries of them are defined and how a run becomes the
contract's metrics all come from the stage's ``Judge`` (``phost.judge``), registered with the contract, so adding a stage
does not edit the checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..judge import Judge, context
from ..registry import Registry, default
from ..session import Selection, Session
from ..stage import Role, StageContract

SENTINEL = 123.0                  # what a driver fills an output with before the call, to see which worlds were written


@dataclass
class Verdict:
    stage: str
    impl: str
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


def check_impl(host, selection: Selection, registry: Registry | None = None, *, calls=None) -> Verdict:
    """Install ``selection`` on ``host``, drive calls, and check every contract rule.

    ``calls`` is a list of keyword dicts for ``host.drive`` (default: the stage's own, from its judge).
    Checks, for every stage: the contract's tolerances, inputs unchanged, outputs finite, the runtime was called, and,
    when the judge says so, that identical calls give identical outputs. What else is measured (the float64 law, the
    host's own computation of the same thing) is the judge's ``evaluate``.
    """
    registry = registry or default
    contract: StageContract = registry.contract(selection.stage, registry.impl(selection.impl).stage_version)
    judge = registry.judge(contract.name, contract.version)
    calls = calls or judge.default_calls(host)
    check = _check_driven if judge.driven else _check_recorded
    verdict = check(host, selection, registry, contract, judge, calls)
    for gate in contract.tolerances:
        value = verdict.metrics.get(gate.metric)
        if value is None:
            verdict.failures.append(f"{gate.metric} was not measured")
        elif not value <= gate.max:
            verdict.failures.append(f"{gate.metric} = {value} exceeds {gate.max}")
    verdict.ok = not verdict.failures
    return verdict


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


def _worst(into: dict, metrics) -> None:
    for name, value in metrics.items():
        into[name] = max(into.get(name, 0.0), value)


def _check_driven(host, selection, registry, contract, judge: Judge, calls) -> Verdict:
    """Stages whose driver observes the host's hook and returns the whole record (inputs built per host call, with worlds
    that are inactive and whose outputs must be left alone)."""
    ports = {p.name: p for p in contract.ports}
    verdict = Verdict(contract.name, selection.impl, True)
    worst: dict[str, float] = {}
    with Session(host, [selection], registry):
        for kwargs in calls:
            run = host.drive(contract.name, **kwargs)
            for name, before in run["unchanged"].items():
                if not np.array_equal(np.asarray(judge.entries(ports[name], before, run)),
                                      np.asarray(judge.entries(ports[name], run["after"][name], run)), equal_nan=True):
                    verdict.failures.append(f"input {name!r} was modified ({kwargs})")
            out = run["outputs"]
            for w in range(run["active"].shape[0]):
                for name, value in out.items():
                    value = np.asarray(value)[w]
                    if run["active"][w]:
                        if not np.all(np.isfinite(value)):
                            verdict.failures.append(f"output {name!r} not finite in world {w} ({kwargs})")
                    elif not np.all(value == SENTINEL):
                        verdict.failures.append(f"output {name!r} written for inactive world {w} ({kwargs})")
            _worst(worst, judge.evaluate(dict(inputs=run["inputs"], outputs=out), None, None, context(run)))
    verdict.metrics = worst
    return verdict


def _check_recorded(host, selection, registry, contract, judge: Judge, calls) -> Verdict:
    """Record what the runtime is handed and returns while the host runs, then judge each call."""
    impl = registry.impl(selection.impl)
    params = impl.resolve_params(selection.params)
    ports = {p.name: p for p in contract.ports}
    verdict = Verdict(contract.name, selection.impl, True)
    worst: dict[str, float] = {}
    if contract.reference == "impl" and impl.reference is None:
        verdict.failures.append("the implementation supplies no float64 reference")
        verdict.ok = False
        return verdict

    def floats(inputs):
        return {k: np.asarray(v, np.float64) if np.asarray(v).dtype.kind == "f" else v for k, v in inputs.items()}

    with Session(host, [selection], registry) as session:
        runtime = session.runtimes[contract.name]
        recorder = _Recorder(runtime, contract.entry, contract)
        for kwargs in calls:
            recorder.calls.clear()
            run = host.drive(contract.name, **kwargs)
            first = list(recorder.calls)
            second: list[dict] = []
            if judge.repeatable:
                recorder.calls.clear()
                host.drive(contract.name, **kwargs)
                second = list(recorder.calls)
            if not first:
                verdict.failures.append(f"the runtime was never called ({kwargs})")
                continue
            if judge.repeatable and len(first) != len(second):
                verdict.failures.append(f"{len(first)} calls the first time, {len(second)} the second ({kwargs})")
            ctx = context(run)
            if contract.reference == "impl":
                ctx.law = lambda inputs: impl.reference(floats(inputs), params)
            elif judge.reference is not None:
                ctx.law = lambda inputs: judge.reference(floats(inputs), ctx)
            for index, call in enumerate(first):
                for name, value in call["inputs"].items():
                    if not np.array_equal(np.asarray(judge.entries(ports[name], value, run)),
                                          np.asarray(judge.entries(ports[name], call["after"][name], run)), equal_nan=True):
                        verdict.failures.append(f"input {name!r} was modified ({kwargs})")
                for name, value in call["outputs"].items():
                    shown = np.asarray(judge.entries(ports[name], value, run))
                    if not np.all(np.isfinite(shown)):
                        verdict.failures.append(f"output {name!r} is not finite ({kwargs})")
                    if index < len(second) and not np.array_equal(
                            shown, np.asarray(judge.entries(ports[name], second[index]["outputs"][name], run)), equal_nan=True):
                        verdict.failures.append(f"output {name!r} differs between identical calls ({kwargs})")
                reference = ctx.law(call["inputs"]) if ctx.law is not None else None
                effect = run.get("effect") if index == len(first) - 1 else None
                _worst(worst, judge.evaluate(call, reference, effect, ctx))
    verdict.metrics = worst
    return verdict
