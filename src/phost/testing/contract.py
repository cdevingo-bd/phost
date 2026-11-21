"""Judge an implementation against its stage contract. The same check gates a leaderboard entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..registry import Registry, default
from ..session import Selection, Session
from ..stage import StageContract
from ..stages import newton_linear as nl

EVALUATORS: dict[str, Callable] = {nl.NAME: nl.evaluate}


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

    ``calls`` is a list of keyword dicts for ``host.drive`` (default: a few skip/active patterns).
    Checks: tolerances, inputs unchanged, inactive outputs untouched, outputs finite.
    """
    registry = registry or default
    contract: StageContract = registry.contract(selection.stage, registry.impl(selection.impl).stage_version)
    evaluate = EVALUATORS[contract.name]
    calls = calls or [dict(), dict(active=[1, 0] + [1] * (host.data.nworld - 2)), dict(skip_unchanged=True, skip_noflip=True)]
    verdict = Verdict(contract.name, selection.impl, True)
    worst: dict[str, float] = {}
    with Session(host, [selection], registry):
        for kwargs in calls:
            run = host.drive(contract.name, **kwargs)
            for name in ("h", "grad"):
                if not np.array_equal(run["unchanged"][name], run["after"][name]):
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
