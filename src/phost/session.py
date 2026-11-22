"""Apply a selection of implementations to a host for a scope, and read their metrics."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import CapabilityError, ContractError, GuardError
from .host import Capability, Host, Provision, StageBinding
from .impl import ImplKind
from .meter import Metered
from .patch import install
from .registry import Registry, default


def _unvouched(binding: StageBinding, provision: Provision) -> str | None:
    """Why the host cannot vouch for this extension on this build (a guard that does not pass), or None."""
    try:
        if provision.guard:
            binding.point.guards[provision.guard].check()
        if provision.point is not None:
            provision.point.guards["target"].check()
    except GuardError as error:
        return str(error)
    return None


@dataclass(frozen=True)
class Selection:
    """Choose an implementation (by id) for a stage, with parameters."""

    stage: str
    impl: str
    params: Mapping[str, Any] = field(default_factory=dict)


class Session:
    """``with Session(host, [Selection(...)]) as s:`` installs the choices; ``s.metrics()`` reads them.

    Warm up and capture graphs inside the ``with`` block. A selection of a native implementation
    installs nothing.
    """

    def __init__(self, host: Host, selections, registry: Registry | None = None):
        self.host = host
        self.registry = registry or default
        self.selections = [s if isinstance(s, Selection) else Selection(*s) for s in selections]
        seen = [s.stage for s in self.selections]
        if len(set(seen)) != len(seen):
            raise ContractError(f"stage selected twice: {sorted(x for x in seen if seen.count(x) > 1)}")
        self.runtimes: dict[str, Any] = {}
        self.meters: dict[str, Any] = {}
        self.advisories: list[str] = []
        self._stack: ExitStack | None = None

    def __enter__(self) -> "Session":
        stack = ExitStack()
        try:
            facts = self.host.facts()
            for selection in self.selections:
                self._attach(selection, facts, stack)
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def _attach(self, selection: Selection, facts, stack: ExitStack) -> None:
        impl = self.registry.impl(selection.impl)
        if impl.stage != selection.stage:
            raise ContractError(f"impl {impl.id!r} is for stage {impl.stage!r}, not {selection.stage!r}")
        params = impl.resolve_params(selection.params)
        impl.check_applicable(facts)
        note = impl.advise(facts)
        if note:
            self.advisories.append(f"{impl.id}: {note}")
        if impl.kind is ImplKind.NATIVE:
            return
        if Capability.PATCH not in self.host.capabilities:
            raise CapabilityError(f"host {self.host.info.name!r} cannot patch stages")
        binding = self.host.stage(selection.stage)
        if binding.contract.version != impl.stage_version:
            raise ContractError(
                f"impl {impl.id!r} targets {impl.stage}@{impl.stage_version}, host offers {binding.contract.key}")
        provided = self._extensions(impl, binding)
        runtime = impl.factory(self.host, params)
        self.runtimes[selection.stage] = runtime
        seen = runtime                                          # what the host's dispatch calls
        if binding.contract.entry:
            meter = self.meters[selection.stage] = self.host.meter()
            seen = Metered(runtime, binding.contract.entry, meter)
        allowed = frozenset(provided)
        stack.enter_context(install(
            binding.point,
            lambda original, b=binding, r=seen, a=allowed: b.make_dispatch(original, r, self.host, a),
            require=[p.guard for p in provided.values() if p.guard]))
        for provision in provided.values():
            if provision.point is not None:
                stack.enter_context(install(
                    provision.point, lambda original, p=provision: p.wrap(original, self.host)))

    def _extensions(self, impl, binding) -> dict[str, Provision]:
        """The extensions this implementation gets: all it needs (refused when the host lacks one), and those it
        would like that the host offers and can vouch for. An optional one that is missing is recorded as an advisory."""
        chosen: dict[str, Provision] = {}
        for name in impl.extensions_used:
            if name not in binding.extensions:
                raise CapabilityError(
                    f"host {self.host.info.name!r} offers no extension {name!r} for {impl.stage}; "
                    f"it offers {sorted(binding.extensions)}")
            chosen[name] = binding.extensions[name]
        for name in impl.extensions_optional:
            provision = binding.extensions.get(name)
            problem = f"the host offers no extension {name!r}" if provision is None else _unvouched(binding, provision)
            if problem:
                self.advisories.append(f"{impl.id}: running without extension {name!r}: {problem}")
            else:
                chosen[name] = provision
        return chosen

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def reset_metrics(self) -> None:
        """Zero the counters of every runtime that supports it (those without ``reset_metrics`` are left alone)."""
        for meter in self.meters.values():
            meter.reset()
        for runtime in self.runtimes.values():
            reset = getattr(runtime, "reset_metrics", None)
            if reset is not None:
                reset()

    def metrics(self) -> dict[str, Mapping[str, Any]]:
        """Metrics of every installed runtime, by stage, plus ``calls`` (executions of the stage, counted by the host,
        replays included). Reading may synchronize the device."""
        out = {}
        for stage, runtime in self.runtimes.items():
            out[stage] = dict(runtime.metrics())
            if stage in self.meters:
                out[stage]["calls"] = self.meters[stage].count()
        return out
