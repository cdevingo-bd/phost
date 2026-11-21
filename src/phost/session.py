"""Apply a selection of implementations to a host for a scope, and read their metrics."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import CapabilityError, ContractError
from .host import Capability, Host
from .impl import ImplKind
from .patch import install
from .registry import Registry, default


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
        self.advisories: list[str] = []
        self._stack: ExitStack | None = None

    def __enter__(self) -> "Session":
        stack = ExitStack()
        try:
            facts = self.host.facts()
            for selection in self.selections:
                impl = self.registry.impl(selection.impl)
                if impl.stage != selection.stage:
                    raise ContractError(f"impl {impl.id!r} is for stage {impl.stage!r}, not {selection.stage!r}")
                params = impl.resolve_params(selection.params)
                impl.check_applicable(facts)
                note = impl.advise(facts)
                if note:
                    self.advisories.append(f"{impl.id}: {note}")
                if impl.kind is ImplKind.NATIVE:
                    continue
                if Capability.PATCH not in self.host.capabilities:
                    raise CapabilityError(f"host {self.host.info.name!r} cannot patch stages")
                binding = self.host.stage(selection.stage)
                if binding.contract.version != impl.stage_version:
                    raise ContractError(
                        f"impl {impl.id!r} targets {impl.stage}@{impl.stage_version}, host offers {binding.contract.key}")
                require = []
                for name in impl.extensions_used:
                    if name not in binding.extensions:
                        raise CapabilityError(
                            f"host {self.host.info.name!r} offers no extension {name!r} for {impl.stage}; "
                            f"it offers {sorted(binding.extensions)}")
                    if binding.extensions[name]:
                        require.append(binding.extensions[name])
                allowed = frozenset(impl.extensions_used)
                runtime = impl.factory(self.host, params)
                self.runtimes[selection.stage] = runtime
                stack.enter_context(install(
                    binding.point,
                    lambda original, b=binding, r=runtime, a=allowed: b.make_dispatch(original, r, self.host, a),
                    require=require))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def metrics(self) -> dict[str, Mapping[str, Any]]:
        """Metrics of every installed runtime, by stage. Reading may synchronize the device."""
        return {stage: dict(runtime.metrics()) for stage, runtime in self.runtimes.items()}
