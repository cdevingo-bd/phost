"""Implementations of a stage: metadata the tools read, plus a factory that builds the runtime."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .errors import ApplicabilityError, ContractError
from .host import Host, ModelFacts
from .stage import MetricSpec


class ImplKind(str, enum.Enum):
    NATIVE = "native"          # leave the host's own implementation in place
    BUILTIN = "builtin"        # shipped with a phost-based package
    EXPRESSION = "expression"  # compiled from a typed math expression
    KERNEL = "kernel"          # user-authored kernel code


class StageRuntime(Protocol):
    """What every built implementation exposes, whatever the stage."""

    def metrics(self) -> Mapping[str, Any]: ...


_TYPES = {"int": int, "float": float, "bool": bool, "str": str}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: Any
    doc: str = ""
    min: float | None = None
    max: float | None = None
    choices: tuple = ()

    def coerce(self, value: Any) -> Any:
        kind = _TYPES.get(self.type)
        if kind is None:
            raise ContractError(f"param {self.name!r}: unknown type {self.type!r}")
        if kind is bool and not isinstance(value, bool):
            raise ContractError(f"param {self.name!r}: expected bool, got {value!r}")
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ContractError(f"param {self.name!r}: expected int, got {value!r}")
        if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ContractError(f"param {self.name!r}: expected float, got {value!r}")
        if kind is str and not isinstance(value, str):
            raise ContractError(f"param {self.name!r}: expected str, got {value!r}")
        if self.min is not None and value < self.min:
            raise ContractError(f"param {self.name!r}: {value} < minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ContractError(f"param {self.name!r}: {value} > maximum {self.max}")
        if self.choices and value not in self.choices:
            raise ContractError(f"param {self.name!r}: {value!r} not in {list(self.choices)}")
        return kind(value) if kind is float else value


@dataclass(frozen=True)
class Impl:
    id: str
    stage: str
    stage_version: int
    version: str
    title: str
    kind: ImplKind
    doc: str = ""
    params: tuple[ParamSpec, ...] = ()
    fit: str = ""                                   # prose: where it helps, where it does not
    metrics: tuple[MetricSpec, ...] = ()            # metrics beyond the stage contract's own
    extensions_used: tuple[str, ...] = ()           # optional host callables it relies on
    applicable: Callable[[ModelFacts], str | None] | None = None   # returns a reason if NOT applicable
    advisory: Callable[[ModelFacts], str | None] | None = None     # returns a warning; the impl still runs
    factory: Callable[[Host, Mapping[str, Any]], StageRuntime] | None = None
    # float64 reference of this implementation's own math: (inputs by port name, resolved params) -> outputs by
    # port name, as numpy arrays. Required when the stage's contract says ``reference="impl"``.
    reference: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind is ImplKind.NATIVE and self.factory is not None:
            raise ContractError(f"impl {self.id!r}: a native implementation has no factory")
        if self.kind is not ImplKind.NATIVE and self.factory is None:
            raise ContractError(f"impl {self.id!r}: needs a factory")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ContractError(f"impl {self.id!r}: duplicate params")

    def resolve_params(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        given = dict(given or {})
        known = {p.name: p for p in self.params}
        unknown = set(given) - set(known)
        if unknown:
            raise ContractError(f"impl {self.id!r}: unknown params {sorted(unknown)}; known {sorted(known)}")
        return {name: spec.coerce(given[name] if name in given else spec.default) for name, spec in known.items()}

    def check_applicable(self, facts: ModelFacts) -> None:
        reason = self.applicable(facts) if self.applicable else None
        if reason:
            raise ApplicabilityError(f"impl {self.id!r} does not apply: {reason}")

    def advise(self, facts: ModelFacts) -> str | None:
        return self.advisory(facts) if self.advisory else None

    def describe(self) -> dict:
        """JSON-able description for tools (no callables)."""
        return dict(
            id=self.id, stage=self.stage, stage_version=self.stage_version, version=self.version,
            title=self.title, kind=self.kind.value, doc=self.doc, fit=self.fit,
            params=[dict(name=p.name, type=p.type, default=p.default, doc=p.doc, min=p.min, max=p.max,
                         choices=list(p.choices)) for p in self.params],
            metrics=[vars(m).copy() for m in self.metrics], extensions_used=list(self.extensions_used),
            has_reference=self.reference is not None, provenance=dict(self.provenance))
