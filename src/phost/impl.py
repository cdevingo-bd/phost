"""Implementations of a stage: metadata the tools read, plus a factory that builds the runtime."""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .errors import ApplicabilityError, ContractError
from .host import Host, ModelFacts
from .stage import MetricSpec, SwapClass, Tolerance


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
    extensions_used: tuple[str, ...] = ()           # host extensions it needs: the host must offer them
    extensions_optional: tuple[str, ...] = ()       # host extensions it uses when the host offers them (and they pass)
    applicable: Callable[[ModelFacts], str | None] | None = None   # returns a reason if NOT applicable
    advisory: Callable[[ModelFacts], str | None] | None = None     # returns a warning; the impl still runs
    factory: Callable[[Host, Mapping[str, Any]], StageRuntime] | None = None
    # float64 reference of this implementation's own math: (inputs by port name, resolved params) -> outputs by
    # port name, as numpy arrays. Required when the stage's contract says ``reference="impl"``.
    reference: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None
    # Extra check on the resolved parameters (for example "this expression parses"). Raise ContractError to refuse.
    check_params: Callable[[Mapping[str, Any]], None] | None = None
    # An implementation may be less exact than its stage: an exact stage (a linear solve) can be answered by an approximate
    # method (a few iterations). ``swap_class`` says so (only ``APPROXIMATE`` is accepted, and only for an exact or approximate
    # stage), and ``tolerances(params)`` gives the gates it is held to in place of the stage's for the same metrics: the
    # accuracy the method claims, which the check then enforces. The scorer ranks it on speed and accuracy, not speed alone.
    swap_class: SwapClass | None = None
    tolerances: Callable[[Mapping[str, Any]], tuple[Tolerance, ...]] | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind is ImplKind.NATIVE and self.factory is not None:
            raise ContractError(f"impl {self.id!r}: a native implementation has no factory")
        if self.kind is not ImplKind.NATIVE and self.factory is None:
            raise ContractError(f"impl {self.id!r}: needs a factory")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ContractError(f"impl {self.id!r}: duplicate params")
        if self.swap_class not in (None, SwapClass.APPROXIMATE):
            raise ContractError(f"impl {self.id!r}: an implementation can only declare itself approximate")
        if self.tolerances is not None and self.swap_class is None:
            raise ContractError(f"impl {self.id!r}: only an approximate implementation has gates of its own")

    def resolve_params(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        given = dict(given or {})
        known = {p.name: p for p in self.params}
        unknown = set(given) - set(known)
        if unknown:
            raise ContractError(f"impl {self.id!r}: unknown params {sorted(unknown)}; known {sorted(known)}")
        resolved = {name: spec.coerce(given[name] if name in given else spec.default) for name, spec in known.items()}
        if self.check_params is not None:
            self.check_params(resolved)
        return resolved

    def check_applicable(self, facts: ModelFacts) -> None:
        reason = self.applicable(facts) if self.applicable else None
        if reason:
            raise ApplicabilityError(f"impl {self.id!r} does not apply: {reason}")

    def advise(self, facts: ModelFacts) -> str | None:
        return self.advisory(facts) if self.advisory else None

    def run_reference(self, inputs: Mapping[str, Any], params: Mapping[str, Any], dims: Mapping[str, int] | None = None) -> Mapping[str, Any]:
        """Run ``reference``. A reference that declares a ``dims`` parameter also gets the sizes of the stage's dimensions that its
        inputs do not show (a padded system does not say how many unknowns it has)."""
        if self.reference is None:
            raise ContractError(f"impl {self.id!r} has no float64 reference")
        if "dims" in inspect.signature(self.reference).parameters:
            return self.reference(inputs, params, dims=dict(dims or {}))
        return self.reference(inputs, params)

    def gates(self, contract_tolerances: tuple[Tolerance, ...], params: Mapping[str, Any]) -> tuple[Tolerance, ...]:
        """The gates this implementation is held to: the stage's, with this implementation's own in place of the same metrics."""
        own = self.tolerances(params) if self.tolerances is not None else ()
        replaced = {t.metric for t in own}
        return (*[t for t in contract_tolerances if t.metric not in replaced], *own)

    def describe(self) -> dict:
        """JSON-able description for tools (no callables)."""
        return dict(
            id=self.id, stage=self.stage, stage_version=self.stage_version, version=self.version,
            title=self.title, kind=self.kind.value, doc=self.doc, fit=self.fit,
            params=[dict(name=p.name, type=p.type, default=p.default, doc=p.doc, min=p.min, max=p.max,
                         choices=list(p.choices)) for p in self.params],
            metrics=[vars(m).copy() for m in self.metrics], extensions_used=list(self.extensions_used),
            extensions_optional=list(self.extensions_optional),
            has_reference=self.reference is not None, swap_class=self.swap_class.value if self.swap_class else None,
            provenance=dict(self.provenance))
