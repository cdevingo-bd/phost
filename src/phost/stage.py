"""Machine-readable stage contracts.

A stage is a named step of a host's computation with a typed interface. The contract
is data: tools (a runner, a UI) read it without importing a host. It states the ports,
their layout conventions, the invariants an implementation must keep, how it is judged
(swap class, tolerances, reference) and which metrics it can emit.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import ContractError


class SwapClass(str, enum.Enum):
    """What kind of replacement a stage accepts. Scoring depends on it."""

    EXACT = "exact"                    # same math, different algorithm; judged against a reference
    APPROXIMATE = "approximate"        # same math, bounded error; judged on a speed/error front
    MODEL_CHANGING = "model_changing"  # different physics; judged against ground truth, not the host
    OBSERVER = "observer"              # reads state, changes nothing; judged on correctness and overhead


class Role(str, enum.Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"


_DIM = re.compile(r"^[a-z][a-z0-9_]*\+?$")


@dataclass(frozen=True)
class Port:
    """One array crossing the stage boundary.

    ``shape`` entries are ints or symbols. A symbol exactly binds a dimension; a symbol with a
    trailing ``+`` ("nv+") means "at least that size" (hosts pad matrices).
    """

    name: str
    role: Role
    dtype: str
    shape: tuple[int | str, ...]
    doc: str = ""
    layout: str = ""

    def __post_init__(self):
        for dim in self.shape:
            if isinstance(dim, bool) or not (isinstance(dim, int) and dim >= 0 or isinstance(dim, str) and _DIM.match(dim)):
                raise ContractError(f"port {self.name!r}: bad dimension {dim!r}")


@dataclass(frozen=True)
class MetricSpec:
    """A quantity an implementation or the runner can report for this stage."""

    name: str
    unit: str
    better: str                 # "lower" | "higher" | "none"
    kind: str                   # "gauge" | "counter" | "timing" | "bitmask"
    doc: str = ""

    def __post_init__(self):
        if self.better not in ("lower", "higher", "none"):
            raise ContractError(f"metric {self.name!r}: better must be lower/higher/none")
        if self.kind not in ("gauge", "counter", "timing", "bitmask"):
            raise ContractError(f"metric {self.name!r}: bad kind {self.kind!r}")


@dataclass(frozen=True)
class Tolerance:
    """A gate: the metric must stay at or below ``max`` for an implementation to be accepted."""

    metric: str
    max: float
    doc: str = ""


@dataclass(frozen=True)
class Extension:
    """An optional capability the host may hand an implementation (a callable), described in prose."""

    name: str
    doc: str


@dataclass(frozen=True)
class StageContract:
    name: str
    version: int
    doc: str
    swap_class: SwapClass
    dims: tuple[str, ...]
    ports: tuple[Port, ...]
    invariants: tuple[str, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    tolerances: tuple[Tolerance, ...] = ()
    extensions: tuple[Extension, ...] = ()
    reference: str | None = None    # "impl": every implementation supplies its own float64 reference;
                                    # otherwise the name of a float64 reference that ships with the stage

    def __post_init__(self):
        names = [p.name for p in self.ports]
        if len(set(names)) != len(names):
            raise ContractError(f"stage {self.name!r}: duplicate port names")
        for port in self.ports:
            for dim in port.shape:
                if isinstance(dim, str) and dim.rstrip("+") not in self.dims:
                    raise ContractError(f"stage {self.name!r}: port {port.name!r} uses undeclared dimension {dim!r}")
        metric_names = {m.name for m in self.metrics}
        if len(metric_names) != len(self.metrics):
            raise ContractError(f"stage {self.name!r}: duplicate metric names")
        for gate in self.tolerances:
            if gate.metric not in metric_names:
                raise ContractError(f"stage {self.name!r}: tolerance on unknown metric {gate.metric!r}")
        if self.version < 1:
            raise ContractError(f"stage {self.name!r}: version must be >= 1")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise ContractError(f"stage {self.name!r} has no port {name!r}")

    # -- serialization -------------------------------------------------------------------
    def to_dict(self) -> dict:
        def port(p):
            return dict(name=p.name, role=p.role.value, dtype=p.dtype, shape=list(p.shape), doc=p.doc, layout=p.layout)
        return dict(
            name=self.name, version=self.version, doc=self.doc, swap_class=self.swap_class.value,
            dims=list(self.dims), ports=[port(p) for p in self.ports], invariants=list(self.invariants),
            metrics=[vars(m).copy() for m in self.metrics], tolerances=[vars(t).copy() for t in self.tolerances],
            extensions=[vars(e).copy() for e in self.extensions], reference=self.reference)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StageContract":
        try:
            return cls(
                name=d["name"], version=int(d["version"]), doc=d["doc"], swap_class=SwapClass(d["swap_class"]),
                dims=tuple(d["dims"]),
                ports=tuple(Port(p["name"], Role(p["role"]), p["dtype"], tuple(p["shape"]), p.get("doc", ""), p.get("layout", ""))
                            for p in d["ports"]),
                invariants=tuple(d.get("invariants", ())),
                metrics=tuple(MetricSpec(**m) for m in d.get("metrics", ())),
                tolerances=tuple(Tolerance(**t) for t in d.get("tolerances", ())),
                extensions=tuple(Extension(**e) for e in d.get("extensions", ())),
                reference=d.get("reference"))
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"malformed contract: {error!r}") from error

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "StageContract":
        return cls.from_dict(json.loads(text))


def _dtype_name(dtype: Any) -> str:
    """Name of a numpy or Warp dtype without importing either library."""
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return {"bool_": "bool", "float": "float32", "int": "int32"}.get(name, name)


def check_ports(contract: StageContract, values: Mapping[str, Any], bindings: Mapping[str, int] | None = None) -> list[str]:
    """Return a list of violations (empty when ``values`` satisfy the contract's shapes and dtypes).

    ``values`` maps port names to array-likes with ``.shape`` and ``.dtype``. ``bindings`` pre-binds
    dimension symbols (for example ``{"world": 8, "nv": 330}``) so every port is checked against the
    model, not only against the other ports.
    """
    problems: list[str] = []
    bound = dict(bindings or {})
    exact_first = sorted(contract.ports, key=lambda p: any(isinstance(d, str) and d.endswith("+") for d in p.shape))
    for port in exact_first:
        if port.name not in values:
            problems.append(f"missing port {port.name!r}")
            continue
        array = values[port.name]
        shape, dtype = tuple(getattr(array, "shape", ())), _dtype_name(getattr(array, "dtype", None))
        if dtype != port.dtype:
            problems.append(f"port {port.name!r}: dtype {dtype}, expected {port.dtype}")
        if len(shape) != len(port.shape):
            problems.append(f"port {port.name!r}: rank {len(shape)}, expected {len(port.shape)}")
            continue
        for axis, (want, got) in enumerate(zip(port.shape, shape)):
            if isinstance(want, int):
                if got != want:
                    problems.append(f"port {port.name!r} axis {axis}: size {got}, expected {want}")
            elif want.endswith("+"):
                symbol = want[:-1]
                if symbol in bound and got < bound[symbol]:
                    problems.append(f"port {port.name!r} axis {axis}: size {got} < {symbol}={bound[symbol]}")
            elif bound.setdefault(want, got) != got:
                problems.append(f"port {port.name!r} axis {axis}: size {got} but {want}={bound[want]}")
    extra = set(values) - {p.name for p in contract.ports}
    if extra:
        problems.append(f"unknown ports {sorted(extra)}")
    return problems
