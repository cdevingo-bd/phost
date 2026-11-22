"""The host abstraction: who the host is, what it can do, and which stages it exposes."""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .errors import CapabilityError
from .meter import Meter, PythonMeter
from .patch import PatchPoint
from .stage import StageContract


class Capability(enum.Flag):
    NONE = 0
    PATCH = enum.auto()    # stages can be replaced
    NATIVE = enum.auto()   # native() exposes the host's own objects


@dataclass(frozen=True)
class HostInfo:
    name: str                  # "mjwarp" | "mjlab" | "fake" | ...
    version: str
    build_id: str | None = None   # source revision for source builds
    patched: bool | None = None  # the build carries local changes; None when unknown


@dataclass(frozen=True)
class ModelFacts:
    """Portable facts an implementation checks to decide whether it applies."""

    nv: int
    world_count: int
    solver: str = "newton"
    integrator: str = "euler"
    cone: str = "pyramidal"
    sparse: bool = True
    sleep: bool = False
    device_kind: str = "cpu"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Provision:
    """How a host supplies one extension named in a stage contract.

    ``guard`` names a guard of the stage's patch point that must pass when an implementation uses the extension.
    Some extensions need more than a value handed to the runtime: the host must also change what it does at a
    second function (for example integrate a quantity the runtime reports). ``point`` is that function and
    ``wrap(original, host)`` builds what is installed there; it is installed, guarded and restored with the stage,
    and only when an implementation uses the extension.
    """

    guard: str | None = None
    point: PatchPoint | None = None
    wrap: Callable[[Callable, "Host"], Callable] | None = None

    def __post_init__(self):
        if (self.point is None) != (self.wrap is None):
            raise ValueError("a provision's point and wrap go together")


@dataclass(frozen=True)
class StageBinding:
    """How one host implements one stage contract.

    ``make_dispatch(original, runtime, host, allowed)`` builds the function installed at ``point``:
    it translates the host's native call into the contract's ports, hands them to ``runtime``, and
    forwards every call that is not the host's owned model to ``original``. ``allowed`` is the set
    of extension names the implementation declared; the dispatch supplies only those.
    ``extensions`` lists the extensions this host can supply, by name, each as a ``Provision``.
    """

    contract: StageContract
    point: PatchPoint
    make_dispatch: Callable[[Callable, Any, "Host", frozenset], Callable]
    extensions: Mapping[str, Provision] = field(default_factory=dict)


class Host(abc.ABC):
    info: HostInfo
    capabilities: Capability
    backend: str = "numpy"        # array type the stages' ports carry: "numpy" or "warp"

    @abc.abstractmethod
    def facts(self) -> ModelFacts: ...

    @abc.abstractmethod
    def stages(self) -> Mapping[str, StageBinding]: ...

    def stage(self, name: str) -> StageBinding:
        found = self.stages().get(name)
        if found is None:
            raise CapabilityError(f"host {self.info.name!r} has no stage {name!r}; it offers {sorted(self.stages())}")
        return found

    def native(self) -> Any:
        raise CapabilityError(f"host {self.info.name!r} does not expose native objects")

    def meter(self) -> Meter:
        """A new counter of stage executions. A host that replays captured device work overrides this with a
        meter whose increment is device work too, so that every replay counts."""
        return PythonMeter()
