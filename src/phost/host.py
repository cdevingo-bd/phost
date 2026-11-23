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
    backend: str = "numpy"       # the array type the stages' ports carry (``Host.backend``): "numpy", "warp" or "jax"


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
class NativeTiming:
    """How to time the host's own execution of a stage, so a tool can say how much of a step the stage is worth.

    ``run(host)`` returns a function that executes the host's native version of the stage once on ``host.data`` (it must
    be capturable into a graph and leave the data fit to run again), or None when the host cannot do that for this model.
    ``enclosing`` is empty when the function is the stage itself. When the stage cannot run on its own it names the larger
    phase that contains it (``run`` then times that), and the result is an upper bound.
    """

    run: Callable[["Host"], Callable[[], None] | None]
    enclosing: str = ""


@dataclass(frozen=True)
class StageBinding:
    """How one host implements one stage contract.

    ``make_dispatch(original, runtime, host, allowed)`` builds the function installed at ``point``:
    it translates the host's native call into the contract's ports, hands them to ``runtime``, and
    forwards every call that is not the host's owned model to ``original``. ``allowed`` is the set
    of extension names the implementation declared; the dispatch supplies only those.
    ``extensions`` lists the extensions this host can supply, by name, each as a ``Provision``.

    The rest is what a tool needs to know about the stage on this host, as data:
    ``applicable(host)`` returns why the stage cannot be installed on this host's model, or None (a session refuses with
    that reason; a tool can list it for every scene without running anything).
    ``native_timing`` says how to time the host's own version of the stage (see ``NativeTiming``).
    ``oracle(host)`` is what a contract check calls after the host has run with an implementation installed: it measures what
    the host did with the stage's outputs, from the host's own data, and returns the ``effect`` the stage's judge reads.
    ``sparsity(host)`` returns the nonzero pattern of the stage's system matrix (a ``phost.sparsity.Sparsity``), or None when
    the host cannot say or the stage has no matrix. It is the same fact the ``sparsity`` extension hands an implementation,
    available to a tool without installing anything.
    """

    contract: StageContract
    point: PatchPoint
    make_dispatch: Callable[[Callable, Any, "Host", frozenset], Callable]
    extensions: Mapping[str, Provision] = field(default_factory=dict)
    applicable: Callable[["Host"], str | None] | None = None
    native_timing: NativeTiming | None = None
    oracle: Callable[["Host"], Mapping[str, Any] | None] | None = None
    sparsity: Callable[["Host"], Any] | None = None


class Host(abc.ABC):
    info: HostInfo
    capabilities: Capability
    backend: str = "numpy"        # array type the stages' ports carry: "numpy", "warp" or "jax" (also in ``ModelFacts.backend``)

    @abc.abstractmethod
    def facts(self) -> ModelFacts: ...

    @abc.abstractmethod
    def stages(self) -> Mapping[str, StageBinding]: ...

    def stage(self, name: str) -> StageBinding:
        found = self.stages().get(name)
        if found is None:
            raise CapabilityError(f"host {self.info.name!r} has no stage {name!r}; it offers {sorted(self.stages())}")
        return found

    def refusal(self, name: str) -> str | None:
        """Why the stage cannot be installed on this host's model, or None. Reads attributes only: safe at any time."""
        binding = self.stage(name)
        return binding.applicable(self) if binding.applicable else None

    def native(self) -> Any:
        raise CapabilityError(f"host {self.info.name!r} does not expose native objects")

    def hook_support(self):
        """What a ``phost.hook.Hook`` needs from this host (see ``HookSupport``). A host whose bindings are all hand-written
        does not need to provide it."""
        raise CapabilityError(f"host {self.info.name!r} does not support table-driven stage bindings")

    def meter(self) -> Meter:
        """A new counter of stage executions. A host that replays captured device work overrides this with a
        meter whose increment is device work too, so that every replay counts."""
        return PythonMeter()
