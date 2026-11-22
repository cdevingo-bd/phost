"""What a stage's dispatch does, written as data instead of a function. Host-neutral: a host supplies the few operations
a table needs (``HookSupport``), and the same table machinery serves a GPU host and a simulator-free one.

Every dispatch of a stage that wraps one host function does the same few things: pass on any call that is not for the
host's own model, derive the contract's ports from the host's arrays (running a kernel for those that need one), hand
them to the runtime, and fold what it wrote back into the host's arrays. A ``Hook`` states those steps and ``make``
turns it into the function the session installs, so a stage's binding is a table of where each port comes from.

A source says where a value comes from:

* ``Data("qvel")``: an array of the host's data, read at every call (``d.qvel``)
* ``Model("body_mass")``: an array of the model (``m.body_mass``)
* ``Buf("force")``: a persistent buffer the host owns (``support.buffers().force``)
* ``Arg(2)``: an argument of the hooked call after ``(m, d)``
* ``Dim("nv")``: a size of the model (``world``, ``nv``, ``nu``, ``nbody``)
* ``Once(make)``: ``make(host, runtime)``, evaluated once when the dispatch is built
* ``When("ext", source)``: the source if the implementation declared the extension ``ext``, else None

A step is run around the runtime: ``Launch`` runs a named kernel of the host over a size, ``Zero`` clears an array.

A function that does something a table cannot express (the Newton solve builds masks and a fallback closure) is still
written by hand: ``StageBinding.make_dispatch`` takes any callable, ``Hook.make`` is one way to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class Data:
    name: str


@dataclass(frozen=True)
class Model:
    name: str


@dataclass(frozen=True)
class Buf:
    name: str


@dataclass(frozen=True)
class Arg:
    index: int


@dataclass(frozen=True)
class Dim:
    name: str


@dataclass(frozen=True)
class Once:
    make: Callable[[Any, Any], Any]


@dataclass(frozen=True)
class When:
    extension: str
    source: Any


class HookSupport(Protocol):
    """What a host provides so a ``Hook`` can run on it (returned by the host's ``hook_support()``)."""

    def sizes(self) -> Mapping[str, int]:
        """``world``, ``nv``, ``nu`` and ``nbody`` of this host's model and data."""

    def check(self, d) -> None:
        """Raise ``ValueError`` if ``d`` is not data this host was built for."""

    def zero(self, array) -> None:
        """Set every entry of one of the host's arrays to zero."""

    def launch(self, kernel: str, dims: tuple, inputs: list, outputs: list) -> None:
        """Run the host's kernel called ``kernel`` over ``dims``."""

    def buffers(self) -> Any:
        """An object whose attributes are the host's persistent buffers (``Buf("force")`` reads ``.force``)."""


@dataclass(frozen=True)
class Launch:
    """Run the host's kernel called ``kernel`` over ``dim`` (sizes or ``Dim`` names); the host resolves the name."""

    kernel: str
    dim: tuple
    inputs: tuple = ()
    outputs: tuple = ()


@dataclass(frozen=True)
class Zero:
    """Zero an array before the runtime is called (nothing if its source is absent)."""

    target: Any


@dataclass(frozen=True)
class Hook:
    """One stage's dispatch: when it runs relative to the host function, what it derives and what it writes back.

    ``when``: ``"before"`` or ``"after"`` the host function runs, or ``"instead"`` of it (the runtime does the host
    function's job, so the original is not called). ``system`` is the dataclass the runtime is handed, built from ``ports``.
    ``prepare`` runs before the runtime is called (zero a buffer, launch a kernel that derives a port), ``finish`` after.
    ``handles(args, kwargs)`` says whether a call is one this stage covers; any other call goes to the original.
    """

    when: str
    system: type
    ports: Mapping[str, Any]
    prepare: tuple = ()
    finish: tuple = ()
    handles: Callable[[tuple, Mapping[str, Any]], bool] | None = None

    def __post_init__(self):
        if self.when not in ("before", "after", "instead"):
            raise ValueError(f"when must be 'before', 'after' or 'instead', not {self.when!r}")

    def make(self, entry: str) -> Callable:
        """The ``make_dispatch`` of a ``StageBinding``; ``entry`` is the runtime method (the contract's ``entry``)."""

        def make_dispatch(original, runtime, host, allowed: frozenset):
            support = host.hook_support()
            sizes = dict(support.sizes())

            def bind(source) -> Callable:
                """A function ``(m, d, args) -> value`` for a source; constants are resolved here, once."""
                if isinstance(source, When):
                    return bind(source.source) if source.extension in allowed else (lambda m, d, args: None)
                if isinstance(source, Data):
                    return lambda m, d, args: getattr(d, source.name)
                if isinstance(source, Model):
                    return lambda m, d, args: getattr(m, source.name)
                if isinstance(source, Arg):
                    return lambda m, d, args: args[source.index]
                if isinstance(source, Buf):
                    value = getattr(support.buffers(), source.name)
                elif isinstance(source, Dim):
                    value = sizes[source.name]
                elif isinstance(source, Once):
                    value = source.make(host, runtime)
                else:
                    value = source                                  # a literal
                return lambda m, d, args: value

            def size(dim):
                return sizes[dim] if isinstance(dim, str) else int(dim)

            def compile_step(step) -> Callable:
                if isinstance(step, Zero):
                    target = bind(step.target)
                    def zero(m, d, args):
                        array = target(m, d, args)
                        if array is not None:
                            support.zero(array)
                    return zero
                dims = tuple(size(x) for x in step.dim)
                ins, outs = [bind(x) for x in step.inputs], [bind(x) for x in step.outputs]
                return lambda m, d, args: support.launch(step.kernel, dims, [f(m, d, args) for f in ins],
                                                         [f(m, d, args) for f in outs])

            prepare, finish = [compile_step(s) for s in self.prepare], [compile_step(s) for s in self.finish]
            ports = {name: bind(source) for name, source in self.ports.items()}

            def run(m, d, args):
                for step in prepare:
                    step(m, d, args)
                getattr(runtime, entry)(self.system(**{name: get(m, d, args) for name, get in ports.items()}))   # looked up per call: a test may wrap it
                for step in finish:
                    step(m, d, args)

            def dispatch(m, d, *args, **kwargs):
                if m is not host.model or (self.handles is not None and not self.handles(args, kwargs)):
                    return original(m, d, *args, **kwargs)
                support.check(d)
                if self.when == "before":
                    run(m, d, args)
                    return original(m, d, *args, **kwargs)
                if self.when == "after":
                    result = original(m, d, *args, **kwargs)
                    run(m, d, args)
                    return result
                run(m, d, args)

            return dispatch

        return make_dispatch
