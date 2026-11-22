"""What a stage's dispatch does on MuJoCo Warp, written as data instead of a function.

Every dispatch of a stage that wraps one host function does the same few things: pass on any call that is not for the
host's own model, derive the contract's ports from the host's arrays (launching a kernel for those that need one), hand
them to the runtime, and fold what it wrote back into the host's arrays. A ``Hook`` states those steps and ``make``
turns it into the function the session installs, so a stage's binding is a table of where each port comes from.

A source says where a value comes from:

* ``Data("qvel")``: an array of the host's data, read at every call (``d.qvel``)
* ``Model("body_mass")``: an array of the model (``m.body_mass``)
* ``Buf("force")``: a persistent buffer the host owns (``host.buffers().force``)
* ``Arg(2)``: an argument of the hooked call after ``(m, d)``
* ``Dim("nv")``: a size of the model (``world``, ``nv``, ``nu``, ``nbody``)
* ``Once(make)``: ``make(host, runtime)``, evaluated once when the dispatch is built
* ``When("ext", source)``: the source if the implementation declared the extension ``ext``, else None

A function that does something a table cannot express (the Newton solve builds masks and a fallback closure) is still
written by hand: ``StageBinding.make_dispatch`` takes any callable, ``Hook.make`` is one way to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


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


@dataclass(frozen=True)
class Launch:
    """Launch a kernel of ``phost.hosts._mjwarp_kernels`` over ``dim`` (sizes or ``Dim`` names)."""

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
            wp, k, device = host._wp, host._k, host.device
            sizes = dict(world=int(host.data.nworld), nv=int(host.model.nv), nu=int(host.model.nu),
                         nbody=int(host.model.nbody))

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
                    value = getattr(host.buffers(), source.name)
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
                            array.zero_()
                    return zero
                kernel, dims = getattr(k, step.kernel), tuple(size(x) for x in step.dim)
                ins, outs = [bind(x) for x in step.inputs], [bind(x) for x in step.outputs]
                return lambda m, d, args: wp.launch(kernel, dim=dims, device=device,
                                                    inputs=[f(m, d, args) for f in ins],
                                                    outputs=[f(m, d, args) for f in outs])

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
                if d.nworld != sizes["world"] or d.qpos.device != device:
                    raise ValueError("data world count or device mismatch")
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
