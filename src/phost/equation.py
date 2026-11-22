"""How a stage's law can be typed: data a stage declares next to its contract, read by whatever turns text into a runtime.

A stage that accepts a typed law says which names the text may use and what it writes. It does not say how the text becomes
code: that belongs to the tool that compiles it (``phost-author``) and to the array backend it compiles for, so a new backend
(NumPy, Warp, JAX, ...) adds an emitter and no stage changes, and a new stage adds a declaration and no emitter changes.

Three forms cover what the shipped stages need.

``Pointwise``  one law per element (a DOF, an actuator, a body) of every world. Names are scalars: one component of an input
               port at that element. The law writes one component of an output port per output name. Passive force, control
               law and body wrench are this form.
``Channels``   one expression per column of a ``(world, width)`` output. Names are arrays along an axis of the model; sums,
               norms and maxima collapse an axis and ``name[3]`` picks one entry. The probe is this form.
``Iteration``  a short program, repeated a fixed number of times, whose result is the solution of ``A x = b`` per world: the
               typed form of a linear solve. The two SPD solves are this form.

A declaration is checked against the contract when it is attached (``StageSpec``), so a name that points at a port that does
not exist, or at the wrong role, fails when the stage is defined and not when someone first types an equation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import ContractError
from .stage import Role, StageContract


@dataclass(frozen=True)
class Name:
    """A name the text may use (an input) or write (an output), and the port entry it stands for.

    ``column`` picks one entry of the port's last axis when that axis is a fixed-size vector (a wrench has six).
    """

    name: str
    port: str
    column: int | None = None
    doc: str = ""

    def to_dict(self) -> dict:
        return dict(name=self.name, port=self.port, column=self.column, doc=self.doc)


@dataclass(frozen=True)
class Slope:
    """A law's derivative with respect to one of its inputs, reported to the host on an output port as ``sign * d law / d name``.

    The derivative is taken symbolically from the typed tree, so a host that integrates a velocity dependence implicitly can
    do so for the typed law. ``param`` is the implementation parameter that turns the report off.
    """

    of: str
    port: str
    sign: float = -1.0
    param: str = "implicit"

    def to_dict(self) -> dict:
        return dict(of=self.of, port=self.port, sign=self.sign, param=self.param)


Predicate = Callable[[Any], "str | None"]


@dataclass(frozen=True)
class Pointwise:
    """One law per element of every world.

    ``element`` is the dimension looped over. Inputs are ports shaped ``(world, element)``, ``(world, element, k)`` (one name
    per column) or ``(world,)`` (the same value for every element). ``mask`` names an int port shaped ``(element,)`` that is 1
    where the law applies; the outputs are 0 elsewhere. ``applicable`` and ``advisory`` are the stage's rules for whether a
    law fits a model (``ModelFacts`` in, a reason or None out); they are code, so they are not serialized.
    """

    element: str
    inputs: tuple[Name, ...]
    outputs: tuple[Name, ...]
    mask: str | None = None
    slope: Slope | None = None
    applicable: Predicate | None = field(default=None, compare=False, repr=False)
    advisory: Predicate | None = field(default=None, compare=False, repr=False)
    kind: str = field(default="pointwise", init=False)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, element=self.element, inputs=[n.to_dict() for n in self.inputs],
                    outputs=[n.to_dict() for n in self.outputs], mask=self.mask,
                    slope=self.slope.to_dict() if self.slope else None)


@dataclass(frozen=True)
class Channels:
    """One expression per column of ``output``, a ``(world, width)`` port. Inputs are arrays along an axis of the model."""

    inputs: tuple[Name, ...]
    output: str
    applicable: Predicate | None = field(default=None, compare=False, repr=False)
    advisory: Predicate | None = field(default=None, compare=False, repr=False)
    kind: str = field(default="channels", init=False)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, inputs=[n.to_dict() for n in self.inputs], output=self.output)


@dataclass(frozen=True)
class Iteration:
    """A linear solve ``matrix x = rhs`` per world, written as a program that is repeated.

    ``matrix`` is a symmetric port whose upper triangle is valid; ``rhs`` a vector port; both along ``dim``. The program's
    ``x`` is written to ``solution`` multiplied by ``sign``. ``derived`` lists the other outputs the stage defines in terms of
    ``x``, as ``(port, quantity)``: ``"square_norm"`` is ``x . x`` and ``"rhs_dot"`` is ``rhs . x``. ``active`` names an int
    port per world: where it is 0 the world is left alone.
    """

    matrix: str
    rhs: str
    solution: str
    dim: str
    sign: float = 1.0
    active: str | None = None
    derived: tuple[tuple[str, str], ...] = ()
    applicable: Predicate | None = field(default=None, compare=False, repr=False)
    advisory: Predicate | None = field(default=None, compare=False, repr=False)
    kind: str = field(default="iteration", init=False)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, matrix=self.matrix, rhs=self.rhs, solution=self.solution, dim=self.dim, sign=self.sign,
                    active=self.active, derived=[list(d) for d in self.derived])


EquationForm = Pointwise | Channels | Iteration
QUANTITIES = ("square_norm", "rhs_dot")


def _port(contract: StageContract, name: str, role: Role | None, what: str):
    port = next((p for p in contract.ports if p.name == name), None)
    if port is None:
        raise ContractError(f"stage {contract.name!r}: the equation form's {what} {name!r} is not a port")
    if role is not None and (port.role is Role.IN) != (role is Role.IN):
        raise ContractError(f"stage {contract.name!r}: the equation form's {what} {name!r} must be an "
                            f"{'input' if role is Role.IN else 'output'} port")
    return port


def _names_unique(contract: StageContract, names) -> None:
    seen = [n.name for n in names]
    if len(set(seen)) != len(seen):
        raise ContractError(f"stage {contract.name!r}: the equation form repeats a name")
    for name in seen:
        if not name.isidentifier() or name.startswith("_") or name == "pi":
            raise ContractError(f"stage {contract.name!r}: {name!r} cannot be a name in an equation")


def validate(form: EquationForm, contract: StageContract) -> None:
    """Raise ``ContractError`` unless ``form`` is consistent with ``contract``."""
    if isinstance(form, Pointwise):
        if form.element not in contract.dims:
            raise ContractError(f"stage {contract.name!r}: the equation's element {form.element!r} is not a dimension")
        _names_unique(contract, (*form.inputs, *form.outputs))
        for n in form.inputs:
            _check_pointwise_port(contract, form, n, Role.IN)
        for n in form.outputs:
            _check_pointwise_port(contract, form, n, Role.OUT)
        if not form.outputs:
            raise ContractError(f"stage {contract.name!r}: a pointwise equation form needs an output")
        if form.mask is not None:
            mask = _port(contract, form.mask, Role.IN, "mask")
            if mask.shape != (form.element,) or mask.dtype != "int32":
                raise ContractError(f"stage {contract.name!r}: the mask {form.mask!r} must be an int32 port over {form.element!r}")
        if form.slope is not None:
            if form.slope.of not in {n.name for n in form.inputs}:
                raise ContractError(f"stage {contract.name!r}: the slope is taken in {form.slope.of!r}, which is not an input name")
            target = _port(contract, form.slope.port, Role.OUT, "slope port")
            if target.shape != ("world", form.element):
                raise ContractError(f"stage {contract.name!r}: the slope port must be shaped (world, {form.element})")
            if len(form.outputs) != 1:
                raise ContractError(f"stage {contract.name!r}: a slope needs a single output")
    elif isinstance(form, Channels):
        _names_unique(contract, form.inputs)
        for n in form.inputs:
            port = _port(contract, n.port, Role.IN, "input")
            if not (port.shape == ("world",) or len(port.shape) == 2 and port.shape[0] == "world"):
                raise ContractError(f"stage {contract.name!r}: channel input {n.name!r} must be shaped (world,) or (world, axis)")
            if n.column is not None:
                raise ContractError(f"stage {contract.name!r}: channel inputs are whole ports, not columns")
        out = _port(contract, form.output, Role.OUT, "output")
        if len(out.shape) != 2 or out.shape[0] != "world":
            raise ContractError(f"stage {contract.name!r}: the channel output {form.output!r} must be shaped (world, width)")
    elif isinstance(form, Iteration):
        matrix = _port(contract, form.matrix, Role.IN, "matrix")
        rhs = _port(contract, form.rhs, Role.IN, "rhs")
        solution = _port(contract, form.solution, Role.OUT, "solution")
        if form.dim not in contract.dims:
            raise ContractError(f"stage {contract.name!r}: the iteration's dim {form.dim!r} is not a dimension")
        padded = f"{form.dim}+"
        if matrix.shape != ("world", padded, padded) or rhs.shape != ("world", padded) or solution.shape != ("world", form.dim):
            raise ContractError(f"stage {contract.name!r}: an iteration needs matrix (world, {padded}, {padded}), rhs "
                                f"(world, {padded}) and solution (world, {form.dim})")
        if form.active is not None and _port(contract, form.active, Role.IN, "active").shape != ("world",):
            raise ContractError(f"stage {contract.name!r}: the active port must be shaped (world,)")
        for port, quantity in form.derived:
            if quantity not in QUANTITIES:
                raise ContractError(f"stage {contract.name!r}: unknown derived quantity {quantity!r}; known {list(QUANTITIES)}")
            if _port(contract, port, Role.OUT, "derived port").shape != ("world",):
                raise ContractError(f"stage {contract.name!r}: the derived port {port!r} must be shaped (world,)")
    else:                                                        # pragma: no cover
        raise ContractError(f"stage {contract.name!r}: {form!r} is not an equation form")


def _check_pointwise_port(contract: StageContract, form: Pointwise, n: Name, role: Role) -> None:
    port = _port(contract, n.port, role, "input" if role is Role.IN else "output")
    shape = port.shape
    if shape == ("world",) and role is Role.IN and n.column is None:
        return
    if shape == ("world", form.element) and n.column is None:
        return
    if len(shape) == 3 and shape[:2] == ("world", form.element) and isinstance(shape[2], int) and n.column is not None \
            and 0 <= n.column < shape[2]:
        return
    raise ContractError(f"stage {contract.name!r}: {n.name!r} does not fit port {n.port!r} {shape}: a pointwise name is one "
                        f"entry per {form.element!r} (a column for a (world, {form.element}, k) port, none otherwise)")
