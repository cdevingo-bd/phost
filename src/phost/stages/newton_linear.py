"""Stage ``newton.linear_solve``: the SPD solve inside a Newton iteration.

Given the Hessian H and gradient g of a convex per-world problem, produce the Newton step
s = -H^-1 g and two scalars the host's line search needs. This is the stage the coupled-block
Cholesky replaces. The contract is host-neutral: a host adapter maps its own call onto these ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from ..impl import Impl, ImplKind
from ..registry import Registry
from ..stage import (Extension, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance)

NAME = "newton.linear_solve"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Solve H x = g per world for a symmetric positive definite H and return the Newton step "
         "search = -x, search_dot = |search|^2 and newton_decrement = g.x."),
    swap_class=SwapClass.EXACT,
    dims=("world", "nv"),
    ports=(
        Port("h", Role.IN, "float32", ("world", "nv+", "nv+"),
             doc="Hessian per world. The leading nv x nv block is the system.",
             layout="Symmetric positive definite. Only entries with row <= col are valid; the lower triangle is undefined."),
        Port("grad", Role.IN, "float32", ("world", "nv+"), doc="Gradient; the first nv entries are used."),
        Port("active", Role.IN, "int32", ("world",),
             doc="1: solve this world this call. 0: leave its outputs untouched."),
        Port("refactor", Role.IN, "int32", ("world",),
             doc=("1: H changed since this world's last solve, rebuild any cached factorization. "
                  "0: a cached factorization may be reused; the implementation owns that cache. "
                  "Only meaningful where active is 1.")),
        Port("search", Role.OUT, "float32", ("world", "nv"), doc="Newton step -H^-1 g."),
        Port("search_dot", Role.OUT, "float32", ("world",), doc="Squared norm of search."),
        Port("newton_decrement", Role.OUT, "float32", ("world",), doc="g . H^-1 g, equal to -g . search."),
    ),
    invariants=(
        "For every active world, H_sym * (-search) = grad to the tolerance below, where H_sym is rebuilt from the upper triangle.",
        "search_dot equals the squared 2-norm of search and newton_decrement equals -grad . search.",
        "Outputs of inactive worlds are not written.",
        "No input is modified.",
        "Worlds are independent: a world's outputs do not depend on any other world's inputs.",
    ),
    metrics=(
        MetricSpec("max_relative_residual", "1", "lower", "gauge",
                   "max over solves and active worlds of |H_sym x - g| / |g|, evaluated in float64"),
        MetricSpec("max_invariant_error", "1", "lower", "gauge",
                   "max relative error of search_dot and newton_decrement against their definitions"),
        MetricSpec("status", "bitmask", "none", "bitmask",
                   "sticky OR over worlds: 1 non-finite or non-positive pivot, 2 residual above the gate"),
        MetricSpec("fallbacks", "count", "lower", "counter", "worlds handed back to the host's own solver"),
    ),
    tolerances=(
        Tolerance("max_relative_residual", 1e-4, "gate shared with the host-side status bit"),
        Tolerance("max_invariant_error", 1e-3, "chosen loosely; tighten from measured values"),
    ),
    extensions=(
        Extension("structure_hint",
                  "Callable hint(flags, refactor): ORs 1 into flags[world, dof] for every DOF that may have an "
                  "off-diagonal entry in H (a superset is allowed), for worlds with refactor[world] == 1. "
                  "flags persist between calls: the implementation clears them only for worlds it rescans and "
                  "the hint clears them when the host starts a new solve. Relies on how the host assembles H."),
        Extension("fallback",
                  "Callable fallback(solve_mask): runs the host's own solve for every world with solve_mask[world] == 1 "
                  "that is also active, and writes that world's outputs. Inactive worlds are ignored."),
    ),
    reference="newton.linear_solve.float64",
)


@dataclass
class NewtonSystem:
    """The contract's ports for one call, plus the extensions the implementation asked for."""

    h: Any
    grad: Any
    active: Any
    refactor: Any
    search: Any
    search_dot: Any
    newton_decrement: Any
    structure_hint: Callable | None = None
    fallback: Callable | None = None


class NewtonLinearSolver(Protocol):
    """Runtime an implementation of this stage builds."""

    def solve(self, system: NewtonSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl(registry_id: str = "native") -> Impl:
    """The host's own solver, selected by leaving it in place."""
    return Impl(id=f"{NAME}/{registry_id}", stage=NAME, stage_version=CONTRACT.version, version="host",
                title="Host's own solver", kind=ImplKind.NATIVE,
                doc="Leave the host's implementation untouched. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_contract(CONTRACT)
    registry.add_impl(native_impl())


# -- float64 reference and contract evaluation (needs numpy) -------------------------------------
def _symmetric(h, nv):
    import numpy as np
    upper = np.triu(np.asarray(h, np.float64)[:nv, :nv])
    return upper + upper.T - np.diag(np.diag(upper))


def reference(inputs: Mapping[str, Any], nv: int) -> dict:
    """float64 solution for every active world. Returns arrays for all worlds; inactive rows are NaN."""
    import numpy as np
    h, g, active = np.asarray(inputs["h"]), np.asarray(inputs["grad"]), np.asarray(inputs["active"])
    worlds = h.shape[0]
    out = dict(search=np.full((worlds, nv), np.nan), search_dot=np.full(worlds, np.nan),
               newton_decrement=np.full(worlds, np.nan))
    for w in range(worlds):
        if not active[w]:
            continue
        gw = np.asarray(g[w, :nv], np.float64)
        x = np.linalg.solve(_symmetric(h[w], nv), gw)
        out["search"][w], out["search_dot"][w], out["newton_decrement"][w] = -x, float(x @ x), float(gw @ x)
    return out


def evaluate(inputs: Mapping[str, Any], outputs: Mapping[str, Any], nv: int) -> dict:
    """Measure outputs against the contract in float64. Returns the contract's gauge metrics.

    Only active worlds are judged. ``max_relative_residual`` is the accuracy gate.
    """
    import numpy as np
    h, g, active = np.asarray(inputs["h"]), np.asarray(inputs["grad"]), np.asarray(inputs["active"])
    search = np.asarray(outputs["search"], np.float64)
    sdot, decr = np.asarray(outputs["search_dot"], np.float64), np.asarray(outputs["newton_decrement"], np.float64)
    residual, invariant, judged = 0.0, 0.0, 0
    for w in range(h.shape[0]):
        if not active[w]:
            continue
        judged += 1
        x, gw = -search[w, :nv], np.asarray(g[w, :nv], np.float64)
        gnorm = np.linalg.norm(gw)
        err = np.linalg.norm(_symmetric(h[w], nv) @ x - gw)
        rel = err / gnorm if gnorm > 0 else (0.0 if err == 0 else float("inf"))
        residual = max(residual, rel) if np.isfinite(rel) else float("inf")
        want_dot, want_dec = float(search[w, :nv] @ search[w, :nv]), float(gw @ x)
        e1 = abs(sdot[w] - want_dot) / max(abs(want_dot), 1e-300)
        e2 = abs(decr[w] - want_dec) / max(abs(want_dec), 1e-300)
        invariant = max(invariant, e1, e2) if np.isfinite(e1) and np.isfinite(e2) else float("inf")
    return dict(max_relative_residual=float(residual), max_invariant_error=float(invariant), worlds_judged=judged)
