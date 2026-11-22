"""Stage ``effective.solve``: the SPD solve of a discrete step's effective metric.

A host that integrates with the discrete step map solves, before its constraint solver runs, a linear system
M_hat x = rhs per world, where M_hat is the mass matrix plus a diagonal plus a sparse stiffness block (flex vertices
and edges), all scaled by the time step. The host's own solver is an iterative one, so on small systems its cost is
mostly launch latency. A direct solve of the assembled matrix is exact and has a fixed cost.

The contract is host-neutral. The host adapter assembles M_hat (it knows how its own pieces add up), hands it over as
a dense symmetric matrix per world, and takes the solution back. An implementation never sees the host's layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..impl import Impl, ImplKind
from ..judge import Judge, StageSpec
from ..registry import Registry
from ..stage import CALLS, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance
from ._common import padded_entries

NAME = "effective.solve"

CONTRACT = StageContract(
    name=NAME,
    version=1,
    doc=("Solve M_hat x = rhs for every world, for a symmetric positive definite M_hat, and write x to qacc. "
         "M_hat is the host's effective metric: mass plus time-scaled damping and stiffness."),
    swap_class=SwapClass.EXACT,
    dims=("world", "nv"),
    ports=(
        Port("h", Role.IN, "float32", ("world", "nv+", "nv+"),
             doc="M_hat per world, assembled by the host. The leading nv x nv block is the system.",
             layout="Symmetric positive definite. Only entries with row <= col are valid; the lower triangle is undefined."),
        Port("rhs", Role.IN, "float32", ("world", "nv+"), doc="Right-hand side; the first nv entries are used."),
        Port("qacc", Role.OUT, "float32", ("world", "nv"), doc="The solution x = M_hat^-1 rhs, written for every world."),
    ),
    invariants=(
        "For every world, H_sym * qacc = rhs to the tolerance below, where H_sym is rebuilt from the upper triangle.",
        "No defined input entry is modified. Undefined entries (the lower triangle of h, anything beyond the leading nv "
        "entries) may hold any value, including NaN, and may be overwritten.",
        "Worlds are independent: a world's output does not depend on any other world's inputs.",
        "Every world is solved on every call: the host does not skip any.",
    ),
    metrics=(
        MetricSpec("max_relative_residual", "1", "lower", "gauge",
                   "max over solves and worlds of |H_sym x - rhs| / |rhs|, evaluated in float64"),
        MetricSpec("status", "bitmask", "none", "bitmask",
                   "sticky OR over worlds: 1 non-finite or non-positive pivot, 2 residual above the gate"),
        MetricSpec("max_native_difference", "1", "lower", "gauge",
                   "max over worlds of |qacc - qacc_host| / |qacc_host|, where qacc_host is the host's own solve of the same "
                   "state. Catches an assembly of M_hat that differs from the matrix the host iterates with"),
        CALLS,
    ),
    tolerances=(
        Tolerance("max_native_difference", 1e-3,
                  "the host's own solver is iterative and float32, so two correct solves differ by about 1e-5"),
        Tolerance("max_relative_residual", 1e-4,
                  "gate shared with the host-side status bit. Above a condition number of about 1e4 no float32 "
                  "solver meets it; an implementation reports that in status instead of hiding it"),
    ),
    reference="effective.solve.float64",
    entry="solve",
)


@dataclass
class EffectiveSystem:
    """The contract's ports for one call."""

    h: Any
    rhs: Any
    qacc: Any


class EffectiveSolver(Protocol):
    """Runtime an implementation of this stage builds."""

    def solve(self, system: EffectiveSystem) -> None: ...
    def metrics(self) -> Mapping[str, Any]: ...


def native_impl(registry_id: str = "native") -> Impl:
    """The host's own solver, selected by leaving it in place."""
    return Impl(id=f"{NAME}/{registry_id}", stage=NAME, stage_version=CONTRACT.version, version="host",
                title="Host's own solver", kind=ImplKind.NATIVE,
                doc="Leave the host's implementation untouched. The control for every comparison.")


def register(registry: Registry) -> None:
    registry.add_stage(SPEC)
    registry.add_impl(native_impl())


# -- float64 reference and contract evaluation (needs numpy) -------------------------------------
def _symmetric(h, nv):
    import numpy as np
    upper = np.triu(np.asarray(h, np.float64)[:nv, :nv])
    return upper + upper.T - np.diag(np.diag(upper))


def reference(inputs: Mapping[str, Any], nv: int) -> dict:
    """float64 solution for every world."""
    import numpy as np
    h, rhs = np.asarray(inputs["h"]), np.asarray(inputs["rhs"])
    out = np.zeros((h.shape[0], nv))
    for w in range(h.shape[0]):
        out[w] = np.linalg.solve(_symmetric(h[w], nv), np.asarray(rhs[w, :nv], np.float64))
    return dict(qacc=out)


def native_difference(actual, expected) -> float:
    """Largest per-world relative difference between two solutions (rows are worlds)."""
    import numpy as np
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    worst = 0.0
    for w in range(a.shape[0]):
        scale = np.linalg.norm(b[w])
        err = np.linalg.norm(a[w] - b[w])
        rel = err / scale if scale > 0 else (0.0 if err == 0 else float("inf"))
        worst = max(worst, rel) if np.isfinite(rel) else float("inf")
    return float(worst)


def evaluate(inputs: Mapping[str, Any], outputs: Mapping[str, Any], nv: int) -> dict:
    """Measure the solution against the contract in float64. ``max_relative_residual`` is the accuracy gate."""
    import numpy as np
    h, rhs = np.asarray(inputs["h"]), np.asarray(inputs["rhs"])
    x = np.asarray(outputs["qacc"], np.float64)
    worst = 0.0
    for w in range(h.shape[0]):
        b = np.asarray(rhs[w, :nv], np.float64)
        bnorm = np.linalg.norm(b)
        err = np.linalg.norm(_symmetric(h[w], nv) @ x[w, :nv] - b)
        rel = err / bnorm if bnorm > 0 else (0.0 if err == 0 else float("inf"))
        worst = max(worst, rel) if np.isfinite(rel) else float("inf")
    return dict(max_relative_residual=float(worst), worlds_judged=int(h.shape[0]))


def _judge_evaluate(call, reference, effect, ctx) -> dict:
    """The residual of the runtime's solution, and (when the driver measured the host's own solve of the same state) how far
    it is from that."""
    out = evaluate(call["inputs"], call["outputs"], ctx.nv)
    if effect is not None:
        out["max_native_difference"] = native_difference(effect["qacc"], effect["expected"])
    return out


SPEC = StageSpec(CONTRACT, Judge(evaluate=_judge_evaluate, defined=padded_entries))
