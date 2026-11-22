"""The spring-mass chain again, over JAX arrays: the host that tests what a stage assumes about writing in place.

``phost.hosts.chain`` steps NumPy arrays that a stage's implementation writes into. A JAX array cannot be written into, so
this host keeps every array of its data, its buffers and its meters in a ``Ref`` (``phost.ports``): a cell that takes numpy's
in-place syntax and replaces the immutable array behind it. The stages, the ``Hook`` tables and every implementation written
against the contract are the ones the NumPy chain uses; only what a cell holds differs.

What that buys is a compiled step. The whole step is also a pure function of the state (``pure_step``): the cells are bound to
the arguments, the host and every installed stage run as ordinary Python while JAX traces them, and what the cells then hold is
the result. ``jax.jit`` of it is the JAX counterpart of capturing a CUDA graph, with the same property: the stages that were
installed when it was traced are the ones it replays. The meter is counted in the state too, so a stage's ``calls`` counts
executions of the compiled step and not traces.

The physics, the stage bindings and the checks are ``chain``'s; this module redefines only the array operations.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from ..host import HostInfo
from ..ports import Ref, unwrap
from . import chain

NAME = "jaxchain"


def _jnp():
    import jax.numpy as jnp
    return jnp


def _spring_force(m, x):
    jnp = _jnp()
    k = jnp.asarray(m.stiffness)
    behind = jnp.concatenate([jnp.zeros((x.shape[0], 1), x.dtype), x[:, :-1]], axis=1)
    pull = k * (x - behind)
    return -pull + jnp.concatenate([pull[:, 1:], jnp.zeros((x.shape[0], 1), x.dtype)], axis=1)


def _passive(m, d):
    d.passive_force[...] = -_jnp().asarray(m.damping) * unwrap(d.v)


def _actuate(m, d):
    d.drive_force[...] = m.gain * unwrap(d.command)


def _assemble(m, d):
    jnp = _jnp()
    h, n = m.timestep, m.nv
    x, v = unwrap(d.x), unwrap(d.v)
    force = unwrap(d.passive_force) + _spring_force(m, x)
    force = force.at[:, :m.nu].add(unwrap(d.drive_force))
    k = jnp.asarray(m.stiffness)
    kdiag, koff = k + jnp.concatenate([k[1:], jnp.zeros(1, k.dtype)]), -k[1:]
    damping = jnp.asarray(m.damping)
    diag = jnp.asarray(m.mass) + h * damping + h * h * kdiag
    off = jnp.broadcast_to(h * h * koff, (x.shape[0], n - 1)) if n > 1 else jnp.zeros((x.shape[0], 0), x.dtype)
    zero = jnp.zeros((x.shape[0], 1), x.dtype)
    kv = kdiag * v + jnp.concatenate([koff * v[:, 1:], zero], axis=1) + jnp.concatenate([zero, koff * v[:, :-1]], axis=1)
    idx = jnp.arange(n)
    big = jnp.where(jnp.asarray(m.upper), 0.0, jnp.nan).astype(x.dtype)
    big = jnp.broadcast_to(big, (x.shape[0], n, n))
    big = big.at[:, idx, idx].set(jnp.broadcast_to(diag, (x.shape[0], n)))
    if n > 1:
        big = big.at[:, idx[:-1], idx[:-1] + 1].set(off)
    d.band_diag[...] = jnp.broadcast_to(diag, (x.shape[0], n))
    d.band_off[...] = off
    d.rhs[...] = force - h * kv
    d.H[...] = big


def thomas(diag, off, rhs):
    """Solve a symmetric tridiagonal system per row (worlds along axis 0) with two scans: forward elimination, back substitution."""
    import jax
    jnp = _jnp()
    diag, off, rhs = (jnp.asarray(a) for a in (diag, off, rhs))
    n = diag.shape[1]
    if n == 1:
        return rhs / diag

    def forward(carry, xs):
        pivot, g_prev = carry
        d_i, o_prev, r_i = xs
        c = o_prev / pivot
        pivot = d_i - o_prev * c
        g = (r_i - o_prev * g_prev) / pivot
        return (pivot, g), (c, g)

    first = rhs[:, 0] / diag[:, 0]
    _, (c, g) = jax.lax.scan(forward, (diag[:, 0], first), (diag[:, 1:].T, off.T, rhs[:, 1:].T))
    g_all = jnp.concatenate([first[None], g], axis=0)                     # (n, worlds)

    def back(x_next, xs):
        g_i, c_i = xs
        x_i = g_i - c_i * x_next
        return x_i, x_i

    _, rest = jax.lax.scan(back, g_all[-1], (g_all[:-1][::-1], c[::-1]))
    return jnp.concatenate([rest[::-1], g_all[-1:]], axis=0).T


def _effective_solve(m, d, a, rhs=None):
    solved = thomas(unwrap(d.band_diag), unwrap(d.band_off), unwrap(d.rhs) if rhs is None else unwrap(rhs))
    if isinstance(a, Ref):
        a[...] = solved
    else:
        a[...] = np.asarray(solved)


def _forward(m, d, ns):
    d.drive_length[...] = unwrap(d.x)[:, :m.nu]
    d.drive_velocity[...] = unwrap(d.v)[:, :m.nu]
    ns.passive(m, d)
    ns.actuate(m, d)
    ns.assemble(m, d)
    ns.effective_solve(m, d, d.a)


def _step(m, d, ns):
    ns.forward(m, d)
    h = m.timestep
    d.v[...] = unwrap(d.v) + h * unwrap(d.a)
    d.x[...] = unwrap(d.x) + h * unwrap(d.v)
    d.t[...] = unwrap(d.t) + h


# The kernels of the table-driven bindings, as this host runs them: each replaces its output whole.
def _gather_scalar_q(x, scalar, q):
    jnp = _jnp()
    q[...] = jnp.where(unwrap(scalar).astype(bool), unwrap(x)[:, :q.shape[1]], 0.0)


def _add_scalar_force(scalar, force, passive_force):
    jnp = _jnp()
    passive_force[...] = unwrap(passive_force) + jnp.where(unwrap(scalar).astype(bool), unwrap(force), 0.0)


KERNELS = dict(gather_scalar_q=_gather_scalar_q, add_scalar_force=_add_scalar_force)


class JaxMeter:
    """Counts executions of a stage in the host's state, so a compiled step that replays the stage counts every replay."""

    def __init__(self, host: "JaxChainHost"):
        self._cell = Ref(_jnp().zeros((), _jnp().int32))
        host._extra_cells.append(self._cell)

    def tick(self) -> None:
        self._cell += 1

    def count(self) -> int:
        return int(np.asarray(self._cell))

    def reset(self) -> None:
        self._cell.value = _jnp().zeros((), _jnp().int32)


class JaxChainHost(chain.ChainHost):
    info = HostInfo(name=NAME, version="1", build_id="jaxchain-1", patched=False)
    backend = "jax"
    kernels = KERNELS
    _step = staticmethod(_step)

    def __init__(self, model=None, worlds: int = 4, seed: int = 0):
        self._extra_cells: list[Ref] = []
        super().__init__(model, worlds, seed)

    def _make_data(self, w: int) -> SimpleNamespace:
        jnp = _jnp()
        n, nu = self.model.nv, self.model.nu
        z = lambda *shape: Ref(jnp.zeros(shape, jnp.float32))
        return SimpleNamespace(
            nworld=w, x=z(w, n), v=z(w, n), a=z(w, n), t=z(w), command=z(w, nu), drive_length=z(w, nu), drive_velocity=z(w, nu),
            drive_force=z(w, nu), passive_force=z(w, n), band_diag=z(w, n), band_off=z(w, max(n - 1, 0)), rhs=z(w, n),
            H=Ref(jnp.full((w, n, n), jnp.nan, jnp.float32)))

    def _make_ns(self) -> SimpleNamespace:
        ns = SimpleNamespace(passive=_passive, actuate=_actuate, assemble=_assemble, effective_solve=_effective_solve)
        ns.forward = lambda m, d: _forward(m, d, ns)
        return ns

    def _new_buffer(self, shape, dtype):
        return Ref(_jnp().zeros(shape, dtype))

    def _zero(self, array) -> None:
        array.value = _jnp().zeros(array.shape, array.dtype)             # never reads the old value: it may be a stale trace

    def meter(self):
        return JaxMeter(self)

    # -- the compiled step --------------------------------------------------------------------------
    def cells(self) -> dict[str, Ref]:
        """Every cell the step reads or writes, by name: the data, the buffers stages created and the meters."""
        found = {f"data.{k}": v for k, v in vars(self.data).items() if isinstance(v, Ref)}
        found.update({f"buffer.{key[0]}.{'x'.join(map(str, key[1]))}": ref for key, ref in self._buffers.items()})
        found.update({f"extra.{i}": ref for i, ref in enumerate(self._extra_cells)})
        return found

    def get_state(self) -> dict:
        return {name: ref.value for name, ref in self.cells().items()}

    def set_state(self, state: dict) -> None:
        for name, ref in self.cells().items():
            ref.value = state[name]

    def pure(self, function):
        """``function()``, which changes the cells, as a function of the state: ``state -> state``, what ``jax.jit`` traces. The
        cells hold the concrete values again afterwards, so nothing a trace made outlives it."""
        def run(state: dict) -> dict:
            cells = self.cells()
            saved = {name: ref.value for name, ref in cells.items()}
            try:
                for name, value in state.items():
                    cells[name].value = value
                function()
                return {name: ref.value for name, ref in cells.items()}
            finally:
                for name, value in saved.items():
                    cells[name].value = value
        return run

    def pure_step(self, state: dict) -> dict:
        """One step as a function of the state."""
        return self.pure(self.step)(state)

    def drive(self, stage: str, *, owned: bool = True, seed: int = 0, **_ignored):
        return super().drive(stage, owned=owned, seed=seed)
