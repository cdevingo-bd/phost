"""Drive the real MJWarp hook with a chosen Hessian: for contract checks against a real host.

Builds a genuine solver context (``solver._create_solver_context``, a private MJWarp function: this
is test support, not part of the adapter) and calls whatever is installed at the hook.
"""

from __future__ import annotations

import copy

import numpy as np

from ..hosts.mjwarp import MJWarpHost, _solver_module
from ..stages import newton_linear as nl

SENTINEL = 123.0


def synthetic_batch(nworld, nv, nvpad, coupled, seed=0):
    """SPD systems with a dense block over the first ``coupled`` DOFs; garbage in the lower triangle."""
    rng = np.random.default_rng(seed)
    h = np.zeros((nworld, nvpad, nvpad), np.float32)
    grad = np.zeros((nworld, nvpad), np.float32)
    for w in range(nworld):
        a = np.diag(rng.uniform(1.0, 3.0, nv))
        if coupled:
            b = rng.standard_normal((coupled, coupled))
            a[:coupled, :coupled] += b @ b.T / coupled + np.eye(coupled)
        a32 = a.astype(np.float32)
        h[w, :nv, :nv] = np.triu(a32) + np.tril(np.full((nv, nv), 7.0, np.float32), -1)
        idx = np.arange(nv, nvpad)
        h[w, idx, idx] = 1.0
        grad[w, :nv] = rng.standard_normal(nv)
    return h, grad


class DrivableMJWarpHost(MJWarpHost):
    def drive(self, stage: str, *, owned=True, active=None, skip_unchanged=False, skip_noflip=False,
              h=None, grad=None, coupled=8, seed=0):
        import warp as wp
        if stage != nl.NAME:
            raise ValueError(f"only {nl.NAME} can be driven")
        solver, m, d = _solver_module(), self.model, self.data
        n, nv, pad = int(d.nworld), int(m.nv), int(m.nv_pad)
        with wp.ScopedDevice(self.device):
            ctx = solver._create_solver_context(m, d)
            if h is None:
                h, grad = synthetic_batch(n, nv, pad, coupled, seed)
            ctx.h.assign(np.ascontiguousarray(h, np.float32))
            ctx.grad.assign(np.ascontiguousarray(grad, np.float32))
            done = np.zeros(n, bool) if active is None else ~np.asarray(active, bool)
            ctx.done.assign(done)
            for name in ("quad_changed_count", "state_changed_count"):
                if getattr(ctx, name).shape[0] != n:
                    setattr(ctx, name, wp.ones(n, dtype=wp.int32, device=self.device))
                else:
                    getattr(ctx, name).fill_(1)
            target = m if owned else copy.copy(m)
            call = solver._cholesky_factorize_solve
            if skip_unchanged or skip_noflip:
                call(target, d, ctx, False, False)       # prime the host's factor cache first
            for name in ("search", "search_dot", "newton_decrement"):
                getattr(ctx, name).fill_(SENTINEL)
            wp.synchronize()
            before = dict(h=ctx.h.numpy().copy(), grad=ctx.grad.numpy().copy())
            call(target, d, ctx, skip_unchanged, skip_noflip)
            wp.synchronize()
            act = ~done
            if skip_noflip:
                act &= ctx.state_changed_count.numpy() > 0
            return dict(inputs=dict(h=before["h"], grad=before["grad"], active=act.astype(np.int32)),
                        unchanged=before, after=dict(h=ctx.h.numpy(), grad=ctx.grad.numpy()),
                        outputs=dict(search=ctx.search.numpy(), search_dot=ctx.search_dot.numpy(),
                                     newton_decrement=ctx.newton_decrement.numpy()),
                        nv=nv, active=act)
