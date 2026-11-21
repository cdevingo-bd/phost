"""Drive the real MJWarp hook with a chosen Hessian: for contract checks against a real host.

Builds a genuine solver context (``solver._create_solver_context``, a private MJWarp function: this
is test support, not part of the adapter) and calls whatever is installed at the hook.
"""

from __future__ import annotations

import copy

import numpy as np

from ..hosts.mjwarp import MJWarpHost, _forward_module, _passive_module, _solver_module
from ..patch import original_of
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import newton_linear as nl
from ..stages import probe_state as ps

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
    """MJWarpHost plus ``drive``. Pass ``cpu_data`` (a state with contacts) to use ``mode="step"``."""

    def __init__(self, model, data, *, cpu_model=None, cpu_data=None, name="mjwarp"):
        super().__init__(model, data, cpu_model=cpu_model, name=name)
        self.cpu_data = cpu_data
        self._base_qpos = data.qpos.numpy().copy()            # the state every seeded drive perturbs

    def drive(self, stage: str, *, mode="synthetic", index=0, **kwargs):
        """Run one hook call and return its inputs and outputs.

        ``mode="synthetic"``: a genuine solver context with a chosen H (any SPD matrix). Right for
        implementations that read H only.
        ``mode="step"``: the ``index``-th hook call of a real ``mjw.step`` on ``cpu_data``, so H is what the
        host assembled. Needed for implementations that use the structure hint, which derives from the
        host's constraint Jacobian and mass matrix.
        """
        if stage in (pf.NAME, cl.NAME, ps.NAME):
            return self._drive_forward(stage, **kwargs)
        if stage != nl.NAME:
            raise ValueError(f"cannot drive {stage}")
        if mode == "step":
            return self._drive_step(index)
        return self._drive_synthetic(**kwargs)

    def _drive_forward(self, stage, *, seed=0, owned=True):
        """One forward-dynamics pass from a random state. The runtime's calls are recorded by ``check_impl``.

        Returns the host-side effect the stage's contract promises: for ``dof.passive_force`` the change of
        ``qfrc_passive`` against the host's own passive force; for ``control.law`` the actuator force MuJoCo
        (C, float64) computes for the ctrl that was used. ``owned=False`` runs a copy of the model, which the
        stage must leave alone.
        """
        import copy
        import mujoco
        import warp as wp
        m, d = self.model, self.data
        n, nv, nu = int(d.nworld), int(m.nv), int(m.nu)
        rng = np.random.default_rng(seed)
        scalar, address = self._scalar_dofs()
        qpos = self._base_qpos.copy()
        for dof in np.nonzero(scalar)[0]:
            qpos[:, address[dof]] += 0.1 * rng.standard_normal(n)
        with wp.ScopedDevice(self.device):
            d.qpos.assign(qpos)
            d.qvel.assign((0.3 * rng.standard_normal((n, nv))).astype(np.float32))
            if nu:
                d.ctrl.assign(rng.uniform(-0.5, 0.5, (n, nu)).astype(np.float32))
            d.time.assign(rng.uniform(0, 5, n).astype(np.float32))
            target = m if owned else copy.copy(m)
            _forward_module().forward(target, d)
            wp.synchronize()
            effect = None
            if stage == pf.NAME:
                actual = d.qfrc_passive.numpy().astype(np.float64)
                original_of(_passive_module().passive)(m, d)           # the host's own passive force, same state
                wp.synchronize()
                effect = dict(delta=actual - d.qfrc_passive.numpy().astype(np.float64))
            elif stage == cl.NAME and nu:
                if self.cpu_model is None:
                    raise ValueError("driving control.law needs cpu_model for the float64 actuator-force reference")
                cpu = mujoco.MjData(self.cpu_model)
                expected = np.zeros((n, nu))
                q, v, c, t = d.qpos.numpy(), d.qvel.numpy(), d.ctrl.numpy(), d.time.numpy()
                for w in range(n):
                    cpu.qpos[:], cpu.qvel[:], cpu.ctrl[:], cpu.time = q[w], v[w], c[w], t[w]
                    mujoco.mj_forward(self.cpu_model, cpu)
                    expected[w] = cpu.actuator_force
                effect = dict(actuator_force=d.actuator_force.numpy(), expected_actuator_force=expected)
        return dict(effect=effect, nv=nv)

    def _drive_step(self, index):
        import warp as wp
        import mujoco_warp as mjw
        if self.cpu_data is None or self.cpu_model is None:
            raise ValueError("mode='step' needs cpu_model and cpu_data")
        solver, d = _solver_module(), self.data
        n = int(d.nworld)
        seen, record = [0], {}
        installed = solver._cholesky_factorize_solve

        def spy(m, dd, ctx, skip_unchanged=False, skip_noflip=False):
            if m is not self.model:
                return installed(m, dd, ctx, skip_unchanged, skip_noflip)
            mine = seen[0] == index
            seen[0] += 1
            if not mine:
                return installed(m, dd, ctx, skip_unchanged, skip_noflip)
            for name in ("search", "search_dot", "newton_decrement"):
                getattr(ctx, name).fill_(SENTINEL)
            wp.synchronize()
            done = ctx.done.numpy().copy()
            act = ~done
            if skip_noflip:
                act &= ctx.state_changed_count.numpy() > 0
            before = dict(h=ctx.h.numpy().copy(), grad=ctx.grad.numpy().copy())
            installed(m, dd, ctx, skip_unchanged, skip_noflip)
            wp.synchronize()
            record.update(inputs=dict(h=before["h"], grad=before["grad"], active=act.astype(np.int32)),
                          unchanged=before, after=dict(h=ctx.h.numpy().copy(), grad=ctx.grad.numpy().copy()),
                          outputs=dict(search=ctx.search.numpy().copy(), search_dot=ctx.search_dot.numpy().copy(),
                                       newton_decrement=ctx.newton_decrement.numpy().copy()),
                          nv=int(self.model.nv), active=act)

        solver._cholesky_factorize_solve = spy
        try:
            with wp.ScopedDevice(self.device):
                data = mjw.put_data(self.cpu_model, self.cpu_data, nworld=n, nconmax=64 * n, njmax=512)
                mjw.step(self.model, data)
                wp.synchronize()
        finally:
            solver._cholesky_factorize_solve = installed
        if not record:
            raise IndexError(f"the step made only {seen[0]} hook calls; asked for index {index}")
        return record

    def _drive_synthetic(self, *, owned=True, active=None, skip_unchanged=False, skip_noflip=False,
                         h=None, grad=None, coupled=8, seed=0):
        import warp as wp
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
