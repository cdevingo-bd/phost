"""A host with no simulator: plain numpy arrays and a Python "solver" that can be patched.

It implements the stage ``newton.linear_solve`` the way a real adapter does, so every piece of
phost (install rule, guards, session, contract checks) can be tested without a GPU or a simulator.
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..errors import CapabilityError
from ..guard import Guard, source_hash
from ..host import Capability, Host, HostInfo, ModelFacts, Provision, StageBinding
from ..patch import PatchPoint, Target, original_of
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import effective_solve as es
from ..stages import newton_linear as nl

SENTINEL = 123.0

BINDING_FACTORIES: list = []


def register_binding(factory):
    """Register ``factory(host) -> StageBinding`` as one more stage the fake host offers (see ``fake_stages``)."""
    BINDING_FACTORIES.append(factory)
    return factory


def _native_solve(m, d, ctx, skip_unchanged=False, skip_noflip=False):
    """The fake host's own Newton solve: float64 solve per world, written as float32."""
    for w in range(d.nworld):
        if ctx.done[w] or (skip_noflip and ctx.state_changed_count[w] == 0):
            continue
        x = np.linalg.solve(nl._symmetric(ctx.h[w], m.nv), np.asarray(ctx.grad[w, :m.nv], np.float64))
        ctx.search[w] = (-x).astype(np.float32)
        ctx.search_dot[w] = np.float32(x @ x)
        ctx.newton_decrement[w] = np.float32(np.asarray(ctx.grad[w, :m.nv], np.float64) @ x)


def _native_eff_solve(m, d, qacc, qfrc=None):
    """The fake host's own effective-metric solve: float64 per world, from the system the host assembled."""
    rhs = d.eff_rhs if qfrc is None else qfrc
    for w in range(d.nworld):
        qacc[w, :m.nv] = np.linalg.solve(es._symmetric(d.eff_h[w], m.nv), np.asarray(rhs[w, :m.nv], np.float64)).astype(np.float32)


def _assemble(m, d, ctx):
    """Stands in for the host code that assembles H; an implementation may rely on it."""
    return ctx.h


# The fake host's own physics for the pointwise stages. Simple on purpose, but wired like a real host:
# forward() looks its steps up on ``ns`` at call time, so patching an entry changes what forward() does.
DAMPING = 0.1                         # native passive damping on every DOF (the model's dof_damping)
TIMESTEP = 0.05
GAIN, BIAS_LENGTH, BIAS_VELOCITY = 2.0, -0.5, -0.05   # native actuator: force = GAIN ctrl + bias terms


def _native_passive(m, d):
    d.qfrc_passive[:] = (-m.dof_damping * d.qvel).astype(np.float32)


def _native_euler(m, d):
    """Unit masses, semi-implicit in velocity: the model's joint damping is integrated implicitly, as MuJoCo's Euler does.
    qacc is the explicit acceleration (passive damper force included), so v+ = v + h qacc / (1 + h c)."""
    d.qvel[:] = (d.qvel + TIMESTEP * d.qacc / (1.0 + TIMESTEP * m.dof_damping)).astype(np.float32)


def _native_actuation(m, d):
    d.actuator_force[:] = (GAIN * d.ctrl.astype(np.float64) + BIAS_LENGTH * d.actuator_length
                           + BIAS_VELOCITY * d.actuator_velocity).astype(np.float32)


def _native_xfrc_accumulate(m, d, qfrc):
    """Fold the wrench on each body into a generalized force through the body Jacobians, as MuJoCo does: qfrc += J^T w."""
    qfrc += np.einsum("bkn,wbk->wn", m.jac, d.xfrc_applied.astype(np.float64)).astype(np.float32)


def _native_forward(m, d, ns=None):
    nu = m.nu
    d.actuator_length[:] = d.qpos[:, :nu]
    d.actuator_velocity[:] = d.qvel[:, :nu]
    d.xpos[:] = (d.qpos @ m.kin_pos).reshape(d.xpos.shape)            # kinematics: body pose and velocity from the state
    d.cvel[:] = (d.qvel @ m.kin_vel).reshape(d.cvel.shape)
    ns.passive(m, d)
    ns.fwd_actuation(m, d)
    pad = np.zeros_like(d.qvel)
    pad[:, :nu] = d.actuator_force
    d.qfrc_smooth[:] = d.qfrc_passive + pad
    ns.xfrc_accumulate(m, d, d.qfrc_smooth)
    d.qacc[:] = -d.qpos + d.qfrc_smooth


# The fake host's kernels: what ``Launch`` names run as on this host. Same names and argument order as the GPU host's.
def _gather_scalar_q(qpos, scalar, q):
    q[:] = np.where(scalar.astype(bool), qpos[:, :q.shape[1]], 0.0)


def _add_scalar_force(scalar, force, qfrc_passive):
    qfrc_passive += np.where(scalar.astype(bool), force, 0.0).astype(qfrc_passive.dtype)


KERNELS = dict(gather_scalar_q=_gather_scalar_q, add_scalar_force=_add_scalar_force)


class FakeHost(Host):
    info = HostInfo(name="fake", version="1", build_id="fake-build-1", patched=False)
    capabilities = Capability.PATCH | Capability.NATIVE

    def __init__(self, nv: int = 12, world_count: int = 4, nv_pad: int = 16, seed: int = 0, coupled: int = 4,
                 nu: int = 3, damping: float = DAMPING, eff_bandwidth: int | None = None):
        self.ns = SimpleNamespace(solve_hook=_native_solve, assemble=_assemble, passive=_native_passive,
                                  fwd_actuation=_native_actuation, euler=_native_euler, eff_solve=_native_eff_solve,
                                  xfrc_accumulate=_native_xfrc_accumulate)
        self.ns.forward = lambda m, d: _native_forward(m, d, self.ns)
        scalar = np.ones(nv, np.int32)
        scalar[-3:] = 0                                  # the last DOFs stand in for a free joint
        dof_damping = np.full(nv, damping, np.float32)
        nbody = max(1, min(4, nv // 3))
        shape = np.random.default_rng(12345)                       # the fake robot's geometry: the same for every host
        kin = dict(nbody=nbody, kin_pos=shape.standard_normal((nv, 3 * nbody)) / np.sqrt(nv),
                   kin_vel=shape.standard_normal((nv, 6 * nbody)) / np.sqrt(nv), jac=shape.standard_normal((nbody, 6, nv)))
        self.model = SimpleNamespace(nv=nv, nq=nv, nv_pad=nv_pad, nu=nu, scalar=scalar, dof_damping=dof_damping, **kin)
        self.other_model = SimpleNamespace(nv=nv, nq=nv, nv_pad=nv_pad, nu=nu, scalar=scalar, dof_damping=dof_damping, **kin)
        f32 = lambda *shape: np.zeros(shape, np.float32)
        self.data = SimpleNamespace(nworld=world_count, qpos=f32(world_count, nv), qvel=f32(world_count, nv),
                                    qacc=f32(world_count, nv), time=f32(world_count), ctrl=f32(world_count, nu),
                                    qfrc_passive=f32(world_count, nv), actuator_length=f32(world_count, nu),
                                    actuator_velocity=f32(world_count, nu), actuator_force=f32(world_count, nu),
                                    eff_h=f32(world_count, nv_pad, nv_pad), eff_rhs=f32(world_count, nv_pad),
                                    eff_qacc=f32(world_count, nv), xpos=f32(world_count, nbody, 3),
                                    cvel=f32(world_count, nbody, 6), xfrc_applied=f32(world_count, nbody, 6),
                                    qfrc_smooth=f32(world_count, nv))
        self._buffers: dict = {}
        self.rng = np.random.default_rng(seed)
        self.coupled = coupled
        self.eff_bandwidth = eff_bandwidth          # the effective system's half bandwidth; None: a dense system
        self._reviewed_hook = source_hash(self.ns.solve_hook)
        self._reviewed_assemble = source_hash(self.ns.assemble)
        self._reviewed = {name: source_hash(getattr(self.ns, name)) for name in
                          ("passive", "fwd_actuation", "forward", "euler", "eff_solve", "xfrc_accumulate")}
        self._stages = None

    # -- Host ---------------------------------------------------------------------------------
    def facts(self) -> ModelFacts:
        return ModelFacts(nv=self.model.nv, world_count=self.data.nworld, device_kind="cpu",
                          extra=dict(nq=self.model.nq, nu=self.model.nu, nhistory=0,
                                     scalar_dofs=int(self.model.scalar.sum())))

    def native(self) -> Any:
        return SimpleNamespace(model=self.model, data=self.data, ns=self.ns)

    def stages(self) -> Mapping[str, StageBinding]:
        if self._stages is None:
            target = Target("fake.ns.solve_hook", lambda: self.ns.solve_hook, lambda f: setattr(self.ns, "solve_hook", f))
            point = PatchPoint(nl.NAME, target, {
                "target": Guard("target", lambda: source_hash(target.get()),
                                {self._reviewed_hook: "fake build 1"}, "the function being replaced"),
                "assembly": Guard("assembly", lambda: source_hash(self.ns.assemble),
                                  {self._reviewed_assemble: "fake build 1"}, "how H is assembled"),
            })
            self._stages = {nl.NAME: StageBinding(nl.CONTRACT, point, _make_dispatch,
                                                  extensions={"structure_hint": Provision(guard="assembly"), "fallback": Provision()})}
            for factory in BINDING_FACTORIES:                 # every other stage is a table (see fake_stages.py)
                binding = factory(self)
                self._stages[binding.contract.name] = binding
        return self._stages

    def _point(self, stage, attribute):
        ns = self.ns
        target = Target(f"fake.ns.{attribute}", lambda: getattr(ns, attribute), lambda f: setattr(ns, attribute, f))
        return PatchPoint(stage, target, {"target": Guard(
            "target", lambda: source_hash(getattr(ns, attribute)), {self._reviewed[attribute]: "fake build 1"},
            f"the {attribute} function being wrapped")})

    def buffer(self, name, shape, dtype=np.float32):
        """A host-owned output buffer, created once per (name, shape)."""
        key = (name, tuple(shape))
        if key not in self._buffers:
            self._buffers[key] = np.zeros(shape, dtype)
        return self._buffers[key]

    def buffers(self) -> SimpleNamespace:
        """The host's persistent buffers a table names with ``Buf(...)``."""
        shape = self.data.qvel.shape
        return SimpleNamespace(q=self.buffer("q", shape), force=self.buffer("force", shape), damping=self.buffer("damping", shape))

    def hook_support(self):
        host = self

        class Support:
            def sizes(self):
                return dict(world=int(host.data.nworld), nv=int(host.model.nv), nu=int(host.model.nu), nbody=int(host.model.nbody))

            def check(self, d):
                if d.nworld != host.data.nworld:
                    raise ValueError("data world count mismatch")

            def zero(self, array):
                array[...] = 0

            def launch(self, kernel, dims, inputs, outputs):
                KERNELS[kernel](*inputs, *outputs)

            def buffers(self):
                return host.buffers()

        return Support()

    # -- driving a call, for tests and conformance --------------------------------------------
    def make_ctx(self, active=None, *, quad=None, state=None):
        n, pad, worlds = self.model.nv, self.model.nv_pad, self.data.nworld
        h = np.full((worlds, pad, pad), 7.0, np.float32)           # garbage everywhere ...
        grad = np.zeros((worlds, pad), np.float32)
        for w in range(worlds):
            a = np.zeros((n, n))
            block = self.rng.standard_normal((self.coupled, self.coupled))
            a[:self.coupled, :self.coupled] = block @ block.T + self.coupled * np.eye(self.coupled)
            a[np.arange(n), np.arange(n)] += self.rng.uniform(1.0, 3.0, n) * (np.arange(n) >= self.coupled) + 0.0
            upper = np.triu(a).astype(np.float32)
            h[w, :n, :n] = upper + np.tril(np.full((n, n), 7.0, np.float32), -1)  # ... below the diagonal
            grad[w, :n] = self.rng.standard_normal(n)
        done = np.zeros(worlds, bool) if active is None else ~np.asarray(active, bool)
        return SimpleNamespace(
            h=h, grad=grad, done=done,
            search=np.full((worlds, n), SENTINEL, np.float32), search_dot=np.full(worlds, SENTINEL, np.float32),
            newton_decrement=np.full(worlds, SENTINEL, np.float32),
            quad_changed_count=np.ones(worlds, np.int32) if quad is None else np.asarray(quad, np.int32),
            state_changed_count=np.ones(worlds, np.int32) if state is None else np.asarray(state, np.int32))

    def drive(self, stage: str, *, owned: bool = True, active=None, skip_unchanged=False, skip_noflip=False, seed: int = 0):
        """Run one call of ``stage`` through whatever is installed. Returns inputs, outputs, nv."""
        if stage == es.NAME:
            return self._drive_effective(owned=owned, seed=seed)
        if stage != nl.NAME:
            return self._drive_forward(stage, owned=owned, seed=seed)
        ctx = self.make_ctx(active)
        before = copy.deepcopy(ctx)
        self.ns.solve_hook(self.model if owned else self.other_model, self.data, ctx, skip_unchanged, skip_noflip)
        act = ~ctx.done & (not skip_noflip or ctx.state_changed_count > 0)
        return dict(inputs=dict(h=before.h, grad=before.grad, active=act.astype(np.int32)),
                    unchanged=dict(h=before.h, grad=before.grad),
                    after=dict(h=ctx.h, grad=ctx.grad),
                    outputs=dict(search=ctx.search, search_dot=ctx.search_dot, newton_decrement=ctx.newton_decrement),
                    nv=self.model.nv, active=act)


    def _drive_effective(self, *, owned, seed):
        """One effective-metric solve of a random SPD system. ``effect`` is the host's own solve of the same system."""
        rng, d, n, pad = np.random.default_rng(seed), self.data, self.model.nv, self.model.nv_pad
        d.eff_h[:] = 7.0                                      # garbage everywhere, then the defined upper triangle
        d.eff_rhs[:] = 0.0
        for w in range(d.nworld):
            a = rng.standard_normal((n, n))
            sym = a @ a.T / n + np.diag(rng.uniform(1.0, 3.0, n))
            if self.eff_bandwidth is not None:                  # a system with structure: entries beyond the band are zero
                sym = np.where(np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= self.eff_bandwidth, sym, 0.0)
                sym += np.eye(n) * (np.abs(sym).sum(axis=1).max())       # diagonally dominant, so still positive definite
            d.eff_h[w, :n, :n] = np.triu(sym).astype(np.float32) + np.tril(np.full((n, n), 7.0, np.float32), -1)
            d.eff_rhs[w, :n] = rng.standard_normal(n)
        d.eff_qacc[:] = SENTINEL
        self.ns.eff_solve(self.model if owned else self.other_model, d, d.eff_qacc)
        actual = d.eff_qacc.copy()
        original_of(self.ns.eff_solve)(self.model, d, d.eff_qacc)
        return dict(effect=dict(qacc=actual, expected=d.eff_qacc.copy()), nv=n)

    def _random_state(self, seed):
        rng, d = np.random.default_rng(seed), self.data
        for name in ("qpos", "qvel", "ctrl"):
            getattr(d, name)[:] = rng.standard_normal(getattr(d, name).shape).astype(np.float32)
        d.time[:] = rng.uniform(0, 5, d.nworld).astype(np.float32)
        for name in ("qacc", "qfrc_passive", "actuator_force", "qfrc_smooth"):
            getattr(d, name)[:] = SENTINEL
        d.xfrc_applied[:] = 0.0

    def integrate(self, *, seed: int = 0, owned: bool = True):
        """One forward() and one Euler step from a random state. Returns the velocity afterwards."""
        self._random_state(seed)
        model = self.model if owned else self.other_model
        self.ns.forward(model, self.data)
        self.ns.euler(model, self.data)
        return self.data.qvel.copy()

    def _drive_forward(self, stage, *, owned, seed):
        """Run one forward() on a random state; what the host did with the stage's output is the binding's ``oracle``.

        Any stage a binding offers is driven this way, so adding a stage to the fake host needs no change here.
        """
        binding = self.stages().get(stage)
        if binding is None:
            raise CapabilityError(f"fake host cannot drive {stage}")
        self._random_state(seed)
        self.ns.forward(self.model if owned else self.other_model, self.data)
        return dict(effect=binding.oracle(self) if binding.oracle else None, nv=self.model.nv)


def _make_dispatch(original, runtime, host: FakeHost, allowed: frozenset):
    initialized = np.zeros(host.data.nworld, bool)

    def dispatch(m, d, ctx, skip_unchanged=False, skip_noflip=False):
        if m is not host.model:
            return original(m, d, ctx, skip_unchanged, skip_noflip)
        active = ~ctx.done & (not skip_noflip or ctx.state_changed_count > 0)
        stale = np.full(active.shape, not skip_unchanged) | (ctx.quad_changed_count > 0) | ~initialized
        refactor = active & stale
        initialized[active] = True
        system = nl.NewtonSystem(
            h=ctx.h, grad=ctx.grad, active=active.astype(np.int32), refactor=refactor.astype(np.int32),
            search=ctx.search, search_dot=ctx.search_dot, newton_decrement=ctx.newton_decrement)
        if "fallback" in allowed:
            def fallback(mask):
                shadow = copy.copy(ctx)
                shadow.done = (np.asarray(mask) == 0) | ~active     # never solve inactive worlds
                original(m, d, shadow, False, False)
            system.fallback = fallback
        if "structure_hint" in allowed:
            def hint(flags, refactor_mask):
                for w in range(flags.shape[0]):
                    if refactor_mask[w]:
                        sym = nl._symmetric(ctx.h[w], m.nv)
                        flags[w, :m.nv] = ((np.abs(sym - np.diag(np.diag(sym))).sum(axis=1)) > 0).astype(flags.dtype)
            system.structure_hint = hint
        runtime.solve(system)

    return dispatch


from . import fake_body_wrench, fake_stages   # noqa: E402,F401  (they register their bindings on import)
