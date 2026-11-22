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
from ..stages import newton_linear as nl
from ..stages import probe_state as ps

SENTINEL = 123.0


def _native_solve(m, d, ctx, skip_unchanged=False, skip_noflip=False):
    """The fake host's own Newton solve: float64 solve per world, written as float32."""
    for w in range(d.nworld):
        if ctx.done[w] or (skip_noflip and ctx.state_changed_count[w] == 0):
            continue
        x = np.linalg.solve(nl._symmetric(ctx.h[w], m.nv), np.asarray(ctx.grad[w, :m.nv], np.float64))
        ctx.search[w] = (-x).astype(np.float32)
        ctx.search_dot[w] = np.float32(x @ x)
        ctx.newton_decrement[w] = np.float32(np.asarray(ctx.grad[w, :m.nv], np.float64) @ x)


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


def _native_forward(m, d, ns=None):
    nu = m.nu
    d.actuator_length[:] = d.qpos[:, :nu]
    d.actuator_velocity[:] = d.qvel[:, :nu]
    ns.passive(m, d)
    ns.fwd_actuation(m, d)
    pad = np.zeros_like(d.qvel)
    pad[:, :nu] = d.actuator_force
    d.qacc[:] = -d.qpos + d.qfrc_passive + pad


class FakeHost(Host):
    info = HostInfo(name="fake", version="1", build_id="fake-build-1", patched=False)
    capabilities = Capability.PATCH | Capability.NATIVE

    def __init__(self, nv: int = 12, world_count: int = 4, nv_pad: int = 16, seed: int = 0, coupled: int = 4,
                 nu: int = 3, damping: float = DAMPING):
        self.ns = SimpleNamespace(solve_hook=_native_solve, assemble=_assemble, passive=_native_passive,
                                  fwd_actuation=_native_actuation, euler=_native_euler)
        self.ns.forward = lambda m, d: _native_forward(m, d, self.ns)
        scalar = np.ones(nv, np.int32)
        scalar[-3:] = 0                                  # the last DOFs stand in for a free joint
        dof_damping = np.full(nv, damping, np.float32)
        self.model = SimpleNamespace(nv=nv, nq=nv, nv_pad=nv_pad, nu=nu, scalar=scalar, dof_damping=dof_damping)
        self.other_model = SimpleNamespace(nv=nv, nq=nv, nv_pad=nv_pad, nu=nu, scalar=scalar, dof_damping=dof_damping)
        f32 = lambda *shape: np.zeros(shape, np.float32)
        self.data = SimpleNamespace(nworld=world_count, qpos=f32(world_count, nv), qvel=f32(world_count, nv),
                                    qacc=f32(world_count, nv), time=f32(world_count), ctrl=f32(world_count, nu),
                                    qfrc_passive=f32(world_count, nv), actuator_length=f32(world_count, nu),
                                    actuator_velocity=f32(world_count, nu), actuator_force=f32(world_count, nu))
        self._buffers: dict = {}
        self.rng = np.random.default_rng(seed)
        self.coupled = coupled
        self._reviewed_hook = source_hash(self.ns.solve_hook)
        self._reviewed_assemble = source_hash(self.ns.assemble)
        self._reviewed = {name: source_hash(getattr(self.ns, name)) for name in ("passive", "fwd_actuation", "forward", "euler")}
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
            for contract, attribute, make in ((pf.CONTRACT, "passive", _passive_dispatch),
                                              (cl.CONTRACT, "fwd_actuation", _control_dispatch),
                                              (ps.CONTRACT, "forward", _probe_dispatch)):
                self._stages[contract.name] = StageBinding(contract, self._point(contract.name, attribute), make)
            self._stages[pf.NAME] = dataclasses.replace(self._stages[pf.NAME], extensions={
                "implicit_damping": Provision(point=self._point(pf.NAME, "euler"), wrap=_euler_wrap)})
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


    def _random_state(self, seed):
        rng, d = np.random.default_rng(seed), self.data
        for name in ("qpos", "qvel", "ctrl"):
            getattr(d, name)[:] = rng.standard_normal(getattr(d, name).shape).astype(np.float32)
        d.time[:] = rng.uniform(0, 5, d.nworld).astype(np.float32)
        for name in ("qacc", "qfrc_passive", "actuator_force"):
            getattr(d, name)[:] = SENTINEL

    def integrate(self, *, seed: int = 0, owned: bool = True):
        """One forward() and one Euler step from a random state. Returns the velocity afterwards."""
        self._random_state(seed)
        model = self.model if owned else self.other_model
        self.ns.forward(model, self.data)
        self.ns.euler(model, self.data)
        return self.data.qvel.copy()

    def _drive_forward(self, stage, *, owned, seed):
        """Run one forward() on a random state, return what the host did with the stage's output."""
        if stage not in (pf.NAME, cl.NAME, ps.NAME):
            raise CapabilityError(f"fake host cannot drive {stage}")
        d, m = self.data, self.model
        self._random_state(seed)
        baseline = copy.deepcopy(d)
        original_of(self.ns.passive)(m, baseline)
        self.ns.forward(m if owned else self.other_model, d)
        effect = None
        if stage == pf.NAME:
            effect = dict(delta=d.qfrc_passive.astype(np.float64) - baseline.qfrc_passive.astype(np.float64))
        elif stage == cl.NAME:
            ctrl = d.ctrl.astype(np.float64)
            effect = dict(actuator_force=d.actuator_force, expected_actuator_force=(
                GAIN * ctrl + BIAS_LENGTH * d.actuator_length + BIAS_VELOCITY * d.actuator_velocity))
        return dict(effect=effect, nv=m.nv)


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


def _passive_dispatch(original, runtime, host: FakeHost, allowed: frozenset):
    damping = host.buffer("damping", host.data.qvel.shape) if "implicit_damping" in allowed else None

    def dispatch(m, d):
        original(m, d)
        if m is not host.model:
            return
        scalar = m.scalar.astype(bool)
        q = np.where(scalar, d.qpos[:, :m.nv], 0.0).astype(np.float32)
        force = np.zeros_like(d.qvel)
        if damping is not None:
            damping[:] = 0.0
        runtime.apply(pf.DofForceSystem(q=q, v=d.qvel, time=d.time, scalar=m.scalar, force=force, damping=damping))
        d.qfrc_passive += np.where(scalar, force, 0.0).astype(np.float32)

    return dispatch


def _euler_wrap(original, host: FakeHost):
    """Euler integrates the model's damping implicitly: give it the law's damping on top, for this call."""
    damping = host.buffer("damping", host.data.qvel.shape)

    def dispatch(m, d):
        if m is not host.model:
            return original(m, d)
        seen = copy.copy(m)
        seen.dof_damping = (m.dof_damping + damping).astype(np.float32)
        original(seen, d)

    return dispatch


def _control_dispatch(original, runtime, host: FakeHost, allowed: frozenset):
    def dispatch(m, d):
        if m is host.model:
            runtime.apply(cl.ControlSystem(length=d.actuator_length, velocity=d.actuator_velocity, time=d.time, ctrl=d.ctrl))
        original(m, d)

    return dispatch


def _probe_dispatch(original, runtime, host: FakeHost, allowed: frozenset):
    value = host.buffer(f"probe/{runtime.width}", (host.data.nworld, runtime.width))

    def dispatch(m, d):
        original(m, d)
        if m is host.model:
            runtime.observe(ps.ProbeSystem(qpos=d.qpos, qvel=d.qvel, qacc=d.qacc, time=d.time, value=value))

    return dispatch
