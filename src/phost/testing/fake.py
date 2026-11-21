"""A host with no simulator: plain numpy arrays and a Python "solver" that can be patched.

It implements the stage ``newton.linear_solve`` the way a real adapter does, so every piece of
phost (install rule, guards, session, contract checks) can be tested without a GPU or a simulator.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..errors import CapabilityError
from ..guard import Guard, source_hash
from ..host import Capability, Host, HostInfo, ModelFacts, StageBinding
from ..patch import PatchPoint, Target
from ..stages import newton_linear as nl

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


class FakeHost(Host):
    info = HostInfo(name="fake", version="1", build_id="fake-build-1", patched=False)
    capabilities = Capability.PATCH | Capability.NATIVE

    def __init__(self, nv: int = 12, world_count: int = 4, nv_pad: int = 16, seed: int = 0, coupled: int = 4):
        self.ns = SimpleNamespace(solve_hook=_native_solve, assemble=_assemble)
        self.model = SimpleNamespace(nv=nv, nv_pad=nv_pad)
        self.other_model = SimpleNamespace(nv=nv, nv_pad=nv_pad)
        self.data = SimpleNamespace(nworld=world_count)
        self.rng = np.random.default_rng(seed)
        self.coupled = coupled
        self._reviewed_hook = source_hash(self.ns.solve_hook)
        self._reviewed_assemble = source_hash(self.ns.assemble)
        self._stages = None

    # -- Host ---------------------------------------------------------------------------------
    def facts(self) -> ModelFacts:
        return ModelFacts(nv=self.model.nv, world_count=self.data.nworld, device_kind="cpu")

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
                                                  extensions={"structure_hint": "assembly", "fallback": None})}
        return self._stages

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

    def drive(self, stage: str, *, owned: bool = True, active=None, skip_unchanged=False, skip_noflip=False):
        """Run one call of ``stage`` through whatever is installed. Returns inputs, outputs, nv."""
        if stage != nl.NAME:
            raise CapabilityError(f"fake host drives only {nl.NAME}")
        ctx = self.make_ctx(active)
        before = copy.deepcopy(ctx)
        self.ns.solve_hook(self.model if owned else self.other_model, self.data, ctx, skip_unchanged, skip_noflip)
        act = ~ctx.done & (not skip_noflip or ctx.state_changed_count > 0)
        return dict(inputs=dict(h=before.h, grad=before.grad, active=act.astype(np.int32)),
                    unchanged=dict(h=before.h, grad=before.grad),
                    after=dict(h=ctx.h, grad=ctx.grad),
                    outputs=dict(search=ctx.search, search_dot=ctx.search_dot, newton_decrement=ctx.newton_decrement),
                    nv=self.model.nv, active=act)


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
