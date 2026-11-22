"""A host that is not MuJoCo-shaped: a damped spring-mass chain, integrated in NumPy.

It exists to answer one question: do the stage contracts, the typed equations and the lab belong to phost, or did they
only ever fit one simulator? Nothing here is MuJoCo: no ``qpos``, no ``mjw.step``, no Warp, no GPU. The data is
particles on a line (``x``, ``v``, ``a``, ``t``), the model is masses and springs, and the step is linearly implicit Euler:

    (M + h C + h^2 K) a = f(x, v) - h K v        then   v += h a,  x += h v,  t += h

``M + h C + h^2 K`` is tridiagonal and symmetric positive definite, which is exactly what ``effective.solve`` is for, and
its pattern is a band of half-width 1, which is exactly what ``Sparsity.banded`` says. The host's own solve uses that band
(the Thomas algorithm in float64); the stage hands an implementation the same system as the dense matrix the contract
asks for, with the lower triangle left undefined (NaN here, so an implementation that reads it fails loudly).

The stages offered are the ones that mean something here, bound by the same ``Hook`` tables the GPU host uses:

* ``dof.passive_force``  an extra force law on each particle (a drag, an anchor spring, a pulse); the ``implicit_damping``
  extension adds the law's damping to ``C`` for the step, as the chain's own damper is
* ``control.law``        a controller for the ``nu`` driven particles, before the host turns commands into forces
* ``probe.state``        a measurement of ``x``, ``v``, ``a`` and ``t`` after every forward pass
* ``effective.solve``    the SPD solve above, with the band as its ``sparsity``

Not offered, because a chain has nothing for them to be: ``body.wrench`` (no bodies) and ``newton.linear_solve`` (no
constraint solver). A host offers what it has; the lab and the dashboard read the list from the host.

Hooks dispatch on the host function's ``(model, data, ...)`` call shape, which is a convention of how simulator code is
usually written and not a MuJoCo idea; that is the one assumption the table-driven bindings make of a host.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..errors import CapabilityError
from ..guard import Guard, source_hash
from ..hook import Arg, Buf, Data, Hook, Launch, Model, Once, When, Zero
from ..host import Capability, Host, HostInfo, ModelFacts, NativeTiming, Provision, StageBinding
from ..patch import PatchPoint, Target, original_of
from ..ports import unwrap
from ..sparsity import Sparsity
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import effective_solve as es
from ..stages import probe_state as ps

NAME = "chain"
SENTINEL = 123.0


# ---- the simulator --------------------------------------------------------------------------------------

def make_model(n: int = 32, nu: int = 2, *, timestep: float = 0.01, stiffness: float = 400.0, damping: float = 0.2,
               gain: float = 5.0, seed: int = 0) -> SimpleNamespace:
    """A chain of ``n`` unit-ish masses joined by springs, the first joined to a wall, the first ``nu`` driven.

    Masses and stiffnesses vary a little with ``seed`` so no two particles are alike.
    """
    if not 1 <= nu <= n:
        raise ValueError("nu must be between 1 and the number of particles")
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        nv=n, nq=n, nu=nu, timestep=float(timestep), gain=float(gain),
        mass=rng.uniform(0.8, 1.2, n), stiffness=stiffness * rng.uniform(0.8, 1.2, n),   # stiffness[i]: wall or i-1 to i
        damping=np.full(n, float(damping)), particle_scalar=np.ones(n, np.int32),
        upper=np.triu(np.ones((n, n), bool)))


def _spring_force(m, x):
    """-K x for the chain: spring i joins particle i-1 (the wall for i = 0) to particle i."""
    stretch = x - np.concatenate([np.zeros((x.shape[0], 1)), x[:, :-1]], axis=1)       # extension of spring i
    pull = m.stiffness * stretch
    return -pull + np.concatenate([pull[:, 1:], np.zeros((x.shape[0], 1))], axis=1)


def _stiffness_bands(m):
    """The diagonal and the first off-diagonal of K."""
    k = m.stiffness
    return k + np.concatenate([k[1:], [0.0]]), -k[1:]


def _native_passive(m, d):
    d.passive_force[:] = (-m.damping * d.v).astype(np.float32)


def _native_actuate(m, d):
    d.drive_force[:] = (m.gain * d.command).astype(np.float32)


def _native_assemble(m, d):
    """Total force and the effective system, into the data the solve reads. Banded storage for the host's own solve and the
    dense upper triangle for the contract."""
    h, n = m.timestep, m.nv
    x, v = d.x.astype(np.float64), d.v.astype(np.float64)
    force = d.passive_force.astype(np.float64) + _spring_force(m, x)
    force[:, :m.nu] += d.drive_force
    kdiag, koff = _stiffness_bands(m)
    d.band_diag[:] = m.mass + h * m.damping + h * h * kdiag
    d.band_off[:] = h * h * koff
    d.rhs[:] = force - h * (kdiag * v + np.concatenate([koff * v[:, 1:], np.zeros((v.shape[0], 1))], axis=1)
                            + np.concatenate([np.zeros((v.shape[0], 1)), koff * v[:, :-1]], axis=1))
    d.H[:] = np.where(m.upper, 0.0, np.nan)                              # the lower triangle is not defined
    idx = np.arange(n)
    d.H[:, idx, idx] = d.band_diag
    d.H[:, idx[:-1], idx[:-1] + 1] = d.band_off


def thomas(diag, off, rhs) -> np.ndarray:
    """Solve a symmetric tridiagonal system per row (worlds along axis 0), in float64."""
    diag, off, rhs = (np.asarray(a, np.float64) for a in (diag, off, rhs))
    n = diag.shape[1]
    c, g = np.zeros_like(rhs), np.zeros_like(rhs)
    pivot = diag[:, 0].copy()
    g[:, 0] = rhs[:, 0] / pivot
    for i in range(1, n):
        c[:, i - 1] = off[:, i - 1] / pivot
        pivot = diag[:, i] - off[:, i - 1] * c[:, i - 1]
        g[:, i] = (rhs[:, i] - off[:, i - 1] * g[:, i - 1]) / pivot
    out = g
    for i in range(n - 2, -1, -1):
        out[:, i] = g[:, i] - c[:, i] * out[:, i + 1]
    return out


def _native_effective_solve(m, d, a, rhs=None):
    """The host's own solve, from the banded system it assembled."""
    a[:] = thomas(d.band_diag, d.band_off, d.rhs if rhs is None else rhs).astype(np.float32)


def _forward(m, d, ns):
    d.drive_length[:] = d.x[:, :m.nu]
    d.drive_velocity[:] = d.v[:, :m.nu]
    ns.passive(m, d)
    ns.actuate(m, d)
    ns.assemble(m, d)
    ns.effective_solve(m, d, d.a)


def _step(m, d, ns):
    ns.forward(m, d)
    h = np.float32(m.timestep)
    d.v += h * d.a
    d.x += h * d.v
    d.t += h


# The chain's kernels: what ``Launch`` names run as here. The same names the GPU host's tables use, with this host's arrays.
def _gather_scalar_q(x, scalar, q):
    q[:] = np.where(scalar.astype(bool), x[:, :q.shape[1]], 0.0)


def _add_scalar_force(scalar, force, passive_force):
    passive_force += np.where(scalar.astype(bool), force, 0.0).astype(passive_force.dtype)


KERNELS = dict(gather_scalar_q=_gather_scalar_q, add_scalar_force=_add_scalar_force)


# ---- the host -------------------------------------------------------------------------------------------

class ChainHost(Host):
    info = HostInfo(name=NAME, version="1", build_id="chain-1", patched=False)
    capabilities = Capability.PATCH | Capability.NATIVE
    backend = "numpy"

    def __init__(self, model: SimpleNamespace | None = None, worlds: int = 4, seed: int = 0):
        self.model = model if model is not None else make_model(seed=seed)
        self.other_model = copy.copy(self.model)                       # not the host's own: calls on it are left alone
        self.data = self._make_data(int(worlds))
        self.ns = self._make_ns()
        self._reviewed = {name: source_hash(getattr(self.ns, name))
                          for name in ("passive", "actuate", "assemble", "effective_solve", "forward")}
        self._buffers: dict = {}
        self._stages: dict | None = None
        self._rng = np.random.default_rng(seed)

    # -- what a host over other arrays replaces (see ``jaxchain``) ----------------------------------
    kernels = KERNELS

    def _make_data(self, w: int) -> SimpleNamespace:
        n, nu = self.model.nv, self.model.nu
        f32 = lambda *shape: np.zeros(shape, np.float32)
        return SimpleNamespace(
            nworld=w, x=f32(w, n), v=f32(w, n), a=f32(w, n), t=f32(w), command=f32(w, nu),
            drive_length=f32(w, nu), drive_velocity=f32(w, nu), drive_force=f32(w, nu), passive_force=f32(w, n),
            band_diag=f32(w, n), band_off=f32(w, max(n - 1, 0)), rhs=f32(w, n), H=np.full((w, n, n), np.nan, np.float32))

    def _make_ns(self) -> SimpleNamespace:
        ns = SimpleNamespace(passive=_native_passive, actuate=_native_actuate, assemble=_native_assemble,
                             effective_solve=_native_effective_solve)
        ns.forward = lambda m, d: _forward(m, d, ns)
        return ns

    def _new_buffer(self, shape, dtype):
        return np.zeros(shape, dtype)

    def _zero(self, array) -> None:
        array[...] = 0

    # -- running ----------------------------------------------------------------------------------
    def step(self) -> None:
        """One step of every world, through whatever is installed."""
        self._step(self.model, self.data, self.ns)

    _step = staticmethod(_step)

    def forward(self) -> None:
        self.ns.forward(self.model, self.data)

    def load(self, x, v=None) -> None:
        """Set every world's state (a vector shared by all worlds, or ``(worlds, n)``) and zero the time."""
        d = self.data
        d.x[...] = x
        d.v[...] = 0.0 if v is None else v
        d.t[...] = 0.0
        d.a[...] = 0.0

    # -- Host -------------------------------------------------------------------------------------
    def facts(self) -> ModelFacts:
        return ModelFacts(nv=self.model.nv, world_count=self.data.nworld, solver="direct", integrator="linearly-implicit-euler",
                          sparse=True, device_kind="cpu",
                          extra=dict(nq=self.model.nq, nu=self.model.nu, nhistory=0, scalar_dofs=int(self.model.particle_scalar.sum()),
                                     bandwidth=1))

    def native(self) -> Any:
        return SimpleNamespace(model=self.model, data=self.data, ns=self.ns)

    def buffer(self, name, shape, dtype=np.float32):
        key = (name, tuple(shape))
        if key not in self._buffers:
            self._buffers[key] = self._new_buffer(shape, dtype)
        return self._buffers[key]

    def buffers(self) -> SimpleNamespace:
        shape = self.data.v.shape
        return SimpleNamespace(q=self.buffer("q", shape), force=self.buffer("force", shape), damping=self.buffer("damping", shape))

    def hook_support(self):
        host = self

        class Support:
            def sizes(self):
                return dict(world=int(host.data.nworld), nv=int(host.model.nv), nu=int(host.model.nu), nbody=0)

            def check(self, d):
                if d.nworld != host.data.nworld:
                    raise ValueError("data world count mismatch")

            def zero(self, array):
                host._zero(array)

            def launch(self, kernel, dims, inputs, outputs):
                host.kernels[kernel](*inputs, *outputs)

            def buffers(self):
                return host.buffers()

        return Support()

    def _point(self, stage: str, attribute: str) -> PatchPoint:
        ns = self.ns
        target = Target(f"chain.ns.{attribute}", lambda: getattr(ns, attribute), lambda f: setattr(ns, attribute, f))
        return PatchPoint(stage, target, {"target": Guard(
            "target", lambda: source_hash(getattr(ns, attribute)), {self._reviewed[attribute]: "chain build 1"},
            f"the {attribute} function being wrapped")})

    def stages(self) -> Mapping[str, StageBinding]:
        if self._stages is None:
            self._stages = {}
            for factory in _BINDINGS:
                binding = factory(self)
                self._stages[binding.contract.name] = binding
        return self._stages

    # -- driving a call, for conformance ----------------------------------------------------------
    def drive(self, stage: str, *, owned: bool = True, seed: int = 0, **_ignored):
        """Run one forward pass on a random state; what the host did with the stage's output is the binding's ``oracle``."""
        binding = self.stages().get(stage)
        if binding is None:
            raise CapabilityError(f"chain host cannot drive {stage}")
        rng, d, m = np.random.default_rng(seed), self.data, self.model
        d.x[...] = rng.standard_normal(tuple(d.x.shape)).astype(np.float32) * 0.1
        d.v[...] = rng.standard_normal(tuple(d.v.shape)).astype(np.float32)
        d.command[...] = rng.standard_normal(tuple(d.command.shape)).astype(np.float32)
        d.t[...] = rng.uniform(0, 5, d.nworld).astype(np.float32)
        for name in ("a", "passive_force", "drive_force"):
            getattr(d, name)[...] = SENTINEL
        self.ns.forward(m if owned else self.other_model, d)
        return dict(effect=binding.oracle(self) if binding.oracle else None, nv=m.nv)


# ---- the stage bindings, as tables ------------------------------------------------------------------

def _damping_wrap(original, host):
    """The assembly adds the law's damping to the chain's own dashpots for this call: the law then acts as the damper does."""
    damping = host.buffers().damping

    def dispatch(m, d):
        if m is not host.model:
            return original(m, d)
        seen = copy.copy(m)
        seen.damping = m.damping + unwrap(damping)
        original(seen, d)

    return dispatch


def _native_call(attribute: str, *extra) -> NativeTiming:
    """Time the host's own function once on its own data: the chain calls these as plain Python, so a call is the stage."""
    return NativeTiming(run=lambda host: lambda: getattr(host.ns, attribute)(host.model, host.data, *[getattr(host.data, e) for e in extra]))


def _passive_oracle(host):
    d = host.data
    baseline = copy.deepcopy(d)
    original_of(host.ns.passive)(host.model, baseline)
    return dict(delta=np.asarray(d.passive_force, np.float64) - np.asarray(baseline.passive_force, np.float64))


def _passive_force(host) -> StageBinding:
    hook = Hook(
        "after", pf.DofForceSystem,
        ports=dict(q=Buf("q"), v=Data("v"), time=Data("t"), scalar=Model("particle_scalar"), force=Buf("force"),
                   damping=When("implicit_damping", Buf("damping"))),
        prepare=(Launch("gather_scalar_q", ("world", "nv"), (Data("x"), Model("particle_scalar")), (Buf("q"),)),
                 Zero(Buf("force")), Zero(When("implicit_damping", Buf("damping")))),
        finish=(Launch("add_scalar_force", ("world", "nv"), (Model("particle_scalar"), Buf("force")), (Data("passive_force"),)),))
    return StageBinding(pf.CONTRACT, host._point(pf.NAME, "passive"), hook.make(pf.CONTRACT.entry),
                        extensions={"implicit_damping": Provision(point=host._point(pf.NAME, "assemble"), wrap=_damping_wrap)},
                        oracle=_passive_oracle, native_timing=_native_call("passive"))


def _control_oracle(host):
    d, m = host.data, host.model
    return dict(actuator_force=np.asarray(d.drive_force), expected_actuator_force=m.gain * np.asarray(d.command, np.float64))


def _control_law(host) -> StageBinding:
    hook = Hook("before", cl.ControlSystem, ports=dict(length=Data("drive_length"), velocity=Data("drive_velocity"),
                                                       time=Data("t"), ctrl=Data("command")))
    return StageBinding(cl.CONTRACT, host._point(cl.NAME, "actuate"), hook.make(cl.CONTRACT.entry), oracle=_control_oracle,
                        native_timing=_native_call("actuate"))


def _probe_state(host) -> StageBinding:
    hook = Hook("after", ps.ProbeSystem, ports=dict(
        qpos=Data("x"), qvel=Data("v"), qacc=Data("a"), time=Data("t"),
        value=Once(lambda host, runtime: host.buffer(f"probe/{runtime.width}", (host.data.nworld, int(runtime.width))))))
    return StageBinding(ps.CONTRACT, host._point(ps.NAME, "forward"), hook.make(ps.CONTRACT.entry))


def _effective_sparsity(host) -> Sparsity:
    return Sparsity.banded(int(host.model.nv), 1)


def _effective_oracle(host):
    """The host's own solve of the system it just assembled, against what is now in ``a``."""
    expected = np.zeros(tuple(host.data.a.shape), np.float32)
    original_of(host.ns.effective_solve)(host.model, host.data, expected)
    return dict(qacc=np.asarray(host.data.a), expected=expected)


def _effective_solve(host) -> StageBinding:
    hook = Hook("instead", es.EffectiveSystem,
                ports=dict(h=Data("H"), rhs=Data("rhs"), qacc=Arg(0),
                           sparsity=When("sparsity", Once(lambda host, runtime: _effective_sparsity(host)))),
                handles=lambda args, kwargs: len(args) == 1 and not kwargs)
    return StageBinding(es.CONTRACT, host._point(es.NAME, "effective_solve"), hook.make(es.CONTRACT.entry),
                        extensions={"sparsity": Provision()}, oracle=_effective_oracle, sparsity=_effective_sparsity,
                        native_timing=_native_call("effective_solve", "a"))


_BINDINGS = [_passive_force, _control_law, _probe_state, _effective_solve]
