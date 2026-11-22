"""The fake host's stages, as data: the same ``Hook`` tables the GPU host uses, with numpy arrays behind the sources.

Adding a stage to the fake host is one more factory like these and a ``register_binding``. A stage that has a binding here
can be driven by ``FakeHost.drive`` and checked by ``check_impl`` with no simulator and no GPU.
"""

from __future__ import annotations

import copy

import numpy as np

from ..host import Provision, StageBinding
from ..hook import Arg, Buf, Data, Hook, Launch, Model, Once, When, Zero
from ..patch import original_of
from ..sparsity import Sparsity
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import effective_solve as es
from ..stages import probe_state as ps
from . import fake as adapter


def _euler_wrap(original, host):
    """Euler integrates the model's damping implicitly: give it the law's damping on top, for this call."""
    damping = host.buffers().damping

    def dispatch(m, d):
        if m is not host.model:
            return original(m, d)
        seen = copy.copy(m)
        seen.dof_damping = (m.dof_damping + damping).astype(np.float32)
        original(seen, d)

    return dispatch


def _passive_oracle(host):
    """The change the stage made to ``qfrc_passive``, against the host's own passive force on the same state."""
    d = host.data
    baseline = copy.deepcopy(d)
    original_of(host.ns.passive)(host.model, baseline)
    return dict(delta=d.qfrc_passive.astype(np.float64) - baseline.qfrc_passive.astype(np.float64))


@adapter.register_binding
def passive_force(host) -> StageBinding:
    hook = Hook(
        "after", pf.DofForceSystem,
        ports=dict(q=Buf("q"), v=Data("qvel"), time=Data("time"), scalar=Model("scalar"), force=Buf("force"),
                   damping=When("implicit_damping", Buf("damping"))),
        prepare=(Launch("gather_scalar_q", ("world", "nv"), (Data("qpos"), Model("scalar")), (Buf("q"),)),
                 Zero(Buf("force")), Zero(When("implicit_damping", Buf("damping")))),
        finish=(Launch("add_scalar_force", ("world", "nv"), (Model("scalar"), Buf("force")), (Data("qfrc_passive"),)),))
    return StageBinding(
        pf.CONTRACT, host._point(pf.NAME, "passive"), hook.make(pf.CONTRACT.entry),
        extensions={"implicit_damping": Provision(point=host._point(pf.NAME, "euler"), wrap=_euler_wrap)},
        oracle=_passive_oracle)


def _control_oracle(host):
    """The actuator force the host computed, against the native law applied to the same ctrl."""
    d = host.data
    ctrl = d.ctrl.astype(np.float64)
    return dict(actuator_force=d.actuator_force, expected_actuator_force=(
        adapter.GAIN * ctrl + adapter.BIAS_LENGTH * d.actuator_length + adapter.BIAS_VELOCITY * d.actuator_velocity))


@adapter.register_binding
def control_law(host) -> StageBinding:
    hook = Hook("before", cl.ControlSystem, ports=dict(length=Data("actuator_length"), velocity=Data("actuator_velocity"),
                                                       time=Data("time"), ctrl=Data("ctrl")))
    return StageBinding(cl.CONTRACT, host._point(cl.NAME, "fwd_actuation"), hook.make(cl.CONTRACT.entry),
                        oracle=_control_oracle)


@adapter.register_binding
def probe_state(host) -> StageBinding:
    hook = Hook("after", ps.ProbeSystem, ports=dict(
        qpos=Data("qpos"), qvel=Data("qvel"), qacc=Data("qacc"), time=Data("time"),
        value=Once(lambda host, runtime: host.buffer(f"probe/{runtime.width}", (host.data.nworld, int(runtime.width))))))
    return StageBinding(ps.CONTRACT, host._point(ps.NAME, "forward"), hook.make(ps.CONTRACT.entry))


def _effective_sparsity(host):
    """The pattern of the fake host's effective system: dense, unless the host was built with a band."""
    n = int(host.model.nv)
    return Sparsity.dense(n) if host.eff_bandwidth is None else Sparsity.banded(n, host.eff_bandwidth)


@adapter.register_binding
def effective_solve(host) -> StageBinding:
    # the host makes one call, eff_solve(m, d, qacc); a call that passes its own right-hand side is not this one
    hook = Hook("instead", es.EffectiveSystem,
                ports=dict(h=Data("eff_h"), rhs=Data("eff_rhs"), qacc=Arg(0),
                           sparsity=When("sparsity", Once(lambda host, runtime: _effective_sparsity(host)))),
                handles=lambda args, kwargs: len(args) == 1 and not kwargs)
    return StageBinding(es.CONTRACT, host._point(es.NAME, "eff_solve"), hook.make(es.CONTRACT.entry),
                        extensions={"sparsity": Provision()}, sparsity=_effective_sparsity)
