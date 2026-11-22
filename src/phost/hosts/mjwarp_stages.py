"""The stages the MuJoCo Warp adapter ships, as data: where each one attaches, how its ports are derived, when it applies.

Each factory below builds one ``StageBinding`` for a host. The binding is a table (see ``_dispatch``) plus, per stage:

* the patch point and the reviewed hashes that guard it (the tables live in ``mjwarp.py``, which is where a build is
  reviewed);
* ``applicable``: why the stage cannot be installed on this model, or None;
* ``native_timing``: how to run the host's own version once, so a tool can say what the stage is worth;
* ``oracle``: what a contract check reads off the host after a run, for the judge's host-side metric.

Adding a stage to this host is one more factory in a module like this one and a call to ``register_binding``.
"""

from __future__ import annotations

from ..host import NativeTiming, Provision, StageBinding
from ..patch import original_of
from ..stages import control_law as cl
from ..stages import dof_passive_force as pf
from ..stages import effective_solve as es
from ..stages import probe_state as ps
from . import mjwarp as adapter
from ..hook import Arg, Buf, Data, Dim, Hook, Launch, Model, Once, When, Zero


def _timing(module, attribute) -> NativeTiming:
    """Time ``module.attribute(m, d)`` as the host runs it, even while a stage is installed on it."""
    return NativeTiming(lambda host: (lambda: original_of(getattr(module, attribute))(host.model, host.data)))


# ---- dof.passive_force: after passive.passive, add a force per scalar DOF --------------------------------------------

def _passive_oracle(host):
    """The change the stage made to ``qfrc_passive``, against the host's own passive force on the same state."""
    import numpy as np
    import warp as wp
    m, d = host.model, host.data
    actual = d.qfrc_passive.numpy().astype(np.float64)
    original_of(adapter._passive_module().passive)(m, d)
    wp.synchronize()
    return dict(delta=actual - d.qfrc_passive.numpy().astype(np.float64))


@adapter.register_binding
def passive_force(host) -> StageBinding:
    forward, passive = adapter._forward_module(), adapter._passive_module()
    hook = Hook(
        "after", pf.DofForceSystem,
        ports=dict(q=Buf("q"), v=Data("qvel"), time=Data("time"), scalar=Buf("scalar"), force=Buf("force"),
                   damping=When("implicit_damping", Buf("damping"))),
        prepare=(Launch("gather_scalar_q", ("world", "nv"), (Data("qpos"), Buf("dof_qposadr")), (Buf("q"),)),
                 Zero(Buf("force")), Zero(When("implicit_damping", Buf("damping")))),
        finish=(Launch("add_scalar_force", ("world", "nv"), (Buf("scalar"), Buf("force")), (Data("qfrc_passive"),)),))
    return StageBinding(
        pf.CONTRACT, adapter._point(pf.NAME, passive, "passive", adapter.REVIEWED_PASSIVE), hook.make(pf.CONTRACT.entry),
        extensions={"implicit_damping": Provision(
            point=adapter._point(pf.NAME, forward, "euler", adapter.REVIEWED_EULER, also=adapter._EULER[1:]),
            wrap=adapter._wrap_euler)},
        native_timing=_timing(passive, "passive"), oracle=_passive_oracle)


# ---- control.law: before fwd_actuation, write ctrl -----------------------------------------------------------------

def _control_oracle(host):
    """The actuator force the host computed, against what MuJoCo (C, float64) computes for the same ctrl."""
    import mujoco
    import numpy as np
    d, nu, n = host.data, int(host.model.nu), int(host.data.nworld)
    if not nu:
        return None
    if host.cpu_model is None:
        raise ValueError("driving control.law needs cpu_model for the float64 actuator-force reference")
    cpu, expected = mujoco.MjData(host.cpu_model), np.zeros((n, nu))
    q, v, c, t = d.qpos.numpy(), d.qvel.numpy(), d.ctrl.numpy(), d.time.numpy()
    for w in range(n):
        cpu.qpos[:], cpu.qvel[:], cpu.ctrl[:], cpu.time = q[w], v[w], c[w], t[w]
        mujoco.mj_forward(host.cpu_model, cpu)
        expected[w] = cpu.actuator_force
    return dict(actuator_force=d.actuator_force.numpy(), expected_actuator_force=expected)


@adapter.register_binding
def control_law(host) -> StageBinding:
    forward = adapter._forward_module()
    hook = Hook("before", cl.ControlSystem, ports=dict(length=Data("actuator_length"), velocity=Data("actuator_velocity"),
                                                       time=Data("time"), ctrl=Data("ctrl")))
    return StageBinding(cl.CONTRACT, adapter._point(cl.NAME, forward, "fwd_actuation", adapter.REVIEWED_ACTUATION),
                        hook.make(cl.CONTRACT.entry), native_timing=_timing(forward, "fwd_actuation"),
                        oracle=_control_oracle)


# ---- probe.state: after forward, read the state --------------------------------------------------------------------

@adapter.register_binding
def probe_state(host) -> StageBinding:
    forward = adapter._forward_module()
    hook = Hook("after", ps.ProbeSystem, ports=dict(
        qpos=Data("qpos"), qvel=Data("qvel"), qacc=Data("qacc"), time=Data("time"),
        value=Once(lambda host, runtime: host.probe_buffer(int(runtime.width)))))
    return StageBinding(ps.CONTRACT, adapter._point(ps.NAME, forward, "forward", adapter.REVIEWED_FORWARD),
                        hook.make(ps.CONTRACT.entry))        # an observer attaches after the whole forward: nothing to time


# ---- effective.solve: instead of derivative.eff_solve ----------------------------------------------------------------

def _effective_oracle(host):
    """The solve the stage did, against the host's own solve of the same state."""
    import numpy as np
    import warp as wp
    from mujoco_warp._src import derivative
    m, d = host.model, host.data
    actual = d.qacc_smooth.numpy().astype(np.float64)
    scratch = wp.zeros_like(d.qacc_smooth)
    original_of(derivative.eff_solve)(m, d, scratch)
    wp.synchronize()
    return dict(qacc=actual, expected=scratch.numpy().astype(np.float64))


def _effective_sparsity(host):
    """The pattern of M_hat = M + diag + K, read from the model's own index arrays: where the assembly kernel writes."""
    from ..sparsity import Sparsity
    if host._effective_blocker:
        return None
    m, nv = host.model, int(host.model.nv)
    m_rowadr, m_col = m.M_mulm_rowadr.numpy(), m.M_mulm_col.numpy()
    k_rownnz, k_rowadr, k_colind = m.efm_K_rownnz.numpy(), m.efm_K_rowadr.numpy(), m.efm_K_colind.numpy()
    pairs = []
    for row in range(nv):
        pairs.extend((row, int(c)) for c in m_col[m_rowadr[row]:m_rowadr[row + 1]])
        pairs.extend((row, int(c)) for c in k_colind[k_rowadr[row]:k_rowadr[row] + k_rownnz[row]])
    return Sparsity.from_pairs(nv, pairs)


@adapter.register_binding
def effective_solve(host) -> StageBinding:
    derivative = adapter._derivative_module()
    matrix, rhs = Once(lambda host, runtime: host.effective_buffers().matrix), Once(lambda host, runtime: host.effective_buffers().rhs)
    hook = Hook(
        "instead", es.EffectiveSystem,
        ports=dict(h=matrix, rhs=rhs, qacc=Arg(0), sparsity=When("sparsity", Once(lambda host, runtime: _effective_sparsity(host)))),
        # the host makes one call, eff_solve(m, d, qacc); a call that passes its own right-hand side is not this one
        handles=lambda args, kwargs: len(args) == 1 and not kwargs,
        prepare=(Launch("effective_rhs", ("world", "nv"), (Data("qfrc_smooth"), Data("efm_c"), Data("efm_ca")), (rhs,)),
                 Launch("effective_assemble", ("world", "nv"),
                        (Model("M_mulm_rowadr"), Model("M_mulm_col"), Model("M_mulm_madr"), Model("efm_K_rownnz"),
                         Model("efm_K_rowadr"), Model("efm_K_colind"), Dim("nv"), Data("M"), Data("efm_diag"),
                         Data("efm_K_val")), (matrix,))))
    return StageBinding(
        es.CONTRACT, adapter._point(es.NAME, derivative, "eff_solve", adapter.REVIEWED_EFFECTIVE, also=adapter._EFFECTIVE[1:]),
        hook.make(es.CONTRACT.entry), applicable=lambda host: host._effective_blocker or None,
        native_timing=NativeTiming(lambda host: None if host._effective_blocker else (
            lambda: original_of(derivative.eff_solve)(host.model, host.data, host.data.qacc_smooth))),
        oracle=_effective_oracle, extensions={"sparsity": Provision()}, sparsity=_effective_sparsity)
