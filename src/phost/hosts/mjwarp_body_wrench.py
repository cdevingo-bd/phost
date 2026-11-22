"""``body.wrench`` on MuJoCo Warp: before ``forward.xfrc_accumulate`` folds ``d.xfrc_applied`` into ``qfrc_smooth``.

Reviewed in the mujoco_warp 3.17.0 source build: ``fwd_acceleration`` builds ``qfrc_smooth`` from the passive, actuator, bias
and applied generalized forces, then calls ``xfrc_accumulate(m, d, d.qfrc_smooth)``, which maps ``d.xfrc_applied`` (force then
torque per body, in the world frame) through the Jacobians and adds it. The name is bound into ``forward``'s namespace, so
replacing that attribute changes every call the step makes. The stage writes ``d.xfrc_applied`` and calls the original.
"""

from __future__ import annotations

from ..host import NativeTiming, StageBinding
from ..patch import original_of
from ..stages import body_wrench as bw
from . import mjwarp as adapter
from ..hook import Data, Hook, Model

REVIEWED_XFRC = {"f8d20c77e72530e6f4f8e8fbbf52fdf20b274c44ced82fa1bf25c6a3c7c63387": adapter._BUILD}


def _oracle(host):
    """The host's smooth force, against MuJoCo (C, float64) for the same state and the wrench that was applied."""
    import mujoco
    import numpy as np
    d, n = host.data, int(host.data.nworld)
    if host.cpu_model is None:
        raise ValueError("driving body.wrench needs cpu_model for the float64 reference")
    cpu, expected = mujoco.MjData(host.cpu_model), np.zeros((n, int(host.model.nv)))
    q, v, c, t, wrench = d.qpos.numpy(), d.qvel.numpy(), d.ctrl.numpy(), d.time.numpy(), d.xfrc_applied.numpy()
    for w in range(n):
        cpu.qpos[:], cpu.qvel[:], cpu.time = q[w], v[w], t[w]
        if c.shape[1]:
            cpu.ctrl[:] = c[w]
        cpu.xfrc_applied[:] = wrench[w]
        mujoco.mj_forward(host.cpu_model, cpu)
        expected[w] = cpu.qfrc_smooth
    return dict(qfrc_smooth=d.qfrc_smooth.numpy(), expected_qfrc_smooth=expected)


@adapter.register_binding
def body_wrench(host) -> StageBinding:
    forward = adapter._forward_module()
    hook = Hook("before", bw.WrenchSystem, ports=dict(xpos=Data("xpos"), cvel=Data("cvel"), time=Data("time"),
                                                      wrench=Data("xfrc_applied")))
    return StageBinding(
        bw.CONTRACT, adapter._point(bw.NAME, forward, "xfrc_accumulate", REVIEWED_XFRC), hook.make(bw.CONTRACT.entry),
        native_timing=NativeTiming(lambda host: (lambda: original_of(forward.xfrc_accumulate)(host.model, host.data,
                                                                                               host.data.qfrc_smooth))),
        oracle=_oracle)
