"""``body.wrench`` on the fake host: before ``xfrc_accumulate`` folds the wrench on each body into the smooth force.

The same stage the GPU host has (``phost.hosts.mjwarp_body_wrench``), as one table and one oracle. The oracle is the
float64 sum of J^T w over bodies, computed apart from the host's own accumulation.
"""

from __future__ import annotations

import numpy as np

from ..host import StageBinding
from ..hook import Data, Hook
from ..stages import body_wrench as bw
from . import fake as adapter


def _oracle(host):
    """The host's smooth force, against a float64 sum over bodies of J^T w for the wrench that was applied."""
    d, m = host.data, host.model
    pad = np.zeros_like(d.qvel, dtype=np.float64)
    pad[:, :m.nu] = d.actuator_force
    expected = d.qfrc_passive.astype(np.float64) + pad
    for b in range(m.nbody):
        expected += d.xfrc_applied[:, b, :].astype(np.float64) @ m.jac[b]
    return dict(qfrc_smooth=d.qfrc_smooth, expected_qfrc_smooth=expected)


@adapter.register_binding
def body_wrench(host) -> StageBinding:
    hook = Hook("before", bw.WrenchSystem, ports=dict(xpos=Data("xpos"), cvel=Data("cvel"), time=Data("time"),
                                                      wrench=Data("xfrc_applied")))
    return StageBinding(bw.CONTRACT, host._point(bw.NAME, "xfrc_accumulate"), hook.make(bw.CONTRACT.entry), oracle=_oracle)
