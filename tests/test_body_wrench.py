"""``body.wrench``: a stage added with a contract module, a binding module per host and this file, and nothing else.

It is the held-out test of the stage machinery: its dimension (``body``), its hook (``forward.xfrc_accumulate``) and its
host-side oracle are all new, and ``check_impl``, the test drivers and the session were not touched to make it work. The
tests run on both hosts: the fake host (numpy, no simulator) always, and MuJoCo Warp when mujoco, mujoco_warp and CUDA are
available (its oracle is MuJoCo (C) for the same wrench).
"""

import numpy as np
import pytest

from phost import GuardError, Impl, ImplKind, ParamSpec, Registry, Selection, Session, stages
from phost.guard import source_hash
from phost.stages import body_wrench as bw
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost

try:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    HAVE_GPU = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE_GPU = False

XML = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <body pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>
 <body pos="0.5 0 1.0"><joint type="hinge" axis="0 1 0"/>
   <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/></body>
</worldbody></mujoco>"""


def make_mjwarp_host(nworld=3):
    from phost.testing.mjwarp import DrivableMJWarpHost
    cm = mujoco.MjModel.from_xml_string(XML)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=16 * nworld, njmax=128)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd)


def to_numpy(array):
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)


def put(array, value):
    if hasattr(array, "assign"):
        array.assign(value)
    else:
        array[...] = value


def drag(velocity, c):
    """A linear drag force and a small rotational drag torque, per body: force first, torque second."""
    out = np.zeros(velocity.shape[:-1] + (6,), np.float64)
    out[..., 0:3] = -c * velocity[..., 3:6]
    out[..., 3:6] = -0.1 * c * velocity[..., 0:3]
    return out


class Drag:
    def __init__(self, host, params):
        self.c = params["c"]

    def apply(self, system):
        put(system.wrench, drag(to_numpy(system.cvel).astype(np.float64), self.c).astype(np.float32))

    def metrics(self):
        return {}


class TorqueFirst(Drag):
    """The right law written into the wrong layout: torque then force. It disagrees with its own float64 law."""

    def apply(self, system):
        w = drag(to_numpy(system.cvel).astype(np.float64), self.c)
        put(system.wrench, np.concatenate([w[..., 3:], w[..., :3]], axis=-1).astype(np.float32))


def impl(name, runtime):
    return Impl(id=f"test/{name}", stage=bw.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                params=(ParamSpec("c", "float", 0.7, "drag coefficient"),),
                factory=lambda host, params: runtime(host, params),
                reference=lambda inputs, params: dict(wrench=drag(inputs["cvel"], params["c"])))


@pytest.fixture
def registry():
    reg = Registry()
    stages.register_all(reg)
    reg.add_impl(impl("drag", Drag))
    reg.add_impl(impl("swapped", TorqueFirst))
    return reg


@pytest.fixture(params=["fake", pytest.param("mjwarp", marks=pytest.mark.skipif(not HAVE_GPU, reason="needs mujoco_warp and CUDA"))])
def host(request):
    return FakeHost(nv=12, world_count=3) if request.param == "fake" else make_mjwarp_host()


def test_the_stage_is_registered_with_its_judge_and_a_binding(host, registry):
    assert registry.judge(bw.NAME).repeatable
    assert host.stage(bw.NAME).contract is bw.CONTRACT and host.refusal(bw.NAME) is None


def test_an_implementation_that_applies_the_wrench_the_host_expects_passes(host, registry):
    verdict = check_impl(host, Selection(bw.NAME, "test/drag"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5 and verdict.metrics["max_effect_error"] < 1e-4


def test_a_wrench_in_the_wrong_layout_fails_its_own_law_while_the_host_still_applies_it_faithfully(host, registry):
    verdict = check_impl(host, Selection(bw.NAME, "test/swapped"), registry)
    assert not verdict.ok and any("max_law_error" in f for f in verdict.failures)
    assert not any("max_effect_error" in f for f in verdict.failures)     # the host did what it does with that wrench


def test_the_wrench_actually_moves_the_trajectory(host, registry):
    with Session(host, [Selection(bw.NAME, "test/drag", {"c": 5.0})], registry) as session:
        host.drive(bw.NAME, seed=1)
        assert session.metrics()[bw.NAME]["calls"] >= 1
        assert np.abs(to_numpy(host.data.xfrc_applied)).max() > 0


# ---- the fake host only: a host that applies the wrench wrongly is caught by the oracle, not by the stage ----

def test_a_host_that_applies_the_wrench_with_the_wrong_sign_fails_the_effect_check(registry):
    host = FakeHost(nv=12, world_count=3)

    def flipped(m, d, qfrc):
        qfrc -= np.einsum("bkn,wbk->wn", m.jac, d.xfrc_applied.astype(np.float64)).astype(np.float32)

    host.ns.xfrc_accumulate = flipped
    host._reviewed["xfrc_accumulate"] = source_hash(flipped)         # this build is the one that is reviewed
    verdict = check_impl(host, Selection(bw.NAME, "test/drag"), registry)
    assert not verdict.ok and any("max_effect_error" in f for f in verdict.failures)
    assert not any("max_law_error" in f for f in verdict.failures)       # the implementation did its job


def test_the_fake_host_refuses_an_unreviewed_xfrc_accumulate(registry):
    host = FakeHost(nv=12, world_count=2)
    host._reviewed["xfrc_accumulate"] = "0" * 64
    with pytest.raises(GuardError):
        with Session(host, [Selection(bw.NAME, "test/drag")], registry):
            pass


# ---- MuJoCo Warp only ----

@pytest.mark.skipif(not HAVE_GPU, reason="needs mujoco_warp and CUDA")
def test_native_timing_runs_the_hosts_own_function():
    host = make_mjwarp_host()
    run = host.stage(bw.NAME).native_timing.run(host)
    assert run is not None
    run()
    wp.synchronize()


@pytest.mark.skipif(not HAVE_GPU, reason="needs mujoco_warp and CUDA")
def test_a_build_that_is_not_in_the_reviewed_table_is_refused(registry, monkeypatch):
    from phost.hosts import mjwarp_body_wrench as binding
    monkeypatch.setattr(binding, "REVIEWED_XFRC", {})
    fresh = make_mjwarp_host(nworld=1)
    with pytest.raises(GuardError, match="unreviewed source sha256 f8d20c77"):
        with Session(fresh, [Selection(bw.NAME, "test/drag")], registry):
            pass
