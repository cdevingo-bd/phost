"""``body.wrench``: a stage added with a contract module, a binding module and this file, and nothing else.

It is the held-out test of the stage machinery: its dimension (``body``), its hook (``forward.xfrc_accumulate``) and its host-side
oracle (MuJoCo (C) for the same wrench) are all new, and ``check_impl``, the test driver, the session and the fake host were not
touched to make it work. Needs mujoco, mujoco_warp and CUDA, and a reviewed build.
"""

import pytest

try:
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp
    HAVE = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs mujoco, mujoco_warp and CUDA")

if HAVE:
    from phost import GuardError, Impl, ImplKind, ParamSpec, Registry, Selection, Session, stages
    from phost.hosts import mjwarp_body_wrench as binding
    from phost.stages import body_wrench as bw
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <body pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>
 <body pos="0.5 0 1.0"><joint type="hinge" axis="0 1 0"/>
   <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/></body>
</worldbody></mujoco>"""


def make_host(nworld=3):
    cm = mujoco.MjModel.from_xml_string(XML)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=16 * nworld, njmax=128)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd)


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
        system.wrench.assign(drag(system.cvel.numpy().astype(np.float64), self.c).astype(np.float32))

    def metrics(self):
        return {}


class TorqueFirst(Drag):
    """The right law written into the wrong layout: torque then force. It disagrees with its own float64 law."""

    def apply(self, system):
        w = drag(system.cvel.numpy().astype(np.float64), self.c)
        system.wrench.assign(np.concatenate([w[..., 3:], w[..., :3]], axis=-1).astype(np.float32))


def impl(name, runtime, reference):
    return Impl(id=f"test/{name}", stage=bw.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                params=(ParamSpec("c", "float", 0.7, "drag coefficient"),),
                factory=lambda host, params: runtime(host, params),
                reference=lambda inputs, params: dict(wrench=drag(inputs["cvel"], params["c"])))


@pytest.fixture
def registry():
    reg = Registry()
    stages.register_all(reg)
    reg.add_impl(impl("drag", Drag, None))
    reg.add_impl(impl("swapped", TorqueFirst, None))
    return reg


@pytest.fixture(scope="module")
def host():
    return make_host()


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
    assert not any("max_effect_error" in f for f in verdict.failures)     # the host did what MuJoCo does with that wrench


def test_the_wrench_actually_moves_the_trajectory(host, registry):
    with Session(host, [Selection(bw.NAME, "test/drag", {"c": 5.0})], registry) as session:
        host.drive(bw.NAME, seed=1)
        assert session.metrics()[bw.NAME]["calls"] >= 1
        assert np.abs(host.data.xfrc_applied.numpy()).max() > 0


def test_native_timing_runs_the_hosts_own_function(host):
    run = host.stage(bw.NAME).native_timing.run(host)
    assert run is not None
    run()
    wp.synchronize()


def test_a_build_that_is_not_in_the_reviewed_table_is_refused(registry, monkeypatch):
    monkeypatch.setattr(binding, "REVIEWED_XFRC", {})
    fresh = make_host(nworld=1)
    with pytest.raises(GuardError, match="unreviewed source sha256 f8d20c77"):
        with Session(fresh, [Selection(bw.NAME, "test/drag")], registry):
            pass
