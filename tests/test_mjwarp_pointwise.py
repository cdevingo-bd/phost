"""The pointwise stages on the real MuJoCo Warp host: conformance, then physics against MuJoCo (C).

Needs mujoco, mujoco_warp and CUDA, and a build whose functions are in the reviewed tables.
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
    from phost import GuardError, Registry, Selection, Session
    from phost.hosts import mjwarp as adapter
    from phost.impls import controllers, forces, probes, register_builtin
    from phost.stages import control_law as cl
    from phost.stages import dof_passive_force as pf
    from phost.stages import probe_state as ps
    from phost.stages import register_all
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <body pos="0 0 2"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>
 <body pos="0.5 0 1.0"><joint name="h1" type="hinge" axis="0 1 0" {spring1}/>
   <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/>
   <body pos="0.3 0 0"><joint name="h2" type="hinge" axis="0 1 0" {spring2}/>
     <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/></body></body>
 <body pos="-0.5 0 0.5"><joint name="s1" type="slide" axis="0 0 1" {spring3}/><geom type="sphere" size=".05" mass=".2"/></body>
</worldbody>
<actuator><motor joint="h1" gear="2"/><position joint="h2" kp="5"/></actuator></mujoco>"""
PLAIN = dict(spring1="", spring2="", spring3="")


def cpu_model(**kw):
    return mujoco.MjModel.from_xml_string(XML.format(**{**PLAIN, **kw}))


def make_host(nworld=3, **xml):
    cm = cpu_model(**xml)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=16 * nworld, njmax=128)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd), cm, cd


@pytest.fixture(scope="module")
def host():
    return make_host()[0]


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    return reg


@pytest.fixture(scope="module", autouse=True)
def reviewed(host):
    """These tests need a reviewed build; on any other build the adapter must refuse, which is checked here."""
    stages = host.stages()
    try:
        for name in (pf.NAME, cl.NAME, ps.NAME):
            stages[name].point.guards["target"].check()
    except GuardError:
        pytest.skip("this mujoco_warp build is not in the reviewed tables")


def test_facts_describe_the_model(host):
    extra = host.facts().extra
    assert (extra["nq"], extra["nu"], extra["nhistory"], extra["scalar_dofs"]) == (10, 2, 0, 3)


@pytest.mark.parametrize("impl_id,params", [
    (forces.SPRING_DAMPER_ID, dict(k=3.0, c=0.4, q_ref=0.2)),
    (controllers.PD_ID, dict(kp=5.0, kd=0.3, target=0.5)),
    (probes.PEAK_ID, {}),
])
def test_builtin_implementations_pass_the_contract_on_the_real_host(host, registry, impl_id, params):
    verdict = check_impl(host, Selection(registry.impl(impl_id).stage, impl_id, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5
    if "max_effect_error" in verdict.metrics:
        assert verdict.metrics["max_effect_error"] < 1e-4


def test_other_models_are_left_alone(host, registry):
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=9.0))], registry) as session:
        host.drive(pf.NAME, owned=False)
        assert session.metrics()[pf.NAME]["calls"] == 0


def test_scope_restores_every_function(host, registry):
    from mujoco_warp._src import forward, passive
    before = (passive.passive, forward.fwd_actuation, forward.forward)
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=1.0)), Selection(cl.NAME, controllers.PD_ID),
                        Selection(ps.NAME, probes.PEAK_ID)], registry):
        assert passive.passive is not before[0] and forward.forward is not before[2]
    assert (passive.passive, forward.fwd_actuation, forward.forward) == before


def run_gpu(nsteps, selections=(), registry=None, nworld=2, graph=False, **xml):
    host, cm, cd = make_host(nworld, **xml)
    m, d = host.model, host.data

    def go():
        for _ in range(nsteps):
            mjw.step(m, d)

    with wp.ScopedDevice("cuda:0"):
        if selections:
            with Session(host, selections, registry):
                if graph:
                    mjw.step(m, d)                       # warmup, outside the capture
                    d.qpos.assign(np.tile(cd.qpos, (nworld, 1)).astype(np.float32))
                    d.qvel.assign(np.tile(cd.qvel, (nworld, 1)).astype(np.float32))
                    d.time.zero_()
                    with wp.ScopedCapture() as cap:
                        mjw.step(m, d)
                    for _ in range(nsteps):
                        wp.capture_launch(cap.graph)
                else:
                    go()
        else:
            go()
        wp.synchronize()
        return d.qpos.numpy()[0].astype(np.float64)


def cpu_truth(k, c, native_damper, steps=150):
    """MuJoCo (C, float64): springs in the model; damping either native (implicit in Euler) or applied explicitly."""
    joint = f'stiffness="{k}"' + (f' damping="{c}"' if native_damper else "")
    model = cpu_model(spring1=joint, spring2=joint, spring3=joint)
    data = mujoco.MjData(model)
    dofs = [6, 7, 8]                                             # h1, h2, s1 (the free joint holds DOFs 0..5)
    for _ in range(steps):
        if not native_damper:
            data.qfrc_applied[dofs] = -c * data.qvel[dofs]
        mujoco.mj_step(model, data)
    return data.qpos


SCALAR = [7, 8, 9]                                               # h1, h2, s1 in qpos (the free joint holds 0..6)


def err(a, b):
    return np.abs(a[SCALAR] - b[SCALAR]).max()


def test_spring_law_matches_mujoco_springs():
    """A position-dependent law is equivalent to the same spring in the model: it must match as closely as MJWarp's
    own native springs do (the control), and the stage must have changed the motion."""
    k = 6.0
    registry = Registry(); register_all(registry); register_builtin(registry)
    truth = cpu_truth(k, 0.0, native_damper=True)
    staged = run_gpu(150, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=0.0))], registry)
    joint = f'stiffness="{k}"'
    control = run_gpu(150, spring1=joint, spring2=joint, spring3=joint)      # same spring, written into the model
    assert err(staged, truth) <= 3 * err(control, truth) + 1e-6, (err(staged, truth), err(control, truth))
    assert err(run_gpu(150), truth) > 100 * max(err(staged, truth), 1e-6)


def test_damping_law_matches_an_explicit_damper():
    """The stage applies its force explicitly, so the truth is MuJoCo with the damper applied explicitly."""
    k, c = 6.0, 0.15
    registry = Registry(); register_all(registry); register_builtin(registry)
    explicit = cpu_truth(k, c, native_damper=False)
    staged = run_gpu(150, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=c))], registry)
    assert err(staged, explicit) < 1e-5


def test_native_implicit_damping_differs_from_the_explicit_law_by_a_small_documented_amount():
    """Records the semantic difference the stage's documentation states (O(timestep), about 0.1% of displacement here)."""
    k, c = 6.0, 0.15
    registry = Registry(); register_all(registry); register_builtin(registry)
    staged = run_gpu(150, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=c))], registry)
    gap = err(staged, cpu_truth(k, c, native_damper=True))
    assert 1e-4 < gap < 1e-2, gap


def test_controller_matches_a_mujoco_pd_loop():
    kp, kd, target = 4.0, 0.3, 0.2
    cm = cpu_model()
    cd = mujoco.MjData(cm)
    for _ in range(150):
        mujoco.mj_forward(cm, cd)                                # actuator length and velocity at the current state
        cd.ctrl[:] = kp * (target - cd.actuator_length) - kd * cd.actuator_velocity
        mujoco.mj_step(cm, cd)
    registry = Registry(); register_all(registry); register_builtin(registry)
    got = run_gpu(150, [Selection(cl.NAME, controllers.PD_ID, dict(kp=kp, kd=kd, target=target))], registry)
    assert np.abs(got[[7, 8, 9]] - cd.qpos[[7, 8, 9]]).max() < 5e-4
    assert np.abs(run_gpu(150)[[7, 8, 9]] - cd.qpos[[7, 8, 9]]).max() > 5e-3      # the stage did something


def test_probe_does_not_change_the_trajectory_and_reports_values():
    registry = Registry(); register_all(registry); register_builtin(registry)
    native = run_gpu(60)
    probed = run_gpu(60, [Selection(ps.NAME, probes.PEAK_ID)], registry)
    assert np.array_equal(native, probed)


def test_captured_graph_replays_the_installed_stages():
    registry = Registry(); register_all(registry); register_builtin(registry)
    selections = [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=6.0, c=0.15)),
                  Selection(cl.NAME, controllers.PD_ID, dict(kp=4.0, kd=0.3, target=0.2)),
                  Selection(ps.NAME, probes.PEAK_ID)]
    eager = run_gpu(40, selections, registry)
    replayed = run_gpu(40, selections, registry, graph=True)
    assert np.abs(eager - replayed).max() < 1e-5
