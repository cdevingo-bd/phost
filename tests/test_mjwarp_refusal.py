"""On a mujoco_warp build whose functions are not in the reviewed tables, installing a stage must be refused."""

import pytest

try:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    HAVE = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs mujoco, mujoco_warp and CUDA")

if HAVE:
    from phost import GuardError, Registry, Selection, Session
    from phost.impls import controllers, forces, probes, register_builtin
    from phost.stages import control_law as cl
    from phost.stages import dof_passive_force as pf
    from phost.stages import probe_state as ps
    from phost.stages import register_all
    from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size=".1"/></body></worldbody>
<actuator><motor joint="j"/></actuator></mujoco>"""

SELECTIONS = (("passive", "REVIEWED_PASSIVE", lambda: Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=1.0))),
              ("fwd_actuation", "REVIEWED_ACTUATION", lambda: Selection(cl.NAME, controllers.PD_ID)),
              ("forward", "REVIEWED_FORWARD", lambda: Selection(ps.NAME, probes.PEAK_ID)))


def test_each_stage_is_installed_on_a_reviewed_build_and_refused_otherwise():
    from phost import source_hash
    from phost.hosts import mjwarp as adapter
    from mujoco_warp._src import forward, passive
    cm = mujoco.MjModel.from_xml_string(XML)
    cd = mujoco.MjData(cm)
    with wp.ScopedDevice("cuda:0"):
        host = DrivableMJWarpHost(mjw.put_model(cm), mjw.put_data(cm, cd, nworld=1), cpu_model=cm)
    registry = Registry(); register_all(registry); register_builtin(registry)
    before = (passive.passive, forward.fwd_actuation, forward.forward)
    for attribute, table, selection in SELECTIONS:
        module = passive if attribute == "passive" else forward
        is_reviewed = source_hash(getattr(module, attribute)) in getattr(adapter, table)
        if is_reviewed:
            with Session(host, [selection()], registry):
                assert (passive.passive, forward.fwd_actuation, forward.forward) != before
        else:
            with pytest.raises(GuardError, match="unreviewed source sha256"):
                Session(host, [selection()], registry).__enter__()
        assert (passive.passive, forward.fwd_actuation, forward.forward) == before     # nothing left installed


def test_an_unreviewed_euler_degrades_implicit_damping_to_explicit_and_says_why(monkeypatch):
    """The law's own stage is installed, but Euler is left alone and the session records why: the damping stays explicit."""
    from phost.hosts import mjwarp as adapter
    from mujoco_warp._src import forward
    monkeypatch.setattr(adapter, "REVIEWED_EULER", {})            # as if this build's euler were not reviewed
    cm = mujoco.MjModel.from_xml_string(XML)
    cd = mujoco.MjData(cm)
    with wp.ScopedDevice("cuda:0"):
        host = DrivableMJWarpHost(mjw.put_model(cm), mjw.put_data(cm, cd, nworld=1), cpu_model=cm)
    registry = Registry(); register_all(registry); register_builtin(registry)
    euler = forward.euler
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(c=0.5))], registry) as session:
        assert forward.euler is euler
        assert any("running without extension 'implicit_damping'" in note and "unreviewed source sha256" in note
                   for note in session.advisories), session.advisories
    assert forward.euler is euler
