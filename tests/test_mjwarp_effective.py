"""``effective.solve`` on the real MuJoCo Warp host: the assembled matrix is the one the host iterates with.

Needs mujoco, mujoco_warp and CUDA, and a build whose functions are in the reviewed tables. The runtime used here
solves on the CPU in float64, so the checks are about what the host hands over and takes back, not about speed.
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
    from phost import GuardError, Impl, ImplKind, Registry, Selection, Session
    from phost.hosts import mjwarp as adapter
    from phost.stages import effective_solve as es
    from phost.testing import runtimes
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco model="sheet"><option timestep="0.002" integrator="{integrator}"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <flexcomp name="sheet" type="grid" dim="2" count="6 6 1" spacing="0.05 0.05 0.1" pos="0 0 .3" mass=".2" radius="5e-3">
   <pin gridrange="0 0 5 0"/><pin gridrange="0 5 5 5"/><pin gridrange="0 0 0 5"/><pin gridrange="5 0 5 5"/>
   <elasticity elastic2d="both" poisson="0" thickness="2e-3" young="1e5" damping="0.01"/>
 </flexcomp>
 <body pos="0.12 0.12 0.6"><freejoint/><geom type="sphere" size="0.03" mass=".1"/></body>
</worldbody></mujoco>"""
PLAIN = """<mujoco><option integrator="{integrator}" timestep="0.002"/><worldbody><geom type="plane" size="3 3 .1"/>
<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>"""


def make_host(xml=XML, integrator="discrete", nworld=3):
    cm = mujoco.MjModel.from_xml_string(xml.format(integrator=integrator))
    cd = mujoco.MjData(cm)
    for _ in range(10):
        mujoco.mj_step(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=64 * nworld, njmax=512)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd)


@pytest.fixture
def registry():
    reg = Registry()
    es.register(reg)
    reg.add_impl(Impl(id="test/reference", stage=es.NAME, stage_version=1, version="1", title="reference", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: runtimes.EffectiveReferenceRuntime(host, params)))
    reg.add_impl(Impl(id="test/wrong", stage=es.NAME, stage_version=1, version="1", title="wrong", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: runtimes.WrongEffectiveRuntime(host, params)))
    return reg


@pytest.fixture(scope="module")
def host():
    return make_host()


def test_the_flex_model_with_the_discrete_integrator_can_be_replaced(host):
    assert host.facts().extra["effective_blocker"] == ""
    assert host.facts().integrator == "discrete"


@pytest.mark.parametrize("xml, integrator, words", [
    (XML, "Euler", "discrete integrator"),
    (PLAIN, "discrete", "no flex stiffness"),
])
def test_a_model_the_host_solves_another_way_names_why(xml, integrator, words):
    blocker = make_host(xml, integrator, nworld=1).facts().extra["effective_blocker"]
    assert words in blocker


def test_the_host_hands_over_the_matrix_it_iterates_with(host, registry):
    """The strongest check: a float64 solve of the assembled matrix equals the host's own solve of the same state."""
    verdict = check_impl(host, Selection(es.NAME, "test/reference"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_native_difference"] < 1e-3
    assert verdict.metrics["max_relative_residual"] < 1e-5


def test_an_assembly_that_differs_from_the_hosts_would_be_caught(host, registry):
    verdict = check_impl(host, Selection(es.NAME, "test/wrong"), registry)
    assert not verdict.ok and any("max_native_difference" in f for f in verdict.failures)


def test_the_hook_is_restored_and_counted(host, registry):
    from mujoco_warp._src import derivative
    original = derivative.eff_solve
    with Session(host, [Selection(es.NAME, "test/reference")], registry) as session:
        assert derivative.eff_solve is not original
        host.drive(es.NAME)
        assert session.metrics()[es.NAME]["calls"] >= 1
    assert derivative.eff_solve is original


def test_a_build_that_is_not_in_the_reviewed_table_is_refused(registry, monkeypatch):
    monkeypatch.setattr(adapter, "REVIEWED_EFFECTIVE", {})
    fresh = make_host(nworld=1)                      # a host reads the table when its stage bindings are first built
    with pytest.raises(GuardError, match="unreviewed source sha256 810004a7"):
        with Session(fresh, [Selection(es.NAME, "test/reference")], registry):
            pass
