"""Needs a MuJoCo + MuJoCo Warp + CUDA environment; skipped otherwise."""

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
    from phost import (Capability, GuardError, Impl, ImplKind, Registry, Selection, Session, StageContract)
    from phost.hosts import mjwarp as adapter
    from phost.stages import newton_linear as nl
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost
    from phost.testing.runtimes import ReferenceRuntime

SCENE = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody><geom type="plane" size="3 3 .1"/>
{bodies}</worldbody></mujoco>"""


def scene(nbox=12):
    bodies = "".join(f'<body pos="{0.3*(i%4)-0.45} {0.3*(i//4)-0.3} {0.1+0.02*i}"><freejoint/>'
                     f'<geom type="box" size=".05 .05 .05" mass=".5"/></body>' for i in range(nbox))
    return SCENE.format(bodies=bodies)


def make_host(nworld=2, nbox=12):
    cm = mujoco.MjModel.from_xml_string(scene(nbox))
    cd = mujoco.MjData(cm)
    for _ in range(60):
        mujoco.mj_step(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=64 * nworld, njmax=512)
    return DrivableMJWarpHost(model, data, cpu_model=cm)


class DeviceReference:
    """Correct-by-construction runtime on the real host: solves in float64 on the CPU."""

    def __init__(self, host, params):
        self.host = host

    def solve(self, s):
        wp.synchronize()
        nv = int(self.host.model.nv)
        h, g, act = s.h.numpy(), s.grad.numpy(), s.active.numpy()
        search, sdot, dec = s.search.numpy(), s.search_dot.numpy(), s.newton_decrement.numpy()
        for w in range(h.shape[0]):
            if act[w]:
                x = np.linalg.solve(nl._symmetric(h[w], nv), g[w, :nv].astype(np.float64))
                search[w], sdot[w], dec[w] = -x, x @ x, g[w, :nv] @ x
        s.search.assign(search); s.search_dot.assign(sdot); s.newton_decrement.assign(dec)

    def metrics(self):
        return {"status": 0, "fallbacks": 0}


@pytest.fixture
def registry():
    reg = Registry()
    nl.register(reg)
    reg.add_impl(Impl(id="test/ref", stage=nl.NAME, stage_version=1, version="1", title="ref", kind=ImplKind.BUILTIN,
                      factory=lambda h, p: DeviceReference(h, p)))
    return reg


@pytest.fixture(scope="module")
def host():
    return make_host()


def reviewed():
    try:
        adapter.Guard("t", adapter._read_hook, adapter.REVIEWED_HOOKS).check()
        return True
    except GuardError:
        return False


needs_reviewed = pytest.mark.skipif(not HAVE or not reviewed(), reason="this MJWarp build is not a reviewed build")


def test_facts_and_info(host):
    f = host.facts()
    assert f.nv == 72 and f.world_count == 2 and f.solver == "newton" and f.device_kind == "cuda"
    assert f.extra["device"].startswith("cuda") and host.info.name == "mjwarp"
    assert Capability.PATCH in host.capabilities


def test_unreviewed_build_is_refused_with_the_hash_found(host, registry, monkeypatch):
    monkeypatch.setattr(adapter, "REVIEWED_HOOKS", {"0" * 64: "nothing"})
    fresh = make_host()
    fresh._stages = None
    with pytest.raises(GuardError, match="unreviewed source sha256 [0-9a-f]{64}"):
        with Session(fresh, [Selection(nl.NAME, "test/ref")], registry):
            pass


@needs_reviewed
def test_native_hook_itself_meets_the_contract(host, registry):
    run = host.drive(nl.NAME)
    metrics = nl.evaluate(run["inputs"], run["outputs"], run["nv"])
    assert metrics["max_relative_residual"] < 1e-4, metrics


@needs_reviewed
def test_a_correct_runtime_on_the_real_host_passes_the_contract_check(host, registry):
    verdict = check_impl(host, Selection(nl.NAME, "test/ref"), registry)
    assert verdict.ok, verdict.failures


@needs_reviewed
def test_install_restores_the_host_and_other_models_reach_the_native_hook(host, registry):
    solver = adapter._solver_module()
    original = solver._cholesky_factorize_solve
    with Session(host, [Selection(nl.NAME, "test/ref")], registry):
        assert solver._cholesky_factorize_solve is not original
        run = host.drive(nl.NAME, owned=False)
        assert nl.evaluate(run["inputs"], run["outputs"], run["nv"])["max_relative_residual"] < 1e-4
    assert solver._cholesky_factorize_solve is original


@needs_reviewed
def test_nested_session_is_refused(host, registry):
    with Session(host, [Selection(nl.NAME, "test/ref")], registry):
        with pytest.raises(GuardError, match="nested"):
            with Session(host, [Selection(nl.NAME, "test/ref")], registry):
                pass


def test_mjlab_wrapper_delegates_to_the_mjwarp_host():
    from types import SimpleNamespace
    base = make_host()
    sim = SimpleNamespace(wp_model=base.model, wp_data=base.data, mj_model=base.cpu_model)
    lab = adapter.MJLabHost(sim)
    assert lab.info.name == "mjlab"
    assert lab.facts().nv == base.facts().nv and lab.cpu_model is base.cpu_model
    from phost import CapabilityError
    with pytest.raises(CapabilityError, match="MJLab"):
        adapter.MJLabHost(SimpleNamespace(wp_model=1))


def test_facts_can_be_read_while_a_graph_is_being_captured():
    """The dispatches ask the host for facts inside a capture, where a device-to-host copy raises. Facts must not read
    the device once the host is built."""
    host = make_host()
    with wp.ScopedDevice("cuda:0"):
        wp.synchronize()
        with wp.ScopedCapture():
            facts = host.facts()
            assert facts.extra["scalar_dofs"] >= 0
