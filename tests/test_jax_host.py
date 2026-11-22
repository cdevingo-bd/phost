"""The chain over JAX arrays: the same stages and implementations on a host that cannot be written in place, eager and compiled."""

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="needs jax")

from phost import Impl, ImplKind, Registry, Selection, Session
from phost.hosts.chain import ChainHost, make_model
from phost.hosts.jaxchain import JaxChainHost, thomas
from phost.impls import controllers, forces, probes, register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import effective_solve as es
from phost.stages import probe_state as ps
from phost.stages import register_all
from phost.testing import runtimes
from phost.testing.contract import check_impl


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    return reg


def make(cls=JaxChainHost, n=24, nu=3, worlds=3, **kw):
    return cls(make_model(n, nu, **kw), worlds=worlds)


def pluck(host, at=6, size=0.1):
    host.load(np.zeros(host.model.nv))
    host.data.x[:, at] = size


def jax_effective(registry, name, cls=runtimes.EffectiveReferenceRuntime):
    registry.add_impl(Impl(id=f"test/{name}", stage=es.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                           factory=lambda host, params: cls(host, params)))
    return Selection(es.NAME, f"test/{name}")


def test_the_host_declares_its_arrays_and_offers_the_chains_stages():
    host = make()
    assert host.backend == "jax" and host.info.name == "jaxchain"
    assert sorted(host.stages()) == [cl.NAME, pf.NAME, es.NAME, ps.NAME]


def test_the_tridiagonal_solve_by_scans_matches_a_dense_solve():
    rng = np.random.default_rng(1)
    diag, off, rhs = rng.uniform(2, 3, (4, 9)), rng.uniform(-0.5, 0.5, (4, 8)), rng.standard_normal((4, 9))
    got = np.asarray(thomas(diag.astype(np.float32), off.astype(np.float32), rhs.astype(np.float32)))
    for w in range(4):
        a = np.diag(diag[w]) + np.diag(off[w], 1) + np.diag(off[w], -1)
        assert np.allclose(a @ got[w], rhs[w], atol=1e-4)
    assert np.allclose(np.asarray(thomas(np.full((2, 1), 2.0), np.zeros((2, 0)), np.full((2, 1), 4.0))), 2.0)


def test_eager_steps_follow_the_numpy_chain():
    jax_host, np_host = make(), make(ChainHost)
    for host in (jax_host, np_host):
        pluck(host)
        for _ in range(80):
            host.step()
    assert np.allclose(np.asarray(jax_host.data.x), np_host.data.x, atol=1e-6)


def test_the_compiled_step_matches_the_eager_one_and_the_state_round_trips():
    eager, compiled = make(), make()
    for host in (eager, compiled):
        pluck(host)
    for _ in range(50):
        eager.step()
    step = jax.jit(compiled.pure_step)
    state = compiled.get_state()
    for _ in range(50):
        state = step(state)
    compiled.set_state(state)
    assert np.allclose(np.asarray(compiled.data.x), np.asarray(eager.data.x), atol=1e-6)
    assert np.allclose(np.asarray(compiled.data.t), 0.5, atol=1e-5)


@pytest.mark.parametrize("impl_id,params", [
    (forces.SPRING_DAMPER_ID, dict(k=3.0, c=0.4, q_ref=0.2)),
    (controllers.PD_ID, dict(kp=5.0, kd=0.3, target=0.5)),
    (probes.PEAK_ID, {}),
])
def test_the_builtin_implementations_pass_their_contracts_on_jax_arrays(registry, impl_id, params):
    verdict = check_impl(make(), Selection(registry.impl(impl_id).stage, impl_id, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5


def test_the_effective_solve_contract_holds_on_jax_arrays(registry):
    verdict = check_impl(make(), jax_effective(registry, "ref"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-4


def test_a_law_that_reports_its_damping_integrates_like_the_chains_own_dashpot_on_jax_arrays(registry):
    c, k = 0.8, 0.3
    own, native = make(damping=0.0), make(damping=c)
    for host in (own, native):
        pluck(host)
    with Session(native, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=0.0))], registry):
        for _ in range(30):
            native.step()
    with Session(own, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=c))], registry):
        for _ in range(30):
            own.step()
    assert np.allclose(np.asarray(own.data.x), np.asarray(native.data.x), rtol=1e-3, atol=1e-6)


def test_installed_stages_are_compiled_into_the_step_and_every_replay_is_counted(registry):
    host = make()
    pluck(host)
    law = Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=0.0, c=5.0))
    probe = Selection(ps.NAME, probes.PEAK_ID, {})
    reference = make()
    pluck(reference)
    with Session(reference, [law], registry):
        for _ in range(40):
            reference.step()
    with Session(host, [law, probe], registry) as session:
        host.step()                                             # the warm-up a driver does: it creates the stages' buffers
        pluck(host)
        session.reset_metrics()
        step = jax.jit(host.pure_step)
        state = host.get_state()
        for _ in range(40):
            state = step(state)
        host.set_state(state)
        calls = session.metrics()
    assert np.allclose(np.asarray(host.data.x), np.asarray(reference.data.x), atol=1e-6)      # the law is in the compiled step
    assert calls[pf.NAME]["calls"] == 40 and calls[ps.NAME]["calls"] == 40                    # replays, not traces
    assert calls[ps.NAME]["last_max_qvel"] > 0


def test_an_installed_law_that_adds_nothing_leaves_the_compiled_trajectory_where_the_native_one_is(registry):
    plain, hooked = make(), make()
    for host in (plain, hooked):
        pluck(host)
    with Session(hooked, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=0.0, c=0.0))], registry):
        step = jax.jit(hooked.pure_step)
        state = hooked.get_state()
        hooked.step()
        pluck(hooked)
        state = hooked.get_state()
        for _ in range(20):
            state = step(state)
        hooked.set_state(state)
    for _ in range(20):
        plain.step()
    assert np.allclose(np.asarray(hooked.data.x), np.asarray(plain.data.x), atol=1e-6)
