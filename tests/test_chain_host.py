"""The chain host: a NumPy spring-mass simulator that shares no vocabulary with MuJoCo, run through the same stage contracts.

If the stage contracts, the table-driven bindings, the implementations that ship with phost and the contract checker only
worked for the simulator they were first written against, these tests would fail here. No GPU, no MuJoCo.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from phost import CapabilityError, Impl, ImplKind, Registry, Selection, Session
from phost.hosts.chain import ChainHost, make_model, thomas
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


def chain(n=24, nu=3, worlds=3, **kw):
    return ChainHost(make_model(n, nu, **kw), worlds=worlds)


def pluck(host, at=5, size=0.1):
    host.load(np.zeros(host.model.nv))
    host.data.x[:, at] = size


def add_effective(registry, name, cls=runtimes.EffectiveReferenceRuntime, **kw):
    registry.add_impl(Impl(id=f"test/{name}", stage=es.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                           factory=lambda host, params: cls(host, params), **kw))
    return Selection(es.NAME, f"test/{name}")


# ---- it is a simulator, and not MuJoCo -----------------------------------------------------------------

def test_the_host_offers_the_stages_a_chain_has_and_no_others():
    host = chain()
    assert sorted(host.stages()) == [cl.NAME, pf.NAME, es.NAME, ps.NAME]
    with pytest.raises(CapabilityError, match="no stage 'body.wrench'"):
        host.stage("body.wrench")
    assert host.backend == "numpy" and host.info.name == "chain"


def test_importing_the_host_loads_neither_mujoco_nor_warp():
    code = ("import sys, phost.hosts.chain, phost.stages, phost.impls; "
            "bad = [m for m in ('mujoco', 'mujoco_warp', 'warp', 'jax') if m in sys.modules]; "
            "print(bad); raise SystemExit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_plucked_chain_rings_down_and_stays_finite():
    host = chain(damping=0.5)
    pluck(host)
    energy = []
    for _ in range(300):
        host.step()
        m, d = host.model, host.data
        stretch = d.x.astype(np.float64) - np.concatenate([np.zeros((d.nworld, 1)), d.x[:, :-1]], axis=1)
        energy.append(float((0.5 * m.mass * d.v ** 2).sum() + (0.5 * m.stiffness * stretch ** 2).sum()))
    assert np.isfinite(host.data.x).all()
    assert energy[-1] < 0.05 * max(energy)                    # damped
    assert np.allclose(host.data.t, 300 * host.model.timestep, rtol=1e-4)


def test_the_tridiagonal_solve_matches_a_dense_solve():
    rng = np.random.default_rng(0)
    diag, off, rhs = rng.uniform(2, 3, (4, 9)), rng.uniform(-0.5, 0.5, (4, 8)), rng.standard_normal((4, 9))
    got = thomas(diag, off, rhs)
    for w in range(4):
        a = np.diag(diag[w]) + np.diag(off[w], 1) + np.diag(off[w], -1)
        assert np.allclose(a @ got[w], rhs[w], atol=1e-12)


def test_the_assembled_system_is_what_the_host_solves():
    host = chain(n=12, nu=2)
    pluck(host, 3)
    host.forward()
    d, n = host.data, host.model.nv
    upper = np.triu(d.H[0].astype(np.float64))
    assert not np.isnan(upper).any() and np.isnan(d.H[0, 5, 2])             # upper defined, lower left undefined
    a = upper + upper.T - np.diag(np.diag(upper))
    assert np.allclose(a @ d.a[0], d.rhs[0], atol=1e-5)
    assert np.allclose(np.linalg.eigvalsh(a) > 0, True)


# ---- the stage contracts hold here ---------------------------------------------------------------------

@pytest.mark.parametrize("impl_id,params", [
    (forces.SPRING_DAMPER_ID, dict(k=3.0, c=0.4, q_ref=0.2)),
    (controllers.PD_ID, dict(kp=5.0, kd=0.3, target=0.5)),
    (probes.PEAK_ID, {}),
])
def test_the_builtin_implementations_pass_their_contracts(registry, impl_id, params):
    verdict = check_impl(chain(), Selection(registry.impl(impl_id).stage, impl_id, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-6


def test_the_effective_solve_contract_is_judged_against_the_hosts_own_banded_solve(registry):
    verdict = check_impl(chain(), add_effective(registry, "ref"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-5
    assert verdict.metrics["max_native_difference"] < 1e-4


def test_an_implementation_that_reads_the_lower_triangle_is_caught(registry):
    class Lower(runtimes.EffectiveReferenceRuntime):
        def solve(self, s):
            a = np.asarray(s.h[:, :self.host.facts().nv, :self.host.facts().nv], np.float64)
            s.qacc[:] = np.linalg.solve(np.triu(a) + np.tril(a, -1), np.asarray(s.rhs, np.float64)[..., None])[..., 0]

    verdict = check_impl(chain(), add_effective(registry, "lower", Lower), registry)
    assert not verdict.ok                                                   # the lower triangle is NaN here


def test_a_wrong_solve_is_caught(registry):
    verdict = check_impl(chain(), add_effective(registry, "wrong", runtimes.WrongEffectiveRuntime), registry)
    assert not verdict.ok and any("max_relative_residual" in f for f in verdict.failures)


def test_the_hosts_band_reaches_an_implementation_that_asks_for_it(registry):
    seen = []

    class Looking(runtimes.EffectiveReferenceRuntime):
        def solve(self, s):
            seen.append(s.sparsity)
            super().solve(s)

    host = chain(n=16)
    selection = add_effective(registry, "looking", Looking, extensions_used=("sparsity",))
    with Session(host, [selection], registry):
        pluck(host)
        host.step()
    assert seen and seen[0].n == 16 and seen[0].bandwidth() == 1
    assert seen[0].describe()["bandwidth"] == 1


def test_an_installed_solve_leaves_the_trajectory_where_the_hosts_own_puts_it(registry):
    native, swapped = chain(), chain()
    selection = add_effective(registry, "traj")
    pluck(native)
    pluck(swapped)
    with Session(swapped, [selection], registry):
        for _ in range(100):
            swapped.step()
    for _ in range(100):
        native.step()
    assert np.allclose(swapped.data.x, native.data.x, atol=1e-5)


def test_a_session_installs_and_removes_every_hook_it_made(registry):
    host = chain()
    before = {name: getattr(host.ns, name) for name in ("passive", "actuate", "assemble", "effective_solve", "forward")}
    selections = [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=1.0, c=0.2)),
                  Selection(cl.NAME, controllers.PD_ID, dict(kp=2.0, kd=0.1, target=0.1)),
                  Selection(ps.NAME, probes.PEAK_ID, {}), add_effective(registry, "scope")]
    with Session(host, selections, registry):
        assert all(getattr(host.ns, name) is not before[name] for name in before)
        pluck(host)
        host.step()
        assert np.isfinite(host.data.x).all()
    assert all(getattr(host.ns, name) is before[name] for name in before)


def test_a_law_on_another_model_is_left_alone(registry):
    host = chain()
    selection = Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=50.0, c=0.0, q_ref=1.0))
    with Session(host, [selection], registry):
        wrapped = host.drive(pf.NAME, owned=False)
    assert wrapped["effect"]["delta"].shape == host.data.x.shape


# ---- a law's damping integrates the way the chain's own damper does ------------------------------------

def test_a_law_that_reports_its_damping_integrates_like_the_chains_own_dashpot(registry):
    c, k = 0.8, 0.3
    own, native = chain(damping=0.0), chain(damping=c)
    for host in (own, native):
        pluck(host)
    with Session(native, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=0.0))], registry):
        for _ in range(40):
            native.step()
    with Session(own, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=k, c=c))], registry):
        for _ in range(40):
            own.step()
    assert np.allclose(own.data.x, native.data.x, rtol=1e-4, atol=1e-6), np.abs(own.data.x - native.data.x).max()


def test_applied_explicitly_the_same_law_is_a_different_trajectory(registry):
    c = 40.0                                                              # stiff enough that the implicit term matters
    implicit, explicit = chain(damping=0.0), chain(damping=0.0)
    for host in (implicit, explicit):
        pluck(host)
    with Session(implicit, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=0.0, c=c))], registry):
        for _ in range(20):
            implicit.step()
    with Session(explicit, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=0.0, c=c, implicit=False))], registry):
        for _ in range(20):
            explicit.step()
    assert np.abs(implicit.data.x - explicit.data.x).max() > 1e-5


def test_the_facts_say_which_array_type_a_host_hands_to_stages():
    """An implementation that needs one array type (Warp arrays, say) can refuse a host that hands it another, from facts alone."""
    from phost.testing.fake import FakeHost
    assert ChainHost(make_model(8, 2), worlds=1).facts().backend == "numpy"
    assert FakeHost().facts().backend == "numpy"


def test_the_jax_chain_says_its_arrays_are_jax():
    pytest.importorskip("jax")
    from phost.hosts.jaxchain import JaxChainHost
    assert JaxChainHost(make_model(8, 2), worlds=1).facts().backend == "jax"
