"""A velocity-dependent law that reports its damping integrates like the model's own damper (fake host, no GPU)."""

import dataclasses

import numpy as np
import pytest

from phost import GuardError, Impl, Registry, Selection, Session
from phost.impls import controllers, forces, register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost.testing.fake import TIMESTEP, FakeHost

C, K = 0.8, 0.3


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    return reg


def law(**extra):
    return [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=K, c=C, **extra))]


def scalar_velocity(host, **kwargs):
    return host.integrate(**kwargs)[:, host.model.scalar.astype(bool)]


def test_the_law_integrates_like_the_models_own_damper(registry):
    own = FakeHost(damping=0.0)
    native = FakeHost(damping=C)
    with Session(native, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=K))], registry):
        want = scalar_velocity(native, seed=3)                  # the model damps (c); the law only adds the spring
    with Session(own, law(), registry):
        got = scalar_velocity(own, seed=3)                      # the model does not damp; the law does
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), np.abs(got - want).max()


def one_step(host):
    """(v before, explicit acceleration, v after) of one forward and one Euler step on a fixed random state."""
    host._random_state(3)
    v0 = host.data.qvel.copy()
    host.ns.forward(host.model, host.data)
    qacc = host.data.qacc.copy()
    host.ns.euler(host.model, host.data)
    return v0, qacc, host.data.qvel.copy()


def test_explicit_and_implicit_application_differ_by_exactly_the_integrators_factor(registry):
    """With no model damping, explicit is v + h a. Implicit divides the step by (1 + h c) on the DOFs the law damps."""
    on = FakeHost().model.scalar.astype(bool)
    host = FakeHost(damping=0.0)
    with Session(host, law(implicit=False), registry):
        v0, qacc, explicit = one_step(host)
    assert np.allclose(explicit, v0 + TIMESTEP * qacc, rtol=1e-6, atol=1e-7)
    host = FakeHost(damping=0.0)
    with Session(host, law(), registry):
        v0, qacc, implicit = one_step(host)
    assert np.allclose(implicit[:, on], v0[:, on] + TIMESTEP * qacc[:, on] / (1.0 + TIMESTEP * C), rtol=1e-6, atol=1e-7)
    assert np.allclose(implicit[:, ~on], v0[:, ~on] + TIMESTEP * qacc[:, ~on], rtol=1e-6, atol=1e-7)   # not damped
    assert np.abs(implicit[:, on] - explicit[:, on]).max() > 1e-3                                      # and it matters


def test_the_euler_wrapper_exists_only_inside_the_scope_and_only_for_the_owned_model(registry):
    host = FakeHost(damping=0.0)
    original = host.ns.euler
    with Session(host, law(), registry):
        assert host.ns.euler is not original
        wrapped = host.integrate(seed=1, owned=False)             # a model the session does not own
    assert host.ns.euler is original
    plain = FakeHost(damping=0.0).integrate(seed=1, owned=False)
    assert np.array_equal(wrapped, plain)


def test_the_stage_alone_leaves_euler_alone_when_the_impl_does_not_ask(registry):
    host = FakeHost()
    original = host.ns.euler
    with Session(host, [Selection(cl.NAME, controllers.PD_ID)], registry):
        assert host.ns.euler is original


def test_an_optional_extension_the_host_lacks_is_an_advisory_not_a_refusal(registry):
    class Bare(FakeHost):
        def stages(self):
            stages = dict(super().stages())
            stages[pf.NAME] = dataclasses.replace(stages[pf.NAME], extensions={})
            return stages

    host = Bare(damping=0.0)
    with Session(host, law(), registry) as session:
        assert any("running without extension 'implicit_damping'" in note for note in session.advisories)
        got = scalar_velocity(host, seed=3)
    explicit_host = FakeHost(damping=0.0)
    with Session(explicit_host, law(implicit=False), registry):
        assert np.allclose(got, scalar_velocity(explicit_host, seed=3), rtol=1e-6)    # explicit, as asked of a bare host


def test_an_unreviewed_euler_degrades_with_the_guard_message_for_an_optional_use(registry):
    host = FakeHost(damping=0.0)
    host.stages()
    host.ns.euler = lambda m, d: None                              # not the function that was reviewed
    with Session(host, law(), registry) as session:
        assert any("running without extension 'implicit_damping'" in note and "unreviewed" in note
                   for note in session.advisories)
        assert host.ns.euler is not None


def test_an_unreviewed_euler_refuses_an_implementation_that_requires_the_extension(registry):
    needs = dataclasses.replace(registry.impl(forces.SPRING_DAMPER_ID), id="dof.passive_force/needs-implicit",
                                extensions_used=("implicit_damping",), extensions_optional=())
    registry.add_impl(needs)
    host = FakeHost(damping=0.0)
    host.stages()
    host.ns.euler = lambda m, d: None
    with pytest.raises(GuardError, match="unreviewed"):
        Session(host, [Selection(pf.NAME, needs.id, dict(c=C))], registry).__enter__()


def test_a_required_extension_the_host_does_not_offer_is_refused(registry):
    from phost import CapabilityError

    class Bare(FakeHost):
        def stages(self):
            stages = dict(super().stages())
            stages[pf.NAME] = dataclasses.replace(stages[pf.NAME], extensions={})
            return stages

    needs = dataclasses.replace(registry.impl(forces.SPRING_DAMPER_ID), id="dof.passive_force/needs-implicit",
                                extensions_used=("implicit_damping",), extensions_optional=())
    registry.add_impl(needs)
    with pytest.raises(CapabilityError, match="implicit_damping"):
        Session(Bare(), [Selection(pf.NAME, needs.id)], registry).__enter__()


# -- conformance --------------------------------------------------------------------------------------

@pytest.mark.parametrize("params", [dict(k=K, c=C), dict(k=K, c=C, implicit=False), dict(k=K, c=0.0)])
def test_the_builtin_passes_the_contract_with_and_without_reported_damping(registry, params):
    verdict = check_impl(FakeHost(), Selection(pf.NAME, forces.SPRING_DAMPER_ID, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_damping_error"] < 1e-5


def test_the_conformance_check_catches_damping_that_is_not_the_laws_derivative(registry):
    class Doubled(forces._Numpy):
        def apply(self, system):
            super().apply(system)
            system.damping *= 2.0

    wrong = dataclasses.replace(registry.impl(forces.SPRING_DAMPER_ID), id="dof.passive_force/doubled",
                                factory=lambda host, params: Doubled(params))
    registry.add_impl(wrong)
    verdict = check_impl(FakeHost(), Selection(pf.NAME, wrong.id, dict(k=K, c=C)), registry)
    assert not verdict.ok
    assert verdict.metrics["max_damping_error"] > 0.5
    assert any("max_damping_error" in failure for failure in verdict.failures)


def test_damping_written_on_a_DOF_that_is_not_scalar_is_caught(registry):
    class Everywhere(forces._Numpy):
        def apply(self, system):
            super().apply(system)
            system.damping[:] = self.p["c"]

    wrong = dataclasses.replace(registry.impl(forces.SPRING_DAMPER_ID), id="dof.passive_force/everywhere",
                                factory=lambda host, params: Everywhere(params))
    registry.add_impl(wrong)
    verdict = check_impl(FakeHost(), Selection(pf.NAME, wrong.id, dict(k=K, c=C)), registry)
    assert not verdict.ok and verdict.metrics["max_damping_error"] > 0.5


# -- the advisories used to raise NameError on an implicit integrator ---------------------------------

@pytest.mark.parametrize("integrator", ["implicit", "implicitfast"])
@pytest.mark.parametrize("module", [forces, controllers])
def test_advisory_text_for_implicit_integrators(module, integrator):
    from phost import ModelFacts
    note = module._advisory(ModelFacts(nv=3, world_count=1, integrator=integrator))
    assert "velocity Jacobian" in note


def test_a_session_on_an_implicit_integrator_model_records_the_advisory(registry):
    host = FakeHost()
    host.facts = lambda: dataclasses.replace(FakeHost.facts(host), integrator="implicitfast")
    with Session(host, law(), registry) as session:
        assert any("velocity Jacobian" in note for note in session.advisories)
