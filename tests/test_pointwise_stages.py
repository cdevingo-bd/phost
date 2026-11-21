"""The three pointwise stages (passive force law, controller, probe) on the fake host."""

import numpy as np
import pytest

from phost import ApplicabilityError, ContractError, Impl, ImplKind, Registry, Selection, Session, SwapClass
from phost.impls import controllers, forces, probes, register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import probe_state as ps
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost.testing.fake import BIAS_LENGTH, DAMPING, GAIN, FakeHost


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    return reg


@pytest.fixture
def host():
    return FakeHost(nv=9, world_count=3, nu=3)


def add(registry, stage, name, runtime_cls, reference, **kw):
    registry.add_impl(Impl(id=f"test/{name}", stage=stage.NAME, stage_version=1, version="1", title=name,
                           kind=ImplKind.BUILTIN, factory=lambda host, params: runtime_cls(),
                           reference=reference, **kw))
    return Selection(stage.NAME, f"test/{name}")


# ---- the catalog ------------------------------------------------------------------------------------

def test_catalog_swap_classes_and_references(registry):
    classes = {c.name: (c.swap_class, c.reference) for c in registry.contracts()}
    assert classes["newton.linear_solve"] == (SwapClass.EXACT, "newton.linear_solve.float64")
    assert classes[pf.NAME] == (SwapClass.MODEL_CHANGING, "impl")
    assert classes[cl.NAME] == (SwapClass.MODEL_CHANGING, "impl")
    assert classes[ps.NAME] == (SwapClass.OBSERVER, "impl")


def test_contracts_round_trip_through_json(registry):
    for contract in registry.contracts():
        assert type(contract).from_json(contract.to_json()) == contract


def test_an_impl_without_a_reference_is_refused_where_one_is_required(registry):
    with pytest.raises(ContractError, match="float64 reference"):
        registry.add_impl(Impl(id="test/none", stage=pf.NAME, stage_version=1, version="1", title="x",
                               kind=ImplKind.BUILTIN, factory=lambda host, params: object()))


# ---- the built-in implementations pass their own contracts ---------------------------------------------

@pytest.mark.parametrize("impl_id,params", [
    (forces.SPRING_DAMPER_ID, dict(k=3.0, c=0.4, q_ref=0.2)),
    (controllers.PD_ID, dict(kp=5.0, kd=0.3, target=0.5)),
    (probes.PEAK_ID, {}),
])
def test_builtin_implementations_pass(host, registry, impl_id, params):
    verdict = check_impl(host, Selection(registry.impl(impl_id).stage, impl_id, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-6


def test_passive_force_changes_the_hosts_result_by_exactly_the_law(host, registry):
    host.drive(pf.NAME)
    native = host.data.qfrc_passive.copy()
    q, v = host.data.qpos.copy(), host.data.qvel.copy()
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=2.0, c=0.0))], registry):
        host.drive(pf.NAME)
    scalar = host.model.scalar.astype(bool)
    expected = np.where(scalar, -2.0 * q[:, :host.model.nv], 0.0)
    assert np.allclose(host.data.qfrc_passive - native, expected, atol=1e-6)
    assert np.all((host.data.qfrc_passive - native)[:, ~scalar] == 0)
    assert np.allclose(native, -DAMPING * v)                       # native passive force untouched by the stage's absence


def test_controller_reaches_the_actuators(host, registry):
    with Session(host, [Selection(cl.NAME, controllers.PD_ID, dict(kp=4.0, kd=0.0, target=1.0))], registry):
        host.drive(cl.NAME)
    d = host.data
    assert np.allclose(d.ctrl, 4.0 * (1.0 - d.actuator_length), atol=1e-5)
    assert np.allclose(d.actuator_force, GAIN * d.ctrl + BIAS_LENGTH * d.actuator_length
                       + (-0.05) * d.actuator_velocity, atol=1e-4)


def test_probe_reports_without_changing_the_simulation(host, registry):
    host.drive(ps.NAME)
    baseline = (host.data.qacc.copy(), host.data.qfrc_passive.copy())
    with Session(host, [Selection(ps.NAME, probes.PEAK_ID)], registry) as session:
        host.drive(ps.NAME)
        metrics = session.metrics()[ps.NAME]
    assert np.array_equal(host.data.qacc, baseline[0]) and np.array_equal(host.data.qfrc_passive, baseline[1])
    assert metrics["traces"] == 1 and metrics["last_max_qvel"] > 0


def test_other_models_are_left_alone(host, registry):
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=9.0))], registry) as session:
        host.drive(pf.NAME, owned=False)
        assert session.metrics()[pf.NAME]["traces"] == 0


def test_applicability_is_checked(registry):
    sleepy = FakeHost()
    sleepy.facts = lambda: __import__("dataclasses").replace(FakeHost.facts(sleepy), sleep=True)
    with pytest.raises(ApplicabilityError, match="sleeping"):
        Session(sleepy, [Selection(pf.NAME, forces.SPRING_DAMPER_ID)], registry).__enter__()


def test_the_scope_restores_the_hosts_functions(host, registry):
    before = (host.ns.passive, host.ns.fwd_actuation, host.ns.forward)
    with Session(host, [Selection(pf.NAME, forces.SPRING_DAMPER_ID, dict(k=1.0)),
                        Selection(cl.NAME, controllers.PD_ID), Selection(ps.NAME, probes.PEAK_ID)], registry):
        assert host.ns.passive is not before[0] and host.ns.forward is not before[2]
    assert (host.ns.passive, host.ns.fwd_actuation, host.ns.forward) == before


# ---- faulty implementations are caught -------------------------------------------------------------------

def spring_reference(inputs, params):
    return dict(force=np.where(inputs["scalar"].astype(bool), -2.0 * inputs["q"], 0.0))


class WrongLaw:                         # computes something other than its own reference
    def apply(self, s): s.force[:] = np.where(s.scalar.astype(bool), -3.0 * s.q, 0.0)
    def metrics(self): return {}


class IgnoresMask:                      # right on scalar DOFs, but forgets to write 0 on the others
    def apply(self, s): s.force[:] = -2.0 * s.q - 0.5 * s.v
    def metrics(self): return {}


class WritesInput:
    def apply(self, s):
        s.q[:] += 1.0
        s.force[:] = np.where(s.scalar.astype(bool), -2.0 * (s.q - 1.0), 0.0)
    def metrics(self): return {}


class Random:                           # not repeatable
    def apply(self, s): s.force[:] = np.where(s.scalar.astype(bool), np.random.rand(*s.force.shape) * 1e-4 - 2.0 * s.q, 0.0)
    def metrics(self): return {}


class NaN:
    def apply(self, s): s.force[:] = np.nan
    def metrics(self): return {}


class Silent:                           # leaves force at zero while claiming the spring law
    def apply(self, s): pass
    def metrics(self): return {}


def spring_with_velocity_reference(inputs, params):
    return dict(force=np.where(inputs["scalar"].astype(bool), -2.0 * inputs["q"] - 0.5 * inputs["v"], 0.0))


@pytest.mark.parametrize("cls,needle", [
    (WrongLaw, "max_law_error"), (IgnoresMask, "max_law_error"), (WritesInput, "modified"),
    (Random, "differs between identical calls"), (NaN, "not finite"), (Silent, "max_law_error"),
])
def test_faulty_force_laws_are_caught(host, registry, cls, needle):
    reference = spring_with_velocity_reference if cls is IgnoresMask else spring_reference
    selection = add(registry, pf, cls.__name__.lower(), cls, reference)
    verdict = check_impl(host, selection, registry)
    assert not verdict.ok and any(needle in f for f in verdict.failures), verdict.failures


def test_a_law_that_matches_its_reference_but_not_the_hosts_effect_is_caught(host, registry):
    class Honest:
        def apply(self, s): s.force[:] = np.where(s.scalar.astype(bool), -2.0 * s.q, 0.0)
        def metrics(self): return {}

    selection = add(registry, pf, "honest", Honest, spring_reference)
    assert check_impl(host, selection, registry).ok
    # simulate a faulty adapter: the host's passive force does not change, whatever the law returned
    real_drive = host.drive

    def drive(stage, **kw):
        run = real_drive(stage, **kw)
        run["effect"] = dict(delta=np.zeros_like(run["effect"]["delta"]))
        return run
    host.drive = drive
    verdict = check_impl(host, selection, registry)
    assert not verdict.ok and any("max_effect_error" in f for f in verdict.failures)


def test_controller_that_disagrees_with_its_reference_is_caught(host, registry):
    class Off:
        def apply(self, s): s.ctrl[:] = 1.1 * (0.5 - s.length)
        def metrics(self): return {}

    selection = add(registry, cl, "off", Off, lambda inputs, params: dict(ctrl=0.5 - inputs["length"]))
    verdict = check_impl(host, selection, registry)
    assert not verdict.ok and any("max_law_error" in f for f in verdict.failures)


def test_probe_that_writes_a_state_array_is_caught(host, registry):
    class Meddler:
        width = 1
        def observe(self, s):
            s.qacc[:] *= 1.0001
            s.value[:] = s.qvel.sum(axis=1, keepdims=True)
        def metrics(self): return {}

    selection = add(registry, ps, "meddler", Meddler, lambda inputs, params: dict(value=inputs["qvel"].sum(axis=1, keepdims=True)))
    verdict = check_impl(host, selection, registry)
    assert not verdict.ok and any("modified" in f for f in verdict.failures)
