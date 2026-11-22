"""Stage ``effective.solve`` on the fake host: the contract is judged in float64 and against the host's own solve."""

import numpy as np
import pytest

from phost import Impl, ImplKind, Registry, Selection, Session, stages
from phost.errors import ApplicabilityError
from phost.stages import effective_solve as es
from phost.testing import runtimes
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost


def make(cls, name, **kw):
    return Impl(id=f"test/{name}", stage=es.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                factory=lambda host, params, cls=cls: cls(host, params), **kw)


@pytest.fixture
def registry():
    reg = Registry()
    es.register(reg)
    for name, cls in dict(reference=runtimes.EffectiveReferenceRuntime, wrong=runtimes.WrongEffectiveRuntime,
                          inputwriter=runtimes.EffectiveInputWriterRuntime).items():
        reg.add_impl(make(cls, name))
    return reg


def test_the_stage_ships_with_phost_and_roundtrips_through_json():
    shipped = Registry()
    stages.register_all(shipped)
    assert es.NAME in {c.name for c in shipped.contracts()}
    assert es.CONTRACT == type(es.CONTRACT).from_json(es.CONTRACT.to_json())
    assert es.CONTRACT.swap_class.value == "exact" and es.CONTRACT.entry == "solve"


def test_a_correct_solver_passes_both_gates(registry):
    verdict = check_impl(FakeHost(nv=12), Selection(es.NAME, "test/reference"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-5 and verdict.metrics["max_native_difference"] < 1e-5


def test_a_slightly_wrong_solution_fails_the_residual_and_the_host_comparison(registry):
    verdict = check_impl(FakeHost(), Selection(es.NAME, "test/wrong"), registry)
    assert not verdict.ok
    text = " ".join(verdict.failures)
    assert "max_relative_residual" in text and "max_native_difference" in text


def test_changing_an_input_is_caught(registry):
    verdict = check_impl(FakeHost(), Selection(es.NAME, "test/inputwriter"), registry)
    assert not verdict.ok and any("'rhs' was modified" in f for f in verdict.failures)


def test_the_host_counts_every_solve_and_the_session_restores_its_own(registry):
    host = FakeHost()
    original = host.ns.eff_solve
    with Session(host, [Selection(es.NAME, "test/reference")], registry) as session:
        assert host.ns.eff_solve is not original
        host.drive(es.NAME)
        assert session.metrics()[es.NAME]["calls"] >= 1
    assert host.ns.eff_solve is original


def test_another_model_is_solved_by_the_host_and_never_reaches_the_runtime(registry):
    host = FakeHost()
    with Session(host, [Selection(es.NAME, "test/reference")], registry) as session:
        run = host.drive(es.NAME, owned=False)
        assert session.metrics()[es.NAME]["calls"] == 0
    assert es.native_difference(run["effect"]["qacc"], run["effect"]["expected"]) < 1e-6


def test_a_host_that_reports_a_blocker_refuses_an_implementation_that_checks_it(registry):
    def applicable(facts):
        return facts.extra.get("effective_blocker") or None
    registry.add_impl(make(runtimes.EffectiveReferenceRuntime, "checked", applicable=applicable))
    host = FakeHost()
    with Session(host, [Selection(es.NAME, "test/checked")], registry):
        pass                                                           # the fake host reports no blocker
    blocked = FakeHost()
    blocked.facts = lambda: type(host.facts())(**{**vars(host.facts()), "extra": {"effective_blocker": "has a fluid term"}})
    with pytest.raises(ApplicabilityError, match="has a fluid term"):
        with Session(blocked, [Selection(es.NAME, "test/checked")], registry):
            pass


def test_the_reference_matches_a_direct_solve():
    rng = np.random.default_rng(3)
    n, pad = 6, 8
    h = np.zeros((2, pad, pad), np.float32)
    rhs = np.zeros((2, pad), np.float32)
    truth = []
    for w in range(2):
        a = rng.standard_normal((n, n))
        sym = a @ a.T + n * np.eye(n)
        h[w, :n, :n] = np.triu(sym)
        rhs[w, :n] = rng.standard_normal(n)
        truth.append(np.linalg.solve(sym.astype(np.float32).astype(np.float64), rhs[w, :n].astype(np.float64)))
    ref = es.reference(dict(h=h, rhs=rhs), n)["qacc"]
    assert np.allclose(ref, np.array(truth), rtol=1e-6)
    assert es.evaluate(dict(h=h, rhs=rhs), dict(qacc=ref), n)["max_relative_residual"] < 1e-9
