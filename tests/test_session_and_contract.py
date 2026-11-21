import numpy as np
import pytest

from phost import (ApplicabilityError, CapabilityError, ContractError, GuardError, Impl, ImplKind, ParamSpec, Selection,
                   Session, source_hash)
from phost.stages import newton_linear as nl
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost
from phost.testing.runtimes import ReferenceRuntime


def test_pristine_host_passes_its_own_guards(host):
    binding = host.stage(nl.NAME)
    for guard in binding.point.guards.values():
        assert guard.check() == "fake build 1"


def test_reference_runtime_passes_the_contract(host, registry):
    verdict = check_impl(host, Selection(nl.NAME, "test/reference"), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-6


@pytest.mark.parametrize("name,needle", [
    ("wrong", "max_relative_residual"),
    ("lowerreader", "max_relative_residual"),
    ("inputwriter", "modified"),
    ("inactivewriter", "inactive"),
])
def test_faulty_runtimes_are_caught(host, registry, name, needle):
    verdict = check_impl(host, Selection(nl.NAME, f"test/{name}"), registry)
    assert not verdict.ok
    assert any(needle in f for f in verdict.failures), verdict.failures


def test_fallback_to_the_host_solver_is_correct(host, registry):
    assert check_impl(host, Selection(nl.NAME, "test/fallback"), registry).ok


def test_native_selection_installs_nothing(host, registry):
    original = host.ns.solve_hook
    with Session(host, [Selection(nl.NAME, "native/native".replace("native/native", f"{nl.NAME}/native"))], registry):
        assert host.ns.solve_hook is original


def test_session_restores_the_host_after_exit_and_after_error(host, registry):
    original = host.ns.solve_hook
    with Session(host, [Selection(nl.NAME, "test/reference")], registry) as session:
        assert host.ns.solve_hook is not original
        host.drive(nl.NAME)
        assert session.metrics()[nl.NAME]["calls"] == 1
    assert host.ns.solve_hook is original
    with pytest.raises(ValueError):
        with Session(host, [Selection(nl.NAME, "test/reference")], registry):
            raise ValueError("x")
    assert host.ns.solve_hook is original


def test_other_models_still_reach_the_host_solver(host, registry):
    calls = []
    real = ReferenceRuntime.solve
    ReferenceRuntime.solve = lambda self, s: (calls.append(1), real(self, s))[1]
    try:
        with Session(host, [Selection(nl.NAME, "test/reference")], registry):
            run = host.drive(nl.NAME, owned=False)
        assert not calls
        assert np.all(np.isfinite(run["outputs"]["search"][run["active"]]))
    finally:
        ReferenceRuntime.solve = real


def test_unreviewed_host_build_is_refused(host, registry):
    host.ns.solve_hook = lambda *a, **k: None            # a host build nobody reviewed
    with pytest.raises(GuardError, match="unreviewed"):
        with Session(host, [Selection(nl.NAME, "test/reference")], registry):
            pass


def test_extension_requirements(host, registry):
    reg = registry
    reg.add_impl(Impl(id="test/hint", stage=nl.NAME, stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
                      extensions_used=("structure_hint",), factory=lambda h, p: ReferenceRuntime(h, p)))
    with Session(host, [Selection(nl.NAME, "test/hint")], reg):
        pass
    host.ns.assemble = lambda *a: None                   # the assembly guard no longer matches
    with pytest.raises(GuardError, match="assembly"):
        with Session(host, [Selection(nl.NAME, "test/hint")], reg):
            pass
    with Session(host, [Selection(nl.NAME, "test/reference")], reg):    # does not rely on it: still fine
        pass


def test_unknown_extension_and_missing_capability(host, registry):
    registry.add_impl(Impl(id="test/bad", stage=nl.NAME, stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
                           extensions_used=("telepathy",), factory=lambda h, p: ReferenceRuntime(h, p)))
    with pytest.raises(CapabilityError, match="telepathy"):
        with Session(host, [Selection(nl.NAME, "test/bad")], registry):
            pass
    from phost import Capability
    host.capabilities = Capability.NATIVE
    with pytest.raises(CapabilityError, match="cannot patch"):
        with Session(host, [Selection(nl.NAME, "test/reference")], registry):
            pass


def test_applicability_and_params(host, registry):
    registry.add_impl(Impl(
        id="test/big", stage=nl.NAME, stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
        params=(ParamSpec("tile", "int", 32, choices=(16, 32)), ParamSpec("scale", "float", 1.0, min=0.0)),
        applicable=lambda f: None if f.nv >= 100 else f"nv={f.nv} is below 100",
        factory=lambda h, p: ReferenceRuntime(h, p)))
    with pytest.raises(ApplicabilityError, match="nv=12"):
        with Session(host, [Selection(nl.NAME, "test/big")], registry):
            pass
    big = FakeHost(nv=120, nv_pad=128, coupled=6)
    with Session(big, [Selection(nl.NAME, "test/big", {"tile": 16})], registry):
        pass
    for bad in ({"tile": 8}, {"scale": -1.0}, {"tile": "x"}, {"nope": 1}):
        with pytest.raises(ContractError):
            with Session(big, [Selection(nl.NAME, "test/big", bad)], registry):
                pass


def test_session_input_validation(host, registry):
    with pytest.raises(ContractError, match="twice"):
        Session(host, [Selection(nl.NAME, "test/reference"), Selection(nl.NAME, "test/wrong")], registry)
    with pytest.raises(ContractError, match="no implementation"):
        with Session(host, [Selection(nl.NAME, "test/missing")], registry):
            pass
    other = Impl(id="x/other", stage="other.stage", stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
                 factory=lambda h, p: ReferenceRuntime(h, p))
    registry.add_contract(__import__("phost").StageContract("other.stage", 1, "", __import__("phost").SwapClass.EXACT, (), ()))
    registry.add_impl(other)
    with pytest.raises(ContractError, match="not 'newton.linear_solve'"):
        with Session(host, [Selection(nl.NAME, "x/other")], registry):
            pass


def test_registry_describes_itself_as_json(registry):
    import json
    text = json.dumps(registry.describe())
    again = json.loads(text)
    assert any(c["name"] == nl.NAME for c in again["contracts"])
    assert {i["id"] for i in again["impls"]} >= {"test/reference", f"{nl.NAME}/native"}


def test_registry_rejects_conflicts(registry):
    from phost import StageContract, SwapClass
    with pytest.raises(ContractError, match="different content"):
        registry.add_contract(StageContract(nl.NAME, 1, "changed", SwapClass.EXACT, (), ()))
    with pytest.raises(ContractError, match="already registered"):
        registry.add_impl(registry.impl("test/reference"))
    with pytest.raises(ContractError, match="unknown contract"):
        registry.add_impl(Impl(id="y", stage="nope", stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
                               factory=lambda h, p: None))
    with pytest.raises(ContractError, match="redefines"):
        from phost import MetricSpec
        registry.add_impl(Impl(id="z", stage=nl.NAME, stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN,
                               metrics=(MetricSpec("status", "x", "none", "gauge"),), factory=lambda h, p: None))


def test_impl_kind_rules():
    with pytest.raises(ContractError, match="no factory"):
        Impl(id="a", stage="s", stage_version=1, version="1", title="t", kind=ImplKind.NATIVE, factory=lambda h, p: None)
    with pytest.raises(ContractError, match="needs a factory"):
        Impl(id="a", stage="s", stage_version=1, version="1", title="t", kind=ImplKind.BUILTIN)


def test_reference_agrees_with_evaluate(host):
    run = host.drive(nl.NAME)
    ref = nl.reference(run["inputs"], run["nv"])
    got = run["outputs"]["search"]
    assert np.allclose(got[run["active"]], ref["search"][run["active"]], rtol=1e-4, atol=1e-5)
