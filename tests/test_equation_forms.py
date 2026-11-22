"""Equation forms (``phost.equation``) and implementation-level approximate classes.

A stage declares how a law for it can be typed; the declaration is checked against the contract when the stage is defined.
An implementation of an exact stage may declare itself approximate and is then held to the gates it claims.
"""

import dataclasses

import numpy as np
import pytest

from phost import (Channels, ContractError, Impl, ImplKind, Iteration, Judge, ParamSpec, Pointwise, Registry, Selection, Slope,
                   StageSpec, SwapClass, Tolerance, stages)
from phost.equation import Name, validate
from phost.stages import body_wrench as bw
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl
from phost.stages import probe_state as ps
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost


@pytest.fixture
def shipped():
    reg = Registry()
    stages.register_all(reg)
    return reg


def test_every_stage_that_takes_a_typed_law_declares_its_form(shipped):
    assert set(shipped.equation_stages()) == {pf.NAME, cl.NAME, bw.NAME, ps.NAME, es.NAME, nl.NAME}
    assert [shipped.equation(n).kind for n in (pf.NAME, ps.NAME, es.NAME)] == ["pointwise", "channels", "iteration"]


def test_the_forms_are_in_the_registry_description_as_data(shipped):
    import json
    again = json.loads(json.dumps(shipped.describe()))
    forms = {c["name"]: c["equation"] for c in again["contracts"]}
    assert forms[bw.NAME]["outputs"][3] == dict(name="tx", port="wrench", column=3, doc="torque about the centre of mass, world x")
    assert forms[pf.NAME]["slope"] == dict(of="v", port="damping", sign=-1.0, param="implicit") and forms[pf.NAME]["mask"] == "scalar"
    assert forms[nl.NAME]["derived"] == [["search_dot", "square_norm"], ["newton_decrement", "rhs_dot"]]
    assert forms[nl.NAME]["active"] == "active" and forms[nl.NAME]["sign"] == -1.0


def test_a_stage_without_a_form_takes_no_typed_law(shipped):
    reg = Registry()
    nl.register(reg)
    with pytest.raises(ContractError, match="different equation form"):
        reg.add_stage(StageSpec(nl.CONTRACT, nl.SPEC.judge))                   # the same stage, registered without its form
    bare = Registry()
    bare.add_stage(StageSpec(es.CONTRACT, es.SPEC.judge))
    assert bare.equation(es.NAME) is None and bare.equation_stages() == []


def test_a_stage_registered_twice_with_different_forms_is_refused(shipped):
    other = dataclasses.replace(pf.SPEC, equation=dataclasses.replace(pf.EQUATION, mask=None))
    with pytest.raises(ContractError, match="different equation form"):
        shipped.add_stage(other)


# ---- the form is checked against the contract when the stage is defined -----------------------------------------

def spec_with(contract, form):
    return StageSpec(contract, Judge(evaluate=lambda *a: {}), equation=form)


@pytest.mark.parametrize("form,message", [
    (Pointwise("nv", (Name("q", "nope"),), (Name("force", "force"),)), "is not a port"),
    (Pointwise("nv", (Name("q", "force"),), (Name("force", "force"),)), "must be an input port"),
    (Pointwise("nv", (Name("q", "q"),), (Name("force", "v"),)), "must be an output port"),
    (Pointwise("nope", (Name("q", "q"),), (Name("force", "force"),)), "not a dimension"),
    (Pointwise("nv", (Name("q", "q", 2),), (Name("force", "force"),)), "does not fit port"),
    (Pointwise("nv", (Name("q", "q"), Name("q", "v")), (Name("force", "force"),)), "repeats a name"),
    (Pointwise("nv", (Name("_q", "q"),), (Name("force", "force"),)), "cannot be a name"),
    (Pointwise("nv", (Name("q", "q"),), ()), "needs an output"),
    (Pointwise("nv", (Name("q", "q"),), (Name("force", "force"),), mask="time"), "must be an int32 port"),
    (Pointwise("nv", (Name("q", "q"),), (Name("force", "force"),), slope=Slope("v", "damping")), "not an input name"),
    (Pointwise("nv", (Name("v", "v"),), (Name("force", "force"),), slope=Slope("v", "q")), "must be an output port"),
])
def test_a_pointwise_form_that_does_not_fit_the_contract_is_refused(form, message):
    with pytest.raises(ContractError, match=message):
        spec_with(pf.CONTRACT, form)


def test_a_vector_port_needs_a_column_in_range():
    ok = Pointwise("body", (Name("vx", "cvel", 5),), (Name("fx", "wrench", 0),))
    spec_with(bw.CONTRACT, ok)
    for column in (6, -1, None):
        with pytest.raises(ContractError, match="does not fit port"):
            spec_with(bw.CONTRACT, Pointwise("body", (Name("vx", "cvel", column),), (Name("fx", "wrench", 0),)))


@pytest.mark.parametrize("form,message", [
    (Channels((Name("qpos", "nope"),), "value"), "is not a port"),
    (Channels((Name("qpos", "qpos"),), "qpos"), "must be an output port"),
    (Channels((Name("qpos", "qpos", 1),), "value"), "whole ports"),
    (Channels((Name("qpos", "qpos"),), "time"), "output port"),
])
def test_a_channels_form_is_checked(form, message):
    with pytest.raises(ContractError, match=message):
        spec_with(ps.CONTRACT, form)


@pytest.mark.parametrize("form,message", [
    (Iteration("rhs", "rhs", "qacc", "nv"), "needs matrix"),
    (Iteration("h", "rhs", "qacc", "nope"), "not a dimension"),
    (Iteration("h", "rhs", "qacc", "nv", derived=(("qacc", "bogus"),)), "unknown derived quantity"),
    (Iteration("h", "rhs", "qacc", "nv", derived=(("qacc", "square_norm"),)), "must be shaped"),
    (Iteration("h", "rhs", "qacc", "nv", active="h"), "must be an|shaped"),
])
def test_an_iteration_form_is_checked(form, message):
    with pytest.raises(ContractError, match=message):
        spec_with(es.CONTRACT, form)


def test_validate_is_the_function_the_stage_spec_uses():
    validate(pf.EQUATION, pf.CONTRACT)
    with pytest.raises(ContractError):
        validate(pf.EQUATION, cl.CONTRACT)


# ---- an implementation may be less exact than its stage --------------------------------------------------------

class Jacobi:
    """An approximate solve of M_hat x = rhs: a few damped Jacobi passes. The last pass count is a parameter."""

    def __init__(self, host, params):
        self.passes = params["passes"]

    def solve(self, system):
        h, rhs = np.asarray(system.h, np.float64), np.asarray(system.rhs, np.float64)
        n = system.qacc.shape[1]
        for w in range(h.shape[0]):
            upper = np.triu(h[w, :n, :n])
            a = upper + upper.T - np.diag(np.diag(upper))
            d, x = np.diag(a), np.zeros(n)
            for _ in range(self.passes):
                x = x + 0.5 * (rhs[w, :n] - a @ x) / d
            system.qacc[w] = x

    def metrics(self):
        return {}


def approximate(registry, passes_default=4, tolerance=0.5, stage=es):
    def gates(params):
        return (Tolerance("max_relative_residual", tolerance, "claimed"),
                Tolerance("max_native_difference", 1.0, "near the host's"))

    impl = Impl(id="test/jacobi", stage=stage.NAME, stage_version=1, version="1", title="jacobi", kind=ImplKind.BUILTIN,
                params=(ParamSpec("passes", "int", passes_default),), factory=lambda host, params: Jacobi(host, params),
                swap_class=SwapClass.APPROXIMATE, tolerances=gates)
    registry.add_impl(impl)
    return impl


def test_an_approximate_solver_is_held_to_the_accuracy_it_claims_and_not_to_the_exact_gate(shipped):
    approximate(shipped, tolerance=0.9)
    verdict = check_impl(FakeHost(nv=12), Selection(es.NAME, "test/jacobi"), shipped)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] > es.CONTRACT.tolerances[1].max      # it would fail the exact gate


def test_an_approximate_solver_that_misses_its_own_claim_fails(shipped):
    approximate(shipped, tolerance=1e-6)
    verdict = check_impl(FakeHost(nv=12), Selection(es.NAME, "test/jacobi"), shipped)
    assert not verdict.ok and any("max_relative_residual" in f and "exceeds 1e-06" in f for f in verdict.failures)


def test_the_stages_gates_stay_for_metrics_the_implementation_does_not_replace(shipped):
    impl = approximate(shipped)
    gates = {t.metric: t.max for t in impl.gates(es.CONTRACT.tolerances, {})}
    assert gates == {"max_relative_residual": 0.5, "max_native_difference": 1.0}
    only_residual = dataclasses.replace(impl, tolerances=lambda p: (Tolerance("max_relative_residual", 0.1),))
    assert {t.metric: t.max for t in only_residual.gates(es.CONTRACT.tolerances, {})} == \
        {"max_native_difference": 1e-3, "max_relative_residual": 0.1}


def test_an_implementation_that_declares_nothing_is_gated_by_its_stage(shipped):
    plain = Impl(id="test/plain", stage=es.NAME, stage_version=1, version="1", title="p", kind=ImplKind.BUILTIN,
                 factory=lambda host, params: Jacobi(host, dict(passes=2)))
    assert plain.gates(es.CONTRACT.tolerances, {}) == es.CONTRACT.tolerances


def test_only_an_approximate_implementation_may_have_gates_of_its_own():
    with pytest.raises(ContractError, match="only an approximate"):
        Impl(id="x", stage=es.NAME, stage_version=1, version="1", title="x", kind=ImplKind.BUILTIN,
             factory=lambda h, p: None, tolerances=lambda p: ())
    with pytest.raises(ContractError, match="only declare itself approximate"):
        Impl(id="x", stage=es.NAME, stage_version=1, version="1", title="x", kind=ImplKind.BUILTIN,
             factory=lambda h, p: None, swap_class=SwapClass.EXACT)


def test_a_stage_that_changes_the_physics_or_only_observes_cannot_be_relaxed(shipped):
    for stage in (pf, ps):
        impl = Impl(id=f"x/{stage.NAME}", stage=stage.NAME, stage_version=1, version="1", title="x", kind=ImplKind.BUILTIN,
                    factory=lambda h, p: None, reference=lambda i, p: {}, swap_class=SwapClass.APPROXIMATE)
        with pytest.raises(ContractError, match="cannot relax"):
            shipped.add_impl(impl)


def test_the_description_says_which_class_an_implementation_declares(shipped):
    approximate(shipped)
    classes = {i["id"]: i["swap_class"] for i in shipped.describe()["impls"]}
    assert classes["test/jacobi"] == "approximate" and classes[f"{es.NAME}/native"] is None


# ---- the program error: the generated code against what was typed -------------------------------------------------

def test_an_implementation_with_a_reference_gets_a_program_error_even_where_the_stage_does_not_ask_for_one(shipped):
    def reference(inputs, params, dims):
        n = dims["nv"]
        out = np.zeros((inputs["h"].shape[0], n))
        for w in range(out.shape[0]):
            upper = np.triu(inputs["h"][w, :n, :n])
            out[w] = np.linalg.solve(upper + upper.T - np.diag(np.diag(upper)), inputs["rhs"][w, :n])
        return dict(qacc=out)

    impl = Impl(id="test/ref", stage=es.NAME, stage_version=1, version="1", title="r", kind=ImplKind.BUILTIN,
                factory=lambda host, params: Jacobi(host, dict(passes=60)), reference=reference)
    shipped.add_impl(impl)
    verdict = check_impl(FakeHost(nv=12), Selection(es.NAME, "test/ref"), shipped)
    assert "max_program_error" in verdict.metrics and verdict.metrics["max_program_error"] < 1e-2
    bad = dataclasses.replace(impl, id="test/ref2", reference=lambda inputs, params, dims: dict(qacc=np.zeros((inputs["h"].shape[0], dims["nv"]))))
    shipped.add_impl(bad)
    far = check_impl(FakeHost(nv=12), Selection(es.NAME, "test/ref2"), shipped)
    assert far.metrics["max_program_error"] > 0.5


def test_a_reference_without_a_dims_parameter_is_called_as_before():
    impl = Impl(id="x", stage=pf.NAME, stage_version=1, version="1", title="x", kind=ImplKind.BUILTIN,
                factory=lambda h, p: None, reference=lambda inputs, params: dict(seen=sorted(inputs)))
    assert impl.run_reference(dict(a=1), {}, dims=dict(nv=3)) == dict(seen=["a"])
    with_dims = dataclasses.replace(impl, reference=lambda inputs, params, dims=None: dict(dims=dims))
    assert with_dims.run_reference({}, {}, dims=dict(nv=3)) == dict(dims=dict(nv=3))
    with pytest.raises(ContractError, match="no float64 reference"):
        dataclasses.replace(impl, reference=None).run_reference({}, {})
