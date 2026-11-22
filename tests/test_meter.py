"""The host counts stage executions; implementations do not count themselves."""

import json

import numpy as np
import pytest

from phost import CALLS, PythonMeter, Registry, Selection, Session, StageContract
from phost.impls import forces, register_builtin
from phost.meter import Metered
from phost.stages import control_law, dof_passive_force, newton_linear, probe_state, register_all
from phost.testing.fake import FakeHost
from phost.testing.runtimes import ReferenceRuntime


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    return reg


def test_every_shipped_contract_names_its_entry_and_reports_calls():
    for module, entry in ((newton_linear, "solve"), (dof_passive_force, "apply"), (control_law, "apply"),
                          (probe_state, "observe")):
        assert module.CONTRACT.entry == entry
        assert CALLS in module.CONTRACT.metrics
        assert "traces" not in {m.name for m in module.CONTRACT.metrics}


def test_the_entry_survives_serialization():
    contract = dof_passive_force.CONTRACT
    assert StageContract.from_json(contract.to_json()).entry == "apply"
    assert json.loads(contract.to_json())["entry"] == "apply"


def test_python_meter_counts_and_resets():
    meter = PythonMeter()
    for _ in range(3):
        meter.tick()
    assert meter.count() == 3
    meter.reset()
    assert meter.count() == 0


def test_metered_ticks_the_entry_method_only():
    class Runtime:
        def __init__(self):
            self.seen = []

        def apply(self, system):
            self.seen.append(system)
            return "done"

        def metrics(self):
            return {"x": 1}

    runtime, meter = Runtime(), PythonMeter()
    proxy = Metered(runtime, "apply", meter)
    assert proxy.apply("s") == "done" and runtime.seen == ["s"]
    assert proxy.metrics() == {"x": 1}
    assert meter.count() == 1                                    # metrics() is not an execution


def test_metered_still_calls_a_wrapper_installed_on_the_runtime_later():
    class Runtime:
        def apply(self, system):
            return 1

    runtime, meter = Runtime(), PythonMeter()
    proxy = Metered(runtime, "apply", meter)
    original = runtime.apply
    runtime.apply = lambda system: original(system) + 10         # what the conformance recorder does
    assert proxy.apply(None) == 11 and meter.count() == 1


def test_session_reports_host_counted_calls_and_a_runtime_cannot_override_them(registry):
    host = FakeHost()
    registry.add_impl(__import__("phost").Impl(
        id="newton.linear_solve/claims", stage=newton_linear.NAME, stage_version=1, version="1", title="Lies",
        kind=__import__("phost").ImplKind.KERNEL,
        factory=lambda h, p: type("R", (ReferenceRuntime,), {"metrics": lambda self: {"calls": 999, "status": 0}})(h, p)))
    with Session(host, [Selection(newton_linear.NAME, "newton.linear_solve/claims")], registry) as session:
        host.drive(newton_linear.NAME)
        host.drive(newton_linear.NAME)
        assert session.metrics()[newton_linear.NAME]["calls"] == 2


def test_calls_count_executions_on_the_owned_model_only(registry):
    host = FakeHost()
    with Session(host, [Selection(dof_passive_force.NAME, forces.SPRING_DAMPER_ID, dict(k=1.0))], registry) as session:
        host.drive(dof_passive_force.NAME)
        host.drive(dof_passive_force.NAME, owned=False)
        host.drive(dof_passive_force.NAME)
        assert session.metrics()[dof_passive_force.NAME]["calls"] == 2
        session.reset_metrics()
        assert session.metrics()[dof_passive_force.NAME]["calls"] == 0


def test_a_native_selection_installs_nothing_and_counts_nothing(registry):
    host = FakeHost()
    with Session(host, [Selection(dof_passive_force.NAME, "dof.passive_force/native")], registry) as session:
        host.drive(dof_passive_force.NAME)
        assert session.metrics() == {}
