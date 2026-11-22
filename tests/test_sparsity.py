"""``Sparsity``: the structure of a system matrix as a host-provided fact, and the ``sparsity`` extension that hands it over."""

import numpy as np
import pytest

from phost import Impl, ImplKind, Registry, Selection, Session
from phost.errors import CapabilityError
from phost.sparsity import Sparsity
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost


# ---- the type ----

def test_a_pattern_is_symmetric_with_its_diagonal_and_sorted_rows():
    s = Sparsity.from_pairs(4, [(0, 2), (3, 1)])
    assert [list(s.row(i)) for i in range(4)] == [[0, 2], [1, 3], [0, 2], [1, 3]]
    assert s.nnz == 8 and s.degrees() == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="outside"):
        Sparsity.from_pairs(2, [(0, 5)])


def test_from_dense_reads_the_nonzeros_and_banded_and_dense_are_special_cases():
    a = np.diag([1.0, 2.0, 3.0, 4.0])
    a[0, 3] = 0.5
    s = Sparsity.from_dense(a)
    assert list(s.row(0)) == [0, 3] and list(s.row(3)) == [0, 3] and s.coupled_rows() == [1, 0, 0, 1]
    assert Sparsity.banded(6, 1).bandwidth() == 1 and Sparsity.dense(5).nnz == 25
    assert Sparsity.from_dense(np.ones((3, 3))) == Sparsity.dense(3)


def test_reverse_cuthill_mckee_narrows_a_scrambled_band():
    n, rng = 40, np.random.default_rng(3)
    scramble = rng.permutation(n)
    chain = [(int(scramble[i]), int(scramble[i + 1])) for i in range(n - 1)]      # a path, with its rows in random order
    s = Sparsity.from_pairs(n, chain)
    assert s.bandwidth() > 10
    order = s.rcm_order()
    assert sorted(order) == list(range(n)) and s.bandwidth(order) == 1
    with pytest.raises(ValueError, match="permutation"):
        s.bandwidth([0, 0, 1])


def test_describe_counts_the_work_a_structure_exploiting_factorization_could_save():
    n, c = 100, 10
    pairs = [(i, j) for i in range(c) for j in range(i)]                           # a dense block of 10 coupled rows
    d = Sparsity.from_pairs(n, pairs).describe()
    assert d["n"] == n and d["coupled"] == c and d["coupled_flops"] == c ** 3 / 3 and d["dense_flops"] == n ** 3 / 3
    assert d["coupled_flops"] < d["dense_flops"] / 500
    assert d["bandwidth_rcm"] <= d["bandwidth"]


def test_it_answers_the_question_a_newton_structure_hint_answers():
    """The coupled-row flags a Newton hint sets are a function of the pattern: one fact serves both stages."""
    host = FakeHost(nv=12, coupled=4)
    ctx = host.make_ctx()
    n = host.model.nv
    sym = nl._symmetric(ctx.h[0], n)
    flags = np.zeros((host.data.nworld, n), np.int32)
    seen = {}

    class Hinted:
        def __init__(self, host, params): pass
        def solve(self, system):
            system.structure_hint(flags, np.ones(host.data.nworld, np.int32))
            seen["pattern"] = Sparsity.from_dense(nl._symmetric(system.h[0], n))
            system.search[:] = 0.0
        def metrics(self): return {}

    registry = Registry()
    nl.register(registry)
    registry.add_impl(Impl(id="test/hinted", stage=nl.NAME, stage_version=1, version="1", title="hinted", kind=ImplKind.BUILTIN,
                           extensions_used=("structure_hint",), factory=lambda host, params: Hinted(host, params)))
    with Session(host, [Selection(nl.NAME, "test/hinted")], registry):
        host.ns.solve_hook(host.model, host.data, ctx, False, False)
    assert list(flags[0]) == seen["pattern"].coupled_rows() == Sparsity.from_dense(sym).coupled_rows()
    assert sum(flags[0]) == host.coupled


# ---- the extension ----

class BandedSolver:
    """Reorders by reverse Cuthill-McKee from the host's pattern, then solves. It never scans h for structure."""

    def __init__(self, host, params):
        self.seen = []

    def solve(self, system):
        self.seen.append(system.sparsity)
        n = system.qacc.shape[1]
        order = np.array(system.sparsity.rcm_order())
        for w in range(system.qacc.shape[0]):
            h = es._symmetric(system.h[w], n)[np.ix_(order, order)]
            x = np.linalg.solve(h, np.asarray(system.rhs[w, :n], np.float64)[order])
            out = np.empty(n)
            out[order] = x
            system.qacc[w] = out.astype(np.float32)

    def metrics(self):
        return {}


class Plain(BandedSolver):
    def solve(self, system):
        self.seen.append(system.sparsity)
        for w in range(system.qacc.shape[0]):
            n = system.qacc.shape[1]
            system.qacc[w] = np.linalg.solve(es._symmetric(system.h[w], n), np.asarray(system.rhs[w, :n], np.float64)).astype(np.float32)


@pytest.fixture
def registry():
    reg = Registry()
    es.register(reg)
    reg.add_impl(Impl(id="test/banded", stage=es.NAME, stage_version=1, version="1", title="banded", kind=ImplKind.BUILTIN,
                      extensions_used=("sparsity",), factory=lambda host, params: BandedSolver(host, params)))
    reg.add_impl(Impl(id="test/plain", stage=es.NAME, stage_version=1, version="1", title="plain", kind=ImplKind.BUILTIN,
                      factory=lambda host, params: Plain(host, params)))
    return reg


def test_the_stage_declares_the_extension_and_the_fake_host_offers_it():
    assert "sparsity" in {e.name for e in es.CONTRACT.extensions}
    host = FakeHost(nv=12)
    assert "sparsity" in host.stage(es.NAME).extensions and host.stage(es.NAME).sparsity is not None


def test_a_tool_can_read_the_pattern_without_installing_anything():
    dense, banded = FakeHost(nv=12), FakeHost(nv=12, eff_bandwidth=2)
    assert dense.stage(es.NAME).sparsity(dense).describe()["nnz"] == 144
    s = banded.stage(es.NAME).sparsity(banded)
    assert s.bandwidth() == 2 and s.describe()["bandwidth_rcm"] <= 2


def test_an_implementation_that_asks_for_the_extension_gets_the_hosts_pattern_and_passes_the_check(registry):
    host = FakeHost(nv=24, nv_pad=28, eff_bandwidth=2)
    selection = Selection(es.NAME, "test/banded")
    verdict = check_impl(host, selection, registry)
    assert verdict.ok, verdict.failures
    with Session(host, [selection], registry) as session:
        host.drive(es.NAME, seed=2)
    assert session.metrics()[es.NAME]["calls"] >= 1


def test_the_runtime_sees_the_pattern_only_when_it_declared_the_extension(registry):
    host = FakeHost(nv=24, nv_pad=28, eff_bandwidth=2)
    seen = {}
    for impl_id in ("test/banded", "test/plain"):
        impl = registry.impl(impl_id)
        made = []
        registry._impls[impl_id] = Impl(**{**impl.__dict__, "factory": lambda h, p, f=impl.factory, m=made: m.append(f(h, p)) or m[-1]})
        with Session(host, [Selection(es.NAME, impl_id)], registry):
            host.drive(es.NAME, seed=0)
        seen[impl_id] = made[0].seen
    assert all(isinstance(s, Sparsity) and s.bandwidth() == 2 for s in seen["test/banded"]) and seen["test/banded"]
    assert seen["test/plain"] and all(s is None for s in seen["test/plain"])


def test_a_host_that_does_not_offer_the_extension_refuses_an_implementation_that_needs_it(registry):
    host = FakeHost(nv=12)
    binding = host.stage(es.NAME)
    host._stages[es.NAME] = type(binding)(**{**binding.__dict__, "extensions": {}})
    with pytest.raises(CapabilityError, match="sparsity"):
        with Session(host, [Selection(es.NAME, "test/banded")], registry):
            pass
