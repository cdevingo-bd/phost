import threading
from types import SimpleNamespace

import pytest

from phost import GuardError, Guard, PatchPoint, Target, install, source_hash


def native(x):
    return ("native", x)


def other(x):
    return ("other", x)


def make_point(ns, reviewed=None):
    target = Target("ns.f", lambda: ns.f, lambda f: setattr(ns, "f", f))
    reviewed = reviewed if reviewed is not None else {source_hash(native): "build A"}
    return PatchPoint("s", target, {"target": Guard("target", lambda: source_hash(target.get()), reviewed)})


def routed(replacement):
    return lambda original: (lambda x: replacement(x) if x == "mine" else original(x))


def test_install_routes_owner_and_forwards_others_then_restores():
    ns = SimpleNamespace(f=native)
    with install(make_point(ns), routed(lambda x: "replaced")) as dispatch:
        assert ns.f is dispatch
        assert ns.f("mine") == "replaced"
        assert ns.f("theirs") == ("native", "theirs")
    assert ns.f is native


def test_restores_even_when_the_body_raises():
    ns = SimpleNamespace(f=native)
    with pytest.raises(RuntimeError):
        with install(make_point(ns), routed(lambda x: x)):
            raise RuntimeError("boom")
    assert ns.f is native


def test_unreviewed_source_is_refused_with_the_hash_found():
    ns = SimpleNamespace(f=other)
    point = make_point(ns)
    with pytest.raises(GuardError) as error:
        with install(point, routed(lambda x: x)):
            pass
    assert source_hash(other) in str(error.value)
    assert "build A" in str(error.value)
    assert ns.f is other


def test_nested_install_is_refused_and_the_outer_scope_survives():
    ns = SimpleNamespace(f=native)
    point = make_point(ns)
    with install(point, routed(lambda x: "outer")) as outer:
        with pytest.raises(GuardError, match="nested"):
            with install(point, routed(lambda x: "inner")):
                pass
        assert ns.f is outer
        assert ns.f("mine") == "outer"
    assert ns.f is native


def test_extra_guards_are_checked_only_when_required():
    ns = SimpleNamespace(f=native)
    target = Target("ns.f", lambda: ns.f, lambda f: setattr(ns, "f", f))
    guards = {"target": Guard("target", lambda: source_hash(target.get()), {source_hash(native): "A"}),
              "assembly": Guard("assembly", lambda: "deadbeef", {"cafe": "A"})}
    point = PatchPoint("s", target, guards)
    with install(point, routed(lambda x: x)):
        pass                                                 # not required: fine
    with pytest.raises(GuardError, match="assembly"):
        with install(point, routed(lambda x: x), require=["assembly"]):
            pass
    assert ns.f is native
    with pytest.raises(GuardError, match="no guard named"):
        with install(point, routed(lambda x: x), require=["nope"]):
            pass


def test_patch_point_requires_a_target_guard():
    with pytest.raises(ValueError, match="target"):
        PatchPoint("s", Target("x", lambda: None, lambda f: None), {})


def test_concurrent_installs_are_serialized():
    ns = SimpleNamespace(f=native)
    point = make_point(ns)
    inside, overlap = [], []

    def worker():
        try:
            with install(point, routed(lambda x: x)):
                inside.append(1)
                if len(inside) > 1:
                    overlap.append(1)
                inside.pop()
        except GuardError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap and ns.f is native
