import pytest

from phost import Impl, ImplKind, ParamSpec, Registry, Selection
from phost.stages import newton_linear as nl
from phost.testing import runtimes
from phost.testing.fake import FakeHost


def make_impl(cls, name, **kw):
    return Impl(id=f"test/{name}", stage=nl.NAME, stage_version=1, version="1", title=name, kind=ImplKind.BUILTIN,
                factory=lambda host, params, cls=cls: cls(host, params), **kw)


@pytest.fixture
def registry():
    reg = Registry()
    nl.register(reg)
    for name, cls in dict(reference=runtimes.ReferenceRuntime, wrong=runtimes.WrongRuntime,
                          lowerreader=runtimes.LowerReaderRuntime, fallback=runtimes.FallbackRuntime,
                          inputwriter=runtimes.InputWriterRuntime,
                          inactivewriter=runtimes.InactiveWriterRuntime).items():
        reg.add_impl(make_impl(cls, name, extensions_used=("fallback",) if name == "fallback" else ()))
    return reg


@pytest.fixture
def host():
    return FakeHost()
