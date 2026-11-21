"""Built-in implementations of ``probe.state``."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind
from ..registry import Registry
from ..stages import probe_state as stage

PEAK_ID = f"{stage.NAME}/peak"


def _reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    """[max |qvel|, max |qacc|, |qvel|_2] per world, in float64."""
    import numpy as np
    v, a = np.asarray(inputs["qvel"], np.float64), np.asarray(inputs["qacc"], np.float64)
    return dict(value=np.stack([np.abs(v).max(axis=1), np.abs(a).max(axis=1), np.sqrt((v * v).sum(axis=1))], axis=1))


class _Numpy:
    width = 3

    def __init__(self, params):
        self.calls = 0
        self._last = None

    def observe(self, system):
        import numpy as np
        v, a = system.qvel.astype(np.float64), system.qacc.astype(np.float64)
        system.value[:] = np.stack([np.abs(v).max(axis=1), np.abs(a).max(axis=1), np.sqrt((v * v).sum(axis=1))],
                                   axis=1).astype(system.value.dtype)
        self._last = system.value
        self.calls += 1

    def metrics(self):
        import numpy as np
        out: dict = dict(calls=self.calls)
        if self._last is not None:
            last = self._last.numpy() if hasattr(self._last, "numpy") else np.asarray(self._last)
            out["last_max_qvel"], out["last_max_qacc"] = float(last[:, 0].max()), float(last[:, 1].max())
        return out


class _Warp(_Numpy):
    def observe(self, system):
        import warp as wp
        from . import _warp
        wp.launch(_warp.peak, dim=system.value.shape[0], inputs=[system.qvel, system.qacc], outputs=[system.value],
                  device=system.value.device)
        self._last = system.value
        self.calls += 1


def _factory(host, params):
    return (_Warp if host.backend == "warp" else _Numpy)(params)


def register(registry: Registry) -> None:
    registry.add_impl(Impl(
        id=PEAK_ID, stage=stage.NAME, stage_version=1, version="1", title="Peak speed and acceleration",
        kind=ImplKind.BUILTIN, doc="value = [max |qvel|, max |qacc|, |qvel|_2] per world.",
        fit="An example probe: three numbers that show a simulation blowing up before it produces NaNs.",
        factory=_factory, reference=_reference))
