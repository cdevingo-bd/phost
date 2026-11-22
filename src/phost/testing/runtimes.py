"""Small runtimes used by tests: a correct one, and deliberately faulty ones."""

from __future__ import annotations

import numpy as np

from ..stages import effective_solve as es
from ..stages import newton_linear as nl


class ReferenceRuntime:
    """Solves in float64 with numpy. Correct by construction."""

    def __init__(self, host, params):
        self.host = host

    def solve(self, s: nl.NewtonSystem) -> None:
        nv = self.host.model.nv
        for w in range(s.h.shape[0]):
            if not s.active[w]:
                continue
            g = np.asarray(s.grad[w, :nv], np.float64)
            x = np.linalg.solve(nl._symmetric(s.h[w], nv), g)
            s.search[w] = (-x).astype(np.float32)
            s.search_dot[w] = np.float32(x @ x)
            s.newton_decrement[w] = np.float32(g @ x)

    def metrics(self):
        return {"status": 0, "fallbacks": 0}


class WrongRuntime(ReferenceRuntime):
    """Returns a step that is a little wrong: fails the accuracy gate."""

    def solve(self, s):
        super().solve(s)
        s.search *= 1.01


class LowerReaderRuntime(ReferenceRuntime):
    """Reads the undefined lower triangle (treats H as a full matrix): must fail the gate."""

    def solve(self, s):
        nv = self.host.model.nv
        for w in range(s.h.shape[0]):
            if s.active[w]:
                a = np.asarray(s.h[w, :nv, :nv], np.float64)
                a = np.triu(a) + np.tril(a, -1)
                try:
                    x = np.linalg.solve(a, np.asarray(s.grad[w, :nv], np.float64))
                except np.linalg.LinAlgError:
                    x = np.zeros(nv)
                s.search[w], s.search_dot[w], s.newton_decrement[w] = -x, x @ x, s.grad[w, :nv] @ x


class FallbackRuntime(ReferenceRuntime):
    """Hands every world back to the host."""

    def solve(self, s):
        s.fallback(np.ones_like(s.active))

    def metrics(self):
        return {"status": 0, "fallbacks": 1}


class InputWriterRuntime(ReferenceRuntime):
    """Modifies an input: must be caught."""

    def solve(self, s):
        super().solve(s)
        s.grad[:] += 1.0


class InactiveWriterRuntime(ReferenceRuntime):
    """Writes outputs of inactive worlds: must be caught."""

    def solve(self, s):
        super().solve(s)
        s.search[:] += 0.5


class EffectiveReferenceRuntime:
    """Solves the effective-metric system in float64 with numpy. Correct by construction."""

    def __init__(self, host, params):
        self.host = host

    def solve(self, s: es.EffectiveSystem) -> None:
        nv = self.host.facts().nv
        h, rhs = np.asarray(s.h.numpy() if hasattr(s.h, "numpy") else s.h), np.asarray(s.rhs.numpy() if hasattr(s.rhs, "numpy") else s.rhs)
        out = np.zeros((h.shape[0], nv), np.float32)
        for w in range(h.shape[0]):
            out[w] = np.linalg.solve(es._symmetric(h[w], nv), np.asarray(rhs[w, :nv], np.float64)).astype(np.float32)
        if hasattr(s.qacc, "assign"):
            s.qacc.assign(out)
        else:
            s.qacc[:, :nv] = out

    def metrics(self):
        return {"status": 0}


class WrongEffectiveRuntime(EffectiveReferenceRuntime):
    """A solution that is a little wrong: fails the residual gate."""

    def solve(self, s):
        super().solve(s)
        if hasattr(s.qacc, "assign"):
            s.qacc.assign(s.qacc.numpy() * 1.01)
        else:
            s.qacc *= 1.01


class EffectiveInputWriterRuntime(EffectiveReferenceRuntime):
    """Modifies the right-hand side: must be caught."""

    def solve(self, s):
        super().solve(s)
        if hasattr(s.rhs, "assign"):
            s.rhs.assign(s.rhs.numpy() + 1.0)
        else:
            s.rhs[:] += 1.0
