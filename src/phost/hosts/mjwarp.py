"""MuJoCo Warp host adapter, and the MJLab wrapper around it.

Stage ``newton.linear_solve`` replaces ``mujoco_warp._src.solver._cholesky_factorize_solve``.
That is a private MJWarp function, so every install is guarded by a hash of its source (and, for
implementations that use the structure hint, of the code that assembles the Hessian). A build whose
hash is not in the reviewed tables is refused, with the hash found in the message.

Keep the host (and the session) alive for as long as graphs captured under it are replayed: the mask
buffers the dispatch passes to the implementation belong to the host.
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any, Mapping

from ..errors import CapabilityError
from ..guard import Guard, source_hash
from ..host import Capability, Host, HostInfo, ModelFacts, StageBinding
from ..patch import PatchPoint, Target
from ..stages import newton_linear as nl

# Reviewed builds. Add a hash only after re-reviewing the new source. The label says what was reviewed.
REVIEWED_HOOKS = {
    "5f04c3acf13da87ed1d913c53da048c384e7f1312ae5bb8bf74f8ff0213d0433":
        "mujoco_warp 3.17.0 source build (f34c6dd5 + local patches)",
}
REVIEWED_ASSEMBLIES = {
    "fdc952e6c5fc771d60cc43e6618bb955f7147d55eb7de811057d92d4a657602a":
        "mujoco_warp 3.17.0 source build (f34c6dd5 + local patches)",
}
# Functions that assemble the Newton Hessian. The structure hint depends on how they build H.
_ASSEMBLY = ("_update_gradient", "_update_gradient_incremental", "_JTDACJ_sparse",
             "_update_gradient_h_incremental_sparse", "_update_gradient_init_h_sparse")


def _solver_module():
    from mujoco_warp._src import solver
    return solver


def _read_hook() -> str:
    return source_hash(_solver_module()._cholesky_factorize_solve)


def _read_assembly() -> str:
    solver = _solver_module()
    try:
        return source_hash(*[getattr(solver, name) for name in _ASSEMBLY])
    except AttributeError as error:
        from ..errors import GuardError
        raise GuardError(f"cannot find the Hessian assembly functions: {error}") from error


def _mass_pairs(model):
    """Stored off-diagonal mass-matrix entries as (row dof, column dof, element id) arrays."""
    import numpy as np
    elements = model.M_elemid.numpy()
    rows, cols = np.nonzero(np.tril(elements >= 0, -1))
    return rows.astype(np.int32), cols.astype(np.int32), elements[rows, cols].astype(np.int32)


class MJWarpHost(Host):
    capabilities = Capability.PATCH | Capability.NATIVE

    def __init__(self, model, data, *, cpu_model=None, name: str = "mjwarp"):
        import warp as wp
        from mujoco_warp._src import types
        from . import _mjwarp_kernels as kernels

        self.model, self.data, self.cpu_model = model, data, cpu_model
        self._k, self._wp, self._types = kernels, wp, types
        self.device = model.body_mass.device
        self._buffers: dict[str, Any] = {}
        self._stages: Mapping[str, StageBinding] | None = None
        self.info = HostInfo(name=name, version=_version(), build_id=None, patched=None)

    # -- Host ---------------------------------------------------------------------------------
    def facts(self) -> ModelFacts:
        m, t = self.model, self._types
        enable = m.opt.enableflags
        return ModelFacts(
            nv=int(m.nv), world_count=int(self.data.nworld), solver=t.SolverType(m.opt.solver).name.lower(),
            integrator=t.IntegratorType(m.opt.integrator).name.lower(), cone=t.ConeType(m.opt.cone).name.lower(),
            sparse=bool(m.is_sparse), sleep=bool(enable & t.EnableBit.SLEEP),
            device_kind="cuda" if self.device.is_cuda else "cpu", extra={"device": str(self.device)})

    def native(self) -> Any:
        return SimpleNamespace(model=self.model, data=self.data, cpu_model=self.cpu_model, solver=_solver_module())

    def stages(self) -> Mapping[str, StageBinding]:
        if self._stages is None:
            solver = _solver_module()
            target = Target("mujoco_warp._src.solver._cholesky_factorize_solve",
                            lambda: solver._cholesky_factorize_solve,
                            lambda f: setattr(solver, "_cholesky_factorize_solve", f))
            point = PatchPoint(nl.NAME, target, {
                "target": Guard("target", _read_hook, REVIEWED_HOOKS, "the Newton factor+solve hook"),
                "assembly": Guard("assembly", _read_assembly, REVIEWED_ASSEMBLIES, "how MJWarp assembles H"),
            })
            self._stages = {nl.NAME: StageBinding(
                nl.CONTRACT, point, _make_dispatch, extensions={"structure_hint": "assembly", "fallback": None})}
        return self._stages

    # -- buffers the dispatch hands to implementations (they must outlive captured graphs) -------
    def buffers(self) -> SimpleNamespace:
        if not self._buffers:
            wp, n = self._wp, int(self.data.nworld)
            rows, cols, element = _mass_pairs(self.model)
            self._buffers = dict(
                active=wp.zeros(n, dtype=wp.int32, device=self.device),
                refactor=wp.zeros(n, dtype=wp.int32, device=self.device),
                shadow_done=wp.ones(n, dtype=wp.bool, device=self.device),
                mass_rows=wp.array(rows, dtype=wp.int32, device=self.device),
                mass_cols=wp.array(cols, dtype=wp.int32, device=self.device),
                mass_element=wp.array(element, dtype=wp.int32, device=self.device))
        return SimpleNamespace(**self._buffers)


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("mujoco-warp")
    except Exception:
        import mujoco_warp
        return str(getattr(mujoco_warp, "__version__", "unknown"))


def _hint_supported(host: MJWarpHost, ctx) -> bool:
    """The structural hint is valid only for the full (non-compact) sparse system."""
    facts = host.facts()
    return facts.sparse and facts.integrator != "discrete" and facts.cone != "elliptic" and ctx.compact_d_full is None


def _make_dispatch(original, runtime, host: MJWarpHost, allowed: frozenset):
    wp, k = host._wp, host._k
    nworld, device = int(host.data.nworld), host.device
    bufs = host.buffers()

    def dispatch(m, d, ctx, skip_unchanged=False, skip_noflip=False):
        if m is not host.model:
            return original(m, d, ctx, skip_unchanged, skip_noflip)
        if d.nworld != nworld or d.qpos.device != device:
            raise ValueError("data world count or device mismatch")
        quad = ctx.quad_changed_count if skip_unchanged else d.nefc
        state = ctx.state_changed_count if skip_noflip else d.nefc
        wp.launch(k.masks, dim=nworld, inputs=[ctx.done, quad, state, skip_unchanged, skip_noflip],
                  outputs=[bufs.active, bufs.refactor], device=device)
        system = nl.NewtonSystem(h=ctx.h, grad=ctx.grad, active=bufs.active, refactor=bufs.refactor,
                                 search=ctx.search, search_dot=ctx.search_dot, newton_decrement=ctx.newton_decrement)
        if "fallback" in allowed:
            def fallback(mask):
                # Re-solve the requested active worlds with the host's own hook, always refactoring: the
                # host's cached factor may be stale because the implementation solved earlier calls.
                shadow = dataclasses.replace(ctx) if dataclasses.is_dataclass(ctx) else copy.copy(ctx)
                wp.launch(k.shadow_done, dim=nworld, inputs=[mask, bufs.active], outputs=[bufs.shadow_done],
                          device=device)
                shadow.done = bufs.shadow_done
                original(m, d, shadow, False, False)
            system.fallback = fallback
        if "structure_hint" in allowed and _hint_supported(host, ctx):
            nv = int(m.nv)

            def hint(flags, refactor):
                # flags accumulate over one solve (MJWarp updates H incrementally, and a subtracted row can
                # leave rounding residue on its support); the first call of a solve clears them.
                if not skip_unchanged:
                    wp.launch(k.clear_flags, dim=(nworld, nv), inputs=[refactor, flags], device=device)
                efc = d.efc
                wp.launch(k.mark_jacobian, dim=(nworld, efc.J_rownnz.shape[1]),
                          inputs=[refactor, d.nefc, efc.state, efc.J_rownnz, efc.J_rowadr, efc.J_colind, efc.J],
                          outputs=[flags], device=device)
                if bufs.mass_rows.shape[0]:
                    wp.launch(k.mark_mass, dim=(nworld, bufs.mass_rows.shape[0]),
                              inputs=[refactor, bufs.mass_rows, bufs.mass_cols, bufs.mass_element, d.M],
                              outputs=[flags], device=device)
            system.structure_hint = hint
        runtime.solve(system)

    return dispatch


class MJLabHost(MJWarpHost):
    """MJLab simulation: wraps MJWarp's model and data (``sim.wp_model``, ``sim.wp_data``).

    MJLab runs MuJoCo Warp, so everything MJWarpHost does applies; ``num_envs`` is the world count.
    """

    def __init__(self, sim):
        for attribute in ("wp_model", "wp_data", "mj_model"):
            if not hasattr(sim, attribute):
                raise CapabilityError(f"object {type(sim).__name__!r} has no {attribute!r}; expected an MJLab Simulation")
        super().__init__(sim.wp_model, sim.wp_data, cpu_model=sim.mj_model, name="mjlab")
