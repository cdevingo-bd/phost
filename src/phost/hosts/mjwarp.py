"""MuJoCo Warp host adapter, and the MJLab wrapper around it.

Stage ``newton.linear_solve`` replaces ``mujoco_warp._src.solver._cholesky_factorize_solve`` and stage
``effective.solve`` replaces ``mujoco_warp._src.derivative.eff_solve``. The stages
``dof.passive_force``, ``control.law`` and ``probe.state`` wrap ``passive.passive``, ``forward.fwd_actuation``
and ``forward.forward``. These are private MJWarp functions, so every install is guarded by a hash of the
function's source (and, for implementations that use the structure hint, of the code that assembles the
Hessian). A build whose hash is not in the reviewed tables is refused, with the hash found in the message.
``docs/MJWARP_STEP.md`` records what each stage relies on.

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
from ..host import Capability, Host, HostInfo, ModelFacts, NativeTiming, Provision, StageBinding
from ..patch import PatchPoint, Target, original_of
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
_BUILD = "mujoco_warp 3.17.0 source build (f34c6dd5 + local patches)"
# Reviewed for the pointwise stages. What each stage relies on, as read in that build:
#   passive.passive      rebuilds d.qfrc_passive from scratch on every call; fwd_acceleration reads it afterwards
#   forward.fwd_actuation reads d.ctrl directly when the model has no history (nhistory == 0)
#   forward.forward      ends with the solved d.qacc; step() calls it through the module, once per step
REVIEWED_PASSIVE = {"908ad56584892abc221c6b98b47ea20e3582aa71fc0515aeb1e925101f5e2ba5": _BUILD}
REVIEWED_ACTUATION = {"8c0bf41d54b65408e230e0d0500ea35a17b5477987f9e2e3a40af63024e71271": _BUILD}
REVIEWED_FORWARD = {"24421d5832eceecadd616dce6424b9db38efc7af5b7e80e3a645814d9ebc1db7": _BUILD}
# forward.euler, with the two kernels it launches for its implicit damping: euler reads m.dof_damping and
# m.dof_dampingpoly through _compute_damping_deriv (linear + 2 p0 |v| + 3 p1 v^2, so adding to the linear term adds
# to the derivative) and _euler_damp_qfrc adds timestep * that to the diagonal of the mass matrix it solves with.
REVIEWED_EULER = {"7023bc7c8027667766292b6038e3e43cd446169a28edebc050ca0200ee7f06cc": _BUILD}
_EULER = ("euler", "_compute_damping_deriv", "_euler_damp_qfrc")
# forward.fwd_acceleration calls derivative.eff_solve(m, d, d.qacc_smooth) when the integrator is DISCRETE. The stage
# replaces it for the model where the effective metric is exactly M + diag + K: the guard hashes the solve, the
# matrix-vector product it iterates with, the CSR kernel that product launches, and the right-hand-side kernel. M_hat is
# assembled from the same arrays that kernel reads (M_mulm_*, efm_K_*, d.M, d.efm_diag, d.efm_K_val).
REVIEWED_EFFECTIVE = {"810004a7688ebb507b79f59d6fc52fc3ba9f8cffeafb1fd991cef5e21ab32f5a": _BUILD}
_EFFECTIVE = ("eff_solve", "eff_mul_m", "_eff_rhs", "_eff_mul_diag_and_csr")
# Functions that assemble the Newton Hessian. The structure hint depends on how they build H.
_ASSEMBLY = ("_update_gradient", "_update_gradient_incremental", "_JTDACJ_sparse",
             "_update_gradient_h_incremental_sparse", "_update_gradient_init_h_sparse")


def _solver_module():
    from mujoco_warp._src import solver
    return solver


def _forward_module():
    from mujoco_warp._src import forward
    return forward


def _passive_module():
    from mujoco_warp._src import passive
    return passive


def _derivative_module():
    from mujoco_warp._src import derivative
    return derivative


def effective_blocker(model, types) -> str:
    """Why ``effective.solve`` cannot replace the host's solve for this model, or an empty string.

    The host solves M_hat qacc = rhs with an iterative method. A direct solve of the assembled matrix is exact only when
    M_hat is exactly the mass matrix plus a diagonal plus the CSR flex stiffness: every other term the host can add (flex
    contact, matrix-free flex terms, fluid, tendons, actuators, a backbone preconditioner) is absent. Read from Python
    attributes of the model only, so it is safe to ask while a graph is being captured."""
    try:
        disable, enable = int(model.opt.disableflags), int(model.opt.enableflags)
        checks = [
            (model.opt.integrator == types.IntegratorType.DISCRETE, "the host only calls this solve with the discrete integrator"),
            (model.nefmK > 0, "the model has no flex stiffness block: the host solves with its stored factorization"),
            (not (disable & types.DisableBit.SPRING and disable & types.DisableBit.DAMPER),
             "spring and damper are both disabled, so the host takes its direct path"),
            (not model.has_non_simple_flex, "the model has flex terms the host applies matrix-free"),
            (not model.has_flex_passive, "the model has a flex contact term"),
            (not model.has_fluid, "the model has a fluid term"),
            (not model.has_efm_actuator, "the model has an actuator term"),
            (not model.has_tendon_stiffness, "the model has a tendon stiffness term"),
            (not model.has_tendon_damping, "the model has a tendon damping term"),
            (not model.efm0_active, "the model uses the host's backbone preconditioner path"),
            (model.flex_interp_assemblable, "the model's flex interpolation cannot be assembled"),
            (not enable & types.EnableBit.SLEEP, "sleeping is enabled"),
        ]
    except AttributeError as error:
        return f"this MuJoCo Warp build does not expose {error.args[0] if error.args else 'a field'} the stage reads"
    for ok, reason in checks:
        if not ok:
            return reason
    return ""


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


# Stage bindings other than the Newton solve are built by factories, each ``(host) -> StageBinding``, registered here. A
# module that adds a stage to this host calls ``register_binding`` once at import; the shipped ones live in
# ``mjwarp_stages`` and are loaded at the bottom of this module.
BINDING_FACTORIES: list = []


def register_binding(factory):
    """Add a stage to every MuJoCo Warp host built from now on (and to hosts that have not yet listed their stages)."""
    if factory not in BINDING_FACTORIES:
        BINDING_FACTORIES.append(factory)
    return factory


class MJWarpHost(Host):
    capabilities = Capability.PATCH | Capability.NATIVE
    backend = "warp"

    def __init__(self, model, data, *, cpu_model=None, name: str = "mjwarp"):
        import warp as wp
        from mujoco_warp._src import types
        from . import _mjwarp_kernels as kernels

        self.model, self.data, self.cpu_model = model, data, cpu_model
        self._k, self._wp, self._types = kernels, wp, types
        self.device = model.body_mass.device
        self._buffers: dict[str, Any] = {}
        self._stages: Mapping[str, StageBinding] | None = None
        self._scalar = self._read_scalar_dofs()          # the only device read facts() needs; see _scalar_dofs
        self._effective_blocker = effective_blocker(model, types)
        self.info = HostInfo(name=name, version=_version(), build_id=None, patched=None)

    # -- Host ---------------------------------------------------------------------------------
    def facts(self) -> ModelFacts:
        m, t = self.model, self._types
        enable = m.opt.enableflags
        return ModelFacts(
            nv=int(m.nv), world_count=int(self.data.nworld), solver=t.SolverType(m.opt.solver).name.lower(),
            integrator=t.IntegratorType(m.opt.integrator).name.lower(), cone=t.ConeType(m.opt.cone).name.lower(),
            sparse=bool(m.is_sparse), sleep=bool(enable & t.EnableBit.SLEEP),
            device_kind="cuda" if self.device.is_cuda else "cpu",
            extra={"device": str(self.device), "nq": int(m.nq), "nu": int(m.nu),
                   "nhistory": int(getattr(m, "nhistory", 0)), "scalar_dofs": int(self._scalar_dofs()[0].sum())})

    def native(self) -> Any:
        return SimpleNamespace(model=self.model, data=self.data, cpu_model=self.cpu_model, solver=_solver_module())

    def meter(self):
        return self._k.DeviceMeter(self.device)

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
                nl.CONTRACT, point, _make_dispatch,
                extensions={"structure_hint": Provision(guard="assembly"), "fallback": Provision()},
                native_timing=NativeTiming(
                    lambda host: (lambda: original_of(solver.solve)(host.model, host.data)),
                    enclosing="the solver phase (solver.solve), which runs the stage once per Newton iteration"))}
            for factory in BINDING_FACTORIES:                  # every other stage: see mjwarp_stages.py
                binding = factory(self)
                self._stages[binding.contract.name] = binding
        return self._stages

    def _scalar_dofs(self):
        """(scalar mask, qpos address) per DOF: scalar DOFs belong to hinge and slide joints.

        Joint topology does not change, so it is read from the device once, when the host is built. ``facts()`` uses
        it, and the dispatches call ``facts()`` while a graph is being captured, where a device-to-host copy is an error.
        """
        return self._scalar

    def _read_scalar_dofs(self):
        import numpy as np
        m, t = self.model, self._types
        jtype, qadr, dadr = m.jnt_type.numpy(), m.jnt_qposadr.numpy(), m.jnt_dofadr.numpy()
        scalar = np.zeros(int(m.nv), np.int32)
        address = np.full(int(m.nv), -1, np.int32)
        for j in range(int(m.njnt)):
            if jtype[j] in (int(t.JointType.HINGE), int(t.JointType.SLIDE)):
                scalar[dadr[j]], address[dadr[j]] = 1, qadr[j]
        return scalar, address

    # -- buffers the dispatch hands to implementations (they must outlive captured graphs) -------
    def buffers(self) -> SimpleNamespace:
        if not self._buffers:
            wp, n = self._wp, int(self.data.nworld)
            rows, cols, element = _mass_pairs(self.model)
            scalar, address = self._scalar_dofs()
            nv = int(self.model.nv)
            self._buffers = dict(
                scalar=wp.array(scalar, dtype=wp.int32, device=self.device),
                dof_qposadr=wp.array(address, dtype=wp.int32, device=self.device),
                q=wp.zeros((n, nv), dtype=wp.float32, device=self.device),
                force=wp.zeros((n, nv), dtype=wp.float32, device=self.device),
                damping=wp.zeros((n, nv), dtype=wp.float32, device=self.device),
                effective_damping=wp.zeros((n, nv), dtype=wp.float32, device=self.device),
                active=wp.zeros(n, dtype=wp.int32, device=self.device),
                refactor=wp.zeros(n, dtype=wp.int32, device=self.device),
                shadow_done=wp.ones(n, dtype=wp.bool, device=self.device),
                mass_rows=wp.array(rows, dtype=wp.int32, device=self.device),
                mass_cols=wp.array(cols, dtype=wp.int32, device=self.device),
                mass_element=wp.array(element, dtype=wp.int32, device=self.device))
        return SimpleNamespace(**self._buffers)

    def effective_buffers(self) -> SimpleNamespace:
        """The dense M_hat and right-hand side the effective-solve dispatch fills (allocated when first asked for)."""
        if "effective/matrix" not in self._buffers:
            wp, n, nv = self._wp, int(self.data.nworld), int(self.model.nv)
            self._buffers["effective/matrix"] = wp.zeros((n, nv, nv), dtype=wp.float32, device=self.device)
            self._buffers["effective/rhs"] = wp.zeros((n, nv), dtype=wp.float32, device=self.device)
        return SimpleNamespace(matrix=self._buffers["effective/matrix"], rhs=self._buffers["effective/rhs"])

    def hook_support(self):
        host, wp = self, self._wp

        class Support:
            def sizes(self):
                return dict(world=int(host.data.nworld), nv=int(host.model.nv), nu=int(host.model.nu), nbody=int(host.model.nbody))

            def check(self, d):
                if d.nworld != int(host.data.nworld) or d.qpos.device != host.device:
                    raise ValueError("data world count or device mismatch")

            def zero(self, array):
                array.zero_()

            def launch(self, kernel, dims, inputs, outputs):
                wp.launch(getattr(host._k, kernel), dim=dims, device=host.device, inputs=inputs, outputs=outputs)

            def buffers(self):
                return host.buffers()

        return Support()

    def probe_buffer(self, width: int):
        """The host-owned ``value`` array of a probe of this width (created once, so graphs can capture it)."""
        key = f"probe/{width}"
        if key not in self._buffers:
            self._buffers[key] = self._wp.zeros((int(self.data.nworld), width), dtype=self._wp.float32, device=self.device)
        return self._buffers[key]


def _point(stage, module, attribute, reviewed, also=()) -> PatchPoint:
    """A function to wrap. ``also`` names further functions or kernels of the module the guard hashes with it, for
    behaviour of the wrapped function that lives in what it calls."""
    target = Target(f"{module.__name__}.{attribute}", lambda: getattr(module, attribute),
                    lambda f: setattr(module, attribute, f))
    return PatchPoint(stage, target, {"target": Guard(
        "target", lambda: source_hash(*[getattr(module, name) for name in (attribute, *also)]), reviewed,
        f"{attribute}, the function being wrapped" + (f", with {', '.join(also)}" if also else ""))})


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


def _check_same_model(host: MJWarpHost, d) -> None:
    if d.nworld != int(host.data.nworld) or d.qpos.device != host.device:
        raise ValueError("data world count or device mismatch")


def _wrap_euler(original, host: MJWarpHost):
    """Euler integrates the model's joint damping implicitly. Give it the damping the law reported on top of the
    model's own, for this call, by handing it a copy of the model whose ``dof_damping`` is the sum. The user's model is
    not touched. The passive force of the law is already in qfrc_passive, so the force and its implicit term agree."""
    wp, k = host._wp, host._k
    bufs, nworld, nv, device = host.buffers(), int(host.data.nworld), int(host.model.nv), host.device

    def dispatch(m, d):
        if m is not host.model:
            return original(m, d)
        _check_same_model(host, d)
        wp.launch(k.fold_damping, dim=(nworld, nv), inputs=[m.dof_damping, bufs.damping],
                  outputs=[bufs.effective_damping], device=device)
        seen = copy.copy(m)
        seen.dof_damping = bufs.effective_damping
        original(seen, d)

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


from . import mjwarp_stages  # noqa: E402,F401  (registers the shipped stage bindings)
from . import mjwarp_body_wrench  # noqa: E402,F401
