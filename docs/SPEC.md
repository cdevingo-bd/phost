# phost: design spec (draft)

phost is the layer between a physics host (MuJoCo, MJWarp, MJLab, others later)
and features that need that host: reading its state, or replacing one of its
stages. Features are written against phost's interfaces and never import a host
directly. Swapping hosts means swapping an adapter.

## Scope

In scope:

* Reading host state (today: flex vertex positions) through one interface.
* Replacing a named host stage (today: MJWarp's Newton Cholesky, `eff_solve`)
  with one guarded, scoped install mechanism.
* Declaring which host builds are known, and refusing the rest with a clear error.
* A conformance suite every adapter must pass, and a simulator-free `FakeHost`.

Out of scope:

* Being a simulator, or owning physics, contacts or rendering.
* Making kernels portable. A solver replacement is written against one host's
  internals. phost centralizes how it is attached, guarded and versioned, not what
  it computes.

## What the existing code asks of its host (audit)

Read side (`FlexEmbedding`, bindings):

* `data.flexvert_xpos`, shape `[world, vertex]`, device array, plus its device
* a refresh step (`kinematics`, `flex`), because a step leaves derived geometry stale
* per-flex layout (`flex_vertnum`, vertex offset, `nflexvert`) and the rest pose
  (hash-checked), taken from a CPU `MjModel`

Patch side:

| Stage | Replaced attribute | Source guard today | Lock today |
| --- | --- | --- | --- |
| Newton Cholesky (`NewtonFactorPolicy`) | `solver._cholesky_factorize_solve` | hash, reviewed for one patched 3.17 build | yes |
| Newton (`ParallelNewtonSolver`, older) | `solver._cholesky_factorize_solve` | different hash, reviewed for one 4.0.1 build | yes |
| Effective solve (`Dense/BandedEffectiveSolve`) | `derivative.eff_solve` | none, only a `supported(model)` check | no |

The three share one pattern and differ only in how well they guard it:
replace a module attribute, route the owner's model to the replacement and every
other model to the original, restore on exit. Policies also read model facts
(`nv`, sparse, solver, integrator, cone, sleep flag) through the host's native
enums, and the Newton path reads solver internals (`ctx.quad_changed_count`,
`ctx.state_changed_count`, `ctx.compact_d_full`, `M_elemid`).

## Dependencies

The core imports only the standard library. It does not import Warp, MuJoCo or
NumPy.

* Core types are plain data and `typing.Protocol`s. An array is whatever the host
  natively holds: `wp.array` on GPU hosts, `numpy.ndarray` on the CPU host. The
  core only requires `shape` and, for device arrays, `device`.
* Each adapter imports its own host lazily and declares an extra:
  `phost[mjwarp]`, `phost[mujoco]`, `phost[mjlab]`.
* Conversions live in optional helpers (for example `phost.interop.warp`), used by
  features that already depend on that library. `deform3d`'s kernels need Warp,
  so Warp is a dependency of the feature, not of phost.
* `FakeHost` and the conformance suite need NumPy and live in `phost.testing`.

## Core model

```python
class Capability(Flag):
    FLEX_POSITIONS; REFRESH; STEP_PATCH; NATIVE

@dataclass(frozen=True)
class HostInfo:            # who the host is
    name: str              # "mjwarp" | "mjlab" | "mujoco" | "fake"
    version: str
    build_id: str | None   # source sha for source builds; None for wheels
    patched: bool          # build carries local changes

@dataclass(frozen=True)
class FlexLayout:          # one flex, host-independent
    flex_id: int; vertex_offset: int; vertex_count: int
    rest_positions: Any    # (vertex_count, 3), for the rest-pose hash

@dataclass(frozen=True)
class ModelFacts:          # portable facts policies check
    nv: int; world_count: int
    solver: str; integrator: str; cone: str    # lower-case names
    sparse: bool; sleep: bool
    device_kind: str       # "cuda" | "cpu"

class Host(Protocol):
    info: HostInfo
    capabilities: Capability
    def facts(self) -> ModelFacts: ...
    def native(self) -> NativeHandles: ...    # escape hatch, needs Capability.NATIVE

class FlexSource(Host, Protocol):
    def flex_layout(self, flex_id: int) -> FlexLayout: ...
    def positions(self) -> Array: ...         # [world, vertex, 3]; zero-copy on GPU hosts
    def refresh(self, stream=None) -> None: ...

class PatchSurface(Host, Protocol):
    def resolve(self, point: PatchPoint) -> Target: ...
```

Portable facts cover what a policy needs to decide whether it applies. A policy
that needs host-specific model fields (the effective solve checks about a dozen)
asks for `native()`, which requires `Capability.NATIVE` and documents that it is
tied to that host. This is a deliberate escape hatch, not a leak to hide.

### PatchPoint

```python
@dataclass(frozen=True)
class PatchPoint:
    name: str                     # "newton.cholesky", "effective.solve"
    reviewed: Mapping[str, str]   # {sha256 of host source: label}, per host name
```

`PatchSurface.resolve` returns an object that can read the original, set the
replacement, and report the current source hash. Reviewed hash tables belong to
the feature that reviewed the replacement (`mjwarp-newton` owns the Newton hashes),
not to phost, which knows nothing about Newton.

### Install semantics (one rule for every patch point)

1. Reading the target's source hash must match a reviewed entry, else raise an
   error that names the hash found and the hosts known. A nested install fails
   here too, because the target is already a replacement.
2. The replacement routes calls whose model is the owner's, and forwards all other
   models to the original unchanged.
3. Install is scoped: `with phost.install(host, point, dispatch): ...` restores the
   original on every exit path. A process-wide re-entrant lock covers install and
   restore.
4. Install happens at warmup or graph capture. A graph captured under the
   replacement keeps using it after exit. This is documented, and the scope
   exists to bracket capture, not replay.
5. A policy-specific fallback (for example Hessian too dense) stays inside the
   policy. phost does not hide an unrecognized build behind a silent fallback.

## Adapters

| Adapter | FLEX_POSITIONS | REFRESH | STEP_PATCH | NATIVE | Notes |
| --- | --- | --- | --- | --- | --- |
| `MJWarpHost(model, data, cpu_model)` | yes, zero-copy | yes | yes | yes | Knows MJWarp builds; world count from `data.nworld` |
| `MJLabHost(sim)` | yes | yes | yes | yes | Resolves `sim.mj_model`, `sim.wp_model`, `sim.wp_data`, delegates to `MJWarpHost`; env count is the world count |
| `MuJoCoHost(model, data)` | yes, uploaded or numpy | yes (`mj_kinematics`, `mj_flex`) | no | yes | No solver hook exists in the CPU engine |
| `FakeHost` | yes | no-op | optional | no | Plain arrays, no simulator |

`phost.attach(obj)` picks an adapter by duck typing as a convenience. Explicit
constructors are the supported path. Asking for a capability the host lacks
raises an error naming it.

## Conformance suite

`phost.testing.conformance(make_host)` runs against any adapter:

* positions have shape `[world, vertex, 3]` and the declared device
* `refresh` brings stale derived geometry up to date, and positions are stale
  without it where the host defines that
* world-count and device mismatches are rejected
* `flex_layout` agrees with the CPU model for every flex
* for `STEP_PATCH` hosts: install then exit restores the original object, a
  nested install is rejected, an unknown hash is rejected with the hash in the
  message, and other models still reach the original

A new host is done when it passes the same suite as the others.

## How features use it

* **Geometry (`deform3d` today):** takes a `FlexSource`. It keeps its Warp
  kernels and its C++ core, but no host-specific code. Its C++ MuJoCo adapter
  stays as the C++ reader and shares only the `.bind` file format.
* **Solver (`mjwarp-newton` today):** takes a `PatchSurface`, defines its
  `PatchPoint`s and reviewed hashes, and keeps its kernels.

## Migration map (current modules)

| Today | Becomes |
| --- | --- |
| `mujoco_bridge.FlexEmbedding` host reads | `FlexSource` + `MJWarpHost`/`MJLabHost` |
| `bindings.read_bindings(..., model)` host reads | `FlexLayout` from `FlexSource.flex_layout` |
| `_LOCK`, hash checks, `capture_policy` in `mujoco_newton_factor` | `phost.install` + a `PatchPoint` in the solver package |
| `ParallelNewtonSolver.capture_policy` | removed or ported, see open questions |
| `_EffectiveSolvePolicy.capture_policy` | `phost.install` + a new `PatchPoint` with reviewed hashes |
| model option reads via `mujoco_warp._src.types` | `ModelFacts`, `native()` for the rest |

## Open questions

1. **Effective solves have no source guard today.** Porting them needs reviewed
   hashes for each supported build, which means reviewing `eff_solve` on those
   builds first.
2. **`ParallelNewtonSolver`** is reviewed only for one 4.0.1 build and is the
   older design. Port it, or delete it once its tests are covered by the
   coupled policy.
3. **Supported builds.** The reviewed Newton hash is for a patched 3.17 source
   build. The upstream gap should be written down so a release can replace it.
4. **Repo layout.** One repo with features as subpackages, or phost separate from
   the geometry and solver projects. Either works with these interfaces.
5. **Names for the feature projects** once the geometry code is built on phost.
6. **Beyond MuJoCo hosts.** Nothing here assumes MuJoCo, but no other host has
   been audited, so no other adapter is specified yet.

## Milestones

1. `phost` core, `FakeHost` and the conformance suite, with no host installed.
2. `MJWarpHost`; port `FlexEmbedding` to `FlexSource` and run its existing tests on
   `FakeHost` and `MJWarpHost`.
3. `MJLabHost` and `MuJoCoHost`, all four passing conformance.
4. Port the Newton policy and the effective solves to `phost.install`.
5. CI matrix over supported host builds, plus a scheduled run against upstream
   MJWarp `main` to flag drift.
