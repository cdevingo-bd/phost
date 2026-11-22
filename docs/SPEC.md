# phost: design (matches the code)

phost is the layer between a physics host (MuJoCo Warp, MJLab, others later) and anything that
replaces one of the host's stages. A stage is a named step with a typed contract. An
implementation of a stage is written against the contract, never against a host. A host adapter
maps its own function onto the contract. Swapping hosts means swapping an adapter; swapping math
means swapping an implementation.

## What exists

| Piece | Where | Needs |
| --- | --- | --- |
| Stage contract (machine-readable: ports, layouts, invariants, swap class, tolerances, metrics, reference) | `stage.py` | stdlib |
| How a stage's law can be typed (`Pointwise`, `Channels`, `Iteration`), as data checked against the contract | `equation.py` | stdlib |
| Source guards and one scoped install rule | `guard.py`, `patch.py` | stdlib |
| Implementations (metadata, params, applicability, advisories) and registry (describes itself as JSON) | `impl.py`, `registry.py` | stdlib |
| Session: apply a selection to a host for a scope, read metrics | `session.py` | stdlib |
| Contracts `newton.linear_solve` and `effective.solve`, each with its float64 reference and evaluator | `stages/newton_linear.py`, `stages/effective_solve.py` | numpy for reference |
| Contracts `dof.passive_force`, `control.law`, `probe.state` (see [CATALOG.md](CATALOG.md)) | `stages/` | numpy for evaluation |
| Built-in implementations of those three, numpy and Warp backends, each with its float64 reference | `impls/` | numpy, warp |
| MuJoCo Warp and MJLab adapters (what each stage relies on: [MJWARP_STEP.md](MJWARP_STEP.md)) | `hosts/mjwarp.py` | warp, mujoco_warp |
| A host that is not MuJoCo-shaped: a spring-mass chain in NumPy, with passive force, control, probe and a banded effective solve | `hosts/chain.py` | numpy |
| Stage dispatch as data (`Hook`), shared by every host | `hook.py` | stdlib |
| The nonzero structure of a system matrix, as a host-provided extension | `sparsity.py` | stdlib |
| Simulator-free fake host, faulty runtimes, `check_impl` | `testing/` | numpy |

The core imports only the standard library. Adapters import their simulator lazily. Arrays are
whatever the host holds (Warp arrays on GPU hosts); the core needs only `.shape` and `.dtype`.

## Swap classes

Every contract declares what kind of replacement it accepts, because scoring depends on it:

* `exact`: same math, different algorithm. Judged against a float64 reference with a tolerance.
* `approximate`: same math, bounded error. Judged on a speed/error front.
* `model_changing`: different physics. Judged against ground truth, not against the host. The contract says
  `reference="impl"`: each implementation supplies its own float64 law (`Impl.reference`), and the registry
  refuses one that does not.
* `observer`: reads state and writes only its own buffer. Judged on matching its own definition and on cost.

`newton.linear_solve` and `effective.solve` are `exact`; `dof.passive_force`, `control.law` and `body.wrench` are `model_changing`;
`probe.state` is `observer`.

A stage's class is the default for its implementations. An implementation of an `exact` stage may declare itself `approximate`
(`Impl.swap_class`) with gates of its own (`Impl.tolerances(params)`, replacing the stage's for the metrics they name): a typed
solver step claims a residual and is held to it, and is ranked as approximate. A `model_changing` or `observer` stage cannot be
relaxed.

## Install rule (one for every stage)

1. The target must not already be patched by phost (no nested installs).
2. Every required guard must match a reviewed source hash, else `GuardError` names the hash found.
3. Install and restore are serialized by one process-wide re-entrant lock.
4. The original is restored on every exit path.
5. The replacement serves the owner's model; every other call goes to the original.

A graph captured while installed keeps the replacement after the scope ends. The scope brackets
warmup and capture, not replay. Keep the host and session alive while replaying captured graphs
(the mask buffers the dispatch passes to an implementation belong to the host).

## Decisions the implementation forced

* **Reviewed hashes live with the adapter, not the algorithm.** They protect the adapter's
  assumptions about the host's solver context. An algorithm package has none.
* **The structure hint is host knowledge.** Marking coupled DOFs from MJWarp's constraint Jacobian and
  mass matrix moved out of the algorithm into the adapter, behind its own guard (the Hessian
  assembly). An implementation that uses it declares `extensions_used=("structure_hint",)`.
* **A hint is exact only for Hessians the host assembled.** A synthetic H has no relation to the
  host's Jacobian, so the hint misses its coupling. The contract check therefore has a `step` mode
  that captures real hook calls during a real `mjw.step`.
* **The host counts executions, not the implementation.** An implementation's Python runs once per eager call and per
  capture, never per replay, and could report anything. A contract names its `entry` method, the session wraps it, and
  the host's `Meter` ticks at every execution; on a device host the increment is device work, so it repeats on replay.
  Measured cost: under 1% on the smallest scene (0.17 ms per step), unmeasurable at nv=330.
* **An extension can need a second host function.** `StageBinding.extensions` maps a name to a `Provision`: the guard
  that must pass and, when the host must also change what it does elsewhere, a second patch point and its wrapper,
  installed, guarded and restored with the stage. `dof.passive_force`'s `implicit_damping` is one: the law reports
  `damping` (its -d force / d v) and the host's Euler step integrates it implicitly, like the model's own. An
  implementation lists extensions it needs in `extensions_used` (refused if the host lacks one) and extensions it
  merely likes in `extensions_optional` (a missing or unreviewed one is recorded as an advisory and the law runs without).
* **Structure is a fact about the model, so the host provides it.** `effective.solve` offers the `sparsity` extension: a
  `Sparsity` (the symmetric pattern in compressed rows, with coupled rows, bandwidth and a reverse Cuthill-McKee order)
  read from the model's own index arrays, not from the values in `h`. It is static; the Newton `structure_hint` is the
  per-call analogue. Operation counts for a dense, a coupled-rows and a banded factorization come from the same object.
* **The cache belongs to the implementation.** The contract's `refactor` means "H changed"; an
  implementation also refactors worlds for which it holds no valid factor (first call, or a call it
  handed to the host).
* **The fallback is clamped to active worlds** by the adapter, so an implementation cannot make the
  host write outputs the contract says stay untouched.
* **Only defined input entries must be preserved.** The lower triangle of H, and anything beyond the
  leading nv entries, may hold any value (including NaN) and may be overwritten.
* **Advisories are separate from applicability.** A hard `applicable` refuses; an `advisory`
  warns. Measured: the coupled path is slower than the host at nv=138 and wins from about nv=300.

## Reviewed hosts

Only one build is reviewed: MuJoCo Warp 3.17.0 source build (f34c6dd5 plus local patches), for both
the Newton hook, the Hessian assembly, the discrete step's effective-metric solve (`derivative.eff_solve` and the three functions it is built from), the three functions the pointwise stages wrap
(`passive.passive`, `forward.fwd_actuation`, `forward.forward`) and `forward.euler` with the two kernels it launches for
its implicit damping. Any other build is refused with its hash in the
message (mujoco_warp 4.0.1 is refused today). Supporting another build means reviewing those functions on it
first, then adding the hashes.
The MJLab adapter wraps `sim.wp_model`, `sim.wp_data`, `sim.mj_model`; it has been tested against a
stand-in object, not against a real MJLab.

## Not built yet

* More stages. Five contracts exist; the audit in [MJWARP_STEP.md](MJWARP_STEP.md) lists what else the step
  offers and why it is not built (mass-matrix solve, line search, constraint softness, collision, integrators).
  The graph spec, scene registry, isolated runner and CLI live in the sibling package `phost-lab`.
* Authoring lives in the sibling package `phost-author`: typed equations for the two law stages, and a sandbox
  for user-written runtimes. Its README states what the sandbox does not protect.
* Scoring and the leaderboard live in the sibling package `phost-score`, and the browser UI in `phost-dash`. Both are
  separate packages, not part of this one.

## Roadmap

1. **Headless lab:** done, in `phost-lab`.
2. **Stage catalog:** done for five stages (see [CATALOG.md](CATALOG.md)).
3. **Authoring:** done, in `phost-author` (typed equations; sandboxed kernel submissions).
