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
| Source guards and one scoped install rule | `guard.py`, `patch.py` | stdlib |
| Implementations (metadata, params, applicability, advisories) and registry (describes itself as JSON) | `impl.py`, `registry.py` | stdlib |
| Session: apply a selection to a host for a scope, read metrics | `session.py` | stdlib |
| Contract `newton.linear_solve`, its float64 reference and evaluator | `stages/newton_linear.py` | numpy for reference |
| MuJoCo Warp and MJLab adapters | `hosts/mjwarp.py` | warp, mujoco_warp |
| Simulator-free fake host, faulty runtimes, `check_impl` | `testing/` | numpy |

The core imports only the standard library. Adapters import their simulator lazily. Arrays are
whatever the host holds (Warp arrays on GPU hosts); the core needs only `.shape` and `.dtype`.

## Swap classes

Every contract declares what kind of replacement it accepts, because scoring depends on it:

* `exact`: same math, different algorithm. Judged against a float64 reference with a tolerance.
* `approximate`: same math, bounded error. Judged on a speed/error front.
* `model_changing`: different physics. Judged against ground truth, not against the host.

`newton.linear_solve` is `exact`.

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
the hook and the Hessian assembly. Any other build is refused with its hash in the message. Supporting
another build means reviewing the hook and the assembly on it first, then adding the hash.
The MJLab adapter wraps `sim.wp_model`, `sim.wp_data`, `sim.mj_model`; it has been tested against a
stand-in object, not against a real MJLab.

## Not built yet

* The graph spec (a serializable description of which implementation sits on which stage), a scene
  registry, a runner that executes a graph on a scene in an isolated process and writes JSON, and
  metric plugins. See the roadmap below.
* More stages. Only `newton.linear_solve` has a contract. A stage catalog needs an audit of the
  host's whole step.
* Authoring: an expression DSL for pointwise stages, and kernel authoring with a sandbox.
* A scoring and leaderboard layer, an event stream, and a UI. These are internal products and live in
  a private repository, not here.

## Roadmap

1. **Headless lab:** graph spec, scene registry, runner, metric plugins, CLI. Exit: one command takes a
   graph file and a scene and writes a JSON result that reproduces the A/B numbers.
2. **Stage catalog:** audit the host step and define contracts, each with a reference and a swap class.
3. **Authoring:** expression DSL for one pointwise stage; kernel authoring for the linear solve, both
   in a sandbox.
