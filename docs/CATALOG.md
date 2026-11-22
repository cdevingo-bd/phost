# Stage catalog

Each stage is a contract (`phost.stages.*.CONTRACT`, machine-readable, `Registry.describe()` lists them) with a
swap class, a float64 reference, metrics and tolerances. `phost.testing.contract.check_impl` judges an
implementation against it; the same check gates a leaderboard entry. Each stage also carries a judge that says how a run of
it is measured, so a new stage needs no change to the check ([docs/ADDING_A_STAGE.md](ADDING_A_STAGE.md)).

| stage | swap class | reference | hook on MJWarp | what an implementation provides |
| --- | --- | --- | --- | --- |
| `newton.linear_solve` | exact | `newton.linear_solve.float64`, shipped with the stage | `solver._cholesky_factorize_solve` | `solve(system)`: the Newton step for `H`, `g` |
| `effective.solve` | exact | `effective.solve.float64`, shipped with the stage | `derivative.eff_solve` | `solve(system)`: `x = M_hat^-1 rhs` for the host-assembled effective metric (discrete integrator only) |
| `dof.passive_force` | model changing | the implementation's own, `Impl.reference` | after `passive.passive` | `apply(system)`: force per scalar DOF from `q`, `v`, `time` |
| `control.law` | model changing | the implementation's own | before `forward.fwd_actuation` | `apply(system)`: `ctrl` from actuator `length`, `velocity`, `time` |
| `probe.state` | observer | the implementation's own | after `forward.forward` | `observe(system)`: `value[world, width]` from `qpos`, `qvel`, `qacc`, `time` |
| `body.wrench` | model changing | the implementation's own | before `forward.xfrc_accumulate` | `apply(system)`: force and torque per body from `xpos`, `cvel`, `time` |

`effective.solve` exists only under the discrete integrator, where MuJoCo Warp solves a mass-plus-stiffness system with
conjugate gradients before its constraint solver runs. The host adapter assembles that matrix densely per world (it
knows how the pieces add up) and hands it over, so an implementation never sees the host's layouts. The adapter refuses
the hook, naming why, whenever the host's system holds anything beyond mass, diagonal and flex stiffness: a model on
another integrator, no flex stiffness, flex contact or passive terms, fluid, an actuator or tendon term, or sleeping
(`host.refusal("effective.solve")` returns the reason, or None when the stage can be installed). Besides the float64 residual, the check
reports `max_native_difference`: the host's own solve of the same state, which catches an assembly that differs from the
matrix the host iterates with. A direct solve is only worth installing where the host's iteration is slow; measure first
(see the numbers in `phost-coupledchol`'s README).

## What the swap class means

* **exact**: same math, a different algorithm. Judged against the stage's reference to a tolerance, and on speed.
* **approximate**: same math to a bounded error. Judged on a speed and error front. (No stage uses it yet.)
* **model changing**: different physics. It cannot be compared with the host, because the host is not the truth; it is
  judged against its own float64 law (the implementation supplies it, and the registry refuses an implementation
  that does not) and against the effect it has on the host (the host's passive force changed by exactly the law;
  the host's actuator force is what MuJoCo (C) computes for that `ctrl`).
* **observer**: reads state, writes only to its own buffer. Judged on matching its own definition, on not touching the
  state, and on cost.

## What `check_impl` verifies for the stages that are not solver swaps

Inputs unchanged; outputs finite; the same inputs give bitwise the same outputs; the law matches its float64 reference
(`max_law_error` <= 1e-5); the host-side effect matches (`max_effect_error`); the runtime was called. For
`dof.passive_force` that reports `damping`, each entry is 0 or the law's own -d force / d v, checked by central
differences of the float64 reference one DOF at a time (`max_damping_error` <= 1e-4). Faults are tested: a wrong law,
forgetting the scalar-DOF mask, writing an input, nondeterminism, NaN, an adapter that drops the effect, a probe that
meddles with state, damping that is not the law's derivative. For `body.wrench`, `max_effect_error` is the host's `qfrc_smooth`
against MuJoCo (C) for the same state and the wrench the runtime wrote.

Every stage also gets `calls` from the host, not from the implementation: executions of the stage, counted on the
device, so every replay of a captured graph counts (RK4 runs forward dynamics, and so the stage, four times a step).

## Ground truth against MuJoCo (C), MJWarp host

* A spring law matches MuJoCo's own joint spring as closely as MJWarp's native spring does.
* A damping law that reports its damping (the default) matches MuJoCo's own joint damper to 1.6e-7, where MJWarp's native
  damper is 1.9e-7 away: the host integrates it implicitly, as it does the model's damping. With `implicit=False` the
  law is applied as a plain force; it matches MuJoCo with the damper applied explicitly to 1e-7 and is 3.4e-4 from the
  model's implicit damper over 150 steps at 2 ms (see `MJWARP_STEP.md`).
* A PD controller matches a MuJoCo loop that sets `ctrl` the same way.
* A probe leaves the trajectory bitwise unchanged; a captured graph replays all three stages.

## Not built, and why

See the table in `MJWARP_STEP.md`: mass-matrix solve, line search, constraint softness, collision, integrators.
