# Stage catalog

Each stage is a contract (`phost.stages.*.CONTRACT`, machine-readable, `Registry.describe()` lists them) with a
swap class, a float64 reference, metrics and tolerances. `phost.testing.contract.check_impl` judges an
implementation against it; the same check gates a leaderboard entry.

| stage | swap class | reference | hook on MJWarp | what an implementation provides |
| --- | --- | --- | --- | --- |
| `newton.linear_solve` | exact | `newton.linear_solve.float64`, shipped with the stage | `solver._cholesky_factorize_solve` | `solve(system)`: the Newton step for `H`, `g` |
| `dof.passive_force` | model changing | the implementation's own, `Impl.reference` | after `passive.passive` | `apply(system)`: force per scalar DOF from `q`, `v`, `time` |
| `control.law` | model changing | the implementation's own | before `forward.fwd_actuation` | `apply(system)`: `ctrl` from actuator `length`, `velocity`, `time` |
| `probe.state` | observer | the implementation's own | after `forward.forward` | `observe(system)`: `value[world, width]` from `qpos`, `qvel`, `qacc`, `time` |

## What the swap class means

* **exact**: same math, a different algorithm. Judged against the stage's reference to a tolerance, and on speed.
* **approximate**: same math to a bounded error. Judged on a speed and error front. (No stage uses it yet.)
* **model changing**: different physics. It cannot be compared with the host, because the host is not the truth; it is
  judged against its own float64 law (the implementation supplies it, and the registry refuses an implementation
  that does not) and against the effect it has on the host (the host's passive force changed by exactly the law;
  the host's actuator force is what MuJoCo (C) computes for that `ctrl`).
* **observer**: reads state, writes only to its own buffer. Judged on matching its own definition, on not touching the
  state, and on cost.

## What `check_impl` verifies for the three non-solver stages

Inputs unchanged; outputs finite; the same inputs give bitwise the same outputs; the law matches its float64 reference
(`max_law_error` <= 1e-5); the host-side effect matches (`max_effect_error`); the runtime was called. Faults are
tested: a wrong law, forgetting the scalar-DOF mask, writing an input, nondeterminism, NaN, an adapter that drops the
effect, a probe that meddles with state.

## Ground truth against MuJoCo (C), MJWarp host

* A spring law matches MuJoCo's own joint spring as closely as MJWarp's native spring does.
* A damping law matches MuJoCo with the same damper applied explicitly to 1e-7; it differs from the model's implicit
  damping by about 0.1% (see `MJWARP_STEP.md`).
* A PD controller matches a MuJoCo loop that sets `ctrl` the same way.
* A probe leaves the trajectory bitwise unchanged; a captured graph replays all three stages.

## Not built, and why

See the table in `MJWARP_STEP.md`: mass-matrix solve, line search, constraint softness, collision, integrators.
