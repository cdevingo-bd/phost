# MuJoCo Warp's step, and where phost can attach

Read from the reviewed build (mujoco_warp 3.17.0 source build, f34c6dd5 plus local patches). This is what the
adapter's stages rely on; a different build needs the same reading before its hashes are added.

## The step

`mjw.step(m, d)` is `forward.step`: `forward(m, d)`, then the integrator.

```
forward
  fwd_position (factorize=False)
    kinematics, com_pos, camlight, flex, tendon          fwd_kinematics
    crb, tendon_armature
    collision_driver.collision
    constraint.make_constraint                           builds the efc rows: J, pos, D, aref
    transmission                                         actuator_length, actuator_moment
  sensor_pos, energy
  fwd_velocity
    actuator_velocity, tendon_velocity, com_vel
    passive.passive                                      qfrc_passive (springs, dampers, gravcomp, fluid, adhesion)
    rne, tendon_bias                                     qfrc_bias
  sensor_vel, control callback
  fwd_actuation                                          reads d.ctrl -> actuator_force -> qfrc_actuator
  fwd_acceleration (factorize=True)
    qfrc_smooth, factor_solve_i / solve_m                qacc_smooth = M^-1 qfrc_smooth
                                                         (DISCRETE only: derivative.eff_solve instead, PCG on M_hat)
  solver.solve
    Newton iterations: update_constraint, update_gradient (assembles H),
                       _cholesky_factorize_solve, _linesearch
  rne_postconstraint, sensor_acc
integrator: euler | rungekutta4 (calls forward 4 times) | implicit(fast) | discrete
```

## What can be attached

A boundary is attachable when it is a module-level Python function that its caller reaches through the module
(`passive.passive(m, d)`, or a global of `forward.py`). Replacing the module attribute then changes every call made
by `step`. Anything inside a kernel (a `wp.func` such as `_efc_row`) is not attachable without editing MJWarp.

| boundary | form | status |
| --- | --- | --- |
| `solver._cholesky_factorize_solve` | private function, called per Newton iteration | **built**: `newton.linear_solve`, exact |
| `derivative.eff_solve` | function, called once per forward when the integrator is DISCRETE (the Euler path never calls it) | **built**: `effective.solve`, exact; refused unless the system is exactly mass + diagonal + CSR flex stiffness (see below) |
| `passive.passive` | public function, called once per forward | **built**: `dof.passive_force`, post-hook, model changing |
| `forward.fwd_actuation` | function, called by `forward` and `step2` | **built**: `control.law`, pre-hook, model changing |
| `forward.forward` | function, called by `step` (4 times under RK4) | **built**: `probe.state`, post-hook, observer |
| `forward.xfrc_accumulate` | function imported into `forward`'s namespace, called once per `fwd_acceleration` as `xfrc_accumulate(m, d, d.qfrc_smooth)` | **built**: `body.wrench`, pre-hook, model changing; writes `d.xfrc_applied`, which the call maps through the Jacobians |
| `smooth.solve_m`, `factor_solve_i` | functions on block-compressed CSR `M` with tile and LDL factors | not built: a contract needs the host to densify `M`; candidate exact stage |
| `solver._linesearch` | private; inputs are per-row quadratic coefficients with branches for friction loss and elliptic cones | not built: the float64 reference must cover those branches; candidate approximate stage |
| constraint softness (`_efc_row`, `_contact_kbimp`) | `wp.func` inlined in per-constraint kernels | not attachable; would need a post-pass over `efc.D`, `efc.aref` and per-row `solref`/`solimp` lookup |
| `collision_driver.collision` | function | attachable, but the contract is a contact list; not built |
| `forward.euler`, `_advance` | per-integrator functions | **built**: wrapped as `dof.passive_force`'s `implicit_damping` extension (Euler only); other integrators not built |

## Things that bit, or will

* `mjw.forward`, `mjw.fwd_actuation` and the other names re-exported by the package are bound at import. A caller that
  uses them directly bypasses a patch; `mjw.step` goes through the modules and does not.
* `passive.passive` rebuilds `d.qfrc_passive` from scratch on every call (and zeroes it when springs and dampers are
  both disabled). A force added afterwards is seen by `fwd_acceleration`. `qfrc_spring` and `qfrc_damper`, which sensors
  read, do not contain it.
* Euler integrates `dof_damping` implicitly: `_compute_damping_deriv` reads `m.dof_damping` and `m.dof_dampingpoly`
  (derivative `linear + 2 p0 |v| + 3 p1 v^2`) and `_euler_damp_qfrc` adds `timestep` times it to the diagonal of the
  mass matrix it solves with. A force added to `qfrc_passive` is explicit and differs from the same damping in the model
  by O(timestep): 3.4e-4 in position (0.1% of displacement) over 150 steps at 2 ms with c/m = 0.75. The adapter closes
  the gap for a law that reports `damping`: it hands `euler` a shallow copy of the model whose `dof_damping` is the
  model's plus the reported damping (the user's model is not touched), which gives 1.6e-7 against MuJoCo's own damper.
  The `implicit` and `implicitfast` integrators take damping from `derivative.py`'s own kernels, which this does not
  cover (the advisory says so); RK4 has no implicit damping anywhere, so an explicit law is exact there.
* `fwd_actuation` reads a delayed copy of `ctrl` when `nhistory > 0`; `control.law` is refused for such models.
* The discrete integrator builds its own effective-force solve; the two law stages are refused under it.
* `derivative.eff_solve(m, d, qacc, qfrc=None)` solves `M_hat qacc = qfrc_smooth + efm_c + efm_ca` by PCG. `effective.solve`
  replaces it only when `M_hat` is exactly `M + diag(efm_diag) + K` with `K` in CSR form (`efm_K_*`), assembled from
  `m.M_mulm_*`, `d.M`, `d.efm_diag`, `d.efm_K_val`. Anything else adds a term the assembly would miss, so the adapter
  refuses with the reason: non-DISCRETE integrator, `nefmK == 0`, springs and dampers both disabled, non-simple flex,
  flex passive or contact terms, fluid, an `efm` actuator, tendon stiffness or damping, `efm0_active`, a flex
  interpolation that cannot be assembled, or sleeping. The dispatch is written with the model's arrays only
  (`_eff_rhs`, `_eff_mul_m`, `_eff_mul_diag_and_csr` and `eff_solve` are all hashed).
* The native PCG's cost depends on the system's conditioning, which depends on the time step and stiffness. It can be
  well under a millisecond (the lab's 12x12 membrane at 0.5 ms), where no direct solve can win, or several (a stiffer
  sheet at a larger step).
* The probe runs once per forward call. Under RK4 that is four times per step, and the buffer holds the last.
