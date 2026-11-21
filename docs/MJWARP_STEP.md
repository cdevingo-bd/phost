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
| `passive.passive` | public function, called once per forward | **built**: `dof.passive_force`, post-hook, model changing |
| `forward.fwd_actuation` | function, called by `forward` and `step2` | **built**: `control.law`, pre-hook, model changing |
| `forward.forward` | function, called by `step` (4 times under RK4) | **built**: `probe.state`, post-hook, observer |
| `smooth.solve_m`, `factor_solve_i` | functions on block-compressed CSR `M` with tile and LDL factors | not built: a contract needs the host to densify `M`; candidate exact stage |
| `solver._linesearch` | private; inputs are per-row quadratic coefficients with branches for friction loss and elliptic cones | not built: the float64 reference must cover those branches; candidate approximate stage |
| constraint softness (`_efc_row`, `_contact_kbimp`) | `wp.func` inlined in per-constraint kernels | not attachable; would need a post-pass over `efc.D`, `efc.aref` and per-row `solref`/`solimp` lookup |
| `collision_driver.collision` | function | attachable, but the contract is a contact list; not built |
| `forward.euler`, `_advance` | per-integrator functions | attachable; Euler integrates joint damping implicitly (below) |

## Things that bit, or will

* `mjw.forward`, `mjw.fwd_actuation` and the other names re-exported by the package are bound at import. A caller that
  uses them directly bypasses a patch; `mjw.step` goes through the modules and does not.
* `passive.passive` rebuilds `d.qfrc_passive` from scratch on every call (and zeroes it when springs and dampers are
  both disabled). A force added afterwards is seen by `fwd_acceleration`. `qfrc_spring` and `qfrc_damper`, which sensors
  read, do not contain it.
* Euler integrates `dof_damping` implicitly. A damping law written into `dof.passive_force` is explicit, and differs
  from the same damping in the model by O(timestep): 0.1% of displacement over 150 steps at 2 ms with c/m = 0.75
  (`tests/test_mjwarp_pointwise.py`). Springs match the model's own to float32 rounding.
* `fwd_actuation` reads a delayed copy of `ctrl` when `nhistory > 0`; `control.law` is refused for such models.
* The discrete integrator builds its own effective-force solve; the two law stages are refused under it.
* The probe runs once per forward call. Under RK4 that is four times per step, and the buffer holds the last.
