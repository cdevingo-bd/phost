"""Warp kernels the MJWarp adapter launches. Imported only when an MJWarp host is built."""

import warp as wp


@wp.kernel(enable_backward=False)
def masks(done: wp.array(dtype=wp.bool), quad: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32),
          skip_unchanged: bool, skip_noflip: bool, active: wp.array(dtype=wp.int32),
          refactor: wp.array(dtype=wp.int32)):
    # MJWarp skips worlds that are done, and (incremental mode) worlds whose constraint states did not flip.
    # H is "changed" unless MJWarp is reusing a factor and no quadratic constraint changed.
    w = wp.tid()
    on = not done[w] and (not skip_noflip or state[w] > 0)
    active[w] = int(on)
    refactor[w] = int(on and (not skip_unchanged or quad[w] > 0))


@wp.kernel(enable_backward=False)
def shadow_done(mask: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32), done: wp.array(dtype=wp.bool)):
    # The host skips a world unless the implementation asked for it AND it is active.
    w = wp.tid()
    done[w] = not (mask[w] != 0 and active[w] != 0)


@wp.kernel(enable_backward=False)
def clear_flags(refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32)):
    w, i = wp.tid()
    if refactor[w] != 0:
        flag[w, i] = 0


@wp.kernel(enable_backward=False)
def mark_jacobian(refactor: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),
                   state: wp.array2d(dtype=wp.int32),
                   rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
                   colind: wp.array3d(dtype=wp.int32), jac: wp.array3d(dtype=wp.float32),
                   flag: wp.array2d(dtype=wp.int32)):
    # With pyramidal cones, J^T D J is a sum over quadratic rows (state 1) of D * outer(J_r, J_r),
    # so a row couples the columns where its Jacobian is nonzero, provided there are at least two
    # (a floor contact along an axis stores explicit zeros and couples nothing). The flags are NOT
    # cleared between the calls of one solve: MJWarp updates H incrementally, adding and
    # subtracting a row's contribution as it changes state, and the subtraction can leave rounding
    # residue on that row's support. Accumulating over the solve therefore covers every entry H can
    # hold; the first call of a solve clears.
    w, r = wp.tid()
    if refactor[w] == 0 or r >= nefc[w]:
        return
    if state[w, r] != 1:
        return
    address = rowadr[w, r]
    count = rownnz[w, r]
    nonzero = int(0)
    for k in range(count):
        if jac[w, 0, address + k] != 0.0:
            nonzero += 1
    if nonzero < 2:
        return
    for k in range(count):
        if jac[w, 0, address + k] != 0.0:
            flag[w, colind[w, 0, address + k]] = 1


@wp.kernel(enable_backward=False)
def mark_mass(refactor: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
               cols: wp.array(dtype=wp.int32), element: wp.array(dtype=wp.int32),
               mass: wp.array2d(dtype=wp.float32), flag: wp.array2d(dtype=wp.int32)):
    # The mass matrix couples two DOFs only where a stored off-diagonal entry is nonzero. A
    # flex vertex's 3x3 block is structurally present but numerically diagonal, so test values.
    w, e = wp.tid()
    if refactor[w] == 0:
        return
    if mass[w, element[e]] != 0.0:
        flag[w, rows[e]] = 1
        flag[w, cols[e]] = 1


@wp.kernel(enable_backward=False)
def effective_assemble(m_rowadr: wp.array(dtype=wp.int32), m_col: wp.array(dtype=wp.int32),
                       m_madr: wp.array(dtype=wp.int32), k_rownnz: wp.array(dtype=wp.int32),
                       k_rowadr: wp.array(dtype=wp.int32), k_colind: wp.array(dtype=wp.int32), nv: int,
                       mass: wp.array2d(dtype=wp.float32), diag: wp.array2d(dtype=wp.float32),
                       kval: wp.array2d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32)):
    # M_hat = M + diag + K, as the host's matrix-vector product applies it: one thread per (world, row) gathers the
    # mass entries, adds the diagonal and scatters the CSR stiffness row. No atomics, so the result is repeatable.
    w, row = wp.tid()
    for j in range(nv):
        out[w, row, j] = 0.0
    for k in range(m_rowadr[row], m_rowadr[row + 1]):
        out[w, row, m_col[k]] += mass[w, m_madr[k]]
    out[w, row, row] += diag[w, row]
    adr = k_rowadr[row]
    for t in range(k_rownnz[row]):
        out[w, row, k_colind[adr + t]] += kval[w, adr + t]


@wp.kernel(enable_backward=False)
def effective_rhs(qfrc_smooth: wp.array2d(dtype=wp.float32), efm_c: wp.array2d(dtype=wp.float32),
                  efm_ca: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    w, i = wp.tid()
    out[w, i] = qfrc_smooth[w, i] + efm_c[w, i] + efm_ca[w, i]


@wp.kernel
def gather_scalar_q(qpos: wp.array2d[float], dof_qposadr: wp.array[int], q: wp.array2d[float]):
    """q[world, dof] = qpos at the DOF's joint when that joint is a scalar (hinge, slide), else 0."""
    w, i = wp.tid()
    adr = dof_qposadr[i]
    if adr >= 0:
        q[w, i] = qpos[w, adr]
    else:
        q[w, i] = 0.0


@wp.kernel
def add_scalar_force(scalar: wp.array[int], force: wp.array2d[float], qfrc_passive: wp.array2d[float]):
    """Add force to the host's passive generalized force on scalar DOFs only."""
    w, i = wp.tid()
    if scalar[i] == 1:
        qfrc_passive[w, i] = qfrc_passive[w, i] + force[w, i]


@wp.kernel(enable_backward=False)
def fold_damping(dof_damping: wp.array2d[float], law: wp.array2d[float], out: wp.array2d[float]):
    # MJWarp's damping arrays are shared across worlds when they have one row: index them the way MJWarp does.
    w, i = wp.tid()
    out[w, i] = dof_damping[w % dof_damping.shape[0], i] + law[w, i]


@wp.kernel(enable_backward=False)
def tick(counter: wp.array[wp.int64]):
    counter[0] = counter[0] + wp.int64(1)


class DeviceMeter:
    """Counts stage executions on the device: the increment is a kernel, so a captured graph repeats it on replay."""

    def __init__(self, device):
        self.device = device
        self._counter = wp.zeros(1, dtype=wp.int64, device=device)

    def tick(self) -> None:
        wp.launch(tick, dim=1, inputs=[self._counter], device=self.device)

    def count(self) -> int:
        return int(self._counter.numpy()[0])          # numpy() synchronizes

    def reset(self) -> None:
        self._counter.zero_()
