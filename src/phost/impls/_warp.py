"""Warp kernels of the built-in implementations. Imported only when a Warp host asks for them."""

import warp as wp


@wp.kernel
def spring_damper(q: wp.array2d[float], v: wp.array2d[float], scalar: wp.array[int], k: float, c: float, q_ref: float,
                  force: wp.array2d[float]):
    w, i = wp.tid()
    if scalar[i] == 1:
        force[w, i] = -k * (q[w, i] - q_ref) - c * v[w, i]
    else:
        force[w, i] = 0.0


@wp.kernel
def spring_damper_implicit(q: wp.array2d[float], v: wp.array2d[float], scalar: wp.array[int], k: float, c: float,
                           q_ref: float, force: wp.array2d[float], damping: wp.array2d[float]):
    w, i = wp.tid()
    if scalar[i] == 1:
        force[w, i] = -k * (q[w, i] - q_ref) - c * v[w, i]
        damping[w, i] = c
    else:
        force[w, i] = 0.0
        damping[w, i] = 0.0


@wp.kernel
def pd(length: wp.array2d[float], velocity: wp.array2d[float], kp: float, kd: float, target: float,
       ctrl: wp.array2d[float]):
    w, i = wp.tid()
    ctrl[w, i] = kp * (target - length[w, i]) - kd * velocity[w, i]


@wp.kernel
def peak(qvel: wp.array2d[float], qacc: wp.array2d[float], value: wp.array2d[float]):
    w = wp.tid()
    vmax = float(0.0)
    amax = float(0.0)
    vsq = float(0.0)
    for i in range(qvel.shape[1]):
        x = qvel[w, i]
        vmax = wp.max(vmax, wp.abs(x))
        vsq += x * x
        amax = wp.max(amax, wp.abs(qacc[w, i]))
    value[w, 0] = vmax
    value[w, 1] = amax
    value[w, 2] = wp.sqrt(vsq)
