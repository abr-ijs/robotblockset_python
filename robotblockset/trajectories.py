import numpy as np
import math
import quaternionic as Quaternion
import scipy.interpolate as spi
import matplotlib.pyplot as plt

from robotblockset.tools import isscalar, vector, isvector, ismatrix, ismatrixarray, check_option, normalize, vecnormalize, gradientPath, gradientCartesianPath
from robotblockset.transformations import rbs_type, ang4v, rot_v, q2r, r2q, t2x, x2t, prpy2x, xerr, qerr, qlog, qmtimes, qinv, qexp, xnormalize, uniqueCartesianPath
from robotblockset.RBF import encodeRBF, decodeRBF, decodeCartesianRBF
from robotblockset.graphics import plotcpath


_eps = 100 * np.finfo(np.float64).eps


def arc(p0, p1, pC, s, short: bool = True):
    """Points on arc defined by two points and center point

    Arc center at pC, arc is starting in p0=p(0) and ending
    at p1=s(1). If distance from pC to p0 and p1 is not equal,
    pC is projected to point on midline between p0 and p1

    If s<0 long path is used for arc

    Parameters
    ----------
    p0 : array of floats
        inital arc point (3,)
    p1 : array of floats
        final arc point (3,)
    pC : array of floats
        arc center point (3,)
    s : float
        normalized arc distance [0..1]
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    p : array of floats
        points on ark (n, 3) or (3,)

    Raises
    ------
    ValueError
        Points are not distinct
    """
    p0 = vector(p0, dim=3)
    p1 = vector(p1, dim=3)
    pC = vector(pC, dim=3)
    v1 = p0 - pC
    v2 = p1 - pC
    v1n = np.linalg.norm(v1)
    v2n = np.linalg.norm(v2)
    phi = ang4v(v1, v2)
    if v1n > 0 and v2n > 0 and phi > 0.001 and phi < (np.pi - 0.001):
        v3 = normalize(np.cross(v1, v2))
        v4 = normalize(np.cross(p1 - p0, v3))
        p01 = (p0 + p1) / 2
        pCx = np.dot(pC - p01, v4) * v4 + p01
        v1 = p0 - pCx
        v2 = p1 - pCx
        phi = ang4v(v1, v2)
        if s < 0:
            short = False
        if not short:
            phi = 2 * np.pi - phi
            s = -np.abs(s)
        return rot_v(v3, s * phi, out="R") @ v1 + pCx
    else:
        raise ValueError("Points must be distinct and not on the same line")


def carctraj(x0, x1, pC, t, traj="poly", short: bool = True):
    """Cartesian trajectory on arc form x0 to x1

    Arc is defined by initial and final pose and arc center position.

    Parameters
    ----------
    x0 : array of floats
        initial Cartesian pose (7,)
    x1 : array of floats
        final Cartesian pose (7,)
    pC : array of floats
        arc center position (3,)
    t : array of floats
        trajectory time  (nsamp,)
    traj : str
        trajectory type (`poly`, `trap` or `line`)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    xt : array of floats
        Cartesian trajectory - pose (nsamp,7)
    xdt : array of floats
        Cartesian trajectory - velocizy (nsamp,7)
    xddt : array of floats
        Cartesian trajectory - acceleration (nsamp,7)
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    pC = vector(pC, dim=3)
    if t[-1] < 0:
        t = np.abs(t)
        short = False
    s, _, _ = jtraj(0.0, 1.0, t, traj=traj)
    xt = xarcinterp(x0, x1, pC, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def jline(q0, q1, t):
    """Trajectory form q0 to q1 with constant velocity

    Parameters
    ----------
    q0 : array of floats
        initial joint positions (n,)
    q1 : array of floats
        final joint position (n,)
    t : array of floats
        trajectory time (nsamp,)

    Returns
    -------
    qt : array of floats
        interpolated joint position (nsamp, n)
    qdt : array of floats
        interpolated joint velocities  (nsamp, n)
    qddt : array of floats
        interpolated joint accelerations  (nsamp, n)

    Raises
    ------
    TypeError
        Wrong input verctor size
    """
    q0 = vector(q0)
    q1 = vector(q1)
    t = vector(t)
    if q0.size == q1.size:
        t = t - t[0]
        tmax = np.max(t)
        _t = t / tmax
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")
        if q0.size == 1:
            qt = q0 + (q1 - q0) * _t
        else:
            qt = q0 + np.einsum("i,j->ji", q1 - q0, _t)
        dq = (q1 - q0) / tmax
        qdt = np.ones(qt.shape) * dq
        return qt, qdt, np.zeros(qt.shape)
    else:
        raise TypeError("Input vecotrs must be same size")


def jtrap(q0, q1, t, ta=0.1):
    """Trajectory form q0 to q1 with trapezoidal velocity

    Parameters
    ----------
    q0 : array of floats
        initial joint positions (n,)
    q1 : array of floats
        final joint position (n,)
    t : array of floats
        trajectory time (nsamp,)
    ta : float, optional
        acceleration/deceleration time

    Returns
    -------
    qt : array of floats
        interpolated joint position (nsamp, n)
    qdt : array of floats
        interpolated joint velocities  (nsamp, n)
    qddt : array of floats
        interpolated joint accelerations  (nsamp, n)

    Raises
    ------
    TypeError
        Wrong input verctor size
    """
    q0 = vector(q0)
    q1 = vector(q1)
    t = vector(t)
    if q0.size == q1.size:
        t = t - t[0]
        tmax = np.max(t)
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")
        acc = 1 / (ta * (tmax - ta))
        s = lambda t: (t <= ta) * 0.5 * acc * t**2 + (t > ta and t <= (tmax - ta)) * (0.5 * acc * ta**2 + acc * ta * (t - ta)) + (t > (tmax - ta)) * (1 - 0.5 * acc * (tmax - t) ** 2)
        v = lambda t: (t <= ta) * acc * t + (t > ta and t <= (tmax - ta)) * acc * ta + (t > (tmax - ta)) * acc * (tmax - t)
        a = lambda t: (t <= ta) * acc - (t > (tmax - ta)) * acc
        st = np.array([s(x) for x in t])
        vt = np.array([v(x) for x in t])
        at = np.array([a(x) for x in t])
        if q0.size == 1:
            qt = q0 + (q1 - q0) * st
            qdt = q0 + (q1 - q0) * vt
            qddt = q0 + (q1 - q0) * at
        else:
            qt = q0 + np.einsum("i,j->ji", q1 - q0, st)
            qdt = np.einsum("i,j->ji", q1 - q0, vt)
            qddt = np.einsum("i,j->ji", q1 - q0, at)
        return qt, qdt, qddt
    else:
        raise TypeError("Input vecotrs must be same size")


def jpoly(q0, q1, t, qd0=None, qd1=None):
    """Trajectory form q0 to q1 using 5th order polynomial

    Parameters
    ----------
    q0 : array of floats
        initial joint positions (n,)
    q1 : array of floats
        final joint position (n,)
    t : array of floats
        trajectory time (nsamp,)
    qd0 : array of floats
        Initial joint velocities (n,)
    qd1 : array of floats
        Final joint velocities (n,)

    Returns
    -------
    qt : array of floats
        interpolated joint position (nsamp, n)
    qdt : array of floats
        interpolated joint velocities  (nsamp, n)
    qddt : array of floats
        interpolated joint accelerations  (nsamp, n)

    Raises
    ------
    TypeError
        Wrong input verctor size
    """
    q0 = vector(q0)
    q1 = vector(q1)
    t = vector(t)
    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = vector(qd0)
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd1 = vector(qd1)
    if q0.size == q1.size and qd0.size == q0.size and qd1.size == q1.size:
        tmax = max(t)
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")
        t = np.copy(vector(t).T) / tmax

        A = 6 * (q1 - q0) - 3 * (qd1 + qd0) * tmax
        B = -15 * (q1 - q0) + (8 * qd0 + 7 * qd1) * tmax
        C = 10 * (q1 - q0) - (6 * qd0 + 4 * qd1) * tmax
        E = qd0 * tmax
        F = q0

        tt = np.array([t**5, t**4, t**3, t**2, t, np.ones(t.shape)])
        s = np.array([A, B, C, np.zeros(A.shape), E, F]).reshape((6, q0.size))
        v = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E]).reshape((6, q0.size)) / tmax
        a = (
            np.array(
                [
                    np.zeros(A.shape),
                    np.zeros(A.shape),
                    20 * A,
                    12 * B,
                    6 * C,
                    np.zeros(A.shape),
                ]
            ).reshape((6, q0.size))
            / tmax**2
        )
        qt = np.einsum("ij,ik->kj", s, tt)
        qdt = np.einsum("ij,ik->kj", v, tt)
        qddt = np.einsum("ij,ik->kj", a, tt)
        if q0.size == 1:
            return qt.flatten(), qdt.flatten(), qddt.flatten()
        else:
            return qt, qdt, qddt
    else:
        raise TypeError("Input vecotrs must be same size")


def jtraj(q0, q1, t, traj="poly", qd0=None, qd1=None):
    """Trajectory form q0 to q1

    Parameters
    ----------
    q0 : array of floats
        initial joint positions (n,)
    q1 : array of floats
        final joint position (n,)
    t : array of floats
        trajectory time (nsamp,)
    typ : str
        trajectory type (`poly`, `trap` or `line`)
    qd0 : array of floats
        Initial joint velocities (n,)
    qd1 : array of floats
        Final joint velocities (n,)

    Returns
    -------
    qt : array of floats
        interpolated joint position (nsamp, n)
    qdt : array of floats
        interpolated joint velocities  (nsamp, n)
    qddt : array of floats
        interpolated joint accelerations  (nsamp, n)

    Raises
    ------
    ValueError
        Wrong trajectory type
    """
    q0 = vector(q0)
    q1 = vector(q1)
    if check_option(traj, "poly"):
        _traj = jpoly
    elif check_option(traj, "trap"):
        _traj = jtrap
    elif check_option(traj, "line"):
        _traj = jline
    else:
        raise ValueError(f"Trajectory type {traj} not supported")
    return _traj(q0, q1, t)


def cline(x0, x1, t, short=True):
    """Cartesian trajectory form x0 to x1 with constant velocity

    Initial and final pose are defined by position and quaternion.

    Parameters
    ----------
    x0 : array of floats
        initial Cartesian pose (7,)
    x1 : array of floats
        final Cartesian pose (7,)
    t : array of floats
        trajectory time  (nsamp,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    xt : array of floats
        Cartesian trajectory - pose (nsamp,7)
    xdt : array of floats
        Cartesian trajectory - velocizy (nsamp,7)
    xddt : array of floats
        Cartesian trajectory - acceleration (nsamp,7)
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jline(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def ctrap(x0, x1, t, short=True):
    """Cartesian trajectory form x0 to x1 with trapezoidal velocity

    Initial and final pose are defined by position and quaternion.

    Parameters
    ----------
    x0 : array of floats
        initial Cartesian pose (7,)
    x1 : array of floats
        final Cartesian pose (7,)
    t : array of floats
        trajectory time  (nsamp,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    xt : array of floats
        Cartesian trajectory - pose (nsamp,7)
    xdt : array of floats
        Cartesian trajectory - velocizy (nsamp,7)
    xddt : array of floats
        Cartesian trajectory - acceleration (nsamp,7)
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jtrap(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def cpoly(x0, x1, t, short=True):
    """Cartesian trajectory form x0 to x1 using 5th order polynomial

    Initial and final pose are defined by position and quaternion.

    Parameters
    ----------
    x0 : array of floats
        initial Cartesian pose (7,)
    x1 : array of floats
        final Cartesian pose (7,)
    t : array of floats
        trajectory time  (nsamp,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    xt : array of floats
        Cartesian trajectory - pose (nsamp,7)
    xdt : array of floats
        Cartesian trajectory - velocizy (nsamp,7)
    xddt : array of floats
        Cartesian trajectory - acceleration (nsamp,7)
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jpoly(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def ctraj(x0, x1, t, traj="poly", short=True):
    """Cartesian trajectory form x0 to x1

    Initial and final pose are defined by position and quaternion.

    Parameters
    ----------
    x0 : array of floats
        initial Cartesian pose (7,)
    x1 : array of floats
        final Cartesian pose (7,)
    t : array of floats
        trajectory time  (nsamp,)
    traj : str
        trajectory type (`poly`, `trap` or `line`)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    xt : array of floats
        Cartesian trajectory - pose (nsamp,7)
    xdt : array of floats
        Cartesian trajectory - velocizy (nsamp,7)
    xddt : array of floats
        Cartesian trajectory - acceleration (nsamp,7)
    Raises
    ------
    ValueError
        Wrong trajectory type
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    if check_option(traj, "poly"):
        _traj = cpoly
    elif check_option(traj, "trap"):
        _traj = ctrap
    elif check_option(traj, "line"):
        _traj = cline
    else:
        raise ValueError(f"Trajectory type {traj} not supported")
    return _traj(x0, x1, t, short=short)


def interp(y1, y2, s):
    """Multidimensional linear interpolation

    Returns linear interpolated data points between y1 and y2 at s

    Parameters
    ----------
    y1 : array of floats
        initial data points (n,)
    y2 : array of floats
        final data points (n,)
    s : array of floats
        query data points (ns,)

    Returns
    -------
    array of floats
        interpolated data points (ns, n)
    """
    y1 = vector(y1)
    y2 = vector(y2)
    s = vector(s)
    if y1.size == y2.size:
        if y1.size == 1:
            return y1 + (y2 - y1) * s
        else:
            return y1 + np.einsum("i,j->ji", y2 - y1, s)
    else:
        raise TypeError("Input vecotrs must be same size")


def slerp(Q1, Q2, s, short=True):
    """Spherical linear interpolation of unit quaternion like arrays

    Returns linear interpolated data points between Q1 and Q2
    using SLERP

    Parameters
    ----------
    Q1 : array of floats
        initial quaternion (4,)
    Q2 : array of floats
        final quaternion (4,)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray of quaternions
        interpolated quaternion elements (ns, 4)
    """
    Q1 = vector(Q1, dim=4)
    Q2 = vector(Q2, dim=4)
    s = np.asarray(s, dtype="float").flatten()

    qq = np.clip(np.dot(Q1, Q2), -1.0, 1.0)
    if short:
        if qq < 0:
            Q2 = -Q2  # pylint: disable=invalid-unary-operand-type
            qq = -qq  # pylint: disable=invalid-unary-operand-type
    else:
        if qq > 0:
            Q2 = -Q2  # pylint: disable=invalid-unary-operand-type
            qq = -qq  # pylint: disable=invalid-unary-operand-type
    phi = np.arccos(qq)
    sinphi = np.sin(phi)
    n = s.size
    Q = np.empty((n, 4))
    for i in range(n):
        ss = s[i]
        if ss == 0:
            Q[i] = Q1
        elif ss == 1:
            Q[i] = Q2
        else:
            if abs(phi) < _eps:
                Q[i] = Q1
            else:
                Q[i] = (np.sin((1 - ss) * phi) * Q1 + np.sin(ss * phi) * Q2) / sinphi
    return Q


def qspline(Q, s, mode="squad"):
    """
    Spline interpolation of N quaternions in the spherical space of SO(3).

    Parameters
    ----------
    q: array of floats
        Quaternion array of shape (n, 4).
    s: array of floasts
        Path parameters [0..1] as a numpy array (m, ).
    mode: str
        Mode of spline interpolation ('hermite_cubic' or 'squad').

    Returns:
    array of floats
        Interpolated quaternions as a numpy array of shape (m, 4).
    """

    def _get_intermediate_control_point(j, q, dir_flip):
        L = q.shape[1]

        if j == 0:
            qa = q[0]
        elif j == L - 1:
            qa = q[L - 1]
        else:
            qji = qinv(q[j])
            qiqm1 = qmtimes(qji, q[j - 1])
            qiqp1 = qmtimes(qji, q[j + 1])
            ang_vel = -((qlog(qiqp1) + qlog(qiqm1)) / 4)

            if dir_flip:
                qa = qmtimes(q[j], qinv(qexp(ang_vel)))
            else:
                qa = qmtimes(q[j], qexp(ang_vel))

        return qa

    def _eval_cumulative_berstein_basis(s, i, order):
        N = order
        beta = 0

        for j in range(i, N + 1):
            term1 = math.comb(N, j)
            term2 = (1 - s) ** (N - j)
            term3 = s**j
            beta += term1 * term2 * term3

        return beta

    def _eval_alpha(s, i, L):
        k = s * (L - 1) + 1

        if i < k:
            alpha = 1
        elif i > k and i < k + 1:
            alpha = k - (i - 1)
        else:
            alpha = 0

        return alpha

    q = np.asarray(Q, dtype="float")
    if q.shape[1] != 4:
        raise ValueError("Quaternion vector 'q' must have 4 columns.")

    if not np.all((s >= 0) & (s <= 1)) or not np.all(np.diff(s) >= 0):
        raise ValueError("Path parameters 's' must be in the range [0, 1] and in increasing order.")

    n = q.shape[0]
    m = len(s)
    order = 3

    for j in range(1, n):
        C = np.dot(q[j - 1], q[j])
        if C < 0:
            q[j] = -q[j]

    qout = np.empty((m, 4))

    for i in range(m):
        si = s[i]
        qout[i] = q[0]

        if si != 0 and si != 1:
            val = q[0]
            EPS = 1e-9

            for j in range(1, n):
                alpha = _eval_alpha(si, j + 1, n)
                t = alpha
                if alpha > 0:
                    C = np.dot(q[j - 1], q[j])

                    if np.abs(1 - C) <= EPS:
                        val = (1 - si) * q[j - 1] + si * q[j]
                        val = qnormalize(val)
                    elif np.abs(1 + C) <= EPS:
                        qtemp = np.array([q[j, 3], -q[j, 2], q[j, 1], -q[j, 0]])
                        qtemp_array = np.copy(q)
                        qtemp_array[j] = qtemp

                        if mode == "hermite_cubic":
                            qi = qinv(q[j - 1])
                            qa = _get_intermediate_control_point(j - 1, qtemp_array, 0)
                            qap1 = _get_intermediate_control_point(j, qtemp_array, 1)
                            qai = qinv(qa)
                            qap1i = qinv(qap1)
                            qiqa = qmtimes(qi, qa)
                            qaiqap1 = qmtimes(qai, qap1)
                            qap1iqp1 = qmtimes(qap1i, q[j])
                            omega1 = qlog(qiqa)
                            omega2 = qlog(qaiqap1)
                            omega3 = qlog(qap1iqp1)
                            beta1 = _eval_cumulative_berstein_basis(t, 1, order)
                            beta2 = _eval_cumulative_berstein_basis(t, 2, order)
                            beta3 = _eval_cumulative_berstein_basis(t, 3, order)
                            val = qmtimes(q[j - 1], qexp(omega1 * beta1))
                            val = qmtimes(val, qexp(omega2 * beta2))
                            val = qmtimes(val, qexp(omega3 * beta3))
                        elif mode == "squad":
                            qa = _get_intermediate_control_point(j - 1, qtemp_array, 0)
                            qap1 = _get_intermediate_control_point(j, qtemp_array, 0)
                            qtemp1 = slerp(q[j - 1], qtemp, t)
                            qtemp2 = slerp(qa, qap1, t)
                            squad = slerp(qtemp1, qtemp2, 2 * t * (1 - t))
                            val = squad

                    else:
                        if mode == "hermite_cubic":
                            qi = qinv(q[j - 1])
                            qa = _get_intermediate_control_point(j - 1, q, 0)
                            qap1 = _get_intermediate_control_point(j, q, 1)
                            qai = qinv(qa)
                            qap1i = qinv(qap1)
                            qiqa = qmtimes(qi, qa)
                            qaiqap1 = qmtimes(qai, qap1)
                            qap1iqp1 = qmtimes(qap1i, q[j])
                            omega1 = qlog(qiqa)
                            omega2 = qlog(qaiqap1)
                            omega3 = qlog(qap1iqp1)
                            beta1 = _eval_cumulative_berstein_basis(t, 1, order)
                            beta2 = _eval_cumulative_berstein_basis(t, 2, order)
                            beta3 = _eval_cumulative_berstein_basis(t, 3, order)
                            val = qmtimes(q[j - 1], qexp(omega1 * beta1))
                            val = qmtimes(val, qexp(omega2 * beta2))
                            val = qmtimes(val, qexp(omega3 * beta3))
                            val = qnormalize(val)
                        elif mode == "squad":
                            qa = _get_intermediate_control_point(j - 1, q, 0)
                            qap1 = _get_intermediate_control_point(j, q, 0)
                            qtemp1 = slerp(q[j - 1], q[j], t)
                            qtemp2 = slerp(qa, qap1, t)
                            squad = slerp(qtemp1, qtemp2, 2 * t * (1 - t))
                            val = squad

                qout[i] = qnormalize(val)

    return qout


def qinterp(Q1, Q2, s, short=True):
    """Spherical linear interpolation of unit quaternion like arrays

    Returns linear interpolated data points between Q1 and Q2
    using SLERP

    Parameters
    ----------
    Q1 : array of floats
        initial quaternion (4,)
    Q2 : array of floats
        final quaternion (4,)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray of quaternions
        interpolated quaternion elements (ns, 4)
    """
    return slerp(Q1, Q2, s, short=short)


def rinterp(R1, R2, s, short=True):
    """Spherical linear interpolation of rotational matrices

    Returns linear interpolated data points between R1 and R2
    using SLERP

    Parameters
    ----------
    R1 : array of floats
        initial quaternion (3, 3)
    R2 : array of floats
        final quaternion (3, 3)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray of quaternions
        interpolated quaternion elements (ns, 3, 3)
    """
    return q2r(slerp(r2q(R1), r2q(R2), s, short=short))


def xinterp(x1, x2, s, short=True):
    """Linear interpolation of spatial poses (SE3)

    Returns linear interpolated data points between x1 and x2
    using LERP for positions and SLERP for rotations

    Spatial poses are represented as array of 3 positions and
    4 quaternion elements

    Parameters
    ----------
    x1 : array of floats
        initial Cartesian pose (7,)
    x2 : array of floats
        final Cartesian pose (7,)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray
        interpolated Cartesian poses (ns, 7)
    """
    x1 = vector(x1, dim=7)
    x2 = vector(x2, dim=7)
    s = vector(s)
    p = interp(x1[:3], x2[:3], s)
    Q = qinterp(x1[3:], x2[3:], s, short=short)
    return np.hstack((p, Q))


def tinterp(T1, T2, s, short=True):
    """Linear interpolation of spatial poses (SE3)

    Returns linear interpolated data points between T1 and T2
    using LERP for positions and SLERP for rotations

    Spatial poses are represented as homogenous matrices

    Parameters
    ----------
    T1 : array of floats
        initial Cartesian pose (4, 4)
    T2 : array of floats
        final Cartesian pose (4, 4)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray
        interpolated Cartesian poses (ns, 4, 4)
    """
    x1 = t2x(T1)
    x2 = t2x(T2)
    return x2t(xinterp(t2x(T1), t2x(T2), s, short=short))


def xarcinterp(x1, x2, pC, s, short: bool = True):
    """Linear interpolation of spatial poses (SE3) along arc

    Returns linear interpolated data points between x1 and x2
    using LERP for positions on arc and SLERP for rotations

    Spatial poses are represented as array of 3 positions and
    4 quaternion elements

    Parameters
    ----------
    x1 : array of floats
        initial Cartesian pose (7,)
    x2 : array of floats
        final Cartesian pose (7,)
    pC : array of floats
        arc center position (3,)
    s : array of floats
        query data points (ns,)
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    ndarray
        interpolated Cartesian poses (ns, 7)
    """
    x1 = vector(x1, dim=7)
    x2 = vector(x2, dim=7)
    pC = vector(pC, dim=3)
    s = vector(s)
    p = [arc(x1[:3], x2[:3], pC, ss, short=short) for ss in s]
    Q = qinterp(x1[3:], x2[3:], s, short=short)
    return np.hstack((p, Q))


def interp1(s, y, si):
    """Wrapper for SciPy interp1d

    Parameters
    ----------
    s : array of floats
        query points (ns,)
    y : array of floats
        data points (ns, n)
    si : array of floats
        quary points (ni,)

    Returns
    -------
    array of floats
        data in query points (ni, n)
    """
    f = spi.interp1d(s, y, axis=0, fill_value="extrapolate")
    return f(si)


def interpPath(s, path, squery):
    """Interpolate path for query path values

    Parameters
    ----------
    s : array of floats
        path parameter (ns,)
    path : array of floats
        path data (ns, n)
    squery : array of floats
        query path points (ni, )

    Returns
    -------
    array of floats
        path values at query points (ni, n)
    """
    s = vector(s)
    path = np.array(path)
    if (ismatrix(path) and path.shape[0] == len(s)) or isvector(path, dim=len(s)):
        return interp1(s, path, squery)
    else:
        raise TypeError(f"s and path must have same first dimension, but have shapes {s.shape} and {path.shape}")


def interpQuaternionPath(s, path, squery, short=True):
    """Interpolate quaternion path for query path values

    Returns sequentially linear interpolated data points between path points
    using SLERP.

    Parameters
    ----------
    s : array of floats
        path parameter (ns,)
    path : array of quaternions
        path quaternions (ns, 4)
    squery : array of floats
        query path points (ni, )
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    array of quaternions
        path quaternions at query points (ni, 4)
    """
    s = vector(s)
    path = np.array(path)
    if not ismatrix(path, shape=(len(s), 4)):
        raise TypeError(f"path must have dimension {(len(s), 4)}, but has {path.shape}")
    n = len(s)
    m = len(squery)
    i1 = np.clip(np.floor(interp1(s, range(n), squery)), 0, n - 2).astype(int)
    i2 = np.clip(i1 + 1, 0, n - 1).astype(int)
    ss = (squery - s[i1]) / (s[i2] - s[i1])
    newpath = np.empty(shape=(m, 4))
    for i in range(m):
        xx = qinterp(path[i1[i], :], path[i2[i], :], ss[i], short=short)
        newpath[i, :] = xx
    return newpath


def interpCartesianPath(s, path, squery, short=True):
    """Interpolate Cartesian path for query path values

    Returns sequentially linear interpolated data points between path points
    using LERP for positions and SLERP for rotations.

    Spatial poses are represented as array of 3 positions and
    4 quaternion elements

    Parameters
    ----------
    s : array of floats
        path parameter (ns,)
    path : array of floats
        Cartesian path poses (ns, 7)
    squery : array of floats
        query path points (ni, )
    short : bool, optional
        if true, shortest rotation is taken

    Returns
    -------
    array of floats
        Cartesian path poses at query points (ni, 7)
    """
    s = vector(s)
    path = np.array(path)
    if not ismatrix(path, shape=(len(s), 7)):
        raise TypeError(f"Qpath must have dimension {(len(s), 7)}, but has {path.shape}")
    p = interpPath(s, path[:, :3], squery)
    Q = interpQuaternionPath(s, path[:, 3:], squery, short=short)
    return np.hstack((p, Q))


def pathauxpoints(pnt: np.ndarray, auxpoints="relative", auxdistance=[0.1, 0.1], viapoints=False) -> np.ndarray:
    """Generates auxiliary points for path points

    Parameters
    ----------
    pnt : array of floats
        waypoints for path (n, 3) or (n, 7) or (n, 6)
    auxpoints : str, optional
        auxiliary points (default "none"):
            'absolute'  absolute distance
            'relative'  relative distance of path segment
    auxdistance : array of floats, optional
        distance of auxiliary points (2, ), by default [0.1, 0.1]
    viapoints : bool, optional
        include viapoints when auxpoints are used, by default False

    Returns
    -------
    array of floats
        path vith auxilary points (m, 7) or (m, 3)

    Raises
    ------
    ValueError
        Wrong parameters values or shape
    """
    if not auxpoints in ["absolute", "relative", "none"]:
        raise ValueError("Invalid selection of auxilary points")

    if auxpoints == "none":
        return pnt

    auxdistance = np.asarray(auxdistance, dtype="float").flatten()
    if auxpoints == "relative":
        if any(auxdistance > 0.5) or any(auxdistance < 0):
            raise ValueError("auxdistance should be in the range [0, 0.5] for 'relative' mode.")
    else:
        if any(auxdistance < 0):
            raise ValueError("auxdistance should be positive.")

    if isscalar(auxdistance):
        auxdistance = [auxdistance, auxdistance]

    npts, nd = pnt.shape
    nptsaux = (npts - 2) * 3 + 2
    yy = np.zeros((nptsaux, nd))
    yy[0, :] = pnt[0, :]

    for i in range(1, npts - 1):
        j = i * 3 - 1

        if auxpoints == "absolute":
            if auxdistance[0] > (np.linalg.norm(pnt[i - 1, :3] - pnt[i, :3]) / 2):
                print(f"Auxiliary point too far from basic point {i}. Moved to the middle of the segment.")
                yy[j - 1, :3] = pnt[i, :3] + (pnt[i - 1, :3] - pnt[i, :3]) / 2
            else:
                yy[j - 1, :3] = pnt[i, :3] + (pnt[i - 1, :3] - pnt[i, :3]) / np.linalg.norm(pnt[i - 1, :3] - pnt[i, :3]) * auxdistance[0]

            yy[j, :3] = pnt[i, :3]

            if auxdistance[0] > (np.linalg.norm(pnt[i + 1, :3] - pnt[i, :3]) / 2):
                print(f"Auxiliary point too far from basic point {i}. Moved to the middle of the segment.")
                yy[j + 1, :3] = pnt[i, :3] + (pnt[i + 1, :3] - pnt[i, :3]) / 2
            else:
                yy[j + 1, :3] = pnt[i, :3] + (pnt[i + 1, :3] - pnt[i, :3]) / np.linalg.norm(pnt[i + 1, :3] - pnt[i, :3]) * auxdistance[0]
        else:
            yy[j - 1, :3] = pnt[i, :3] + (pnt[i - 1, :3] - pnt[i, :3]) * auxdistance[0]
            yy[j, :3] = pnt[i, :3]
            yy[j + 1, :3] = pnt[i, :3] + (pnt[i + 1, :3] - pnt[i, :3]) * auxdistance[0]

        if nd == 7:
            if auxdistance[1] > 0:
                if auxpoints == "absolute":
                    qen = np.linalg.norm(qerr(pnt[i - 1, 3:], pnt[i, 3:]))
                    if auxdistance[1] > qen / 2:
                        print(f"Auxiliary point too far from basic point {i}. Moved to the middle of the segment.")
                        yy[j - 1, 3:] = slerp(pnt[i, 3:], pnt[i - 1, 3:], 0.5)
                    else:
                        yy[j - 1, 3:] = slerp(pnt[i, 3:], pnt[i - 1, 3:], auxdistance[1] / qen)

                    yy[j, 3:] = pnt[i, 3:]

                    qen = np.linalg.norm(qerr(pnt[i + 1, 3:], pnt[i, 3:]))
                    if auxdistance[1] > qen / 2:
                        print(f"Auxiliary point too far from basic point {i}. Moved to the middle of the segment.")
                        yy[j + 1, 3:] = slerp(pnt[i, 3:], pnt[i + 1, 3:], 0.5)
                    else:
                        yy[j + 1, 3:] = slerp(pnt[i, 3:], pnt[i + 1, 3:], auxdistance[1] / qen)
                else:
                    yy[j - 1, 3:] = slerp(pnt[i, 3:], pnt[i - 1, 3:], auxdistance[1])
                    yy[j, 3:] = pnt[i, 3:]
                    yy[j + 1, 3:] = slerp(pnt[i, 3:], pnt[i + 1, 3:], auxdistance[1])

            else:
                yy[j - 1, 3:] = pnt[i - 1, 3:]
                yy[j, 3:] = pnt[i, 3:]
                yy[j + 1, 3:] = pnt[i, 3:]

    yy[-1, :] = pnt[-1, :]

    if viapoints:
        auxpnt = yy
    else:
        npoints = yy.shape[0]
        indices = np.setdiff1d(np.arange(npoints), np.arange(2, npoints, 3))
        auxpnt = yy[indices, :]

    return auxpnt


def pathoverpoints(
    pnt: np.ndarray,
    interp="inner",
    order=4,
    step=0.02,
    n_points=0,
    auxpoints="none",
    auxdistance=[0.1, 0.1],
    viapoints=False,
    natural=False,
    normscale=[1, 1],
    plot=False,
    ori_sel=(1, 2),
) -> np.ndarray:
    """Generates path over points using spline interpolation

    Parameters
    ----------
    pnt : array of floats
        waypoints for path (n, 3) or (n, 7) or (n, 6)
    interp : str, optional
        Interpolation type: (default "inner")
            "inner"     spline by uniform subdivision
            "spline"    cubic spline curve
            'RBF'       interpolation using RBF
            "none"      no interpolation,
    order : int, optional
        order of inner spline, by default 4
    step : float, optional
        maximal difference in path parameter, by default 0.02
    n_points : int, optional
        minimal number of path points - if 0 then "step" is used, by default 0
    auxpoints : str, optional (used only for "inner")
        auxiliary points (default "none"):
            'absolute'  absolute distance
            'relative'  relative distance of path segment
    auxdistance : array of floats, optional
        distance of auxiliary points (2, ), by default [0.1, 0.1]
    viapoints : bool, optional
        include viapoints when auxpoints are used, by default False
    natural : bool, optional
        make path parameter natural, by default False
    normscale : list of int, optional
        scaling factor for rotation norm, by default [1, 1]
    plot : bool, optional
        plot generated path, by default False
    ori_sel : list, optional
        selection of quaternion angles for 2D plot, by default (1, 2)

    Returns
    -------
    array of floats
        interpolated path (m, 7) or (m, 3)
    array of floats
        path parameter (m,)

    Raises
    ------
    ValueError
        Wrong parameters values or shape
    """

    def _spcrv(x, k=4, maxpnt=None):
        y = x.copy()
        kntstp = 1
        if k is None:
            k = 4
        n, d = y.shape

        if n < k:
            raise ValueError(f"Too few points ({n}) for the specified order ({k}).")
        elif k < 2:
            raise ValueError(f"Order ({k}) is too small; it must be at least 2.")
        else:
            if k > 2:
                if maxpnt is None:
                    maxpnt = 100

                while n < maxpnt:
                    kntstp = 2 * kntstp
                    m = 2 * n
                    yy = np.zeros((m, d))
                    yy[1:m:2, :] = y
                    yy[0:m:2, :] = y

                    for r in range(2, k + 1):
                        yy[1:m, :] = (yy[1:m, :] + yy[: m - 1, :]) * 0.5

                    y = yy[k - 1 : m, :].copy()
                    n = m + 1 - k

            return y

    if order < 2:
        raise ValueError("Order must be >= 2")
    auxdistance = np.asarray(auxdistance, dtype="float").flatten()
    if not all(0 <= val <= 0.5 for val in auxdistance):
        raise ValueError("auxdistance values out of range")
    if not interp in ["inner", "spline", "RBF", "none"]:
        raise ValueError("Invalid selected interpolation")
    if not auxpoints in ["absolute", "relative", "none"]:
        raise ValueError("Invalid selection of auxilary points")

    pnt = rbs_type(pnt)
    if len(pnt.shape) == 3:
        if ismatrixarray(pnt, shape=(4, 4)):
            _xx = uniqueCartesianPath(t2x(pnt))
        else:
            raise ValueError("Input 'pnt' must be a (..., 4, 4) array")
    elif ismatrix(pnt):
        nd = pnt.shape[1]
        if not (nd == 3 or nd == 6 or nd == 7):
            raise ValueError("Wrong input points dimension")
        if nd == 6:
            _xx = prpy2x(pnt)
        else:
            _xx = pnt
    else:
        raise ValueError("Input 'pnt' must be a (..., 3) or (..., 6) or (..., 7) array")

    points = _xx
    xaux = pathauxpoints(_xx, auxpoints=auxpoints, auxdistance=auxdistance, viapoints=viapoints)
    npoints, nd = xaux.shape

    if natural:
        kpoints = 4
    else:
        kpoints = 1

    # Inner spline interpolation
    if interp == "inner":
        ya = np.vstack(
            (
                (
                    np.repeat([xaux[0, :]], order - 2, axis=0),
                    xaux,
                    np.repeat([xaux[-1, :]], order - 2, axis=0),
                )
            )
        )
        if n_points == 0:
            ys = _spcrv(ya, order)
            vx = np.diff(ys, axis=0)
            s1 = np.cumsum(np.hstack((0, np.linalg.norm(vx, axis=1))))
            npoints = int(np.ceil(np.max(s1) / step)) + 1
        else:
            npoints = n_points
        xi = _spcrv(ya, order, npoints * kpoints)
    elif interp == "spline":
        _npts = xaux.shape[0]
        sp = spi.CubicSpline(np.arange(_npts), xaux)
        if n_points == 0:
            s2 = np.arange(0, _npts - 1, step / kpoints)
        else:
            s2 = np.linspace(0, _npts - 1, n_points * kpoints)
        xi = sp(s2)
    elif interp == "RBF":
        sp = pathlen(xaux, normscale)
        init_cond = np.zeros((4, xaux.shape[1]))
        RBF = encodeRBF(sp, xaux, N=len(sp), sigma2=0.6, bc=init_cond)
        if n_points == 0:
            t = np.arange(sp[0], sp[-1], step)
        else:
            t = np.linspace(sp[0], sp[-1], n_points)
        if nd == 7:
            xi = decodeCartesianRBF(t, RBF)
        else:
            xi = decodeRBF(t, RBF)
    else:
        xi = xaux

    if nd == 7:
        xi = xnormalize(xi)

    # remove duplicates
    n = xi.shape[0]
    idx = []
    for i in range(1, n):
        if np.array_equal(xi[i, :], xi[0, :]):
            idx.append(i)
        else:
            break
    for i in range(n - 2, -1, -1):
        if np.array_equal(xi[i, :], xi[n - 1, :]):
            idx.append(i)
        else:
            break
    xi = np.delete(xi, idx, axis=0)

    # Velocity
    si = np.linspace(0, 1, xi.shape[0])
    if natural:
        if nd == 7:
            xi = uniqueCartesianPath(xi)
            xid = gradientCartesianPath(xi, si)
        else:
            xid = gradientPath(xi, si)
        si = pathlen(xi, normscale)
        sid = np.diff(si)
        ff = np.where(sid == 0)[0]
        si = np.delete(si, ff)
        xi = np.delete(xi, ff, axis=0)
        sie = np.linspace(0, np.max(si), len(si) // kpoints)
        if nd == 7:
            xi = interpCartesianPath(si, xi, sie)
        else:
            xi = interpPath(si, xi, sie)
        si = pathlen(xi, normscale)
    sid = gradientPath(si)
    if nd == 7:
        xi = uniqueCartesianPath(xi)
        xid = gradientCartesianPath(xi, si)
        nxid = xerrnorm(xid, normscale)
    else:
        xid = gradientPath(xi, si)
        nxid = np.linalg.norm(xid, axis=1)
    sidd = gradientPath(sid)
    xidd = gradientPath(xid, si)

    # Plots
    if plot:
        plotcpath(si, xi, points=points, auxpoints=xaux, ori_sel=ori_sel, fig_num="Generated path over points")

    return xi, si


def pathlen(path: np.ndarray, Cartesian=True, scale: float = [1.0, 1.0]) -> np.ndarray:
    """Path length in m dimensional space

    Parameters
    ----------
    path : ndarray
        path (n, m)
    Cartesian : bool
        Flag for path type (True -> Cartesian)
    scale : float, optional
        position and orientaion norm scales for Cartesian path, by default [1.0, 1.0]

    Returns
    -------
    ndarray
        natural path parameter (n,)
    """
    path = rbs_type(path)
    if isscalar(scale):
        scale = [1, scale]

    m = path.shape[1]

    if m == 7 and Cartesian:
        dp = np.diff(path[:, 0:3], axis=0)
        dq = 2 * qlog(qmtimes(path[1:, 3:7], qinv(path[:-1, 3:7])))
        dx = (scale[0] * np.sum(dp**2, axis=1) + scale[1] * np.sum(dq**2, axis=1)) ** 0.5
    elif m == 4 and Cartesian:
        dq = 2 * qlog(qmtimes(path[1:, :], qinv(path[:-1, :])))
        dx = np.sum(dq**2, axis=1) ** 0.5
    else:
        dp = np.diff(path, axis=0)
        dx = np.sum(dp**2, axis=1) ** 0.5

    si = np.cumsum(np.concatenate(([0], np.abs(dx))))

    return si


def distance2path(x: np.ndarray, path: np.ndarray, s: np.ndarray, *args) -> np.ndarray:
    """Find the closest point on path and the distance to it

    Parameters
    ----------
    x : np.ndarray
        point (3 x 1) or (1 x 7)
    path : np.ndarray
        path (n x 3) or (n x 7)
    s : np.ndarray
        path parameter (n x 1)
    scale: float
        rotation norm scaling used only for Cartesian path (nsamp x 7) (optional)


    Returns
    -------
    np.ndarray
        closest point on path (3 x 1) or (1 x 7)
    float
        distance
    float
        path parameter of closest point

    Raises
    ------
    ValueError
        Wrong input dimension
    """

    x = vector(x)
    s2 = path.shape
    if isvector(x, dim=3) and (x.size >= 3):
        path = path[:, :3]
        tmp1 = path - x
        tmp2 = np.linalg.norm(tmp1, axis=1)
        i = np.argmin(tmp2)
    elif isvector(x1, dim=7) and (s2[1] == 7):
        if len(args) < 4:
            scale = 1.0
        else:
            scale = args[0]
        x = np.expand_dims(x, 0)
        x = np.repeat(x, s2[1], axis=0)
        tmp1 = xerr(path, x)
        tmp2 = np.linalg.norm(tmp1[:, :3], axis=1) + scale * np.linalg.norm(tmp1[:, 3:6], axis=1)
        i = np.argmin(tmp2)
    else:
        raise ValueError("Wrong input dimension")
    d = tmp2[i]
    px = path[i, :]
    return px, d, s[i]


if __name__ == "__main__":
    from transformations import map_pose, rot_x, rot_y, rot_z, rpy2q, qnormalize

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    import matplotlib.pyplot as plt

    t = np.linspace(0, 2, num=201)

    # # Joint trajectories
    q0 = np.array((0, 1, 6, -3))
    q1 = np.array((1, 2, 3, 4))
    q2 = np.array((2, 3, -5, 7))

    fig, ax = plt.subplots(3, 3, num="Joint trajectories using 'jline'", figsize=(8, 8))
    qt, qdt, qddt = jline(q1, q2, t)
    ax[0, 0].plot(t, qt)
    ax[0, 0].grid()
    ax[0, 0].set_title("Line")
    ax[1, 0].plot(t, qdt)
    gqt = np.gradient(qt, t, axis=0)
    ax[1, 0].plot(t, gqt, "--")
    ax[1, 0].grid()
    ax[1, 0].set_title("Velocity")
    ax[2, 0].plot(t, qddt)
    ax[2, 0].grid()
    ax[2, 0].set_title("Acceleration")

    qt, qdt, qddt = jtrap(q1, q2, t)
    ax[0, 1].plot(t, qt)
    ax[0, 1].grid()
    ax[0, 1].set_title("Trap")
    ax[1, 1].plot(t, qdt)
    gqt = np.gradient(qt, t, axis=0)
    ax[1, 1].plot(t, gqt, "--")
    ax[1, 1].grid()
    ax[1, 1].set_title("Velocity")
    ax[2, 1].plot(t, qddt)
    ax[2, 1].grid()
    ax[2, 1].set_title("Acceleration")

    qt, qdt, qddt = jtraj(q1, q2, t)
    ax[0, 2].plot(t, qt)
    ax[0, 2].grid()
    ax[0, 2].set_title("Traj")
    ax[1, 2].plot(t, qdt)
    gqt = np.gradient(qt, t, axis=0)
    ax[1, 2].plot(t, gqt, "--")
    ax[1, 2].grid()
    ax[1, 2].set_title("Velocity")
    ax[2, 2].plot(t, qddt)
    ax[2, 2].grid()
    ax[2, 2].set_title("Acceleration")

    # Cartesian trajectories
    p0 = np.array([0, 1, 3])
    p1 = np.array([1, 4, -1])
    p2 = np.array([-1, 1, 1])
    p3 = np.array([0, 3, 2])
    Q0 = rot_x(0, unit="deg")
    Q1 = rot_x(60, unit="deg")
    Q2 = rot_y(30, unit="deg")
    Q3 = rot_z(45, unit="deg")
    x0 = map_pose(Q=Q0, p=p0, out="x")
    x1 = map_pose(Q=Q1, p=p1, out="x")
    x2 = map_pose(Q=Q2, p=p2, out="x")
    x3 = map_pose(Q=Q3, p=p3, out="x")

    x0 = np.array([0.0349, -0.4928, 0.6526, 0.0681, 0.7280, -0.6782, 0.0730])
    x1 = np.array([0.4941, 0.0000, 0.6526, 0.0000, -0.9950, 0.0000, -0.0998])
    xt, xdt, xddt = ctrap(x0, x1, t)
    xt1, xdt1, xddt1 = cline(x0, x1, t)
    xt2, xdt2, xddt2 = cpoly(x0, x1, t)
    fig, ax = plt.subplots(
        3,
        2,
        num="Cartesian trajectories using 'ctrap', 'cline and 'cpoly'",
        figsize=(8, 8),
    )
    ax[0, 0].plot(t, xt[:, :3])
    ax[0, 0].plot(t, xt1[:, :3], "--")
    ax[0, 0].plot(t, xt2[:, :3], ":")
    ax[0, 0].grid()
    ax[0, 0].set_title("$p$")
    ax[1, 0].plot(t, xdt[:, :3])
    ax[1, 0].plot(t, xdt1[:, :3], "--")
    ax[1, 0].plot(t, xdt2[:, :3], ":")
    ax[1, 0].grid()
    ax[1, 0].set_title("$\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].plot(t, xddt1[:, :3], "--")
    ax[2, 0].plot(t, xddt2[:, :3], ":")
    ax[2, 0].grid()
    ax[2, 0].set_title("$\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].plot(t, xt1[:, 3:], "--")
    ax[0, 1].plot(t, xt2[:, 3:], ":")
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].plot(t, xdt1[:, 3:], "--")
    ax[1, 1].plot(t, xdt2[:, 3:], ":")
    ax[1, 1].grid()
    ax[1, 1].set_title("$\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].plot(t, xddt1[:, 3:], "--")
    ax[2, 1].plot(t, xddt2[:, 3:], ":")
    ax[2, 1].grid()
    ax[2, 1].set_title("$\dot\omega$")

    # Trajectory - Multi point interpolation
    t = np.linspace(0, 4, num=401)
    ti, _, _ = jtraj(0, 2, t)
    s = [0, 1, 1.75, 2]
    xx = np.vstack((x0, x1, x2, x3))

    xt = interpCartesianPath(s, xx, ti)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)

    fig, ax = plt.subplots(3, 2, num="Trajectory using multi point interpolation", figsize=(8, 8))
    ax[0, 0].plot(t, xt[:, :3])
    ax[0, 0].grid()
    ax[0, 0].set_title("$p$")
    ax[1, 0].plot(t, xdt[:, :3])
    ax[1, 0].grid()
    ax[1, 0].set_title("$\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].grid()
    ax[2, 0].set_title("$\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].grid()
    ax[1, 1].set_title("$\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].grid()
    ax[2, 1].set_title("$\dot\omega$")

    # Cartesian arc trajectories
    p0 = np.array([0, 1, 3])
    p1 = np.array([1, 4, -1])
    pC = np.array([-1, 1, 1])
    Q0 = rot_x(0, unit="deg")
    Q1 = rot_x(60, unit="deg")
    Q2 = rot_y(30, unit="deg")
    x0 = map_pose(Q=Q0, p=p0)
    x1 = map_pose(Q=Q1, p=p1)

    xt, xdt, xddt = carctraj(x0, x1, pC, -t)
    pp = np.array([2, 1, 2])
    px, d, sx = distance2path(pp, xt, t)
    print("Distance to path:", d)
    print("Closest point on path:", px, " at path parameter:", sx)

    fig, ax = plt.subplots(3, 2, num="Cartesian trajectory using 'carctraj'", figsize=(8, 8))
    ax[0, 0].plot(t, xt[:, :3])
    ax[0, 0].grid()
    ax[0, 0].set_title("$p$")
    ax[1, 0].plot(t, xdt[:, :3])
    ax[1, 0].grid()
    ax[1, 0].set_title("$\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].grid()
    ax[2, 0].set_title("$\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].grid()
    ax[1, 1].set_title("$\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].grid()
    ax[2, 1].set_title("$\dot\omega$")

    from mpl_toolkits import mplot3d

    fig = plt.figure(num="Cartesian trajectory using 'carctraj'")
    ax = plt.axes(projection="3d")
    ax.plot(xt[:, 0], xt[:, 1], xt[:, 2])
    ax.plot(
        [pp[0], px[0]],
        [pp[1], px[1]],
        [pp[2], px[2]],
        color="y",
        linestyle="-",
        linewidth=2,
    )
    ax.scatter(x0[0], x0[1], x0[2], color="k")
    ax.text(x0[0], x0[1], x0[2], "$P_0$")
    ax.scatter(x1[0], x1[1], x1[2], color="k")
    ax.text(x1[0], x1[1], x1[2], "$P_1$")
    ax.scatter(pC[0], pC[1], pC[2], color="blue")
    ax.text(pC[0], pC[1], pC[2], "$P_C$")
    ax.scatter(pp[0], pp[1], pp[2], color="green")
    ax.text(pp[0], pp[1], pp[2], "$P$")
    ax.scatter(px[0], px[1], px[2], color="red")
    ax.text(px[0], px[1], px[2], "$P_x$")
    ax.grid()
    ax.set_aspect("equal")
    ax.set_title("Arc traj in $3D$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    rpy = np.array([[0, 0, 0], [np.pi, -np.pi / 2, 0], [0, np.pi / 2, 0], [0, 0, 0]])
    print(rpy)
    q = rpy2q(rpy)
    q[:, 3] = 0
    q = qnormalize(q)
    print(q)
    n = q.shape[0]

    s = np.linspace(0, 1, n)
    t = np.linspace(0, 1, 11)
    q_slerp = interpQuaternionPath(s, q, t)
    print("Q slerp interpolation:\n", q_slerp)

    q_squad = qspline(q, t, "squad")
    print("Q squad interpolation:\n", q_squad)

    q_hermite = qspline(q, t, "hermite_cubic")
    print("Q hermite interpolation:\n", q_hermite)

    pte = np.array(
        [
            [-0.2, -0.2, -0.175, 0, 0, 0],
            [-0.2, -0.2, 0.075, 0, 0, -np.pi / 2],
            [-0.2, 0.1, 0.075, -np.pi / 2, 0, -np.pi / 2],
            [0.2, 0.1, 0.075, -np.pi, 0, -np.pi / 2],
        ]
    )
    pt = prpy2x(pte)

    print("Points: \n", pt)
    print(
        "Auxpoints: \n",
        pathauxpoints(pt, auxpoints="relative", viapoints=False, auxdistance=[0.25, 0.1]),
    )

    path, si = pathoverpoints(
        pt,
        interp="spline",
        step=0.01,
        natural=False,
        plot=True,
        auxpoints="relative",
        auxdistance=0.25,
    )

    plt.show()  # Display the generated plot
