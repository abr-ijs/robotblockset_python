"""Trajectory generation and interpolation utilities.

This module provides a collection of functions to generate, interpolate, and process different types of trajectories for robotic or
path planning systems. The main functionalities include creating Cartesian and joint space trajectories, performing interpolation on
spatial poses (SE3), computing velocities and accelerations, and generating auxiliary points on paths. It supports various interpolation
methods such as linear, cubic splines, and radial basis functions (RBF), along with rotation interpolation (SLERP) for quaternion-based
orientations.

Trajectory Types:
-----------------
- **Cartesian trajectory**: Includes both position and orientation (quaternion), used for path planning in Cartesian space.
- **Joint space trajectory**: Describes movements in a robot's joint space, typically used for articulated robot arm movements.
- **Spline trajectory**: Smoothing or interpolation between points using cubic splines or other methods to ensure smooth transitions.
- **RBF-based trajectory**: Trajectories that are generated using Radial Basis Function interpolation, useful for more complex path generation.

The module is designed to be flexible and easily adaptable for various robotic applications, path planners, and trajectory optimization tasks.
It can handle 2D and 3D space and supports various interpolation techniques to meet different planning requirements.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy.interpolate as spi

from copy import copy
from ruckig import InputParameter, OutputParameter, Result, Trajectory, Ruckig

try:
    import matplotlib.pyplot as plt
    from robotblockset.graphics import plotcpath
except Exception:
    plt = None
    print("Warning: matplotlib.pyplot could not be imported. Graphics related functions will not work!")

from robotblockset.tools import check_option, gradientCartesianPath, gradientPath, ismatrix, ismatrixarray, isscalar, isvector, normalize, vector
from robotblockset.transformations import ang4v, prpy2x, qerr, qexp, qinv, qlog, qmtimes, q2r, r2q, rbs_type, t2x, uniqueCartesianPath, x2t, xerr, xnormalize, rot_v, qnormalize
from robotblockset.rbf import encodeRBF, decodeRBF, decodeCartesianRBF
from robotblockset.rbs_typing import (
    Velocity3DType,
    Acceleration3DType,
    Accelerations3DType,
    ArrayLike,
    HomogeneousMatricesType,
    HomogeneousMatrixType,
    JointConfigurationType,
    JointPathType,
    Pose3DType,
    Poses3DType,
    QuaternionType,
    QuaternionsType,
    RotationMatricesType,
    RotationMatrixType,
    TimesType,
    Velocities3DType,
    Vector3DType,
    Vector3DArrayType,
)

_eps = 100 * np.finfo(np.float64).eps


def arc(p0: Vector3DType, p1: Vector3DType, pC: Vector3DType, s: float, short: bool = True) -> Vector3DArrayType:
    """
    Generates points on an arc defined by two points and a center point.

    The arc is centered at `pC`, starting at point `p0` and ending at point `p1`.
    If the distances from `pC` to `p0` and `p1` are not equal, `pC` is projected to a point
    on the midline between `p0` and `p1`.

    If `s` < 0, the long path is used for the arc. If `short` is `True`, the shortest rotation
    is taken.

    Parameters
    ----------
    p0 : Vector3DType
        Initial point on the arc (3,)
    p1 : Vector3DType
        Final point on the arc (3,)
    pC : Vector3DType
        Center point of the arc (3,)
    s : float
        Normalized arc distance in the range [0..1]
    short : bool, optional
        If `True`, the shortest rotation is used (default is `True`)

    Returns
    -------
    Vector3DType
        Points on the arc (3,) or (n, 3)

    Raises
    ------
    ValueError
        If the points are not distinct or collinear.
    """
    p0 = vector(p0, dim=3)
    p1 = vector(p1, dim=3)
    pC = vector(pC, dim=3)

    v1 = p0 - pC
    v2 = p1 - pC
    v1n = np.linalg.norm(v1)
    v2n = np.linalg.norm(v2)

    # Angle between vectors v1 and v2
    phi = ang4v(v1, v2)

    if v1n > 0 and v2n > 0 and phi > 0.001 and phi < (np.pi - 0.001):
        v3 = normalize(np.cross(v1, v2))  # Normal to the plane
        v4 = normalize(np.cross(p1 - p0, v3))  # Second normal for reorientation
        p01 = (p0 + p1) / 2  # Midpoint between p0 and p1
        pCx = np.dot(pC - p01, v4) * v4 + p01  # Projection of pC onto the midline

        v1 = p0 - pCx
        v2 = p1 - pCx
        phi = ang4v(v1, v2)

        # Adjust for long arc if s < 0
        if s < 0:
            short = False

        if not short:
            phi = 2 * np.pi - phi  # Use long path
            s = -np.abs(s)  # Ensure s is negative for long arc

        # Perform rotation
        return rot_v(v3, s * phi, out="R") @ v1 + pCx
    else:
        raise ValueError("Points must be distinct and not on the same line")


def carctraj(x0: Pose3DType, x1: Pose3DType, pC: Vector3DType, t: TimesType, traj: str = "Poly", short: bool = True) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generates a Cartesian trajectory on an arc from `x0` to `x1`, with the arc defined by the initial and final poses and an arc center.

    The arc is parameterized by time `t`, and the trajectory type (`poly`, `trap`, or `line`) determines the interpolation method.
    The trajectory considers the shortest rotation if `short` is `True`.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,)
    x1 : Pose3DType
        Final Cartesian pose (7,)
    pC : Vector3DType
        Arc center position (3,)
    t : TimesType
        Time array (nsamp,)
    traj : str, optional
        Trajectory type: `"Poly"` for polynomial, `"Trap"` for trapezoidal, or `"Line"` for linear interpolation. Default is `"Poly"`.
    short : bool, optional
        If `True`, the shortest rotation path is used. Default is `True`.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7)
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6)
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6)

    Raises
    ------
    ValueError
        If the trajectory time `t` is invalid or if the arc parameters are incompatible.
    """
    # Convert input arguments to the appropriate dimensions
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    pC = vector(pC, dim=3)

    # If the last time value is negative, adjust the trajectory time
    if t[-1] < 0:
        t = np.abs(t)
        short = False  # Use the long path if the time is negative

    # Generate the normalized trajectory parameters
    s, _, _ = jtraj(0.0, 1.0, t, traj=traj)

    # Interpolate the Cartesian trajectory along the arc
    xt = xarcinterp(x0, x1, pC, s, short=short)

    # Calculate the velocity and acceleration along the trajectory
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)

    return xt, xdt, xddt


def jline(q0: JointConfigurationType, q1: JointConfigurationType, t: TimesType, **kwargs) -> Tuple[JointPathType, JointPathType, JointPathType]:
    """
    Generates a trajectory from joint position `q0` to `q1` with constant velocity.

    This function calculates a linear trajectory from an initial joint position `q0` to a final
    joint position `q1` with constant velocity over the provided time interval `t`. It also
    computes the joint velocities and accelerations, which are constant for the entire trajectory.

    Parameters
    ----------
    q0 : JointConfigurationType
        Initial joint positions (n,), where `n` is the number of joints.
    q1 : JointConfigurationType
        Final joint positions (n,), where `n` is the number of joints.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.

    Returns
    -------
    tuple
        - qt : JointPathType
          Interpolated joint positions (nsamp, n).
        - qdt : JointPathType
          Interpolated joint velocities (nsamp, n), constant throughout the trajectory.
        - qddt : JointPathType
          Interpolated joint accelerations (nsamp, n), which are zero since velocity is constant.

    Raises
    ------
    ValueError
        If the trajectory time values are invalid (i.e., non-positive or incorrect).
    TypeError
        If the input vectors `q0` and `q1` do not have the same size.
    """
    # Convert input arguments to appropriate format
    q0 = vector(q0)
    m = q0.shape[0]
    q1 = vector(q1, dim=m)
    t = vector(t)

    # Check that the sizes of the initial and final positions match
    if q0.size == q1.size:
        t = t - t[0]  # Normalize time so it starts from zero
        tmax = np.max(t)  # Find the maximum time value
        _t = t / tmax  # Normalize time to the range [0, 1]

        # Check if the maximum time is valid (positive)
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")

        # Linear interpolation of joint positions
        if q0.size == 1:  # Scalar case (single joint)
            qt = q0 + (q1 - q0) * _t
        else:
            # Vectorized case (multiple joints)
            qt = q0 + np.einsum("i,j->ji", q1 - q0, _t)

        # Constant joint velocities
        dq = (q1 - q0) / tmax
        qdt = np.ones(qt.shape) * dq  # Constant velocity for all time samples

        # Zero acceleration, as velocity is constant
        return qt, qdt, np.zeros(qt.shape)
    else:
        # Raise an error if input vectors don't match in size
        raise TypeError("Input vectors must be of the same size")


def jtrap(q0: JointConfigurationType, q1: JointConfigurationType, t: TimesType, ta: float = 0.1, **kwargs) -> Tuple[JointPathType, JointPathType, JointPathType]:
    """
    Generates a trajectory from `q0` to `q1` using trapezoidal velocity profile.

    This function generates a trajectory from an initial joint position `q0` to a final joint
    position `q1` using a trapezoidal velocity profile. The trajectory consists of three phases:
    acceleration, constant velocity, and deceleration. The time spent on acceleration and
    deceleration is specified by `ta`.

    Parameters
    ----------
    q0 : JointConfigurationType
        Initial joint positions (n,). The number of elements corresponds to the number of joints.
    q1 : JointConfigurationType
        Final joint positions (n,). The number of elements corresponds to the number of joints.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    ta : float, optional
        Acceleration/deceleration time (default is 0.1). This parameter controls the duration
        of the acceleration and deceleration phases.

    Returns
    -------
    tuple
        - qt : JointPathType
          Interpolated joint positions (nsamp, n).
        - qdt : JointPathType
          Interpolated joint velocities (nsamp, n).
        - qddt : JointPathType
          Interpolated joint accelerations (nsamp, n).

    Raises
    ------
    ValueError
        If the trajectory time values are non-positive or incorrect.
    TypeError
        If the input vectors `q0` and `q1` do not have the same size.
    """
    # Convert inputs to proper vector format
    q0 = vector(q0)
    m = q0.shape[0]
    q1 = vector(q1, dim=m)
    t = vector(t)

    ta = min(ta, t[-1] / 2)  # Ensure ta is not greater than half the total time

    # Ensure the initial and final positions have the same size
    if q0.size == q1.size:
        # Normalize time to start from zero
        t = t - t[0]
        tmax = np.max(t)  # Find the maximum time value

        # Check if the trajectory time is valid
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")

        # Calculate the acceleration constant
        acc = 1 / (ta * (tmax - ta))

        # Define the piecewise functions for position, velocity, and acceleration
        s = lambda t: ((t <= ta) * 0.5 * acc * t**2 + (np.logical_and(t > ta, t <= (tmax - ta))) * (0.5 * acc * ta**2 + acc * ta * (t - ta)) + (t > (tmax - ta)) * (1 - 0.5 * acc * (tmax - t) ** 2))
        v = lambda t: ((t <= ta) * acc * t + (np.logical_and(t > ta, t <= (tmax - ta))) * acc * ta + (t > (tmax - ta)) * acc * (tmax - t))
        a = lambda t: (t <= ta) * acc - (t > (tmax - ta)) * acc

        # Apply the functions to time array
        st = np.array([s(x) for x in t])
        vt = np.array([v(x) for x in t])
        at = np.array([a(x) for x in t])

        # Calculate the joint positions, velocities, and accelerations
        if q0.size == 1:  # Scalar case (single joint)
            qt = q0 + (q1 - q0) * st
            qdt = q0 + (q1 - q0) * vt
            qddt = q0 + (q1 - q0) * at
        else:  # Vectorized case (multiple joints)
            qt = q0 + np.einsum("i,j->ji", q1 - q0, st)
            qdt = np.einsum("i,j->ji", q1 - q0, vt)
            qddt = np.einsum("i,j->ji", q1 - q0, at)

        return qt, qdt, qddt
    else:
        # Raise an error if the input vectors don't match in size
        raise TypeError("Input vectors must be the same size")


def jpoly(
    q0: JointConfigurationType,
    q1: JointConfigurationType,
    t: TimesType,
    qd0: Optional[JointConfigurationType] = None,
    qd1: Optional[JointConfigurationType] = None,
    **kwargs,
) -> Tuple[JointPathType, JointPathType, JointPathType]:
    """
    Generates a trajectory from `q0` to `q1` using a 5th order polynomial.

    This function computes a smooth trajectory from an initial joint position `q0` to a final joint
    position `q1` using a 5th order polynomial. The polynomial is determined such that the joint
    positions, velocities, and accelerations are continuous and smooth. Optional initial and final
    joint velocities (`qd0` and `qd1`) can be provided to control the velocity at the start and end.

    Parameters
    ----------
    q0 : JointConfigurationType
        Initial joint positions (n,). The number of elements corresponds to the number of joints.
    q1 : JointConfigurationType
        Final joint positions (n,). The number of elements corresponds to the number of joints.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    qd0 : JointConfigurationType, optional
        Initial joint velocities (n,). Defaults to zero if not provided.
    qd1 : JointConfigurationType, optional
        Final joint velocities (n,). Defaults to zero if not provided.

    Returns
    -------
    tuple
        - qt : JointPathType
          Interpolated joint positions (nsamp, n).
        - qdt : JointPathType
          Interpolated joint velocities (nsamp, n).
        - qddt : JointPathType
          Interpolated joint accelerations (nsamp, n).

    Raises
    ------
    TypeError
        If the input vectors `q0`, `q1`, `qd0`, and `qd1` do not have the same size.
    ValueError
        If the trajectory time values are non-positive or incorrect.
    """
    q0 = vector(q0)
    m = q0.shape[0]
    q1 = vector(q1, dim=m)
    t = vector(t)
    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = vector(qd0, dim=m)
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd1 = vector(qd1, dim=m)
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
        a = np.array([np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)]).reshape((6, q0.size)) / tmax**2
        qt = np.einsum("ij,ik->kj", s, tt)
        qdt = np.einsum("ij,ik->kj", v, tt)
        qddt = np.einsum("ij,ik->kj", a, tt)
        if q0.size == 1:
            return qt.flatten(), qdt.flatten(), qddt.flatten()
        else:
            return qt, qdt, qddt
    else:
        raise TypeError("Input vecotrs must be same size")


def jjerk(
    q0: JointConfigurationType,
    q1: JointConfigurationType,
    t: TimesType,
    qd0: Optional[JointConfigurationType] = None,
    qd1: Optional[JointConfigurationType] = None,
    qd_max: Optional[JointConfigurationType] = None,
    qd_min: Optional[JointConfigurationType] = None,
    qdd_max: Optional[JointConfigurationType] = None,
    qdd_min: Optional[JointConfigurationType] = None,
    jerk_max: Optional[JointConfigurationType] = None,
    resample: bool = True,
    **kwargs,
) -> Tuple[JointPathType, JointPathType, JointPathType]:
    """
    Generates a trajectory from `q0` to `q1` using a limited jerk profile (Ruckig: Jerk-Limited Trajectory Generation).

    This function computes a smooth trajectory from an initial joint position `q0` to a final joint
    position `q1` using a limited jerk profile. The profile is determined such that the joint
    positions, velocities, and accelerations are continuous and smooth. Optional initial and final
    joint velocities (`qd0` and `qd1`) can be provided to control the velocity at the start and end.

    Parameters
    ----------
    q0 : JointConfigurationType
        Initial joint positions (n,). The number of elements corresponds to the number of joints.
    q1 : JointConfigurationType
        Final joint positions (n,). The number of elements corresponds to the number of joints.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    qd0 : JointConfigurationType, optional
        Initial joint velocities (n,). Defaults to zero if not provided.
    qd1 : JointConfigurationType, optional
        Final joint velocities (n,). Defaults to zero if not provided.
    qd_max : JointConfigurationType, optional
        Maximum joint velocities (n,). Defaults to 1 if not provided.
    qd_min : JointConfigurationType, optional
        Minimum joint velocities (n,). Defaults to -1 if not provided.
    qdd_max : JointConfigurationType, optional
        Maximum joint accelerations (n,). Defaults to 2 if not provided.
    jerk_max : JointConfigurationType, optional
        Maximum joint jerks (n,). Defaults to 10 if not provided.
    resample : bool, optional
        Whether to resample the trajectory due to longer duration. Defaults to True.

    Returns
    -------
    tuple
        - qt : JointPathType
          Interpolated joint positions (nsamp, n).
        - qdt : JointPathType
          Interpolated joint velocities (nsamp, n).
        - qddt : JointPathType
          Interpolated joint accelerations (nsamp, n).

    Raises
    ------
    TypeError
        If the input vectors `q0`, `q1`, `qd0`, and `qd1` do not have the same size.
    ValueError
        If the trajectory time values are non-positive or incorrect.
    """
    q0 = vector(q0)
    m = q0.shape[0]
    q1 = vector(q1, dim=m)
    t = vector(t)
    if qd0 is None:
        qd0 = np.zeros(m)
    else:
        qd0 = vector(qd0, dim=m)
    if qd1 is None:
        qd1 = np.zeros(m)
    else:
        qd1 = vector(qd1, dim=m)
    if qd_max is None:
        qd_max = np.ones(m) * 2
    elif isinstance(qd_max, (int, float)):
        qd_max = np.ones(m) * qd_max
    else:
        qd_max = vector(qd_max, dim=m)
    if qdd_max is None:
        qdd_max = np.ones(m) * 10
    elif isinstance(qdd_max, (int, float)):
        qdd_max = np.ones(m) * qdd_max
    else:
        qdd_max = vector(qdd_max, dim=m)
    if jerk_max is None:
        jerk_max = np.ones(m) * 100
    elif isinstance(jerk_max, (int, float)):
        jerk_max = np.ones(m) * jerk_max
    else:
        jerk_max = vector(jerk_max, dim=m)
    if q0.size == q1.size and qd0.size == q0.size and qd1.size == q1.size:
        tmax = max(t)
        dt = t[1] - t[0]
        if tmax <= 0:
            raise ValueError("Incorrect trajectory time values")

        otg = Ruckig(m, dt)
        inp = InputParameter(m)
        out = OutputParameter(m)

        # Set input parameters
        inp.current_position = q0.tolist()
        inp.current_velocity = qd0.tolist()
        inp.current_acceleration = np.zeros(m)
        inp.minimum_duration = tmax

        inp.target_position = q1.tolist()
        inp.target_velocity = qd1.tolist()
        inp.target_acceleration = np.zeros(m)

        inp.max_velocity = qd_max
        inp.max_acceleration = qdd_max
        inp.max_jerk = jerk_max

        if qd_min is not None:
            qd_min = vector(qd_min, dim=m)
            inp.min_velocity = qd_min
        if qdd_min is not None:
            qdd_min = vector(qdd_min, dim=m)
            inp.min_acceleration = qdd_min

        trajectory = Trajectory(m)

        result = otg.calculate(inp, trajectory)
        if result == Result.ErrorInvalidInput:
            raise Exception("Invalid input!")

        if tmax < trajectory.duration:
            if resample:
                # print(f"Trajectory duration {trajectory.duration} is longer than the provided time interval {tmax}. It will be prolonged to fit the trajectory duration.")
                tt = np.arange(0, trajectory.duration + dt, dt)
            else:
                raise ValueError(f"Trajectory duration {trajectory.duration} is longer than the provided time interval {tmax}")
        else:
            tt = t - t[0]
        q = []
        qd = []
        qdd = []
        for ti in tt:
            new_position, new_velocity, new_acceleration = trajectory.at_time(ti)
            q.append(new_position)
            qd.append(new_velocity)
            qdd.append(new_acceleration)

        qt = np.array(q)
        qdt = np.array(qd)
        qddt = np.array(qdd)
        return qt, qdt, qddt
    else:
        raise TypeError("Input vecotrs must be same size")


def jtraj(q0: JointConfigurationType, q1: JointConfigurationType, t: TimesType, traj: str = "Poly", qd0: Optional[JointConfigurationType] = None, qd1: Optional[JointConfigurationType] = None, **kwargs) -> Tuple[JointPathType, JointPathType, JointPathType]:
    """
    Generate a trajectory from initial joint positions `q0` to final joint positions `q1` over time `t`.

    Parameters
    ----------
    q0 : JointConfigurationType
        Initial joint positions (n,).
    q1 : JointConfigurationType
        Final joint positions (n,).
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    traj : str, optional
        Trajectory type. Possible values are:
        - 'poly': Polynomial trajectory.
        - 'jerk': Jerk-limited trajectory using Ruckig.
        - 'trap': Trapezoidal trajectory.
        - 'line': Linear trajectory.
        By default, 'poly'.
    qd0 : JointConfigurationType, optional
        Initial joint velocities (n,). Required for 'trap' and 'poly' types with velocity constraints.
    qd1 : JointConfigurationType, optional
        Final joint velocities (n,). Required for 'trap' and 'poly' types with velocity constraints.
    **kwargs : dict, optional

    Returns
    -------
    tuple
        - qt : JointPathType
            Interpolated joint positions (nsamp, n).
        - qdt : JointPathType
            Interpolated joint velocities (nsamp, n).
        - qddt : JointPathType
            Interpolated joint accelerations (nsamp, n).

    Raises
    ------
    ValueError
        If the trajectory type is unsupported.
    """

    q0 = vector(q0)
    q1 = vector(q1)
    if check_option(traj, "Poly"):
        _traj = jpoly
    elif check_option(traj, "Trap"):
        _traj = jtrap
    elif check_option(traj, "Line"):
        _traj = jline
    elif check_option(traj, "Jerk"):
        _traj = jjerk
    else:
        raise ValueError(f"Trajectory type {traj} not supported")
    return _traj(q0, q1, t, **kwargs)


def cline(x0: Pose3DType, x1: Pose3DType, t: TimesType, short: bool = True, **kwargs) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generate a Cartesian trajectory from `x0` to `x1` with constant velocity.

    This function generates a trajectory that interpolates between an initial
    Cartesian pose `x0` and a final Cartesian pose `x1`. The trajectory is
    computed with constant velocity, and the poses are defined by both position
    and quaternion. Optionally, the shortest rotation can be chosen.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,). The pose is represented by a 7-element vector,
        where the first three elements are the position, and the last four are the
        quaternion orientation.
    x1 : Pose3DType
        Final Cartesian pose (7,). Similar to `x0`, this is a 7-element vector
        representing the final pose.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    short : bool, optional
        If True (default), the shortest rotation is taken between the initial and
        final orientations. If False, the long path is taken.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7).
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6).
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6).

    Raises
    ------
    ValueError
        If the time array `t` contains non-positive values.
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jline(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def ctrap(x0: Pose3DType, x1: Pose3DType, t: TimesType, short: bool = True, **kwargs) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generate a Cartesian trajectory from `x0` to `x1` with trapezoidal velocity profile.

    This function generates a trajectory between the initial Cartesian pose `x0`
    and the final Cartesian pose `x1`, using a trapezoidal velocity profile. The
    poses are defined by both position and quaternion. Optionally, the shortest
    rotation can be chosen between the two poses.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,). The pose is represented by a 7-element vector,
        where the first three elements are the position, and the last four are the
        quaternion orientation.
    x1 : Pose3DType
        Final Cartesian pose (7,). Similar to `x0`, this is a 7-element vector
        representing the final pose.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    short : bool, optional
        If True (default), the shortest rotation is taken between the initial and
        final orientations. If False, the long path is taken.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7).
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6).
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6).

    Raises
    ------
    ValueError
        If the time array `t` contains non-positive values.
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jtrap(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def cpoly(x0: Pose3DType, x1: Pose3DType, t: TimesType, short: bool = True, **kwargs) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generate a Cartesian trajectory from `x0` to `x1` using a 5th order polynomial.

    This function generates a trajectory between the initial Cartesian pose `x0`
    and the final Cartesian pose `x1`, using a 5th order polynomial. The poses
    are defined by both position and quaternion. Optionally, the shortest rotation
    can be chosen between the two poses.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,). The pose is represented by a 7-element vector,
        where the first three elements are the position, and the last four are the
        quaternion orientation.
    x1 : Pose3DType
        Final Cartesian pose (7,). Similar to `x0`, this is a 7-element vector
        representing the final pose.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    short : bool, optional
        If True (default), the shortest rotation is taken between the initial and
        final orientations. If False, the long path is taken.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7).
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6).
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6).

    Raises
    ------
    ValueError
        If the time array `t` contains non-positive values.
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    s, _, _ = jpoly(0.0, 1.0, t)
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def cjerk(
    x0: Pose3DType,
    x1: Pose3DType,
    t: TimesType,
    short: bool = True,
    v_max: Optional[Velocity3DType] = None,
    v_min: Optional[Velocity3DType] = None,
    a_max: Optional[Acceleration3DType] = None,
    a_min: Optional[Acceleration3DType] = None,
    jerk_max: Optional[Acceleration3DType] = None,
    resample: bool = True,
    **kwargs,
) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generate a Cartesian trajectory from `x0` to `x1` using a limited jerk profile
    (Ruckig: Jerk-Limited Trajectory Generation).

    This function generates a trajectory between the initial Cartesian pose `x0`
    and the final Cartesian pose `x1`, using a limited jerk profile. Ruckig treats
    every DOF as an independent scalar trajectory. That is problematic because
    quaternions must satisfy unit norm constraint. Therefore, we generate a scalar
    trajectory:

            s(t) ∈ [0,1]

    and apply Ruckig trajectory generation to it.

    The poses are defined by both position and quaternion. Optionally, the shortest
    rotation can be chosen between the two poses.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,). The pose is represented by a 7-element vector,
        where the first three elements are the position, and the last four are the
        quaternion orientation.
    x1 : Pose3DType
        Final Cartesian pose (7,). Similar to `x0`, this is a 7-element vector
        representing the final pose.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    short : bool, optional
        If True (default), the shortest rotation is taken between the initial and
        final orientations. If False, the long path is taken.
    v_max : Velocity3DType, optional
        Maximum Cartesian velocities (6,). If not provided, defaults are used.
    v_min : Velocity3DType, optional
        Minimum Cartesian velocities (6,). If not provided, defaults are used.
    a_max : Acceleration3DType, optional
        Maximum Cartesian accelerations (6,). If not provided, defaults are used.
    a_min : Acceleration3DType, optional
        Minimum Cartesian accelerations (6,). If not provided, defaults are used.
    jerk_max : Acceleration3DType, optional
        Maximum Cartesian jerk (6,). If not provided, defaults are used.
    resample : bool, optional
        Whether to resample the trajectory due to longer duration. Defaults to True.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7).
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6).
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6).

    Raises
    ------
    ValueError
        If the time array `t` contains non-positive values.
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    tmax = max(t)
    dt = t[1] - t[0]
    if tmax <= 0:
        raise ValueError("Incorrect trajectory time values")

    dx = xerr(x0, x1)
    dist = np.linalg.norm(dx[:3])
    ang = np.linalg.norm(dx[3:])
    if not short:
        ang = 2 * np.pi - ang
        dx[3:] = 2 * np.pi - dx[3:]
    dx = np.clip(np.abs(dx), _eps, np.inf)

    if v_max is None:
        v_max = np.ones(6) * 2
    elif isinstance(v_max, (int, float)):
        v_max = np.ones(6) * v_max
    else:
        v_max = vector(v_max, dim=6)
    # max_velocity = [np.min(np.concatenate((v_max[:3] / dist, v_max[3:] / ang)))]
    max_velocity = [np.min(v_max / np.abs(dx))]

    if a_max is None:
        a_max = np.ones(6) * 10
    elif isinstance(a_max, (int, float)):
        a_max = np.ones(6) * a_max
    else:
        a_max = vector(a_max, dim=6)
    # max_acceleration = [np.min(np.concatenate((a_max[:3] / dist, a_max[3:] / ang)))]
    max_acceleration = [np.min(a_max / np.abs(dx))]

    if jerk_max is None:
        jerk_max = np.ones(6) * 100
    elif isinstance(jerk_max, (int, float)):
        jerk_max = np.ones(6) * jerk_max
    else:
        jerk_max = vector(jerk_max, dim=6)
    # max_jerk = [np.min(np.concatenate((jerk_max[:3] / dist, jerk_max[3:] / ang)))]
    max_jerk = [np.min(jerk_max / np.abs(dx))]

    min_velocity = None
    min_acceleration = None
    if v_min is not None:
        v_min = vector(v_min, dim=6)
        # min_velocity = [np.min(np.concatenate((v_min[:3] / dist, v_min[3:] / ang)))]
        min_velocity = [np.min(v_min / np.abs(dx))]
    if a_min is not None:
        a_min = vector(a_min, dim=6)
        # min_acceleration = [np.min(np.concatenate((a_min[:3] / dist, a_min[3:] / ang)))]
        min_acceleration = [np.min(a_min / np.abs(dx))]

    s, sd, sdd = jjerk(0, 1, t, qd_max=max_velocity, qd_min=min_velocity, qdd_max=max_acceleration, qdd_min=min_acceleration, jerk_max=max_jerk, resample=resample)
    t = np.arange(s.shape[0]) * dt
    xt = xinterp(x0, x1, s, short=short)
    xdt = gradientCartesianPath(xt, t)
    xddt = gradientPath(xdt, t)
    return xt, xdt, xddt


def ctraj(x0: Pose3DType, x1: Pose3DType, t: TimesType, traj: str = "Poly", short: bool = True, **kwargs) -> Tuple[Poses3DType, Velocities3DType, Accelerations3DType]:
    """
    Generate a Cartesian trajectory from `x0` to `x1` based on the specified trajectory type.

    This function generates a trajectory between two Cartesian poses, `x0` and `x1`,
    with options for different types of trajectories: polynomial, trapezoidal, or linear.
    The poses are defined by both position and quaternion.

    Parameters
    ----------
    x0 : Pose3DType
        Initial Cartesian pose (7,). The pose is represented by a 7-element vector,
        where the first three elements are the position, and the last four are the
        quaternion orientation.
    x1 : Pose3DType
        Final Cartesian pose (7,). Similar to `x0`, this is a 7-element vector
        representing the final pose.
    t : TimesType
        Time array (nsamp,) for the trajectory evaluation.
    traj : str, optional
        The type of trajectory to generate. Options are:
        - "Poly" for polynomial (default),
        - "Jerk" for jerk-limited trajectory using Ruckig,
        - "Trap" for trapezoidal velocity,
        - "Line" for linear velocity.
    short : bool, optional
        If True (default), the shortest rotation is taken between the initial and
        final orientations. If False, the long path is taken.

    Returns
    -------
    tuple
        - xt : Poses3DType
          Cartesian trajectory - pose (nsamp, 7).
        - xdt : Velocities3DType
          Cartesian trajectory - velocity (nsamp, 6).
        - xddt : Accelerations3DType
          Cartesian trajectory - acceleration (nsamp, 6).

    Raises
    ------
    ValueError
        If the trajectory type is not one of the supported types: "Jerk", "Poly", "Trap", or "Line".
    """
    x0 = vector(x0, dim=7)
    x1 = vector(x1, dim=7)
    if check_option(traj, "Poly"):
        _traj = cpoly
    elif check_option(traj, "Trap"):
        _traj = ctrap
    elif check_option(traj, "Line"):
        _traj = cline
    elif check_option(traj, "Jerk"):
        _traj = cjerk
    else:
        raise ValueError(f"Trajectory type {traj} not supported")
    return _traj(x0, x1, t, short=short, **kwargs)


def interp(y1: ArrayLike, y2: ArrayLike, s: ArrayLike) -> np.ndarray:
    """
    Perform multidimensional linear interpolation between two sets of data points.

    This function linearly interpolates between two sets of data points `y1` and `y2`
    for a set of query points `s`. The interpolation is performed element-wise.

    Parameters
    ----------
    y1 : ArrayLike
        Initial data points (n,). A 1-dimensional array of floats representing the initial data points.
    y2 : ArrayLike
        Final data points (n,). A 1-dimensional array of floats representing the final data points.
    s : ArrayLike
        Query data points (ns,). A 1-dimensional array of floats representing the query points at which
        the interpolation is to be evaluated.

    Returns
    -------
    np.ndarray
        Interpolated data points (ns, n). A 2-dimensional array of floats containing the interpolated
        data points for each query in `s`.

    Raises
    ------
    TypeError
        If the input vectors `y1` and `y2` do not have the same size.
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


def slerp(Q1: QuaternionType, Q2: QuaternionType, s: ArrayLike, short: bool = True) -> QuaternionsType:
    """
    Interpolate unit quaternions with spherical linear interpolation.

    This function interpolates between two unit quaternions ``Q1`` and ``Q2``
    at the normalized interpolation points specified by ``s``.

    Parameters
    ----------
    Q1 : QuaternionType
        Initial quaternion ``(4,)``.
    Q2 : QuaternionType
        Final quaternion ``(4,)``.
    s : ArrayLike
        Query interpolation points ``(n,)``.
    short : bool, optional
        If ``True``, use the shortest rotation path. If ``False``, use the
        longer path. Default is ``True``.

    Returns
    -------
    QuaternionsType
        Interpolated quaternions with shape ``(n, 4)``.

    Notes
    -----
    SLERP preserves unit length and provides smooth interpolation on the
    quaternion sphere. When the two quaternions are nearly identical, the
    function falls back to a numerically safe limit case.
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
    return np.squeeze(Q)


def qspline(Q: QuaternionsType, s: ArrayLike, mode: str) -> QuaternionsType:
    """
    Spline interpolation of N quaternions in the spherical space of SO(3).

    This function computes spline interpolation between quaternions given as input, in the space of SO(3) (3D rotation group).
    The interpolation is performed using either Hermite cubic or Squad interpolation, based on the specified mode.

    Parameters
    ----------
    Q : QuaternionsType
        Quaternion array of shape (n, 4), where each row represents a quaternion (4 elements).
    s : ArrayLike
        Path parameters as a numpy array of shape (m,). These parameters define the interpolation points between [0..1].
    mode : str
        Mode of spline interpolation. Can be 'hermite_cubic' or 'squad'. Default is 'squad'.

    Returns
    -------
    QuaternionsType
        Interpolated quaternions as a numpy array of shape (m, 4).

    Raises
    ------
    ValueError
        If quaternion vector 'Q' does not have 4 columns or if path parameters 's' are not in the range [0, 1].
    TypeError
        If input quaternion vectors are not of the same size.
    """

    def _get_intermediate_control_point(j: int, q: np.ndarray, dir_flip: bool) -> np.ndarray:
        """Compute an intermediate spline control point.

        Calculate intermediate control point for spline interpolation."""
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

    def _eval_cumulative_berstein_basis(s: float, i: int, order: int) -> float:
        """Evaluate Bernstein basis for cumulative interpolation."""
        N = order
        beta = 0
        for j in range(i, N + 1):
            term1 = math.comb(N, j)
            term2 = (1 - s) ** (N - j)
            term3 = s**j
            beta += term1 * term2 * term3
        return beta

    def _eval_alpha(s: float, i: int, L: int) -> float:
        """Evaluate alpha parameter for spline interpolation."""
        k = s * (L - 1) + 1
        if i < k:
            return 1
        elif i > k and i < k + 1:
            return k - (i - 1)
        else:
            return 0

    Q = np.asarray(Q, dtype="float")
    if Q.shape[1] != 4:
        raise ValueError("Quaternion vector 'Q' must have 4 columns.")

    if not np.all((s >= 0) & (s <= 1)) or not np.all(np.diff(s) >= 0):
        raise ValueError("Path parameters 's' must be in the range [0, 1] and in increasing order.")

    n = Q.shape[0]
    m = len(s)
    order = 3

    for j in range(1, n):
        C = np.dot(Q[j - 1], Q[j])
        if C < 0:
            Q[j] = -Q[j]

    qout = np.empty((m, 4))

    for i in range(m):
        si = s[i]
        qout[i] = Q[0]

        if si != 0 and si != 1:
            val = Q[0]
            EPS = 1e-9

            for j in range(1, n):
                alpha = _eval_alpha(si, j + 1, n)
                t = alpha
                if alpha > 0:
                    C = np.dot(Q[j - 1], Q[j])

                    if np.abs(1 - C) <= EPS:
                        val = (1 - si) * Q[j - 1] + si * Q[j]
                        val = qnormalize(val)
                    elif np.abs(1 + C) <= EPS:
                        qtemp = np.array([Q[j, 3], -Q[j, 2], Q[j, 1], -Q[j, 0]])
                        qtemp_array = np.copy(Q)
                        qtemp_array[j] = qtemp

                        if mode == "hermite_cubic":
                            qi = qinv(Q[j - 1])
                            qa = _get_intermediate_control_point(j - 1, qtemp_array, 0)
                            qap1 = _get_intermediate_control_point(j, qtemp_array, 1)
                            qai = qinv(qa)
                            qap1i = qinv(qap1)
                            qiqa = qmtimes(qi, qa)
                            qaiqap1 = qmtimes(qai, qap1)
                            qap1iqp1 = qmtimes(qap1i, Q[j])
                            omega1 = qlog(qiqa)
                            omega2 = qlog(qaiqap1)
                            omega3 = qlog(qap1iqp1)
                            beta1 = _eval_cumulative_berstein_basis(t, 1, order)
                            beta2 = _eval_cumulative_berstein_basis(t, 2, order)
                            beta3 = _eval_cumulative_berstein_basis(t, 3, order)
                            val = qmtimes(Q[j - 1], qexp(omega1 * beta1))
                            val = qmtimes(val, qexp(omega2 * beta2))
                            val = qmtimes(val, qexp(omega3 * beta3))
                        elif mode == "squad":
                            qa = _get_intermediate_control_point(j - 1, qtemp_array, 0)
                            qap1 = _get_intermediate_control_point(j, qtemp_array, 0)
                            qtemp1 = slerp(Q[j - 1], qtemp, t)
                            qtemp2 = slerp(qa, qap1, t)
                            squad = slerp(qtemp1, qtemp2, 2 * t * (1 - t))
                            val = squad

                    else:
                        if mode == "hermite_cubic":
                            qi = qinv(Q[j - 1])
                            qa = _get_intermediate_control_point(j - 1, Q, 0)
                            qap1 = _get_intermediate_control_point(j, Q, 1)
                            qai = qinv(qa)
                            qap1i = qinv(qap1)
                            qiqa = qmtimes(qi, qa)
                            qaiqap1 = qmtimes(qai, qap1)
                            qap1iqp1 = qmtimes(qap1i, Q[j])
                            omega1 = qlog(qiqa)
                            omega2 = qlog(qaiqap1)
                            omega3 = qlog(qap1iqp1)
                            beta1 = _eval_cumulative_berstein_basis(t, 1, order)
                            beta2 = _eval_cumulative_berstein_basis(t, 2, order)
                            beta3 = _eval_cumulative_berstein_basis(t, 3, order)
                            val = qmtimes(Q[j - 1], qexp(omega1 * beta1))
                            val = qmtimes(val, qexp(omega2 * beta2))
                            val = qmtimes(val, qexp(omega3 * beta3))
                            val = qnormalize(val)
                        elif mode == "squad":
                            qa = _get_intermediate_control_point(j - 1, Q, 0)
                            qap1 = _get_intermediate_control_point(j, Q, 0)
                            qtemp1 = slerp(Q[j - 1], Q[j], t)
                            qtemp2 = slerp(qa, qap1, t)
                            squad = slerp(qtemp1, qtemp2, 2 * t * (1 - t))
                            val = squad

                qout[i] = qnormalize(val)

    return qout


def qinterp(Q1: QuaternionType, Q2: QuaternionType, s: ArrayLike, short: bool = True) -> QuaternionsType:
    """
    Spherical Linear Interpolation (SLERP) of unit quaternion-like arrays.

    This function returns interpolated quaternion data points between two quaternions `Q1` and `Q2`
    using Spherical Linear Interpolation (SLERP), which is useful for smoothly interpolating rotations in 3D space.

    Parameters
    ----------
    Q1 : QuaternionType
        The initial quaternion (4,). A unit quaternion representing the starting rotation.
    Q2 : QuaternionType
        The final quaternion (4,). A unit quaternion representing the ending rotation.
    s : ArrayLike
        Query data points (n,). These values range from 0 to 1 and define the interpolation progression between `Q1` and `Q2`.
    short : bool, optional
        If True (default), the shortest rotation path will be chosen. If False, the longer path is used for interpolation.

    Returns
    -------
    QuaternionsType
        Interpolated quaternions at the requested `s` values, with shape (n, 4), where `n` is the length of the `s` array.
        Each row represents an interpolated quaternion.

    Notes
    -----
    SLERP provides a smooth, constant velocity interpolation between two unit quaternions and is commonly used for smooth rotation transitions.
    """
    return slerp(Q1, Q2, s, short=short)


def rinterp(R1: RotationMatrixType, R2: RotationMatrixType, s: ArrayLike, short: bool = True) -> RotationMatricesType:
    """
    Spherical Linear Interpolation (SLERP) of rotation matrices.

    This function interpolates between two rotation matrices `R1` and `R2` using Spherical Linear Interpolation (SLERP),
    by first converting the rotation matrices into quaternions, performing the interpolation, and then converting back to rotation matrices.

    Parameters
    ----------
    R1 : RotationMatrixType
        The initial rotation matrix (3, 3). A 3x3 matrix representing the starting rotation.
    R2 : RotationMatrixType
        The final rotation matrix (3, 3). A 3x3 matrix representing the ending rotation.
    s : ArrayLike
        Query data points (n,). These values range from 0 to 1 and define the interpolation progression between `R1` and `R2`.
    short : bool, optional
        If True (default), the shortest rotation path will be chosen. If False, the longer path is used for interpolation.

    Returns
    -------
    RotationMatricesType
        Interpolated rotation matrices at the requested `s` values, with shape (n, 3, 3), where `n` is the length of the `s` array.
        Each matrix represents an interpolated rotation.

    Notes
    -----
    SLERP provides a smooth, constant velocity interpolation between two quaternions, which is then mapped back to rotation matrices.
    The function internally converts the rotation matrices to quaternions, performs SLERP, and converts back to matrices.

    This method ensures that the interpolation is geometrically correct and smooth for 3D rotations.
    """
    return q2r(slerp(r2q(R1), r2q(R2), s, short=short))


def xinterp(x1: Pose3DType, x2: Pose3DType, s: ArrayLike, short: bool = True) -> Poses3DType:
    """
    Linear interpolation of spatial poses (SE3).

    This function interpolates between two spatial poses `x1` and `x2` using linear interpolation (LERP) for the positions
    and spherical linear interpolation (SLERP) for the rotations. Spatial poses are represented as arrays of 3 positions and
    4 quaternion elements (position + orientation).

    Parameters
    ----------
    x1 : Pose3DType
        Initial Cartesian pose (7,). The first three elements represent the position, and the last four elements represent the rotation as a quaternion.
    x2 : Pose3DType
        Final Cartesian pose (7,). The first three elements represent the position, and the last four elements represent the rotation as a quaternion.
    s : ArrayLike
        Query data points (n,). These values range from 0 to 1 and define the interpolation progression between `x1` and `x2`.
    short : bool, optional
        If True (default), the shortest rotation path will be chosen for SLERP. If False, the longer path is used.

    Returns
    -------
    Poses3DType
        Interpolated Cartesian poses (n, 7), where each row represents an interpolated pose: [position, quaternion].

    Notes
    -----
    - The positions are interpolated using linear interpolation (LERP).
    - The rotations (quaternions) are interpolated using spherical linear interpolation (SLERP).
    - The resulting interpolated poses combine the position and rotation at each query point `s`.

    Example
    --------
    # Example usage of the function:
    x1 = np.array([1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0])  # Initial pose (position + quaternion)
    x2 = np.array([4.0, 5.0, 6.0, 0.0, 0.7071, 0.7071, 0.0])  # Final pose (position + quaternion)
    s = np.linspace(0, 1, 10)  # 10 query points for interpolation
    interpolated_poses = xinterp(x1, x2, s)
    print(interpolated_poses)
    """
    x1 = vector(x1, dim=7)
    x2 = vector(x2, dim=7)
    s = vector(s)
    p = interp(x1[:3], x2[:3], s)
    Q = qinterp(x1[3:], x2[3:], s, short=short)
    return np.hstack((p, Q))


def tinterp(T1: HomogeneousMatrixType, T2: HomogeneousMatrixType, s: ArrayLike, short: bool = True) -> HomogeneousMatricesType:
    """
    Linear interpolation of spatial poses (SE3) represented as homogeneous transformation matrices.

    This function interpolates between two spatial poses `T1` and `T2`, which are represented as 4x4 homogeneous matrices.
    The positions are interpolated using linear interpolation (LERP), and the rotations are interpolated using spherical linear interpolation (SLERP).

    Parameters
    ----------
    T1 : HomogeneousMatrixType
        Initial Cartesian pose represented as a homogeneous transformation matrix (4, 4).
    T2 : HomogeneousMatrixType
        Final Cartesian pose represented as a homogeneous transformation matrix (4, 4).
    s : ArrayLike
        Query data points (ns,). These values range from 0 to 1 and define the interpolation progression between `T1` and `T2`.
    short : bool, optional
        If True (default), the shortest rotation path will be chosen for SLERP. If False, the longer path is used.

    Returns
    -------
    HomogeneousMatricesType
        Interpolated Cartesian poses (ns, 4, 4), where each pose is a 4x4 homogeneous transformation matrix.

    Notes
    -----
    - The translation part of the transformation matrix is interpolated using linear interpolation (LERP).
    - The rotation part of the transformation matrix is interpolated using spherical linear interpolation (SLERP) for quaternions.
    - The resulting interpolated poses combine the interpolated translation and rotation for each query point `s`.

    Example
    --------
    # Example usage of the function:
    T1 = np.eye(4)  # Initial pose (identity matrix as an example)
    T2 = np.eye(4)  # Final pose (identity matrix as an example)
    s = np.linspace(0, 1, 10)  # 10 query points for interpolation
    interpolated_poses = tinterp(T1, T2, s)
    print(interpolated_poses)
    """
    return x2t(xinterp(t2x(T1), t2x(T2), s, short=short))


def xarcinterp(x1: Pose3DType, x2: Pose3DType, pC: Vector3DType, s: ArrayLike, short: bool = True) -> Poses3DType:
    """
    Linear interpolation of spatial poses (SE3) along an arc.

    This function interpolates between two spatial poses `x1` and `x2` along an arc with a given center point `pC`.
    The positions are interpolated along the arc using LERP (Linear Interpolation), and the rotations are interpolated using SLERP (Spherical Linear Interpolation) for quaternions.

    Parameters
    ----------
    x1 : Pose3DType
        Initial Cartesian pose represented as an array of 7 elements (3 positions and 4 quaternion elements).
    x2 : Pose3DType
        Final Cartesian pose represented as an array of 7 elements (3 positions and 4 quaternion elements).
    pC : Vector3DType
        Arc center position, an array of 3 elements representing the center of the arc.
    s : ArrayLike
        Query data points (s,). These values range from 0 to 1 and define the interpolation progression between `x1` and `x2`.
    short : bool, optional
        If True (default), the shortest rotation path will be chosen for SLERP. If False, the longer path is used.

    Returns
    -------
    Poses3DType
        Interpolated Cartesian poses along the arc (n, 7), where each pose is a combination of interpolated position and rotation.

    Notes
    -----
    - The translation part of the transformation is interpolated along the arc using LERP.
    - The rotation part of the transformation is interpolated using SLERP for quaternions.
    - The resulting interpolated poses combine the interpolated translations and rotations for each query point `s`.

    Example
    --------
    # Example usage of the function:
    x1 = np.array([0, 0, 0, 1, 0, 0, 0])  # Initial pose (position and quaternion)
    x2 = np.array([1, 1, 1, 0, 1, 0, 0])  # Final pose (position and quaternion)
    pC = np.array([0.5, 0.5, 0.5])  # Arc center position
    s = np.linspace(0, 1, 10)  # 10 query points for interpolation
    interpolated_poses = xarcinterp(x1, x2, pC, s)
    print(interpolated_poses)
    """
    x1 = vector(x1, dim=7)
    x2 = vector(x2, dim=7)
    pC = vector(pC, dim=3)
    s = vector(s)
    p = [arc(x1[:3], x2[:3], pC, ss, short=short) for ss in s]
    Q = qinterp(x1[3:], x2[3:], s, short=short)
    return np.hstack((p, Q))


def interp1(s: ArrayLike, y: ArrayLike, si: ArrayLike) -> np.ndarray:
    """
    Wrapper for SciPy's `interp1d` function to perform 1D interpolation.

    This function interpolates data points `y` at query points `s` and returns
    the interpolated values at new query points `si`.

    Parameters
    ----------
    s : ArrayLike
        for which the values in `y` are provided.
    y : ArrayLike
        to the query points `s`.
    si : ArrayLike
        the new x-coordinates where you want to interpolate the corresponding values.

    Returns
    -------
    np.ndarray
        Interpolated data points at the new query points (ni, n). The shape of the result
        matches the shape of `si`, with interpolated values for each corresponding query point.

    Example
    --------
    # Example usage of the function:
    s = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    si = np.array([1.5, 2.5, 3.5])
    interpolated_values = interp1(s, y, si)
    print(interpolated_values)  # Output: array([2.25, 6.25, 12.25])
    """
    f = spi.interp1d(s, y, axis=0, fill_value="extrapolate")
    return f(si)


def interpPath(s: ArrayLike, path: ArrayLike, squery: ArrayLike) -> np.ndarray:
    """
    Interpolate path for query path values.

    This function interpolates the path data for given query points using `interp1` for linear interpolation.
    The path data is assumed to be defined for a path parameter `s`, and the query points are provided
    in `squery`. The function will return the path values at the query points.

    Parameters
    ----------
    s : ArrayLike
        Path parameter (ns,). The path parameter defines the progression along the path.
    path : ArrayLike
        Path data (ns, n). Each row corresponds to a data point along the path.
    squery : ArrayLike
        Query path points (ni,). These are the values at which interpolation is performed.

    Returns
    -------
    np.ndarray
        Path values at query points (ni, n). These are the path values interpolated at the specified query
        path points.

    Raises
    ------
    TypeError
        If `s` and `path` do not have the same first dimension.
    """
    s = vector(s)
    path = np.array(path)
    if (ismatrix(path) and path.shape[0] == len(s)) or isvector(path, dim=len(s)):
        return interp1(s, path, squery)
    else:
        raise TypeError(f"s and path must have same first dimension, but have shapes {s.shape} and {path.shape}")


def interpQuaternionPath(s: ArrayLike, path: QuaternionsType, squery: ArrayLike, short: bool = True) -> QuaternionsType:
    """
    Interpolate quaternion path for query path values using spherical linear interpolation (SLERP).

    This function returns interpolated quaternion data points at the specified query path values
    based on a path of quaternions. Linear interpolation is performed sequentially between path points.

    Parameters
    ----------
    s : ArrayLike
        Path parameter (ns,). The path parameter defines the progression along the path.
    path : QuaternionsType
        Path quaternions (ns, 4). Each row corresponds to a quaternion representing a point along the path.
    squery : ArrayLike
        Query path points (ni,). These are the values at which interpolation is performed.
    short : bool, optional
        If True, the shortest rotation is taken between quaternions during interpolation. Default is True.

    Returns
    -------
    QuaternionsType
        Interpolated quaternions at query points (ni, 4). These are the quaternions corresponding to the query
        path points, interpolated between the path quaternions.

    Raises
    ------
    TypeError
        If the `path` does not have the expected shape of (ns, 4).
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


def interpCartesianPath(s: ArrayLike, path: Poses3DType, squery: ArrayLike, short: bool = True) -> Poses3DType:
    """
    Interpolate Cartesian path for query path values.

    This function performs linear interpolation for position values (using LERP)
    and quaternion interpolation (using SLERP) for rotations along a Cartesian path.

    Parameters
    ----------
    s : ArrayLike
        Path parameter (ns,). The parameter defining the path progression.
    path : Poses3DType
        Cartesian path poses (ns, 7). The Cartesian poses should be a matrix where each row
        corresponds to a spatial pose (3 positions and 4 quaternion elements).
    squery : ArrayLike
        Query path points (ni,). These are the values at which interpolation is performed.
    short : bool, optional
        If True, the shortest rotation (SLERP) will be used for quaternion interpolation.
        Defaults to True.

    Returns
    -------
    Poses3DType
        Cartesian path poses at query points (ni, 7). This is the interpolated path where
        each row contains the position and rotation (quaternion) at the respective query point.

    Raises
    ------
    TypeError
        If `path` does not have the expected shape.
    """
    s = vector(s)
    path = np.array(path)
    if not ismatrix(path, shape=(len(s), 7)):
        raise TypeError(f"Qpath must have dimension {(len(s), 7)}, but has {path.shape}")
    p = interpPath(s, path[:, :3], squery)
    Q = interpQuaternionPath(s, path[:, 3:], squery, short=short)
    return np.hstack((p, Q))


def pathauxpoints(pnt: ArrayLike, auxpoints: str = "absolute", auxdistance: ArrayLike = [0.1, 0.1], viapoints: bool = False) -> np.ndarray:
    """
    Generates auxiliary points for path points.

    This function generates additional points along a path for refinement. The auxiliary points can be created
    using either absolute or relative distances from the original path points. If specified, the function can
    also generate viapoints for more detailed path representation.

    Parameters
    ----------
    pnt : ArrayLike
        Waypoints for the path, which can be of shape (n, 3), (n, 7), or (n, 6). Each row represents a point in 3D,
        Cartesian coordinates (position and orientation as quaternion), or other representations of poses.

    auxpoints : str, optional
        Auxiliary points generation method. Options are:
            - "absolute": Use absolute distance between points
            - "relative": Use relative distance of path segments
            - "none" (default): No auxiliary points are added.

    auxdistance : ArrayLike, optional
        The distance of auxiliary points, default is [0.1, 0.1], where the first value is for position and
        the second value is for orientation (if applicable).

    viapoints : bool, optional
        Whether to include viapoints when generating auxiliary points, by default False.

    Returns
    -------
    np.ndarray
        Path with auxiliary points, shape (m, 7) or (m, 3), depending on the input dimensions. The returned path
        contains positions and optionally orientations (in quaternion form) for each interpolated point.

    Raises
    ------
    ValueError
        If the input parameters or shapes are incorrect or incompatible.

    Notes
    -----
    - The function supports both 3D paths and Cartesian paths with orientations (7D).
    - Auxiliary points can be generated based on the specified distance and method.
    - The function can include viapoints to ensure finer control of the path.
    """
    if auxpoints not in ["absolute", "relative", "none"]:
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


def pathoverpoints(pnt: ArrayLike, interp: str = "inner", order: int = 4, step: float = 0.02, n_points: int = 0, auxpoints: str = "none", auxdistance: list = [0.1, 0.1], viapoints: bool = False, natural: bool = False, normscale: list = [1, 1], plot: bool = False, ori_sel: list = [1, 2]) -> np.ndarray:
    """
    Generates a path over points using spline interpolation.

    This function takes a set of waypoints and generates a smooth path that interpolates
    between them using different methods such as cubic splines, radial basis functions (RBF),
    or uniform subdivision. It can also generate auxiliary points to refine the path and
    return the corresponding path parameterization.

    Parameters
    ----------
    pnt : np.ndarray
        Waypoints for the path, which can be of shape (n, 3), (n, 7), or (n, 6) where each row represents
        a point in 3D, Cartesian coordinates (with position and orientation as quaternion), or
        some other representation of poses.

    interp : str, optional
        Interpolation method. Options include:
            - "inner" (default): Spline by uniform subdivision
            - "spline": Cubic spline interpolation
            - "RBF": Radial basis function interpolation
            - "none": No interpolation

    order : int, optional
        The order of the inner spline, by default 4.

    step : float, optional
        The maximum difference in the path parameter, by default 0.02.

    n_points : int, optional
        The minimal number of path points. If 0, `step` is used for subdivision, by default 0.

    auxpoints : str, optional
        Auxiliary points generation method, used only for the "inner" interpolation method. Options are:
            - "absolute": Use absolute distance between points
            - "relative": Use relative distance of path segments
            - "none" (default): No auxiliary points are added.

    auxdistance : list, optional
        The distance of auxiliary points, default is [0.1, 0.1], where the first value is for position and
        the second value is for orientation (if applicable).

    viapoints : bool, optional
        Whether to include viapoints when auxiliary points are used, by default False.

    natural : bool, optional
        If True, makes the path parameter natural (i.e., uses path length for parameterization), by default False.

    normscale : list, optional
        Scaling factor for rotation norm, by default [1, 1], where the first value is for position and the second for rotation.

    plot : bool, optional
        If True, plots the generated path, by default False.

    ori_sel : list, optional
        Selection of quaternion angles for 2D plotting, by default [1, 2], which represents the first and second quaternion components.

    Returns
    -------
    np.ndarray
        Interpolated path (m, 7) or (m, 3), depending on the input dimensions. The returned path contains
        positions and optionally orientations (in quaternion form) for each interpolated point.

    np.ndarray
        Path parameter (m, ). The parameterization of the generated path.

    Raises
    ------
    ValueError
        If the input parameters or shapes are incorrect or incompatible.

    Notes
    -----
    - The function supports both 3D paths and Cartesian paths with orientations (7D).
    - When using "inner" interpolation, auxiliary points are generated between the given waypoints to refine the path.
    - The path parameterization can be adjusted to use either natural path lengths or custom intervals based on `step` or `n_points`.
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

                    y = yy[(k - 1) : m, :].copy()
                    n = m + 1 - k

            return y

    if order < 2:
        raise ValueError("Order must be >= 2")
    auxdistance = np.asarray(auxdistance, dtype="float").flatten()
    if not all(0 <= val <= 0.5 for val in auxdistance):
        raise ValueError("auxdistance values out of range")
    if interp not in ["inner", "spline", "RBF", "none"]:
        raise ValueError("Invalid selected interpolation")
    if auxpoints not in ["absolute", "relative", "none"]:
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
    # Cubic spline interpolation
    elif interp == "spline":
        _npts = xaux.shape[0]
        sp = spi.CubicSpline(np.arange(_npts), xaux)
        if n_points == 0:
            s2 = np.arange(0, _npts - 1, step / kpoints)
        else:
            s2 = np.linspace(0, _npts - 1, n_points * kpoints)
        xi = sp(s2)
    # RBF interpolation
    elif interp == "RBF":
        sp = pathlen(xaux, scale=normscale)
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
        xi = uniqueCartesianPath(xi)

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
    if not natural:
        si = np.linspace(0, 1, xi.shape[0])
    elif interp == "RBF":
        si = t
    else:
        si = pathlen(xi, scale=normscale)
        sid = np.diff(si)
        ff = np.where(sid == 0)[0]
        si = np.delete(si, ff)
        xi = np.delete(xi, ff, axis=0)
        sie = np.linspace(0, np.max(si), len(si) // kpoints)
        if nd == 7:
            xi = interpCartesianPath(si, xi, sie)
        else:
            xi = interpPath(si, xi, sie)
        si = pathlen(xi, scale=normscale)

    # Plots
    if plot:
        plotcpath(si, xi, points=points, auxpoints=xaux, ori_sel=ori_sel, fig_num="Generated path over points")

    return xi, si


def pathlen(path: ArrayLike, scale: Union[float, ArrayLike] = [1.0, 1.0], Cartesian: bool = True) -> np.ndarray:
    """
    Calculates the path length (natural path parameter) for a given path in m-dimensional space.

    This function computes the natural parameterization of a path based on its geometry. The
    path can either be in Cartesian space (with 3D positions and quaternion orientations)
    or general m-dimensional space. If Cartesian space is used, scaling factors are applied
    to the position and orientation components to compute the distance.

    Parameters
    ----------
    path : ArrayLike
        The path data, where each row represents a point in the path. The shape of the path
        should be (n, m), where n is the number of points, and m is the dimensionality of each point.
        For Cartesian paths, m should be 7 (3D position and 4D quaternion). For other types of paths,
        m can vary based on the path's structure.

    scale : Union[float, ArrayLike], optional
        Scaling factors for the position and orientation norms, by default [1.0, 1.0].
        If the path is in Cartesian space (7D), this parameter scales the position and orientation
        components differently. The first value scales the position (default is 1.0) and the second
        value scales the orientation (default is 1.0).

    Cartesian : bool, optional
        A flag to indicate if the path is in Cartesian space. If True, the function computes the path length
        considering both position and orientation using L2 norms for position and orientation differences.
        If False, the path is assumed to be in general m-dimensional space, and the function computes the path length
        by considering only the position differences.

    Returns
    -------
    np.ndarray
        The natural path parameter (n,), where each entry corresponds to the cumulative path length
        up to the respective point in the path.

    Raises
    ------
    ValueError
        If the path data is invalid (e.g., mismatched dimensions or unsupported path format).

    Notes
    -----
    - If the path is a Cartesian path (7D), the function computes the distance using both position and orientation.
    - The path length is calculated by accumulating the distances between consecutive points, considering the specified scaling.
    """
    path = rbs_type(path)
    if isscalar(scale):
        scale = [1, scale]

    m = path.shape[1]

    if m == 7 and Cartesian:
        dp = np.atleast_2d(np.diff(path[:, 0:3], axis=0))
        dq = np.atleast_2d(2 * qlog(qmtimes(path[1:, 3:7], qinv(path[:-1, 3:7]))))
        dx = (scale[0] * np.sum(dp**2, axis=1) + scale[1] * np.sum(dq**2, axis=1)) ** 0.5
    elif m == 4 and Cartesian:
        dq = np.atleast_2d(2 * qlog(qmtimes(path[1:, :], qinv(path[:-1, :]))))
        dx = np.sum(dq**2, axis=1) ** 0.5
    else:
        dp = np.atleast_2d(np.diff(path, axis=0))
        dx = np.sum(dp**2, axis=1) ** 0.5

    si = np.cumsum(np.concatenate(([0], np.abs(dx))))

    return si


def distance2path(x: ArrayLike, path: ArrayLike, s: ArrayLike, *args) -> Tuple[np.ndarray, float, float]:
    """
    Find the closest point on a given path and compute the distance to it.

    This function computes the closest point on a path to a given point `x`,
    along with the distance to that point and the path parameter associated with it.
    The path can either be represented in Cartesian coordinates or in a more general space.

    Parameters
    ----------
    x : ArrayLike
        (3,) for 3D points or (1, 7) for Cartesian points (position and quaternion).

    path : ArrayLike
        The path, represented as an array of points. The shape can be either (n, 3) for 3D points
        or (n, 7) for Cartesian points (position and quaternion).

    s : ArrayLike
        The path parameter associated with each point in the path (n, 1). These parameters typically
        represent the progression along the path.

    scale : float, optional
        A scaling factor for rotation norm, used only for Cartesian paths (when `path` has shape (n, 7)).
        Default is 1.0, meaning no scaling is applied to the orientation component.

    Returns
    -------
    np.ndarray
        The closest point on the path, either in 3D or Cartesian coordinates (3,) or (1, 7).

    float
        The distance to the closest point on the path.

    float
        The path parameter of the closest point.

    Raises
    ------
    ValueError
        If the input dimensions are incorrect or if `x` or `path` have incompatible shapes.

    Notes
    -----
    - The distance is calculated using the Euclidean norm for the position and orientation (if applicable).
    - For Cartesian paths (7D), both position and orientation are considered in the distance calculation.
    - The function handles 3D and Cartesian paths differently.
    """
    x = vector(x)
    s2 = path.shape
    if isvector(x, dim=3) and (x.size >= 3):
        path = path[:, :3]
        tmp1 = path - x
        tmp2 = np.linalg.norm(tmp1, axis=1)
        i = np.argmin(tmp2)
    elif isvector(x, dim=7) and (s2[1] == 7):
        if len(args) < 4:
            scale = 1.0
        else:
            scale = args[0]
        x = np.expand_dims(x, 0)
        x = np.repeat(x, s2[0], axis=0)
        tmp1 = xerr(path, x)
        tmp2 = np.linalg.norm(tmp1[:, :3], axis=1) + scale * np.linalg.norm(tmp1[:, 3:6], axis=1)
        i = np.argmin(tmp2)
    else:
        raise ValueError("Wrong input dimension")
    d = tmp2[i]
    px = path[i, :]
    return px, d, s[i]


if __name__ == "__main__":
    from robotblockset.transformations import map_pose, rot_x, rot_y, rot_z, rpy2q

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    t = np.linspace(0, 2, num=201)

    # # Joint trajectories
    q0 = np.array((0, 1, 6, -3))
    q1 = np.array((1, 2, 3, 4))
    q2 = np.array((2, 3, -5, 7))

    fig, ax = plt.subplots(3, 4, num="Joint trajectories using 'jline'", figsize=(12, 8))
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

    qt, qdt, qddt = jpoly(q1, q2, t)
    ax[0, 2].plot(t, qt)
    ax[0, 2].grid()
    ax[0, 2].set_title("Poly")
    ax[1, 2].plot(t, qdt)
    gqt = np.gradient(qt, t, axis=0)
    ax[1, 2].plot(t, gqt, "--")
    ax[1, 2].grid()
    ax[1, 2].set_title("Velocity")
    ax[2, 2].plot(t, qddt)
    ax[2, 2].grid()
    ax[2, 2].set_title("Acceleration")

    qt, qdt, qddt = jtraj(q1, q2, t, traj="Jerk", qd_max=10, qdd_max=10, jerk_max=100, resample=True)
    if qt.shape[0] > t.shape[0]:
        t = np.arange(qt.shape[0]) * (t[1] - t[0])
    ax[0, 3].plot(t, qt)
    ax[0, 3].grid()
    ax[0, 3].set_title("Jerk")
    ax[1, 3].plot(t, qdt)
    gqt = np.gradient(qt, t, axis=0)
    ax[1, 3].plot(t, gqt, "--")
    ax[1, 3].grid()
    ax[1, 3].set_title("Velocity")
    ax[2, 3].plot(t, qddt)
    ax[2, 3].grid()
    ax[2, 3].set_title("Acceleration")

    # Cartesian trajectories
    t = np.linspace(0, 2, num=201)
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
    xt3, xdt3, xddt3 = cjerk(x0, x1, t)
    fig, ax = plt.subplots(3, 2, num="Cartesian trajectories using 'ctrap', 'cline, 'cpoly', and 'cjerk'", figsize=(8, 8))
    ax[0, 0].plot(t, xt[:, :3], label="Trap")
    ax[0, 0].plot(t, xt1[:, :3], "--", label="Line")
    ax[0, 0].plot(t, xt2[:, :3], ":", label="Poly")
    ax[0, 0].plot(t, xt3[:, :3], "-.", label="Jerk")
    ax[0, 0].grid()
    ax[0, 0].set_title("$p$")
    ax[1, 0].plot(t, xdt[:, :3])
    ax[1, 0].plot(t, xdt1[:, :3], "--")
    ax[1, 0].plot(t, xdt2[:, :3], ":")
    ax[1, 0].plot(t, xdt3[:, :3], "-.")
    ax[1, 0].grid()
    ax[1, 0].set_title("$\\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].plot(t, xddt1[:, :3], "--")
    ax[2, 0].plot(t, xddt2[:, :3], ":")
    ax[2, 0].plot(t, xddt3[:, :3], "-.")
    ax[2, 0].grid()
    ax[2, 0].set_title("$\\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].plot(t, xt1[:, 3:], "--")
    ax[0, 1].plot(t, xt2[:, 3:], ":")
    ax[0, 1].plot(t, xt3[:, 3:], "-.")
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].plot(t, xdt1[:, 3:], "--")
    ax[1, 1].plot(t, xdt2[:, 3:], ":")
    ax[1, 1].plot(t, xdt3[:, 3:], "-.")
    ax[1, 1].grid()
    ax[1, 1].set_title("$\\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].plot(t, xddt1[:, 3:], "--")
    ax[2, 1].plot(t, xddt2[:, 3:], ":")
    ax[2, 1].plot(t, xddt3[:, 3:], "-.")
    ax[2, 1].grid()
    ax[2, 1].set_title("$\\dot\\omega$")

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
    ax[1, 0].set_title("$\\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].grid()
    ax[2, 0].set_title("$\\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].grid()
    ax[1, 1].set_title("$\\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].grid()
    ax[2, 1].set_title("$\\dot\\omega$")

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
    ax[1, 0].set_title("$\\dot p$")
    ax[2, 0].plot(t, xddt[:, :3])
    ax[2, 0].grid()
    ax[2, 0].set_title("$\\ddot p$")

    ax[0, 1].plot(t, xt[:, 3:])
    ax[0, 1].grid()
    ax[0, 1].set_title("$Q$")
    ax[1, 1].plot(t, xdt[:, 3:])
    ax[1, 1].grid()
    ax[1, 1].set_title("$\\omega$")
    ax[2, 1].plot(t, xddt[:, 3:])
    ax[2, 1].grid()
    ax[2, 1].set_title("$\\dot\\omega$")

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
