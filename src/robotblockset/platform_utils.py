"""Mobile platform helper utilities.

This module provides helper functions for reduced kinematics and fully local.
vector-field control of a nonholonomic mobile platform tracking a manipulator
TCP treated as a point (pointer). The reduced kinematic model uses only the
first five joints of a Franka Emika Panda arm to compute the translational TCP
position and the corresponding geometric Jacobian. These quantities are then
used to derive distance-to-workspace metrics and to generate smooth,
rate-limited linear and angular velocity commands for the mobile base directly
in its local frame, without relying on global localization or onboard sensing.

Key functionalities include:
- Reduced 5-DoF Panda forward kinematics for translational TCP position and Jacobian.
- Signed distance computations to spherical and sphere-or-cylinder workspace boundaries.
- Angle wrapping and smooth activation functions for rotational alignment.
- Rate limiting and exponential smoothing for stable velocity commands.
- Vector-field-based tracking for forward, hold, and retreat behaviors.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Tadej Petric.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from robotblockset.platforms import platform
from robotblockset.robot_models import kinmodel_panda_reduced_pos
from robotblockset.robots import robot
from robotblockset.transformations import x2t
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, RotationMatrixType, Vector2DType, Vector3DType

ReconfigureDebugType = Dict[str, Any]


def distance_to_sphere_or_cylinder(p: ArrayLike, h1: float, s1: float) -> float:
    """
    Compute the signed distance to a sphere-or-cylinder workspace boundary.

    The workspace is modeled as a sphere centered at ``[0, 0, h1]`` with
    radius ``s1`` for points above ``z = h1``, and as an infinite cylinder
    aligned with the ``z`` axis with the same radius for points at or below
    ``z = h1``.

    Parameters
    ----------
    p : ArrayLike
        Cartesian point ``[x, y, z]`` expressed in the workspace frame.
    h1 : float
        Height of the spherical cap center along the ``z`` axis.
    s1 : float
        Radius of both the sphere and the cylinder.

    Returns
    -------
    float
        Signed distance to the boundary. Positive values indicate that the
        point lies inside the modeled workspace, negative values indicate
        that it lies outside, and zero corresponds to the boundary.
    """
    p = np.asarray(p).flatten()

    x, y, z = p

    if z > h1:
        center = np.array([0.0, 0.0, h1])
        d = s1 - np.linalg.norm(p - center)
    else:
        r = np.linalg.norm([x, y])
        d = s1 - r

    return float(d)


def distance_to_sphere(p: ArrayLike, height: float, radius: float) -> float:
    """
    Compute the signed distance to a spherical workspace boundary.

    Parameters
    ----------
    p : ArrayLike
        Cartesian point ``[x, y, z]`` expressed in the workspace frame.
    height : float
        ``z`` coordinate of the sphere center.
    radius : float
        Sphere radius.

    Returns
    -------
    float
        Signed distance to the sphere surface. Positive values indicate that
        the point lies inside the sphere, negative values indicate that it
        lies outside, and zero corresponds to the surface.
    """
    p = np.asarray(p).flatten()
    center = np.array([0.0, 0.0, height])
    d = radius - np.linalg.norm(p - center)
    return float(d)


def wrap_to_pi(angle: float) -> float:
    """
    Wrap an angle to the interval ``[-pi, pi]``.

    Parameters
    ----------
    angle : float
        Input angle in radians.

    Returns
    -------
    float
        Angle wrapped to the range ``[-pi, pi]``.
    """
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def smoothstep(x: float, edge0: float, edge1: float) -> float:
    """
    Compute a smooth activation value between two thresholds.

    Parameters
    ----------
    x : float
        Input value.
    edge0 : float
        Lower transition bound.
    edge1 : float
        Upper transition bound.

    Returns
    -------
    float
        Smoothed value in the range ``[0, 1]`` with continuous first
        derivatives inside the transition region.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(x * x * (3 - 2 * x))


def limit_rate(value_prev: float, value_target: float, max_delta: float) -> float:
    """
    Limit the step-to-step change of a scalar signal.

    Parameters
    ----------
    value_prev : float
        Previously applied signal value.
    value_target : float
        Desired signal value for the current control step.
    max_delta : float
        Maximum allowed absolute change between consecutive values.

    Returns
    -------
    float
        Rate-limited signal value.
    """
    delta = value_target - value_prev
    delta_clipped = max(-max_delta, min(max_delta, delta))
    return float(value_prev + delta_clipped)


def check_panda_reach(robot: robot, **kwargs: Any) -> bool:
    """
    Check whether the reduced Panda TCP is close to the workspace boundary.

    A reduced 5-DoF Panda kinematic model is used to evaluate the TCP
    position and its signed distance to a spherical workspace limit.

    Parameters
    ----------
    robot : robot
        robot
        Robot instance with a valid joint configuration in ``robot.q``.
    **kwargs : dict
        Optional keyword arguments. Supported key:

        ``d_min`` : float, optional
            Minimum allowed signed distance to the workspace boundary.

    Returns
    -------
    bool
        ``True`` if the TCP is closer than ``d_min`` to the boundary,
        otherwise ``False``.
    """
    d_min = kwargs.get("d_min", 0.05)
    l0 = 0.333
    l1 = np.linalg.norm([0.316, 0.088])
    l2 = np.linalg.norm([0.384, 0.088])
    rp5dof, _, _ = kinmodel_panda_reduced_pos(robot.q, out="pR")
    d = distance_to_sphere(rp5dof, l0, l1 + l2)
    return bool(d < d_min)


def yaw_from_tcp_direction(R: RotationMatrixType, alpha: float, sign: float = 1.0) -> float:
    """
    Compute a world-frame yaw angle from a TCP-frame planar direction.

    Parameters
    ----------
    R : RotationMatrixType
        TCP rotation matrix expressed in the world frame.
    alpha : float
        Direction angle in the TCP ``xy`` plane in radians, where ``0``
        corresponds to ``+x`` and ``pi / 2`` to ``+y``.
    sign : float, optional
        Direction multiplier. Use ``1.0`` for the forward direction and
        ``-1.0`` for the opposite direction.

    Returns
    -------
    float
        World-frame yaw angle in radians corresponding to the transformed
        planar direction vector.
    """
    v_tcp = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    v_world = sign * (R @ v_tcp)
    return float(np.arctan2(v_world[1], v_world[0]))


def reconfigure_base_from_tcp(
    robot: robot,
    platform: platform,
    base_goal_mode: str = "reconfigure",
    yaw_axis: float = 0.0,
    task_DOF: ArrayLike = np.array([0, 0, 1, 1, 1, 0]),
    null_space_task: str = "JointLimit",
    Kns: float = 100,
    pos_err: float = -1,
    approach_dist: float = 0.05,
    return_debug: bool = False,
    motion_enable: bool = False,
    **kwargs: Any,
) -> Union[Tuple[Vector2DType, float], Tuple[Vector2DType, float, ReconfigureDebugType]]:
    """
    Compute a mobile-base target position and yaw from a TCP reconfiguration.

    The function evaluates an inverse-kinematics-based manipulator
    reconfiguration and derives a planar platform target that preserves the
    current TCP pose. Two goal-generation modes are supported:
    ``"reconfigure"``, which uses the relative transform between the current
    and reconfigured robot poses, and ``"tcp_direction"``, which places the
    base behind the TCP along a selected tool-frame direction.

    Parameters
    ----------
    robot : robot
        robot
        Manipulator instance attached to the platform.
    platform : platform
        Mobile platform to be repositioned.
    base_goal_mode : str, optional
        ``"reconfigure"`` and ``"tcp_direction"``.
    yaw_axis : float, optional
        Direction angle in the TCP ``xy`` plane, in radians, used when
        ``base_goal_mode`` is ``"tcp_direction"``.
    task_DOF : ArrayLike, optional
        solver.
    null_space_task : str, optional
        solver.
    Kns : float, optional
        Null-space gain passed to the inverse kinematics solver.
    pos_err : float, optional
        Position tolerance passed to the inverse kinematics solver.
    approach_dist : float, optional
        Final approach distance passed to ``platform.CMoveToLocation(...)``
        when ``motion_enable`` is enabled.
    return_debug : bool, optional
        If ``True``, return intermediate kinematic quantities together with
        the platform target.
    motion_enable : bool, optional
        If ``True``, execute the coordinated robot-platform motion
        immediately.
    **kwargs : dict
        Additional keyword arguments reserved for future compatibility.

    Returns
    -------
    tuple[Vector2DType, float] or tuple[Vector2DType, float, dict[str, Any]]
        Platform target position ``pt`` in the platform plane and target yaw.
        When ``return_debug`` is ``True``, the returned tuple also contains a
        dictionary with intermediate values used during the computation.
    """
    _ = kwargs
    pt: Vector2DType = np.zeros(2, dtype=float)
    p0: Optional[Vector3DType] = None
    v: Optional[Vector2DType] = None

    robot.ResetCurrentTarget()
    robot.GetState()
    platform.ResetCurrentTarget()
    platform.GetState()

    x0 = robot.GetPose(task_space="Robot")
    q0: JointConfigurationType = robot.q_ref.copy()
    qq = robot.IKin(x0, q0=robot.q_ref, task_space="Robot", task_DOF=task_DOF, null_space_task=null_space_task, Kns=Kns, pos_err=pos_err)[0]

    x1 = robot.Kinmodel(qq)[0]

    xx0: HomogeneousMatrixType = x2t(x0)
    xx1: HomogeneousMatrixType = x2t(x1)

    if base_goal_mode == "reconfigure":
        x_target = xx0 @ np.linalg.inv(xx1)

        p0 = xx0[:3, 3]
        pt = x_target[:2, 3]
        v = p0[:2] - pt
        v_xy_norm = np.hypot(v[0], v[1])
        if v_xy_norm < 1e-9:
            yaw = 0.0
        else:
            yaw = float(np.arctan2(v[1], v[0]))

    elif base_goal_mode == "tcp_direction":
        xy_dist_vec = xx1[:2, 3]
        behind_dist = np.linalg.norm(xy_dist_vec) - 0.1
        p0 = robot.GetPos(task_space="Robot")
        R = robot.GetOri(task_space="Robot", out="R")
        yaw = yaw_from_tcp_direction(R, yaw_axis)
        pt[0] = p0[0] - behind_dist * np.cos(yaw)
        pt[1] = p0[1] - behind_dist * np.sin(yaw)

    else:
        raise ValueError(f"Unsupported base_goal_mode: {base_goal_mode}")

    if motion_enable:
        platform.Message("Reconfiguration started", 1)
        robot.EEFixed = True
        robot.Message("TCP fixed", 2)

        platform.ResetCurrentTarget()
        robot.ResetCurrentTarget()

        xfixed = robot.x_ref.copy()

        platform.ResetTime()
        robot.ResetTime()

        robot.CMove(xfixed, 1000, asynchronous=True)

        platform.Wait(0.5)
        platform.CMoveToLocation(pt[:2], yaw, allow_backward=True, task_space="Platform", reach_check_fn=check_panda_reach, d_min=0.1, approach_dist=approach_dist, final_orientation_correction=True, pos_err=0.01, vel_fac=0.2)
        platform.Wait(1)

        robot.Abort()
        robot.GetState()
        robot.EEFixed = False
        robot.Message("TCP released", 2)
        platform.Message("Reconfiguration finished", 1)

    if return_debug:
        debug: ReconfigureDebugType = {
            "x0": x0,
            "x1": x1,
            "q0": q0,
            "qq": qq,
            "xx0": xx0,
            "xx1": xx1,
            "p0": p0,
            "pt": pt,
            "v": v,
        }
        return pt, yaw, debug

    return pt, yaw


def track_panda_vf(
    platform: platform,
    K_v: float = 2.0,
    K_omega: float = 1.0,
    K_back: float = 2,
    a_max: float = 1,
    w_max: float = 1,
    angle_thresh_deg_low: float = 30.0,
    angle_thresh_deg_high: float = 45.0,
    theta_retreat_target_deg: float = 30.0,
    alpha: float = 0.25,
    max_omega: float = 1.0,
    f_zone: float = 0.15,
    b_zone: float = 0.15,
    l0: float = 0.333,
    l1: float = np.linalg.norm([0.316, 0.088]),
    l2: float = np.linalg.norm([0.384, 0.088]),
) -> int:
    """
    Run a local vector-field controller for a nonholonomic mobile platform.

    This controller drives the platform from the relative position of the
    robot TCP only. The TCP is modeled as a point obtained from the reduced
    5-DoF Panda kinematics. No global localization or environment sensing is
    required.

    Parameters
    ----------
    platform : platform
        Mobile platform instance providing access to platform state, robot
        state, sample time, and velocity commands.
    K_v : float, optional
        region.
    K_omega : float, optional
        Gain for angular velocity used to align the platform with the TCP
        direction.
    K_back : float, optional
        Gain for backward velocity during retreat behavior.
    a_max : float, optional
        Maximum linear acceleration.
    w_max : float, optional
        Maximum angular acceleration.
    angle_thresh_deg_low : float, optional
        Lower angular threshold in degrees below which rotation is suppressed.
    angle_thresh_deg_high : float, optional
        activated.
    theta_retreat_target_deg : float, optional
        Desired orientation offset in degrees during retreat behavior.
    alpha : float, optional
        Exponential smoothing factor for the commanded velocities.
    max_omega : float, optional
        Maximum allowed absolute angular velocity.
    f_zone : float, optional
        Forward activation threshold near the outer workspace boundary.
    b_zone : float, optional
        Retreat activation threshold near the inner safety zone.
    l0 : float, optional
        Base-to-first-joint distance used in the reduced workspace model.
    l1 : float, optional
        Effective upper-arm length in the reduced workspace model.
    l2 : float, optional
        Effective forearm length in the reduced workspace model.

    Returns
    -------
    int
        Status code indicating the termination condition. ``0`` denotes an
        abort request from the platform, while ``2`` indicates that no robot
        is attached to the platform.
    """
    v_prev = 0.0
    w_prev = 0.0

    angle_thresh_low = np.deg2rad(angle_thresh_deg_low)
    angle_thresh_high = np.deg2rad(angle_thresh_deg_high)

    in_retreat_mode = False

    while True:
        if platform._abort_autonomous_motion:
            return 0

        if platform.Robot is None:
            platform.Message("No robot attached to platform!", 1)
            return 2

        dt = platform.tsamp
        qq = platform.Robot.q
        rp5dof, _, _ = kinmodel_panda_reduced_pos(qq, out="pR")
        dmax = distance_to_sphere(rp5dof, l0, l1 + l2)
        dmin = -distance_to_sphere_or_cylinder(rp5dof, l0, l0)

        rp = platform.Robot.GetPos(task_space="Robot")[:3]
        robot_rel = rp[:3]
        angle_to_tcp = np.arctan2(robot_rel[1], robot_rel[0])
        theta_error = wrap_to_pi(angle_to_tcp)
        rotation_activation = smoothstep(abs(theta_error), angle_thresh_low, angle_thresh_high)
        w_cmd_temp = K_omega * np.sign(theta_error) * rotation_activation

        if not in_retreat_mode and dmin < b_zone:
            in_retreat_mode = True
            theta_retreat_target = -np.sign(theta_error) * np.deg2rad(theta_retreat_target_deg)

        elif in_retreat_mode and dmin > b_zone:
            in_retreat_mode = False

        if in_retreat_mode:
            theta_error_retreat = wrap_to_pi(theta_retreat_target - theta_error)
            rotation_distance_activation = smoothstep(b_zone - dmin, 0.0, 0.1)
            w_cmd = -K_omega * theta_error_retreat * rotation_distance_activation
            v_cmd = -K_back * (b_zone - dmin)

        elif dmax < f_zone:
            v_cmd = K_v * (f_zone - dmax)
            w_cmd = w_cmd_temp
        else:
            v_cmd = 0.0
            w_cmd = w_cmd_temp

        v_target = (1 - alpha) * v_prev + alpha * v_cmd
        w_target = (1 - alpha) * w_prev + alpha * w_cmd

        v_smooth = limit_rate(v_prev, v_target, a_max * dt)
        w_smooth = limit_rate(w_prev, w_target, w_max * dt)
        w_smooth = np.clip(w_smooth, -max_omega, max_omega)

        platform.Set_vel([v_smooth, w_smooth])
        v_prev = v_smooth
        w_prev = w_smooth

        platform.GetState()
        platform.Update()
