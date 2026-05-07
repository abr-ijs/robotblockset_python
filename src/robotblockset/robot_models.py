"""Robot models.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Optional, Tuple, Union

from robotblockset.transformations import map_pose
from robotblockset.rbs_typing import TCPType
from robotblockset.tools import search_valid_range

pi = np.pi


def invkin_model_panda_valid(self, x: np.ndarray, q7: float, q_initial: np.ndarray, tcp: TCPType, closest: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Check if the inverse kinematics solution is valid for the given end-effector pose and joint angle q7.

    if the joint angle q7 is within the valid range, it computes the inverse kinematics solution, otherwise it finds the closest valid joint configuration.

    Parameters
    ----------
    x : np.ndarray
        The end-effector pose as a 4x4 transformation matrix.
    q7 : float
        The joint angle for the 7th joint (end-effector rotation around the z-axis).
    q0 : np.ndarray
        The initial joint positions.
    tcp : TCPType
        The tool_center_point transformation matrix.
    closest : bool, optional
        If True, find the closest valid joint configuration. Default is True.

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple containing the valid joint configurations for the given end-effector pose and joint angle q7.
    """
    fun = lambda q7: invkin_model_panda(self, x, q7, q_initial, tcp)
    q7_range = search_valid_range(fun, self.q_min[-1], self.q_max[-1])
    if q7_range[0] <= q7 <= q7_range[1]:
        return invkin_model_panda(self, x, q7, q_initial, tcp)
    else:
        distance_to_min = abs(q7 - q7_range[0])
        distance_to_max = abs(q7 - q7_range[1])
    if distance_to_min < distance_to_max:
        return invkin_model_panda(self, x, q7_range[0], q_initial, tcp, closest=closest)
    else:
        return invkin_model_panda(self, x, q7_range[1], q_initial, tcp, closest=closest)


def invkin_model_panda(self, x: np.ndarray, q7: float, q_initial: np.ndarray, tcp: TCPType, closest: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Calculate the inverse kinematics solution the given end-effector pose and joint angle q7.

    Parameters
    ----------
    x : np.ndarray
        The end-effector pose as a 4x4 transformation matrix.
    q7 : float
        The joint angle for the 7th joint (end-effector rotation around the z-axis).
    q0 : np.ndarray
        The initial joint positions.
    tcp : TCPType
        The tool_center_point transformation matrix.
    closest : bool, optional
        If True, find the closest valid joint configuration. Default is True.

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple containing the valid joint configurations for the given end-effector pose and joint angle q7.
    """
    # Initialize variables
    q_all = np.full((4, 7), np.nan)

    x = self.spatial(x)  # Convert to spatial vector if not already
    T_flange = x @ np.linalg.inv(tcp)

    # DH parameters
    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d_flange = 0.1070
    a4 = 0.0825
    a7 = 0.0880

    # pre-compute some constants
    LL24 = 0.10666225  # a4^2 + d3^2
    LL46 = 0.15426225  # a4^2 + d5^2
    L24 = 0.326591870689  # sqrt(LL24)
    L46 = 0.392762332715  # sqrt(LL46)

    thetaH46 = 1.35916951803  # atan(d5/a4)
    theta342 = 1.31542071191  # atan(d3/a4)
    theta46H = 0.211626808766  # acot(d5/a4)

    if q7 <= self.q_min[6] or q7 >= self.q_max[6]:
        return None

    q_all[:, 6] = q7

    # compute p_6
    R_EE = T_flange[:3, :3]
    z_EE = T_flange[:3, 2]
    p_EE = T_flange[:3, 3]
    p_7 = p_EE - (d_flange * z_EE)

    x_EE_6 = np.array([np.cos(q7), -np.sin(q7), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 = x_6 / np.linalg.norm(x_6)
    p_6 = p_7 - (a7 * x_6)

    # compute q4
    p_2 = np.array([0, 0, d1])
    V26 = p_6 - p_2

    LL26 = (V26[0] * V26[0]) + (V26[1] * V26[1]) + (V26[2] * V26[2])
    L26 = np.sqrt(LL26)

    if ((L24 + L46) < L26) or ((L24 + L26) < L46) or (L26 + L46) < L24:
        return None

    theta246 = np.arccos((LL24 + LL46 - LL26) / 2.0 / L24 / L46)
    q4 = theta246 + thetaH46 + theta342 - (2.0 * np.pi)

    if q4 <= self.q_min[3] or q4 >= self.q_max[3]:
        return None

    q_all[:, 3] = q4

    # compute q6
    _tmp = (LL26 + LL46 - LL24) / 2.0 / L26 / L46
    if np.abs(_tmp) > 1.0:
        return None  # Avoid NaN from arcsin if out of bounds
    theta462 = np.arccos(_tmp)
    theta26H = theta46H + theta462
    D26 = -(L26 * np.cos(theta26H))

    Z_6 = np.cross(z_EE, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.empty((3, 3), dtype=np.float64)
    R_6[:, 0] = x_6.copy()
    R_6[:, 1] = Y_6 / np.linalg.norm(Y_6)
    R_6[:, 2] = Z_6 / np.linalg.norm(Z_6)

    V_6_62 = R_6.T @ (-V26)
    Phi6 = np.arctan2(V_6_62[1], V_6_62[0])
    _tmp = D26 / np.sqrt((V_6_62[0] * V_6_62[0]) + (V_6_62[1] * V_6_62[1]))
    if np.abs(_tmp) > 1.0:
        return None  # Avoid NaN from arcsin if out of bounds
    Theta6 = np.arcsin(_tmp)

    q6 = np.zeros(2)
    q6[0] = np.pi - Theta6 - Phi6
    q6[1] = Theta6 - Phi6

    for idx, elem in enumerate(q6):
        if q6[idx] <= self.q_min[5]:
            q6[idx] += 2.0 * np.pi
        elif q6[idx] >= self.q_max[5]:
            q6[idx] -= 2.0 * np.pi

        if q6[idx] <= self.q_min[5] or q6[idx] >= self.q_max[5]:
            # q6 is out of bounds
            q_all[2 * idx] = np.full(7, np.nan)
            q_all[2 * idx + 1] = np.full(7, np.nan)
        else:
            # q6 is within bounds
            q_all[2 * idx, 5] = q6[idx]
            q_all[2 * idx + 1, 5] = q6[idx]

    if np.isnan(q_all[0, 5]) and np.isnan(q_all[2, 5]):
        return None

    # compute q1 & q2
    thetaP26 = (3.0 * np.pi / 2) - theta462 - theta246 - theta342
    thetaP = np.pi - thetaP26 - theta26H
    LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

    z_5_all = np.zeros((4, 3))
    V2P_all = np.zeros((4, 3))

    for idx, elem in enumerate(q6):
        z_6_5 = np.array([np.sin(elem), np.cos(elem), 0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - (LP6 * z_5) - p_2

        z_5_all[2 * idx] = z_5.copy()
        z_5_all[2 * idx + 1] = z_5.copy()
        V2P_all[2 * idx] = V2P.copy()
        V2P_all[2 * idx + 1] = V2P.copy()

        L2P = np.linalg.norm(V2P)

        if np.fabs(V2P[2] / L2P) > 0.999:
            print("V2P is vertical, setting q1 and q2 to q0")
            q_all[2 * idx, 0] = q_initial[0]
            q_all[2 * idx, 1] = 0.0
            q_all[2 * idx + 1, 0] = q_initial[0]
            q_all[2 * idx + 1, 1] = 0.0

        else:
            q_all[2 * idx, 0] = np.arctan2(V2P[1], V2P[0])
            q_all[2 * idx, 1] = np.arccos(V2P[2] / L2P)
            if q_all[2 * idx, 0] < 0:
                q_all[2 * idx + 1, 0] = q_all[2 * idx, 0] + np.pi
            else:
                q_all[2 * idx + 1, 0] = q_all[2 * idx, 0] - np.pi
            q_all[2 * idx + 1, 1] = -q_all[2 * idx, 1]

    for idx in range(4):
        if q_all[idx, 0] <= self.q_min[0] or q_all[idx, 0] >= self.q_max[0] or q_all[idx, 1] <= self.q_min[1] or q_all[idx, 1] >= self.q_max[1]:
            q_all[idx] = np.full(7, np.nan)
            continue

        # compute q3
        z_3 = V2P_all[idx] / np.linalg.norm(V2P_all[idx])
        Y_3 = -np.cross(V26, V2P_all[idx])
        y_3 = Y_3 / np.linalg.norm(Y_3)
        x_3 = np.cross(y_3, z_3)
        c1 = np.cos(q_all[idx, 0])
        s1 = np.sin(q_all[idx, 0])

        R_1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])

        c2 = np.cos(q_all[idx, 1])
        s2 = np.sin(q_all[idx, 1])

        R_1_2 = np.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])

        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3

        q_all[idx, 2] = np.arctan2(x_2_3[2], x_2_3[0])

        # mask = (q_all[idx, 2] <= q_min[2]) | (q_all[idx, 2] >= q_max[2])
        # q_all[mask] = np.full(7, np.nan)
        if q_all[idx, 2] <= self.q_min[2] or q_all[idx, 2] >= self.q_max[2]:
            q_all[idx] = np.full(7, np.nan)
            continue

        # compute q5
        VH4 = p_2 + (d3 * z_3) + (a4 * x_3) - p_6 + (d5 * z_5_all[idx])
        c6 = np.cos(q_all[idx, 5])
        s6 = np.sin(q_all[idx, 5])
        R_5_6 = np.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])

        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4

        q_all[idx, 4] = -np.arctan2(V_5_H4[1], V_5_H4[0])
        if q_all[idx, 4] <= self.q_min[4] or q_all[idx, 4] >= self.q_max[4]:
            q_all[idx] = np.full(7, np.nan)
            continue

    # Copy the results to the output array
    valid_rows = ~np.isnan(q_all).any(axis=1)
    if np.any(valid_rows):
        result = q_all[valid_rows]
        if closest:
            # Find the closest solution to the initial joint positions
            closest_idx = np.argmin(np.linalg.norm(result - q_initial, axis=1))
            return result[closest_idx, :]
        return result
    else:
        return None


def kinmodel_panda_reduced_pos(q: np.ndarray, out: str = "x") -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Abstract method to compute the forward kinematics of the robot.

    This function implements a reduced 5-DoF kinematic model of the
    Franka Emika Panda manipulator, where only the translational motion
    of the Tool Center Point (TCP) is considered. The TCP is treated as
    a point (pointer), and the end-effector orientation is not used for
    control purposes. The model is intended for applications where the
    mobile base or higher-level controller depends solely on the TCP
    position, such as local vector-field-based base–manipulator control.

    Parameters
    ----------
    q : np.ndarray
        Joint angles of the first five Panda joints, used as input to the
        reduced kinematic model.
    out : str, optional
        Output format selector. Depending on the value, the function
        returns the TCP position or a homogeneous transformation with
        translation only. By default "x".

    Returns
    -------
    tuple
        The computed reduced kinematic output. This is typically the
        Cartesian position of the TCP (pointer) and, if requested,
        the associated geometric Jacobian with respect to the first
        five joints.
    """
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    c3, s3 = np.cos(q[2]), np.sin(q[2])
    c4, s4 = np.cos(q[3]), np.sin(q[3])
    c5, s5 = np.cos(q[4]), np.sin(q[4])

    a3 = 0.0825
    a4 = -0.0825

    d1 = 0.333
    d3 = 0.316
    d5 = 0.384

    p = np.zeros(3)
    p[0] = d5 * (s4 * (s1 * s3 - c1 * c2 * c3) + c1 * c4 * s2) + d3 * c1 * s2 - a3 * s1 * s3 - a4 * c4 * (s1 * s3 - c1 * c2 * c3) + a3 * c1 * c2 * c3 + a4 * c1 * s2 * s4
    p[1] = a3 * c1 * s3 - d5 * (s4 * (c1 * s3 + c2 * c3 * s1) - c4 * s1 * s2) + d3 * s1 * s2 + a4 * c4 * (c1 * s3 + c2 * c3 * s1) + a3 * c2 * c3 * s1 + a4 * s1 * s2 * s4
    p[2] = d1 + d5 * (c2 * c4 + c3 * s2 * s4) + d3 * c2 - a3 * c3 * s2 + a4 * c2 * s4 - a4 * c3 * c4 * s2

    R = np.zeros((3, 3))
    R[0, 0] = -c5 * (c4 * (s1 * s3 - c1 * c2 * c3) - c1 * s2 * s4)
    -s5 * (c3 * s1 + c1 * c2 * s3)
    R[0, 1] = s4 * (s1 * s3 - c1 * c2 * c3) + c1 * c4 * s2
    R[0, 2] = c5 * (c3 * s1 + c1 * c2 * s3)
    -s5 * (c4 * (s1 * s3 - c1 * c2 * c3) - c1 * s2 * s4)
    R[1, 0] = c5 * (c4 * (c1 * s3 + c2 * c3 * s1) + s1 * s2 * s4)
    +s5 * (c1 * c3 - c2 * s1 * s3)
    R[1, 1] = c4 * s1 * s2 - s4 * (c1 * s3 + c2 * c3 * s1)
    R[1, 2] = s5 * (c4 * (c1 * s3 + c2 * c3 * s1) + s1 * s2 * s4)
    -c5 * (c1 * c3 - c2 * s1 * s3)
    R[2, 0] = c5 * (c2 * s4 - c3 * c4 * s2) + s2 * s3 * s5
    R[2, 1] = c2 * c4 + c3 * s2 * s4
    R[2, 2] = s5 * (c2 * s4 - c3 * c4 * s2) - c5 * s2 * s3

    Jp = np.zeros((3, 5))
    Jp[0, 0] = d5 * (s4 * (c1 * s3 + c2 * c3 * s1) - c4 * s1 * s2) - a3 * c1 * s3 - d3 * s1 * s2 - a4 * c4 * (c1 * s3 + c2 * c3 * s1) - a3 * c2 * c3 * s1 - a4 * s1 * s2 * s4
    Jp[0, 1] = d5 * (c1 * c2 * c4 + c1 * c3 * s2 * s4) + d3 * c1 * c2 - a3 * c1 * c3 * s2 + a4 * c1 * c2 * s4 - a4 * c1 * c3 * c4 * s2
    Jp[0, 2] = d5 * s4 * (c3 * s1 + c1 * c2 * s3) - a3 * c3 * s1 - a4 * c4 * (c3 * s1 + c1 * c2 * s3) - a3 * c1 * c2 * s3
    Jp[0, 3] = d5 * (c4 * (s1 * s3 - c1 * c2 * c3) - c1 * s2 * s4) + a4 * s4 * (s1 * s3 - c1 * c2 * c3) + a4 * c1 * c4 * s2
    Jp[0, 4] = 0
    Jp[1, 0] = d5 * (s4 * (s1 * s3 - c1 * c2 * c3) + c1 * c4 * s2) + d3 * c1 * s2 - a3 * s1 * s3 - a4 * c4 * (s1 * s3 - c1 * c2 * c3) + a3 * c1 * c2 * c3 + a4 * c1 * s2 * s4
    Jp[1, 1] = d5 * (c2 * c4 * s1 + c3 * s1 * s2 * s4) + d3 * c2 * s1 - a3 * c3 * s1 * s2 + a4 * c2 * s1 * s4 - a4 * c3 * c4 * s1 * s2
    Jp[1, 2] = a3 * c1 * c3 - d5 * s4 * (c1 * c3 - c2 * s1 * s3) + a4 * c4 * (c1 * c3 - c2 * s1 * s3) - a3 * c2 * s1 * s3
    Jp[1, 3] = a4 * c4 * s1 * s2 - a4 * s4 * (c1 * s3 + c2 * c3 * s1) - d5 * (c4 * (c1 * s3 + c2 * c3 * s1) + s1 * s2 * s4)
    Jp[1, 4] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -d5 * (c4 * s2 - c2 * c3 * s4) - d3 * s2 - a3 * c2 * c3
    -a4 * s2 * s4 - a4 * c2 * c3 * c4
    Jp[2, 2] = a3 * s2 * s3 + a4 * c4 * s2 * s3 - d5 * s2 * s3 * s4
    Jp[2, 3] = a4 * c2 * c4 - d5 * (c2 * s4 - c3 * c4 * s2) + a4 * c3 * s2 * s4
    Jp[2, 4] = 0

    Jr = np.zeros((3, 5))
    Jr[0, 0] = 0
    Jr[0, 1] = -s1
    Jr[0, 2] = c1 * s2
    Jr[0, 3] = c3 * s1 + c1 * c2 * s3
    Jr[0, 4] = s4 * (s1 * s3 - c1 * c2 * c3) + c1 * c4 * s2
    Jr[1, 0] = 0
    Jr[1, 1] = c1
    Jr[1, 2] = s1 * s2
    Jr[1, 3] = c2 * s1 * s3 - c1 * c3
    Jr[1, 4] = c4 * s1 * s2 - s4 * (c1 * s3 + c2 * c3 * s1)
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c2
    Jr[2, 3] = -s2 * s3
    Jr[2, 4] = c2 * c4 + c3 * s2 * s4

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return rp2t(R=R, p=p, out=out), J


def kinmodel_panda(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])
    c6 = np.cos(q[6])
    s6 = np.sin(q[6])

    a2 = 0.0825
    a3 = -0.0825
    a5 = 0.088

    d0 = 0.333
    d2 = 0.316
    d4 = 0.384
    d6 = 0.107

    p = np.zeros(3)
    p[0] = (
        -a2 * s0 * s2
        + a2 * c0 * c1 * c2
        + a3 * (-s0 * s2 + c0 * c1 * c2) * c3
        + a3 * s1 * s3 * c0
        + a5 * (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5
        + a5 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5
        + d2 * s1 * c0
        + d4 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3)
        + d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c5)
    )
    p[1] = (
        a2 * s0 * c1 * c2
        + a2 * s2 * c0
        + a3 * (s0 * c1 * c2 + s2 * c0) * c3
        + a3 * s0 * s1 * s3
        + a5 * (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5
        + a5 * (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5
        + d2 * s0 * s1
        + d4 * (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3)
        + d6 * ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * c5)
    )
    p[2] = -a2 * s1 * c2 - a3 * s1 * c2 * c3 + a3 * s3 * c1 + a5 * ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + a5 * (s1 * s3 * c2 + c1 * c3) * s5 + d0 + d2 * c1 + d4 * (s1 * s3 * c2 + c1 * c3) + d6 * (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (s1 * s3 * c2 + c1 * c3) * c5)
    R = np.zeros((3, 3))
    R[0, 0] = ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5) * c6 + (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 - (-s0 * c2 - s2 * c0 * c1) * c4) * s6
    R[0, 1] = -((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5) * s6 + (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 - (-s0 * c2 - s2 * c0 * c1) * c4) * c6
    R[0, 2] = (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c5
    R[1, 0] = ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5) * c6 + (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 - (-s0 * s2 * c1 + c0 * c2) * c4) * s6
    R[1, 1] = -((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5) * s6 + (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 - (-s0 * s2 * c1 + c0 * c2) * c4) * c6
    R[1, 2] = (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * c5
    R[2, 0] = (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (s1 * s3 * c2 + c1 * c3) * s5) * c6 + ((-s1 * c2 * c3 + s3 * c1) * s4 - s1 * s2 * c4) * s6
    R[2, 1] = -(((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (s1 * s3 * c2 + c1 * c3) * s5) * s6 + ((-s1 * c2 * c3 + s3 * c1) * s4 - s1 * s2 * c4) * c6
    R[2, 2] = ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (s1 * s3 * c2 + c1 * c3) * c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = (
        -a2 * s0 * c1 * c2
        - a2 * s2 * c0
        + a3 * (-s0 * c1 * c2 - s2 * c0) * c3
        - a3 * s0 * s1 * s3
        + a5 * (((-s0 * c1 * c2 - s2 * c0) * c3 - s0 * s1 * s3) * c4 + (s0 * s2 * c1 - c0 * c2) * s4) * c5
        + a5 * (-(-s0 * c1 * c2 - s2 * c0) * s3 - s0 * s1 * c3) * s5
        - d2 * s0 * s1
        + d4 * (-(-s0 * c1 * c2 - s2 * c0) * s3 - s0 * s1 * c3)
        + d6 * ((((-s0 * c1 * c2 - s2 * c0) * c3 - s0 * s1 * s3) * c4 + (s0 * s2 * c1 - c0 * c2) * s4) * s5 - (-(-s0 * c1 * c2 - s2 * c0) * s3 - s0 * s1 * c3) * c5)
    )
    Jp[0, 1] = (
        -a2 * s1 * c0 * c2
        - a3 * s1 * c0 * c2 * c3
        + a3 * s3 * c0 * c1
        + a5 * ((-s1 * c0 * c2 * c3 + s3 * c0 * c1) * c4 + s1 * s2 * s4 * c0) * c5
        + a5 * (s1 * s3 * c0 * c2 + c0 * c1 * c3) * s5
        + d2 * c0 * c1
        + d4 * (s1 * s3 * c0 * c2 + c0 * c1 * c3)
        + d6 * (((-s1 * c0 * c2 * c3 + s3 * c0 * c1) * c4 + s1 * s2 * s4 * c0) * s5 - (s1 * s3 * c0 * c2 + c0 * c1 * c3) * c5)
    )
    Jp[0, 2] = (
        -a2 * s0 * c2
        - a2 * s2 * c0 * c1
        + a3 * (-s0 * c2 - s2 * c0 * c1) * c3
        + a5 * ((s0 * s2 - c0 * c1 * c2) * s4 + (-s0 * c2 - s2 * c0 * c1) * c3 * c4) * c5
        - a5 * (-s0 * c2 - s2 * c0 * c1) * s3 * s5
        - d4 * (-s0 * c2 - s2 * c0 * c1) * s3
        + d6 * (((s0 * s2 - c0 * c1 * c2) * s4 + (-s0 * c2 - s2 * c0 * c1) * c3 * c4) * s5 + (-s0 * c2 - s2 * c0 * c1) * s3 * c5)
    )
    Jp[0, 3] = (
        -a3 * (-s0 * s2 + c0 * c1 * c2) * s3
        + a3 * s1 * c0 * c3
        + a5 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c4 * c5
        + a5 * (-(-s0 * s2 + c0 * c1 * c2) * c3 - s1 * s3 * c0) * s5
        + d4 * (-(-s0 * s2 + c0 * c1 * c2) * c3 - s1 * s3 * c0)
        + d6 * ((-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5 * c4 - (-(-s0 * s2 + c0 * c1 * c2) * c3 - s1 * s3 * c0) * c5)
    )
    Jp[0, 4] = a5 * (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * c5 + d6 * (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * s5
    Jp[0, 5] = -a5 * (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 + a5 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c5 + d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5)
    Jp[0, 6] = 0
    Jp[1, 0] = (
        -a2 * s0 * s2
        + a2 * c0 * c1 * c2
        + a3 * (-s0 * s2 + c0 * c1 * c2) * c3
        + a3 * s1 * s3 * c0
        + a5 * (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5
        + a5 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5
        + d2 * s1 * c0
        + d4 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3)
        + d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c5)
    )
    Jp[1, 1] = (
        -a2 * s0 * s1 * c2
        - a3 * s0 * s1 * c2 * c3
        + a3 * s0 * s3 * c1
        + a5 * ((-s0 * s1 * c2 * c3 + s0 * s3 * c1) * c4 + s0 * s1 * s2 * s4) * c5
        + a5 * (s0 * s1 * s3 * c2 + s0 * c1 * c3) * s5
        + d2 * s0 * c1
        + d4 * (s0 * s1 * s3 * c2 + s0 * c1 * c3)
        + d6 * (((-s0 * s1 * c2 * c3 + s0 * s3 * c1) * c4 + s0 * s1 * s2 * s4) * s5 - (s0 * s1 * s3 * c2 + s0 * c1 * c3) * c5)
    )
    Jp[1, 2] = (
        -a2 * s0 * s2 * c1
        + a2 * c0 * c2
        + a3 * (-s0 * s2 * c1 + c0 * c2) * c3
        + a5 * ((-s0 * s2 * c1 + c0 * c2) * c3 * c4 + (-s0 * c1 * c2 - s2 * c0) * s4) * c5
        - a5 * (-s0 * s2 * c1 + c0 * c2) * s3 * s5
        - d4 * (-s0 * s2 * c1 + c0 * c2) * s3
        + d6 * (((-s0 * s2 * c1 + c0 * c2) * c3 * c4 + (-s0 * c1 * c2 - s2 * c0) * s4) * s5 + (-s0 * s2 * c1 + c0 * c2) * s3 * c5)
    )
    Jp[1, 3] = (
        -a3 * (s0 * c1 * c2 + s2 * c0) * s3
        + a3 * s0 * s1 * c3
        + a5 * (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * c4 * c5
        + a5 * (-(s0 * c1 * c2 + s2 * c0) * c3 - s0 * s1 * s3) * s5
        + d4 * (-(s0 * c1 * c2 + s2 * c0) * c3 - s0 * s1 * s3)
        + d6 * ((-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5 * c4 - (-(s0 * c1 * c2 + s2 * c0) * c3 - s0 * s1 * s3) * c5)
    )
    Jp[1, 4] = a5 * (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * c5 + d6 * (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * s5
    Jp[1, 5] = -a5 * (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 + a5 * (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * c5 + d6 * ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -a2 * c1 * c2 - a3 * s1 * s3 - a3 * c1 * c2 * c3 + a5 * ((-s1 * s3 - c1 * c2 * c3) * c4 + s2 * s4 * c1) * c5 + a5 * (-s1 * c3 + s3 * c1 * c2) * s5 - d2 * s1 + d4 * (-s1 * c3 + s3 * c1 * c2) + d6 * (((-s1 * s3 - c1 * c2 * c3) * c4 + s2 * s4 * c1) * s5 - (-s1 * c3 + s3 * c1 * c2) * c5)
    Jp[2, 2] = a2 * s1 * s2 + a3 * s1 * s2 * c3 + a5 * (s1 * s2 * c3 * c4 + s1 * s4 * c2) * c5 - a5 * s1 * s2 * s3 * s5 - d4 * s1 * s2 * s3 + d6 * ((s1 * s2 * c3 * c4 + s1 * s4 * c2) * s5 + s1 * s2 * s3 * c5)
    Jp[2, 3] = a3 * s1 * s3 * c2 + a3 * c1 * c3 + a5 * (s1 * s3 * c2 + c1 * c3) * c4 * c5 + a5 * (s1 * c2 * c3 - s3 * c1) * s5 + d4 * (s1 * c2 * c3 - s3 * c1) + d6 * ((s1 * s3 * c2 + c1 * c3) * s5 * c4 - (s1 * c2 * c3 - s3 * c1) * c5)
    Jp[2, 4] = a5 * (-(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4) * c5 + d6 * (-(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4) * s5
    Jp[2, 5] = -a5 * ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 + a5 * (s1 * s3 * c2 + c1 * c3) * c5 + d6 * (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (s1 * s3 * c2 + c1 * c3) * s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3, 7))
    Jr[0, 0] = 0
    Jr[0, 1] = -s0
    Jr[0, 2] = s1 * c0
    Jr[0, 3] = s0 * c2 + s2 * c0 * c1
    Jr[0, 4] = -(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3
    Jr[0, 5] = ((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 - (-s0 * c2 - s2 * c0 * c1) * c4
    Jr[0, 6] = (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * c5
    Jr[1, 0] = 0
    Jr[1, 1] = c0
    Jr[1, 2] = s0 * s1
    Jr[1, 3] = s0 * s2 * c1 - c0 * c2
    Jr[1, 4] = -(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3
    Jr[1, 5] = ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 - (-s0 * s2 * c1 + c0 * c2) * c4
    Jr[1, 6] = (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1 * s2
    Jr[2, 4] = s1 * s3 * c2 + c1 * c3
    Jr[2, 5] = (-s1 * c2 * c3 + s3 * c1) * s4 - s1 * s2 * c4
    Jr[2, 6] = ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (s1 * s3 * c2 + c1 * c3) * c5

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_ur10(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])

    a1 = -0.612
    a2 = -0.5723

    d0 = 0.1273
    d3 = 0.163941
    d4 = 0.1157
    d5 = 0.0922

    p = np.zeros(3)
    p[0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    p[1] = a1 * s0 * c1 - a2 * s0 * s1 * s2 + a2 * s0 * c1 * c2 - d3 * c0 + d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3) + d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4)
    p[2] = a1 * s1 + a2 * s1 * c2 + a2 * s2 * c1 + d0 + d4 * (-(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) - d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    R = np.zeros((3, 3))
    R[0, 0] = (((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * c5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s5
    R[0, 1] = -(((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * s5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * c5
    R[0, 2] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    R[1, 0] = (((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * c5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s5
    R[1, 1] = -(((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * s5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * c5
    R[1, 2] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    R[2, 0] = ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4 * c5 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s5
    R[2, 1] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s5 * c4 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * c5
    R[2, 2] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1 * s0 * c1 + a2 * s0 * s1 * s2 - a2 * s0 * c1 * c2 + d3 * c0 + d4 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 - (s0 * s1 * c2 + s0 * s2 * c1) * c3) + d5 * (-((s0 * s1 * s2 - s0 * c1 * c2) * c3 + (s0 * s1 * c2 + s0 * s2 * c1) * s3) * s4 + c0 * c4)
    Jp[0, 1] = -a1 * s1 * c0 - a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 2] = -a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 3] = d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 4] = d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    Jp[1, 1] = -a1 * s0 * s1 - a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 2] = -a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 3] = d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 4] = d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1 * c1 - a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 2] = -a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 3] = d4 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 4] = -d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4
    Jp[2, 5] = 0
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3
    Jr[0, 5] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3
    Jr[1, 5] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3
    Jr[2, 5] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_ur10e(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])

    a1 = -0.6127
    a2 = -0.57155

    d0 = 0.1807
    d3 = 0.17415
    d4 = 0.11985
    d5 = 0.11655

    p = np.zeros(3)
    p[0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    p[1] = a1 * s0 * c1 - a2 * s0 * s1 * s2 + a2 * s0 * c1 * c2 - d3 * c0 + d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3) + d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4)
    p[2] = a1 * s1 + a2 * s1 * c2 + a2 * s2 * c1 + d0 + d4 * (-(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) - d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    R = np.zeros((3, 3))
    R[0, 0] = (((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * c5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s5
    R[0, 1] = -(((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * s5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * c5
    R[0, 2] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    R[1, 0] = (((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * c5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s5
    R[1, 1] = -(((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * s5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * c5
    R[1, 2] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    R[2, 0] = ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4 * c5 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s5
    R[2, 1] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s5 * c4 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * c5
    R[2, 2] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1 * s0 * c1 + a2 * s0 * s1 * s2 - a2 * s0 * c1 * c2 + d3 * c0 + d4 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 - (s0 * s1 * c2 + s0 * s2 * c1) * c3) + d5 * (-((s0 * s1 * s2 - s0 * c1 * c2) * c3 + (s0 * s1 * c2 + s0 * s2 * c1) * s3) * s4 + c0 * c4)
    Jp[0, 1] = -a1 * s1 * c0 - a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 2] = -a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 3] = d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 4] = d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    Jp[1, 1] = -a1 * s0 * s1 - a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 2] = -a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 3] = d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 4] = d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1 * c1 - a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 2] = -a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 3] = d4 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 4] = -d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4
    Jp[2, 5] = 0
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3
    Jr[0, 5] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3
    Jr[1, 5] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3
    Jr[2, 5] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_ur5(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])

    a1 = -0.425
    a2 = -0.39225

    d0 = 0.089159
    d3 = 0.10915
    d4 = 0.09456
    d5 = 0.0823

    p = np.zeros(3)
    p[0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    p[1] = a1 * s0 * c1 - a2 * s0 * s1 * s2 + a2 * s0 * c1 * c2 - d3 * c0 + d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3) + d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4)
    p[2] = a1 * s1 + a2 * s1 * c2 + a2 * s2 * c1 + d0 + d4 * (-(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) - d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    R = np.zeros((3, 3))
    R[0, 0] = (((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * c5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s5
    R[0, 1] = -(((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * s5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * c5
    R[0, 2] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    R[1, 0] = (((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * c5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s5
    R[1, 1] = -(((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * s5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * c5
    R[1, 2] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    R[2, 0] = ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4 * c5 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s5
    R[2, 1] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s5 * c4 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * c5
    R[2, 2] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1 * s0 * c1 + a2 * s0 * s1 * s2 - a2 * s0 * c1 * c2 + d3 * c0 + d4 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 - (s0 * s1 * c2 + s0 * s2 * c1) * c3) + d5 * (-((s0 * s1 * s2 - s0 * c1 * c2) * c3 + (s0 * s1 * c2 + s0 * s2 * c1) * s3) * s4 + c0 * c4)
    Jp[0, 1] = -a1 * s1 * c0 - a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 2] = -a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 3] = d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 4] = d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    Jp[1, 1] = -a1 * s0 * s1 - a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 2] = -a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 3] = d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 4] = d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1 * c1 - a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 2] = -a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 3] = d4 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 4] = -d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4
    Jp[2, 5] = 0
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3
    Jr[0, 5] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3
    Jr[1, 5] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3
    Jr[2, 5] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_ur5e(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])

    a1 = -0.425
    a2 = -0.3922

    d0 = 0.1625
    d3 = 0.1333
    d4 = 0.0997
    d5 = 0.0996

    p = np.zeros(3)
    p[0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    p[1] = a1 * s0 * c1 - a2 * s0 * s1 * s2 + a2 * s0 * c1 * c2 - d3 * c0 + d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3) + d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4)
    p[2] = a1 * s1 + a2 * s1 * c2 + a2 * s2 * c1 + d0 + d4 * (-(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) - d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    R = np.zeros((3, 3))
    R[0, 0] = (((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * c5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s5
    R[0, 1] = -(((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4) * s5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * c5
    R[0, 2] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    R[1, 0] = (((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * c5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s5
    R[1, 1] = -(((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0) * s5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * c5
    R[1, 2] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    R[2, 0] = ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4 * c5 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s5
    R[2, 1] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s5 * c4 + ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * c5
    R[2, 2] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1 * s0 * c1 + a2 * s0 * s1 * s2 - a2 * s0 * c1 * c2 + d3 * c0 + d4 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 - (s0 * s1 * c2 + s0 * s2 * c1) * c3) + d5 * (-((s0 * s1 * s2 - s0 * c1 * c2) * c3 + (s0 * s1 * c2 + s0 * s2 * c1) * s3) * s4 + c0 * c4)
    Jp[0, 1] = -a1 * s1 * c0 - a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 2] = -a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + d4 * (-(s1 * s2 * c0 - c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 3] = d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) - d5 * (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 4] = d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + d3 * s0 + d4 * ((-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3) + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4)
    Jp[1, 1] = -a1 * s0 * s1 - a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 2] = -a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + d4 * (-(s0 * s1 * s2 - s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 3] = d4 * ((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) - d5 * (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 4] = d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1 * c1 - a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 2] = -a2 * s1 * s2 + a2 * c1 * c2 + d4 * ((-s1 * s2 + c1 * c2) * s3 - (-s1 * c2 - s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 3] = d4 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) - d5 * ((-s1 * s2 + c1 * c2) * c3 - (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 4] = -d5 * ((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * c4
    Jp[2, 5] = 0
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3
    Jr[0, 5] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 + s0 * c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3
    Jr[1, 5] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 - c0 * c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1 * s2 + c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3
    Jr[2, 5] = -((-s1 * s2 + c1 * c2) * s3 + (s1 * c2 + s2 * c1) * c3) * s4

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_iiwa(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])
    c6 = np.cos(q[6])
    s6 = np.sin(q[6])

    d0 = 0.36
    d2 = 0.42
    d4 = 0.4
    d6 = 0.126

    p = np.zeros(3)
    p[0] = d2 * s1 * c0 + d4 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) + d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5)
    p[1] = d2 * s0 * s1 + d4 * (-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) + d6 * ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5)
    p[2] = d0 + d2 * c1 + d4 * (s1 * s3 * c2 + c1 * c3) + d6 * (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (-s1 * s3 * c2 - c1 * c3) * c5)
    R = np.zeros((3, 3))
    R[0, 0] = ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5) * c6 + (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * s6
    R[0, 1] = -((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5) * s6 + (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * c6
    R[0, 2] = (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5
    R[1, 0] = ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5) * c6 + (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * s6
    R[1, 1] = -((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5) * s6 + (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * c6
    R[1, 2] = (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5
    R[2, 0] = (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (-s1 * s3 * c2 - c1 * c3) * s5) * c6 + (-(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4) * s6
    R[2, 1] = -(((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (-s1 * s3 * c2 - c1 * c3) * s5) * s6 + (-(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4) * c6
    R[2, 2] = ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (-s1 * s3 * c2 - c1 * c3) * c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = -d2 * s0 * s1 + d4 * (-(-s0 * c1 * c2 - s2 * c0) * s3 - s0 * s1 * c3) + d6 * ((((-s0 * c1 * c2 - s2 * c0) * c3 - s0 * s1 * s3) * c4 + (s0 * s2 * c1 - c0 * c2) * s4) * s5 - ((-s0 * c1 * c2 - s2 * c0) * s3 + s0 * s1 * c3) * c5)
    Jp[0, 1] = d2 * c0 * c1 + d4 * (s1 * s3 * c0 * c2 + c0 * c1 * c3) + d6 * (((-s1 * c0 * c2 * c3 + s3 * c0 * c1) * c4 + s1 * s2 * s4 * c0) * s5 - (-s1 * s3 * c0 * c2 - c0 * c1 * c3) * c5)
    Jp[0, 2] = -d4 * (-s0 * c2 - s2 * c0 * c1) * s3 + d6 * (((s0 * s2 - c0 * c1 * c2) * s4 + (-s0 * c2 - s2 * c0 * c1) * c3 * c4) * s5 - (-s0 * c2 - s2 * c0 * c1) * s3 * c5)
    Jp[0, 3] = d4 * (-(-s0 * s2 + c0 * c1 * c2) * c3 - s1 * s3 * c0) + d6 * ((-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5 * c4 - ((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c5)
    Jp[0, 4] = d6 * (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * s5
    Jp[0, 5] = d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5)
    Jp[0, 6] = 0
    Jp[1, 0] = d2 * s1 * c0 + d4 * (-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) + d6 * ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5)
    Jp[1, 1] = d2 * s0 * c1 + d4 * (s0 * s1 * s3 * c2 + s0 * c1 * c3) + d6 * (((-s0 * s1 * c2 * c3 + s0 * s3 * c1) * c4 + s0 * s1 * s2 * s4) * s5 - (-s0 * s1 * s3 * c2 - s0 * c1 * c3) * c5)
    Jp[1, 2] = -d4 * (-s0 * s2 * c1 + c0 * c2) * s3 + d6 * (((-s0 * s2 * c1 + c0 * c2) * c3 * c4 + (-s0 * c1 * c2 - s2 * c0) * s4) * s5 - (-s0 * s2 * c1 + c0 * c2) * s3 * c5)
    Jp[1, 3] = d4 * (-(s0 * c1 * c2 + s2 * c0) * c3 - s0 * s1 * s3) + d6 * ((-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5 * c4 - ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c5)
    Jp[1, 4] = d6 * (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * s5
    Jp[1, 5] = d6 * ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -d2 * s1 + d4 * (-s1 * c3 + s3 * c1 * c2) + d6 * (((-s1 * s3 - c1 * c2 * c3) * c4 + s2 * s4 * c1) * s5 - (s1 * c3 - s3 * c1 * c2) * c5)
    Jp[2, 2] = -d4 * s1 * s2 * s3 + d6 * ((s1 * s2 * c3 * c4 + s1 * s4 * c2) * s5 - s1 * s2 * s3 * c5)
    Jp[2, 3] = d4 * (s1 * c2 * c3 - s3 * c1) + d6 * ((s1 * s3 * c2 + c1 * c3) * s5 * c4 - (-s1 * c2 * c3 + s3 * c1) * c5)
    Jp[2, 4] = d6 * (-(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4) * s5
    Jp[2, 5] = d6 * (((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * c5 + (-s1 * s3 * c2 - c1 * c3) * s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3, 7))
    Jr[0, 0] = 0
    Jr[0, 1] = -s0
    Jr[0, 2] = s1 * c0
    Jr[0, 3] = s0 * c2 + s2 * c0 * c1
    Jr[0, 4] = -(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3
    Jr[0, 5] = -((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4
    Jr[0, 6] = (((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 - ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5
    Jr[1, 0] = 0
    Jr[1, 1] = c0
    Jr[1, 2] = s0 * s1
    Jr[1, 3] = s0 * s2 * c1 - c0 * c2
    Jr[1, 4] = -(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3
    Jr[1, 5] = -((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4
    Jr[1, 6] = (((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 - ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1 * s2
    Jr[2, 4] = s1 * s3 * c2 + c1 * c3
    Jr[2, 5] = -(-s1 * c2 * c3 + s3 * c1) * s4 + s1 * s2 * c4
    Jr[2, 6] = ((-s1 * c2 * c3 + s3 * c1) * c4 + s1 * s2 * s4) * s5 - (-s1 * s3 * c2 - c1 * c3) * c5

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_lwr(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the robot.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.array
        Position of the end effector.
    R : np.array
        Rotation matrix of the end effector.
    J : np.array
        Jacobian matrix (6 x nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2])
    s2 = np.sin(q[2])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    c5 = np.cos(q[5])
    s5 = np.sin(q[5])
    c6 = np.cos(q[6])
    s6 = np.sin(q[6])

    d0 = 0.31
    d2 = 0.4
    d4 = 0.39
    d6 = 0.078

    p = np.zeros(3)
    p[0] = -d2 * s1 * c0 + d4 * ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) + d6 * (-(((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5)
    p[1] = -d2 * s0 * s1 + d4 * ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) + d6 * (-(((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5)
    p[2] = d0 + d2 * c1 + d4 * (s1 * s3 * c2 + c1 * c3) + d6 * (-((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * s5 + (s1 * s3 * c2 + c1 * c3) * c5)
    R = np.zeros((3, 3))
    R[0, 0] = ((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5) * c6 + (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * s6
    R[0, 1] = -((((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5) * s6 + (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * c6
    R[0, 2] = -(((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5
    R[1, 0] = ((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5) * c6 + (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * s6
    R[1, 1] = -((((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5) * s6 + (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * c6
    R[1, 2] = -(((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5
    R[2, 0] = (((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * c5 + (s1 * s3 * c2 + c1 * c3) * s5) * c6 + (-(s1 * c2 * c3 - s3 * c1) * s4 - s1 * s2 * c4) * s6
    R[2, 1] = -(((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * c5 + (s1 * s3 * c2 + c1 * c3) * s5) * s6 + (-(s1 * c2 * c3 - s3 * c1) * s4 - s1 * s2 * c4) * c6
    R[2, 2] = -((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * s5 + (s1 * s3 * c2 + c1 * c3) * c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = d2 * s0 * s1 + d4 * ((-s0 * c1 * c2 - s2 * c0) * s3 + s0 * s1 * c3) + d6 * (-(((-s0 * c1 * c2 - s2 * c0) * c3 - s0 * s1 * s3) * c4 + (s0 * s2 * c1 - c0 * c2) * s4) * s5 + ((-s0 * c1 * c2 - s2 * c0) * s3 + s0 * s1 * c3) * c5)
    Jp[0, 1] = -d2 * c0 * c1 + d4 * (-s1 * s3 * c0 * c2 - c0 * c1 * c3) + d6 * (-((-s1 * c0 * c2 * c3 + s3 * c0 * c1) * c4 + s1 * s2 * s4 * c0) * s5 + (-s1 * s3 * c0 * c2 - c0 * c1 * c3) * c5)
    Jp[0, 2] = d4 * (-s0 * c2 - s2 * c0 * c1) * s3 + d6 * (-((s0 * s2 - c0 * c1 * c2) * s4 + (-s0 * c2 - s2 * c0 * c1) * c3 * c4) * s5 + (-s0 * c2 - s2 * c0 * c1) * s3 * c5)
    Jp[0, 3] = d4 * ((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) + d6 * (-(-(-s0 * s2 + c0 * c1 * c2) * s3 + s1 * c0 * c3) * s5 * c4 + ((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c5)
    Jp[0, 4] = -d6 * (-((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 + (-s0 * c2 - s2 * c0 * c1) * c4) * s5
    Jp[0, 5] = d6 * (-(((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * c5 - ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * s5)
    Jp[0, 6] = 0
    Jp[1, 0] = -d2 * s1 * c0 + d4 * ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) + d6 * (-(((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5)
    Jp[1, 1] = -d2 * s0 * c1 + d4 * (-s0 * s1 * s3 * c2 - s0 * c1 * c3) + d6 * (-((-s0 * s1 * c2 * c3 + s0 * s3 * c1) * c4 + s0 * s1 * s2 * s4) * s5 + (-s0 * s1 * s3 * c2 - s0 * c1 * c3) * c5)
    Jp[1, 2] = d4 * (-s0 * s2 * c1 + c0 * c2) * s3 + d6 * (-((-s0 * s2 * c1 + c0 * c2) * c3 * c4 + (-s0 * c1 * c2 - s2 * c0) * s4) * s5 + (-s0 * s2 * c1 + c0 * c2) * s3 * c5)
    Jp[1, 3] = d4 * ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) + d6 * (-(-(s0 * c1 * c2 + s2 * c0) * s3 + s0 * s1 * c3) * s5 * c4 + ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c5)
    Jp[1, 4] = -d6 * (-((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 + (-s0 * s2 * c1 + c0 * c2) * c4) * s5
    Jp[1, 5] = d6 * (-(((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * c5 - ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -d2 * s1 + d4 * (-s1 * c3 + s3 * c1 * c2) + d6 * (-((s1 * s3 + c1 * c2 * c3) * c4 - s2 * s4 * c1) * s5 + (-s1 * c3 + s3 * c1 * c2) * c5)
    Jp[2, 2] = -d4 * s1 * s2 * s3 + d6 * (-(-s1 * s2 * c3 * c4 - s1 * s4 * c2) * s5 - s1 * s2 * s3 * c5)
    Jp[2, 3] = d4 * (s1 * c2 * c3 - s3 * c1) + d6 * (-(-s1 * s3 * c2 - c1 * c3) * s5 * c4 + (s1 * c2 * c3 - s3 * c1) * c5)
    Jp[2, 4] = -d6 * (-(s1 * c2 * c3 - s3 * c1) * s4 - s1 * s2 * c4) * s5
    Jp[2, 5] = d6 * (-((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * c5 - (s1 * s3 * c2 + c1 * c3) * s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3, 7))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = -s1 * c0
    Jr[0, 3] = -s0 * c2 - s2 * c0 * c1
    Jr[0, 4] = (-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3
    Jr[0, 5] = ((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * s4 - (-s0 * c2 - s2 * c0 * c1) * c4
    Jr[0, 6] = -(((-s0 * s2 + c0 * c1 * c2) * c3 + s1 * s3 * c0) * c4 + (-s0 * c2 - s2 * c0 * c1) * s4) * s5 + ((-s0 * s2 + c0 * c1 * c2) * s3 - s1 * c0 * c3) * c5
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -s0 * s1
    Jr[1, 3] = -s0 * s2 * c1 + c0 * c2
    Jr[1, 4] = (s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3
    Jr[1, 5] = ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * s4 - (-s0 * s2 * c1 + c0 * c2) * c4
    Jr[1, 6] = -(((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 + (-s0 * s2 * c1 + c0 * c2) * s4) * s5 + ((s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3) * c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1 * s2
    Jr[2, 4] = s1 * s3 * c2 + c1 * c3
    Jr[2, 5] = (s1 * c2 * c3 - s3 * c1) * s4 + s1 * s2 * c4
    Jr[2, 6] = -((s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4) * s5 + (s1 * s3 * c2 + c1 * c3) * c5

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_dh_7dof(q: np.ndarray, DH: dict, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the 7 DOF robot with revolute joints.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    DH : dict
        DH parameters: 'a', 'alpha', 'd', 'theta'.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c0 = np.cos(q[0] + DH["theta"][0])
    s0 = np.sin(q[0] + DH["theta"][0])
    c1 = np.cos(q[1] + DH["theta"][1])
    s1 = np.sin(q[1] + DH["theta"][1])
    c2 = np.cos(q[2] + DH["theta"][2])
    s2 = np.sin(q[2] + DH["theta"][2])
    c3 = np.cos(q[3] + DH["theta"][3])
    s3 = np.sin(q[3] + DH["theta"][3])
    c4 = np.cos(q[4] + DH["theta"][4])
    s4 = np.sin(q[4] + DH["theta"][4])
    c5 = np.cos(q[5] + DH["theta"][5])
    s5 = np.sin(q[5] + DH["theta"][5])
    c6 = np.cos(q[6] + DH["theta"][6])
    s6 = np.sin(q[6] + DH["theta"][6])

    a0 = DH["a"][0]
    a1 = DH["a"][1]
    a2 = DH["a"][2]
    a3 = DH["a"][3]
    a4 = DH["a"][4]
    a5 = DH["a"][5]
    a6 = DH["a"][6]

    ca0 = np.cos(DH["alpha"][0])
    sa0 = np.sin(DH["alpha"][0])
    ca1 = np.cos(DH["alpha"][1])
    sa1 = np.sin(DH["alpha"][1])
    ca2 = np.cos(DH["alpha"][2])
    sa2 = np.sin(DH["alpha"][2])
    ca3 = np.cos(DH["alpha"][3])
    sa3 = np.sin(DH["alpha"][3])
    ca4 = np.cos(DH["alpha"][4])
    sa4 = np.sin(DH["alpha"][4])
    ca5 = np.cos(DH["alpha"][5])
    sa5 = np.sin(DH["alpha"][5])
    ca6 = np.cos(DH["alpha"][6])
    sa6 = np.sin(DH["alpha"][6])

    d0 = DH["d"][0]
    d1 = DH["d"][1]
    d2 = DH["d"][2]
    d3 = DH["d"][3]
    d4 = DH["d"][4]
    d5 = DH["d"][5]
    d6 = DH["d"][6]

    p = np.zeros(3)
    p[0] = (
        a0 * c0
        - a1 * s0 * s1 * ca0
        + a1 * c0 * c1
        + a2 * (-s0 * s1 * ca0 + c0 * c1) * c2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3
        + a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d1 * sa0 * s0
        + d2 * (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0)
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2)
        + d4
        * (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    p[1] = (
        a0 * s0
        + a1 * s0 * c1
        + a1 * s1 * ca0 * c0
        + a2 * (s0 * c1 + s1 * ca0 * c0) * c2
        + a2 * (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2
        + a3 * ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3
        + a4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
        + a4
        * (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        - d1 * sa0 * c0
        + d2 * (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1)
        + d3 * ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2)
        + d4
        * (
            ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    p[2] = (
        a1 * sa0 * s1
        + a2 * (sa0 * ca1 * c1 + sa1 * ca0) * s2
        + a2 * sa0 * s1 * c2
        + a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3
        + a3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3
        + a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        + a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d0
        + d1 * ca0
        + d2 * (-sa0 * sa1 * c1 + ca0 * ca1)
        + d3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2)
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3)
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
        + d6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * sa5
            * s5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * ca5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    R = np.zeros((3, 3))
    R[0, 0] = (
        (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
    ) * c6 + (
        -(
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        * ca5
        + (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * sa5
        + (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * ca5
        * c5
    ) * s6
    R[0, 1] = (
        -(
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * s6
        * ca6
        + (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
        * sa6
        + (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * ca6
        * c6
    )
    R[0, 2] = (
        (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * sa6
        * s6
        + (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
        * ca6
        - (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * sa6
        * c6
    )
    R[1, 0] = (
        (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
    ) * c6 + (
        -(
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        * ca5
        + (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * sa5
        + (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * ca5
        * c5
    ) * s6
    R[1, 1] = (
        -(
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * s6
        * ca6
        + (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
        * sa6
        + (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * ca6
        * c6
    )
    R[1, 2] = (
        (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * sa6
        * s6
        + (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
        * ca6
        - (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * sa6
        * c6
    )
    R[2, 0] = (
        (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * c5
        + (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * s5
    ) * c6 + (
        -(
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * s5
        * ca5
        + (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
        * sa5
        + (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * ca5
        * c5
    ) * s6
    R[2, 1] = (
        -(
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * s5
        )
        * s6
        * ca6
        + (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * sa5
            * s5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * ca5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * sa5
            * c5
        )
        * sa6
        + (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * ca6
        * c6
    )
    R[2, 2] = (
        (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * s5
        )
        * sa6
        * s6
        + (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * sa5
            * s5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * ca5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * sa5
            * c5
        )
        * ca6
        - (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * sa6
        * c6
    )
    Jp = np.zeros((3, 7))
    Jp[0, 0] = (
        -a0 * s0
        - a1 * s0 * c1
        - a1 * s1 * ca0 * c0
        + a2 * (-s0 * c1 - s1 * ca0 * c0) * c2
        + a2 * (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2
        + a3 * ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3
        + a3 * (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3
        + a4 * (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d1 * sa0 * c0
        + d2 * (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1)
        + d3 * ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2)
        + d4
        * (
            ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[0, 1] = (
        -a1 * s0 * ca0 * c1
        - a1 * s1 * c0
        + a2 * (-s0 * ca0 * c1 - s1 * c0) * c2
        + a2 * (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2
        + a3 * ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3
        + a3 * (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3
        + a4 * (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
        + a4
        * (
            -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
            + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
            + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
            + (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * s4 * ca4
            + (
                ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                    - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * sa4 * s4
                + (
                    ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                    - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                    - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d2 * (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1)
        + d3 * ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2)
        + d4
        * (
            ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
            + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
            - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * sa4 * s4
            + (
                ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * sa4 * s4
                + (
                    ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                    - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                    - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                    + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[0, 2] = (
        -a2 * (-s0 * s1 * ca0 + c0 * c1) * s2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3
        + a4 * ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
        + a4
        * (
            -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
        )
        * s4
        + a5
        * (
            ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
            + (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * s4 * ca4
            + (
                (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
            )
            * sa4
            + (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * sa4 * s4
                + (
                    (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
                )
                * ca4
                - (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2)
        + d4
        * (
            (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
        )
        + d5
        * (
            ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * sa4 * s4
            + (
                (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
            )
            * ca4
            - (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * sa4 * s4
                + (
                    (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
                )
                * ca4
                - (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[0, 3] = (
        -a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3
        + a4 * (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
        + a4 * (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
        + a5
        * (
            (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
            + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * s4 * ca4
            + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * sa4
            + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
            )
            * c5
            + (
                -(-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * s4 * ca4
                + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * sa4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
            )
            * s5
            * ca5
            + (
                (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * sa4 * s4
                + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * ca4
                - (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * sa4 * c4
            )
            * sa5
            + (
                -(-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * s4 * ca4
                + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * sa4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3)
        + d5
        * (
            (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * sa4 * s4
            + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * ca4
            - (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * sa4 * c4
        )
        + d6
        * (
            (
                (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
            )
            * sa5
            * s5
            + (
                (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * sa4 * s4
                + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * ca4
                - (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * sa4 * c4
            )
            * ca5
            - (
                -(-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * s4 * ca4
                + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * sa4
                + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    Jp[0, 4] = (
        -a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * c4
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * c4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * ca4 * c4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
            * ca4
        )
        * s5
        + a6
        * (
            (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * c4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * ca4 * c4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * c4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * s4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * ca4 * c4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * ca5
            * c5
        )
        * s6
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * s4
        )
        + d6
        * (
            (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * c4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * s4
            )
            * ca5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * ca4 * c4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * sa5
            * c5
        )
    )
    Jp[0, 5] = (
        -a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * c5
        + a6
        * (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * c5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * ca5
            * c5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
            * ca5
        )
        * s6
        + d6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * s5
        )
    )
    Jp[0, 6] = (
        -a6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * s6
        + a6
        * (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * c6
    )
    Jp[1, 0] = (
        a0 * c0
        - a1 * s0 * s1 * ca0
        + a1 * c0 * c1
        + a2 * (-s0 * s1 * ca0 + c0 * c1) * c2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3
        + a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d1 * sa0 * s0
        + d2 * (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0)
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2)
        + d4
        * (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                    - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                    + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[1, 1] = (
        -a1 * s0 * s1
        + a1 * ca0 * c0 * c1
        + a2 * (-s0 * s1 + ca0 * c0 * c1) * c2
        + a2 * (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3
        + a4 * (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
            + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
            + (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                    - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                    - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                    - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d2 * (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0)
        + d3 * ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2)
        + d4
        * (
            ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
            - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * sa4 * s4
                + (
                    ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                    - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
                )
                * ca4
                - (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -(((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * s4 * ca4
                + (
                    ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                    - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
                )
                * sa4
                + (
                    -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                    + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                    + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[1, 2] = (
        -a2 * (s0 * c1 + s1 * ca0 * c0) * s2
        + a2 * (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3
        + a4 * ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
        + a4
        * (
            -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
        )
        * s4
        + a5
        * (
            ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
            + (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * s4 * ca4
            + (
                (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
            )
            * sa4
            + (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + a6
        * (
            (
                ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * sa4 * s4
                + (
                    (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
                )
                * ca4
                - (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * s6
        + d3 * ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2)
        + d4
        * (
            (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
        )
        + d5
        * (
            ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * sa4 * s4
            + (
                (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
            )
            * ca4
            - (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * sa4
            * c4
        )
        + d6
        * (
            (
                ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * s5
            + (
                ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * sa4 * s4
                + (
                    (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
                )
                * ca4
                - (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * ca5
            - (
                -((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * s4 * ca4
                + (
                    (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
                )
                * sa4
                + (
                    -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * c5
        )
    )
    Jp[1, 3] = (
        -a3 * ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3
        + a4 * (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
        + a4 * (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
        + a5
        * (
            (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
            + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * s4 * ca4
            + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * sa4
            + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
            )
            * c5
            + (
                -(-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * s4 * ca4
                + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * sa4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
            )
            * s5
            * ca5
            + (
                (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * sa4 * s4
                + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * ca4
                - (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * sa4 * c4
            )
            * sa5
            + (
                -(-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * s4 * ca4
                + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * sa4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3)
        + d5
        * (
            (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * sa4 * s4
            + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * ca4
            - (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * sa4 * c4
        )
        + d6
        * (
            (
                (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
            )
            * sa5
            * s5
            + (
                (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * sa4 * s4
                + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * ca4
                - (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * sa4 * c4
            )
            * ca5
            - (
                -(-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * s4 * ca4
                + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * sa4
                + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    Jp[1, 4] = (
        -a4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
        + a4
        * (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * c4
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * c4
        )
        * c5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * ca4 * c4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
            * ca4
        )
        * s5
        + a6
        * (
            (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * c4
            )
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * ca4 * c4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * c4
            )
            * s5
            * ca5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * s4
            )
            * sa5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * ca4 * c4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * ca5
            * c5
        )
        * s6
        + d5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * s4
        )
        + d6
        * (
            (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * c4
            )
            * sa5
            * s5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * s4
            )
            * ca5
            - (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * ca4 * c4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
                * ca4
            )
            * sa5
            * c5
        )
    )
    Jp[1, 5] = (
        -a5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * c5
        + a6
        * (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * c5
        )
        * c6
        + a6
        * (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * ca5
            * c5
            - (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
            * ca5
        )
        * s6
        + d6
        * (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * sa5
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * sa5
            * s5
        )
    )
    Jp[1, 6] = (
        -a6
        * (
            (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * c5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * s5
        )
        * s6
        + a6
        * (
            -(
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * s4
            )
            * s5
            * ca5
            + (
                (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * ca4
                - (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * sa4
                * c4
            )
            * sa5
            + (
                -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
                + (
                    ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                    - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
                )
                * sa4
                + (
                    -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                    + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                    + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
                )
                * ca4
                * c4
            )
            * ca5
            * c5
        )
        * c6
    )
    Jp[2, 0] = 0
    Jp[2, 1] = (
        a1 * sa0 * c1
        - a2 * sa0 * s1 * s2 * ca1
        + a2 * sa0 * c1 * c2
        + a3 * (-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3
        + a3 * (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3
        + a4 * ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
        + a4 * (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
        + a5
        * (
            ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
            + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
        )
        * c5
        + a5
        * (
            -((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * s4 * ca4
            + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * sa4
            + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
            )
            * c5
            + (
                -((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * s4 * ca4
                + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * sa4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
            )
            * s5
            * ca5
            + (
                ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * sa4 * s4
                + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * ca4
                - (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * sa4 * c4
            )
            * sa5
            + (
                -((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * s4 * ca4
                + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * sa4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d2 * sa0 * sa1 * s1
        + d3 * (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1)
        + d4 * ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3)
        + d5
        * (
            ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * sa4 * s4
            + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * ca4
            - (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * sa4 * c4
        )
        + d6
        * (
            (
                ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
            )
            * sa5
            * s5
            + (
                ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * sa4 * s4
                + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * ca4
                - (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * sa4 * c4
            )
            * ca5
            - (
                -((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * s4 * ca4
                + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * sa4
                + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    Jp[2, 2] = (
        a2 * (sa0 * ca1 * c1 + sa1 * ca0) * c2
        - a2 * sa0 * s1 * s2
        + a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3
        + a3 * (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3
        + a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
        + a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * sa4 * c4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d3 * ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2)
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3)
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * sa4 * c4
        )
        + d6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
            )
            * sa5
            * s5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * sa4 * c4
            )
            * ca5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    Jp[2, 3] = (
        -a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3
        + a3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
        + a5
        * (
            (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + a6
        * (
            (
                (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
            )
            * c5
            + (
                -(-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * ca4 * c4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
            )
            * s5
            * ca5
            + (
                (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * sa4 * c4
            )
            * sa5
            + (
                -(-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * s6
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3)
        + d5
        * (
            (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * sa4 * c4
        )
        + d6
        * (
            (
                (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
            )
            * sa5
            * s5
            + (
                (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * sa4 * c4
            )
            * ca5
            - (
                -(-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * ca4 * c4
            )
            * sa5
            * c5
        )
    )
    Jp[2, 4] = (
        -a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * ca4 * c4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4 * ca4
        )
        * s5
        + a6
        * (
            (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * ca4 * c4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4 * ca4
            )
            * s5
        )
        * c6
        + a6
        * (
            -(
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * s4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * ca4 * c4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4 * ca4
            )
            * ca5
            * c5
        )
        * s6
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * s4
        )
        + d6
        * (
            (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
            )
            * sa5
            * s5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * s4
            )
            * ca5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * ca4 * c4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4 * ca4
            )
            * sa5
            * c5
        )
    )
    Jp[2, 5] = (
        -a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * s5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * c5
        + a6
        * (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * s5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * c5
        )
        * c6
        + a6
        * (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * ca5
            * c5
            - (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * s5
            * ca5
        )
        * s6
        + d6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * sa5
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * sa5
            * s5
        )
    )
    Jp[2, 6] = (
        -a6
        * (
            (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * c5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * s5
        )
        * s6
        + a6
        * (
            -(
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
            )
            * s5
            * ca5
            + (
                (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
                - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
            )
            * sa5
            + (
                -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
                + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
                + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
            )
            * ca5
            * c5
        )
        * c6
    )
    Jr = np.zeros((3, 7))
    Jr[0, 0] = 0
    Jr[0, 1] = sa0 * s0
    Jr[0, 2] = sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0
    Jr[0, 3] = (-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2
    Jr[0, 4] = (
        ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
        + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
        - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
    )
    Jr[0, 5] = (
        (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
        + (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        * ca4
        - (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * sa4
        * c4
    )
    Jr[0, 6] = (
        (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * sa5
        * s5
        + (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * ca5
        - (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * sa5
        * c5
    )
    Jr[1, 0] = 0
    Jr[1, 1] = -sa0 * c0
    Jr[1, 2] = -sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1
    Jr[1, 3] = (s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2
    Jr[1, 4] = (
        ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
        + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
        - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
    )
    Jr[1, 5] = (
        (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
        + (
            ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        * ca4
        - (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * sa4
        * c4
    )
    Jr[1, 6] = (
        (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * sa5
        * s5
        + (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * ca5
        - (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * sa5
        * c5
    )
    Jr[2, 0] = 1
    Jr[2, 1] = ca0
    Jr[2, 2] = -sa0 * sa1 * c1 + ca0 * ca1
    Jr[2, 3] = (-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2
    Jr[2, 4] = ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3
    Jr[2, 5] = (
        (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
        + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
        - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
    )
    Jr[2, 6] = (
        (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * sa5
        * s5
        + (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
        * ca5
        - (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * sa5
        * c5
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_dh_6dof(q: np.ndarray, DH: dict, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the 6 DOF robot with revolute joints.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    DH : dict
        DH parameters: 'a', 'alpha', 'd', 'theta'.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c0 = np.cos(q[0] + DH["theta"][0])
    s0 = np.sin(q[0] + DH["theta"][0])
    c1 = np.cos(q[1] + DH["theta"][1])
    s1 = np.sin(q[1] + DH["theta"][1])
    c2 = np.cos(q[2] + DH["theta"][2])
    s2 = np.sin(q[2] + DH["theta"][2])
    c3 = np.cos(q[3] + DH["theta"][3])
    s3 = np.sin(q[3] + DH["theta"][3])
    c4 = np.cos(q[4] + DH["theta"][4])
    s4 = np.sin(q[4] + DH["theta"][4])
    c5 = np.cos(q[5] + DH["theta"][5])
    s5 = np.sin(q[5] + DH["theta"][5])

    a0 = DH["a"][0]
    a1 = DH["a"][1]
    a2 = DH["a"][2]
    a3 = DH["a"][3]
    a4 = DH["a"][4]
    a5 = DH["a"][5]

    ca0 = np.cos(DH["alpha"][0])
    sa0 = np.sin(DH["alpha"][0])
    ca1 = np.cos(DH["alpha"][1])
    sa1 = np.sin(DH["alpha"][1])
    ca2 = np.cos(DH["alpha"][2])
    sa2 = np.sin(DH["alpha"][2])
    ca3 = np.cos(DH["alpha"][3])
    sa3 = np.sin(DH["alpha"][3])
    ca4 = np.cos(DH["alpha"][4])
    sa4 = np.sin(DH["alpha"][4])
    ca5 = np.cos(DH["alpha"][5])
    sa5 = np.sin(DH["alpha"][5])

    d0 = DH["d"][0]
    d1 = DH["d"][1]
    d2 = DH["d"][2]
    d3 = DH["d"][3]
    d4 = DH["d"][4]
    d5 = DH["d"][5]

    p = np.zeros(3)
    p[0] = (
        a0 * c0
        - a1 * s0 * s1 * ca0
        + a1 * c0 * c1
        + a2 * (-s0 * s1 * ca0 + c0 * c1) * c2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3
        + a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d1 * sa0 * s0
        + d2 * (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0)
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2)
        + d4
        * (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    p[1] = (
        a0 * s0
        + a1 * s0 * c1
        + a1 * s1 * ca0 * c0
        + a2 * (s0 * c1 + s1 * ca0 * c0) * c2
        + a2 * (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2
        + a3 * ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3
        + a4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
        + a4
        * (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        - d1 * sa0 * c0
        + d2 * (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1)
        + d3 * ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2)
        + d4
        * (
            ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    p[2] = (
        a1 * sa0 * s1
        + a2 * (sa0 * ca1 * c1 + sa1 * ca0) * s2
        + a2 * sa0 * s1 * c2
        + a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3
        + a3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3
        + a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        + a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * s5
        + d0
        + d1 * ca0
        + d2 * (-sa0 * sa1 * c1 + ca0 * ca1)
        + d3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2)
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3)
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
    )
    R = np.zeros((3, 3))
    R[0, 0] = (
        (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
        + (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * s4
    ) * c5 + (
        -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
        + (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        * sa4
        + (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * ca4
        * c4
    ) * s5
    R[0, 1] = (
        -(
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        * ca5
        + (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * sa5
        + (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * ca5
        * c5
    )
    R[0, 2] = (
        (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * sa5
        * s5
        + (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * ca5
        - (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * sa5
        * c5
    )
    R[1, 0] = (
        (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
        + (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * s4
    ) * c5 + (
        -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
        + (
            ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        * sa4
        + (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * ca4
        * c4
    ) * s5
    R[1, 1] = (
        -(
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        * ca5
        + (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * sa5
        + (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * ca5
        * c5
    )
    R[1, 2] = (
        (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * sa5
        * s5
        + (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
        * ca5
        - (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * sa5
        * c5
    )
    R[2, 0] = (
        (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
        + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
    ) * c5 + (
        -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
        + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
        + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
    ) * s5
    R[2, 1] = (
        -(
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * s5
        * ca5
        + (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
        * sa5
        + (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * ca5
        * c5
    )
    R[2, 2] = (
        (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * sa5
        * s5
        + (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
        )
        * ca5
        - (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * sa5
        * c5
    )
    Jp = np.zeros((3, 6))
    Jp[0, 0] = (
        -a0 * s0
        - a1 * s0 * c1
        - a1 * s1 * ca0 * c0
        + a2 * (-s0 * c1 - s1 * ca0 * c0) * c2
        + a2 * (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2
        + a3 * ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3
        + a3 * (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3
        + a4 * (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d1 * sa0 * c0
        + d2 * (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1)
        + d3 * ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2)
        + d4
        * (
            ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * c3 + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * c1 - s1 * ca0 * c0) * c2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * c1 - s1 * ca0 * c0) * sa2 * s2 - (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * sa2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(-s0 * c1 - s1 * ca0 * c0) * s2 * ca2 + (sa0 * sa1 * c0 + s0 * s1 * ca1 - ca0 * ca1 * c0 * c1) * ca2 * c2 + (sa0 * ca1 * c0 - sa1 * s0 * s1 + sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[0, 1] = (
        -a1 * s0 * ca0 * c1
        - a1 * s1 * c0
        + a2 * (-s0 * ca0 * c1 - s1 * c0) * c2
        + a2 * (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2
        + a3 * ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3
        + a3 * (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3
        + a4 * (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
        + a4
        * (
            -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
            + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
            + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * c4
            + (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * s4 * ca4
            + (
                ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d2 * (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1)
        + d3 * ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2)
        + d4
        * (
            ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
            + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
            - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * c3 + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * s3) * sa4 * s4
            + (
                ((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * sa3 * s3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * ca3
                - (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * ca0 * c1 - s1 * c0) * c2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * s2) * s3 * ca3
                + ((-s0 * ca0 * c1 - s1 * c0) * sa2 * s2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * ca2 - (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * sa2 * c2) * sa3
                + (-(-s0 * ca0 * c1 - s1 * c0) * s2 * ca2 + (-sa1 * s0 * s1 * ca0 + sa1 * c0 * c1) * sa2 + (s0 * s1 * ca0 * ca1 - ca1 * c0 * c1) * ca2 * c2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[0, 2] = (
        -a2 * (-s0 * s1 * ca0 + c0 * c1) * s2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3
        + a4 * ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
        + a4
        * (
            -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
        )
        * s4
        + a5
        * (
            ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * c4
            + (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * s4 * ca4
            + (
                (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
            )
            * sa4
            + (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2)
        + d4
        * (
            (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
        )
        + d5
        * (
            ((-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * s3) * sa4 * s4
            + (
                (-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * sa3 * c3
            )
            * ca4
            - (
                -(-(-s0 * s1 * ca0 + c0 * c1) * s2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * c2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * s2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * ca2 * c2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2 * ca2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[0, 3] = (
        -a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3
        + a4 * (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
        + a4 * (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
        + a5
        * (
            (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * c4
            + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * s4 * ca4
            + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * sa4
            + (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + d4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3)
        + d5
        * (
            (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * c3) * sa4 * s4
            + (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * s3) * ca4
            - (-((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * ca3 * c3 - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3 * ca3) * sa4 * c4
        )
    )
    Jp[0, 4] = (
        -a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * c4
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * c4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * ca4 * c4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
            * ca4
        )
        * s5
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * s4
        )
    )
    Jp[0, 5] = (
        -a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * c5
    )
    Jp[1, 0] = (
        a0 * c0
        - a1 * s0 * s1 * ca0
        + a1 * c0 * c1
        + a2 * (-s0 * s1 * ca0 + c0 * c1) * c2
        + a2 * (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3
        + a4 * (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * c4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d1 * sa0 * s0
        + d2 * (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0)
        + d3 * ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2)
        + d4
        * (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
                - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
                + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[1, 1] = (
        -a1 * s0 * s1
        + a1 * ca0 * c0 * c1
        + a2 * (-s0 * s1 + ca0 * c0 * c1) * c2
        + a2 * (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2
        + a3 * ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3
        + a3 * (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3
        + a4 * (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
        + a4
        * (
            -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
            + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
        )
        * s4
        + a5
        * (
            (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * c4
            + (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -(((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * s4 * ca4
            + (
                ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
            )
            * sa4
            + (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d2 * (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0)
        + d3 * ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2)
        + d4
        * (
            ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
            - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
        )
        + d5
        * (
            (((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * s3) * sa4 * s4
            + (
                ((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * sa3 * s3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * ca3
                - (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * sa3 * c3
            )
            * ca4
            - (
                -((-s0 * s1 + ca0 * c0 * c1) * c2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * s2) * s3 * ca3
                + ((-s0 * s1 + ca0 * c0 * c1) * sa2 * s2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * ca2 - (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * sa2 * c2) * sa3
                + (-(-s0 * s1 + ca0 * c0 * c1) * s2 * ca2 + (sa1 * s0 * c1 + sa1 * s1 * ca0 * c0) * sa2 + (-s0 * ca1 * c1 - s1 * ca0 * ca1 * c0) * ca2 * c2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[1, 2] = (
        -a2 * (s0 * c1 + s1 * ca0 * c0) * s2
        + a2 * (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3
        + a4 * ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
        + a4
        * (
            -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
        )
        * s4
        + a5
        * (
            ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * c4
            + (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * s4
        )
        * c5
        + a5
        * (
            -((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * s4 * ca4
            + (
                (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
            )
            * sa4
            + (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * s5
        + d3 * ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2)
        + d4
        * (
            (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
        )
        + d5
        * (
            ((-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * s3) * sa4 * s4
            + (
                (-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * sa3 * c3
            )
            * ca4
            - (
                -(-(s0 * c1 + s1 * ca0 * c0) * s2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * c2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * s2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * ca2 * c2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2 * ca2) * ca3 * c3
            )
            * sa4
            * c4
        )
    )
    Jp[1, 3] = (
        -a3 * ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3
        + a3 * (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3
        + a4 * (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
        + a4 * (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
        + a5
        * (
            (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * c4
            + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * s4 * ca4
            + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * sa4
            + (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + d4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3)
        + d5
        * (
            (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * c3) * sa4 * s4
            + (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * s3) * ca4
            - (-((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * ca3 * c3 - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3 * ca3) * sa4 * c4
        )
    )
    Jp[1, 4] = (
        -a4 * (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
        + a4
        * (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * c4
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * c4
        )
        * c5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * ca4 * c4
            - (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
            * ca4
        )
        * s5
        + d5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * sa4
            * s4
        )
    )
    Jp[1, 5] = (
        -a5
        * (
            (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * c4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * s4
        )
        * s5
        + a5
        * (
            -(((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * s4 * ca4
            + (
                ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
                - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
            )
            * sa4
            + (
                -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
                + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
                + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
            )
            * ca4
            * c4
        )
        * c5
    )
    Jp[2, 0] = 0
    Jp[2, 1] = (
        a1 * sa0 * c1
        - a2 * sa0 * s1 * s2 * ca1
        + a2 * sa0 * c1 * c2
        + a3 * (-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3
        + a3 * (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3
        + a4 * ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
        + a4 * (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
        + a5
        * (
            ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * c4
            + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * s4
        )
        * c5
        + a5
        * (
            -((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * s4 * ca4
            + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * sa4
            + (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * ca4 * c4
        )
        * s5
        + d2 * sa0 * sa1 * s1
        + d3 * (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1)
        + d4 * ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3)
        + d5
        * (
            ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * c3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * s3) * sa4 * s4
            + ((-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * sa3 * s3 - (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * sa3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * ca3) * ca4
            - (-(-sa0 * s1 * s2 * ca1 + sa0 * c1 * c2) * s3 * ca3 + (sa0 * sa1 * sa2 * s1 - sa0 * s1 * ca1 * ca2 * c2 - sa0 * s2 * ca2 * c1) * ca3 * c3 + (sa0 * sa1 * s1 * ca2 + sa0 * sa2 * s1 * ca1 * c2 + sa0 * sa2 * s2 * c1) * sa3) * sa4 * c4
        )
    )
    Jp[2, 2] = (
        a2 * (sa0 * ca1 * c1 + sa1 * ca0) * c2
        - a2 * sa0 * s1 * s2
        + a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3
        + a3 * (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3
        + a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
        + a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * s4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * ca4 * c4
        )
        * s5
        + d3 * ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2)
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3)
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * c3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * s3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * sa3 * s3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * ca3 - (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * sa3 * c3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * c2 - sa0 * s1 * s2) * s3 * ca3 + ((sa0 * ca1 * c1 + sa1 * ca0) * sa2 * s2 + sa0 * sa2 * s1 * c2) * sa3 + (-(sa0 * ca1 * c1 + sa1 * ca0) * s2 * ca2 - sa0 * s1 * ca2 * c2) * ca3 * c3) * sa4 * c4
        )
    )
    Jp[2, 3] = (
        -a3 * ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3
        + a3 * ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
        + a5
        * (
            (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * s4
        )
        * c5
        + a5
        * (
            -(-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * ca4 * c4
        )
        * s5
        + d4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3)
        + d5
        * (
            (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * c3) * sa4 * s4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * s3) * ca4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * ca3 * c3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3 * ca3) * sa4 * c4
        )
    )
    Jp[2, 4] = (
        -a4 * (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
        + a4 * (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * c4
        )
        * c5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * ca4 * c4
            - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4 * ca4
        )
        * s5
        + d5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * s4
        )
    )
    Jp[2, 5] = (
        -a5
        * (
            (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * c4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * s4
        )
        * s5
        + a5
        * (
            -(((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * s4 * ca4
            + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * sa4
            + (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * ca4 * c4
        )
        * c5
    )
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = sa0 * s0
    Jr[0, 2] = sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0
    Jr[0, 3] = (-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2
    Jr[0, 4] = (
        ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
        + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
        - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
    )
    Jr[0, 5] = (
        (((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * c3 + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * s3) * sa4 * s4
        + (
            ((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * sa3 * s3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * ca3
            - (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * sa3 * c3
        )
        * ca4
        - (
            -((-s0 * s1 * ca0 + c0 * c1) * c2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * s2) * s3 * ca3
            + ((-s0 * s1 * ca0 + c0 * c1) * sa2 * s2 - (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * sa2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * ca2) * sa3
            + (-(-s0 * s1 * ca0 + c0 * c1) * s2 * ca2 + (sa0 * sa1 * s0 - s0 * ca0 * ca1 * c1 - s1 * ca1 * c0) * ca2 * c2 + (sa0 * s0 * ca1 + sa1 * s0 * ca0 * c1 + sa1 * s1 * c0) * sa2) * ca3 * c3
        )
        * sa4
        * c4
    )
    Jr[1, 0] = 0
    Jr[1, 1] = -sa0 * c0
    Jr[1, 2] = -sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1
    Jr[1, 3] = (s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2
    Jr[1, 4] = (
        ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
        + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
        - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
    )
    Jr[1, 5] = (
        (((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * c3 + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * s3) * sa4 * s4
        + (
            ((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * sa3 * s3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * ca3
            - (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * sa3 * c3
        )
        * ca4
        - (
            -((s0 * c1 + s1 * ca0 * c0) * c2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * s2) * s3 * ca3
            + ((s0 * c1 + s1 * ca0 * c0) * sa2 * s2 - (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * sa2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * ca2) * sa3
            + (-(s0 * c1 + s1 * ca0 * c0) * s2 * ca2 + (-sa0 * sa1 * c0 - s0 * s1 * ca1 + ca0 * ca1 * c0 * c1) * ca2 * c2 + (-sa0 * ca1 * c0 + sa1 * s0 * s1 - sa1 * ca0 * c0 * c1) * sa2) * ca3 * c3
        )
        * sa4
        * c4
    )
    Jr[2, 0] = 1
    Jr[2, 1] = ca0
    Jr[2, 2] = -sa0 * sa1 * c1 + ca0 * ca1
    Jr[2, 3] = (-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2
    Jr[2, 4] = ((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3
    Jr[2, 5] = (
        (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * s3) * sa4 * s4
        + (((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * sa3 * s3 - ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * sa3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * ca3) * ca4
        - (-((sa0 * ca1 * c1 + sa1 * ca0) * s2 + sa0 * s1 * c2) * s3 * ca3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * sa2 + (sa0 * ca1 * c1 + sa1 * ca0) * ca2 * c2 - sa0 * s1 * s2 * ca2) * ca3 * c3 + ((-sa0 * sa1 * c1 + ca0 * ca1) * ca2 - (sa0 * ca1 * c1 + sa1 * ca0) * sa2 * c2 + sa0 * sa2 * s1 * s2) * sa3) * sa4 * c4
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_crx20(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the Fanuc CRX20ia_l.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])
    c4 = np.cos(q[3])
    s4 = np.sin(q[3])
    c5 = np.cos(q[4])
    s5 = np.sin(q[4])
    c6 = np.cos(q[5])
    s6 = np.sin(q[5])

    p1 = 0.127700
    p2 = 0.117300
    p3 = 0.710000
    p4 = 0.540000
    p5 = -0.150000
    p6 = 0.160000

    p = np.array(
        [
            p3 * s2 * c1 + p4 * (s2 * s3 * c1 + c1 * c2 * c3) + p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) + p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
            p3 * s1 * s2 + p4 * (s1 * s2 * s3 + s1 * c2 * c3) + p5 * (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) + p6 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5),
            p1 + p2 + p3 * c2 + p4 * (-s2 * c3 + s3 * c2) - p5 * (s2 * s3 + c2 * c3) * s4 + p6 * ((s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5),
        ]
    )

    R = np.array(
        [
            [
                ((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5,
                -(((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * s6 + (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * c6,
                (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * c6 + (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s6,
            ],
            [
                ((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5,
                -(((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * s6 + (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * c6,
                (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * c6 + (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s6,
            ],
            [(s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5, -(s2 * s3 + c2 * c3) * s4 * c6 - ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * s6, -(s2 * s3 + c2 * c3) * s4 * s6 + ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * c6],
        ]
    )

    Jp = np.array(
        [
            [
                -p3 * s1 * s2 + p4 * (-s1 * s2 * s3 - s1 * c2 * c3) + p5 * (-(-s1 * s2 * c3 + s1 * s3 * c2) * s4 - c1 * c4) + p6 * (((-s1 * s2 * c3 + s1 * s3 * c2) * c4 - s4 * c1) * s5 + (-s1 * s2 * s3 - s1 * c2 * c3) * c5),
                p3 * c1 * c2 + p4 * (-s2 * c1 * c3 + s3 * c1 * c2) - p5 * (s2 * s3 * c1 + c1 * c2 * c3) * s4 + p6 * ((s2 * s3 * c1 + c1 * c2 * c3) * s5 * c4 + (-s2 * c1 * c3 + s3 * c1 * c2) * c5),
                p4 * (s2 * c1 * c3 - s3 * c1 * c2) - p5 * (-s2 * s3 * c1 - c1 * c2 * c3) * s4 + p6 * ((-s2 * s3 * c1 - c1 * c2 * c3) * s5 * c4 + (s2 * c1 * c3 - s3 * c1 * c2) * c5),
                p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * c4 + s1 * s4) + p6 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s5,
                p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5),
                0,
            ],
            [
                p3 * s2 * c1 + p4 * (s2 * s3 * c1 + c1 * c2 * c3) + p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) + p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
                p3 * s1 * c2 + p4 * (-s1 * s2 * c3 + s1 * s3 * c2) - p5 * (s1 * s2 * s3 + s1 * c2 * c3) * s4 + p6 * ((s1 * s2 * s3 + s1 * c2 * c3) * s5 * c4 + (-s1 * s2 * c3 + s1 * s3 * c2) * c5),
                p4 * (s1 * s2 * c3 - s1 * s3 * c2) - p5 * (-s1 * s2 * s3 - s1 * c2 * c3) * s4 + p6 * ((-s1 * s2 * s3 - s1 * c2 * c3) * s5 * c4 + (s1 * s2 * c3 - s1 * s3 * c2) * c5),
                p5 * (-(s1 * s2 * c3 - s1 * s3 * c2) * c4 - s4 * c1) + p6 * (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s5,
                p6 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5),
                0,
            ],
            [
                0,
                -p3 * s2 + p4 * (-s2 * s3 - c2 * c3) - p5 * (-s2 * c3 + s3 * c2) * s4 + p6 * ((-s2 * s3 - c2 * c3) * c5 + (-s2 * c3 + s3 * c2) * s5 * c4),
                p4 * (s2 * s3 + c2 * c3) - p5 * (s2 * c3 - s3 * c2) * s4 + p6 * ((s2 * s3 + c2 * c3) * c5 + (s2 * c3 - s3 * c2) * s5 * c4),
                -p5 * (s2 * s3 + c2 * c3) * c4 - p6 * (s2 * s3 + c2 * c3) * s4 * s5,
                p6 * ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5),
                0,
            ],
        ]
    )

    Jr = np.array(
        [
            [0, -s1, s1, -s2 * s3 * c1 - c1 * c2 * c3, (s2 * c1 * c3 - s3 * c1 * c2) * s4 + s1 * c4, -((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 - (s2 * s3 * c1 + c1 * c2 * c3) * c5],
            [0, c1, -c1, -s1 * s2 * s3 - s1 * c2 * c3, (s1 * s2 * c3 - s1 * s3 * c2) * s4 - c1 * c4, -((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 - (s1 * s2 * s3 + s1 * c2 * c3) * c5],
            [1, 0, 0, s2 * c3 - s3 * c2, (s2 * s3 + c2 * c3) * s4, -(s2 * s3 + c2 * c3) * s5 * c4 - (-s2 * c3 + s3 * c2) * c5],
        ]
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

    """
    Compute forward kinematics and Jacobian for the Fanuc CRX20ia_l.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : np.ndarray
        Tool centre point (optional).
    out : string
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])
    c4 = np.cos(q[3])
    s4 = np.sin(q[3])
    c5 = np.cos(q[4])
    s5 = np.sin(q[4])
    c6 = np.cos(q[5])
    s6 = np.sin(q[5])

    p1 = 0.127700
    p2 = 0.117300
    p3 = 0.710000
    p4 = 0.540000
    p5 = -0.150000
    p6 = 0.160000

    p = np.array(
        [
            p3 * s2 * c1 + p4 * (s2 * s3 * c1 + c1 * c2 * c3) + p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) + p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
            p3 * s1 * s2 + p4 * (s1 * s2 * s3 + s1 * c2 * c3) + p5 * (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) + p6 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5),
            p1 + p2 + p3 * c2 + p4 * (-s2 * c3 + s3 * c2) - p5 * (s2 * s3 + c2 * c3) * s4 + p6 * ((s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5),
        ]
    )

    R = np.array(
        [
            [
                ((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5,
                -(((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * s6 + (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * c6,
                (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * c6 + (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s6,
            ],
            [
                ((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5,
                -(((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * s6 + (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * c6,
                (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * c6 + (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s6,
            ],
            [(s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5, -(s2 * s3 + c2 * c3) * s4 * c6 - ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * s6, -(s2 * s3 + c2 * c3) * s4 * s6 + ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * c6],
        ]
    )

    Jp = np.array(
        [
            [
                -p3 * s1 * s2 + p4 * (-s1 * s2 * s3 - s1 * c2 * c3) + p5 * (-(-s1 * s2 * c3 + s1 * s3 * c2) * s4 - c1 * c4) + p6 * (((-s1 * s2 * c3 + s1 * s3 * c2) * c4 - s4 * c1) * s5 + (-s1 * s2 * s3 - s1 * c2 * c3) * c5),
                p3 * c1 * c2 + p4 * (-s2 * c1 * c3 + s3 * c1 * c2) - p5 * (s2 * s3 * c1 + c1 * c2 * c3) * s4 + p6 * ((s2 * s3 * c1 + c1 * c2 * c3) * s5 * c4 + (-s2 * c1 * c3 + s3 * c1 * c2) * c5),
                p4 * (s2 * c1 * c3 - s3 * c1 * c2) - p5 * (-s2 * s3 * c1 - c1 * c2 * c3) * s4 + p6 * ((-s2 * s3 * c1 - c1 * c2 * c3) * s5 * c4 + (s2 * c1 * c3 - s3 * c1 * c2) * c5),
                p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * c4 + s1 * s4) + p6 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s5,
                p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5),
                0,
            ],
            [
                p3 * s2 * c1 + p4 * (s2 * s3 * c1 + c1 * c2 * c3) + p5 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) + p6 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
                p3 * s1 * c2 + p4 * (-s1 * s2 * c3 + s1 * s3 * c2) - p5 * (s1 * s2 * s3 + s1 * c2 * c3) * s4 + p6 * ((s1 * s2 * s3 + s1 * c2 * c3) * s5 * c4 + (-s1 * s2 * c3 + s1 * s3 * c2) * c5),
                p4 * (s1 * s2 * c3 - s1 * s3 * c2) - p5 * (-s1 * s2 * s3 - s1 * c2 * c3) * s4 + p6 * ((-s1 * s2 * s3 - s1 * c2 * c3) * s5 * c4 + (s1 * s2 * c3 - s1 * s3 * c2) * c5),
                p5 * (-(s1 * s2 * c3 - s1 * s3 * c2) * c4 - s4 * c1) + p6 * (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s5,
                p6 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5),
                0,
            ],
            [
                0,
                -p3 * s2 + p4 * (-s2 * s3 - c2 * c3) - p5 * (-s2 * c3 + s3 * c2) * s4 + p6 * ((-s2 * s3 - c2 * c3) * c5 + (-s2 * c3 + s3 * c2) * s5 * c4),
                p4 * (s2 * s3 + c2 * c3) - p5 * (s2 * c3 - s3 * c2) * s4 + p6 * ((s2 * s3 + c2 * c3) * c5 + (s2 * c3 - s3 * c2) * s5 * c4),
                -p5 * (s2 * s3 + c2 * c3) * c4 - p6 * (s2 * s3 + c2 * c3) * s4 * s5,
                p6 * ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5),
                0,
            ],
        ]
    )

    Jr = np.array(
        [
            [0, -s1, s1, -s2 * s3 * c1 - c1 * c2 * c3, (s2 * c1 * c3 - s3 * c1 * c2) * s4 + s1 * c4, -((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 - (s2 * s3 * c1 + c1 * c2 * c3) * c5],
            [0, c1, -c1, -s1 * s2 * s3 - s1 * c2 * c3, (s1 * s2 * c3 - s1 * s3 * c2) * s4 - c1 * c4, -((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 - (s1 * s2 * s3 + s1 * c2 * c3) * c5],
            [1, 0, 0, s2 * c3 - s3 * c2, (s2 * s3 + c2 * c3) * s4, -(s2 * s3 + c2 * c3) * s5 * c4 - (-s2 * c3 + s3 * c2) * c5],
        ]
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_hc20(q: np.ndarray, tcp: Optional[TCPType] = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the Yaskawa HC20.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : TCPType, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])
    c4 = np.cos(q[3])
    s4 = np.sin(q[3])
    c5 = np.cos(q[4])
    s5 = np.sin(q[4])
    c6 = np.cos(q[5])
    s6 = np.sin(q[5])

    p1 = 0.380000
    p2 = 0.820000
    p3 = 0.880000
    p4 = 0.200000

    p = np.array(
        [
            p2 * s2 * c1 + p3 * (s2 * s3 * c1 + c1 * c2 * c3) + p4 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
            p2 * s1 * s2 + p3 * (s1 * s2 * s3 + s1 * c2 * c3) + p4 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5),
            p1 + p2 * c2 + p3 * (-s2 * c3 + s3 * c2) + p4 * ((s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5),
        ]
    )

    R = np.array(
        [
            [
                (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * c6 + (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s6,
                (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5) * s6 - (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * c6,
                ((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5,
            ],
            [
                (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * c6 + (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s6,
                (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5) * s6 - (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * c6,
                ((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 + (s1 * s2 * s3 + s1 * c2 * c3) * c5,
            ],
            [-(s2 * s3 + c2 * c3) * s4 * s6 + ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * c6, (s2 * s3 + c2 * c3) * s4 * c6 + ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5) * s6, (s2 * s3 + c2 * c3) * s5 * c4 + (-s2 * c3 + s3 * c2) * c5],
        ]
    )

    Jp = np.array(
        [
            [
                -p2 * s1 * s2 + p3 * (-s1 * s2 * s3 - s1 * c2 * c3) + p4 * (((-s1 * s2 * c3 + s1 * s3 * c2) * c4 - s4 * c1) * s5 + (-s1 * s2 * s3 - s1 * c2 * c3) * c5),
                p2 * c1 * c2 + p3 * (-s2 * c1 * c3 + s3 * c1 * c2) + p4 * ((s2 * s3 * c1 + c1 * c2 * c3) * s5 * c4 + (-s2 * c1 * c3 + s3 * c1 * c2) * c5),
                p3 * (s2 * c1 * c3 - s3 * c1 * c2) + p4 * ((-s2 * s3 * c1 - c1 * c2 * c3) * s5 * c4 + (s2 * c1 * c3 - s3 * c1 * c2) * c5),
                p4 * (-(s2 * c1 * c3 - s3 * c1 * c2) * s4 - s1 * c4) * s5,
                p4 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * c5 - (s2 * s3 * c1 + c1 * c2 * c3) * s5),
                0,
            ],
            [
                p2 * s2 * c1 + p3 * (s2 * s3 * c1 + c1 * c2 * c3) + p4 * (((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 + (s2 * s3 * c1 + c1 * c2 * c3) * c5),
                p2 * s1 * c2 + p3 * (-s1 * s2 * c3 + s1 * s3 * c2) + p4 * ((s1 * s2 * s3 + s1 * c2 * c3) * s5 * c4 + (-s1 * s2 * c3 + s1 * s3 * c2) * c5),
                p3 * (s1 * s2 * c3 - s1 * s3 * c2) + p4 * ((-s1 * s2 * s3 - s1 * c2 * c3) * s5 * c4 + (s1 * s2 * c3 - s1 * s3 * c2) * c5),
                p4 * (-(s1 * s2 * c3 - s1 * s3 * c2) * s4 + c1 * c4) * s5,
                p4 * (((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * c5 - (s1 * s2 * s3 + s1 * c2 * c3) * s5),
                0,
            ],
            [0, -p2 * s2 + p3 * (-s2 * s3 - c2 * c3) + p4 * ((-s2 * s3 - c2 * c3) * c5 + (-s2 * c3 + s3 * c2) * s5 * c4), p3 * (s2 * s3 + c2 * c3) + p4 * ((s2 * s3 + c2 * c3) * c5 + (s2 * c3 - s3 * c2) * s5 * c4), -p4 * (s2 * s3 + c2 * c3) * s4 * s5, p4 * ((s2 * s3 + c2 * c3) * c4 * c5 - (-s2 * c3 + s3 * c2) * s5), 0],
        ]
    )

    Jr = np.array(
        [
            [0, -s1, s1, -s2 * s3 * c1 - c1 * c2 * c3, (s2 * c1 * c3 - s3 * c1 * c2) * s4 + s1 * c4, -((s2 * c1 * c3 - s3 * c1 * c2) * c4 - s1 * s4) * s5 - (s2 * s3 * c1 + c1 * c2 * c3) * c5],
            [0, c1, -c1, -s1 * s2 * s3 - s1 * c2 * c3, (s1 * s2 * c3 - s1 * s3 * c2) * s4 - c1 * c4, -((s1 * s2 * c3 - s1 * s3 * c2) * c4 + s4 * c1) * s5 - (s1 * s2 * s3 + s1 * c2 * c3) * c5],
            [1, 0, 0, s2 * c3 - s3 * c2, (s2 * s3 + c2 * c3) * s4, -(s2 * s3 + c2 * c3) * s5 * c4 - (-s2 * c3 + s3 * c2) * c5],
        ]
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_z1_DH(q: np.ndarray, tcp: np.ndarray = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the Unitree z1.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : np.ndarray, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c2 = np.cos(q[2] + -0.25574251)
    s2 = np.sin(q[2] + -0.25574251)
    c3 = np.cos(q[3] + 0.25574251)
    s3 = np.sin(q[3] + 0.25574251)
    c4 = np.cos(q[4] + -np.pi / 2)
    s4 = np.sin(q[4] + -np.pi / 2)
    c5 = np.cos(q[5] + np.pi / 2)
    s5 = np.sin(q[5] + np.pi / 2)

    a1 = -0.35
    a2 = 0.22534
    a3 = 0.07

    d0 = 0.1045
    d5 = 0.2492

    p = np.zeros(3)
    p[0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + a3 * (-s1 * s2 * c0 + c0 * c1 * c2) * c3 + a3 * (-s1 * c0 * c2 - s2 * c0 * c1) * s3 + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 - s0 * c4)
    p[1] = a1 * s0 * c1 - a2 * s0 * s1 * s2 + a2 * s0 * c1 * c2 + a3 * (-s0 * s1 * s2 + s0 * c1 * c2) * c3 + a3 * (-s0 * s1 * c2 - s0 * s2 * c1) * s3 + d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 + c0 * c4)
    p[2] = -a1 * s1 - a2 * s1 * c2 - a2 * s2 * c1 + a3 * (s1 * s2 - c1 * c2) * s3 + a3 * (-s1 * c2 - s2 * c1) * c3 + d0 - d5 * ((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * s4
    R = np.zeros((3, 3))
    R[0, 0] = (((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4) * c5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s5
    R[0, 1] = -(((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 - s0 * s4) * s5 + (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * c5
    R[0, 2] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 - s0 * c4
    R[1, 0] = (((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0) * c5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s5
    R[1, 1] = -(((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 + s4 * c0) * s5 + (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * c5
    R[1, 2] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 + c0 * c4
    R[2, 0] = ((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * c4 * c5 + ((s1 * s2 - c1 * c2) * c3 - (-s1 * c2 - s2 * c1) * s3) * s5
    R[2, 1] = -((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * s5 * c4 + ((s1 * s2 - c1 * c2) * c3 - (-s1 * c2 - s2 * c1) * s3) * c5
    R[2, 2] = -((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1 * s0 * c1 + a2 * s0 * s1 * s2 - a2 * s0 * c1 * c2 + a3 * (s0 * s1 * s2 - s0 * c1 * c2) * c3 + a3 * (s0 * s1 * c2 + s0 * s2 * c1) * s3 + d5 * (-((s0 * s1 * s2 - s0 * c1 * c2) * c3 + (s0 * s1 * c2 + s0 * s2 * c1) * s3) * s4 - c0 * c4)
    Jp[0, 1] = -a1 * s1 * c0 - a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + a3 * (s1 * s2 * c0 - c0 * c1 * c2) * s3 + a3 * (-s1 * c0 * c2 - s2 * c0 * c1) * c3 - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 2] = -a2 * s1 * c0 * c2 - a2 * s2 * c0 * c1 + a3 * (s1 * s2 * c0 - c0 * c1 * c2) * s3 + a3 * (-s1 * c0 * c2 - s2 * c0 * c1) * c3 - d5 * ((s1 * s2 * c0 - c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 3] = -a3 * (-s1 * s2 * c0 + c0 * c1 * c2) * s3 + a3 * (-s1 * c0 * c2 - s2 * c0 * c1) * c3 - d5 * (-(-s1 * s2 * c0 + c0 * c1 * c2) * s3 + (-s1 * c0 * c2 - s2 * c0 * c1) * c3) * s4
    Jp[0, 4] = d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * c4 + s0 * s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1 * c0 * c1 - a2 * s1 * s2 * c0 + a2 * c0 * c1 * c2 + a3 * (-s1 * s2 * c0 + c0 * c1 * c2) * c3 + a3 * (-s1 * c0 * c2 - s2 * c0 * c1) * s3 + d5 * (-((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 - s0 * c4)
    Jp[1, 1] = -a1 * s0 * s1 - a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + a3 * (s0 * s1 * s2 - s0 * c1 * c2) * s3 + a3 * (-s0 * s1 * c2 - s0 * s2 * c1) * c3 - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 2] = -a2 * s0 * s1 * c2 - a2 * s0 * s2 * c1 + a3 * (s0 * s1 * s2 - s0 * c1 * c2) * s3 + a3 * (-s0 * s1 * c2 - s0 * s2 * c1) * c3 - d5 * ((s0 * s1 * s2 - s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 3] = -a3 * (-s0 * s1 * s2 + s0 * c1 * c2) * s3 + a3 * (-s0 * s1 * c2 - s0 * s2 * c1) * c3 - d5 * (-(-s0 * s1 * s2 + s0 * c1 * c2) * s3 + (-s0 * s1 * c2 - s0 * s2 * c1) * c3) * s4
    Jp[1, 4] = d5 * (-((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * c4 - s4 * c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -a1 * c1 + a2 * s1 * s2 - a2 * c1 * c2 + a3 * (s1 * s2 - c1 * c2) * c3 + a3 * (s1 * c2 + s2 * c1) * s3 - d5 * ((s1 * s2 - c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 2] = a2 * s1 * s2 - a2 * c1 * c2 + a3 * (s1 * s2 - c1 * c2) * c3 + a3 * (s1 * c2 + s2 * c1) * s3 - d5 * ((s1 * s2 - c1 * c2) * c3 + (s1 * c2 + s2 * c1) * s3) * s4
    Jp[2, 3] = a3 * (s1 * s2 - c1 * c2) * c3 - a3 * (-s1 * c2 - s2 * c1) * s3 - d5 * ((s1 * s2 - c1 * c2) * c3 - (-s1 * c2 - s2 * c1) * s3) * s4
    Jp[2, 4] = -d5 * ((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * c4
    Jp[2, 5] = 0
    Jr = np.zeros((3, 6))
    Jr[0, 0] = 0
    Jr[0, 1] = -s0
    Jr[0, 2] = -s0
    Jr[0, 3] = -s0
    Jr[0, 4] = (-s1 * s2 * c0 + c0 * c1 * c2) * s3 - (-s1 * c0 * c2 - s2 * c0 * c1) * c3
    Jr[0, 5] = -((-s1 * s2 * c0 + c0 * c1 * c2) * c3 + (-s1 * c0 * c2 - s2 * c0 * c1) * s3) * s4 - s0 * c4
    Jr[1, 0] = 0
    Jr[1, 1] = c0
    Jr[1, 2] = c0
    Jr[1, 3] = c0
    Jr[1, 4] = (-s0 * s1 * s2 + s0 * c1 * c2) * s3 - (-s0 * s1 * c2 - s0 * s2 * c1) * c3
    Jr[1, 5] = -((-s0 * s1 * s2 + s0 * c1 * c2) * c3 + (-s0 * s1 * c2 - s0 * s2 * c1) * s3) * s4 + c0 * c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(s1 * s2 - c1 * c2) * c3 + (-s1 * c2 - s2 * c1) * s3
    Jr[2, 5] = -((s1 * s2 - c1 * c2) * s3 + (-s1 * c2 - s2 * c1) * c3) * s4

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J


def kinmodel_z1(q: np.ndarray, tcp: np.ndarray = None, out: str = "x") -> list:
    """
    Compute forward kinematics and Jacobian for the Z! MJCF.

    Parameters
    ----------
    q : np.ndarray
        Joint angles/positions.
    tcp : np.ndarray, optional
        Tool centre point (optional).
    out : str, optional
        Output form (optional).

    Returns
    -------
    p : np.ndarray
        Position of the end effector.
    R : np.ndarray
        Rotation matrix of the end effector (3, 3).
    J : np.ndarray
        Jacobian matrix (6, nj).
    """

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])
    c4 = np.cos(q[3])
    s4 = np.sin(q[3])
    c5 = np.cos(q[4])
    s5 = np.sin(q[4])
    c6 = np.cos(q[5])
    s6 = np.sin(q[5])

    p1 = 0.058500
    p2 = 0.045000
    p3 = -0.350000
    p4 = 0.218000
    p5 = 0.057000
    p6 = 0.070000
    p7 = 0.049200

    p = np.array(
        [
            p3 * c1 * c2 + p4 * (-s2 * s3 * c1 + c1 * c2 * c3) + p5 * (s2 * c1 * c3 + s3 * c1 * c2) + p6 * ((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) + p7 * (((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * c5 - s1 * s5),
            p3 * s1 * c2 + p4 * (-s1 * s2 * s3 + s1 * c2 * c3) + p5 * (s1 * s2 * c3 + s1 * s3 * c2) + p6 * ((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) + p7 * (((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * c5 + s5 * c1),
            p1 + p2 - p3 * s2 + p4 * (-s2 * c3 - s3 * c2) + p5 * (-s2 * s3 + c2 * c3) + p6 * (-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) + p7 * (-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * c5,
        ]
    )

    R = np.array(
        [
            [
                (-((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * s5 - s1 * c5) * s6 - ((-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (s2 * c1 * c3 + s3 * c1 * c2) * c4) * c6,
                (-((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * s5 - s1 * c5) * c6 + ((-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (s2 * c1 * c3 + s3 * c1 * c2) * c4) * s6,
                ((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * c5 - s1 * s5,
            ],
            [
                (-((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * s5 + c1 * c5) * s6 - ((-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (s1 * s2 * c3 + s1 * s3 * c2) * c4) * c6,
                (-((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * s5 + c1 * c5) * c6 + ((-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (s1 * s2 * c3 + s1 * s3 * c2) * c4) * s6,
                ((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * c5 + s5 * c1,
            ],
            [
                -(-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * s5 * s6 - ((-s2 * s3 + c2 * c3) * c4 + (-s2 * c3 - s3 * c2) * s4) * c6,
                -(-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * s5 * c6 + ((-s2 * s3 + c2 * c3) * c4 + (-s2 * c3 - s3 * c2) * s4) * s6,
                (-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * c5,
            ],
        ]
    )

    Jp = np.array(
        [
            [
                -p3 * s1 * c2 + p4 * (s1 * s2 * s3 - s1 * c2 * c3) + p5 * (-s1 * s2 * c3 - s1 * s3 * c2) + p6 * ((s1 * s2 * s3 - s1 * c2 * c3) * c4 - (-s1 * s2 * c3 - s1 * s3 * c2) * s4) + p7 * (((s1 * s2 * s3 - s1 * c2 * c3) * c4 - (-s1 * s2 * c3 - s1 * s3 * c2) * s4) * c5 - s5 * c1),
                -p3 * s2 * c1 + p4 * (-s2 * c1 * c3 - s3 * c1 * c2) + p5 * (-s2 * s3 * c1 + c1 * c2 * c3) + p6 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (-s2 * c1 * c3 - s3 * c1 * c2) * c4) + p7 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (-s2 * c1 * c3 - s3 * c1 * c2) * c4) * c5,
                p4 * (-s2 * c1 * c3 - s3 * c1 * c2) + p5 * (-s2 * s3 * c1 + c1 * c2 * c3) + p6 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (-s2 * c1 * c3 - s3 * c1 * c2) * c4) + p7 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (-s2 * c1 * c3 - s3 * c1 * c2) * c4) * c5,
                p6 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 - (s2 * c1 * c3 + s3 * c1 * c2) * c4) + p7 * (-(-s2 * s3 * c1 + c1 * c2 * c3) * s4 - (s2 * c1 * c3 + s3 * c1 * c2) * c4) * c5,
                p7 * (-((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * s5 - s1 * c5),
                0,
            ],
            [
                p3 * c1 * c2 + p4 * (-s2 * s3 * c1 + c1 * c2 * c3) + p5 * (s2 * c1 * c3 + s3 * c1 * c2) + p6 * ((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) + p7 * (((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * c5 - s1 * s5),
                -p3 * s1 * s2 + p4 * (-s1 * s2 * c3 - s1 * s3 * c2) + p5 * (-s1 * s2 * s3 + s1 * c2 * c3) + p6 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (-s1 * s2 * c3 - s1 * s3 * c2) * c4) + p7 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (-s1 * s2 * c3 - s1 * s3 * c2) * c4) * c5,
                p4 * (-s1 * s2 * c3 - s1 * s3 * c2) + p5 * (-s1 * s2 * s3 + s1 * c2 * c3) + p6 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (-s1 * s2 * c3 - s1 * s3 * c2) * c4) + p7 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (-s1 * s2 * c3 - s1 * s3 * c2) * c4) * c5,
                p6 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 - (s1 * s2 * c3 + s1 * s3 * c2) * c4) + p7 * (-(-s1 * s2 * s3 + s1 * c2 * c3) * s4 - (s1 * s2 * c3 + s1 * s3 * c2) * c4) * c5,
                p7 * (-((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * s5 + c1 * c5),
                0,
            ],
            [
                0,
                -p3 * c2 + p4 * (s2 * s3 - c2 * c3) + p5 * (-s2 * c3 - s3 * c2) + p6 * ((s2 * s3 - c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) + p7 * ((s2 * s3 - c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) * c5,
                p4 * (s2 * s3 - c2 * c3) + p5 * (-s2 * c3 - s3 * c2) + p6 * ((s2 * s3 - c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) + p7 * ((s2 * s3 - c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) * c5,
                p6 * (-(-s2 * s3 + c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) + p7 * (-(-s2 * s3 + c2 * c3) * c4 - (-s2 * c3 - s3 * c2) * s4) * c5,
                -p7 * (-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * s5,
                0,
            ],
        ]
    )

    Jr = np.array(
        [
            [0, -s1, -s1, -s1, (-s2 * s3 * c1 + c1 * c2 * c3) * s4 + (s2 * c1 * c3 + s3 * c1 * c2) * c4, ((-s2 * s3 * c1 + c1 * c2 * c3) * c4 - (s2 * c1 * c3 + s3 * c1 * c2) * s4) * c5 - s1 * s5],
            [0, c1, c1, c1, (-s1 * s2 * s3 + s1 * c2 * c3) * s4 + (s1 * s2 * c3 + s1 * s3 * c2) * c4, ((-s1 * s2 * s3 + s1 * c2 * c3) * c4 - (s1 * s2 * c3 + s1 * s3 * c2) * s4) * c5 + s5 * c1],
            [1, 0, 0, 0, (-s2 * s3 + c2 * c3) * c4 + (-s2 * c3 - s3 * c2) * s4, (-(-s2 * s3 + c2 * c3) * s4 + (-s2 * c3 - s3 * c2) * c4) * c5],
        ]
    )

    if tcp is not None:
        tcp = np.array(tcp)
        if tcp.shape == (4, 4):
            p_tcp = tcp[:3, 3]
            R_tcp = tcp[:3, :3]
        elif tcp.shape[0] == 3:
            p_tcp = tcp[:3]
            R_tcp = np.eye(3)
        elif tcp.shape[0] == 7:
            p_tcp = tcp[:3]
            R_tcp = map_pose(Q=tcp[3:7], out="R")
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(RPY=tcp[3:6], out="R")
        else:
            raise ValueError("kinmodel: tcp is not SE3")
        v = R @ p_tcp
        s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out == "pR":
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J
