"""Robot models

Copyright (c) 2024 by IJS Leon Zlajpah 

"""
import numpy as np 
from robotblockset.transformations import map_pose 

def kinmodel_panda(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = -a2*s0*s2 + a2*c0*c1*c2 + a3*(-s0*s2 + c0*c1*c2)*c3 + a3*s1*s3*c0 + a5*(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + a5*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5 + d2*s1*c0 + d4*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3) + d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c5)
    p[1] = a2*s0*c1*c2 + a2*s2*c0 + a3*(s0*c1*c2 + s2*c0)*c3 + a3*s0*s1*s3 + a5*(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + a5*(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5 + d2*s0*s1 + d4*(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3) + d6*((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*c5)
    p[2] = -a2*s1*c2 - a3*s1*c2*c3 + a3*s3*c1 + a5*((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + a5*(s1*s3*c2 + c1*c3)*s5 + d0 + d2*c1 + d4*(s1*s3*c2 + c1*c3) + d6*(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (s1*s3*c2 + c1*c3)*c5)
    R = np.zeros((3,3))
    R[0, 0] = ((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5)*c6 + (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 - (-s0*c2 - s2*c0*c1)*c4)*s6
    R[0, 1] = -((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5)*s6 + (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 - (-s0*c2 - s2*c0*c1)*c4)*c6
    R[0, 2] = (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c5
    R[1, 0] = ((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5)*c6 + (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 - (-s0*s2*c1 + c0*c2)*c4)*s6
    R[1, 1] = -((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5)*s6 + (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 - (-s0*s2*c1 + c0*c2)*c4)*c6
    R[1, 2] = (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*c5
    R[2, 0] = (((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (s1*s3*c2 + c1*c3)*s5)*c6 + ((-s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*s6
    R[2, 1] = -(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (s1*s3*c2 + c1*c3)*s5)*s6 + ((-s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6
    R[2, 2] = ((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (s1*s3*c2 + c1*c3)*c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = -a2*s0*c1*c2 - a2*s2*c0 + a3*(-s0*c1*c2 - s2*c0)*c3 - a3*s0*s1*s3 + a5*(((-s0*c1*c2 - s2*c0)*c3 - s0*s1*s3)*c4 + (s0*s2*c1 - c0*c2)*s4)*c5 + a5*(-(-s0*c1*c2 - s2*c0)*s3 - s0*s1*c3)*s5 - d2*s0*s1 + d4*(-(-s0*c1*c2 - s2*c0)*s3 - s0*s1*c3) + d6*((((-s0*c1*c2 - s2*c0)*c3 - s0*s1*s3)*c4 + (s0*s2*c1 - c0*c2)*s4)*s5 - (-(-s0*c1*c2 - s2*c0)*s3 - s0*s1*c3)*c5)
    Jp[0, 1] = -a2*s1*c0*c2 - a3*s1*c0*c2*c3 + a3*s3*c0*c1 + a5*((-s1*c0*c2*c3 + s3*c0*c1)*c4 + s1*s2*s4*c0)*c5 + a5*(s1*s3*c0*c2 + c0*c1*c3)*s5 + d2*c0*c1 + d4*(s1*s3*c0*c2 + c0*c1*c3) + d6*(((-s1*c0*c2*c3 + s3*c0*c1)*c4 + s1*s2*s4*c0)*s5 - (s1*s3*c0*c2 + c0*c1*c3)*c5)
    Jp[0, 2] = -a2*s0*c2 - a2*s2*c0*c1 + a3*(-s0*c2 - s2*c0*c1)*c3 + a5*((s0*s2 - c0*c1*c2)*s4 + (-s0*c2 - s2*c0*c1)*c3*c4)*c5 - a5*(-s0*c2 - s2*c0*c1)*s3*s5 - d4*(-s0*c2 - s2*c0*c1)*s3 + d6*(((s0*s2 - c0*c1*c2)*s4 + (-s0*c2 - s2*c0*c1)*c3*c4)*s5 + (-s0*c2 - s2*c0*c1)*s3*c5)
    Jp[0, 3] = -a3*(-s0*s2 + c0*c1*c2)*s3 + a3*s1*c0*c3 + a5*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c4*c5 + a5*(-(-s0*s2 + c0*c1*c2)*c3 - s1*s3*c0)*s5 + d4*(-(-s0*s2 + c0*c1*c2)*c3 - s1*s3*c0) + d6*((-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5*c4 - (-(-s0*s2 + c0*c1*c2)*c3 - s1*s3*c0)*c5)
    Jp[0, 4] = a5*(-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*c5 + d6*(-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*s5
    Jp[0, 5] = -a5*(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 + a5*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c5 + d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5)
    Jp[0, 6] = 0
    Jp[1, 0] = -a2*s0*s2 + a2*c0*c1*c2 + a3*(-s0*s2 + c0*c1*c2)*c3 + a3*s1*s3*c0 + a5*(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + a5*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5 + d2*s1*c0 + d4*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3) + d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c5)
    Jp[1, 1] = -a2*s0*s1*c2 - a3*s0*s1*c2*c3 + a3*s0*s3*c1 + a5*((-s0*s1*c2*c3 + s0*s3*c1)*c4 + s0*s1*s2*s4)*c5 + a5*(s0*s1*s3*c2 + s0*c1*c3)*s5 + d2*s0*c1 + d4*(s0*s1*s3*c2 + s0*c1*c3) + d6*(((-s0*s1*c2*c3 + s0*s3*c1)*c4 + s0*s1*s2*s4)*s5 - (s0*s1*s3*c2 + s0*c1*c3)*c5)
    Jp[1, 2] = -a2*s0*s2*c1 + a2*c0*c2 + a3*(-s0*s2*c1 + c0*c2)*c3 + a5*((-s0*s2*c1 + c0*c2)*c3*c4 + (-s0*c1*c2 - s2*c0)*s4)*c5 - a5*(-s0*s2*c1 + c0*c2)*s3*s5 - d4*(-s0*s2*c1 + c0*c2)*s3 + d6*(((-s0*s2*c1 + c0*c2)*c3*c4 + (-s0*c1*c2 - s2*c0)*s4)*s5 + (-s0*s2*c1 + c0*c2)*s3*c5)
    Jp[1, 3] = -a3*(s0*c1*c2 + s2*c0)*s3 + a3*s0*s1*c3 + a5*(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*c4*c5 + a5*(-(s0*c1*c2 + s2*c0)*c3 - s0*s1*s3)*s5 + d4*(-(s0*c1*c2 + s2*c0)*c3 - s0*s1*s3) + d6*((-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5*c4 - (-(s0*c1*c2 + s2*c0)*c3 - s0*s1*s3)*c5)
    Jp[1, 4] = a5*(-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*c5 + d6*(-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*s5
    Jp[1, 5] = -a5*(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 + a5*(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*c5 + d6*((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -a2*c1*c2 - a3*s1*s3 - a3*c1*c2*c3 + a5*((-s1*s3 - c1*c2*c3)*c4 + s2*s4*c1)*c5 + a5*(-s1*c3 + s3*c1*c2)*s5 - d2*s1 + d4*(-s1*c3 + s3*c1*c2) + d6*(((-s1*s3 - c1*c2*c3)*c4 + s2*s4*c1)*s5 - (-s1*c3 + s3*c1*c2)*c5)
    Jp[2, 2] = a2*s1*s2 + a3*s1*s2*c3 + a5*(s1*s2*c3*c4 + s1*s4*c2)*c5 - a5*s1*s2*s3*s5 - d4*s1*s2*s3 + d6*((s1*s2*c3*c4 + s1*s4*c2)*s5 + s1*s2*s3*c5)
    Jp[2, 3] = a3*s1*s3*c2 + a3*c1*c3 + a5*(s1*s3*c2 + c1*c3)*c4*c5 + a5*(s1*c2*c3 - s3*c1)*s5 + d4*(s1*c2*c3 - s3*c1) + d6*((s1*s3*c2 + c1*c3)*s5*c4 - (s1*c2*c3 - s3*c1)*c5)
    Jp[2, 4] = a5*(-(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c5 + d6*(-(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s5
    Jp[2, 5] = -a5*((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 + a5*(s1*s3*c2 + c1*c3)*c5 + d6*(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (s1*s3*c2 + c1*c3)*s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3,7))
    Jr[0, 0] = 0
    Jr[0, 1] = -s0
    Jr[0, 2] = s1*c0
    Jr[0, 3] = s0*c2 + s2*c0*c1
    Jr[0, 4] = -(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3
    Jr[0, 5] = ((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 - (-s0*c2 - s2*c0*c1)*c4
    Jr[0, 6] = (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - (-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*c5
    Jr[1, 0] = 0
    Jr[1, 1] = c0
    Jr[1, 2] = s0*s1
    Jr[1, 3] = s0*s2*c1 - c0*c2
    Jr[1, 4] = -(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3
    Jr[1, 5] = ((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 - (-s0*s2*c1 + c0*c2)*c4
    Jr[1, 6] = (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - (-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1*s2
    Jr[2, 4] = s1*s3*c2 + c1*c3
    Jr[2, 5] = (-s1*c2*c3 + s3*c1)*s4 - s1*s2*c4
    Jr[2, 6] = ((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (s1*s3*c2 + c1*c3)*c5

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_ur10(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    p[1] = a1*s0*c1 - a2*s0*s1*s2 + a2*s0*c1*c2 - d3*c0 + d4*((-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3) + d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4)
    p[2] = a1*s1 + a2*s1*c2 + a2*s2*c1 + d0 + d4*(-(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3) - d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    R = np.zeros((3,3))
    R[0, 0] = (((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*c5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s5
    R[0, 1] = -(((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*s5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*c5
    R[0, 2] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    R[1, 0] = (((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*c5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s5
    R[1, 1] = -(((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*s5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*c5
    R[1, 2] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    R[2, 0] = ((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4*c5 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s5
    R[2, 1] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s5*c4 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*c5
    R[2, 2] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1*s0*c1 + a2*s0*s1*s2 - a2*s0*c1*c2 + d3*c0 + d4*((s0*s1*s2 - s0*c1*c2)*s3 - (s0*s1*c2 + s0*s2*c1)*c3) + d5*(-((s0*s1*s2 - s0*c1*c2)*c3 + (s0*s1*c2 + s0*s2*c1)*s3)*s4 + c0*c4)
    Jp[0, 1] = -a1*s1*c0 - a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 2] = -a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 3] = d4*((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*(-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 4] = d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 - s0*s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    Jp[1, 1] = -a1*s0*s1 - a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 2] = -a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 3] = d4*((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*(-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 4] = d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 + s4*c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1*c1 - a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 2] = -a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 3] = d4*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s4
    Jp[2, 4] = -d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4
    Jp[2, 5] = 0
    Jr = np.zeros((3,6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3
    Jr[0, 5] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3
    Jr[1, 5] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3
    Jr[2, 5] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_ur10e(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    p[1] = a1*s0*c1 - a2*s0*s1*s2 + a2*s0*c1*c2 - d3*c0 + d4*((-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3) + d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4)
    p[2] = a1*s1 + a2*s1*c2 + a2*s2*c1 + d0 + d4*(-(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3) - d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    R = np.zeros((3,3))
    R[0, 0] = (((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*c5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s5
    R[0, 1] = -(((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*s5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*c5
    R[0, 2] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    R[1, 0] = (((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*c5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s5
    R[1, 1] = -(((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*s5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*c5
    R[1, 2] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    R[2, 0] = ((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4*c5 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s5
    R[2, 1] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s5*c4 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*c5
    R[2, 2] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1*s0*c1 + a2*s0*s1*s2 - a2*s0*c1*c2 + d3*c0 + d4*((s0*s1*s2 - s0*c1*c2)*s3 - (s0*s1*c2 + s0*s2*c1)*c3) + d5*(-((s0*s1*s2 - s0*c1*c2)*c3 + (s0*s1*c2 + s0*s2*c1)*s3)*s4 + c0*c4)
    Jp[0, 1] = -a1*s1*c0 - a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 2] = -a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 3] = d4*((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*(-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 4] = d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 - s0*s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    Jp[1, 1] = -a1*s0*s1 - a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 2] = -a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 3] = d4*((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*(-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 4] = d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 + s4*c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1*c1 - a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 2] = -a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 3] = d4*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s4
    Jp[2, 4] = -d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4
    Jp[2, 5] = 0
    Jr = np.zeros((3,6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3
    Jr[0, 5] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3
    Jr[1, 5] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3
    Jr[2, 5] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_ur5(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    p[1] = a1*s0*c1 - a2*s0*s1*s2 + a2*s0*c1*c2 - d3*c0 + d4*((-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3) + d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4)
    p[2] = a1*s1 + a2*s1*c2 + a2*s2*c1 + d0 + d4*(-(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3) - d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    R = np.zeros((3,3))
    R[0, 0] = (((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*c5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s5
    R[0, 1] = -(((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*s5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*c5
    R[0, 2] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    R[1, 0] = (((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*c5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s5
    R[1, 1] = -(((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*s5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*c5
    R[1, 2] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    R[2, 0] = ((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4*c5 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s5
    R[2, 1] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s5*c4 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*c5
    R[2, 2] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1*s0*c1 + a2*s0*s1*s2 - a2*s0*c1*c2 + d3*c0 + d4*((s0*s1*s2 - s0*c1*c2)*s3 - (s0*s1*c2 + s0*s2*c1)*c3) + d5*(-((s0*s1*s2 - s0*c1*c2)*c3 + (s0*s1*c2 + s0*s2*c1)*s3)*s4 + c0*c4)
    Jp[0, 1] = -a1*s1*c0 - a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 2] = -a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 3] = d4*((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*(-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 4] = d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 - s0*s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    Jp[1, 1] = -a1*s0*s1 - a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 2] = -a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 3] = d4*((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*(-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 4] = d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 + s4*c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1*c1 - a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 2] = -a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 3] = d4*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s4
    Jp[2, 4] = -d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4
    Jp[2, 5] = 0
    Jr = np.zeros((3,6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3
    Jr[0, 5] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3
    Jr[1, 5] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3
    Jr[2, 5] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_ur5e(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    p[1] = a1*s0*c1 - a2*s0*s1*s2 + a2*s0*c1*c2 - d3*c0 + d4*((-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3) + d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4)
    p[2] = a1*s1 + a2*s1*c2 + a2*s2*c1 + d0 + d4*(-(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3) - d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    R = np.zeros((3,3))
    R[0, 0] = (((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*c5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s5
    R[0, 1] = -(((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 + s0*s4)*s5 + (-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*c5
    R[0, 2] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    R[1, 0] = (((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*c5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s5
    R[1, 1] = -(((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 - s4*c0)*s5 + (-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*c5
    R[1, 2] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    R[2, 0] = ((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4*c5 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s5
    R[2, 1] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s5*c4 + ((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*c5
    R[2, 2] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4
    Jp = np.zeros((3, 6))
    Jp[0, 0] = -a1*s0*c1 + a2*s0*s1*s2 - a2*s0*c1*c2 + d3*c0 + d4*((s0*s1*s2 - s0*c1*c2)*s3 - (s0*s1*c2 + s0*s2*c1)*c3) + d5*(-((s0*s1*s2 - s0*c1*c2)*c3 + (s0*s1*c2 + s0*s2*c1)*s3)*s4 + c0*c4)
    Jp[0, 1] = -a1*s1*c0 - a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 2] = -a2*s1*c0*c2 - a2*s2*c0*c1 + d4*(-(s1*s2*c0 - c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*((s1*s2*c0 - c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 3] = d4*((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3) - d5*(-(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*c0*c2 - s2*c0*c1)*c3)*s4
    Jp[0, 4] = d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*c4 - s0*s4)
    Jp[0, 5] = 0
    Jp[1, 0] = a1*c0*c1 - a2*s1*s2*c0 + a2*c0*c1*c2 + d3*s0 + d4*((-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3) + d5*(-((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4)
    Jp[1, 1] = -a1*s0*s1 - a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 2] = -a2*s0*s1*c2 - a2*s0*s2*c1 + d4*(-(s0*s1*s2 - s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*((s0*s1*s2 - s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 3] = d4*((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3) - d5*(-(-s0*s1*s2 + s0*c1*c2)*s3 + (-s0*s1*c2 - s0*s2*c1)*c3)*s4
    Jp[1, 4] = d5*(-((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*c4 + s4*c0)
    Jp[1, 5] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = a1*c1 - a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 2] = -a2*s1*s2 + a2*c1*c2 + d4*((-s1*s2 + c1*c2)*s3 - (-s1*c2 - s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 + (-s1*c2 - s2*c1)*s3)*s4
    Jp[2, 3] = d4*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3) - d5*((-s1*s2 + c1*c2)*c3 - (s1*c2 + s2*c1)*s3)*s4
    Jp[2, 4] = -d5*((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*c4
    Jp[2, 5] = 0
    Jr = np.zeros((3,6))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = s0
    Jr[0, 3] = s0
    Jr[0, 4] = (-s1*s2*c0 + c0*c1*c2)*s3 - (-s1*c0*c2 - s2*c0*c1)*c3
    Jr[0, 5] = -((-s1*s2*c0 + c0*c1*c2)*c3 + (-s1*c0*c2 - s2*c0*c1)*s3)*s4 + s0*c4
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -c0
    Jr[1, 3] = -c0
    Jr[1, 4] = (-s0*s1*s2 + s0*c1*c2)*s3 - (-s0*s1*c2 - s0*s2*c1)*c3
    Jr[1, 5] = -((-s0*s1*s2 + s0*c1*c2)*c3 + (-s0*s1*c2 - s0*s2*c1)*s3)*s4 - c0*c4
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = 0
    Jr[2, 3] = 0
    Jr[2, 4] = -(-s1*s2 + c1*c2)*c3 + (s1*c2 + s2*c1)*s3
    Jr[2, 5] = -((-s1*s2 + c1*c2)*s3 + (s1*c2 + s2*c1)*c3)*s4

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_iiwa(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = d2*s1*c0 + d4*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3) + d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5)
    p[1] = d2*s0*s1 + d4*(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3) + d6*((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5)
    p[2] = d0 + d2*c1 + d4*(s1*s3*c2 + c1*c3) + d6*(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (-s1*s3*c2 - c1*c3)*c5)
    R = np.zeros((3,3))
    R[0, 0] = ((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)*c6 + (-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*s6
    R[0, 1] = -((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)*s6 + (-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*c6
    R[0, 2] = (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5
    R[1, 0] = ((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)*c6 + (-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*s6
    R[1, 1] = -((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)*s6 + (-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*c6
    R[1, 2] = (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5
    R[2, 0] = (((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 - c1*c3)*s5)*c6 + (-(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s6
    R[2, 1] = -(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 - c1*c3)*s5)*s6 + (-(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c6
    R[2, 2] = ((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (-s1*s3*c2 - c1*c3)*c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = -d2*s0*s1 + d4*(-(-s0*c1*c2 - s2*c0)*s3 - s0*s1*c3) + d6*((((-s0*c1*c2 - s2*c0)*c3 - s0*s1*s3)*c4 + (s0*s2*c1 - c0*c2)*s4)*s5 - ((-s0*c1*c2 - s2*c0)*s3 + s0*s1*c3)*c5)
    Jp[0, 1] = d2*c0*c1 + d4*(s1*s3*c0*c2 + c0*c1*c3) + d6*(((-s1*c0*c2*c3 + s3*c0*c1)*c4 + s1*s2*s4*c0)*s5 - (-s1*s3*c0*c2 - c0*c1*c3)*c5)
    Jp[0, 2] = -d4*(-s0*c2 - s2*c0*c1)*s3 + d6*(((s0*s2 - c0*c1*c2)*s4 + (-s0*c2 - s2*c0*c1)*c3*c4)*s5 - (-s0*c2 - s2*c0*c1)*s3*c5)
    Jp[0, 3] = d4*(-(-s0*s2 + c0*c1*c2)*c3 - s1*s3*c0) + d6*((-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5*c4 - ((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c5)
    Jp[0, 4] = d6*(-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*s5
    Jp[0, 5] = d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)
    Jp[0, 6] = 0
    Jp[1, 0] = d2*s1*c0 + d4*(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3) + d6*((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5)
    Jp[1, 1] = d2*s0*c1 + d4*(s0*s1*s3*c2 + s0*c1*c3) + d6*(((-s0*s1*c2*c3 + s0*s3*c1)*c4 + s0*s1*s2*s4)*s5 - (-s0*s1*s3*c2 - s0*c1*c3)*c5)
    Jp[1, 2] = -d4*(-s0*s2*c1 + c0*c2)*s3 + d6*(((-s0*s2*c1 + c0*c2)*c3*c4 + (-s0*c1*c2 - s2*c0)*s4)*s5 - (-s0*s2*c1 + c0*c2)*s3*c5)
    Jp[1, 3] = d4*(-(s0*c1*c2 + s2*c0)*c3 - s0*s1*s3) + d6*((-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5*c4 - ((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c5)
    Jp[1, 4] = d6*(-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*s5
    Jp[1, 5] = d6*((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -d2*s1 + d4*(-s1*c3 + s3*c1*c2) + d6*(((-s1*s3 - c1*c2*c3)*c4 + s2*s4*c1)*s5 - (s1*c3 - s3*c1*c2)*c5)
    Jp[2, 2] = -d4*s1*s2*s3 + d6*((s1*s2*c3*c4 + s1*s4*c2)*s5 - s1*s2*s3*c5)
    Jp[2, 3] = d4*(s1*c2*c3 - s3*c1) + d6*((s1*s3*c2 + c1*c3)*s5*c4 - (-s1*c2*c3 + s3*c1)*c5)
    Jp[2, 4] = d6*(-(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s5
    Jp[2, 5] = d6*(((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 - c1*c3)*s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3,7))
    Jr[0, 0] = 0
    Jr[0, 1] = -s0
    Jr[0, 2] = s1*c0
    Jr[0, 3] = s0*c2 + s2*c0*c1
    Jr[0, 4] = -(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3
    Jr[0, 5] = -((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4
    Jr[0, 6] = (((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 - ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5
    Jr[1, 0] = 0
    Jr[1, 1] = c0
    Jr[1, 2] = s0*s1
    Jr[1, 3] = s0*s2*c1 - c0*c2
    Jr[1, 4] = -(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3
    Jr[1, 5] = -((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4
    Jr[1, 6] = (((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 - ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1*s2
    Jr[2, 4] = s1*s3*c2 + c1*c3
    Jr[2, 5] = -(-s1*c2*c3 + s3*c1)*s4 + s1*s2*c4
    Jr[2, 6] = ((-s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (-s1*s3*c2 - c1*c3)*c5

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

def kinmodel_lwr(q, tcp=None, out='x'):
    """
    Compute forward kinematics and Jacobian for the robot.
    Parameters:
    ----------
    q : array-like
        Joint angles/positions.
    tcp : array-like
        Tool centre point (optional).
    out : string
        Output form (optional).
    Returns:
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
    p[0] = -d2*s1*c0 + d4*((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3) + d6*(-(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5)
    p[1] = -d2*s0*s1 + d4*((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3) + d6*(-(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5)
    p[2] = d0 + d2*c1 + d4*(s1*s3*c2 + c1*c3) + d6*(-((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*s5 + (s1*s3*c2 + c1*c3)*c5)
    R = np.zeros((3,3))
    R[0, 0] = ((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)*c6 + (-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*s6
    R[0, 1] = -((((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)*s6 + (-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*c6
    R[0, 2] = -(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5
    R[1, 0] = ((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)*c6 + (-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*s6
    R[1, 1] = -((((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)*s6 + (-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*c6
    R[1, 2] = -(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5
    R[2, 0] = (((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*c5 + (s1*s3*c2 + c1*c3)*s5)*c6 + (-(s1*c2*c3 - s3*c1)*s4 - s1*s2*c4)*s6
    R[2, 1] = -(((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*c5 + (s1*s3*c2 + c1*c3)*s5)*s6 + (-(s1*c2*c3 - s3*c1)*s4 - s1*s2*c4)*c6
    R[2, 2] = -((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*s5 + (s1*s3*c2 + c1*c3)*c5
    Jp = np.zeros((3, 7))
    Jp[0, 0] = d2*s0*s1 + d4*((-s0*c1*c2 - s2*c0)*s3 + s0*s1*c3) + d6*(-(((-s0*c1*c2 - s2*c0)*c3 - s0*s1*s3)*c4 + (s0*s2*c1 - c0*c2)*s4)*s5 + ((-s0*c1*c2 - s2*c0)*s3 + s0*s1*c3)*c5)
    Jp[0, 1] = -d2*c0*c1 + d4*(-s1*s3*c0*c2 - c0*c1*c3) + d6*(-((-s1*c0*c2*c3 + s3*c0*c1)*c4 + s1*s2*s4*c0)*s5 + (-s1*s3*c0*c2 - c0*c1*c3)*c5)
    Jp[0, 2] = d4*(-s0*c2 - s2*c0*c1)*s3 + d6*(-((s0*s2 - c0*c1*c2)*s4 + (-s0*c2 - s2*c0*c1)*c3*c4)*s5 + (-s0*c2 - s2*c0*c1)*s3*c5)
    Jp[0, 3] = d4*((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0) + d6*(-(-(-s0*s2 + c0*c1*c2)*s3 + s1*c0*c3)*s5*c4 + ((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c5)
    Jp[0, 4] = -d6*(-((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 + (-s0*c2 - s2*c0*c1)*c4)*s5
    Jp[0, 5] = d6*(-(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*c5 - ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*s5)
    Jp[0, 6] = 0
    Jp[1, 0] = -d2*s1*c0 + d4*((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3) + d6*(-(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5)
    Jp[1, 1] = -d2*s0*c1 + d4*(-s0*s1*s3*c2 - s0*c1*c3) + d6*(-((-s0*s1*c2*c3 + s0*s3*c1)*c4 + s0*s1*s2*s4)*s5 + (-s0*s1*s3*c2 - s0*c1*c3)*c5)
    Jp[1, 2] = d4*(-s0*s2*c1 + c0*c2)*s3 + d6*(-((-s0*s2*c1 + c0*c2)*c3*c4 + (-s0*c1*c2 - s2*c0)*s4)*s5 + (-s0*s2*c1 + c0*c2)*s3*c5)
    Jp[1, 3] = d4*((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3) + d6*(-(-(s0*c1*c2 + s2*c0)*s3 + s0*s1*c3)*s5*c4 + ((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c5)
    Jp[1, 4] = -d6*(-((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 + (-s0*s2*c1 + c0*c2)*c4)*s5
    Jp[1, 5] = d6*(-(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*c5 - ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*s5)
    Jp[1, 6] = 0
    Jp[2, 0] = 0
    Jp[2, 1] = -d2*s1 + d4*(-s1*c3 + s3*c1*c2) + d6*(-((s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 + s3*c1*c2)*c5)
    Jp[2, 2] = -d4*s1*s2*s3 + d6*(-(-s1*s2*c3*c4 - s1*s4*c2)*s5 - s1*s2*s3*c5)
    Jp[2, 3] = d4*(s1*c2*c3 - s3*c1) + d6*(-(-s1*s3*c2 - c1*c3)*s5*c4 + (s1*c2*c3 - s3*c1)*c5)
    Jp[2, 4] = -d6*(-(s1*c2*c3 - s3*c1)*s4 - s1*s2*c4)*s5
    Jp[2, 5] = d6*(-((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*c5 - (s1*s3*c2 + c1*c3)*s5)
    Jp[2, 6] = 0
    Jr = np.zeros((3,7))
    Jr[0, 0] = 0
    Jr[0, 1] = s0
    Jr[0, 2] = -s1*c0
    Jr[0, 3] = -s0*c2 - s2*c0*c1
    Jr[0, 4] = (-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3
    Jr[0, 5] = ((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*s4 - (-s0*c2 - s2*c0*c1)*c4
    Jr[0, 6] = -(((-s0*s2 + c0*c1*c2)*c3 + s1*s3*c0)*c4 + (-s0*c2 - s2*c0*c1)*s4)*s5 + ((-s0*s2 + c0*c1*c2)*s3 - s1*c0*c3)*c5
    Jr[1, 0] = 0
    Jr[1, 1] = -c0
    Jr[1, 2] = -s0*s1
    Jr[1, 3] = -s0*s2*c1 + c0*c2
    Jr[1, 4] = (s0*c1*c2 + s2*c0)*s3 - s0*s1*c3
    Jr[1, 5] = ((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*s4 - (-s0*s2*c1 + c0*c2)*c4
    Jr[1, 6] = -(((s0*c1*c2 + s2*c0)*c3 + s0*s1*s3)*c4 + (-s0*s2*c1 + c0*c2)*s4)*s5 + ((s0*c1*c2 + s2*c0)*s3 - s0*s1*c3)*c5
    Jr[2, 0] = 1
    Jr[2, 1] = 0
    Jr[2, 2] = c1
    Jr[2, 3] = -s1*s2
    Jr[2, 4] = s1*s3*c2 + c1*c3
    Jr[2, 5] = (s1*c2*c3 - s3*c1)*s4 + s1*s2*c4
    Jr[2, 6] = -((s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*s5 + (s1*s3*c2 + c1*c3)*c5

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
            R_tcp = map_pose(Q=tcp[3:7], out='R')
        elif tcp.shape[0] == 6:
            p_tcp = tcp[:3]
            R_tcp = map_pose(TPY=tcp[3:6], out='R')
        else:
            raise ValueError('kinmodel: tcp is not SE3')
        v = R @ p_tcp
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
        p = p + R @ p_tcp
        Jp = Jp + s.T @ Jr
        R = R @ R_tcp

    J = np.vstack((Jp, Jr))

    if out=='pR':
        return p, R, J
    else:
        return map_pose(R=R, p=p, out=out), J

