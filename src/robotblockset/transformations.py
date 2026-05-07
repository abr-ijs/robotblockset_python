"""Transformation representation utilities.

Utilities for transformations between different representations.
of spatial and other variables

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy.linalg import eigh
import quaternionic as Quaternion

from robotblockset.tools import _eps, rbs_type, check_shape, isscalar, isvector, vector, ismatrix, ismatrixarray, isskewsymmetric, matrix, isquaternion, normalize, vecnormalize, getunit
from robotblockset.rbs_typing import ArrayLike, EulerAnglesType, HomogeneousMatrixType, HomogeneousMatricesType, Pose3DType, Poses3DType, QuaternionType, QuaternionObjectArrayType, QuaternionsType, RotationMatrixType, RotationMatricesType, TwistType, Vector3DType, Vectors3DType


def map_pose(x: ArrayLike = None, T: ArrayLike = None, pa: ArrayLike = None, pRPY: ArrayLike = None, Q: ArrayLike = None, R: ArrayLike = None, A: ArrayLike = None, p: ArrayLike = None, RPY: ArrayLike = None, p2d: ArrayLike = None, out: str = "x", unit: str = "rad") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Map and transform pose data between different representations.

    This function supports multiple input formats and outputs the pose in a specified form:

    - Input formats include position and orientation in quaternion, rotation matrix, axis-angle,
      roll-pitch-yaw (RPY), or transformation matrix.

    - Output formats can be the full pose (position + orientation), position only, quaternion, rotation matrix,
      axis-angle, RPY, transformation matrix, or others.

    Parameters
    ----------
    x : ArrayLike, optional
        A 7-element vector (position (3) + quaternion (4)) representing the pose.
        Either `x`, `T`, `pa`, `p2d`, or other parameters must be provided.

    T : ArrayLike, optional
        A 4x4 transformation matrix representing pose (rotation matrix (3x3) and translation vector).

    pa : ArrayLike, optional
        A 6-element vector with the first 3 elements representing position and the last 3 as axis-angle.

    pRPY : ArrayLike, optional
        A 6-element vector with the first 3 elements representing position and the last 3-elements roll-pitch-yaw (RPY) vector.

    Q : ArrayLike, optional
        A 4-element quaternion representing orientation.

    R : ArrayLike, optional
        A 3x3 rotation matrix representing orientation.

    A : ArrayLike, optional
        A 3-element axis-angle vector or a 4-element axis (first 3) and angle (4th element) representation.

    p : ArrayLike, optional
        A 3-element position vector.

    RPY : ArrayLike, optional
        A 3-element roll-pitch-yaw (RPY) vector.

    p2d : ArrayLike, optional
        A 3-element vector (x, y, theta) representing a 2D pose, where theta is an axis-angle representation.

    out : str, optional
        Specifies the output format (default 'x'):

        - 'x' or 'Pose': Full pose as position and quaternion.
        - 'T' or 'TransformationMatrix': 4x4 transformation matrix.
        - 'pa': Position and axis-angle.
        - 'pR': Position and rotation matrix (list).
        - 'pRPY': Position and roll-pitch-yaw.
        - 'Q' or 'Quaternion': Quaternion representation of orientation.
        - 'R' or 'RotationMatrix': Rotation matrix corresponding to the quaternion.
        - 'RPY': Roll-pitch-yaw representation corresponding to the quaternion.
        - 'A' or 'Axis/Angle': Axis-angle representation corresponding to the quaternion.
        - 'p' or 'Position': Position only.
        - '2d': 2D pose with position and orientation error.
        - 'XY': Only the first two components of position.
        - 'Angle' or 'theta': Orientation angle from the quaternion.

    unit : str, optional
        Unit of angle for RPY or axis-angle. Options: 'rad' (default) or 'deg'.

    Returns
    -------
    result : numpy.ndarray
        Pose in the requested output format. The output could be:

        - Position and quaternion as a combined vector.
        - A 4x4 transformation matrix.
        - A rotation matrix.
        - Axis-angle representation.
        - Roll-pitch-yaw representation.
        - Position vector.
        - Or any other specified format.

    Raises
    ------
    TypeError
        If the input does not match the expected shape or type.

    ValueError
        If an unsupported output format is requested.

    """
    if x is not None:
        x = rbs_type(x)
        if check_shape(x, shape=7):
            p = x[..., :3]
            Q = x[..., 3:]
        else:
            raise TypeError(f"Input form x: {x .shape} not supported")
    elif T is not None:
        T = rbs_type(T)
        if check_shape(T, shape=(4, 4)):
            p = T[..., :3, 3]
            Q = Quaternion.array.from_rotation_matrix(T[..., :3, :3]).ndarray
        else:
            raise TypeError(f"Input form T: {T .shape} not supported")
    elif pa is not None:
        pa = rbs_type(pa)
        if check_shape(pa, shape=6):
            p = pa[..., :3]
            Q = Quaternion.array.from_axis_angle(pa[..., 3:]).ndarray
        else:
            raise TypeError(f"Input form pa: {pa .shape} not supported")
    elif pRPY is not None:
        pRPY = rbs_type(pRPY)
        if check_shape(pRPY, shape=6):
            p = pRPY[..., :3]
            Q = rpy2q(pRPY[..., 3:], unit=unit)
        else:
            raise TypeError(f"Input form pa: {pa .shape} not supported")
    elif p2d is not None:
        p2d = rbs_type(p2d)
        if check_shape(p2d, shape=3):
            if isvector(p2d, dim=3):
                p = np.append(p2d[:2], 0)
                _axisangle = np.array([0, 0, p2d[2]])
            else:
                p = np.column_stack((p2d[..., :2], np.zeros(p2d.shape[0])))
                _axisangle = np.column_stack((np.zeros((p2d.shape[0], 2)), p2d[..., 2]))
            Q = Quaternion.array.from_axis_angle(_axisangle).ndarray
        else:
            raise TypeError(f"Input form p2d: {p2d .shape} not supported")
    else:
        _n = 0
        if p is not None:
            p = rbs_type(p)
            if not check_shape(p, shape=3):
                raise TypeError(f"Input form p: {p .shape} not supported")
            _n = 1 if len(p.shape) == 1 else p.shape[0]

        if Q is not None:
            Q = rbs_type(Q)
            if not check_shape(Q, shape=4):
                raise TypeError(f"Input form Q: {Q .shape} not supported")
        elif R is not None:
            R = rbs_type(R)
            if not check_shape(R, shape=(3, 3)):
                raise TypeError(f"Input form R: {R .shape} not supported")
            Q = Quaternion.array.from_rotation_matrix(R).ndarray
        elif A is not None:
            A = rbs_type(A)
            if check_shape(A, shape=3):
                Q = Quaternion.array.from_axis_angle(A).ndarray
            elif check_shape(A, shape=4):
                _tmp = normalize(A[:3]) * A[3]
                Q = Quaternion.array.from_axis_angle(_tmp).ndarray
            else:
                raise TypeError(f"Input form A: {A .shape} not supported")
        elif RPY is not None:
            RPY = rbs_type(RPY)
            if not check_shape(RPY, shape=3):
                raise TypeError(f"Input form RPY: {RPY .shape} not supported")
            Q = rpy2q(RPY, unit=unit)
        else:
            if _n <= 1:
                Q = np.array([1, 0, 0, 0])
            else:
                Q = np.repeat(np.array([[1, 0, 0, 0]]), _n, axis=0)
        if p is None:
            if len(Q.shape) == 1:
                p = np.zeros(3)
            else:
                p = np.zeros((Q.shape[0], 3))

    if (out == "x") or (out == "Pose"):
        return np.hstack((p, Q))
    elif (out == "T") or (out == "TransformationMatrix"):
        if len(Q.shape) == 1:
            T = np.eye(4)
            T[:3, :3] = Quaternion.array(np.array(Q)).to_rotation_matrix
            T[:3, 3] = p
        else:
            _R = Quaternion.array(np.array(Q)).to_rotation_matrix
            _px = np.swapaxes(np.expand_dims(p, 1), 1, 2)
            _Tx = np.concatenate((_R, _px), axis=2)
            _nx = np.expand_dims(np.repeat(np.array([[0, 0, 0, 1]]), Q.shape[0], axis=0), 1)
            T = np.concatenate((_Tx, _nx), axis=1)
        return T
    elif out == "pa":
        return np.hstack((p, Quaternion.array(np.array(Q)).to_axis_angle))
    elif out == "pR":
        return p, Quaternion.array(np.array(Q)).to_rotation_matrix
    elif out == "pRPY":
        return np.hstack((p, q2rpy(Q)))
    elif (out == "Q") or (out == "Quaternion"):
        return Q
    elif (out == "R") or (out == "RotationMatrix"):
        return Quaternion.array(np.array(Q)).to_rotation_matrix
    elif out == "RPY":
        return q2rpy(Q)
    elif (out == "A") or (out == "Axis/Angle"):
        return Quaternion.array(np.array(Q)).to_axis_angle
    elif (out == "p") or (out == "Position"):
        return p
    elif out == "2d":
        return xerr(np.hstack((p, Q)))[..., [0, 1, 5]]
    elif out == "XY":
        return p[..., :2]
    elif (out == "Angle") or (out == "theta"):
        return qerr(Q)[..., 2]
    else:
        raise ValueError(f"Output form {out} not supported")


def checkx(x: ArrayLike) -> Poses3DType:
    """
    Make quaternion scalar component positive.

    This function ensures that the scalar component of the quaternion in a spatial pose is positive.
    It checks the quaternion part of a pose, and if the scalar component is negative, it negates the entire quaternion.

    Parameters
    ----------
    x : ArrayLike
        Spatial pose to check. The input should be a 7-element vector (position and quaternion) or a higher-dimensional
        array with the last 4 elements representing the quaternion in the form [w, x, y, z].

    Returns
    -------
    Poses3DType
        Pose with the quaternion scalar component made positive. The quaternion part of the pose is adjusted
        if the scalar component is negative.

    Notes
    -----
    - If the input is a single 7-element pose vector, the function checks whether the scalar component of the quaternion
      (the first element of the quaternion) is negative. If so, it negates the quaternion.
    - If the input is an array of poses, the function iterates over each pose and ensures the quaternion scalar is positive
      by adjusting the corresponding quaternion.
    - The function assumes the input `x` has a shape that includes a quaternion as the last 4 elements in the pose,
      i.e., the quaternion is expected to be in the form [w, x, y, z].

    Raises
    ------
    ValueError
        If the input `x` does not have the expected shape or is not a valid spatial pose.
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        if x[3] < 0:
            x[3:] = -x[3:]
    elif ismatrix(x, shape=7):
        Q = x[..., 3:]
        for j in range(1, Q.shape[0]):
            C = np.dot(Q[j - 1, :], Q[j, :])
            if C < 0:
                Q[j, :] = -Q[j, :]
        x[..., 3:] = Q
    else:
        raise ValueError(f"Input shape {x .shape} not supported")
    return x


def checkQ(Q: ArrayLike) -> QuaternionsType:
    """
    Make quaternion scalar component positive.

    This function ensures that the scalar component of the quaternion is positive.
    If the scalar component of the quaternion is negative, the quaternion is negated to make the scalar positive.

    Parameters
    ----------
    Q : ArrayLike
        Quaternion(s) to check. The input can either be a single 4-element quaternion or an array of quaternions
        where the last dimension is of size 4 (representing [w, x, y, z] for the quaternion).

    Returns
    -------
    QuaternionsType
        Quaternion(s) with the scalar component made positive. If the scalar component of the quaternion was negative,
        the entire quaternion is negated.

    Raises
    ------
    ValueError
        If the input `Q` does not have a shape that matches a quaternion representation (4 elements) or an array of quaternions.

    Notes
    -----
    - The input `Q` is expected to be a 4-element vector or a higher-dimensional array where the last dimension represents quaternions.
    - If the scalar component (first element) of the quaternion is negative, the entire quaternion is negated.
    - The function uses dot products to check the sign of the scalar components for arrays of quaternions.

    """
    Q = rbs_type(Q)
    if isvector(Q, dim=4):
        if Q[0] < 0:
            Q = -Q
    elif ismatrix(Q, shape=4):
        for j in range(1, Q.shape[0]):
            C = np.dot(Q[j - 1, :], Q[j, :])
            if C < 0:
                Q[j, :] = -Q[j, :]
    else:
        raise ValueError(f"Input shape {Q .shape} not supported")
    return Q


def q2q(Q: ArrayLike) -> QuaternionsType:
    """
    Converts an input array of shape (..., 3) to (..., 4) by prepending a zero,
    or returns the array unchanged if already of shape (..., 4).

    Parameters
    ----------
    Q : ArrayLike
        Input array representing quaternions or vectors, expected to have shape (..., 3) or (..., 4).

    Returns
    -------
    np.ndarray
        An array of shape (..., 4), where a leading zero has been added if the input had shape (..., 3).

    Raises
    ------
    ValueError
        If the input does not have a last dimension of size 3 or 4.
    """
    Q = rbs_type(Q)
    if check_shape(Q, shape=4):
        return Q
    elif check_shape(Q, shape=3):
        zeros = np.zeros(Q.shape[:-1] + (1,))
        return np.concatenate((zeros, Q), axis=-1)
    else:
        raise ValueError(f"Wrong input size {Q .shape} - expected (..., 4) or (..., 3)")


def q_xyzw2wxyz(q: ArrayLike) -> Union[QuaternionsType, Poses3DType]:
    """
    Convert quaternion or pose from (x, y, z, w) to (w, x, y, z) format.

    Parameters
    ----------
    q : ArrayLike
        Input array with last dimension of size 4 (quaternion) or 7 (pose).
        If shape is (..., 4), interpreted as quaternion (x, y, z, w).
        If shape is (..., 7), interpreted as pose (x, y, z, qx, qy, qz, qw).

    Returns
    -------
    QuaternionsType or Poses3DType
        Converted array with same shape, where the quaternion or pose has
        the quaternion part in (w, x, y, z) order.

    Raises
    ------
    ValueError
        If the last dimension of the input is not 4 or 7.
    """
    q = rbs_type(q)
    if check_shape(q, 4):  # quaternion array
        return np.take(q, [3, 0, 1, 2], axis=-1)
    elif check_shape(q, 7):  # pose array
        return np.take(q, [0, 1, 2, 6, 3, 4, 5], axis=-1)
    else:
        raise ValueError("Last dimension of input array must be 4 or 7.")


def q_wxyz2xyzw(q: ArrayLike) -> Union[QuaternionsType, Poses3DType]:
    """
    Convert quaternion or pose from (w, x, y, z) to (x, y, z, w) format.

    Parameters
    ----------
    q : ArrayLike
        Input array with last dimension of size 4 (quaternion) or 7 (pose).
        If shape is (..., 4), interpreted as quaternion (w, x, y, z).
        If shape is (..., 7), interpreted as pose (x, y, z, qw, qx, qy, qz).

    Returns
    -------
    QuaternionsType or Poses3DType
        Converted array with same shape, where the quaternion or pose has
        the quaternion part in (x, y, z, w) order.

    Raises
    ------
    ValueError
        If the last dimension of the input is not 4 or 7.
    """
    q = rbs_type(q)
    if check_shape(q, 4):  # quaternion array
        return np.take(q, [1, 2, 3, 0], axis=-1)
    elif check_shape(q, 7):  # pose array
        return np.take(q, [0, 1, 2, 4, 5, 6, 3], axis=-1)
    else:
        raise ValueError("Last dimension of input array must be 4 or 7.")


def q2Q(Q: ArrayLike) -> QuaternionObjectArrayType:
    """
    Convert a quaternion array to a quaternion object.

    This function takes an array representing a quaternion (or an array of quaternions)
    and converts it into a quaternion object. A quaternion object allows for easier manipulation and conversion
    to other representations (e.g., rotation matrix, axis-angle).

    Parameters
    ----------
    Q : ArrayLike
        The input quaternion(s) to be converted. This can either be a single 4-element quaternion or an array
        of quaternions where the last dimension is of size 4.

    Returns
    -------
    Quaternion
        A quaternion object corresponding to the input quaternion array.

    Raises
    ------
    TypeError
        If the input does not match the expected shape of a quaternion array (4 elements or a higher-dimensional array
        with the last dimension of size 4).

    Notes
    -----
    - The input quaternion `Q` must have 4 elements, or be an array where the last dimension is of size 4.
    - The returned result is a quaternion object, which may offer additional methods for quaternion manipulation.

    """
    if check_shape(Q, shape=4):
        _Q = np.copy(Q)
        return Quaternion.array(_Q)
    else:
        raise TypeError("Input is not quaternion array")


def Q2q(Q: QuaternionObjectArrayType) -> QuaternionsType:
    """Quaternion object to quaternion array

    Parameters
    ----------
    Q : QuaternionObjectArrayType
        quaternion object

    Returns
    -------
    array-like
        quaternion (4,) or (..., 4)
    Raises
    ------
    TypeError
        Input is not quaternion object
    """
    if isquaternion(Q):
        return Q.ndarray
    else:
        raise TypeError("Input is not quternion object")


def q2r(Q: ArrayLike) -> RotationMatricesType:
    """Quaternion to rotation matrix

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (...,4)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or (..., 3, 3)
    """
    _Q = np.copy(Q)
    return Quaternion.array(_Q).to_rotation_matrix


def q2t(Q: ArrayLike) -> HomogeneousMatricesType:
    """Quaternion to homogenous matrix

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (...,4)

    Returns
    -------
    array-like
        rotation matrix (4, 4) or (..., 4, 4)
    """
    return map_pose(Q=Q, out="T")


def q2x(Q: ArrayLike) -> Poses3DType:
    """Quaternion to pose

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (...,4)

    Returns
    -------
    array-like
        pose (7, ) or (..., 7)
    """
    return map_pose(Q=Q, out="x")


def r2q(R: ArrayLike) -> QuaternionsType:
    """Rotation matrix to quaternion

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3) or (...,3, 3)

    Returns
    -------
    array-like
        quaternion (4, ) or (..., 4)
    """
    Q = Quaternion.array.from_rotation_matrix(R).ndarray
    if isvector(Q, dim=4):
        if Q[0] < 0:
            Q = -Q
    else:
        Q[np.where(Q[..., 0] < 0)] = -Q[np.where(Q[..., 0] < 0)]
    return Q


def rp2t(R: ArrayLike, p: ArrayLike, out: str = "T") -> Union[HomogeneousMatricesType, Poses3DType, Tuple[Vectors3DType, RotationMatricesType]]:
    """Convert rotation and/or translation to homogenous matrix

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3)
    p : ArrayLike
        translation vector (3,)
    out : str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))


    Returns
    -------
    HomogeneousMatricesType or Poses3DType or tuple[Vectors3DType, RotationMatricesType]
        Homogeneous matrix, pose array, or translation and rotation pair,
        depending on `out`.
    """
    if ismatrix(R, shape=(3, 3)):
        return map_pose(R=R, p=p, out=out)
    elif isvector(R, dim=4):
        return map_pose(Q=R, p=p, out=out)
    elif isvector(R, dim=3):
        return map_pose(RPY=R, p=p, out=out)
    else:
        raise ValueError(f"Wrong input size {R .shape} - expected (3, 3) or (4,)")


def p2t(p: ArrayLike, out: str = "T") -> Union[HomogeneousMatricesType, Poses3DType, Tuple[Vectors3DType, RotationMatricesType]]:
    """Convert translation to homogenous matrix

    Parameters
    ----------
    p : ArrayLike
        translation vector (3,)
    out : str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))


    Returns
    -------
    HomogeneousMatricesType or Poses3DType or tuple[Vectors3DType, RotationMatricesType]
        Homogeneous matrix, pose array, or translation and rotation pair,
        depending on `out`.
    """
    return map_pose(p=p, out=out)


def x2x(x: ArrayLike) -> Poses3DType:
    """Any pose to Cartesian pose

    Parameters
    ----------
    x : ArrayLike
        Pose (7,) or (4,4) or (3, 4)

    Returns
    -------
    array-like
        Cartesian pose (7,)
    """
    x = rbs_type(x)
    if x.shape == (4, 4):
        return map_pose(T=x)
    elif x.shape == (3, 4):
        return map_pose(T=np.vstack((x, np.array([0, 0, 0, 1]))))
    elif isvector(x, dim=6):
        return map_pose(pa=x)
    elif isvector(x, dim=7):
        return x
    else:
        raise TypeError(f"Pose shape {x .shape} not supported")


def x2t(x: ArrayLike) -> HomogeneousMatricesType:
    """Cartesian pose to homogenous matrix

    Parameters
    ----------
    x : ArrayLike
        Cartesian pose (7,) or (...,7)

    Returns
    -------
    array-like
        homogenous matrix (4, 4) or (..., 4, 4)
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        return map_pose(x=x, out="T")
    elif ismatrix(x, shape=7):
        return map_pose(x=x, out="T")
    else:
        raise TypeError(f"Expected parameter shape (...,7) but is {x .shape}")


def x2pa(x: ArrayLike) -> np.ndarray:
    """Cartesian pose to position + axis/angle

    Parameters
    ----------
    x : ArrayLike
        Cartesian pose (7,) or (...,7)

    Returns
    -------
    array-like
        position+axis/angle (6,) or (..., 6)
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        return map_pose(x=x, out="pa")
    elif ismatrix(x, shape=7):
        return map_pose(x=x, out="pa")
    else:
        raise TypeError(f"Expected parameter shape (...,7) but is {x .shape}")


def x2prpy(x: ArrayLike) -> np.ndarray:
    """Cartesian pose to position + RPY Euler angles

    Parameters
    ----------
    x : ArrayLike
        Cartesian pose (7,) or (...,7)

    Returns
    -------
    array-like
        position+RPY (6,) or (..., 6)
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        return map_pose(x=x, out="pRPY")
    elif ismatrix(x, shape=7):
        return map_pose(x=x, out="pRPY")
    else:
        raise TypeError(f"Expected parameter shape (...,7) but is {x .shape}")


def pa2x(pa: ArrayLike) -> Poses3DType:
    """Position + axis/angle to Cartesian pose

    Parameters
    ----------
    pa : ArrayLike
        position+axis/angle (6,4) or (..., 6)

    Returns
    -------
    array-like
        Cartesian pose (7,) or (...,7)

    """
    pa = rbs_type(pa)
    if isvector(pa, dim=6):
        return map_pose(pa=pa, out="x")
    elif ismatrix(pa, shape=6):
        return map_pose(pa=pa, out="x")
    else:
        raise TypeError(f"Expected parameter shape (...,6) but is {pa .shape}")


def t2x(T: ArrayLike) -> Poses3DType:
    """Homogenous matrix to cartesian pose

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array-like
        Cartesian pose (7,) or (...,7)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        p = T[..., :3, 3]
        R = T[..., :3, :3]
        return np.hstack((p, r2q(R)))
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def t2q(T: ArrayLike) -> QuaternionsType:
    """Homogenous matrix to quaternions

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array-like
        quaternions (...,4)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        R = T[..., :3, :3]
        return r2q(R)
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def t2p(T: ArrayLike) -> Vectors3DType:
    """Extract position form homogenous matrix

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    Vectors3DType
        Position vector `(3,)` or array of position vectors `(..., 3)`.
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        return T[..., :3, 3]
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def t2r(T: ArrayLike) -> RotationMatricesType:
    """Extract rotation matrix from homogenous matrix

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array-like
        rotation matrix (...,3, 3)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        return T[..., :3, :3]
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def t2rp(T: ArrayLike) -> Tuple[RotationMatricesType, Vectors3DType]:
    """Extract rotation matrix from homogenous matrix

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    tuple[RotationMatricesType, Vectors3DType]
        Rotation matrices and position vectors extracted from `T`.
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        return T[..., :3, :3], T[..., :3, 3]
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def t2prpy(T: ArrayLike, unit: str = "rad") -> np.ndarray:
    """Homogenous matrix to position and RPY Euler angles

    Parameters
    ----------
    T : ArrayLike
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array-like
        Position and RPY Euler angles (6,) or (...,6)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        p = T[..., :3, 3]
        RPY = r2rpy(T[..., :3, :3], unit=unit)
        return np.hstack((p, RPY))
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T .shape}")


def q2rpy(Q: ArrayLike, unit: str = "rad") -> np.ndarray:
    """Quaternion to RPY Euler angles

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    array-like
        RPY Euler angles (3,) or (..., 3)
    """
    _fac = getunit(unit=unit)
    Q = rbs_type(Q)
    if isvector(Q):
        Q = Q.reshape(1, 4)
    _qa = Q[:, 0]
    _qb = Q[:, 1]
    _qc = Q[:, 2]
    _qd = Q[:, 3]
    _theta1 = np.ones_like(_qa)
    _theta2 = 2 * _theta1

    _tmp = _qb * _qd * _theta2 - _qa * _qc * _theta2
    _tmp = np.clip(_tmp, -_theta1[0], _theta1[0])
    _b = -np.arcsin(_tmp)
    _a = np.arctan2(
        (_qa * _qd * _theta2 + _qb * _qc * _theta2),
        (_qa**2 * _theta2 - _theta1 + _qb**2 * _theta2),
    )
    _c = np.arctan2(
        (_qa * _qb * _theta2 + _qc * _qd * _theta2),
        (_qa**2 * _theta2 - _theta1 + _qd**2 * _theta2),
    )

    _rpy = np.column_stack((_a, _b, _c))
    return np.squeeze(_rpy) / _fac


def r2rpy(R: ArrayLike, unit: str = "rad") -> np.ndarray:
    """Rotation matrix to RPY Euler angles

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    array-like
        RPY Euler angles (3,) or (..., 3)
    """
    _Q = r2q(R)
    return q2rpy(_Q, unit=unit)


def rpy2q(rpy: ArrayLike, out: str = "Q", unit: str = "rad") -> Union[QuaternionsType, RotationMatricesType]:
    """Euler angles RPY to quaternion or rotation matrix

    Parameters
    ----------
    rpy : ArrayLike
        Euler angles Roll or RPY
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Args
    ----
    p : float or array-like, optional
        Euler angle pitch
    y : float or array-like, optional
        Euler angle yaw

    Returns
    -------
    q : array-like
        quaternion (..., 4) or rotation matrix (..., 4, 4)

    Raises
    ------
    TypeError
        Not supported input or output form
    """
    rpy = rbs_type(rpy)
    _fac = getunit(unit=unit)
    if isvector(rpy, dim=3):
        Q = np.zeros((1, 4))
    elif ismatrix(rpy, shape=3):
        Q = np.zeros((rpy.shape[0], 4))
    else:
        raise TypeError("Parameters has to be array (..., 3)")

    y = rpy[..., 0] * _fac
    p = rpy[..., 1] * _fac
    r = rpy[..., 2] * _fac

    Q[..., 0] = np.cos(r / 2) * np.cos(p / 2) * np.cos(y / 2) + np.sin(r / 2) * np.sin(p / 2) * np.sin(y / 2)
    Q[..., 1] = np.sin(r / 2) * np.cos(p / 2) * np.cos(y / 2) - np.cos(r / 2) * np.sin(p / 2) * np.sin(y / 2)
    Q[..., 2] = np.cos(r / 2) * np.sin(p / 2) * np.cos(y / 2) + np.sin(r / 2) * np.cos(p / 2) * np.sin(y / 2)
    Q[..., 3] = np.cos(r / 2) * np.cos(p / 2) * np.sin(y / 2) - np.sin(r / 2) * np.sin(p / 2) * np.cos(y / 2)

    Q = np.squeeze(Q)

    if out == "Q":
        return Q
    elif out == "R":
        return Quaternion.array(Q).to_rotation_matrix
    else:
        raise ValueError(f"Output form {out} not supported")


def rpy2r(rpy: ArrayLike, unit: str = "rad") -> RotationMatricesType:
    """Euler angles RPY to rotation matrix

    Parameters
    ----------
    rpy : ArrayLike
        Euler angles Roll or RPY
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Args
    ----
    p : float or array-like, optional
        Euler angle pitch
    y : float or array-like, optional
        Euler angle yaw

    Returns
    -------
    array-like
        rotation matrix

    Raises
    ------
    ValueError
        Not supported output form
    """
    return rpy2q(rpy, out="R", unit=unit)


def prpy2t(prpy: ArrayLike, unit: str = "rad") -> HomogeneousMatricesType:
    """
    Convert position and RPY angles to a homogeneous transformation matrix.

    Parameters
    ----------
    prpy : ArrayLike
        Position and Euler roll-pitch-yaw angles ``(..., 6)``.
    unit : str, optional
        Angular unit, either ``"rad"`` or ``"deg"``. Default is ``"rad"``.

    Returns
    -------
    HomogeneousMatricesType
        Homogeneous transformation matrix with shape ``(..., 4, 4)``.
    """
    prpy = rbs_type(prpy)
    return map_pose(p=prpy[..., :3], RPY=prpy[..., 3:], out="T", unit=unit)


def prpy2x(prpy: ArrayLike, unit: str = "rad") -> Poses3DType:
    """Pose defined by translation Euler angles RPY to pose

    Parameters
    ----------
    prpy : ArrayLike
        Position and Euler angles Roll or RPY
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array-like
        poses (..., 7)

    Raises
    ------
    ValueError
        Not supported output form
    """
    prpy = rbs_type(prpy)
    return map_pose(p=prpy[..., :3], RPY=prpy[..., 3:], out="x", unit=unit)


def spatial2x(x: ArrayLike, strict: bool = False) -> Poses3DType:
    """
    Convert a spatial representation SE3 into a pose (position + quaternion).

    This function accepts several common spatial representations of a pose or
    transform (such as rotation matrices, position vectors, quaternions, or
    pose vectors) and converts them into a pose vector `[x, y, z, qw, qx, qy, qz]`.

    Parameters
    ----------
    x : ArrayLike
        Input spatial representation. The following formats are supported:

        - (4, 4): Homogeneous transformation matrix (returned unchanged).
        - (3, 3): Rotation matrix `R`.
        - (7,): Pose vector `[x, y, z, qw, qx, qy, qz]`.
        - (3,): Position vector `p`.
        - (4,): Quaternion `Q = [qw, qx, qy, qz]`.
        - (6,): Pose as position + axis-angle vector `[x, y, z, rx, ry, rz]`.
    strict : bool, optional
        If True, only accept full SE3 representations (4x4 matrices or 7-element pose vectors).
        If False, accept SO3 or position vectors and fill in missing components with
        identity rotation or zero position respectively.

    Returns
    -------
    Poses3DType
        Pose vector `(7,)` or pose array `(..., 7)` representing the same
        spatial pose as the input.

    Raises
    ------
    ValueError
        If the input shape is not one of the supported formats.

    Notes
    -----
    - The function always outputs a valid SE(3) transformation matrix.
    - If input in not SE3 but only SO3 (e.g. rotation matrix or quaternion) or position
      vector, the output pose will have an identity rotation or zero position respectively.
    """
    _x = rbs_type(x)
    if _x.shape == (4, 4):
        _xx = map_pose(T=_x)
    elif _x.shape == (3, 3) and not strict:
        _xx = map_pose(R=_x)
    elif isvector(_x, dim=7):
        _xx = map_pose(x=_x)
    elif isvector(_x, dim=3) and not strict:
        _xx = map_pose(p=_x)
    elif isvector(_x, dim=4) and not strict:
        _xx = map_pose(Q=_x)
    elif isvector(_x, dim=6):
        _xx = map_pose(pa=_x)
    else:
        raise ValueError(f"Input argument shape {_x .shape} not supported")
    return _xx


def spatial2t(T: ArrayLike, strict: bool = False) -> HomogeneousMatricesType:
    """
    Convert a spatial representation SE3 into a 4x4 homogeneous transformation matrix.

    This function accepts several common spatial representations of a pose or
    transform (such as rotation matrices, position vectors, quaternions, or
    pose vectors) and converts them into a homogeneous transformation matrix `T`.

    Parameters
    ----------
    T : ArrayLike
        Input spatial representation. The following formats are supported:

        - (4, 4): Homogeneous transformation matrix (returned unchanged).
        - (3, 3): Rotation matrix `R`.
        - (7,): Pose vector `[x, y, z, qw, qx, qy, qz]`.
        - (3,): Position vector `p`.
        - (4,): Quaternion `Q = [qw, qx, qy, qz]`.
        - (6,): Pose as position + axis-angle vector `[x, y, z, rx, ry, rz]`.
    strict : bool, optional
        If True, only accept full SE3 representations (4x4 matrices or 7-element pose vectors).
        If False, accept SO3 or position vectors and fill in missing components with
        identity rotation or zero position respectively.

    Returns
    -------
    HomogeneousMatricesType
        Homogeneous transformation matrix `(4, 4)` or array of matrices
        `(..., 4, 4)` representing the same spatial pose as the input.

    Raises
    ------
    ValueError
        If the input shape is not one of the supported formats.

    Notes
    -----
    - The function always outputs a valid SE(3) transformation matrix.
    - If input in not SE3 but only SO3 (e.g. rotation matrix or quaternion) or position
      vector, the output pose will have an identity rotation or zero position respectively.

    """
    _x = rbs_type(T)
    if _x.shape == (4, 4):
        _xx = _x
    elif _x.shape == (3, 3) and not strict:
        _xx = map_pose(R=_x, out="T")
    elif isvector(_x, dim=7):
        _xx = map_pose(x=_x, out="T")
    elif isvector(_x, dim=3) and not strict:
        _xx = map_pose(p=_x, out="T")
    elif isvector(_x, dim=4) and not strict:
        _xx = map_pose(Q=_x, out="T")
    elif isvector(_x, dim=6):
        _xx = map_pose(pa=_x, out="T")
    else:
        raise ValueError(f"Input argument shape {_x .shape} not supported")
    return _xx


def t4rpy(rpy: EulerAnglesType) -> RotationMatrixType:
    """
    Matrix to convert Euler angles RPY velocities to rotation velocities
    for R = rot_z(rpy[0]) * rot_y(rpy[1]) * rot_x(rpy[2]).

    Parameters
    ----------
    rpy : EulerAnglesType
        RPY Euler angles `(3,)`.

    Returns
    -------
    RotationMatrixType
        Transformation matrix `(3, 3)`.
    """
    rpy = rbs_type(rpy)
    if isvector(rpy, dim=3) == 3:
        c2 = np.cos(rpy[1])
        s2 = np.sin(rpy[1])
        c3 = np.cos(rpy[0])
        s3 = np.sin(rpy[0])
        return np.array([[0, -s3, c2 * c3], [0, c3, c2 * s3], [1, 0, -s2]])


def t42point_sets(p1: ArrayLike, p2: ArrayLike) -> HomogeneousMatrixType:
    """
    Find a rigid transformation matrix between two poses of a rigid object defined by two sets of points.

    Parameters
    ----------
    p1 : ArrayLike
        First set of 3D points `(n, 3)`.
    p2 : ArrayLike
        Second set of 3D points `(n, 3)`.

    Returns
    -------
    HomogeneousMatrixType
        Homogeneous transformation matrix `(4, 4)`.
    """
    p1 = rbs_type(p1)
    p2 = rbs_type(p2)
    if p1.shape[1] != 3:
        raise ValueError("p1 must have 3 columns")
    if p2.shape[1] != 3:
        raise ValueError("p2 must have 3 columns")
    n = p1.shape[0]
    if p2.shape[0] != n:
        raise ValueError("p1 and p2 must have the same number of rows")

    c1 = np.mean(p1, axis=0)
    p1c = p1 - np.tile(c1, (n, 1))

    c2 = np.mean(p2, axis=0)
    p2c = p2 - np.tile(c2, (n, 1))

    X = np.dot(p1c.T, p2c)

    U, S, VT = np.linalg.svd(X)
    V = VT.T
    R = np.dot(V, U.T)
    if np.linalg.det(R) < 0:
        V[:, 2] = -V[:, 2]
        R = np.dot(V, U.T)

    d = c2 - np.dot(R, c1)

    T = rp2t(R, d)
    return T


def rot_x(phi: float, out: str = "Q", unit: str = "rad") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix for rotation around x-axis

    Parameters
    ----------
    phi : float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    TypeError
        Parameters is not scalar
    """
    if isscalar(phi):
        phi = phi * getunit(unit=unit)
        cx = np.cos(phi)
        sx = np.sin(phi)
        R = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        if out == "R":
            return R
        elif out == "Q":
            return Quaternion.array.from_rotation_matrix(R).ndarray
        else:
            raise ValueError(f"Output form {out} not supported")
    else:
        raise TypeError("Parameter has to be scalar")


def rot_y(phi: float, out: str = "Q", unit: str = "rad") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix for rotation around y-axis

    Parameters
    ----------
    phi : float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    TypeError
        Parameters is not scalar
    """
    if isscalar(phi):
        phi = np.array(phi) * getunit(unit=unit)
        cx = np.cos(phi)
        sx = np.sin(phi)
        R = np.array([[cx, 0, sx], [0, 1, 0], [-sx, 0, cx]])
        if out == "R":
            return R
        elif out == "Q":
            return Quaternion.array.from_rotation_matrix(R).ndarray
        else:
            raise ValueError(f"Output form {out} not supported")
    else:
        raise TypeError("Parameter has to be scalar")


def rot_z(phi: float, out: str = "Q", unit: str = "rad") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix for rotation around z-axis

    Parameters
    ----------
    phi : float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    TypeError
        Incorect parameter type
    """
    if isscalar(phi):
        phi = np.array(phi) * getunit(unit=unit)
        cx = np.cos(phi)
        sx = np.sin(phi)
        R = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
        if out == "R":
            return R
        elif out == "Q":
            return Quaternion.array.from_rotation_matrix(R).ndarray
        else:
            raise ValueError(f"Output form {out} not supported")
    else:
        raise TypeError("Parameter has to be scalar")


def rot_v(v: ArrayLike, *phi: float, out: str = "Q", unit: str = "rad") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix for rotation around v-axis

    if phi is not defined, rotation angle equals norm of ``v``

    Parameters
    ----------
    v : ArrayLike
        3-dimensional rotation axis
    *phi : int or float, optional
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    TypeError
        Incorect parameter type
    """
    v = vector(v, dim=3)
    if out == "R":
        if not phi:
            phi = np.linalg.norm(v)
            v = v / phi
            unit = "rad"
        else:
            phi = phi[0]
            v = v / np.linalg.norm(v)
        if isscalar(phi):
            phi = np.array(phi) * getunit(unit=unit)
            cx = np.cos(phi)
            sx = np.sin(phi)
            vx = 1 - cx
            R = np.array(
                [
                    [cx, -v[2] * sx, v[1] * sx],
                    [v[2] * sx, cx, -v[0] * sx],
                    [-v[1] * sx, v[0] * sx, cx],
                ]
            )
            vv = v.reshape(3, 1)
            R = (vv @ vv.T) * vx + R
            return R
        else:
            raise TypeError("Parameter has to be scalar")
    elif out == "Q":
        if not phi:
            return Quaternion.array.from_rotation_vector(v).ndarray
        else:
            v = v / np.linalg.norm(v) * phi[0]
            return Quaternion.array.from_rotation_vector(v).ndarray
    else:
        raise ValueError(f"Output form {out} not supported")


def vx2r(v: ArrayLike, out: str = "R") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix to rotate x-axis to vector

    Parameters
    ----------
    v : ArrayLike
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    """
    _v = vector(v, dim=3)
    _v = _v / np.linalg.norm(_v)
    _u = np.array([1, 0, 0])
    _k = np.cross(_u, _v)
    if np.all(np.abs(_k) < _eps):
        if _v[0] < 0:
            _R = np.diag([-1, -1, 1])
        else:
            _R = np.eye(3)
    else:
        _costheta = np.dot(_u, _v)
        _kk = _k.reshape(3, 1)
        _R = _costheta * np.eye(3) + v2s(_k) + (_kk @ _kk.T) * (1 - _costheta) / np.linalg.norm(_k) ** 2
    if out == "R":
        return _R
    elif out == "Q":
        return Quaternion.array.from_rotation_matrix(_R).ndarray
    else:
        raise ValueError(f"Output form {out} not supported")


def vy2r(v: ArrayLike, out: str = "R") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix to rotate y-axis to vector

    Parameters
    ----------
    v : ArrayLike
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    """
    _v = vector(v, dim=3)
    _v = _v / np.linalg.norm(_v)
    _u = np.array([0, 1, 0])
    _k = np.cross(_u, _v)
    if np.all(np.abs(_k) < _eps):
        if _v[1] < 0:
            _R = np.diag([1, -1, -1])
        else:
            _R = np.eye(3)
    else:
        _costheta = np.dot(_u, _v)
        _kk = _k.reshape(3, 1)
        _R = _costheta * np.eye(3) + v2s(_k) + (_kk @ _kk.T) * (1 - _costheta) / np.linalg.norm(_k) ** 2
    if out == "R":
        return _R
    elif out == "Q":
        return Quaternion.array.from_rotation_matrix(_R).ndarray
    else:
        raise ValueError(f"Output form {out} not supported")


def vz2r(v: ArrayLike, out: str = "R") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix to rotate z-axis to vector

    Parameters
    ----------
    v : ArrayLike
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    """
    _v = vector(v, dim=3)
    _v = _v / np.linalg.norm(_v)
    _u = np.array([0, 0, 1])
    _k = np.cross(_u, _v)
    if np.all(np.abs(_k) < _eps):
        if _v[2] < 0:
            _R = np.diag([-1, 1, -1])
        else:
            _R = np.eye(3)
    else:
        _costheta = np.dot(_u, _v)
        _kk = _k.reshape(3, 1)
        _R = _costheta * np.eye(3) + v2s(_k) + (_kk @ _kk.T) * (1 - _costheta) / np.linalg.norm(_k) ** 2
    if out == "R":
        return _R
    elif out == "Q":
        return Quaternion.array.from_rotation_matrix(_R).ndarray
    else:
        raise ValueError(f"Output form {out} not supported")


def vv2r(u: ArrayLike, v: ArrayLike, out: str = "R") -> Union[RotationMatricesType, QuaternionsType]:
    """Rotation matrix to rotate vector u to vector v

    Parameters
    ----------
    u, v : array-like
        3-dimensional vectors
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or quaternion (4,)

    Raises
    ------
    ValueError
        Not supported output form
    """
    _v = vector(v, dim=3)
    _v = _v / np.linalg.norm(_v)
    _u = vector(u, dim=3)
    _u = _u / np.linalg.norm(_u)
    _k = np.cross(_u, _v)
    if np.linalg.norm(_k) < _eps:
        if np.allclose(_u, _v, atol=_eps):
            _R = np.eye(3)
        else:
            axis = np.array([1.0, 0.0, 0.0])
            if np.isclose(abs(_u[0]), 1.0):
                axis = np.array([0.0, 1.0, 0.0])
            axis = np.cross(_u, axis)
            axis = axis / np.linalg.norm(axis)
            _R = -np.eye(3) + 2 * np.outer(axis, axis)

    else:
        _costheta = np.dot(_u, _v)
        _kk = _k.reshape(3, 1)
        _R = _costheta * np.eye(3) + v2s(_k) + (_kk @ _kk.T) * (1 - _costheta) / np.linalg.norm(_k) ** 2
    if out == "R":
        return _R
    elif out == "Q":
        return Quaternion.array.from_rotation_matrix(_R).ndarray
    else:
        raise ValueError(f"Output form {out} not supported")


def q2v(Q: ArrayLike) -> np.ndarray:
    """Axis/angle from quaternion

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    numpy.ndarray
        Axis-angle representation `(3,)` or `(..., 3)`.
    """
    _Q = q2Q(Q)
    return _Q.to_axis_angle


def r2v(R: ArrayLike) -> np.ndarray:
    """Axis/angle from rotation matrix

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    numpy.ndarray
        Axis-angle representation `(3,)` or `(..., 3)`.
    """
    _Q = q2Q(r2q(R))
    return _Q.to_axis_angle


def v2r(v: ArrayLike) -> RotationMatricesType:
    """Axis/angle to rotation matrix

    Parameters
    ----------
    v : ArrayLike
        axis/angles representation of rotation (3,) or (..., 3)

    Returns
    -------
    array-like
        rotation matrix (3, 3) or (..., 3, 3)
    """
    _v = rbs_type(v)
    if check_shape(_v, shape=3):
        Q = Quaternion.array.from_axis_angle(_v).ndarray
    elif check_shape(_v, shape=4):
        _tmp = normalize(_v[:3]) * _v[3]
        Q = Quaternion.array.from_axis_angle(_tmp).ndarray
    else:
        raise TypeError(f"Input form A: {_v .shape} not supported")
    return Quaternion.array(np.array(Q)).to_rotation_matrix


def ang4v(v1: ArrayLike, v2: ArrayLike, *vn: ArrayLike, unit: str = "rad") -> float:
    """Absolute angle between two vectors

    If vector ``vn`` is given, the angle is signed assuming ``vn`` is
    pointing in the same side as the normal.


    Parameters
    ----------
    v1, v2 : array-like
        3-dimensional vectors
    *vn : array-like, optional
        vector pointing in the direction of the normal
    unit : str, optional
        angular unit (``rad`` or ``deg``), by default ``rad``

    Returns
    -------
    array-like
        angle between vectors
    """
    v1 = vector(v1, dim=3)
    v2 = vector(v2, dim=3)
    a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if a > 1:
        a = 1
    phi = np.arccos(a)
    if len(vn) > 0:
        vn = vector(vn[0], dim=3)
        b = np.cross(v1, v2)
        if np.dot(np.array(vn), b) < 0:
            phi = -phi
    return phi / getunit(unit=unit)


def side4v(v1: ArrayLike, v2: ArrayLike, vn: ArrayLike) -> int:
    """Side of plane (v1,v2) vector vn is

    Parameters
    ----------
    v1, v2, vn : array-like
        3-dimensional vectors
    Returns
    -------
    int
        1: on same side as normal; -1: on opposite side; 0: on plane
    """
    v1 = vector(v1, dim=3)
    v2 = vector(v2, dim=3)
    vn = vector(vn, dim=3)
    b = np.cross(v1, v2)
    return np.sign(np.dot(vn, b))


def v2s(v: Vector3DType) -> RotationMatrixType:
    """Map vector to matrix operator performing cross product

    Parameters
    ----------
    v : Vector3DType
        Three-dimensional vector.

    Returns
    -------
    RotationMatrixType
        Skew-symmetric matrix `(3, 3)`.
    """
    v = vector(v, dim=3)
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S


def skew(v: Vector3DType) -> RotationMatrixType:
    """Map vector to matrix operator performing cross product

    Parameters
    ----------
    v : Vector3DType
        Three-dimensional vector.

    Returns
    -------
    RotationMatrixType
        Skew-symmetric matrix `(3, 3)`.
    """
    return v2s(v)


def s2v(S: RotationMatrixType) -> Vector3DType:
    """Generate vector from skew-symmetric matrix

    Parameters
    ----------
    S : RotationMatrixType
        Skew-symmetric matrix `(3, 3)`.

    Returns
    -------
    Vector3DType
        Three-dimensional vector.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    if ismatrix(S, shape=(3, 3)) and isskewsymmetric(S):
        v = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2
        return v
    else:
        raise TypeError("Parameter has to be (3, 3) array")


def invskew(S: RotationMatrixType) -> Vector3DType:
    """Generate vector from skew-symmetric matrix

    Parameters
    ----------
    S : RotationMatrixType
        Skew-symmetric matrix `(3, 3)`.

    Returns
    -------
    Vector3DType
        Three-dimensional vector.
    """
    return s2v(S)


def qerr(Q2: ArrayLike, *Q1: ArrayLike) -> Vectors3DType:
    """Error of quaternions

    Angle between Q2 and Q1. If Q1 is ommited then Q2 is comapred
    to unit quaternion

    Parameters
    ----------
    Q2 : ArrayLike
        quaternion (4,) or (..., 4)
    Q1 : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    Vectors3DType
        Quaternion error vector `(3,)` or array `(..., 3)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q2 = np.array(Q2)
    Q2 = uniqueQuaternionPath(Q2)
    if len(Q1) == 0:
        Q2 = uniqueQuaternionPath(Q2)
        return 2 * qlog(Q2)[..., 1:]
    else:
        Q1 = uniqueQuaternionPath(np.array(Q1[0]))
        if Q2.shape[-1] == 4:
            if Q2.shape == Q1.shape:
                pass
            elif isvector(Q1, dim=4):
                Q1 = np.tile(Q1, (Q2.shape[0], 1))
            elif isvector(Q2, dim=4):
                Q2 = np.tile(Q2, (Q1.shape[0], 1))
            else:
                raise ValueError("Parameters must have equal shape")

            eq = 2 * np.log(Quaternion.array(Q2) * Quaternion.array(Q1).inverse).ndarray
            eq = np.where(eq > np.pi, np.mod(eq, np.pi), eq)
            return eq[..., 1:]
        else:
            raise TypeError("Parameters have to be (..., 4) array")


def qexp(Q: ArrayLike) -> QuaternionsType:
    """Exp of quaternions

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    QuaternionsType
        Exponential of the quaternion `(4,)` or `(..., 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = rbs_type(Q)
    if Q.shape[-1] == 4:
        return np.exp(Quaternion.array(Q)).ndarray
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qinv(Q: ArrayLike) -> QuaternionsType:
    """Inverse of quaternion

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    QuaternionsType
        Inverse quaternion `(4,)` or `(..., 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = rbs_type(Q)
    if Q.shape[-1] == 4:
        return (Quaternion.array(Q).inverse).ndarray
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qlog(Q: ArrayLike) -> QuaternionsType:
    """Log of quaternions

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    QuaternionsType
        Quaternion logarithm `(4,)` or `(..., 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = np.array(Q)
    if Q.shape[-1] == 4:
        return np.log(Quaternion.array(Q)).ndarray
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qmean(Q: QuaternionsType) -> QuaternionType:
    """Mean of quaternions

    Parameters
    ----------
    Q : QuaternionsType
        Quaternion array `(4,)` or `(..., 4)`.

    Returns
    -------
    QuaternionType
        Mean quaternion `(4,)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    q = np.array(Q)
    if Q.shape[-1] == 4:
        A = np.zeros((4, 4))
        n = q.shape[0]
        for i in range(n):
            qq = q[i, :]
            if qq[0] < 0:
                qq = -qq
            A = np.outer(qq, qq) + A
        A = (1.0 / n) * A
        _, a = eigh(A, subset_by_index=(3, 3))  # Select the last eigenvalue/eigenvector
        qm = a.squeeze()
        return qm
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qnormalize(q: ArrayLike) -> QuaternionsType:
    """
    Normalize array of quaternions

    Parameters
    ----------
    q : ArrayLike
        matrix (n, 4)

    Returns
    -------
    QuaternionsType
        Normalized quaternion array.

    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    return q2Q(q).normalized.ndarray


def qmtimes(Q1: ArrayLike, Q2: ArrayLike) -> QuaternionsType:
    """Multiply quaternion

    Parameters
    ----------
    Q1 : ArrayLike
        quaternion (4,) or (..., 4)
    Q2 : ArrayLike
        quaternion (4,) or (..., 4)

    Returns
    -------
    QuaternionsType
        Quaternion product `(..., 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q1 = rbs_type(Q1)
    Q2 = rbs_type(Q2)
    if Q1.shape[-1] == 4:
        if Q1.shape == Q2.shape:
            _qm = Quaternion.array(Q1) * Quaternion.array(Q2)
            return _qm.ndarray
        else:
            raise TypeError("Parameter has to be (..., 4) array")
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qrotv(Q: ArrayLike, v: ArrayLike) -> Vectors3DType:
    """Rotate vectors v by given quaternions Q

    Parameters
    ----------
    Q : ArrayLike
        quaternion (4,) or (..., 4)
    v : ArrayLike
        vectors (3,) or (..., 3)

    Returns
    -------
    Vectors3DType
        Rotated vector `(3,)` or array of vectors `(..., 3)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = rbs_type(Q)
    if Q.shape[-1] == 4:
        if v.shape[-1] == 3:
            _vr = Quaternion.array(Q).rotate(v)
            return _vr
        else:
            raise TypeError("Parameter v has to be (..., 4) array")
    else:
        raise TypeError("Parameter Q has to be (..., 4) array")


def qtranspose(Q: ArrayLike) -> QuaternionsType:
    """Transpose of quaternions

    Parameters
    ----------
    Q : ArrayLike
        quaternions (4,) or (..., 4)

    Returns
    -------
    QuaternionsType
        Conjugated quaternion `(4,)` or `(..., 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = rbs_type(Q)
    if Q.shape[-1] == 4:
        return q2Q(Q).conj().ndarray
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def rder(R: ArrayLike, w: ArrayLike) -> RotationMatrixType:
    """Rotation matrix derivative

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3)
    w : ArrayLike
        rotation velocity (3, )

    Returns
    -------
    array-like
        Rotation matrix derivative (3, 3)
    Raises
    ------
    TypeError
        Parameter shape error
    """
    R = rbs_type(R)
    if R.shape == (3, 3):
        if isvector(w, dim=3):
            return v2s(w) @ R
        else:
            raise TypeError("Parameter w has o be array (3, ) ")
    else:
        raise TypeError("Parameter R to be array (3, 3) ")


def rerr(R2: ArrayLike, R1: Optional[ArrayLike] = None) -> Vectors3DType:
    """Error between to rotation matrices

    Angle between R2 and R1. If R1 is ommited then R2 is comapred
    to identity matrix

    Parameters
    ----------
    R2 : ArrayLike
        rotation matrix (3, 3) or (..., 3, 3)
    R1 : ArrayLike
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    array-like
        distance between quaternions (3,) or (...,3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    R2 = rbs_type(R2)
    if R1 is not None:
        R1 = rbs_type(R1)
    if ismatrixarray(R2, shape=(3, 3)):
        if len(R1) == 0:
            _err = qerr(r2q(R2))
        else:
            if not R1.shape == R2.shape:
                raise TypeError(f"Input shapes R1: {R1 .shape} and R2: {R2 .shape} are not equal")
            _err = qerr(r2q(R2 @ R1.T))
        return _err
    else:
        raise TypeError(f"Input R2: {R2 .shape} not supported")


def rmean(R2: ArrayLike, R1: Optional[ArrayLike] = None) -> RotationMatrixType:
    """Mean of two rotation matrices

    Angle between R2 and R1. If R1 is ommited then R2 is comapred
    to identity matrix

    Parameters
    ----------
    R2 : ArrayLike
        rotation matrix (3, 3)
    R1 : ArrayLike
        rotation matrix (3, 3)

    Returns
    -------
    array-like
        mean rotation matrix (3,3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    R2 = rbs_type(R2)
    if R1 is not None:
        R1 = rbs_type(R1)
    if ismatrixarray(R2, shape=(3, 3)):
        if len(R1) == 0:
            R1 = np.eye(3)
        else:
            R1 = rbs_type(R1[0])
        if not R1.shape == R2.shape:
            raise TypeError(f"Input shapes R1: {R1 .shape} and R2: {R2 .shape} are not equal")
        _v = r2v(R2 @ R1.T)
        return R1 @ v2r(_v / 2)
    else:
        raise TypeError(f"Input R2: {R2 .shape} not supported")


def rexp(w: ArrayLike) -> RotationMatrixType:
    """Exp of rotation

    Parameters
    ----------
    w : ArrayLike
        rotation velocity (3, )

    Returns
    -------
    array-like
        rotation matrix (3, 3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    w = rbs_type(w)
    if isvector(w, dim=3):
        _fi = np.linalg.norm(w)
        if _fi < _eps:
            return np.eye(3)
        else:
            _K = v2s(w) / _fi
            return np.eye(3) + np.sin(_fi) * _K + (1 - np.cos(_fi)) * _K @ _K
    else:
        raise TypeError("Parameter w has o be array (3, )")


def rlog(R: ArrayLike) -> Vector3DType:
    """Log of rotation matrix

    Parameters
    ----------
    R : ArrayLike
        rotation matrix (3, 3)

    Returns
    -------
    array-like
        rotation vector (3,)

    Raises
    ------
    ValueError
        Parameter is not rotation matrix
    TypeError
        Parameter is not rotation matrix
    """
    R = rbs_type(R)
    if R.shape == (3, 3):
        if np.isclose(np.linalg.det(R), 1.0, atol=1e-10):
            tr = np.trace(R)
            if np.isclose(tr, 3.0, atol=1e-10):
                w = np.array([0, 0, 0])
            elif np.isclose(tr, -1.0, atol=1e-10):
                k = np.argmax(np.diag(R))
                II = np.eye(3)
                w = np.pi * normalize(R[:, k] + II[:, k])
            else:
                theta = np.arccos((tr - 1) / 2)
                w = theta * s2v((R - R.T) / (2 * np.sin(theta)))
        else:
            raise ValueError("Input matrix is not a rotation matrix")
        return w
    else:
        raise ValueError("Input matrix is not a rotation matrix")


def wexp(w: ArrayLike) -> RotationMatrixType:
    """Exp of rotation velocity

    Parameters
    ----------
    w : ArrayLike
        rotation velocity (3, )

    Returns
    -------
    array-like
        rotation matrix (3, 3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    w = rbs_type(w)
    if isvector(w, dim=3):
        _fi = np.linalg.norm(w)
        if _fi < _eps:
            return np.eye(3)
        else:
            _K = v2s(w) / _fi
            return np.eye(3) + 2 * np.cos(_fi / 2) * np.sin(_fi / 2) * _K + 2 * np.cos(_fi / 2) * _K @ _K
    else:
        raise TypeError("Parameter w has o be array (3, )")


def xerr(x2: ArrayLike, x1: Optional[ArrayLike] = None, use_rot: bool = True) -> np.ndarray:
    """Cartesian pose error

    Distance and angle betwee x2 and x1

    Parameters
    ----------
    x2 : ArrayLike
        Cartesian pose (7,) or (..., 7)
    x1 : ArrayLike, optional
        Cartesian pose (7,) or (..., 7)
    use_rot : bool, optional
        Use rotation matrices for error calculation

    Returns
    -------
    numpy.ndarray
        Pose error vector `(6,)` or array `(..., 6)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    x2 = rbs_type(x2)
    if x1 is None:
        x1 = np.zeros(x2.shape) + np.array([0, 0, 0, 1, 0, 0, 0])
    else:
        x1 = rbs_type(x1)
    if x2.shape[-1] == 7:
        if x2.shape == x1.shape:
            pass
        elif isvector(x1, dim=7):
            x1 = np.tile(x1, (x2.shape[0], 1))
        elif isvector(x2, dim=7):
            x2 = np.tile(x2, (x1.shape[0], 1))
        else:
            raise ValueError("Parameters must have equal shape")

        ep = x2[..., :3] - x1[..., :3]
        Q2 = x2[..., 3:]
        Q1 = x1[..., 3:]
        if isvector(Q2) and use_rot:
            eq = rerr(q2r(Q2), q2r(Q1))
        else:
            # eq = qerr(Q2, Q1)
            # eq = np.vstack([qerr(QQ2, QQ1) for QQ2, QQ1 in zip(Q2, Q1)])
            eq = np.vstack([rerr(q2r(QQ2), q2r(QQ1)) for QQ2, QQ1 in zip(Q2, Q1)])
        return np.hstack((ep, eq))
    else:
        raise TypeError("Parameters have to be (..., 7) array")


def xerrnorm(ex: ArrayLike, scale: ArrayLike = [1, 1]) -> np.ndarray:
    """Cartesian pose error norm

    Parameters
    ----------
    ex : ArrayLike
        Cartesian pose error (6,) or (..., 6)
    scale : ArrayLike, optional
        SE3 norm scale (2,)

    Returns
    -------
    numpy.ndarray
        Cartesian pose norm scalar or array of norms.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    ex = rbs_type(ex)
    if isscalar(scale):
        scale = [1, scale]

    if isvector(ex, dim=6):
        return np.sqrt(scale[0] * np.linalg.norm(ex[:3]) ** 2 + scale[1] * np.linalg.norm(ex[3:]) ** 2)
    elif ismatrix(ex, shape=6):
        return (scale[0] * np.sum(np.abs(ex[..., :3]) ** 2, axis=-1) + scale[1] * np.sum(np.abs(ex[..., 3:]) ** 2, axis=-1)) ** (1.0 / 2)
    else:
        raise TypeError("Parameter has to be (..., 6) array")


def xmean(x: ArrayLike) -> Pose3DType:
    """Mean of pose (SE3)

    Parameters
    ----------
    x : ArrayLike
        poses (..., 7)

    Returns
    -------
    Pose3DType
        Mean pose `(7,)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    x = np.array(x)
    if x.shape[-1] == 7:
        p = np.mean(x[..., :3], axis=0)
        Q = qmean(x[..., 3:])
        return np.hstack((p, Q))
    else:
        raise TypeError("Parameter has to be (..., 7) array")


def xnormalize(x: ArrayLike) -> Poses3DType:
    """
    Normalize quaternion part of array of poses

    Parameters
    ----------
    x : ArrayLike
        matrix (n, 7)

    Returns
    -------
    Poses3DType
        Pose array with normalized quaternions.

    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    x = rbs_type(x)
    if check_shape(x, shape=7):
        return np.hstack((x[..., :3], vecnormalize(x[..., 3:])))
    else:
        raise TypeError("Input is not pose array")


def terr(T2: ArrayLike, T1: ArrayLike) -> TwistType:
    """Homogenous matrix distance

    Distance between T2 and T1

    Parameters
    ----------
    T2 : ArrayLike
        Homogenous matrix (4, 4)
    T2 : ArrayLike
        Homogenous matrix (4, 4)

    Returns
    -------
    TwistType
        Homogeneous-transform error vector `(6,)`.
    """
    T2 = matrix(T2, shape=(4, 4))
    T1 = matrix(T1, shape=(4, 4))
    Rerr = rerr(T2[:3, :3], T1[:3, :3])
    perr = T2[:3, 3] - T1[:3, 3]
    return np.concatenate((perr, Rerr))


def tmean(T2: ArrayLike, T1: ArrayLike) -> HomogeneousMatrixType:
    """Mean of two poses (SE3)

    Parameters
    ----------
    T2 : ArrayLike
        Homogenous matrix (4, 4)
    T2 : ArrayLike
        Homogenous matrix (4, 4)

    Returns
    -------
    HomogeneousMatrixType
        Mean homogeneous transform `(4, 4)`.

    Raises
    ------
    TypeError
        Parameter shape error
    """
    T2 = matrix(T2, shape=(4, 4))
    T1 = matrix(T1, shape=(4, 4))
    pmean = (T2[:3, 3] + T1[:3, 3]) / 2
    Rmean = rmean(T2[:3, :3], T1[:3, :3])
    return rp2t(Rmean, pmean)


def frame2world(x: ArrayLike, T: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
    """Map variable from given frame to world frame

    Parameters
    ----------
    x : ArrayLike
        argument to map:
        - pose (n, 7) or (n, 4, 4) or (n, 3, 4)
        - position (n, 3)
        - orientation (n, 4) or (3, 3)
        - velocity or force (n x 6)
    T : ArrayLike
        source frame in which variable is given:
        - translation and rotation (4, 4) or (3, 4) or (7, )
        - translation (3, )
        - rotation (3, 3) or (4, )
    typ : str, optional
        Transformation type (None, ``Twist`` or ``Wrench``)


    Returns
    -------
    array-like
        mapped argument

    Raises
    ------
    TypeError
        Wrong input size
    """
    T = rbs_type(T)
    if T.shape == (4, 4) or T.shape == (3, 4):
        p0 = T[:3, 3]
        R0 = T[:3, :3]
    elif isvector(T, dim=7):
        p0 = T[:3]
        R0 = q2r(T[3:7])
    elif T.shape == (3, 3):
        p0 = np.zeros(3)
        R0 = T
    elif isvector(T, dim=4):
        p0 = np.zeros(3)
        R0 = q2r(T)
    elif isvector(T, dim=3):
        p0 = T
        R0 = np.eye(3)
    else:
        raise TypeError(f"Wrong frame shape {T .shape}")

    x = rbs_type(x)
    if x.shape == (4, 4):
        return rp2t(R0, p0) @ x
    elif x.shape == (3, 4):
        tmp = rp2t(R0, p0) @ x
        return tmp[:3, :]
    elif x.shape == (3, 3):
        return R0 @ x
    else:
        if isvector(x):
            if isvector(x, dim=7):
                pB = x[:3]
                RB = q2r(x[3:7])
                xx = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
            elif isvector(x, dim=4):
                RB = q2r(x)
                xx = r2q(R0 @ RB)
            elif isvector(x, dim=3):
                pB = x.flatten()
                xx = R0 @ pB + p0
            elif isvector(x, dim=6):
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                elif typ == "Twist":  # twist (v)
                    RR[:3, 3:6] = v2s(p0) @ R0
                xx = RR @ x
            else:
                raise TypeError("Wrong input vector size")
        elif len(x.shape) == 2:
            n = x.shape[0]
            xx = np.copy(x)
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            elif typ == "Twist":  # twist (v)
                RR[:3, 3:6] = v2s(p0) @ R0
            for i in range(0, n):
                if check_shape(x, 7):
                    pB = x[i, :3]
                    RB = q2r(x[i, 3:7])
                    xx[i, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
                elif check_shape(x, 4):
                    RB = q2r(x[i, :4])
                    xx[i, :] = r2q(R0 @ RB)
                elif check_shape(x, 3):
                    pB = x[i, :3]
                    xx[i, :] = R0 @ pB + p0
                elif check_shape(x, 6):
                    xx[i, :] = RR @ x[i, :]
                else:
                    raise TypeError(f"Wrong input vector size {x .shape}")
        elif len(x.shape) == 3:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if ismatrixarray(x, shape=4):
                    pB = x[i, :3, 3]
                    RB = x[i, :3, :3]
                    xx[i, :, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB, out="T")
                elif ismatrixarray(x, shape=3):
                    xx[i, :, :] = R0 @ x[i, :3, :3]
                else:
                    raise TypeError(f"Wrong input vector shape {x .shape}")
        else:
            raise TypeError(f"Wrong input vector shape {x .shape}")
        return xx


def world2frame(x: ArrayLike, T: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
    """Map variable from world frame to given frame

    Parameters
    ----------
    x : ArrayLike
        argument to map:
        - pose (n, 7) or (n, 4, 4) or (n, 3, 4)
        - position (n, 3)
        - orientation (n, 4) or (3, 3)
        - velocity or force (n x 6)
    T : ArrayLike
        target frame to which variable is maped:
        - translation and rotation (4, 4) or (3, 4) or (7, )
        - translation (3, )
        - rotation (3, 3) or (4, )
    typ : str, optional
        Transformation type (None, ``Twist`` or ``Wrench``)


    Returns
    -------
    array-like
        mapped argument

    Raises
    ------
    TypeError
        Wrong input vector size
    """
    T = rbs_type(T)
    if T.shape == (4, 4) or T.shape == (3, 4):
        p0 = T[:3, 3]
        R0 = T[:3, :3]
    elif isvector(T, dim=7):
        p0 = T[:3]
        R0 = q2r(T[3:7])
    elif T.shape == (3, 3):
        p0 = np.zeros(3)
        R0 = T
    elif isvector(T, dim=4):
        p0 = np.zeros(3)
        R0 = q2r(T)
    elif isvector(T, dim=3):
        p0 = T
        R0 = np.eye(3)
    else:
        raise TypeError(f"Wrong frame shape {T .shape}")

    R0 = R0.T
    p0 = -R0 @ p0

    x = rbs_type(x)
    if x.shape == (4, 4):
        return rp2t(R0, p0) @ x
    elif x.shape == (3, 4):
        tmp = rp2t(R0, p0) @ x
        return tmp[:3, :]
    elif x.shape == (3, 3):
        return R0 @ x
    else:
        if isvector(x):
            if isvector(x, dim=7):
                pB = x[:3]
                RB = q2r(x[3:7])
                xx = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
            elif isvector(x, dim=4):
                RB = q2r(x)
                xx = r2q(R0 @ RB)
            elif isvector(x, dim=3):
                pB = x.flatten()
                xx = R0 @ pB + p0
            elif isvector(x, dim=6):
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                elif typ == "Twist":  # twist (v)
                    RR[:3, 3:6] = v2s(p0) @ R0
                xx = RR @ x
            else:
                raise TypeError(f"Wrong input vector size {x .shape}")
        elif len(x.shape) == 2:
            n = x.shape[0]
            xx = np.copy(x)
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            elif typ == "Twist":  # twist (v)
                RR[:3, 3:6] = v2s(p0) @ R0
            for i in range(0, n):
                if check_shape(x, 7):
                    pB = x[i, :3]
                    RB = q2r(x[i, 3:7])
                    xx[i, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
                elif check_shape(x, 4):
                    RB = q2r(x[i, :4])
                    xx[i, :] = r2q(R0 @ RB)
                elif check_shape(x, 3):
                    pB = x[i, :3]
                    xx[i, :] = R0 @ pB + p0
                elif check_shape(x, 6):
                    xx[i, :] = RR @ x[i, :]
                else:
                    raise TypeError(f"Wrong input vector shape {x .shape}")
        elif len(x.shape) == 3:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if ismatrixarray(x, shape=4):
                    pB = x[i, :3, 3]
                    RB = x[i, :3, :3]
                    xx[i, :, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB, out="T")
                elif ismatrixarray(x, shape=3):
                    xx[i, :, :] = R0 @ x[i, :3, :3]
                else:
                    raise TypeError(f"Wrong input vector shape {x .shape}")
        else:
            raise TypeError(f"Wrong input vector shape {x .shape}")
        return xx


def frame2world2d(x: ArrayLike, T: ArrayLike) -> np.ndarray:
    """Map variable from frame to world in 2D (x-y plane)

    Parameters
    ----------
    x : ArrayLike
        argument to map:
        - pose (x,y,theta) (n, 3) or (3, )
        - position (x,y) (n, 2) or (2, )
    T : ArrayLike
        target frame (x,y,theta) origin in world (3, )

    Returns
    -------
    array-like
        mapped argument

    Raises
    ------
    TypeError
        Wrong input vector size
    """
    T = rbs_type(T)
    if isvector(T, dim=3):
        p0 = T[:2]
        theta0 = T[2]
        R0 = rot_z(theta0, out="R")[:2, :2]
    else:
        raise TypeError("Wrong frame")

    x = rbs_type(x)
    if isvector(x):
        if isvector(x, dim=3):
            xx = np.hstack((R0 @ x[:2] + p0, x[2] + theta0))
        elif isvector(x, dim=2):
            xx = R0 @ x[:2] + p0
        else:
            raise TypeError("Wrong input vector size")
    elif len(x.shape) == 2:
        n = x.shape[0]
        xx = np.copy(x)
        for i in range(0, n):
            if check_shape(x, 3):
                xx[i, :] = np.hstack((R0 @ x[i, :2] + p0, x[i, 2] + theta0))
            elif check_shape(x, 2):
                xx[i, :] = R0 @ x[i, :2] + p0
            else:
                raise TypeError("Wrong input vector size")
    return xx


def world2frame2d(x: ArrayLike, T: ArrayLike) -> np.ndarray:
    """Map variable from world to given frame in 2D (x-y plane)

    Parameters
    ----------
    x : ArrayLike
        argument to map:
        - pose (x,y,theta) (n, 3) or (3, )
        - position (x,y) (n, 2) or (2, )
    T : ArrayLike
        target frame (x,y,theta) origin in world (3, )

    Returns
    -------
    array-like
        mapped argument

    Raises
    ------
    TypeError
        Wrong input vector size
    """
    T = rbs_type(T)
    if isvector(T, dim=3):
        p0 = T[:2]
        theta0 = T[2]
        R0 = rot_z(theta0, out="R")[:2, :2]
    else:
        raise TypeError("Wrong frame")

    R0 = R0.T
    p0 = -R0 @ p0
    theta0 = -theta0

    x = rbs_type(x)
    if isvector(x):
        if isvector(x, dim=3):
            xx = np.hstack((R0 @ x[:2] + p0, x[2] + theta0))
        elif isvector(x, dim=2):
            xx = R0 @ x[:2] + p0
        else:
            raise TypeError("Wrong input vector size")
    elif len(x.shape) == 2:
        n = x.shape[0]
        xx = np.copy(x)
        for i in range(0, n):
            if check_shape(x, 3):
                xx[i, :] = np.hstack((R0 @ x[i, :2] + p0, x[i, 2] + theta0))
            elif check_shape(x, 2):
                xx[i, :] = R0 @ x[i, :2] + p0
            else:
                raise TypeError("Wrong input vector size")
    return xx


def uniqueCartesianPath(x: ArrayLike) -> Poses3DType:
    """Make quaternion scalar component positive

    Parameters
    ----------
    x : ArrayLike
        spatial pose to check

    Returns
    -------
    array-like
        pose with positive quaternion scalar component
    """
    return checkx(x)


def uniqueQuaternionPath(Q: ArrayLike) -> QuaternionsType:
    """Make quaternion scalar component positive

    Parameters
    ----------
    Q : ArrayLike
        quaternion to check

    Returns
    -------
    quaternion array
        quaternion with positive scalar component
    """
    return checkQ(Q)


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    print("rot_x(45, out='R', unit='deg'):\n", rot_x(45, out="R", unit="deg"))
    print("rot_x(45, out='Q', unit='deg'):\n", rot_x(45, out="Q", unit="deg"))
    print(
        "RPY->Q  rpy2r(45, 0, 0, out='Q', unit='deg':\n",
        rpy2q((45, 0, 0), unit="deg"),
    )
    print(
        "RPY->R  rpy2r(45, 0, 0, out='R', unit='deg'):\n",
        rpy2r((45, 0, 0), unit="deg"),
    )
    print("Rot_v rot_v((1, 2, 3), 1.2):\n", rot_v((1, 2, 3), 1.2))
    print("vz2r((1, 0, 0)):\n", vz2r((1, 0, 0)))
    print(
        "ang4v((1, -2, 3), (1, 1, 1), [2, 2, 3], unit='deg'):",
        ang4v((1, -2, 3), (1, 1, 1), [2, 2, 3], unit="deg"),
    )
    print(
        "side4v((1, -2, 3), (1, 1, 1), [2, 2, -3]):",
        side4v((1, -2, 3), (1, 1, 1), [2, 2, -3]),
    )
    a = (1, 2, 3)
    S = skew(a)
    print("v:\n", a, "\nv2s(v)\n", S, "\ns2v(S)\n", s2v(S))

    Rx = rot_x(45, unit="deg", out="R")
    print("Rx: ", Rx)
    px = np.array([0, 1, 3])
    print("px: ", px)
    X0 = rp2t(Rx, px)
    print("T:\n", rp2t(Rx, px))
    print("x:\n", rp2t(Rx, px, out="x"))

    print("\n")

    p0 = np.array([0, 1, 3])
    p1 = np.array([1.0, 4.0, -1.0])
    p2 = np.array([-1.0, 1.0, 1.0])
    p3 = np.array([0.0, 3.0, 2.0])
    p = np.vstack((p0, p1, p2, p3))
    print("Positions p:\n", p)

    R = vv2r(p0, p1)
    print("R=vv2r(p0,p1)\n", R)
    v = r2v(R)
    print("v=r2v(R)\n", v)

    R0 = rot_x(0, unit="deg", out="R")
    R1 = rot_x(60, unit="deg", out="R")
    R2 = rot_y(30, unit="deg", out="R")
    R3 = rot_z(45, unit="deg", out="R")
    R = np.stack((R0, R1, R2, R3), axis=0)
    print("Rotations R:\n", R)
    rerr(R2, R3)

    Q0 = rot_x(0, unit="deg")
    Q1 = rot_x(60, unit="deg")
    Q2 = rot_y(30, unit="deg")
    Q3 = rot_z(45, unit="deg")
    Q = np.vstack((Q0, Q1, Q2, Q3))
    print("Quaternions Q:\n", Q)
    print("Mean quaternion:", qmean(Q))

    print("Euler RPY angles:\n", q2rpy(Q))

    x0 = rp2t(R0, p0, out="x")
    x1 = rp2t(R1, p1, out="x")
    x2 = rp2t(R2, p2, out="x")
    x3 = rp2t(R3, p3, out="x")
    x = np.vstack((x0, x1, x2, x3))
    print("Poses x:\n", x)

    T = x2t(x)
    print("Homogenous matrices T:\n", T)

    v = np.array([2, -1, 1, 3, 0, 2])
    print("Velocity v: \n", v)

    FT = np.array([1, -2, 1, 0, 2, 1])
    print("Wrench FT: \n", FT)

    Tx = rp2t(rot_z(2, unit="rad", out="R"), [2, -1, 3])
    print("Frame Tx: \n", Tx)

    print("Position p0 from frame to world: \n", frame2world(p0, Tx))
    print("Rotation R0 from frame to world: \n", frame2world(R0, Tx))
    print("Quaternion Q0 from frame to world: \n", frame2world(Q0, Tx))
    print("Pose x0 from frame to world: \n", frame2world(x0, Tx))
    print("Homogenous matrix T0 from frame to world: \n", frame2world(x2t(x0), Tx))
    print("Velocity vfrom frame to world: \n", frame2world(v, Tx))
    print("Wrench from frame to world: \n", frame2world(FT, Tx, typ="Wrench"))

    print("Position p from frame to world: \n", frame2world(p, Tx))
    print("Rotation R from frame to world: \n", frame2world(R, Tx))
    print("Quaternion Q from frame to world: \n", frame2world(Q, Tx))
    print("Pose x from frame to world: \n", frame2world(x, Tx))
    print("Homogenous matrix T from frame to world: \n", frame2world(T, Tx))

    print("Position p0 from world to frame: \n", world2frame(p0, Tx))
    print("Rotation R0 from world to frame: \n", world2frame(R0, Tx))
    print("Quaternion Q0 from world to frame: \n", world2frame(Q0, Tx))
    print("Pose x0 from world to frame: \n", world2frame(x0, Tx))
    print("Homogenous matrix T0 from world to frame: \n", world2frame(x2t(x0), Tx))
    print("Velocity vfrom world to frame: \n", world2frame(v, Tx))
    print("Wrench from world to frame: \n", world2frame(FT, Tx, typ="Wrench"))

    print("Position p from world to frame: \n", world2frame(p, Tx))
    print("Rotation R from world to frame: \n", world2frame(R, Tx))
    print("Quaternion Q from world to frame: \n", world2frame(Q, Tx))
    print("Pose x from world to frame: \n", world2frame(x, Tx))
    print("Homogenous matrix T from world to frame: \n", world2frame(T, Tx))

    Q5 = rot_y(0.27, unit="rad")
    QQ = np.vstack((Q0, Q1, Q2, Q3))
    ppa = np.vstack((p1, p0, p2, p3))
    QQa = np.vstack((Q5, Q1, -Q5, Q3))
    xxa = np.hstack((ppa, QQa))
    TTa = x2t(xxa)
    print("Err Q: ", qerr(QQa, QQ))
    print("Err x: ", xerr(xxa, x))
    print("Err T: ", xerr(xxa, x))
    # p, Q = x2t(xx)
    # R = np.array([Quaternion.array(x).to_rotation_matrix for x in Q])
    # print(R)

    print("Quaternions: \n", QQa)
    RRa = q2r(QQa)
    print("q2r:\n", RRa)
    print("r2q:\n", r2q(RRa))
    print("Check QQ:\n", checkQ(QQa))

    print("Poses xxa:\n", xxa)
    print("Check xxa:\n", checkx(xxa))

    k = np.repeat(np.array([[1, 2, 3, 4]]).T, 4, axis=1)
    q = np.multiply(Q, k)
    print("q: \n", q)
    print("Row norms of q: \n", np.linalg.norm(q, axis=1))
    print("Normalized q: \n", vecnormalize(q))
    print("Normalized x: \n", xnormalize(np.hstack((p, q))))

    o1p = rbs_type(
        [
            [-1.5562, 0.2572, 1.4492],
            [-1.6330, 0.3083, 1.3808],
            [-1.5965, 0.3571, 1.4649],
            [-1.6991, 0.2063, 1.4663],
            [-1.7070, 0.2998, 1.4768],
        ]
    )
    o2p = np.asarray(
        [
            [-0.2731, -0.4744, 1.4389],
            [-0.2718, -0.5672, 1.3712],
            [-0.3304, -0.5649, 1.4582],
            [-0.1481, -0.5588, 1.4520],
            [-0.2185, -0.6201, 1.4654],
        ]
    )

    print("Transformation between two point sets:\n", t42point_sets(o1p, o2p))
