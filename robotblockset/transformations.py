import numpy as np
from scipy.linalg import eigh
import quaternionic as Quaternion

from robotblockset.tools import _eps, rbs_type, check_shape, isscalar, isvector, vector, ismatrix, ismatrixarray, isskewsymmetric, matrix, isquaternion, normalize, vecnormalize, getunit


def map_pose(x=None, T=None, pa=None, Q=None, R=None, A=None, p=None, RPY=None, out="x", unit="rad"):
    """Convert pose form (SE3)

    A spatial pose can be represented as homogenous matrix or 7-dimensional
    array (translation and quaternion)

    Parameters
    ----------
    x : array of floats, optional
        Cartesian pose represented by translation and quaternion (7,)
    T : array of floats, optional
        Cartesian pose represented as homogenous matrix (4, 4)
    pa : array of floats, optional
        Cartesian pose represented by translation and axis/angle (6,)
    Q : array of floats, optional
        quaternion (4,)
    R : array of floats, optional
        rotation matrix (3, 3)
    A : array of floats, optional
        axis/angle vector (3,)
    p : array of floats, optional
        translation vector (3,)
    RPY : array of floats, optional
        rotation as Euler angles Roll, Pitch and Yaw (3,)
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pa``:position and axis/angle,
        ``pR``: rotation matrix and translation), ``Q`` quaternion,
        ``R`` rotation matrix, ``A`` axis/angle, or ``p`` position)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    ndarray
        pose, orientation or osition in seleced form

    Raises
    ------
    TypeError
        Not supported input or output form

    Note
    ----
    X takes precendece over T and both over others.
    Q takes precedence over R and RPY
    """
    if x is not None:
        x = rbs_type(x)
        if check_shape(x, shape=7):
            p = x[..., :3]
            Q = x[..., 3:]
        else:
            raise TypeError(f"Input form x: {x.shape} not supported")
    elif T is not None:
        T = rbs_type(T)
        if check_shape(T, shape=(4, 4)):
            p = T[..., :3, 3]
            Q = Quaternion.array.from_rotation_matrix(T[..., :3, :3]).ndarray
        else:
            raise TypeError(f"Input form T: {T.shape} not supported")
    elif pa is not None:
        pa = rbs_type(pa)
        if check_shape(pa, shape=6):
            p = pa[..., :3]
            Q = Quaternion.array.from_axis_angle(pa[..., 3:]).ndarray
        else:
            raise TypeError(f"Input form pa: {pa.shape} not supported")
    else:
        _n = 0
        if p is not None:
            p = rbs_type(p)
            if not check_shape(p, shape=3):
                raise TypeError(f"Input form p: {p.shape} not supported")
            if len(p.shape) == 1:
                _n = 1
            else:
                _n = p.shape[0]
        if Q is not None:
            Q = rbs_type(Q)
            if not check_shape(Q, shape=4):
                raise TypeError(f"Input form Q: {Q.shape} not supported")
        elif R is not None:
            R = rbs_type(R)
            if not check_shape(R, shape=(3, 3)):
                raise TypeError(f"Input form R: {R.shape} not supported")
            Q = Quaternion.array.from_rotation_matrix(R).ndarray
        elif A is not None:
            A = rbs_type(A)
            if check_shape(A, shape=3):
                Q = Quaternion.array.from_axis_angle(A).ndarray
            elif check_shape(A, shape=4):
                _tmp = normalize(A[:3]) * A[3]
                Q = Quaternion.array.from_axis_angle(_tmp).ndarray
            else:
                raise TypeError(f"Input form A: {A.shape} not supported")
        elif RPY is not None:
            RPY = rbs_type(RPY)
            if not check_shape(RPY, shape=3):
                raise TypeError(f"Input form RPY: {RPY.shape} not supported")
            Q = rpy2q(RPY, unit=unit)
        else:
            if _n == 1:
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
    if out == "pa":
        return np.hstack((p, Quaternion.array(np.array(Q)).to_axis_angle))
    elif out == "pR":
        return p, Quaternion.array(np.array(Q)).to_rotation_matrix
    elif (out == "Q") or (out == "Quaternion"):
        return Q
    elif (out == "R") or (out == "RotationMatrix"):
        return Quaternion.array(np.array(Q)).to_rotation_matrix
    elif (out == "A") or (out == "Axis/Angle"):
        return Quaternion.array(np.array(Q)).to_axis_angle
    elif (out == "p") or (out == "Position"):
        return p
    elif out == "2D":
        return np.hstack((p[:2], r2rpy(q2r(Q))[0]))
    elif out == "XY":
        return p[:2]
    elif out == "Angle":
        return r2rpy(q2r(Q))[0]
    else:
        raise ValueError(f"Output form {out} not supported")


def checkx(x):
    """Make quaternion scalar component positive

    Parameters
    ----------
    x : array of floats
        spatial pose to check

    Returns
    -------
    array of floats
        pose with positive quaternion scalar component
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        if x[3] < 0:
            x[3:] = -x[3:]
    else:
        Q = x[..., 3:]
        # Q[np.where(Q[..., 0] < 0)] = -Q[np.where(Q[..., 0] < 0)]
        for j in range(1, Q.shape[0]):
            C = np.dot(Q[j - 1, :], Q[j, :])
            if C < 0:
                Q[j, :] = -Q[j, :]
        x[..., 3:] = Q
    return x


def checkQ(Q):
    """Make quaternion scalar component positive

    Parameters
    ----------
    Q : array of floats
        quaternion to check (4,) or (..., 4)

    Returns
    -------
    array of floats
        quaternion with positive scalar component (4,) or (..., 4)
    """
    Q = rbs_type(Q)
    if isvector(Q, dim=4):
        if Q[0] < 0:
            Q = -Q
    else:
        # Q[np.where(Q[..., 0] < 0)] = -Q[np.where(Q[..., 0] < 0)]
        for j in range(1, Q.shape[0]):
            C = np.dot(Q[j - 1, :], Q[j, :])
            if C < 0:
                Q[j, :] = -Q[j, :]

    return Q


def q2Q(Q):
    """Quaternion array to quaternion object

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (...,4)

    Returns
    -------
    array of quaternions
        quaternion object
    """
    if check_shape(Q, shape=4):
        return Quaternion.array(Q)


def Q2q(Q):
    """Quaternion object to quaternion array

    Parameters
    ----------
    Q : quaternion array
        quaternion object

    Returns
    -------
    array of floats
        quaternion (4,) or (..., 4)
    Raises
    ------
    TypeError
        Input is not quaternion object
    """
    if isquaternion(Q):
        return Q.ndarray
    else:
        raise TypeError(f"Input is not quternion object")


def q2r(Q):
    """Quaternion to rotation matrix

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (...,4)

    Returns
    -------
    array of floats
        rotation matrix (3, 3) or (..., 3, 3)
    """
    return Quaternion.array(Q).to_rotation_matrix


def q2t(Q):
    """Quaternion to homogenous matrix

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (...,4)

    Returns
    -------
    array of floats
        rotation matrix (4, 4) or (..., 4, 4)
    """
    return map_pose(Q=Q, out="T")


def q2x(Q):
    """Quaternion to pose

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (...,4)

    Returns
    -------
    array of floats
        pose (7, ) or (..., 7)
    """
    return map_pose(Q=Q, out="x")


def r2q(R):
    """Rotation matrix to quaternion

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3) or (...,3, 3)

    Returns
    -------
    array of floats
        quaternion (4, ) or (..., 4)
    """
    Q = Quaternion.array.from_rotation_matrix(R).ndarray
    if isvector(Q, dim=4):
        if Q[0] < 0:
            Q = -Q
    else:
        Q[np.where(Q[..., 0] < 0)] = -Q[np.where(Q[..., 0] < 0)]
    return Q


def rp2t(R, p, out="T"):
    """Convert rotation and/or translation to homogenous matrix

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3)
    p : array of floats, optional
        translation vector (3,)
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))


    Returns
    -------
    ndarray
        homogenous matrix (4, 4)
    """
    return map_pose(R=R, p=p, out=out)


def p2t(p, out="T"):
    """Convert translation to homogenous matrix

    Parameters
    ----------
    p : array of floats, optional
        translation vector (3,)
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))


    Returns
    -------
    ndarray
        homogenous matrix (4, 4)
    """
    return map_pose(p=p, out=out)


def x2x(x):
    """Any pose to Cartesian pose

    Parameters
    ----------
    x : array of floats
        Pose (7,) or (4,4) or (3, 4)

    Returns
    -------
    array of floats
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
        raise TypeError(f"Pose shape {x.shape} not supported")


def x2t(x):
    """Cartesian pose to homogenous matrix

    Parameters
    ----------
    x : array of floats
        Cartesian pose (7,) or (...,7)

    Returns
    -------
    array of floats
        homogenous matrix (4, 4) or (..., 4, 4)
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        return map_pose(x=x, out="T")
    elif ismatrix(x, shape=7):
        return map_pose(x=x, out="T")
    else:
        raise TypeError(f"Expected parameter shape (...,7) but is {x.shape}")


def x2pa(x):
    """Cartesian pose to position + axis/angle

    Parameters
    ----------
    x : array of floats
        Cartesian pose (7,) or (...,7)

    Returns
    -------
    array of floats
        position+axis/angle (6,4) or (..., 6)
    """
    x = rbs_type(x)
    if isvector(x, dim=7):
        return map_pose(x=x, out="pa")
    elif ismatrix(x, shape=7):
        return map_pose(x=x, out="pa")
    else:
        raise TypeError(f"Expected parameter shape (...,7) but is {x.shape}")


def pa2x(pa):
    """Position + axis/angle to Cartesian pose

    Parameters
    ----------
    pa: array of floats
        position+axis/angle (6,4) or (..., 6)

    Returns
    -------
    array of floats
        Cartesian pose (7,) or (...,7)

    """
    pa = rbs_type(pa)
    if isvector(pa, dim=6):
        return map_pose(pa=pa, out="x")
    elif ismatrix(pa, shape=6):
        return map_pose(pa=pa, out="x")
    else:
        raise TypeError(f"Expected parameter shape (...,6) but is {pa.shape}")


def t2x(T):
    """Homogenous matrix to cartesian pose

    Parameters
    ----------
    T : array of floats
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array of floats
        Cartesian pose (7,) or (...,7)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        p = T[..., :3, 3]
        R = T[..., :3, :3]
        return np.hstack((p, r2q(R)))
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {T.shape}")


def t2q(T):
    """Homogenous matrix to quaternions

    Parameters
    ----------
    T : array of floats
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array of floats
        quaternions (...,4)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        R = T[..., :3, :3]
        return r2q(R)
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {x.shape}")


def t2p(T):
    """Extract position form homogenous matrix

    Parameters
    ----------
    T : array of floats
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array of floats
        Position (3,) or (...,3)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        return T[..., :3, 3]
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {x.shape}")


def t2r(T, unit="rad"):
    """Extract rotation matrix from homogenous matrix

    Parameters
    ----------
    T : array of floats
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array of floats
        rotation matrix (...,3, 3)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        return T[..., :3, :3]
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {x.shape}")


def t2prpy(T, unit="rad"):
    """Homogenous matrix to position and RPY Euler angles

    Parameters
    ----------
    T : array of floats
        Cartesian pose represented as homogenous matrix (..., 4, 4)

    Returns
    -------
    array of floats
        Position and RPY Euler angles (6,) or (...,6)
    """
    T = rbs_type(T)
    if check_shape(T, shape=(4, 4)):
        p = T[..., :3, 3]
        RPY = r2rpy(T[..., :3, :3], unit=unit)
        return np.hstack((p, RPY))
    else:
        raise TypeError(f"Expected parameter shape (...,4,4) but is {x.shape}")


def q2rpy(Q, unit="rad"):
    """Quaternion to RPY Euler angles

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
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


def r2rpy(R, unit="rad"):
    """Rotation matrix to RPY Euler angles

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    array of floats
        RPY Euler angles (3,) or (..., 3)
    """
    _Q = r2q(R)
    return q2rpy(_Q, unit=unit)


def rpy2q(rpy, out="Q", unit="rad"):
    """Euler angles RPY to quaternion or rotation matrix

    Parameters
    ----------
    rpy : float or array of floats
        Euler angles Roll or RPY
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Args
    ----
    p : floar or array of floats, optional
        Euler angle pitch
    y : floar or array of floats, optional
        Euler angle yaw

    Returns
    -------
    q : array of floats
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


def rpy2r(rpy, unit="rad"):
    """Euler angles RPY to rotation matrix

    Parameters
    ----------
    rpy : float or array of floats
        Euler angles Roll or RPY
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Args
    ----
    p : floar or array of floats, optional
        Euler angle pitch
    y : floar or array of floats, optional
        Euler angle yaw

    Returns
    -------
    array of floats
        rotation matrix

    Raises
    ------
    ValueError
        Not supported output form
    """
    return rpy2q(rpy, out="R", unit=unit)


def prpy2t(prpy, unit="rad"):
    """Pose defined by translation Euler angles RPY to homogenous matrix

    Parameters
    ----------
    prpy : float or array of floats
        Position and Euler angles Roll or RPY
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array of floats
        homogenous matrix (..., 4, 4)

    Raises
    ------
    ValueError
        Not supported output form
    """
    return map_pose(p=prpy[..., :3], RPY=prpy[..., 3:], out="T", unit=unit)


def prpy2x(prpy, unit="rad"):
    """Pose defined by translation Euler angles RPY to pose

    Parameters
    ----------
    prpy : float or array of floats
        Position and Euler angles Roll or RPY
    out: str, optional
        output form (``T``: Homogenous matrix, ``X``: pose array,
        ``pR``: rotation matrix and translation))
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    array of floats
        poses (..., 7)

    Raises
    ------
    ValueError
        Not supported output form
    """
    return map_pose(p=prpy[..., :3], RPY=prpy[..., 3:], out="x", unit=unit)


def t4rpy(rpy):
    """
    Matrix to convert Euler angles RPY velocities to rotation velocities
    for R = rot_z(rpy[0]) * rot_y(rpy[1]) * rot_x(rpy[2]).

    Parameters
    ----------
    rpy: array of floats
        RPY Euler angles (3, )

    Returns
    -------
    T: array of floats
        Transformation matrix (3, 3)
    """
    rpy = rbs_type(rpy)
    if isvector(rpy, dim=3) == 3:
        c2 = np.cos(rpy[1])
        s2 = np.sin(rpy[1])
        c3 = np.cos(rpy[0])
        s3 = np.sin(rpy[0])
        return np.array([[0, -s3, c2 * c3], [0, c3, c2 * s3], [1, 0, -s2]])


def t42point_sets(p1, p2):
    """
    Find a rigid transformation matrix between two poses of a rigid object defined by two sets of points.

    Parameters
    ----------
    p1: array of flaots
        Set of 3D points (n x 3)
    p2: array of floats
        Set of 3D points (n x 3)

    Returns
    -------
    array of floats
        Homogenous transformation matrix (4 x 4)
    """
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


def rot_x(phi, out="Q", unit="rad"):
    """Rotation matrix for rotation around x-axis

    Parameters
    ----------
    phi : int or float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    ndarray
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


def rot_y(phi, out="Q", unit="rad"):
    """Rotation matrix for rotation around y-axis

    Parameters
    ----------
    phi : int or float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    ndarray
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


def rot_z(phi, out="Q", unit="rad"):
    """Rotation matrix for rotation around z-axis

    Parameters
    ----------
    phi : int or float
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    ndarray
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


def rot_v(v, *phi, out="Q", unit="rad"):
    """Rotation matrix for rotation around v-axis

    if phi is not defined, rotation angle equals norm of ``v``

    Parameters
    ----------
    v : array-like
        3-dimensional rotation axis
    *phi : int or float, optional
        rotation angle
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)
    unit : str, optional
        angular unit (``rad`` or ``deg``)

    Returns
    -------
    ndarray
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


def vx2r(v, out="R"):
    """Rotation matrix to rotate x-axis to vector

    Parameters
    ----------
    v : array-like
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    ndarray
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


def vy2r(v, out="R"):
    """Rotation matrix to rotate y-axis to vector

    Parameters
    ----------
    v : array-like
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    ndarray
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
        if _v[0] < 0:
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


def vz2r(v, out="R"):
    """Rotation matrix to rotate z-axis to vector

    Parameters
    ----------
    v : array-like
        3-dimensional vector
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    ndarray
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


def vv2r(u, v, out="R"):
    """Rotation matrix to rotate vector u to vector v

    Parameters
    ----------
    u, v : array-like
        3-dimensional vectors
    out : str, optional
        output form (``R``: rotation matrix, ``Q``: quaternion)

    Returns
    -------
    ndarray
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
    if np.all(_k < _eps):
        if np.all(_u == _v):
            _R = np.eye(3)
        else:
            _w = _u[[0, 2, 1]]
            _t = np.cross(_u, _w)
            _t = _t / np.linalg.norm(_t)
            _R = rot_v(_t, np.pi, out="R")
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


def q2v(Q):
    """Axis/angle from quaternion

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        axis/angles representation of rotation (3,) or (..., 3)
    """
    _Q = q2Q(Q)
    return _Q.to_axis_angle


def r2v(R):
    """Axis/angle from rotation matrix

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    array of floats
        axis/angles representation of rotation (3,) or (..., 3)
    """
    _Q = q2Q(r2q(R))
    return _Q.to_axis_angle


def ang4v(v1, v2, *vn, unit="rad"):
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
    ndarray
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


def side4v(v1, v2, vn):
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


def v2s(v):
    """Map vector to matrix operator performing cross product

    Parameters
    ----------
    v : array-like
        3-dimensional vector

    Returns
    -------
    ndarray
        skew-symmertic matrix (3, 3)
    """
    v = vector(v, dim=3)
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S


def skew(v):
    """Map vector to matrix operator performing cross product

    Parameters
    ----------
    v : array-like
        3-dimensional vector

    Returns
    -------
    ndarray
        skew-symmertic matrix (3, 3)
    """
    return v2s(v)


def s2v(S):
    """Generate vector from skew-symmetric matrix

    Parameters
    ----------
    S : nparray
        (3, 3) skew-symmetric matrix

    Returns
    -------
    ndarray
        3-dimensional vector

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


def invskew(S):
    """Generate vector from skew-symmetric matrix

    Parameters
    ----------
    S : ndarray
        (3, 3) skew-symmetric matrix

    Returns
    -------
    ndarray
        3-dimensional vector
    """
    return s2v(S)


def qerr(Q2, *Q1):
    """Error of quaternions

    Angle between Q2 and Q1. If Q1 is ommited then Q2 is comapred
    to unit quaternion

    Parameters
    ----------
    Q2 : array of floats
        quaternion (4,) or (..., 4)
    Q1 : array of floats, optional
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        distance between quaternions (3,) or (...,3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q2 = uniqueQuaternionPath(Q2)
    if len(Q1) == 0:
        return 2 * qlog(Q2)[..., 1:]
    else:
        Q2 = np.array(Q2)
        Q1 = uniqueQuaternionPath(np.array(Q1[0]))
        if Q2.shape == Q1.shape and Q2.shape[-1] == 4:
            eq = 2 * np.log(Quaternion.array(Q2) * Quaternion.array(Q1).inverse).ndarray
            eq = np.where(eq > np.pi, np.mod(eq, np.pi), eq)
            return eq[..., 1:]
        else:
            raise TypeError("Parameters have to be (..., 4) array")


def qexp(Q):
    """Exp of quaternions

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        exp of quaternion (4,) or (...,4)

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


def qinv(Q):
    """Inverse of quaternion

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        Inverse of quaternion (4,) or (...,4)

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


def qlog(Q):
    """Log of quaternions

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        log of quaternion (4,) or (...,4)

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


def qmean(Q):
    """Mean of quaternions

    Parameters
    ----------
    Q : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        Mean of quaternion (4,)

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


def qnormalize(q):
    """
    Normalize array of quaternions

    Parameters
    ----------
    q : array of floats
        matrix (n x 4)

    Returns
    -------
    array of floats
        Normalized array of quaternions

    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    if check_shape(q, shape=4):
        return vecnormalize(q)
    else:
        raise TypeError("Input is not quaternion array")


def qmtimes(Q1, Q2):
    """Multiply quaternion

    Parameters
    ----------
    Q1 : array of floats
        quaternion (4,) or (..., 4)
    Q2 : array of floats
        quaternion (4,) or (..., 4)

    Returns
    -------
    array of floats
        quaternion product (..., 4)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    if Q1.shape[-1] == 4:
        if Q1.shape == Q2.shape:
            _qm = Quaternion.array(Q1) * Quaternion.array(Q2)
            return _qm.ndarray
        else:
            raise TypeError("Parameter has to be (..., 4) array")
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def qtranspose(Q):
    """Transpose of quaternions

    Parameters
    ----------
    Q : array of floats
        quaternions (4,) or (..., 4)

    Returns
    -------
    array of floats
        transpose of quaternions (4,) or (..., 4)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    Q = np.array(Q)
    if Q.shape[-1] == 4:
        return q2Q(Q).conj().ndarray
    else:
        raise TypeError("Parameter has to be (..., 4) array")


def rder(R, w):
    """Rotation matrix derivative

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3)
    w : array of floats
        rotation velocity (3, )

    Returns
    -------
    array of floats
        Rotation matrix derivative (3 x 3)
    Raises
    ------
    TypeError
        Parameter shape error
    """
    if R.shape == (3, 3):
        if isvector(w, dim=3):
            return v2s(w) @ R
        else:
            raise TypeError("Parameter w has o be array (3, ) ")
    else:
        raise TypeError("Parameter R to be array (3, 3) ")


def rerr(R2, *R1):
    """Error between to rotation matrices

    Angle between Q2 and Q1. If Q1 is ommited then Q2 is comapred
    to unit quaternion

    Parameters
    ----------
    R2 : array of floats
        rotation matrix (3, 3) or (..., 3, 3)
    R1 : array of floats, optional
        rotation matrix (3, 3) or (..., 3, 3)

    Returns
    -------
    array of floats
        distance between quaternions (3,) or (...,3)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    if ismatrixarray(R2, shape=(3, 3)):
        if len(R1) == 0:
            _err = qerr(r2q(R2))
        else:
            R1 = rbs_type(R1[0])
            if not R1.shape == R2.shape:
                raise TypeError(f"Input shapes R1: {R1.shape} and R2: {R2.shape} are not equal")
            _err = qerr(r2q(R2 @ R1.T))
        return _err
    else:
        raise TypeError(f"Input R2: {R2.shape} not supported")


def rexp(w):
    """Exp of rotation

    Parameters
    ----------
    w : array of floats
        rotation velocity (3, )

    Returns
    -------
    array of floats
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


def rlog(R):
    """Log of rotation matrix

    Parameters
    ----------
    R : array of floats
        rotation matrix (3, 3)

    Returns
    -------
    array of floats
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
                I = np.eye(3)
                w = np.pi * normalize(R[:, k] + I[:, k])
            else:
                theta = np.arccos((tr - 1) / 2)
                w = theta * s2v((R - R.T) / (2 * np.sin(theta)))
        else:
            raise ValueError("Input matrix is not a rotation matrix")
        return w
    else:
        raise ValueError("Input matrix is not a rotation matrix")


def wexp(w):
    """Exp of rotation velocity

    Parameters
    ----------
    w : array of floats
        rotation velocity (3, )

    Returns
    -------
    array of floats
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
            return np.eye(3) + 2 * np.cos(_fi / 2) * np.sin(_fi / 2) * _K + 2 * np.cos(_fi / 2) @ _K @ _K
    else:
        raise TypeError("Parameter w has o be array (3, )")


def xerr(x2, x1):
    """Cartesian pose error

    Distance and angle betwee x2 and x1

    Parameters
    ----------
    x2 : array of floats
        Cartesian pose (7,) or (..., 7)
    x1 : array of floats
        Cartesian pose (7,) or (..., 7)

    Returns
    -------
    array of floats
        distance between Cartesian poses (6,) or (...,6)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    x2 = rbs_type(x2)
    x1 = rbs_type(x1)
    if x2.shape == x1.shape and x2.shape[-1] == 7:
        ep = x2[..., :3] - x1[..., :3]
        Q2 = x2[..., 3:]
        Q1 = x1[..., 3:]
        eq = qerr(Q2, Q1)
        return np.hstack((ep, eq))
    else:
        raise TypeError("Parameters have to be (..., 7) array")


def xerrnorm(ex, scale=[1, 1]):
    """Cartesian pose error norm

    Parameters
    ----------
    ex : array of floats
        Cartesian pose error (6,) or (..., 6)
    scale: array of floats
        SE3 norm scale (2,)

    Returns
    -------
    array of floats
        Cartesian poses norm (1,) or (...,1)

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


def xmean(x):
    """Mean of pose (SE3)

    Parameters
    ----------
    x : array of floats
        poses (..., 7)

    Returns
    -------
    array of floats
        Mean of poses (7,)

    Raises
    ------
    TypeError
        Parameter shape error
    """
    x = np.array(x)
    if x.shape[-1] == 7:
        p = np.mean(x[:, :3])
        Q = qmean(x[:, 3:])
        return np.hstack((p, Q))
    else:
        raise TypeError("Parameter has to be (..., 7) array")


def xnormalize(x):
    """
    Normalize quaternion part of array of poses

    Parameters
    ----------
    x : array of floats
        matrix (n x 7)

    Returns
    -------
    array of floats
        Array of poses with normalized quaternions

    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    if check_shape(x, shape=7):
        if isvector(x):
            return np.hstack((x[:3], vecnormalize(x[3:])))
        if ismatrix(x):
            return np.hstack((x[:, :3], vecnormalize(x[:, 3:])))
        else:
            raise TypeError("Invalid input shape or type")
    else:
        raise TypeError("Input is not pose array")


def terr(T2, T1):
    """Homogenous matrix distance

    Distance between T2 and T1

    Parameters
    ----------
    T2 : array of floats
        Homogenous matrix (4, 4)
    T2 : array of floats
        Homogenous matrix (4, 4)

    Returns
    -------
    array of floats
        distance between Cartesian poses (6,) or (...,6)
    """
    T2 = matrix(T2, shape=(4, 4))
    T1 = matrix(T1, shape=(4, 4))
    Rerr = rerr(T2[:3, :3], T1[:3, :3])
    perr = T2[:3, 3] - T1[:3, 3]
    return np.concatenate((perr, Rerr))


def frame2world(x, T, typ=None):
    """Map variable from given frame to world frame

    Parameters
    ----------
    x : array of floats
        argument to map:
        - pose (n x 7) or (4 x 4) or (3 x 4)
        - position (n x 3)
        - orientation (n x 4) or (3 x 3)
        - velocity or force (n x 6)
    T : array of floats
        source frame in which variable is given:
        - translation and rotation (4 x 4) or (3 x 4) or (1 x 7)
        - translation (3 x 1)
        - rotation (3 x 3) or (1 x 4)
    typ: str, optional
        Transformation type (None or ``Wrench``)


    Returns
    -------
    array of floats
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
    elif T.shape == (1, 7):
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
        raise TypeError("Wrong input vector size")

    x = rbs_type(x)
    if x.shape == (4, 4):
        return rp2t(R0, p0) @ x
    if x.shape == (3, 4):
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
            if isvector(x, dim=4):
                RB = q2r(x)
                xx = r2q(R0 @ RB)
            if isvector(x, dim=3):
                pB = x.flatten()
                xx = R0 @ pB + p0
            if isvector(x, dim=6):
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                xx = RR @ x
            else:
                raise TypeError("Wrong input vector size")
        elif len(x.shape) == 2:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if check_shape(x, 7):
                    pB = x[i, :3]
                    RB = q2r(x[i, 3:7])
                    xx[i, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
                if check_shape(x, 4):
                    RB = q2r(x[i, :4])
                    xx[i, :] = r2q(R0 @ RB)
                if check_shape(x, 3):
                    pB = x[i, :3]
                    xx[i, :] = R0 @ pB + p0
                else:
                    raise TypeError("Wrong input vector size")
        elif len(x.shape) == 3:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if ismatrixarray(x, shape=4):
                    pB = x[i, :3, 3]
                    RB = x[i, :3, :3]
                    xx[i, :, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB, out="T")
                if ismatrixarray(x, shape=3):
                    xx[i, :, :] = R0 @ x[i, :3, :3]
                else:
                    raise TypeError("Wrong input vector size")
        else:
            raise TypeError("Wrong input vector size")
        return xx


def world2frame(x, T, typ=None):
    """Map variable from world frame to given frame

    Parameters
    ----------
    x : array of floats
        argument to map:
        - pose (n x 7) or (4 x 4) or (3 x 4)
        - position (n x 3)
        - orientation (n x 4) or (3 x 3)
        - velocity or force (n x 6)
    T : array of floats
        target frame to which variable is maped:
        - translation and rotation (4 x 4) or (3 x 4) or (1 x 7)
        - translation (3 x 1)
        - rotation (3 x 3) or (1 x 4)
    typ: str, optional
        Transformation type (None or ``Wrench``)


    Returns
    -------
    array of floats
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
    elif T.shape == (1, 7):
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
        raise TypeError("Wrong input vector size")

    R0 = R0.T
    p0 = -R0 @ p0

    x = rbs_type(x)
    if x.shape == (4, 4):
        return rp2t(R0, p0) @ x
    if x.shape == (3, 4):
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
            if isvector(x, dim=4):
                RB = q2r(x)
                xx = r2q(R0 @ RB)
            if isvector(x, dim=3):
                pB = x.flatten()
                xx = R0 @ pB + p0
            if isvector(x, dim=6):
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                xx = RR @ x
            else:
                raise TypeError("Wrong input vector size")
        elif len(x.shape) == 2:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if check_shape(x, 7):
                    pB = x[i, :3]
                    RB = q2r(x[i, 3:7])
                    xx[i, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB)
                if check_shape(x, 4):
                    RB = q2r(x[i, :4])
                    xx[i, :] = r2q(R0 @ RB)
                if check_shape(x, 3):
                    pB = x[i, :3]
                    xx[i, :] = R0 @ pB + p0
                else:
                    raise TypeError("Wrong input vector size")
        elif len(x.shape) == 3:
            n = x.shape[0]
            xx = np.copy(x)
            for i in range(0, n):
                if ismatrixarray(x, shape=4):
                    pB = x[i, :3, 3]
                    RB = x[i, :3, :3]
                    xx[i, :, :] = map_pose(p=R0 @ pB + p0, R=R0 @ RB, out="T")
                if ismatrixarray(x, shape=3):
                    xx[i, :, :] = R0 @ x[i, :3, :3]
                else:
                    raise TypeError("Wrong input vector size")
        else:
            raise TypeError("Wrong input vector size")
        return xx


def uniqueCartesianPath(x):
    """Make quaternion scalar component positive

    Parameters
    ----------
    x : array of floats
        spatial pose to check

    Returns
    -------
    array of floats
        pose with positive quaternion scalar component
    """
    return checkx(x)


def uniqueQuaternionPath(Q):
    """Make quaternion scalar component positive

    Parameters
    ----------
    Q : quaternion array
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
