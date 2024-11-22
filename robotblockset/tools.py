from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import matplotlib.pyplot as plt
import quaternionic as Quaternion

_eps = np.finfo(np.float64).eps
_scalartypes = (int, float, np.int64, np.int32, np.float64)


class rbs_object(metaclass=ABCMeta):
    def __init__(self):
        self._verbose = 1
        self.Name = ""

    def Message(self, msg, verb=0):
        if self._verbose > 0 and self._verbose >= verb:
            print(f"{self.Name}:{msg}")

    def WarningMessage(self, msg):
        print(f"Warning: {self.Name}:{msg}")

    def SetDebugLevel(self, level=1):
        self._verbose = level

    def GetDebugLevel(self):
        return self._verbose


class _struct:
    def asdict(self):
        return vars(self)

    def __iter__(self):
        for key, value in vars(self).items():
            yield key, value

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class _load(_struct):
    def __init__(self, **kwargs):
        kwargs.setdefault("mass", 0)
        kwargs.setdefault("COM", np.zeros(3))
        kwargs.setdefault("inertia", np.zeros((3, 3)))
        self.mass = kwargs["mass"]
        self.COM = kwargs["COM"]
        self.inertia = kwargs["inertia"]


def rbs_type(x):
    """Returns an ndarray of type necessary for RBS

    Parameters
    ----------
    x : any
        input variable

    Returns
    -------
    ndarray
        RBS nd array
    """
    return np.squeeze(np.copy(np.asarray(x, dtype="float")))


def isscalar(x) -> bool:
    """Check if parameter is scalar number

    Parameters
    ----------
    x : any
        value to check

    Returns
    -------
    bool
        True if x is int or real
    """
    return isinstance(x, _scalartypes) or (isinstance(x, np.ndarray) and (x.size == 1))


def isvector(x, dim: int = None) -> bool:
    """Check if parameter is a vector

    Parameters
    ----------
    x : any
        value to check
    dim : int, optional
        expected dimension

    Returns
    -------
    bool
        True if x is vector and optional if it has the required dimension
    """

    x = np.asarray(x)
    s = x.shape
    if dim is None:
        return len(s) == 1 and s[0] > 1
    else:
        return s == (dim,)


def vector(x, dim=None):
    """Return a vector

    Parameters
    ----------
    x : array-like
        values to be transformed to vector
    dim : int, optional
        required dimension; None: no length check

    Returns
    -------
    ndarray
        array in specified format

    Raises
    ------
    TypeError
        Parameter type error

    ValueError
        Vector length error
    """
    if isinstance(x, (list, tuple)):
        x = rbs_type(x).flatten()
    elif isscalar(x):
        x = rbs_type([x]).flatten()
    elif isinstance(x, np.ndarray):
        x = np.copy(x.flatten())
    else:
        raise TypeError("Invalid input type")
    if (dim is not None) and x.size != dim:
        raise ValueError(f"Incorrect vector length {x.size} - expected {dim}")
    return x


def ismatrix(x, shape=None) -> bool:
    """Check if parameter is a matrix

    Tests if the argument is a 2D matrix with a specified ``shape``.
    If ``shape`` is scalar, then only the  last dimension of the argument
    is checked.

    Parameters
    ----------
    x : ndarray
        value to check
    shape : tuple or scalar, optional
        required 2D shape

    Returns
    -------
    bool
        True if x has required dimensions
    """
    if isinstance(x, np.ndarray):
        if shape is None:
            return len(x.shape) == 2
        elif isscalar(shape) or len(shape) == 1:
            return x.shape[-1] == shape
        else:
            return x.shape == shape
    else:
        return False


def ismatrixarray(x, shape=None) -> bool:
    """Check if parameter is a matrix array

    Tests if the argument is a array of 2D matrices with a specified ``shape``.

    Parameters
    ----------
    x : ndarray
        value to check
    shape : tuple, optional
        required 2D shape of submatrix

    Returns
    -------
    bool
        True if x has required dimensions

    Raises
    ------
    ValueError
        Wrong shape value
    """
    if isinstance(x, np.ndarray):
        if shape is None:
            return len(x.shape) == 3
        elif isscalar(shape) or len(shape) == 1:
            return x.shape[-2:] == (shape, shape)
        elif len(shape) == 2:
            return x.shape[-2:] == shape
        else:
            raise ValueError(f"Incorrect shape value {x.shape} - expected {shape}")
    else:
        return False


def matrix(x, shape=None):
    """Return a matrix

    Parameters
    ----------
    x : array-like
        values to be transformed to matrix
    shape : tuple, optional
        required 2D shape

    Returns
    -------
    ndarray
        2D ndarray of specified shape

    Raises
    ------
    TypeError
        Argument type error

    ValueError
        Matrix shape error
    """
    if isscalar(x) or isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype="float")
    if not isinstance(x, np.ndarray):
        raise TypeError("Invalid argument type")
    if shape is None:
        if not x.ndim == 2:
            raise TypeError("Argument is not two-dimensional array")
        return x
    else:
        if x.shape == shape:
            return x
        elif np.prod(x.shape) == np.prod(shape):
            return x.reshape(shape)
        else:
            raise ValueError(f"Cannot reshape {x.shape} to {shape}")


def check_shape(x, shape) -> bool:
    """Check last dimensions of np array

    Parameters
    ----------
    x : array-like
        array to be checked
    shape : tuple
        required dimension

    Returns
    -------
    bool
        True if parameters is (..., shape)

    Raises
    ------
    TypeError
        Parameter type error
    """
    if shape == 1:
        return isscalar(x)
    elif isinstance(x, (list, tuple)):
        x = np.asarray(x)
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError("Invalid input type")
    if isscalar(shape):
        return x.shape[-1] == shape
    else:
        return x.shape[-len(shape) :] == shape


def isskewsymmetric(S, tol=100) -> bool:
    """Check if matrix is skew-symmetric

    Parameters
    ----------
    S : ndarray
        value to check

    Returns
    -------
    bool
        True if S is skew-symmetric
    """
    return isinstance(S, np.ndarray) and np.linalg.norm(S + S.T) < tol * _eps


def isquaternion(Q) -> bool:
    """Check if parameter is quaternionic array ``QArray``

    Parameters
    ----------
    Q : any
        input parameter

    Returns
    -------
    bool
        True if parameters is quaternionc array (``QArray``)
    """
    return Q.__class__.__name__ == "QArray"


def getunit(unit: str) -> float:
    """Calculates unit conversion factor

    Parameters
    ----------
    unit : str, optional
        angular unit, by default 'rad'

    Returns
    -------
    float
        unit -> rad conversion factor

    Raises
    ------
    ValueError
        Invalid unit
    """
    if unit.lower() == "rad":
        return 1
    elif unit.lower() == "deg":
        return np.pi / 180
    else:
        raise ValueError("Invalid units")


def check_option(opt: str, val: str) -> bool:
    """Check if option equals value (case-independent)

    Parameters
    ----------
    opt : str
        option to be checked
    val : str
        value for check

    Returns
    -------
    bool
        Check result

    Note
    ----
    For check the shortest string length is used
    """
    siz = min(len(opt), len(val))
    return opt[:siz].lower() == val[:siz].lower()


def find_rows(row: np.ndarray, x: np.array) -> list:
    idx = np.where((x == row).all(1))[0]
    return idx.tolist()


def grad(fun, x0: float, dx: float = 0.000001) -> float:
    """Gradient of function at values x

    Parameters
    ----------
    fun :
        function handle
    x0 : float or ndarray
        function argument values
    dx : float or ndarray, optional
        deviation to calculate gradient, optional

    Returns
    -------
    float or ndarray
        gradient of fun at x

    Raises
    ------
    ValueError
        Wrong arguments size
    """
    x0 = np.asarray(x0, dtype="float")
    dx = np.asarray(dx, dtype="float")
    n = x0.size
    if n == 1:
        if not x0.size == dx.size:
            raise ValueError("Parameters have to be same size")
        return (fun(x0 + dx) - fun(x0 - dx)) / (2 * dx)
    else:
        if isscalar(dx):
            dx = np.ones(x0.shape) * dx
        elif not x0.shape == dx.shape:
            raise ValueError("Parameters have to be same size")
        g = np.empty(n)
        u = np.copy(x0)
        for i in range(n):
            u[i] = x0[i] + dx[i]
            f1 = fun(u)
            u[i] = x0[i] - dx[i]
            f2 = fun(u)
            g[i] = (f1 - f2) / (2 * dx[i])
        return g


def hessmat(fun, x0, delta=None):
    """
    Hessian matrix of a scalar function with vector argument.

    The Hessian matrix of f(x) is the square matrix of the second partial
    derivatives of f(x).

    Parameters
    ----------
    fun: Scalar function.
    x0 : float or ndarray
        argument (n x 1)
    dx : float or ndarray, optional
        deviation to calculate hessian, optional

    Returns
    -------
    float or ndarray
        Hessian of fun at x0 (n x n).
    """
    if delta is None:
        delta = max(np.linalg.norm(x0) / 1000, 1e-5)
    if isinstance(delta, (int, float)):
        delta = delta * np.ones(x0.shape)

    n = len(x0)
    h = np.empty((n, n))
    g1 = grad(fun, x0, delta)

    for i in range(n):
        for j in range(i, n):
            u = x0.copy()
            u[j] = x0[j] + delta[j]
            g2 = grad(fun, u, delta)
            h[i, j] = (g2[i] - g1[i]) / (1 * delta[j])
            if j > i:
                h[j, i] = h[i, j]

    return h


def deadzone(x, width=1, center=0):
    x = np.asarray(x, dtype="float")
    xx = np.copy(x)
    _lower_limit = center - width
    _upper_limit = center + width
    xx[(x >= _lower_limit) & (x <= _upper_limit)] = 0
    xx[x < _lower_limit] -= _lower_limit
    xx[x > _upper_limit] -= _upper_limit
    return xx


def sigmoid(x, offset=0.0, gain=1.0):
    """Sigmoid function

    Parameters
    ----------
    x : array of floats
        input values
    offset : float, optional
        function offset, by default 0
    gain : float, optional
        function gain, by default 1

    Returns
    -------
    array of floats
        values of sigmoid function
    """
    x = np.asarray(x, dtype="float")
    return 1 / (1 + np.exp(-gain * (x - offset)))


def fit3dcirc(X, pl=False):
    """
    Fit a circle to a set of 3D points.

    Parameters
    ----------
    X : array of floas
        Set of points (n x 3)
    pl : bool
        Flag for plot (optional, default False).

    Returns
    -------
    pc : array of floats
        Circle center point (3 x 1)
    n : array of floats
        Normal to circle plane (3 x 1)
    r : float
        Circle radius.
    R : array of floats
        Circle frame rotation (3 x 3).
    """
    Xm = np.mean(X, axis=0)
    dX = X - Xm
    U, S, V = np.linalg.svd(dX, full_matrices=False)
    Q = V[:, :2]  # basis of the plane
    dX = dX @ Q
    xc = dX[:, 0]
    yc = dX[:, 1]
    A = np.column_stack((xc**2 + yc**2, -2 * xc, -2 * yc))

    if np.linalg.matrix_rank(A) < 3:
        pc = Xm
        n = np.array([0, 0, 0])
        r = 0
        R = np.eye(3)
        h = []
        return pc, n, r, R, h

    P = np.linalg.lstsq(A, np.ones(xc.shape), rcond=None)[0]
    a = P[0]
    P /= a
    r = np.sqrt(P[1] ** 2 + P[2] ** 2 + 1 / a)
    pc = Xm + Q @ P[1:3]
    n = np.cross(Q[:, 0], Q[:, 1])
    R = np.column_stack((Q, n))

    h = []
    if pl:
        theta = np.linspace(0, 2 * np.pi, num=100)
        pc = np.expand_dims(pc, 1)
        pc = np.repeat(pc, 100, axis=1)
        c = pc + r * Q @ np.array([np.cos(theta), np.sin(theta)])
        plt.axes(projection="3d")
        plt.plot(X[:, 0], X[:, 1], X[:, 2], ".", label="Points")
        plt.plot(c[0, :], c[1, :], c[2, :], color=[0.6, 0.6, 0.6])
        plt.plot(pc[0], pc[1], pc[2], "k.", markersize=5)
        plt.title("Fit circle to points")

    return pc, n, r, R


def fitplane(points):
    """
    Fit a plane to a set of 3D points.

    Parameters
    ----------
    points : array of floats
        3D points (n x 3).

    Returns
    -------
    n : array of floats
        Normal to the plane (3 x 1)
    R : array of floats
        Orthonormal basis of the plane (3 x 3)
    p : array of floats
        Point on the plane (3 x 1).
    """
    p = np.mean(points, axis=0)
    X = points - p
    cov_matrix = np.dot(X.T, X)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    n = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    R = eigvecs[:, [1, 2, 0]]  # Permute columns to create an orthonormal basis
    R /= np.linalg.norm(R, axis=0)

    return n, R, p


def smoothstep(x, xmin, xmax):
    """
    Sigmoid-like interpolation and clamping function.

    Parameters
    ----------
    x: array of floats
        Input values
    xmin: float
        Minimal x (output=0)
    xmax: float
        Maximal x (output=1)

    Returns
    -------
    array of floats
        Output values
    """
    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax")

    x = np.asarray(x, dtype="float")
    x = (x - xmin) / (xmax - xmin)
    x = np.minimum(np.maximum(x, 0), 1)
    return x**3 * (3 * x * (2 * x - 5) + 10)


def limit_bounds(x, x_min, x_max, typ=1):
    """
    Calculate gain for a limiter based on given bounds and type.

    Args:
        x: Input value.
        x_min: Lower bound.
        x_max: Upper bound.
        typ: Function type (optional):
             - 1: linear
             - 2-4: x^(typ)
             - 9: Custom type
             Default is 1.

    Returns:
        y: Output values.
    """
    if typ not in [1, 2, 3, 4, 9]:
        raise ValueError("Invalid typ value")

    x = np.clip(x, x_min, x_max)

    if typ in [2, 3, 4]:
        tmp = (x_max + x_min) / 2 - x
        y = tmp**typ * np.sign(tmp)
    elif typ == 9:
        x_mid = (x_max + x_min) / 2
        x_range = (x_max - x_min) / 2
        tmp = x_mid - x
        y = (np.maximum(1 / (np.abs(x - x_max) / x_range), 1 / (np.abs(x - x_min) / x_range)) - 1) * np.sign(tmp)
    else:
        y = (x_max + x_min) / 2 - x

    return y


def load_est(Ft, Rt):
    """
    Estimates F/T sensor load (mass and COM).

    Args:
        Ft: Force/torque measurements (n x 6) or (6 x n) numpy array.
        Rt: Sensor orientation matrix (3 x 3 x n) or quaternions (n x 4) numpy array.

    Returns:
        mass: Estimated mass load.
        COM: Estimated center of mass (3 x 1 numpy array).
        Off: Sensor offset (6 x 1 numpy array).
    """
    Ft = np.asarray(Ft, dtype="float")
    Rt = np.asarray(Rt, dtype="float")
    if Ft.shape[0] == 6:
        F = Ft[0:3, :].T
        M = Ft[3:6, :].T
        n = Ft.shape[1]
    else:
        F = Ft[:, 0:3]
        M = Ft[:, 3:6]
        n = Ft.shape[0]

    if len(Rt.shape) == 3:
        R = np.moveaxis(Rt, -1, 0)
    else:
        R = np.zeros((n, 3, 3))
        if Rt.shape[0] != 4:
            raise TypeError("Wrong input size")
        R = Quaternion.array(Rt).to_rotation_matrix

    A = R[:, 2, :]
    AI = np.hstack((A.reshape(-1, 1), np.tile(np.eye(3), (n, 1))))
    par = np.linalg.pinv(AI) @ F.ravel()
    mass = -par[0] / 9.81
    Foff = par[1:4]

    Fg = (F - np.repeat(np.expand_dims(Foff, 1).T, F.shape[0], axis=0)).T
    B = np.zeros((3 * n, 3))
    for i in range(n):
        v = Fg[:, i]
        B[3 * i : 3 * i + 3, :] = -np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    BI = np.hstack((B, np.tile(np.eye(3), (n, 1))))
    par = np.linalg.pinv(BI) @ M.ravel()
    COM = par[0:3]
    Moff = par[3:6]

    Off = np.hstack((Foff, Moff))

    return mass, COM, Off


def distance2line(p0, p, dir):
    """Find the closest point on line and calculate distance

    Parameters
    ----------
    p0 : array of floats
        point (3 x 1)
    p : array of floats
        point on line 1 (3 x 1)
    dir : array of floats
        direction of line (3 x 1)

    Returns
    -------
    array of floats
        closest point on line (3 x 1)
    float
        normal distance from point to line
    """
    p0 = vector(p0, dim=3)
    p = vector(p, dim=3)
    dir = normalize(vector(dir, dim=3))
    d = np.linalg.norm(np.cross(p0 - p, dir))
    pt = p + np.dot(p0 - p, dir) * dir
    return pt, d


def dist2lines(p1, dir1, p2, dir2, *args):
    p1 = vector(p1, dim=3)
    dir1 = normalize(vector(dir1, dim=3))
    p2 = vector(p2, dim=3)
    dir2 = normalize(vector(dir2, dim=3))
    if len(args) > 0:
        _eps = args[0]
    else:
        _eps = 1e-8

    dir12 = p2 - p1
    n1 = np.dot(dir1, dir1)
    n2 = np.dot(dir2, dir2)
    S1 = np.dot(dir1, dir12)
    S2 = np.dot(dir2, dir12)
    R = np.dot(dir1, dir2)
    den = n1 * n2 - R**2

    if (n1 == 0) or (n2 == 0):  # if one of the segments is a point
        if n1 != 0:  # if line1 is a segment and line2 is a point
            u = 0
            t = S1 / n1
        elif n2 != 0:  # if line2 is a segment and line 1 is a point
            t = 0
            u = -S2 / n2
        else:  # both segments are points
            t = 0
            u = 0
    elif den < _eps:  #  if lines are parallel
        t = 0
        u = -S2 / n2
    else:  # general case
        t = (S1 * n2 - S2 * R) / den
        u = (t * R - S2) / n2

    dist = np.linalg.norm(dir1 * t - dir2 * u - dir12)
    pts = np.vstack((p1 + dir1 * t, p2 + dir2 * u))
    return dist, pts


def normalize(x):
    """
    Normalize homogeneous matrix, rotation matrix, or vector.

    Parameters
    ----------
    x : array of floats
        Homogeneous transformation matrix (4 x 4), rotation matrix (3 x 3),
        or vector

    Returns
    -------
    array of floats
        Normalized transformation matrix (4 x 4), rotation matrix (3 x 3), or vector
    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    x = np.array(x)
    if x.shape == (3, 3):
        n = np.linalg.norm(T)
        if n < _eps:
            raise ValueError("Rotation matrix has zero norm")
        return x / n
    elif x.shape == (4, 4):
        T = x.copy()
        n = np.linalg.norm(T[0:3, 0:3])
        if n < _eps:
            raise ValueError("Rotation part has zero norm")
        T[0:3, 0:3] = T[0:3, 0:3] / n
        return T
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        n = np.linalg.norm(x)
        if n < _eps:
            raise ValueError("Vector has zero norm")
        return x / n
    else:
        raise TypeError("Invalid input shape or type")


def vecnormalize(x):
    """
    Normalize vector or rows of a matrix

    Parameters
    ----------
    x : array of floats
        matrix (n x m)

    Returns
    -------
    array of floats
        Normalized vector or matrix with normalized rows

    Raises
    ------
    TypeError
        Wrong argument shape or type
    """
    if isvector(x):
        return normalize(x)
    if ismatrix(x):
        return np.array([normalize(xx) for xx in x])
    else:
        raise TypeError("Invalid input shape or type")


def gradientPath_np(path, *args):
    """Calculate gradient along path

    Parameters
    ----------
    path : array of floats
        path samples (nsamp, n)
    args : scalar or array of floats, optional
        path parameter (nsamp,) or constant sample distance (scalar)

    Returns
    -------
    array of floats
        gradient along path (nsamp, n)

    Raises
    ------
    TypeError
        Wrong parameter shape
    """
    path = rbs_type(path)
    if len(args) == 0:
        return np.gradient(path, axis=0)
    else:
        if isscalar(args[0]):
            return np.gradient(path, args[0], axis=0)
        else:
            s = vector(args[0])
            if path.shape[0] == len(s):
                return np.gradient(path, s, axis=0)
            else:
                raise TypeError(f"Parameters must have same first dimension, but have shapes {path.shape} and {len(s)}")


def gradientPath(path, *args):
    """Calculate gradient along path

    Parameters
    ----------
    path : array of floats
        path samples (nsamp, n)
    args : scalar or array of floats, optional
        path parameter (nsamp,) or constant sample distance (scalar)

    Returns
    -------
    array of floats
        gradient along path (nsamp, n)

    Raises
    ------
    TypeError
        Wrong parameter shape
    """
    path = rbs_type(path)
    if isvector(path):
        _dpath = np.diff(path)
        _dpath = np.append(_dpath, _dpath[-1])
    elif ismatrix(path):
        _dpath = np.diff(path, axis=0)
        _dpath = np.vstack((_dpath, _dpath[-1, :]))
    else:
        raise TypeError("Wrong input path parameter shape")

    if len(args) == 0:
        return _dpath
    else:
        if isscalar(args[0]):
            return _dpath / args[0]
        else:
            s = vector(args[0])
            if path.shape[0] == len(s):
                _ds = np.diff(s)
                _ds = np.append(_ds, _ds[-1])
                return (_dpath.T / _ds).T
            else:
                raise TypeError(f"Parameters must have same first dimension, but have shapes {path.shape} and {len(s)}")


def gradientQuaternionPath(path, *args):
    """Calculate velocity along quaternion path

    Parameters
    ----------
    path : array of floats
        quaternion elements (nsamp, 4)
    args : scalar or array of floats, optional
        path parameter (nsamp,) or constant sample distance (scalar)

    Returns
    -------
    array of floats
        gradient along quaternion path (nsamp, 3)

    Raises
    ------
    TypeError
        Wrong parameter shape
    """
    path = rbs_type(path)
    if ismatrix(path, shape=4):
        if len(args) == 0:
            grad = gradientPath(path, 1)
        else:
            if isscalar(args[0]):
                grad = gradientPath(path, args[0])
            else:
                s = vector(args[0])
                if ismatrix(path, shape=(len(s), 4)):
                    grad = gradientPath(path, s)
                else:
                    raise TypeError(f"path must have dimension {(len(s), 4)}, but has {path.shape}")
    else:
        raise TypeError(f"path must have dimension (..., 4), but has {path.shape}")
    omega_q = 2 * (Quaternion.array(grad) * Quaternion.array(path).conj()).ndarray
    return omega_q[:, 1:]


def gradientCartesianPath(path, *args):
    """Calculate gradient along Cartesian path

    Poses are defined by position and quaternion.

    Parameters
    ----------
    path : array of floats
        Cartesian poses (nsamp, 7)
    args : scalar or array of floats, optional
        path parameter (nsamp,) or constant sample distance (scalar)

    Returns
    -------
    array of floats
        gradient along Cartesian path (nsamp, 6)
    """
    path = rbs_type(path)
    if ismatrix(path, shape=7):
        if len(args) == 0:
            v = gradientPath(path[:, :3])
            w = gradientQuaternionPath(path[:, 3:])
        else:
            v = gradientPath(path[:, :3], args[0])
            w = gradientQuaternionPath(path[:, 3:], args[0])
    else:
        raise TypeError(f"path must have dimension (..., 7), but has {path.shape}")
    return np.hstack((v, w))


def wrap_to_pi(x):
    if x == np.pi:
        return x
    else:
        return (x + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    a = vector((1, 2, 3), dim=3)
    print("a: ", a)
    print("Check if a is vector:", isvector(a))

    b = matrix((1, 2, 3, 4, 5, 6), shape=(2, 3))
    print("b: ", b)
    print("Check if b is matrix:", ismatrix(b))

    x = np.random.randint(0, 100, size=(4, 3, 3))
    print("x: ", x)
    print("Shape of x: ", x.shape)
    print("Check if shape of x is (..., 3, 3): ", check_shape(x, (3, 3)))

    print("Check if shape of b is 3: ", check_shape(b, 3))

    print("Check option - " "abC=Abc" ": ", check_option("abC", "Abc"))
    print("Check option - " "abC=Abx" ": ", check_option("abC", "Abx"))

    print("pi [rad] =", np.pi / getunit("deg"), "[deg]:")

    def fun(x):
        return (x[0] ** 2 + x[1] ** 2 - 1) ** 2

    x = np.asarray([0.0, 0.1])
    print("Fun: x[0] ** 2 + x[1] ** 2 - 1) ** 2")
    print("Fun([0.0, 0.1]) : ", fun(x))
    print("Grad([0.0, 0.1]): ", grad(fun, x))
    print("Hess([0.0, 0.1]): ", hessmat(fun, x))

    print(
        "Dead zone: deadzone([2.2, 0.3, 4], width=1.5, center=2) =",
        deadzone([2.2, 0.3, 4], width=1.5, center=2),
    )

    print(
        "Limiter: limit_bounds([1,2,3,4,5], 2, 4, typ=3) =",
        limit_bounds([1, 2, 3, 4, 5], 2, 4, typ=3),
    )

    print(
        "Sigmoid([0.1, 0.43], offset=1.2, gain=0.3): ",
        sigmoid([0.1, 0.43], offset=1.2, gain=0.3),
    )

    print(
        "Smoothstep([0., 2.65, 3., 5.], 2.5, 4): ",
        smoothstep([0.0, 2.65, 3.0, 5.0], 2.5, 4),
    )

    import scipy.io as sio

    # Generate some example data
    data = sio.loadmat("load_est_data.mat")
    Ft = data["Ft"]
    Rt = data["Rt"]

    # Estimate load
    mass, COM, Off = load_est(Ft, Rt)
    print("Estimated mass:\n", mass)
    print("Estimated center of mass:\n", COM)
    print("Sensor offset:\n", Off)

    np.random.seed(0)
    X = np.random.rand(50, 3)

    n, R, p = fitplane(X)

    print("Normal:", n)
    print("Basis:\n", R)
    print("Point on the plane:", p)

    pc, n, r, R = fit3dcirc(X, pl=True)
    plt.show()

    pt, d = distance2line([1, 2, 4], [1, -2, 0.5], [2, 1, -3])
    print("Distance to line:\n", pt, d)

    print("Wrap to pi: 5->", wrap_to_pi(5))
