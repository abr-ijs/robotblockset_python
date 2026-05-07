"""Utility Classes and Functions.

This module contains utility classes and functions for various operations commonly used
in robotics, engineering, and computational tasks. It includes object-oriented tools for debugging,
message logging, and mathematical utilities such as vector and matrix transformations, filtering,
interpolation, and optimization techniques. Additionally, several methods for handling rotation,
transformations, and load estimation are provided.

This module serves as a foundational toolset for more advanced robotic simulations, optimization tasks,
and data analysis processes involving vectors, matrices, and rotations.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from abc import ABCMeta
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, TextIO, Tuple, Union
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import quaternionic as Quaternion
import time
from datetime import datetime
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
import os
import importlib
from IPython.display import HTML, display
import pygments
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import logging

from robotblockset.rbs_typing import ArrayLike

_eps = np.finfo(np.float64).eps
_scalartypes = (int, float, np.int64, np.int32, np.float64)


def get_logger(name: str) -> logging.Logger:
    """Create and configure a package logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger configured with the standard Robotblockset stream handler.
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[RBS_%(levelname)s] [%(created).9f] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_rbs_path() -> str:
    """
    Return the absolute filesystem path to the installed `robotblockset` package.

    This function imports the `robotblockset` module and inspects its
    `__file__` attribute to determine the directory in which the package
    resides. It returns the directory containing the module, not the module
    file itself.

    Returns
    -------
    str
        Absolute path to the folder where the `robotblockset` package is located.

    Raises
    ------
    ModuleNotFoundError
        If the `robotblockset` module is not installed or cannot be imported.
    """
    module = importlib.import_module("robotblockset")
    return os.path.dirname(os.path.abspath(module.__file__))


class rbs_object(metaclass=ABCMeta):
    """
    A base class for objects with debugging and messaging capabilities.

    Attributes
    ----------
        Name (str): The name of the object, used in messages.
    """

    def __init__(self) -> None:
        """Initialize the base object state.

        Returns
        -------
        None
            This constructor initializes the object name and verbosity level.
        """
        self._verbose = 1
        self.Name = ""
        # self.logger = logging.getLogger(__name__)
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    def Message(self, msg: str, verb: int = 0, output: Optional[TextIO] = None) -> None:
        """
        Logs or prints a message if the verbosity level is sufficient.

        Parameters
        ----------
        msg : str
            The message to be logged or printed.
        verb : int, optional
            The verbosity level required to log or print the message. Defaults to 0.
        output : io.TextIOBase, optional
            A writable output stream (e.g., a file or StringIO) to write the message to.
            If None, the message is printed to the standard output. Defaults to None.

        Returns
        -------
        None
            This method writes the formatted message to the selected output.
        """
        LOGGER_LEVEL = {
            0: "INFO",
            1: "INFO",
            2: "DEBUG",
            3: "DEBUG",
        }
        if self._verbose >= verb:
            now = datetime.now()
            message = f"[RBS_{LOGGER_LEVEL.get(verb)}] [{now:%H:%M:%S}.{now.microsecond //1000:03d}] [{self.Name}]: {msg}"
            # message = f"[RBS_{LOGGER_LEVEL.get(verb)}] [{datetime.now().strftime('%H:%M:%S')}] [{self.Name}]: {msg}"
            if output:
                output.write(message + "\n")
            else:
                print(message)
        # if output:
        #     output.write(message + "\n")
        # else:
        #     if verb == 1:
        #         self.logger.info(message)
        #     elif verb == 2:
        #         self.logger.debug(message)
        #     elif verb == 0:
        #         self.logger.error(message)

    def WarningMessage(self, msg: str) -> None:
        """
        Displays a warning message with the name of the object and the provided message.

        Parameters
        ----------
        msg : str
            The warning message to display.

        Returns
        -------
        None
            This method prints the formatted warning message.
        """

        message = f"[RBS_WARN] [{time.time():.9f}] [{self.Name}]: {msg}"
        print(message)
        # self.logger.warning(message)

    def SetLogLevel(self, level: int) -> None:
        """
        Sets the log verbosity level for the application.

        Parameters
        ----------
        level : int
            Verbosity level to set. Higher values enable more detailed debug output.

        Returns
        -------
        None
            This method updates the stored verbosity level.
        """
        self._verbose = level
        # self.logger.setLevel(level.upper())

    def GetLogLevel(self) -> int:
        """
        Retrieve the current log level.

        Returns
        -------
        int
            Current verbosity level.
        """
        return self._verbose
        # return self.logger.level


class _struct:
    """
    Lightweight structure with dictionary-style introspection helpers.

    Instances expose convenience methods for converting stored attributes to a
    dictionary, iterating over key-value pairs, and populating attributes from
    an existing mapping.
    """

    def asdict(self) -> Dict[str, Any]:
        """
        Converts the attributes of the instance into a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the instance attributes as key-value pairs.
        """
        return vars(self)

    def __repr__(self) -> str:
        """
        Return the informal string representation of the structure.

        Returns
        -------
        str
            String representation of the stored attributes.
        """
        return self.__str__()

    def __str__(self) -> str:
        """
        Format object attributes so each key-value pair is shown on its own line.

        Returns
        -------
        str
            Multi-line string representation of the object's attributes.
        """
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over stored attributes.

        Returns
        -------
        Iterator[Tuple[str, Any]]
            Iterator yielding attribute name and value pairs.
        """
        for key, value in vars(self).items():
            yield key, value

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Populate attributes from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Mapping of attribute names to values.

        Returns
        -------
        None
            This method updates the instance in place.
        """
        for key, value in data.items():
            setattr(self, key, value)


@dataclass
class load_params(_struct):
    """
    Load parameters consisting of mass, center of mass (COM), and inertia tensor.

    This structure describes the physical properties of a tool or payload attached
    to the robot. These parameters are typically used by the robot controller
    (e.g., UR RTDE or MotoROS2) to properly account for dynamics, force estimation,
    and motion safety.

    Parameters
    ----------
    mass : float, optional
        Mass of the tool or payload in kilograms. Default is ``0.0``.
    COM : numpy.ndarray, optional
        Center of mass expressed in the tool coordinate system, shape ``(3,)``.
        Default is a zero vector.
    inertia : numpy.ndarray, optional
        3×3 inertia matrix about the tool's coordinate frame origin.
        Default is a zero matrix.

    Attributes
    ----------
    mass : float
        Mass of the attached load.
    COM : numpy.ndarray
        Center of mass vector.
    inertia : numpy.ndarray
        Inertia tensor matrix.
    """

    mass: float = 0.0
    COM: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __str__(self) -> str:
        """Return a nicely formatted display string."""
        return f"Load Parameters\n" f"  Mass    : {self.mass:.3f} kg\n" f"  COM     : {self.COM.tolist()}\n" f"  Inertia :\n" f"    {self.inertia[0].tolist()}\n" f"    {self.inertia[1].tolist()}\n" f"    {self.inertia[2].tolist()}"


@dataclass
class tool_params(_struct):
    """
    Represents tool parameters including TCP definition and associated load.

    Parameters
    ----------
    name : str
        Name of the tool (human-readable identifier).
    id : int
        Numeric ID corresponding to the controller's tool file index.
    tcp_position : np.ndarray
        3-element XYZ position of the tool center point (TCP) relative to the flange.
    tcp_orientation : np.ndarray
        4-element quaternion (w, x, y, z) describing orientation of the TCP.
    load : load_params
        Load parameters (mass, COM, inertia) associated with this tool.
    mounted_on : str
        Identifier of the link or frame this tool is mounted on (e.g., "flange").
    """

    name: str
    id: int
    tcp_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tcp_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    load: "load_params" = None
    mounted_on: str = "flange"

    def __str__(self) -> str:
        """Return a formatted parameter summary.

        Return a human-friendly formatted summary of tool parameters."""
        pos = np.array(self.tcp_position).tolist()
        ori = np.array(self.tcp_orientation).tolist()
        return f"Tool Parameters: {self.name} (ID={self.id})\n" f"  Mounted on : {self.mounted_on}\n" f"  TCP Position     : {pos}\n" f"  TCP Orientation  : {ori}\n" f"{str(self.load)}"


def rbs_type(x: ArrayLike) -> np.ndarray:
    """
    Convert input data to a squeezed float NumPy array.

    Parameters
    ----------
    x : ArrayLike
        Input data to convert.

    Returns
    -------
    np.ndarray
        Squeezed copy of the input data with `float` dtype.
    """

    return np.squeeze(np.copy(np.asarray(x, dtype="float")))


def isscalar(x: Any) -> bool:
    """
    Check if the input is a scalar.

    A scalar is defined as either an instance of a scalar type or a NumPy array
    with a single element.

    Parameters
    ----------
    x : Any
        Input value to check.

    Returns
    -------
    bool
        `True` if the input is a scalar value or a single-element NumPy array.
    """

    return isinstance(x, _scalartypes) or (isinstance(x, np.ndarray) and (x.size == 1))


def isvector(x: ArrayLike, dim: Optional[int] = None) -> bool:
    """
    Check if the input is a vector and optionally verify its dimension.

    This function determines whether the input `x` is a vector. A vector is defined
    as a one-dimensional array with more than one element. If the `dim` parameter
    is provided, the function also checks if the vector has the specified dimension.

    Parameters
    ----------
    x : ArrayLike
        The input value to check.
    dim : int, optional
        verify that the vector has this specific dimension.

    Returns
    -------
    bool
        Returns `True` if `x` is a vector. If `dim` is specified, returns `True`
        only if `x` is a vector with the specified dimension. Otherwise, returns `False`.
    """
    x = np.asarray(x)
    s = x.shape
    if dim is None:
        return len(s) == 1 and s[0] > 0
    else:
        return s == (dim,)


def vector(x: Union[ArrayLike, float, int], dim: Optional[int] = None) -> np.ndarray:
    """Convert input to a NumPy vector.

    Transforms input into a NumPy array vector with optional dimension validation.

    Parameters
    ----------
    x : Union[ArrayLike, float, int]
        Input values to transform into a vector.
    dim : int, optional
        Required vector length. If provided, the result is validated against it.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array representation of the input.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ValueError
        If `dim` is invalid or does not match the resulting vector length.
    """
    if isinstance(x, (list, tuple)):
        x = rbs_type(x).flatten()
    elif isscalar(x):
        x = rbs_type([x]).flatten()
    elif isinstance(x, np.ndarray):
        x = x.ravel()
    else:
        raise TypeError("Input must be a list, tuple, scalar, or NumPy array")
    if dim is not None:
        if dim < 1:
            raise ValueError(f"Dimension 'dim' must be a positive integer, but got {dim}")
        if x.size != dim:
            raise ValueError(f"Incorrect vector length {x.size} - expected {dim}")
    return x


def ismatrix(x: np.ndarray, shape: Optional[Union[int, Tuple[int, int]]] = None) -> bool:
    """Check if parameter is a matrix

    Tests if the argument is a 2D matrix with a specified ``shape``.
    If ``shape`` is scalar, then only the  last dimension of the argument
    is checked.

    Parameters
    ----------
    x : np.ndarray
        value to check
    shape : Union[int, Tuple[int, int]], optional
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


def ismatrixarray(x: np.ndarray, shape: Optional[Union[int, Tuple[int, int]]] = None) -> bool:
    """Check if parameter is a matrix array

    Tests if the argument is a array of 2D matrices with a specified ``shape``.

    Parameters
    ----------
    x : np.ndarray
        value to check
    shape : Union[int, Tuple[int, int]], optional
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


def matrix(x: Union[ArrayLike, float, int], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Return a matrix

    Parameters
    ----------
    x : Union[ArrayLike, float, int]
        values to be transformed to matrix
    shape : Tuple[int, int], optional
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


def check_shape(x: ArrayLike, shape: Union[int, Tuple[int, ...]]) -> bool:
    """Check last dimensions of np array

    Parameters
    ----------
    x : ArrayLike
        array to be checked
    shape : Union[int, Tuple[int, ...]]
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


def isskewsymmetric(S: np.ndarray, tol: float = 100) -> bool:
    """Check if matrix is skew-symmetric

    Parameters
    ----------
    S : np.ndarray
        value to check

    Returns
    -------
    bool
        True if S is skew-symmetric
    """
    return isinstance(S, np.ndarray) and np.linalg.norm(S + S.T) < tol * _eps


def isquaternion(Q: Any) -> bool:
    """Check if parameter is quaternionic array ``QArray``

    Parameters
    ----------
    Q : Any
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
    unit : str
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


def find_rows(row: np.ndarray, x: np.ndarray) -> List[int]:
    """Find indices of rows equal to a reference row.

    Parameters
    ----------
    row : np.ndarray
        Reference row to match.
    x : np.ndarray
        Matrix whose rows are searched.

    Returns
    -------
    List[int]
        Indices of rows in `x` that exactly match `row`.
    """
    idx = np.where((x == row).all(1))[0]
    return idx.tolist()


def grad(fun: Callable[[np.ndarray], float], x0: ArrayLike, dx: ArrayLike = 0.000001) -> Union[float, np.ndarray]:
    """Gradient of function at values x

    Parameters
    ----------
    fun :
        function handle
    x0 : ArrayLike
        function argument values
    dx : ArrayLike, optional
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


def hessmat(fun: Callable[[np.ndarray], float], x0: np.ndarray, delta: Optional[ArrayLike] = None) -> np.ndarray:
    """
    Hessian matrix of a scalar function with vector argument.

    The Hessian matrix of f(x) is the square matrix of the second partial
    derivatives of f(x).

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        Scalar function.
    x0 : np.ndarray
        Function argument of shape `(n,)`.
    delta : ArrayLike, optional
        Perturbation step used for numerical differentiation.

    Returns
    -------
    np.ndarray
        Hessian matrix of `fun` evaluated at `x0`.
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


def deadzone(x: ArrayLike, width: float = 1, center: float = 0) -> np.ndarray:
    """Apply a symmetric dead zone to input values.

    Parameters
    ----------
    x : ArrayLike
        Input values.
    width : float, optional
        Half-width of the zero-output interval around `center`.
    center : float, optional
        Center of the dead zone.

    Returns
    -------
    np.ndarray
        Input values shifted so values inside the dead zone map to zero.
    """
    x = np.asarray(x, dtype="float")
    xx = np.copy(x)
    _lower_limit = center - width
    _upper_limit = center + width
    xx[(x >= _lower_limit) & (x <= _upper_limit)] = 0
    xx[x < _lower_limit] -= _lower_limit
    xx[x > _upper_limit] -= _upper_limit
    return xx


def sigmoid(x: ArrayLike, offset: float = 0.0, gain: float = 1.0) -> np.ndarray:
    """Sigmoid function

    Parameters
    ----------
    x : ArrayLike
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


def smoothstep(x: ArrayLike, xmin: float, xmax: float) -> np.ndarray:
    """
    Sigmoid-like interpolation and clamping function.

    Parameters
    ----------
    x : ArrayLike
        Input values
    xmin : float
        Minimal x (output=0)
    xmax : float
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


def smoothstep3(x: float, x_min: float, x_max: float) -> float:
    """
    Performs smooth Hermite interpolation between 0 and 1 when `x` is in the range [`x_min`, `x_max`].

    This function interpolates smoothly between 0 and 1 based on the position of `x` between `x_min` and `x_max`.
    Values of `x` less than `x_min` return 0.0, and values greater than `x_max` return 1.0.

    Parameters
    ----------
    x : float
        The input value to interpolate.
    x_min : float
        The lower edge of the interpolation range.
    x_max : float
        The upper edge of the interpolation range.

    Returns
    -------
    float
        The interpolated value between 0.0 and 1.0.
    """
    x = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def fit3dcirc(X: np.ndarray, pl: bool = False) -> Union[
    Tuple[np.ndarray, np.ndarray, float, np.ndarray],
    Tuple[np.ndarray, np.ndarray, float, np.ndarray, list],
]:
    """
    Fit a circle to a set of 3D points.

    Parameters
    ----------
    X : np.ndarray
        Set of points (n x 3)
    pl : bool, optional
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


def fitplane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a plane to a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
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


def limit_bounds(x: ArrayLike, x_min: float, x_max: float, typ: int = 1) -> np.ndarray:
    """
    Calculate gain for a limiter based on given bounds and type.

    Parameters
    ----------
    x : ArrayLike
        Input value.
    x_min : float
        Lower bound.
    x_max : float
        Upper bound.
    typ : int, optional
        Limiter function type. Supported values are:

        - ``1``: linear
        - ``2`` to ``4``: polynomial ``x**typ``
        - ``9``: custom reciprocal limiter

        Default is ``1``.

    Returns
    -------
    np.ndarray
        Output values after applying the limiter.
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


def motion_for_load_est(r: Any, q_init: Optional[np.ndarray] = None, n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a set of poses using the last 3 joints and record forces & orientations.

    It moves the robot around its home configuration by randomly perturbing
    the last three joints, then measures the force/torque and sensor orientation.

    Parameters
    ----------
    r : Any
        Robot object with an attached force/torque sensor.
    n : int, optional
        Number of samples to collect. Default is 50.

    Returns
    -------
    Ft : np.ndarray, shape (n, 6)
        Measured forces/torques for each sample.
    Rt : np.ndarray, shape (3, 3, n)
        Measured sensor orientations as rotation matrices for each sample.

    Raises
    ------
    ValueError
        If `r` does not appear to be a valid robot object.
    """
    if q_init is None:
        q_init = r.q_home

    Ft = np.full((n, 6), np.nan)
    Rt = np.full((n, 3, 3), np.nan)

    # Move to home
    r.JMove(q_init, 1)

    r.Load.mass = 0
    if r.FTSensor is not None:
        r.FTSensor.SetOffset(np.zeros(6))

    # Main sampling loop
    for i in range(n):
        print(f"{i + 1} of {n}")

        # Random perturbation of last three joints
        mask = np.array([0, 0, 0, 0, 1, 1, 1], dtype=float)
        delta = mask * (np.random.rand(r.nj) - 0.5) * 2
        qx = q_init + delta

        r.JMove(qx)  # move in 1 second
        r.Wait(1)

        # Read sensor FT
        if r.FTSensor is None:
            Ft[i] = np.asarray(r.GetFT())

        else:
            Ft[i] = np.asarray(r.FTSensor.GetFT())

        # Read sensor orientation
        Rt[i] = r.GetFTFramePose(task_space="World", out="R")

    return Ft, Rt


def load_est(Ft: ArrayLike, Rt: ArrayLike) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimates F/T sensor load (mass and COM).

    Parameters
    ----------
        Ft : ArrayLike
            Force/torque measurements (n x 6) or (6 x n) numpy array.
        Rt : ArrayLike
            Sensor orientation matrix (3 x 3 x n) or quaternions (n x 4) numpy array.

    Returns
    -------
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
        R = Rt
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


def load_tools_from_yaml(path: Union[str, Path]) -> tuple[Dict[str, tool_params], Optional[str]]:
    """
    Load tool definitions from a YAML file.

    Parameters
    ----------
    path : Union[str, Path]
        str or Path
        Path to the YAML file containing tool definitions.

    Returns
    -------
    tools : dict[str, tool_params]
        Dictionary mapping tool names to parsed tool_params objects.
    default_tool : str or None
        Name of the default tool from the YAML (if defined), otherwise None.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tools_cfg = cfg.get("tools", {})
    default_tool = cfg.get("default_tool")

    tools: Dict[str, tool_params] = {}

    for name, tool_cfg in tools_cfg.items():
        tcp_cfg = tool_cfg.get("tcp", {})
        load_cfg = tool_cfg.get("load", {})

        tcp_pos = vector(tcp_cfg.get("position", [0.0, 0.0, 0.0]), dim=3)
        tcp_ori = vector(tcp_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]), dim=4)

        id = int(tool_cfg.get("id", 0))
        mass = float(load_cfg.get("mass", 0.0))
        com = vector(load_cfg.get("com", [0.0, 0.0, 0.0]), dim=3)
        inertia = matrix(
            load_cfg.get("inertia", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            shape=(3, 3),
        )
        mounted_on = str(tool_cfg.get("mounted_on", "Flange"))

        load = load_params(mass=mass, COM=com, inertia=inertia)
        tool = tool_params(
            name=name,
            id=id,
            tcp_position=tcp_pos,
            tcp_orientation=tcp_ori,
            load=load,
            mounted_on=mounted_on,
        )
        tools[name] = tool

    return tools, default_tool


def distance2line(p0: ArrayLike, p: ArrayLike, dir: ArrayLike) -> Tuple[np.ndarray, float]:
    """Find the closest point on line and calculate distance

    Parameters
    ----------
    p0 : ArrayLike
        point (3 x 1)
    p : ArrayLike
        point on line 1 (3 x 1)
    dir : ArrayLike
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


def dist2lines(p1: ArrayLike, dir1: ArrayLike, p2: ArrayLike, dir2: ArrayLike, *args: float) -> Tuple[float, np.ndarray]:
    """Compute the shortest distance between two 3D lines.

    Parameters
    ----------
    p1 : ArrayLike
        Point on the first line.
    dir1 : ArrayLike
        Direction vector of the first line.
    p2 : ArrayLike
        Point on the second line.
    dir2 : ArrayLike
        Direction vector of the second line.
    *args : float
        Optional numerical tolerance used for near-parallel lines.

    Returns
    -------
    Tuple[float, np.ndarray]
        Shortest distance and the corresponding displacement vector.
    """
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
    elif den < _eps:  # if lines are parallel
        t = 0
        u = -S2 / n2
    else:  # general case
        t = (S1 * n2 - S2 * R) / den
        u = (t * R - S2) / n2

    dist = np.linalg.norm(dir1 * t - dir2 * u - dir12)
    pts = np.vstack((p1 + dir1 * t, p2 + dir2 * u))
    return dist, pts


def normalize(x: ArrayLike, eps: float = _eps) -> np.ndarray:
    """
    Normalize homogeneous matrix, rotation matrix, or vector.

    Parameters
    ----------
    x : ArrayLike
        Homogeneous transformation matrix (4 x 4), rotation matrix (3 x 3),
        or vector
    eps : float, optional
        Small threshold to prevent division by zero.

    Returns
    -------
    array of floats
        Normalized transformation matrix (4 x 4), rotation matrix (3 x 3), or vector

    Raises
    ------
    TypeError
        Wrong argument shape or type
    ValueError
        If any input has zero norm.
    """
    x = np.array(x)
    if x.shape == (3, 3):
        n = np.linalg.norm(x)
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
    else:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        if np.any(n < eps):  # Check if any norm is too small
            raise ValueError("Vector has zero norm or is too close to zero.")

        return x / n  # Normalize each vector


def vecnormalize(x: ArrayLike) -> np.ndarray:
    """
    Normalize vector or rows of a matrix

    Parameters
    ----------
    x : ArrayLike
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
    if not isscalar(x):
        return normalize(x)
    else:
        raise TypeError("Invalid input shape")


def gradientPath_np(path: ArrayLike, path_par: Optional[ArrayLike] = None) -> np.ndarray:
    """Calculate gradient along path

    Parameters
    ----------
    path : ArrayLike
        path samples (nsamp, n)
    path_par : ArrayLike, optional
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
    if path_par is None:
        return np.gradient(path, axis=0)
    else:
        if isscalar(path):
            return np.gradient(path, path_par, axis=0)
        else:
            s = vector(path_par)
            if path.shape[0] == len(s):
                return np.gradient(path, s, axis=0)
            else:
                raise TypeError(f"Parameters must have same first dimension, but have shapes {path.shape} and {len(s)}")


def gradientPath(path: ArrayLike, path_par: Optional[ArrayLike] = None) -> np.ndarray:
    """Calculate gradient along path

    Parameters
    ----------
    path : ArrayLike
        path samples (nsamp, n)
    path_par : ArrayLike, optional
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

    if path_par is None:
        return _dpath
    else:
        if isscalar(path_par):
            return _dpath / path_par
        else:
            s = vector(path_par)
            if path.shape[0] == len(s):
                _ds = np.diff(s)
                _ds = np.append(_ds, _ds[-1])
                return (_dpath.T / _ds).T
            else:
                raise TypeError(f"Parameters must have same first dimension, but have shapes {path.shape} and {len(s)}")


def gradientQuaternionPath(path: ArrayLike, path_par: Optional[ArrayLike] = None) -> np.ndarray:
    """Calculate velocity along quaternion path

    Parameters
    ----------
    path : ArrayLike
        quaternion elements (nsamp, 4)
    path_par : ArrayLike, optional
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
        if path_par is None:
            grad = gradientPath(path, 1)
        else:
            if isscalar(path_par):
                grad = gradientPath(path, path_par)
            else:
                s = vector(path_par)
                if ismatrix(path, shape=(len(s), 4)):
                    grad = gradientPath(path, s)
                else:
                    raise TypeError(f"path must have dimension {(len(s), 4)}, but has {path.shape}")
    else:
        raise TypeError(f"path must have dimension (..., 4), but has {path.shape}")
    omega_q = 2 * (Quaternion.array(grad) * Quaternion.array(path).conj()).ndarray
    return omega_q[:, 1:]


def gradientCartesianPath(path: ArrayLike, path_par: Optional[ArrayLike] = None) -> np.ndarray:
    """Calculate gradient along Cartesian path

    Poses are defined by position and quaternion.

    Parameters
    ----------
    path : ArrayLike
        Cartesian poses (nsamp, 7)
    path_par : ArrayLike, optional
        path parameter (nsamp,) or constant sample distance (scalar)

    Returns
    -------
    array of floats
        gradient along Cartesian path (nsamp, 6)
    """
    path = rbs_type(path)
    if ismatrix(path, shape=7):
        if path_par is None:
            v = gradientPath(path[:, :3])
            w = gradientQuaternionPath(path[:, 3:])
        else:
            v = gradientPath(path[:, :3], path_par)
            w = gradientQuaternionPath(path[:, 3:], path_par)
    else:
        raise TypeError(f"path must have dimension (..., 7), but has {path.shape}")
    return np.hstack((v, w))


def wrap_to_pi(x: ArrayLike) -> np.ndarray:
    """Wrap an angle or array of angles to the range [-π, π].

    Parameters
    ----------
    x : ArrayLike
        Input angle(s) in radians.

    Returns
    -------
    float or np.ndarray
        Wrapped angle(s) in the range [-π, π).
    """
    x = np.asarray(x)  # Ensure input is a NumPy array
    return (x + np.pi) % (2 * np.pi) - np.pi


def limit_rate(value_prev: float, value_target: float, max_delta: float) -> float:
    """
    Limits the rate of change from a previous value toward a target value.

    Ensures that the change from `value_prev` to `value_target` does not exceed `max_delta`
    in either direction. Useful for smoothing or clamping rate of change in control systems,
    animations, or simulations.

    Parameters
    ----------
    value_prev : float
        The previous or current value.
    value_target : float
        The desired target value.
    max_delta : float
        The maximum allowed change per step.

    Returns
    -------
    float
        The new value after applying the limited rate of change.
    """
    delta = value_target - value_prev
    delta_clipped = max(-max_delta, min(max_delta, delta))
    return value_prev + delta_clipped


def damped_pinv(A: np.ndarray, lambda_factor: float = 0.01) -> np.ndarray:
    """Compute the damped least squares pseudo-inverse of a matrix.

    The function applies Tikhonov regularization to stabilize the computation
    of the pseudoinverse, particularly for ill-conditioned or singular matrices.

    Parameters
    ----------
    A : np.ndarray
        Input matrix for which the damped pseudoinverse is computed.
    lambda_factor : float, optional
        Regularization parameter controlling the damping effect.
        A higher value increases numerical stability but may reduce accuracy.

    Returns
    -------
    array-like, shape (n, m)
        Damped pseudoinverse of the input matrix A.
    """
    m, n = A.shape
    II = np.eye(n)  # Identity matrix of size n x n
    return np.linalg.inv(A.T @ A + lambda_factor**2 * II) @ A.T


def filtfilt(signal_data: np.ndarray, cutoff: Union[float, ArrayLike] = 0.1, fs: float = 1.0, order: int = 4, filter_type: str = "low") -> np.ndarray:
    """
    Applies a zero-phase filter (equivalent to MATLAB's filtfilt) on multidimensional signals.
    Each column in `signal_data` is treated as an independent signal.

    Parameters
    ----------
    signal_data : np.ndarray
        The input noisy signals, where each column is a separate signal.
    cutoff : Union[float, ArrayLike], optional
        The cutoff frequency (normalized: 0 < cutoff < 1, or in Hz if fs is provided).
    fs : float, optional
        Sampling frequency. If provided, cutoff is interpreted in Hz. Defaults to 1.0.
    order : int, optional
        The order of the filter. Higher order provides a sharper cutoff. Defaults to 4.
    filter_type : str, optional
        Type of filter to apply. Options: "low", "high", "bandpass", "bandstop". Defaults to "low".

    Returns
    -------
    np.ndarray, shape (N, M)
        The filtered signals with zero-phase distortion.
    """
    signal_data = np.atleast_2d(signal_data)  # Ensure 2D (N, M) shape
    nyquist = 0.5 * fs
    norm_cutoff = np.array(cutoff) / nyquist

    # Design Butterworth filter
    b, a = signal.butter(order, norm_cutoff, btype=filter_type, analog=False)

    # Apply filtfilt on each column independently
    filtered_signals = np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), axis=0, arr=signal_data)

    return filtered_signals


def find_closest_row(array_2d: np.ndarray, array_1d: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the row in a 2D NumPy array that is closest to a given 1D array based on the Euclidean distance.

    Parameters
    ----------
    array_2d : np.ndarray
        A 2D NumPy array where each row represents a vector.
    array_1d : np.ndarray
        A 1D NumPy array (vector) to compare against the rows of `array_2d`.

    Returns
    -------
    tuple
        A tuple containing:
        - The index of the closest row in `array_2d`.
        - The closest row itself from `array_2d`.

    Notes
    -----
    The distance is calculated using the Euclidean distance formula:
    d(a, b) = sqrt(sum((a_i - b_i)^2) for each element i in the vectors a and b).
    """
    distances = np.linalg.norm(array_2d - array_1d, axis=1)
    closest_row_index = np.argmin(distances)
    return closest_row_index, array_2d[closest_row_index]


def search_valid_range(fun: Callable[[float], Optional[Any]], x_min: float, x_max: float, tolerance: float = 1e-6) -> Tuple[float, float]:
    """
    Use binary search to find the valid subrange [x_valid_min, x_valid_max].

    Parameters
    ----------
    fun : Callable[[float], Optional[Any]]
        The function to check validity of x values.
    x_min : float
        The minimum value of x.
    x_max : float
        The maximum value of x.
    tolerance : float, optional
        The tolerance for checking validity (default is 1e-6).

    Returns
    -------
    tuple
        A tuple containing the valid subrange (q7_valid_min, q7_valid_max).
    """
    # Use binary search to find the valid range
    left = x_min
    right = x_max

    # Find q7_valid_min
    while right - left > tolerance:
        mid = (left + right) / 2
        if fun(mid) is not None:
            right = mid  # valid found, shrink the search space
        else:
            left = mid  # invalid, search on the other side

    q7_valid_min = (left + right) / 2

    # Reset search space to find q7_valid_max
    left = x_min
    right = x_max

    # Find q7_valid_max
    while right - left > tolerance:
        mid = (left + right) / 2
        if fun(mid) is not None:
            left = mid  # valid found, shrink the search space
        else:
            right = mid  # invalid, search on the other side

    q7_valid_max = (left + right) / 2

    return q7_valid_min, q7_valid_max


def parse_rbs_log(lines: Iterable[str]) -> List[Tuple[float, str]]:
    """
    Parse ROS2 log lines and extract **relative timestamp** and event description.

    The first timestamp encountered is treated as t = 0.

    Parameters
    ----------
    lines : Iterable[str]
        Iterable of log lines (e.g. list of strings, open file handler).

    Returns
    -------
    list of (float, str)
        List of tuples:
        - relative timestamp (seconds since first entry),
        - event description text.
    """
    LOG_LINE_RE = re.compile(r"^\[(RBS_DEBUG|RBS_INFO|RBS_WARN|DEBUG|INFO|WARN|ERROR|FATAL)\]\s+\[(\d+\.\d+)\]\s+\[[^\]]+\]:\s*(.*)$")
    events: List[Tuple[float, str]] = []
    t0 = None  # first timestamp reference

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = LOG_LINE_RE.match(line)
        if not m:
            continue

        # level = m.group(1)  # Available if needed
        ts_str = m.group(2)
        msg = m.group(3)

        try:
            ts = float(ts_str)
        except ValueError:
            continue

        if t0 is None:
            t0 = ts  # first timestamp defines t=0

        events.append((ts - t0, msg))
    events.sort(key=lambda e: e[0])
    return events


def print_xml(xml_string: str, is_dark: bool = True) -> None:
    """Render highlighted XML in a notebook environment.

    Parameters
    ----------
    xml_string : str
        XML text to display.
    is_dark : bool, optional
        Select a dark or light Pygments style.

    Returns
    -------
    None
        This function displays formatted HTML output.
    """
    print_style = "monokai" if is_dark else "lovelace"

    formatter = pygments.formatters.HtmlFormatter(style=print_style)
    lexer = pygments.lexers.XmlLexer()
    highlighted = pygments.highlight(xml_string, lexer, formatter)
    display(HTML(f"<style>{formatter.get_style_defs()}</style>{highlighted}"))


# ANSI colors
C_TAG = "\033[95m"  # magenta
C_ATTR = "\033[94m"  # blue
C_VALUE = "\033[92m"  # green
C_TEXT = "\033[0m"  # reset
C_COMMENT = "\033[90m"  # grey
C_RESET = "\033[0m"


def print_xml_for_console(xml_string: str) -> str:
    """Format XML for console output.

    Parameters
    ----------
    xml_string : str
        XML text to pretty-print and colorize.

    Returns
    -------
    str
        Colorized XML string formatted for terminal output.
    """

    # ---- Pretty print XML ----
    pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # ---- Remove blank / whitespace-only lines ----
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip() != "")

    # ---- Color comments <!-- ... --> ----
    pretty = re.sub(
        r"(<!--.*?-->)",
        lambda m: C_COMMENT + m.group(1) + C_RESET,
        pretty,
        flags=re.DOTALL,
    )

    # ---- Color attribute values  attr="value" ----
    pretty = re.sub(
        r'="(.*?)"',
        lambda m: "=" + C_VALUE + '"' + m.group(1) + '"' + C_RESET,
        pretty,
    )

    # ---- Color attribute names ----
    pretty = re.sub(
        r"(\s+)([a-zA-Z0-9_\-:]+)=",
        lambda m: m.group(1) + C_ATTR + m.group(2) + C_RESET + "=",
        pretty,
    )

    # ---- Color tags <tag> and </tag> ----
    pretty = re.sub(
        r"(<[^>]+>)",
        lambda m: C_TAG + m.group(1) + C_RESET,
        pretty,
    )

    print(pretty)


def replace_attr_values_in_xml(xml_text: str, old: str, new: str, substring: bool = False) -> tuple[str, int]:
    """
    Replace attribute values in MJCF XML text.

    Parameters
    ----------
    xml_text : str
        XML document text.
    old : str
        Attribute value to replace.
    new : str
        Replacement value.
    substring : bool, optional
        If `True`, replace matching substrings inside attribute values; otherwise require exact matches.

    Returns
    -------
    tuple[str, int]
        Updated XML text and the number of replacements made.
    """
    root = ET.fromstring(xml_text)

    replacements = 0
    for elem in root.iter():
        for attr, val in list(elem.attrib.items()):
            if substring:
                if old in val:
                    elem.set(attr, val.replace(old, new))
                    replacements += 1
            else:
                if val == old:
                    elem.set(attr, new)
                    print(f"{elem.tag}  {attr}: {old}->{new}")
                    replacements += 1

    new_xml = ET.tostring(root, encoding="unicode")
    return new_xml, replacements


def find_attr_values_in_xml(xml_text: str, value: str, substring: bool = False) -> tuple[str, int]:
    """Find XML attributes whose values match a target string.

    Parameters
    ----------
    xml_text : str
        XML document text.
    value : str
        Attribute value to search for.
    substring : bool, optional
        If `True`, match attributes containing `value`; otherwise require an exact match.

    Returns
    -------
    list[list[str]]
        Matching entries as `[parent, element_type, element_name, attribute_name]`.
    """
    root = ET.fromstring(xml_text)

    parent = ""
    elements = []
    for elem in root.iter():
        elem_type = elem.tag
        elem_name = elem.attrib.get("name", "(no-name)")
        if len(elem.attrib.items()) == 0:
            parent = elem_type

        for attr, val in list(elem.attrib.items()):
            match = False
            if substring:
                if value in val:
                    match = True
            else:
                if value == val:
                    match = True
            if match:
                elements.append([parent, elem_type, elem_name, attr])
                # print(f"Parent: <{parent}>  Type: <{elem_type}>  Name: '{elem_name}'  Attr: '{attr}'  Value: '{val}'")
    return elements


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

    def fun(x: np.ndarray) -> float:
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
