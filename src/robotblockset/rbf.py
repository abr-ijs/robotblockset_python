"""Radial Basis Function (RBF) Interpolation Module.

This module provides a collection of functions to work with Radial Basis Function (RBF) networks,
which are commonly used for interpolation, regression, and function approximation tasks. The RBFs
are computed using Gaussian kernel functions (GKF) and are capable of modeling complex paths and signals.

All functions work with RBFs computed using Gaussian kernels, which are expressed as:

$GKF(x) = exp(-((x - c)^2 / (2 * sigma2)))$

Where `x` is the path parameter, `c` is the center of the kernel, and `sigma2` is the standard deviation of the kernel.

The recursive regression methods applied in `updateRBF` use a least-squares approach to iteratively update the weights,
while other functions provide tools for decoding paths, calculating derivatives, and working with quaternions or Cartesian coordinates.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from robotblockset.tools import _eps, isscalar, isvector
from robotblockset.rbs_typing import ArrayLike, Poses3DType, QuaternionsType


def encodeRBF(x: ArrayLike, y: ArrayLike, N: int = 25, c: Optional[ArrayLike] = None, sigma2: Optional[ArrayLike] = None, bc: Optional[ArrayLike] = None, coff: float = 0.02, sfac: float = 3.0) -> Dict[str, np.ndarray]:
    """
    Encode path y(x) with Radial Basis Functions (RBF) by calculating weights for Gaussian Kernel Functions (GKF).

    The Gaussian Kernel Function (GKF) is defined as:

    $GKF(x) = $exp(-((x - c)^2 / (2 * sigma2)))$

    The Radial Basis Function (RBF) is a weighted sum of these Gaussian kernel functions:
    RBF(x) = $sum(w * GKF(x)) / sum(GKF(x))$

    Optionally, initial and final velocity and acceleration boundary conditions can be defined, which will be
    included in the matrix for solving the RBF weights.

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)

    Parameters
    ----------
    x : ArrayLike
        Path parameter (n, ). A 1D array representing the path or domain of the function.
    y : ArrayLike
        Measured signals (n, m). A 2D array of shape (n, m) where each column corresponds to a signal.
    N : int, optional
        The number of Gaussian kernel functions (GKF) used. Default is 25.
    c : ArrayLike, optional
        equidistantly or set to `x` if `n == N`.
    sigma2 : ArrayLike, optional
        `(diff(c) * 0.75)**2`.
    bc : ArrayLike, optional
        Boundary conditions of the form `[ydot(0), ydot(end), yddot(0), yddot(end)]` (4, m), where m is the number of signals.
        If not provided, no boundary conditions are applied.
    coff : float, optional
        The relative offset for auxiliary GKF centers. Default is 0.02, with typical values between 0.01 and 0.05.
    sfac : float, optional
        A scaling factor for sigma2 in auxiliary GKF. Default is 3.0.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the RBF parameters
    Raises
    ------
    ValueError
        If the input parameters are incorrect, such as mismatched dimensions.

    Notes
    -----
    - The function applies Radial Basis Function (RBF) interpolation to the input path `x` and corresponding measured signals `y`.
    - The boundary conditions `bc` are applied if provided, influencing the RBF computation.
    - The solution involves constructing a matrix `A` and solving for the weights `w` by using a pseudo-inverse.
    """

    if not isvector(x):
        raise ValueError("Parameter x is not vector")
    x = np.asarray(x, dtype="float")
    y = np.asarray(y, dtype="float")
    n = len(x)
    if y.shape[0] != n:
        raise ValueError(f"Parameter x and y do not have corresponding shapes {x.shape} and {y.shape}")

    if c is None:
        if N == n:
            c = x
        else:
            c = np.linspace(np.min(x), np.max(x), N)
    elif len(c) < 2:
        raise ValueError("Parameter c must have at least two elements to compute sigma2")

    if sigma2 is None:
        sigma2 = (np.diff(c) * 0.75) ** 2
        sigma2 = np.concatenate((sigma2, [sigma2[-1]]))
    elif isscalar(sigma2):
        _sigma2 = (np.diff(c) * sigma2) ** 2
        sigma2 = np.concatenate((_sigma2, [_sigma2[-1]]))

    if bc is not None:
        N = N + 4
        dc = np.diff(c) * coff
        c = np.concatenate((c, [c[0] + dc[0], c[-1] - dc[-1], c[0] + 2 * dc[0], c[-1] - 2 * dc[-1]]))
        sigma2 = np.concatenate((sigma2, sigma2[[0, -1, 0, -1]] / sfac**2))
        y = np.concatenate((y, bc), axis=0)

    x = x.reshape(-1, 1)
    RBF = {"N": N, "c": c, "sigma2": sigma2}

    tmp1 = x - c
    tmp2 = -0.5 * tmp1**2
    tmp3 = tmp2 / sigma2
    f = np.exp(tmp3)
    h = np.sum(f, axis=1)

    if bc is not None:
        tmp6 = -tmp1 / sigma2
        fd = tmp6 * f
        hd = np.sum(fd, axis=1)
        u = fd * h[:, np.newaxis] - f * hd[:, np.newaxis]
        Ad = u / (h**2)[:, np.newaxis]

        tmp8 = (-2 * tmp3 - 1) / sigma2
        fdd = tmp8 * f
        hdd = np.sum(fdd, axis=1)

        a1 = fdd / h[:, np.newaxis]
        a2 = 2 * fd * hd[:, np.newaxis] / (h**2)[:, np.newaxis]
        a3 = 2 * f * (hd**2)[:, np.newaxis] / (h**3)[:, np.newaxis]
        a4 = f * hdd[:, np.newaxis] / (h**2)[:, np.newaxis]
        Add = a1 - a2 + a3 - a4

        A = np.concatenate(
            (
                f / (h + _eps)[:, np.newaxis],
                Ad[[0, -1], :],
                Add[[0, -1], :],
            ),
            axis=0,
        )
    else:
        A = f / (h + _eps)[:, np.newaxis]

    AI = np.linalg.pinv(A)
    RBF["w"] = np.dot(AI, y)

    return RBF


def decodeRBF(x: ArrayLike, RBF: Dict[str, np.ndarray], calc_derivative: int = 0) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate path at points `x` encoded by Gaussian Radial Basis Functions (RBF) with Gaussian Kernel Functions (GKF).

    The Gaussian Kernel Function (GKF) is defined as:

    $GKF(x) = exp(-((x - c)^2 / (2 * sigma2)))$

    The Radial Basis Function (RBF) is a weighted sum of these Gaussian kernel functions:

    $RBF(x) = sum(w * GKF(x)) / sum(GKF(x))$

    Optionally, derivatives of the path (velocity, acceleration, jerk) can be calculated by setting the `calc_derivative` parameter.

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)

    Parameters
    ----------
    x : ArrayLike
        Path parameter (n,). A 1D array of the path or domain where the RBF will be evaluated.
    RBF : Dict[str, np.ndarray]
        A dictionary containing the RBF parameters
    calc_derivative : int, optional
        The order of the derivative to calculate from 0 to 3(default is 0, which means no derivatives). The options are:

    Returns
    -------
    y : np.ndarray
        Path values (n, m). The values of the path at the points `x`.
    yd : np.ndarray, optional
        Path velocities (n, m), returned if `calc_derivative >= 1`.
    ydd : np.ndarray, optional
        Path accelerations (n, m), returned if `calc_derivative >= 2`.
    yddd : np.ndarray, optional
        Path jerks (n, m), returned if `calc_derivative == 3`.

    Raises
    ------
    ValueError
        If input parameters are invalid or mismatched in dimensions.

    Notes
    -----
    The function uses the Gaussian kernel function to generate the path and its derivatives.
    If `calc_derivative` is set to a value greater than 0, it computes the respective derivatives (velocity, acceleration, or jerk) using finite differences.
    """
    if not isvector(x):
        raise ValueError("Parameter x is not a vector")

    x = np.asarray(x, dtype="float")
    x = x.reshape(-1, 1)

    # Compute the Gaussian kernel function for the given input
    tmp1 = x - RBF["c"]
    tmp2 = -0.5 * tmp1**2
    tmp3 = tmp2 / RBF["sigma2"]
    f = np.exp(tmp3)
    h = np.sum(f, axis=1)
    A = f / (h + _eps)[:, np.newaxis]

    # Compute the path values
    y = np.dot(A, RBF["w"])

    # Compute the first derivative (velocity) if requested
    if calc_derivative == 0:
        return y
    else:
        tmp6 = -tmp1 / RBF["sigma2"]
        fd = tmp6 * f
        hd = np.sum(fd, axis=1)
        u = fd * h[:, np.newaxis] - f * hd[:, np.newaxis]
        Ad = u / (h**2)[:, np.newaxis]

        ydot = np.dot(Ad, RBF["w"])

        if calc_derivative == 1:
            return y, ydot
        else:
            # Compute the second derivative (acceleration)
            tmp8 = (-2 * tmp3 - 1) / RBF["sigma2"]
            fdd = tmp8 * f
            hdd = np.sum(fdd, axis=1)

            a1 = fdd / h[:, np.newaxis]
            a2 = 2 * fd * hd[:, np.newaxis] / (h**2)[:, np.newaxis]
            a3 = 2 * f * (hd**2)[:, np.newaxis] / (h**3)[:, np.newaxis]
            a4 = f * hdd[:, np.newaxis] / (h**2)[:, np.newaxis]
            Add = a1 - a2 + a3 - a4

            yddot = np.dot(Add, RBF["w"])

            if calc_derivative == 2:
                return y, ydot, yddot
            else:
                # Compute the third derivative (jerk)
                tmp9 = (2 * tmp3 + 3) / RBF["sigma2"]
                tmp10 = -tmp6 * tmp9
                fddd = tmp10 * f
                hddd = np.sum(fddd, axis=1)

                b1 = fddd * (h**2)[:, np.newaxis] + fdd * (2 * h * hd)[:, np.newaxis]
                b2 = 2 * (fdd * (h * hd)[:, np.newaxis] + fd * (hd**2)[:, np.newaxis] + fd * (h * hdd)[:, np.newaxis])
                b3 = 2 * (fd * (hd**2)[:, np.newaxis] + f * (2 * hd * hdd)[:, np.newaxis])
                b4 = fd * (h * hdd)[:, np.newaxis] + f * (hd * hdd)[:, np.newaxis] + f * (h * hddd)[:, np.newaxis]

                c1 = (b1 - b2 + b3 - b4) / (h**3)[:, np.newaxis]
                c2 = 3 * Add * hd[:, np.newaxis] / h[:, np.newaxis]

                Addd = c1 - c2
                ydddot = np.dot(Addd, RBF["w"])

                return y, ydot, yddot, ydddot


def decodeQuaternionRBF(x: ArrayLike, RBF: Dict[str, np.ndarray]) -> QuaternionsType:
    """
    Generate quaternion path at points `x` encoded by Gaussian Radial Basis Functions (RBF) with Gaussian Kernel Functions (GKF).

    The Gaussian Kernel Function (GKF) is defined as:
    GKF(x) = exp(-((x - c)^2 / (2 * sigma2)))

    The Radial Basis Function (RBF) is a weighted sum of these Gaussian kernel functions:
    RBF(x) = sum(w * GKF(x)) / sum(GKF(x))

    This function decodes the quaternion path `y(x)` encoded by the RBF, ensuring the resulting path is normalized to unit quaternions.

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)

    Parameters
    ----------
    x : ArrayLike
        Path parameter (n,). A 1D array representing the path or domain where the quaternion path is evaluated.
    RBF : Dict[str, np.ndarray]
        A dictionary containing the RBF parameters

    Returns
    -------
    path : QuaternionsType
        The computed quaternion path values (n, 4). The path is represented as unit quaternions at the points `x`.

    Raises
    ------
    ValueError
        If the input parameters are invalid or mismatched in dimensions.

    Notes
    -----
    - The function normalizes the resulting quaternion path to ensure that each quaternion has unit magnitude.
    - The quaternion is represented as a 4-dimensional vector, and the function ensures that each decoded quaternion lies on the unit sphere in 4D space.
    """
    if not isvector(x):
        raise ValueError("Parameter x is not a vector")
    if RBF["w"].shape[1] != 4:
        raise ValueError("RBF is not encoding quaternion path")

    q = decodeRBF(x, RBF, calc_derivative=0)
    qn = np.sqrt(np.sum(q**2, axis=1))
    q = q / qn[:, np.newaxis]
    return q


def decodeCartesianRBF(x: ArrayLike, RBF: Dict[str, np.ndarray]) -> Poses3DType:
    """
    Generate Cartesian path at points `x` encoded by Gaussian Radial Basis Functions (RBF) with Gaussian Kernel Functions (GKF).

    The Gaussian Kernel Function (GKF) is defined as:
    GKF(x) = exp(-((x - c)^2 / (2 * sigma2)))

    The Radial Basis Function (RBF) is a weighted sum of these Gaussian kernel functions:
    RBF(x) = sum(w * GKF(x)) / sum(GKF(x))

    This function decodes the Cartesian path `y(x)` encoded by the RBF. It assumes that the path includes 3D spatial data
    along with quaternion orientations in 7-dimensional space (x, y, z, qx, qy, qz, qw).

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)

    Parameters
    ----------
    x : ArrayLike
        Path parameter (n,). A 1D array representing the path or domain where the Cartesian path is evaluated.
    RBF : Dict[str, np.ndarray]
        A dictionary containing the RBF parameters

    Returns
    -------
    path : Poses3DType
        The computed Cartesian path values (n, m). The path includes 3D position and quaternion orientation at each point `x`.

    Raises
    ------
    ValueError
        If the input parameters are invalid or mismatched in dimensions (e.g., RBF weights not encoding a Cartesian path).

    Notes
    -----
    - The function expects the RBF weights `w` to encode both the 3D positions and quaternion orientations, so `w` should have 7 columns.
    - The quaternion is normalized to ensure it has unit magnitude (i.e., it lies on the unit sphere in 4D space).
    """
    if not isvector(x):
        raise ValueError("Parameter x is not a vector")
    if RBF["w"].shape[1] != 7:
        raise ValueError("RBF is not encoding Cartesian path")

    x = np.asarray(x, dtype="float")
    y = decodeRBF(x, RBF, calc_derivative=0)

    # Normalize the quaternions
    q = y[:, 3:]
    qn = np.sqrt(np.sum(q**2, axis=1))
    q = q / qn[:, np.newaxis]
    y[:, 3:] = q

    return y


def jacobiRBF(x: ArrayLike, RBF: Dict[str, np.ndarray], deps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Jacobian and its derivative for RBF encoded path at points `x` using numeric differentiation.

    The Jacobian is calculated as the numerical derivative of the path with respect to the path parameter `x`,
    and the derivative of the Jacobian is also computed using finite differences.

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)

    Parameters
    ----------
    x : ArrayLike
        Path parameter (n,). A 1D array representing the points at which the Jacobian is computed.
    RBF : Dict[str, np.ndarray]
        A dictionary containing the RBF parameters.
    deps : float, optional
        The step size used for numerical differentiation. Default is `1e-5`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        J (n, m) and Jd (n, m) for the path and its Jacobian derivative.

    Raises
    ------
    ValueError
        If the input parameters are invalid or mismatched in dimensions.

    Notes
    -----
    - The function computes the Jacobian `J` and its derivative `Jd` by evaluating the RBF-encoded path at three points:
      - At `x`, at `x + deps`, and at `x - deps`.
    - The result is a numerical differentiation approach using the finite difference method.
    """
    y0 = decodeRBF(x, RBF)
    y1 = decodeRBF(x + deps, RBF)
    J = (y1 - y0) / deps

    y2 = decodeRBF(x - deps, RBF)
    J1 = (y0 - y2) / deps
    Jdot = (J - J1) / deps

    return J, Jdot


def updateRBF(x: float, yn: ArrayLike, RBF: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Update RBF weights using recursive regression.

    This function updates the weights of the Radial Basis Function (RBF) network using a recursive regression
    method. The update is performed by adjusting the weights based on the difference between the measured signal ``yn``
    and the signal predicted by the RBF model at the given path parameter ``x``. The update is done using the recursive
    least squares approach.

    A dictionary containing the RBF parameters is defined as follows:

        - RBF["N"] : Number of Gaussian kernel functions (N)
        - RBF["w"] : Weights of the GKF (N, m)
        - RBF["c"] : Centers of the GKF (N,)
        - RBF["sigma2"] : Standard deviation of the GKF (N,)
        - RBF["p"] : Recursive regression matrix (N, N)
        - RBF["lambda"] : Regularization parameter for recursive regression.

    Parameters
    ----------
    x : float
        The path parameter at which the RBF model is evaluated. It is a scalar value.
    yn : ArrayLike
        Measured signals (m,). A 1D array representing the observed data for which the RBF model is being updated.
    RBF : Dict[str, np.ndarray]
        A dictionary containing the RBF parameters

    Returns
    -------
    y : np.ndarray
        The calculated signal at ``x`` with shape ``(m,)``. This is the
        predicted signal after updating the weights.
    RBF : Dict[str, np.ndarray]
        Updated RBF parameters, including the updated weights and the regression matrix.

    Raises
    ------
    ValueError
        If the input parameters are invalid or mismatched in dimensions. For example, if `x` is not a scalar or if `yn` is not a vector.

    Notes
    -----
    - The recursive regression method adjusts the weights based on the error
      between the predicted and measured values.
    - The ``lambda`` parameter serves as a regularization term to prevent
      overfitting during the weight update process.
    - This method is often used in online learning scenarios where the model is updated iteratively with new data points.
    """
    if not isscalar(x):
        raise ValueError("Parameter x must be scalar")
    if not isvector(yn):
        raise ValueError("Parameter yn must be vector")

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    yn = np.asarray(yn, dtype=float).reshape(1, -1)

    if yn.shape[1] != RBF["w"].shape[1]:
        raise ValueError(f"Measured values size must be {RBF['w'].shape[1]}")

    tmp1 = x - RBF["c"]
    tmp2 = -0.5 * tmp1**2
    tmp3 = tmp2 / RBF["sigma2"]
    tmp4 = np.exp(tmp3)
    tmp5 = np.sum(tmp4, axis=1)
    A = tmp4 / (tmp5 + np.finfo(float).eps)[:, None]

    y = A @ RBF["w"]

    # Recursive regression
    p = RBF["p"]
    ATA = A.T @ A
    den = float(RBF["lambda"] + A @ p @ A.T)
    p = (1.0 / RBF["lambda"]) * (p - (p @ ATA @ p) / den)

    er = yn - y
    RBF["w"] = RBF["w"] + (p @ A.T) @ er
    RBF["p"] = p

    y = A @ RBF["w"]
    return y, RBF
