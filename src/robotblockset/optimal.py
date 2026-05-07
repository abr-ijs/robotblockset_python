"""Optimal Trajectory Generation Module.

This module provides utilities for the generation of time-optimal trajectories with bounded position,
velocity, and acceleration. It includes functions for calculating and optimizing trajectories
using constraints for both Cartesian and joint space. The module leverages numerical optimization
and interpolation to create smooth paths that satisfy given constraints, ensuring that the motion
is as fast as possible without violating the constraints.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fminbound, fmin
import matplotlib.pyplot as plt
from typing import Callable, Optional, Sequence, Tuple, Union

from robotblockset.rbs_typing import ArrayLike
from robotblockset.trajectories import pathlen, interpPath
from robotblockset.transformations import qmtimes, qtranspose, xerrnorm
from robotblockset.tools import isscalar, vector


class path_constraints:
    """
    A class to define constraints on the path motion, including:
    - Maximum velocity and acceleration in Cartesian and joint spaces.

    Attributes
    ----------
    xdnmax : float, optional
        Maximum path velocity in Cartesian space.
    xddnmax : float, optional
        Maximum path acceleration in Cartesian space.
    xdmax : np.ndarray, optional
        Maximum velocity in Cartesian space.
    xddmax : np.ndarray, optional
        Maximum acceleration in Cartesian space.
    qdmax : np.ndarray, optional
        Maximum joint velocity.
    qddmax : np.ndarray, optional
        Maximum joint acceleration.
    """

    def __init__(self) -> None:
        """
        Initialize an empty set of path velocity and acceleration constraints.

        Returns
        -------
        None
            This constructor initializes the constraint container in place.
        """
        self.xdnmax: Optional[float] = None
        self.xddnmax: Optional[float] = None
        self.xdmax: Optional[np.ndarray] = None
        self.xddmax: Optional[np.ndarray] = None
        self.qdmax: Optional[np.ndarray] = None
        self.qddmax: Optional[np.ndarray] = None


def splinedif(s: float, path_s: ArrayLike, path: ArrayLike, ds: float = 0.001, Cartesian: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerically calculates path Jacobian and its derivative.

    Parameters
    ----------
    s : float
        Path parameter.
    path_s : ArrayLike
        Path parameters for path (n, ).
    path : ArrayLike
        Path task positions (n, m).
    ds : float, optional
        Path step, by default 0.001.
    Cartesian : bool, optional
        Whether the path is Cartesian task poses, by default False.

    Returns
    -------
    tuple
        - x (m, ): Task position.
        - sJ (m, ): Jacobian dx/ds.
        - sJd (m, ): Second derivative d2(x)/ds2.
    """

    _s = np.asarray(path_s, dtype="float")
    _x = np.asarray(path, dtype="float")
    m = _x.shape[1]

    si = np.array([s, s + ds / 2, s + ds])
    cs = CubicSpline(_s, _x, axis=0)
    xi = cs(si)
    x = xi[0]

    sJ = np.diff(xi, axis=0) / ds * 2
    if m == 7 and Cartesian:
        omega_q = 2 * qmtimes(sJ[:, 3:], qtranspose(xi[:2, 3:]))
        sJ = np.hstack((sJ[:, :3], omega_q[:, 1:]))

    sJd = np.diff(sJ, axis=0) / ds * 2
    sJ = sJ[0]
    sJd = sJd[0]

    return x, sJ, sJd


class path_kinematics:
    """
    A class that handles the kinematics of a path, including calculating task positions,
    velocities, and accelerations, along with their corresponding Jacobians.
    """

    def __init__(self, path: ArrayLike, path_s: Optional[ArrayLike] = None, path_q: Optional[ArrayLike] = None, Cartesian: bool = False, dkin: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None, ds: float = 0.001, scale: Union[float, Sequence[float]] = (1.0, 1.0)) -> None:
        """
        Initialize the path kinematics model used for trajectory optimization.

        Parameters
        ----------
        path : ArrayLike
            Path samples in task space or joint space.
        path_s : ArrayLike, optional
            Path parameter values associated with the samples. If omitted, they are computed from the path.
        path_q : ArrayLike, optional
            Joint-space samples associated with a Cartesian path.
        Cartesian : bool, optional
            If True, interpret the path as Cartesian poses.
        dkin : callable, optional
            Direct kinematics function used to map joint states to task space and Jacobians.
        ds : float, optional
            Step size used for numerical differentiation along the path.
        scale : float or sequence of float, optional
            Scale factors used in SE(3) path-length and error computations.

        Returns
        -------
        None
            This constructor initializes the path kinematics object in place.
        """
        m = path.shape[1]
        if path_s is None:
            if m == 7 and Cartesian:
                _s = pathlen(path, Cartesian=True, scale=scale)
            else:
                _s = pathlen(path, Cartesian=False)
        else:
            if path_s.shape[0] != path.shape[0]:
                raise ValueError(f"Path is not consistent (path_s: {path_s.shape} - path: {path.shape})")
            _s = np.asarray(path_s, dtype="float")

        if path_q is not None and path.shape[0] != path_q.shape[0]:
            raise ValueError(f"Joint pathis  not consistent (path_s: {path_s.shape} - path_q: {path_q.shape})")
        self.path_s = _s
        self.path = path
        self.path_q = path_q
        self.Cartesian = Cartesian
        self.dkin = dkin
        self.ds = ds
        if isscalar(scale):
            self.scale = [1.0, scale]
        else:
            self.scale = scale
        self.x = None
        self.sJ = None
        self.sJd = None
        self.J = None
        self.Jd = None

    def calc(self, s: float) -> None:
        """
        Calculates task position, velocity, and acceleration at a given path parameter `s`.

        Parameters
        ----------
        s : float
            Path parameter at which to compute the task position, velocity, and acceleration.
        """
        self.x, self.sJ, self.sJd = splinedif(s, self.path_s, self.path, Cartesian=self.Cartesian, ds=self.ds)
        if self.dkin is None:
            pass
        elif self.Cartesian:  # path is in Cartesian space and path_q is in joint space
            if self.path_q is not None:
                qq = splinedif(s, self.path_s, self.path_q, ds=self.ds, Cartesian=False)[0]
                _, self.J = self.dkin(qq)
                qq1 = splinedif(s + 0.001, self.path_s, self.path_q, ds=self.ds, Cartesian=False)[0]
                _, J1 = self.dkin(qq1)
                self.Jd = (J1 - self.J) / 0.001
        else:  # path is in joint space
            _, self.J = self.dkin(self.x)
            qq1 = splinedif(s + 0.001, self.path_s, self.path, ds=self.ds, Cartesian=False)[0]
            _, J1 = self.dkin(qq1)
            self.Jd = (J1 - self.J) / 0.001

    def s2x(self, sp: ArrayLike, sv: Optional[ArrayLike] = None, sa: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms path trajectory to task trajectory.

        Parameters
        ----------
        sp : ArrayLike
            Path positions (nsamp, ).
        sv : ArrayLike, optional
            Path velocities (nsamp, ), by default None.
        sa : ArrayLike, optional
            Path accelerations (nsamp, ), by default None.

        Returns
        -------
        tuple
            - x (nsamp, m): Task positions.
            - xd (nsamp, m): Task velocities.
            - xdd (nsamp, m): Task accelerations.
        """
        nsamp = len(sp)
        m = self.path.shape[1]

        x = np.zeros((nsamp, m))
        if m == 7 and self.Cartesian:
            xd = np.zeros((nsamp, 6))
            xdd = np.zeros((nsamp, 6))
        else:
            xd = np.zeros((nsamp, m))
            xdd = np.zeros((nsamp, m))

        for i in range(nsamp):
            self.calc(sp[i])
            x[i, :] = self.x
            if sv is not None:
                xd[i, :] = self.sJ * sv[i]
                if sa is not None:
                    xdd[i, :] = self.sJ * sa[i] + self.sJd * sv[i] ** 2

        return x, xd, xdd

    def s2q_x(self, sp: ArrayLike, sv: Optional[ArrayLike] = None, sa: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms path trajectory via Cartesian task space into joint space.

        Parameters
        ----------
        sp : ArrayLike
            Path positions (nsamp, ).
        sv : ArrayLike, optional
            Path velocities (nsamp, ), by default None.
        sa : ArrayLike, optional
            Path accelerations (nsamp, ), by default None.

        Returns
        -------
        tuple
            - q (nsamp, nj): Joint positions.
            - qd (nsamp, nj): Joint velocities.
            - qdd (nsamp, nj): Joint accelerations.
        """
        nsamp = len(sp)
        m = self.path.shape[1]
        if m != 7 or not self.Cartesian or self.path_q is None:
            raise ValueError("Path is not correctly defined")

        nj = self.path_q.shape[1]
        q = np.zeros((nsamp, nj))
        qd = np.zeros((nsamp, nj))
        qdd = np.zeros((nsamp, nj))

        for i in range(nsamp):
            q[i, :] = splinedif(sp[i], self.path_s, self.path_q, ds=self.ds, Cartesian=False)[0]
            if sv is not None:
                self.calc(sp[i])
                Ji = np.linalg.pinv(self.J)
                _xd = self.sJ * sv[i]
                _qd = Ji @ _xd
                qd[i, :] = _qd
                if sa is not None:
                    Jd = self.Jd * sv[i]
                    _xdd = self.sJ * sa[i] + self.sJd * sv[i] ** 2
                    qdd[i, :] = Ji @ (_xdd - Jd @ _qd)

        return q, qd, qdd


def accbounds(sd: float, path_kin: path_kinematics, path_con: path_constraints, calc_option: int = 0b111) -> np.ndarray:
    """
    Calculates the path acceleration bounds in (s, sd) space.

    Parameters
    ----------
    sd : float
        Path velocity.
    path_kin : path_kinematics
        Path kinematics parameters.
    path_con : path_constraints
        Velocity and acceleration constraints in Cartesian and joint space.
    calc_option : int, optional
        Bit selection for considered bounds (default is 0b111), where:
        - 0: maximal Cartesian acceleration norm
        - 1: maximal Cartesian acceleration
        - 2: maximal joint acceleration.

    Returns
    -------
    np.ndarray
        [sddmax, sddmin]: Maximum and minimum path acceleration bounds.
    """
    if path_kin.sJ is None or path_kin.sJd is None:
        return np.inf, -np.inf
    else:
        sJ = path_kin.sJ
        sJd = path_kin.sJd
        J = path_kin.J
        Jd = path_kin.Jd
    if J is None:
        calc_option &= ~0b11

    # Task acc norm
    sdd_xddnmax = np.inf
    sdd_xddnmin = -np.inf
    if path_con.xddnmax is not None and calc_option & (1 << 0):
        if path_kin.Cartesian:
            fxddn = lambda sdd: xerrnorm(sJ * sdd + sJd * sd**2, path_kin.scale)
            fxddn1 = lambda sdd: np.abs(xerrnorm(sJ * sdd + sJd * sd**2, path_kin.scale) - path_con.xddnmax)
        else:
            fxddn = lambda sdd: xerrnorm(J @ sJ * sdd + (J @ sJd + Jd @ sJ) * sd**2, path_kin.scale)
            fxddn1 = lambda sdd: np.abs(xerrnorm(J @ sJ * sdd + (J @ sJd + Jd @ sJ) * sd**2, path_kin.scale) - path_con.xddnmax)

        sdd_xddn0 = fmin(fxddn, 0, disp=0)[0]
        sdd_xddn1 = fminbound(fxddn1, sdd_xddn0 - 10, sdd_xddn0 + 20)
        sdd_xddn2 = fminbound(fxddn1, sdd_xddn0 - 20, sdd_xddn0 + 10)
        sdd_xddnmax = max(sdd_xddn1, sdd_xddn2)
        sdd_xddnmin = min(sdd_xddn1, sdd_xddn2)

    # Task acc
    sdd_xddmax = np.inf
    sdd_xddmin = -np.inf
    if path_con.xddmax is not None and calc_option & (1 << 1):
        xddmax = path_con.xddmax
        xddmin = -xddmax
        if path_kin.Cartesian:
            a1 = sJ
            a0 = sJd * sd**2
        else:
            a1 = J @ sJ
            a0 = (J @ sJd + Jd @ sJ) * sd**2

        m = len(xddmax)
        sdd1 = np.zeros(m)  # max
        sdd2 = np.zeros(m)  # min

        for i in range(m):
            if a1[i] > 0.001:
                sdd1[i] = (xddmax[i] - a0[i]) / a1[i]
                sdd2[i] = (xddmin[i] - a0[i]) / a1[i]
            elif a1[i] < -0.001:
                sdd1[i] = (xddmin[i] - a0[i]) / a1[i]
                sdd2[i] = (xddmax[i] - a0[i]) / a1[i]
            else:
                sdd1[i] = np.inf
                sdd2[i] = -np.inf

        sdd_xddmax = np.min(sdd1)
        sdd_xddmin = np.max(sdd2)

    # Joint acc
    sdd_qddmax = np.inf
    sdd_qddmin = -np.inf
    if not (path_con.qddmax is None or (path_kin.Cartesian and (J is None or Jd is None))) and calc_option & (1 << 2):
        qddmax = path_con.qddmax
        qddmin = -qddmax
        if path_kin.Cartesian:
            Ji = np.linalg.pinv(J)
            b1 = Ji @ sJ
            b0 = Ji @ (sJd - Jd @ Ji @ sJ) * sd**2
        else:
            b1 = sJ
            b0 = sJd * sd**2

        m = len(qddmax)
        sdd1 = np.zeros(m)  # max
        sdd2 = np.zeros(m)  # min

        for i in range(m):
            if b1[i] > 0.001:
                sdd1[i] = (qddmax[i] - b0[i]) / b1[i]
                sdd2[i] = (qddmin[i] - b0[i]) / b1[i]
            elif b1[i] < -0.001:
                sdd1[i] = (qddmin[i] - b0[i]) / b1[i]
                sdd2[i] = (qddmax[i] - b0[i]) / b1[i]
            else:
                sdd1[i] = np.inf
                sdd2[i] = -np.inf

        sdd_qddmax = np.min(sdd1)
        sdd_qddmin = np.max(sdd2)

    sdd_max = min(sdd_xddmax, sdd_qddmax, sdd_xddnmax)
    sdd_min = max(sdd_xddmin, sdd_qddmin, sdd_xddnmin)

    return np.array([sdd_max, sdd_min])


def velbounds(path_kin: path_kinematics, path_con: path_constraints) -> np.ndarray:
    """
    Calculates path velocity bounds as path position `s` due to Cartesian and joint constraints.

    Parameters
    ----------
    path_kin : path_kinematics
        Path kinematics parameters.
    path_con : path_constraints
        Velocity and acceleration constraints in Cartesian and joint space.

    Returns
    -------
    np.ndarray
        [sd_xdn, sd_xd, sd_xdd, sd_qd, sd_qdd]: Path velocity bounds due to:
        - maximal Cartesian velocity norm
        - maximal Cartesian velocity
        - maximal Cartesian acceleration
        - maximal joint velocity
        - maximal joint acceleration
    """

    if path_kin.sJ is None:
        return [np.inf, np.inf, np.inf, np.inf, np.inf]
    else:
        sJ = path_kin.sJ
        sJd = path_kin.sJd
        J = path_kin.J
        Jd = path_kin.Jd
        if path_kin.Cartesian:
            Ji = np.linalg.pinv(J)
            JisJ = Ji @ sJ
        elif J is not None:
            JsJ = J @ sJ

    # sJn = np.linalg.norm(sJ)

    # Bounds on sd due to nominal path velocity
    if path_con.xdnmax is None or np.isinf(path_con.xdnmax) or J is None:
        sd_xdn = np.inf
    else:
        if path_kin.Cartesian:
            fxdn = lambda sd: np.abs(xerrnorm(sJ * sd, path_kin.scale) - path_con.xdnmax)
        else:
            fxdn = lambda sd: np.abs(xerrnorm(JsJ * sd, path_kin.scale) - path_con.xdnmax)
        sd_xdn = fmin(fxdn, 0, disp=0)[0]

    # Bounds on sd due to xdmax
    if path_con.xdmax is None or J is None:
        sd_xd = np.inf
    else:
        if path_kin.Cartesian:
            sd_xd = np.min(np.abs(path_con.xdmax / sJ))
        else:
            sd_xd = np.min(np.abs(path_con.xdmax / JsJ))

    # Bounds on sd due to xddmax
    if path_con.xddmax is None or sJd is None or J is None:
        sd_xdd = np.inf
    else:
        if np.any(np.abs(sJd) > 1e-8):
            fxdd = lambda sd: abs(np.diff(accbounds(sd, path_kin, path_con, calc_option=0b010))) - 0.00001 * sd
            sd_xdd = fmin(fxdd, 0, disp=0)[0]
            if path_kin.Cartesian:
                if np.any(np.abs(sJ) < 0.0001):
                    tmp = np.full(len(sJ), np.inf)
                    for i in range(len(sJ)):
                        if abs(sJ[i]) < 0.0001 and abs(sJd[i] > 0.0001):
                            tmp[i] = np.sqrt(abs(path_con.xddmax[i] / sJd[i]))
                    tmp_min = np.min(tmp)
                    sd_xdd = min(sd_xdd, tmp_min)
            else:
                if np.any(np.abs(JsJ) < 0.0001):
                    tmp = np.full(len(JsJ), np.inf)
                    bx = J @ sJd + Jd @ sJ
                    for i in range(len(JsJ)):
                        if abs(JsJ[i]) < 0.0001 and abs(bx[i] > 0.0001):
                            tmp[i] = np.sqrt(abs(path_con.xddmax[i] / bx[i]))
                    tmp_min = np.min(tmp)
                    sd_xdd = min(sd_xdd, tmp_min)
        else:
            sd_xdd = np.inf

        if sd_xdd < 0:
            raise ValueError(f"Not feasible velocity bound due to task acc: {sd_xdd} ")

    # Bounds on sd due to qdmax
    if path_con.qdmax is None or (path_kin.Cartesian and J is None):
        sd_qd = np.inf
    else:
        if path_kin.Cartesian:
            sd_qd = np.min(np.abs(path_con.qdmax / JisJ))
        else:
            sd_qd = np.min(np.abs(path_con.qdmax / sJ))

    # Bounds on sd due to qddmax
    if path_con.qddmax is None or sJd is None or (path_kin.Cartesian and (J is None or Jd is None)):
        sd_qdd = np.inf
    else:
        if np.any(np.abs(sJd) > 1e-8):
            fqdd = lambda sd: abs(np.diff(accbounds(sd, path_kin, path_con, calc_option=0b100))) - 0.00001 * sd
            sd_qdd = fmin(fqdd, 0, disp=0)[0]
            if path_kin.Cartesian:
                if np.any(np.abs(JisJ) < 0.0001):
                    tmp = np.full(len(JisJ), np.inf)
                    bx = Ji @ (sJd - Jd @ Ji @ sJ)
                    for i in range(len(JisJ)):
                        if abs(JisJ[i]) < 0.0001 and abs(bx[i] > 0.0001):
                            tmp[i] = np.sqrt(abs(path_con.qddmax[i] / bx[i]))
                    tmp_min = np.min(tmp)
                    sd_qdd = min(sd_qdd, tmp_min)
            else:
                if np.any(np.abs(sJ) < 0.0001):
                    tmp = np.full(len(sJ), np.inf)
                    for i in range(len(sJ)):
                        if abs(sJ[i]) < 0.0001 and abs(sJd[i] > 0.0001):
                            tmp[i] = np.sqrt(abs(path_con.qddmax[i] / sJd[i]))
                    tmp_min = np.min(tmp)
                    sd_qdd = min(sd_qdd, tmp_min)
        else:
            sd_qdd = np.inf

        if sd_qdd < 0:
            raise ValueError(f"Not feasible task velocity bound - Bounds due to acc: {sd_xdd} ")

    # Bounds on sd
    sd_b = np.array([sd_xdn, sd_xd, sd_xdd, sd_qd, sd_qdd])
    return sd_b


def lineIntersection(L1x1: float, L1y1: float, L1x2: float, L1y2: float, L2x1: float, L2y1: float, L2x2: float, L2y2: float) -> Tuple[float, float]:
    """
    Calculates the intersection point of two 2D lines defined by their endpoints.

    Parameters
    ----------
    L1x1, L1y1 : float
        Coordinates of the first endpoint of the first line.
    L1x2, L1y2 : float
        Coordinates of the second endpoint of the first line.
    L2x1, L2y1 : float
        Coordinates of the first endpoint of the second line.
    L2x2, L2y2 : float
        Coordinates of the second endpoint of the second line.

    Returns
    -------
    tuple
        - x (float): x-coordinate of the intersection point.
        - y (float): y-coordinate of the intersection point.

    If the lines are parallel (i.e., no intersection), the function returns NaN for both coordinates.
    """
    denom = (L1x1 - L1x2) * (L2y1 - L2y2) - (L1y1 - L1y2) * (L2x1 - L2x2)
    if denom == 0:
        return np.nan, np.nan

    x = ((L1x1 * L1y2 - L1y1 * L1x2) * (L2x1 - L2x2) - (L1x1 - L1x2) * (L2x1 * L2y2 - L2y1 * L2x2)) / denom
    y = ((L1x1 * L1y2 - L1y1 * L1x2) * (L2y1 - L2y2) - (L1y1 - L1y2) * (L2x1 * L2y2 - L2y1 * L2x2)) / denom
    return x, y


def timeopttraj(path_kin: path_kinematics, path_con: path_constraints, s0: float = 0, send: Optional[float] = None, sd0: float = 0, sdend: float = 0, tsamp: float = 0.01, plot: bool = False, sd_bounds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a time-optimal trajectory for bounded position, acceleration, and velocity.

    Parameters
    ----------
    path_kin : path_kinematics
        Path kinematics parameters.
    path_con : path_constraints
        Velocity and acceleration constraints in Cartesian and joint space.
    s0 : float, optional
        Path start position, by default 0.
    send : float, optional
        Path end position, by default None.
    sd0 : float, optional
        Path start velocity, by default 0.
    sdend : float, optional
        Path end velocity, by default 0.
    tsamp : float, optional
        Sample time, by default 0.01.
    plot : bool, optional
        Flag for plotting, by default False.
    sd_bounds : np.ndarray, optional
        Path velocity bounds for plot, by default None.

    Returns
    -------
    tuple
        - t (time): Time array.
        - sp (path parameter): Path parameter array.
        - sv (path velocity): Path velocity array.
        - sa (path acceleration): Path acceleration array.
    """

    path_s = vector(path_kin.path_s)
    if s0 is None:
        s0 = path_s[0]
    if send is None:
        send = path_s[-1]

    # Plot (s,sd) plane
    if plot:
        if sd_bounds is None:
            n = len(path_kin.path_s)
            sd_bounds = np.zeros((n, 5))
            for i in range(n):
                if path_kin.path_s[i] >= s0 and path_kin.path_s[i] <= send:
                    path_kin.calc(path_kin.path_s[i])
                    sdb2 = velbounds(path_kin, path_con)
                    sd_bounds[i, :] = sdb2

        sdx = np.max(np.min(sd_bounds, axis=1))

        fig_gt = plt.figure("Generate path trajectory")
        ax1 = fig_gt.add_subplot(1, 1, 1)
        ax1.plot(path_kin.path_s, sd_bounds, "-")
        ln_f = ax1.plot(0, 0, "-", color="#00FF00")
        ln_b = ax1.plot(0, 0, "b-")
        mr_f = ax1.plot(0, 0, "kx")
        mr_f0 = ax1.plot(0, 0, "k.")
        plt.grid(True)
        plt.xlabel("$s$", fontsize=14, fontweight="bold", style="italic")
        plt.ylabel("$\\dot s$", fontsize=14, fontweight="bold", style="italic")
        plt.xlim([s0, send])
        plt.ylim([0, 1.2 * sdx])
        plt.show(block=False)

    sp = np.array([])
    sv = np.array([])
    sa = np.array([])
    ts = np.array([])

    s = s0
    sd = sd0
    tsamp_back = tsamp / 4
    tsamp_search = tsamp / 10

    while s < send:
        # Forward integration
        path_kin.calc(s0)
        sdd_b = accbounds(sd0, path_kin, path_con)
        sdd = sdd_b[0]
        s = s0 + sd0 * tsamp
        if s >= send:
            break
        sd = sd0 + sdd * tsamp
        # Check if sd in bounds and slow down if necessary
        path_kin.calc(s)
        sd_b = velbounds(path_kin, path_con)
        idx = np.argmin(sd_b)
        sd_m = sd_b[idx]
        if sd_m <= sd:  # slow down if necessary (sd decreased to be on boundary)
            sd = sd_m
            s = s0 + sd * tsamp
            sdd = (sd - sd0) / tsamp
        if sd < 0:  # if stopped then no solution exists
            print("Ttajectory time optimization: No solution found")
            t = np.nan
            return None, None, None, None

        # Check if state is feasible
        if (sdd <= sdd_b[0]) and (sdd >= sdd_b[1]):
            # save forward path
            sp = np.append(sp, s0)
            sv = np.append(sv, sd0)
            sa = np.append(sa, sdd)
            ts = np.append(ts, tsamp)
            # make current state last valid point on forward point (s0, sd0)
            s0 = s
            sd0 = sd
            if plot:
                ln_f[0].remove()
                ln_f = ax1.plot(sp, sv, "-", color="#00FF00")
                mr_f[0].remove()
                mr_f = ax1.plot(s, sd, "kx")
                mr_f0[0].remove()
                mr_f0 = ax1.plot(s0, sd0, "k.")
                plt.pause(0.01)

        else:
            cross_forward = False
            spb = np.array([send])
            loop_count = 0
            # Find next breakpoint for backward integration
            while not cross_forward:
                s1 = s
                sd1 = sd
                sp_s = np.array([sp[-1], s0, s])
                sv_s = np.array([sv[-1], sd0, sd])
                path_kin.calc(s)
                sdd_b = accbounds(sd, path_kin, path_con)
                while (sdd > sdd_b[1]) or (sdd < sdd_b[0]):  # Not admisable path
                    s1 = s
                    if spb[-1] == s1:  # same initial state as in previous loop execution
                        loop_count += 1
                    else:
                        loop_count = 0
                    sd1 = sd
                    s = s1 + sd1 * tsamp_search
                    if s >= send:
                        break

                    path_kin.calc(s)
                    sd_b = velbounds(path_kin, path_con)
                    idx = np.argmin(sd_b)
                    sd = sd_b[idx]
                    if sd1 < sd / 100:
                        print("Trajectory time optimization: No solution found - Path_velociy almost 0")
                        t = np.nan
                        return None, None, None, None

                    if plot:
                        mr_f[0].remove()
                        mr_f = ax1.plot(s, sd, "kx")
                        plt.pause(0.01)

                    sdd = (sd - sd1) / tsamp_search
                    sdd_b = accbounds(sd, path_kin, path_con)
                    sp_s = np.append(sp_s, s)
                    sv_s = np.append(sv_s, sd)
                    if idx in [2, 4]:
                        s_f = s + sd1 * tsamp_search
                        path_kin.calc(s_f)
                        sd_b_f = velbounds(path_kin, path_con)
                        idx = np.argmin(sd_b_f)
                        sd_f = sd_b_f[idx]
                        if sd_f > sd:
                            sdd_f = (sd_f - sd) / tsamp_search
                            if sdd > sdd_b[1] or sdd_f < sdd_b[0]:
                                while sdd > sdd_b[1] or sdd_f < sdd_b[0]:
                                    sfac = 1 - (loop_count + 1) / 100
                                    sd = sd * sfac
                                    sp_s = np.append(sp_s, s)
                                    sv_s = np.append(sv_s, sd)
                                    sdd = (sd - sd1) / tsamp_search
                                    sdd_b = accbounds(sd, path_kin, path_con)
                                    sdd_f = (sd_f - sd) / tsamp_search
                            elif loop_count > 0:
                                sfac = 1 - (loop_count) / 100
                                sd = sd * sfac
                                sp_s = np.append(sp_s, s)
                                sv_s = np.append(sv_s, sd)
                            s = s1
                            break
                    elif idx in [1, 3] and (sdd > sdd_b[1]) and (sdd < sdd_b[0]):  # reached feasible sd boundary
                        sfac = 1 - (loop_count + 1) / 100
                        sd = sd * sfac
                        sp_s = np.append(sp_s, s)
                        sv_s = np.append(sv_s, sd)
                        s = s1
                        break
                if s1 >= send:  # end of path reached
                    break

                if sp[-2] == sp[-1]:
                    sp = sp[:-2]
                    sv = sv[:-2]
                    sa = sa[:-2]
                    ts = ts[:-2]

                # (s1,sd1) is breakpoint
                s1 = s
                sd1 = sd
                spb = np.array([])
                svb = np.array([])
                sab = np.array([])
                tsb = np.array([])

                go_back = True
                # Backward integration
                while go_back:
                    # save backward path
                    spb = np.insert(spb, 0, s)
                    svb = np.insert(svb, 0, sd)
                    sab = np.insert(sab, 0, sdd)
                    tsb = np.insert(tsb, 0, tsamp_back)

                    if plot:
                        ln_b[0].remove()
                        ln_b = ax1.plot(spb, svb, "b")
                        plt.pause(0.01)

                    path_kin.calc(s)
                    sdd_b = accbounds(sd, path_kin, path_con)
                    sdd = sdd_b[1]
                    sd = sd - sdd * tsamp_back
                    s = s - sd * tsamp_back
                    if sd < 0:  # if stopped then no solution exists
                        print("Trajectory time optimization: No solution found")
                        t = np.nan
                        return None, None, None, None

                    # check if above forward  or search path
                    k_s = np.where(np.array(sp_s) < s)[0]
                    if k_s.size > 0 and k_s[0] >= 0:
                        # check if above search path
                        path_kin.calc(s)
                        sd_b = velbounds(path_kin, path_con)
                        if sd >= np.min(sd_b):
                            go_back = False
                            s = spb[-1]
                            sd = svb[-1]
                            sdd = np.inf  # this point is invalid
                    else:
                        # check if beyond forward
                        s_i = np.where(np.array(sp) >= s)[0]
                        if s_i.size > 0:  # left of last point on forward path
                            s_i = s_i[0]
                            # Find intersection between forward and backward path
                            s3, sd3 = lineIntersection(sp[s_i - 1], sv[s_i - 1], sp[s_i], sv[s_i], spb[0], svb[0], s, sd)
                            if s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:  # forward path is intersecting backward segment
                                s_j = np.where(np.array(sp) < s3)[0][-1]
                                cross_forward = True
                            elif s3 > sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                                s_j = s_i
                                cross_forward = True
                            elif s_i < sp.shape[0] - 1:
                                s_i += 1
                                s3, sd3 = lineIntersection(sp[s_i - 1], sv[s_i - 1], sp[s_i], sv[s_i], spb[0], svb[0], s, sd)
                                if s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                                    s_j = np.where(np.array(sp) < s3)[0][-1]
                                    cross_forward = True
                                elif s3 > sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                                    s_j = s_i
                                    cross_forward = True
                            if not cross_forward:
                                s_i -= 1
                                s3, sd3 = lineIntersection(sp[s_i - 1], sv[s_i - 1], sp[s_i], sv[s_i], spb[0], svb[0], s, sd)
                                if s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                                    s_j = np.where(np.array(sp) < s3)[0][-1]
                                    cross_forward = True
                                elif s3 > sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                                    s_j = s_i
                                    cross_forward = True

                        # Check is backward path is above last point on forward path
                        s_k = np.where(np.array(spb) <= sp[-1])[0]
                        if s_k.size > 1 and np.all(svb[s_k] > sv[-1]):
                            go_back = 0
                            s = spb[-1]
                            sd = svb[-1] * 0.99
                        else:
                            go_back = not cross_forward
                        s1 = s
                        sd1 = sd

                if cross_forward:
                    # cut the forward path and append backward path
                    if s_j + 1 < sp.shape[0]:
                        sp = sp[: s_j + 1]
                        sv = sv[: s_j + 1]
                        sa = sa[: s_j + 1]
                        ts = ts[: s_j + 1]

                    path_kin.calc(s3)
                    sd_b = velbounds(path_kin, path_con)
                    sdb = min(sd_b)
                    sp[-1] = s3
                    sv[-1] = min(sd3, sdb)

                    # smooth break
                    ts[-2] = (sp[-1] - sp[-2]) / sv[-2]
                    sa[-2] = (sv[-1] - sv[-2]) / ts[-2]
                    ts[-1] = (spb[0] - sp[-1]) / sv[-1]
                    sa[-1] = (svb[0] - sv[-1]) / ts[-1]

                    # new initial point
                    s0 = spb[-1]
                    sd0 = svb[-1]

                    # add backward path
                    sp = np.concatenate((sp, spb[:-1]))
                    sv = np.concatenate((sv, svb[:-1]))
                    sa = np.concatenate((sa, sab[:-1]))
                    ts = np.concatenate((ts, tsb[:-1]))

                    s = s0
                    sd = sd0

                    if plot:
                        ln_f[0].remove()
                        ln_f = ax1.plot(sp, sv, "-", color="#00FF00")
                        ln_b[0].remove()
                        ln_b = ax1.plot(spb, svb, "b")
                        mr_f[0].remove()
                        mr_f = ax1.plot(s, sd, "kx")
                        mr_f0[0].remove()
                        mr_f0 = ax1.plot(s0, sd0, "k.")
                        plt.pause(0.01)

    # (s0,sd0) last valid end point on forward path
    # (s1,sd1) is end breakpoint
    if sp[-2] == sp[-1]:
        sp = sp[:-2]
        sv = sv[:-2]
        sa = sa[:-2]
        ts = ts[:-2]

    if sd < sdend:
        raise ValueError("Desired end-point velocity too high")

    spb = np.array([])
    svb = np.array([])
    sab = np.array([])
    tsb = np.array([])
    s1 = send
    sd1 = sdend
    s = s1
    sd = sd1

    # (s0,sd0) last valid end point on forward path
    # (s1,sd1) is end breakpoint

    # backward segment from end point
    cross_forward = False
    while not cross_forward:
        s1 = s
        sd1 = sd

        # Backward integration
        path_kin.calc(s)
        sdd_b = accbounds(sd, path_kin, path_con)
        sdd = sdd_b[1]
        # save backward path
        spb = np.insert(spb, 0, s)
        svb = np.insert(svb, 0, sd)
        sab = np.insert(sab, 0, sdd)
        tsb = np.insert(tsb, 0, tsamp)

        sd = sd1 - sdd * tsamp
        s = s1 - sd * tsamp
        if sd < 0:  # if stopped then no solution exists
            print("Trajectory time optimization: No solution found")
            t = np.nan
            return None, None, None, None

        # check if beyond forward
        s_i = np.where(np.array(sp) >= s)[0]
        if s_i.size > 0:
            s_i = s_i[0]
            # Find intersection between forward and backward path
            s3, sd3 = lineIntersection(sp[s_i - 1], sv[s_i - 1], sp[s_i], sv[s_i], spb[0], svb[0], s, sd)
            if s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                s_j = np.where(np.array(sp) < s3)[0][-1]
                cross_forward = True
            elif s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                s_j = s_i
                cross_forward = True
            elif s_i < sp.shape[0] - 2:
                s_i += 1
                s3, sd3 = lineIntersection(sp[s_i - 1], sv[s_i - 1], sp[s_i], sv[s_i], spb[0], svb[0], s, sd)
                if s3 <= sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                    s_j = np.where(np.array(sp) < s3)[0][-1]
                    cross_forward = True
                elif s3 > sp[min(s_i + 1, sp.shape[0] - 1)] and s3 >= s and s3 <= spb[0]:
                    s_j = s_i
                    cross_forward = True

    # cut wrong forward path
    if s_j + 1 < sp.shape[0]:
        sp = sp[: s_j + 1]
        sv = sv[: s_j + 1]
        sa = sa[: s_j + 1]
        ts = ts[: s_j + 1]

    path_kin.calc(s3)
    sd_b = velbounds(path_kin, path_con)
    sdb = min(sd_b)
    sp[-1] = s3
    sv[-1] = min(sd3, sdb)

    # smooth break
    ts[-2] = (sp[-1] - sp[-2]) / sv[-2]
    sa[-2] = (sv[-1] - sv[-2]) / ts[-2]
    ts[-1] = (spb[0] - sp[-1]) / sv[-1]
    sa[-1] = (svb[0] - sv[-1]) / ts[-1]

    # add backward path
    sp = np.concatenate((sp, spb))
    sv = np.concatenate((sv, svb))
    sa = np.concatenate((sa, sab))
    ts = np.concatenate((ts, tsb))

    # time
    # t = np.array([0] + list(np.cumsum(ts[:-1])))

    # time equidistant points
    tt = np.array([0] + list(np.cumsum(ts[:-1])))
    t = np.arange(0, tt[-1] + tsamp, tsamp)
    sp = interpPath(tt, sp, t)
    sv = interpPath(tt, sv, t)
    sa = interpPath(tt, sa, t)

    if plot:
        ln_f[0].remove()
        ln_f = ax1.plot(sp, sv, "-", color="#00FF00")
        ln_b[0].remove()
        mr_f[0].remove()
        mr_f0[0].remove()
        plt.pause(0.01)

    return t, sp, sv, sa


def plot_acc_bounds(s: float, sd: float, path_kin: path_kinematics, path_con: path_constraints, tsamp: float = 0.001, fig: Optional[int] = None) -> None:
    """
    Plots max and min acceleration directions to the current s-sd plot.

    Parameters
    ----------
    s : float
        Path position.
    sd : float
        Path velocity.
    path_kin : path_kinematics
        Path kinematics parameters.
    path_con : path_constraints
        Path constraints due to velocity and acceleration.
    tsamp : float, optional
        Sample time, by default 0.001.
    fig : int, optional
        Figure number, by default None.
    """

    if fig is None:
        fig = plt.gcf()

    plt.figure(fig.number)

    path_kin.calc(s)
    sdd_b = accbounds(sd, path_kin, path_con)
    ds = sd * tsamp
    dsd1 = sdd_b[0] * tsamp
    dsd2 = sdd_b[1] * tsamp
    plt.plot([s, s + ds], [sd, sd + dsd1], "b", linewidth=1)
    plt.plot([s, s + ds], [sd, sd + dsd2], "r")
    plt.show(block=False)


def plot_path_bounds(path_kin: path_kinematics, path_con: path_constraints, s0: Optional[float] = None, send: Optional[float] = None) -> np.ndarray:
    """
    Plots path velocity bounds in the s-sd plane.

    Parameters
    ----------
    path_kin : path_kinematics
        Path kinematics parameters.
    path_con : path_constraints
        Path constraints due to velocity and acceleration.
    s0 : float, optional
        Path start position, by default None.
    send : float, optional
        Path end position, by default None.

    Returns
    -------
    np.ndarray
        Path velocity bounds.
    """
    if s0 is None:
        s0 = 0

    if send is None:
        send = path_kin.path_s[-1]

    plt.figure("Path bounds")
    plt.clf()

    n = len(path_kin.path_s)
    sd_bounds = np.zeros((n, 5))

    for i in range(n):
        if path_kin.path_s[i] >= s0 and path_kin.path_s[i] <= send:
            path_kin.calc(path_kin.path_s[i])
            sdb2 = velbounds(path_kin, path_con)
            sd_bounds[i, :] = sdb2

    sdx = np.max(np.min(sd_bounds, axis=1))
    ln_tmp = plt.plot(path_kin.path_s, sd_bounds, ".-", linewidth=1)
    plt.grid(True)
    plt.xlim([s0, send])
    plt.ylim([0, 2 * sdx])
    plt.xlabel("$s$", fontsize=14, fontweight="bold", style="italic")
    plt.ylabel("$\\dot s$", fontsize=14, fontweight="bold", style="italic")
    plt.legend(ln_tmp, ["$\\dot s_{max}^{n}$", "$\\dot s_{max}^{\\dot x}$", "$\\dot s_{max}^{\\ddot x}$", "$\\dot s_{max}^{\\dot q}$", "$\\dot s_{max}^{\\ddot q}$"], loc="best")
    plt.show(block=False)
    return sd_bounds


def timeopt_joint_traj(
    path_q: ArrayLike,
    path_con: path_constraints,
    dkin: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    scale: Union[float, Sequence[float]] = (1.0, 1.0),
    s0: float = 0,
    send: Optional[float] = None,
    sd0: float = 0,
    sdend: float = 0,
    tsamp: float = 0.01,
    plot: bool = False,
    sd_bounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time optimal joint trajectory generation for bounded position, acceleration, and velocity.

    Parameters
    ----------
    path_q : ArrayLike
        Path positions in joint space (n, nj).
    path_con : path_constraints
        Velocity and acceleration constraints in Cartesian and joint space.
    dkin : Callable[..., Tuple[np.ndarray, np.ndarray]], optional
        Direct kinematic function, by default None.
    scale : Union[float, Sequence[float]], optional
        SE3 norm scale factors, by default [1.0, 1.0].
    s0 : float, optional
        Path start position, by default 0.
    send : float, optional
        Path end position, by default None.
    sd0 : float, optional
        Path start velocity, by default 0.
    sdend : float, optional
        Path end velocity, by default 0.
    tsamp : float, optional
        Sample time, by default 0.01.

    Returns
    -------
    tuple
        - t (time): Time array.
        - q (joint positions): Joint position array.
        - qd (joint velocities): Joint velocity array.
        - qdd (joint accelerations): Joint acceleration array.
    """
    path_kin = path_kinematics(path_q, dkin=dkin, Cartesian=False, scale=scale)
    T, sp, sv, sa = timeopttraj(path_kin, path_con, s0=s0, send=send, sd0=sd0, sdend=sdend, tsamp=tsamp, plot=plot, sd_bounds=sd_bounds)
    path_rq, path_rqd, path_rqdd = path_kin.s2x(sp, sv, sa)
    return T, path_rq, path_rqd, path_rqdd
