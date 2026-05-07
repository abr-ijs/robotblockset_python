"""Graphics Module.

This module provides various utilities for 3D plotting and visualization based on `matplotlib`.
It includes functions for visualizing Cartesian trajectories, path points, coordinate systems, and more.
These utilities are designed to aid in robot trajectory visualization and analysis.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3D
except Exception as e:
    raise e from RuntimeError("Python module matplotlib not installed. \nYou can install it through pip:\n   pip install matplotlib")

from robotblockset.transformations import map_pose
from robotblockset.tools import gradientCartesianPath, gradientPath, isscalar, rbs_type, vecnormalize, vector
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatricesType, JointPathType, NumpyFloatImageType, NumpyIntImageType, OpenCVIntImageType, Poses3DType, QuaternionsType, RotationMatricesType, TimesType, Vector3DType, WrenchType


def plotucs(x: ArrayLike, UCS_length: ArrayLike = np.ones(3), UCS_linewidth: float = 1, UCS_labels: Optional[Sequence[str]] = None, UCS_handles: Optional[List[Any]] = None, ax: Optional[plt.Axes] = None) -> Tuple[List[Any], plt.Axes]:
    """
    Plot coordinate frame UCS (User Coordinate System).

    Draw or update a 3D coordinate frame at a pose given as a position, quaternion,
    homogeneous transform, or rotation matrix. Axis labels can be added, and existing
    plot handles can be reused to refresh an already drawn frame.

    Parameters
    ----------
    x : ArrayLike
        Frame pose (position and/or orientation) in form (7,), (4, 4), (3,), (4,) or (3, 3).
    UCS_length : ArrayLike, optional
        Length of UCS axes, by default np.ones(3).
    UCS_linewidth : float, optional
        Line width of UCS axes, by default 1.
    UCS_labels : Sequence[str], optional
        Labels for UCS axes, by default None.
    UCS_handles : List[Any], optional
        Handles used for updating UCS, by default None.
    ax : plt.Axes, optional
        Axes used to plot UCS, by default None.

    Returns
    -------
    hx : list
        Handles of drawn objects (lines and labels).
    ax : plt.Axes
        Axes where UCS has been plotted.

    Raises
    ------
    ValueError
        If the input parameter shapes are incorrect.
    """
    if ax is None:
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
    if ax.name != "3d":
        raise ValueError("Axes projection is not 3D")

    x = rbs_type(x)
    if x.shape == (7,):
        p, R = map_pose(x=x, out="pR")
    elif x.shape == (4, 4):
        p, R = map_pose(T=x, out="pR")
    else:
        p = np.zeros(3)
        R = np.eye(3)
        if x.shape == (3,):
            p = x
        elif x.shape == (4,):
            R = map_pose(Q=x, out="R")
        elif x.shape == (3, 3):
            R = x
        else:
            raise ValueError("Wrong input shape")

    # Check for axes handles
    if ax is None:
        if not plt.get_fignums():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = plt.gca()

    # Plot
    UCS_length = np.asarray(UCS_length, dtype="float")
    if isscalar(UCS_length):
        UCS_length = np.array([UCS_length, UCS_length, UCS_length])
    UCS_linewidth = float(UCS_linewidth)
    axlabel = ["x-axis", "y-axis", "z-axis"]
    c = np.eye(3)

    hx = []
    if not UCS_handles:
        for i in range(3):
            line = Line3D(
                [0, R[0, i] * UCS_length[i]] + p[0],
                [0, R[1, i] * UCS_length[i]] + p[1],
                [0, R[2, i] * UCS_length[i]] + p[2],
                color=c[i],
                linewidth=UCS_linewidth,
                label=axlabel[i],
            )
            ax.add_line(line)
            hx.append(line)
    else:
        for i in range(3):
            UCS_handles[i].set_xdata([0, R[0, i] * UCS_length[i]] + p[0])
            UCS_handles[i].set_ydata([0, R[1, i] * UCS_length[i]] + p[1])
            UCS_handles[i].set_3d_properties([0, R[2, i] * UCS_length[i]] + p[2])

    if UCS_labels:
        for i, _lab in enumerate(UCS_labels):
            hx.append(
                ax.text(
                    R[0, i] * UCS_length[i] + p[0],
                    R[1, i] * UCS_length[i] + p[1],
                    R[2, i] * UCS_length[i] + p[2],
                    _lab,
                    color=c[i],
                )
            )

    return hx, ax


def plotspheregrid(radius: float = 1.0, alpha: float = 1.0, pos: Vector3DType = np.array([0, 0, 0]), N: int = 36, ax: Optional[plt.Axes] = None) -> Tuple[List[Any], plt.Axes]:
    """
    Plot a sphere grid in 3D space.

    Create a spherical reference surface that helps visualize orientation trajectories
    or provides spatial context for 3D plots. The radius, position, transparency, and
    mesh density can be adjusted.

    Parameters
    ----------
    radius : float, optional
        Sphere radius, by default 1.0.
    alpha : float, optional
        Sphere transparency, by default 1.0.
    pos : Vector3DType, optional
        Sphere origin position, by default np.array([0, 0, 0]).
    N : int, optional
        Number of sphere grid lines, by default 36.
    ax : plt.Axes, optional
        Axes used to plot the sphere, by default None.

    Returns
    -------
    hx : list
        Handles of drawn objects (the sphere).
    ax : plt.Axes
        Axes where the sphere has been plotted.

    Raises
    ------
    ValueError
        If the input parameters are incorrect.
    """
    if ax is None:
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
    if ax.name != "3d":
        raise ValueError("Axes projection is not 3D")

    p = np.array(pos).flatten()
    if len(p) != 3:
        raise ValueError("Invalid 'pos' argument. It should be a 3-element numeric array.")

    if not isinstance(radius, (int, float)):
        raise ValueError("Invalid 'radius' argument. It should be a scalar numeric value.")

    if not isinstance(N, int) or N < 12:
        raise ValueError("Invalid 'N' argument. It should be an integer greater than or equal to 12.")

    if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
        raise ValueError("Invalid 'alpha' argument. It should be a numeric value between 0 and 1.")

    # Create sphere grid
    th, phi = np.meshgrid(np.linspace(0, 2 * np.pi, N + 1), np.linspace(0, np.pi, N + 1))
    x, y, z = np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)
    x = radius * x + p[0]
    y = radius * y + p[1]
    z = radius * z + p[2]

    # Plot
    hx = []
    if alpha == 1:
        hx.append(ax.plot_surface(x, y, z, facecolor=(1, 1, 1), alpha=1, edgecolor=(0.8, 0.8, 0.8)))
    else:
        hx.append(
            ax.plot_surface(
                x,
                y,
                z,
                facecolor=(0.9, 0.9, 0.9),
                alpha=alpha,
                edgecolor=(0.7, 0.7, 0.7),
            )
        )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)

    return hx, ax


def plotarrow(p1: Vector3DType, p2: Vector3DType, radius: float = 0.02, head_length: float = 0.12, head_radius: float = 0.04, color: str = "k", ax: Optional[plt.Axes] = None) -> Tuple[List[Any], plt.Axes]:
    """
    Plot an arrow from point p1 to point p2.

    Draw a lightweight 3D arrow between two points using Matplotlib line primitives.
    The shaft width, head geometry, and color can be tuned to annotate positions and
    directions in trajectory plots.

    Parameters
    ----------
    p1 : Vector3DType
        Initial arrow position (3,).
    p2 : Vector3DType
        Final arrow position (3,).
    radius : float, optional
        Arrow line width, by default 0.02.
    head_length : float, optional
        Relative arrow head length, by default 0.12.
    head_radius : float, optional
        Relative arrow head width, by default 0.04.
    color : str, optional
        Arrow color, by default "k" (black).
    ax : plt.Axes, optional
        Axes to be used for plotting the arrow, by default None.

    Returns
    -------
    hx : list
        Handles of drawn objects (the arrow).
    ax : plt.Axes
        Axes where the arrow has been plotted.

    Raises
    ------
    ValueError
        If input parameters are incorrect.
    """
    if ax is None:
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
    if ax.name != "3d":
        raise ValueError("Axes projection is not 3D")

    # Parameters
    p1 = vector(p1, dim=3)
    p2 = vector(p2, dim=3)
    dp = p2 - p1
    dp_norm = np.linalg.norm(dp)
    radius = np.max((1.0, radius))
    head_length = head_length * dp_norm
    head_radius = head_radius * dp_norm

    hx = []
    phi = np.arctan2(head_radius, head_length)
    theta = np.arctan2(dp[1], dp[0])
    hx.append(
        ax.plot(
            [
                p1[0],
                p2[0],
                p2[0] - np.cos(theta + phi) * head_length,
                p2[0],
                p2[0] - np.cos(theta - phi) * head_length,
            ],
            [
                p1[1],
                p2[1],
                p2[1] - np.sin(theta + phi) * head_length,
                p2[1],
                p2[1] - np.sin(theta - phi) * head_length,
            ],
            color=color,
            linewidth=radius,
        )
    )
    return hx, ax


def plotcpos_ori(
    t: TimesType,
    x: Optional[Poses3DType] = None,
    T: Optional[HomogeneousMatricesType] = None,
    p: Optional[ArrayLike] = None,
    R: Optional[RotationMatricesType] = None,
    Q: Optional[QuaternionsType] = None,
    typ: str = "Pos",
    graph: str = "Time",
    grid: bool = True,
    UCS: bool = False,
    label: bool = False,
    alpha: float = 0.1,
    ori_sel: List[int] = [1, 2],
    fig_num: Union[str, int] = "Cartesian poses",
    ax: Optional[plt.Axes] = None,
) -> Tuple[List[Any], plt.Axes]:
    """Plot positions or orientations of Cartesian trajecotry

    Visualize a Cartesian trajectory either as time-domain signals or as a 3D curve.
    Position mode shows translational motion, while orientation mode shows quaternion
    components over time or projected orientation paths on a sphere.

    Trajectory is defined by one representation x, t, p, R, Q

    Parameters
    ----------
    t : TimesType
        time (n,)
    x : Poses3DType, optional
        Cartesian trajectory (n, 7), by default None
    T : HomogeneousMatricesType, optional
        Cartesian trajectory (n, 4, 4), by default None
    p : ArrayLike, optional
        Cartesian positions (n, 3), by default None
    R : RotationMatricesType, optional
        Cartesian rotations (n, 3, 3), by default None
    Q : QuaternionsType, optional
        quaternions (n, 4), by default None
    typ : str, optional
        Plot signal selection: positions ("Pos") or orientations ("Ori"), by default "Pos"
    graph : str, optional
        Plot type selection: time signals ("Time") or 3D plots ("3D"), by default "Time"
    grid : bool, optional
        Grid flag, by default True
    UCS : bool, optional
        Plot UCS flag, by default False
    label : bool, optional
        Plot labels for points in 3D, by default False
    alpha : float, optional
        Transparency of sphere grid for orientations, by default 0.1
    ori_sel : List[int], optional
        Selection of two quaternions rotations for 3D plot (1, 2 or 3), by default [1, 2]
    fig_num : str, optional
        Figure identifier, by default 1
    ax : plt.Axes, optional
        Axes to be used for plot, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : plt.Axes
        axes where UCS has been ploted

    Raises
    ------
    ValueError
        If the input parameters are invalid.
    """
    t = rbs_type(t)
    x = map_pose(x=x, T=T, p=p, R=R, Q=Q, out="x")

    if ax is None:
        fig = plt.figure(num=fig_num)
        fig.clear()
        if np.char.upper(graph) == "3D":
            ax = fig.add_subplot(projection="3d")
        else:
            ax = fig.add_subplot()
    else:
        if np.char.upper(graph) == "3D":
            if ax.name != "3d":
                raise ValueError("Axes projection is not 3D")
        else:
            if ax.name != "rectilinear":
                raise ValueError("Axes projection is not 2D")
        fig = ax.get_figure()

    hx = []
    if np.char.upper(graph) == "TIME":
        if np.char.upper(typ) == "POS":
            hx.append(ax.plot(t, x[:, 0:3]))
            ax.set_ylabel("$p$")
        elif np.char.upper(typ) == "ORI":
            hx.append(ax.plot(t, x[:, 3:]))
            ax.set_ylabel("$Q$")
        ax.set_xlabel("$t$")
        ax.grid(grid)

    elif np.char.upper(graph) == "3D":
        if np.char.upper(typ) == "POS":
            hx.append(ax.plot(x[:, 0], x[:, 1], x[:, 2]))
            if UCS:
                for i in range(x.shape[0]):
                    plotucs(x[i, :], UCS_length=0.04)
            ax.grid(grid)
            ax.axis("equal")
            if label:
                for i in range(x.shape[0]):
                    if i == 0 or not all(x[i, :] == x[i - 1, :]):
                        ax.text(
                            x[i, 0],
                            x[i, 1],
                            x[i, 2],
                            f"$P_{i}$",
                            fontsize=12,
                            verticalalignment="bottom",
                        )
            ax.set_xlabel("$x$", fontsize=12)
            ax.set_ylabel("$y$", fontsize=12)
            ax.set_zlabel("$z$", fontsize=12)
        elif np.char.upper(typ) == "ORI":
            if grid:
                plotspheregrid(ax=ax, alpha=alpha)
            ax.axis("off")
            qq = x[:, np.append(3, np.array(ori_sel) + 3)]
            qq = vecnormalize(qq)
            hx.append(ax.plot(qq[:, 0], qq[:, 1], qq[:, 2]))
            if label:
                pnt = qq * 1.05
                for i in range(pnt.shape[0]):
                    if i == 0 or not np.array_equal(pnt[i, :], pnt[i - 1, :]):
                        hx.append(
                            ax.text(
                                pnt[i, 0],
                                pnt[i, 1],
                                pnt[i, 2],
                                f"$Q_{i}$",
                                fontsize=12,
                                verticalalignment="bottom",
                            )
                        )
    return hx, ax


def plotcpath(s: ArrayLike, path: Poses3DType, points: Optional[Poses3DType] = None, auxpoints: Optional[Poses3DType] = None, grid: bool = True, UCS: bool = True, label: bool = True, ori_sel: List[int] = [0, 1], normscale: float = 1, fig_num: Union[str, int] = "Cartesian path", **kwargs) -> Tuple[List[Any], List[plt.Axes]]:
    """
    Plot positions, orientations, velocities, and accelerations for a Cartesian path vs path parameter.

    Generate a combined overview of a Cartesian path, including 3D position,
    orientation evolution, and derivative signals with respect to the path parameter.
    Optional waypoints, auxiliary points, labels, and UCS frames can be overlaid for
    path debugging.

    Parameters
    ----------
    s : ArrayLike
        Path parameter (n,).
    path : Poses3DType
        Cartesian path (n, 7).
    points : Poses3DType, optional
        Cartesian points used to generate path (m, 7), by default None.
    auxpoints : Poses3DType, optional
        Auxiliary Cartesian points, by default None.
    grid : bool, optional
        Grid flag, by default True.
    UCS : bool, optional
        Plot UCS axes, by default True.
    label : bool, optional
        Label points, by default True.
    ori_sel : List[int], optional
        Selection of quaternion rotations for 3D plot, by default [1, 2].
    normscale : float, optional
        Scale factor for normalizing the path, by default 1.
    fig_num : str, optional
        Figure identifier, by default "Cartesian path".
    ax : plt.Axes, optional
        Axes to be used for plot, by default None.

    Returns
    -------
    hx : list
        Handles of drawn objects (path points, labels, etc.).
    ax : list
        Axes where the plot was drawn.
    """
    si = rbs_type(s)
    xi = rbs_type(path)
    xaux = rbs_type(auxpoints)
    nd = xi.shape[1]

    hx = []
    # 3D positions
    fig_rgt = plt.figure(num=fig_num, figsize=(9, 6))
    ax3d = fig_rgt.add_subplot(position=[0.0, 0.5, 0.38, 0.45], projection="3d")
    # ax3d.set_position([0.0, 0.5, 0.45, 0.45])
    if points is not None:
        hx.append(ax3d.plot(points[:, 0], points[:, 1], points[:, 2], "r--", linewidth=2))
    if auxpoints is not None:
        hx.append(ax3d.plot(auxpoints[:, 0], auxpoints[:, 1], auxpoints[:, 2], "c.", markersize=10))
    hx.append(ax3d.plot(xi[:, 0], xi[:, 1], xi[:, 2], "m", linewidth=2))
    # 3D UCS for orientations
    if nd == 7 and UCS:
        if auxpoints is not None:
            for _x in xaux:
                plotucs(_x, UCS_length=0.04, UCS_linewidth=1)
        for _x in xi:
            plotucs(_x, UCS_length=0.02, UCS_linewidth=0.25)
        plotucs(xi[0, :], UCS_length=0.1, UCS_linewidth=2)
    # Labels for points
    if points is not None and label:
        for i in range(points.shape[0]):
            if i == 1 or any(points[i, :3] != points[i - 1, :3]):
                ax3d.text(points[i, 0], points[i, 1], points[i, 2], "$P_" + str(i + 1) + "$")

    ax3d.grid(visible=grid)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Generated path positions")

    # 3D Euler angles
    if nd == 7:
        ax3do = fig_rgt.add_subplot(position=[0.0, 0.0, 0.38, 0.4], projection="3d")
        # ax3do.set_position([0.0, 0.0, 0.45, 0.45])
        plotspheregrid(ax=ax3do, alpha=0.1)
        ax3do.axis("off")
        ax3do.set_title("Generated path orientations")
        if points is not None:
            ptq = vecnormalize(points[:, np.append(3, np.array(ori_sel) + 3)]) * 1.05
            hx.append(ax3do.plot(ptq[:, 0], ptq[:, 1], ptq[:, 2], "r--", linewidth=2))
        if auxpoints is not None:
            qq = vecnormalize(xaux[:, np.append(3, np.array(ori_sel) + 3)]) * 1.05
            hx.append(ax3do.plot(qq[:, 0], qq[:, 1], qq[:, 2], "c.", markersize=10))
        qi = vecnormalize(xi[:, np.append(3, np.array(ori_sel) + 3)]) * 1.04
        hx.append(ax3do.plot(qi[:, 0], qi[:, 1], qi[:, 2], "m", linewidth=2))

        if points is not None and label:
            for i in range(points.shape[0]):
                if i == 1 or any(ptq[i, :] != ptq[i - 1, :]):
                    ax3do.text(ptq[i, 0], ptq[i, 1], ptq[i, 2], "$Q_" + str(i + 1) + "$")

    # Path responses versis s
    _dx = 0.22
    _dy = 0.20
    _x0 = 0.45
    _y0 = 0.04
    _xd = 0.30
    _yd = 0.25

    sid = gradientPath(si)
    if nd == 7:
        xid = gradientCartesianPath(xi, si)
        nxid = np.sqrt(np.linalg.norm(xid[:, 3:], axis=1) ** 2 * normscale**2 + np.linalg.norm(xid[:, :3], axis=1) ** 2)
    else:
        xid = gradientPath(xi, si)
        nxid = np.linalg.norm(xid, axis=1)
    # sidd = gradientPath(sid)
    xidd = gradientPath(xid, si)

    axs1 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 3, _dx, _dy])
    hx.append(axs1.plot(si, sid, **kwargs))
    axs1.set_ylim([0, np.max(sid) * 1.1])
    axs1.grid(visible=grid)
    axs1.set_ylabel("$\\Delta s$")

    axs2 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 3, _dx, _dy])
    hx.append(axs2.plot(si, nxid, **kwargs))
    axs2.set_ylim([0, np.max(nxid) * 1.1])
    axs2.grid(visible=grid)
    axs2.set_ylabel("$\\|\\dot x\\|$")

    axs3 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 2, _dx, _dy])
    hx.append(axs3.plot(si, xi[:, :3], **kwargs))
    axs3.grid(visible=grid)
    axs3.set_ylabel("$p$")

    axs4 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 1, _dx, _dy])
    hx.append(axs4.plot(si, xid[:, :3], **kwargs))
    hx.append(axs4.plot(si, np.linalg.norm(xid[:, :3], axis=1), "k--"))
    axs4.grid(visible=grid)
    axs4.set_ylabel("$\\dot p$")

    axs5 = fig_rgt.add_subplot(position=[_x0, _y0, _dx, _dy])
    hx.append(axs5.plot(si, xidd[:, :3], **kwargs))
    axs5.grid(visible=grid)
    axs5.set_ylabel("$\\ddot p$")

    if nd == 7:
        axs6 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 2, _dx, _dy])
        hx.append(axs6.plot(si, xi[:, 3:], **kwargs))
        axs6.grid(visible=grid)
        axs6.set_ylabel("$Q$")

        axs7 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 1, _dx, _dy])
        hx.append(axs7.plot(si, xid[:, 3:], **kwargs))
        hx.append(axs7.plot(si, np.linalg.norm(xid[:, 3:], axis=1), "k--"))
        axs7.grid(visible=grid)
        axs7.set_ylabel("$\\omega$")

        axs8 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0, _dx, _dy])
        hx.append(axs8.plot(si, xidd[:, 3:], **kwargs))
        axs8.grid(visible=grid)
        axs8.set_ylabel("$\\dot \\omega$")

    ax = []
    ax.append(ax3d)
    if nd == 7:
        ax.append(ax3do)
    ax.append(axs1)
    ax.append(axs2)
    ax.append(axs3)
    ax.append(axs4)
    ax.append(axs5)
    if nd == 7:
        ax.append(axs6)
        ax.append(axs7)
        ax.append(axs8)
    return hx, ax


def plotctraj(t: TimesType, xt: Poses3DType, *args: ArrayLike, grid: bool = True, fig_num: Union[str, int] = "Cartesian trajectory", ax: Optional[np.ndarray] = None, **kwargs) -> Tuple[List[Any], np.ndarray]:
    """
    Plot positions, orientations, velocities, and accelerations for Cartesian trajectory.

    Plot a Cartesian trajectory and its first and second derivatives in aligned
    subplots. If velocity and acceleration are not provided, they are estimated
    numerically from the trajectory and time vector.

    Parameters
    ----------
    t : TimesType
        Time array (n,).
    xt : Poses3DType
        Cartesian position trajectory (n, 7).
    *args : ArrayLike, optional
        Cartesian velocity and acceleration trajectories (n, 6), by default None.
    grid : bool, optional
        Grid flag, by default True.
    fig_num : str, optional
        Figure identifier, by default "Cartesian trajectory".
    ax : np.ndarray, optional
        List of axes to be used for plot (3, 2), by default None.
    **kwargs : optional
        Additional arguments passed to plot commands.

    Returns
    -------
    hx : list
        Handles of drawn objects.
    ax : np.ndarray
        Axes of subplots.
    """
    t = vector(t)
    xt = rbs_type(xt)
    if len(args) > 0:
        xdt = rbs_type(args[0])
    else:
        xdt = gradientCartesianPath(xt, t)
    if len(args) > 1:
        xddt = rbs_type(args[1])
    else:
        xddt = gradientPath(xdt, t)

    nd = xt.shape[1]
    hx = []
    if nd == 7:
        if ax is None:
            fig = plt.figure(num=fig_num)
            fig.clear()
            ax = fig.subplots(3, 2)
        else:
            if ax.shape != (3, 2):
                raise ValueError("Axes have to represent (3, 2) subplots")

        hx.append(ax[0, 0].plot(t, xt[:, :3], **kwargs))
        ax[0, 0].grid(visible=grid)
        # ax[0, 0].set_xlabel("$t$")
        ax[0, 0].set_ylabel("$p$")

        hx.append(ax[1, 0].plot(t, xdt[:, :3], **kwargs))
        ax[1, 0].grid(visible=grid)
        # ax[1, 0].set_xlabel("$t$")
        ax[1, 0].set_ylabel("$\\dot p$")

        hx.append(ax[2, 0].plot(t, xddt[:, :3], **kwargs))
        ax[2, 0].grid(visible=grid)
        ax[2, 0].set_xlabel("$t$")
        ax[2, 0].set_ylabel("$\\ddot p$")

        hx.append(ax[0, 1].plot(t, xt[:, 3:], **kwargs))
        ax[0, 1].grid(visible=grid)
        # ax[0, 1].set_xlabel("$t$")
        ax[0, 1].set_ylabel("$Q$")

        hx.append(ax[1, 1].plot(t, xdt[:, 3:], **kwargs))
        ax[1, 1].grid(visible=grid)
        # ax[1, 1].set_xlabel("$t$")
        ax[1, 1].set_ylabel("$\\omega$")

        hx.append(ax[2, 1].plot(t, xddt[:, 3:], **kwargs))
        ax[2, 1].grid(visible=grid)
        ax[2, 1].set_xlabel("$t$")
        ax[2, 1].set_ylabel("$\\dot \\omega$")

    else:
        if ax is None:
            fig = plt.figure(num=fig_num)
            fig.clear()
            ax = fig.subplots(3, 1)
        else:
            if ax.shape != (3, 1):
                raise ValueError("Axes have to represent (3, 1) subplots")
        hx.append(ax[0].plot(t, xt[:, :3], **kwargs))
        ax[0].grid(visible=grid)
        # ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$p$")

        hx.append(ax[1].plot(t, xdt[:, :3], **kwargs))
        ax[1].grid(visible=grid)
        # ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$\\dot p$")

        hx.append(ax[2].plot(t, xddt[:, :3], **kwargs))
        ax[2].grid(visible=grid)
        ax[2].set_xlabel("$t$")
        ax[2].set_ylabel("$\\ddot p$")

    return hx, ax


def plotpathpoints(x: Optional[Poses3DType] = None, T: Optional[HomogeneousMatricesType] = None, p: Optional[ArrayLike] = None, label: bool = False, fig_num: Union[str, int] = "Path", ax: Optional[plt.Axes] = None, **kwargs) -> Tuple[List[Any], plt.Axes]:
    """
    Plot the path points of a Cartesian trajectory.

    Draw Cartesian sample points as a dashed 3D path with point markers. Optional
    labels help identify the original path points or waypoints that define the motion.

    Parameters
    ----------
    x : Poses3DType, optional
        Cartesian trajectory (n, 7), by default None.
    T : HomogeneousMatricesType, optional
        Cartesian trajectory (n, 4, 4), by default None.
    p : ArrayLike, optional
        Cartesian positions (n, 3), by default None.
    label : bool, optional
        Label points, by default False.
    fig_num : str, optional
        Figure identifier, by default "Path".
    ax : plt.Axes, optional
        Axes to be used for plot, by default None.

    Returns
    -------
    hx : list
        Handles of drawn objects (points).
    ax : plt.Axes
        Axes where the points were plotted.
    """
    if ax is None:
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig = plt.figure()
            fig.clear()
            ax = fig.add_subplot(projection="3d")
    if ax.name != "3d":
        raise ValueError("Axes projection is not 3D")

    points = map_pose(x=x, T=T, p=p, out="p")
    hx = []
    hx.append(ax.plot(points[:, 0], points[:, 1], points[:, 2], "r--", linewidth=2))
    hx.append(ax.plot(points[:, 0], points[:, 1], points[:, 2], "c.", markersize=10))
    if label:
        for i in range(points.shape[0]):
            if i == 1 or any(points[i, :3] != points[i - 1, :3]):
                hx.append(
                    ax.text(
                        points[i, 0],
                        points[i, 1],
                        points[i, 2],
                        "$P_" + str(i + 1) + "$",
                    )
                )
    return hx, ax


def plotwrench(t: TimesType, FTt: WrenchType, grid: bool = True, ax: Optional[np.ndarray] = None, fig_num: Union[str, int] = "Task forces", **kwargs) -> Tuple[List[Any], np.ndarray]:
    """
    Plot force and torque signals over time.

    Plot the force and torque components of a wrench in separate subplots so task-space
    loads can be inspected over time.

    Parameters
    ----------
    t : TimesType
        Time (n,).
    FTt : WrenchType
        Force and torque signals (n, 6).
    grid : bool, optional
        Grid flag, by default True.
    fig_num : str, optional
        Figure identifier, by default "Task forces".
    ax : np.ndarray, optional
        List of axes to be used for plot, by default None.
    **kwargs : optional
        Additional arguments passed to plot commands.

    Returns
    -------
    hx : list
        Handles of drawn objects.
    ax : np.ndarray
        Axes of subplots.
    """
    t = vector(t)
    FTt = rbs_type(FTt)

    hx = []
    if ax is None:
        fig = plt.figure(num=fig_num)
        fig.clear()
        ax = fig.subplots(2, 1)
    else:
        if ax.shape != (2, 1):
            raise ValueError("Axes have to represent (2, 1) subplots")

    hx.append(ax[0].plot(t, FTt[:, :3], **kwargs))
    ax[0].grid(visible=grid)
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$F$")

    hx.append(ax[1].plot(t, FTt[:, 3:], **kwargs))
    ax[1].grid(visible=grid)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$T$")

    return hx, ax


def plotjtraj(t: TimesType, qt: JointPathType, *args: ArrayLike, grid: bool = True, ax: Optional[np.ndarray] = None, fig_num: Union[str, int] = "Joint trajectory", **kwargs) -> Tuple[List[Any], np.ndarray]:
    """
    Plot positions, velocities, and accelerations for joint trajectory.

    Visualize joint motion together with velocity and acceleration profiles. Missing
    derivative signals are estimated numerically from the supplied joint trajectory.

    Parameters
    ----------
    t : TimesType
        Time (n,).
    qt : JointPathType
        Joint position trajectory (n, nj).
    *args : ArrayLike, optional
        Joint velocity and acceleration trajectories (n, nj), by default None.
    grid : bool, optional
        Grid flag, by default True.
    fig_num : str, optional
        Figure identifier, by default "Joint trajectory".
    ax : np.ndarray, optional
        List of axes to be used for plot (3,), by default None.
    **kwargs : optional
        Additional arguments passed to plot commands.

    Returns
    -------
    hx : list
        Handles of drawn objects.
    ax : np.ndarray
        Axes of subplots.
    """
    t = vector(t)
    qt = rbs_type(qt)
    if len(args) > 0:
        qdt = rbs_type(args[0])
    else:
        qdt = gradientPath(qt, t)
    if len(args) > 1:
        qddt = rbs_type(args[1])
    else:
        qddt = gradientPath(qdt, t)

    hx = []
    if ax is None:
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1)
    else:
        if ax.shape != (3,):
            raise ValueError("Axes have to represent (3, ) subplots")

    hx.append(ax[0].plot(t, qt, **kwargs))
    ax[0].grid(visible=grid)
    # ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$q$")

    hx.append(ax[1].plot(t, qdt, **kwargs))
    ax[1].grid(visible=grid)
    # ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$\\dot q$")

    hx.append(ax[2].plot(t, qddt, **kwargs))
    ax[2].grid(visible=grid)
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$\\ddot q$")

    return hx, ax


def plotjctraj(t: TimesType, qt: JointPathType, xt: Poses3DType, *args: ArrayLike, grid: bool = True, ax: Optional[np.ndarray] = None, fig_num: Union[str, int] = "Joint and task trajectory", **kwargs) -> Tuple[List[Any], np.ndarray]:
    """
    Plot positions, orientations, and velocities for joint and task trajectory.

    Show joint-space and Cartesian-space signals side by side so manipulator motion can
    be compared with the resulting task-space path and velocity behavior.

    Parameters
    ----------
    t : TimesType
        Time (n,).
    qt : JointPathType
        Joint position trajectory (n, nj).
    xt : Poses3DType
        Cartesian position trajectory (n, nj).
    *args : ArrayLike, optional
        Joint and task velocity trajectories, by default None.
    grid : bool, optional
        Grid flag, by default True.
    fig_num : str, optional
        Figure identifier, by default "Joint and task trajectory".
    ax : np.ndarray, optional
        List of axes to be used for plot (3,), by default None.
    **kwargs : optional
        Additional arguments passed to plot commands.

    Returns
    -------
    hx : list
        Handles of drawn objects.
    ax : np.ndarray
        Axes of subplots.
    """
    t = vector(t)
    qt = rbs_type(qt)
    xt = rbs_type(xt)
    if len(args) > 0:
        qdt = rbs_type(args[0])
    else:
        qdt = gradientPath(qt, t)
    if len(args) > 1:
        xdt = rbs_type(args[1])
    else:
        xdt = gradientPath(xt, t)

    hx = []
    if ax is None:
        fig = plt.figure(num=fig_num, figsize=(12, 4))
        ax = fig.subplots(2, 3)
    else:
        if ax.shape != (2, 3):
            raise ValueError("Axes have to represent (2, 3) subplots")

    hx.append(ax[0, 0].plot(t, qt, **kwargs))
    ax[0, 0].grid(visible=grid)
    ax[0, 0].set_xlabel("$t$")
    ax[0, 0].set_ylabel("$q$")

    hx.append(ax[1, 0].plot(t, qdt, **kwargs))
    ax[1, 0].grid(visible=grid)
    ax[1, 0].set_xlabel("$t$")
    ax[1, 0].set_ylabel("$\\dot q$")

    hx.append(ax[0, 1].plot(t, xt[:, :3], **kwargs))
    ax[0, 1].grid(visible=grid)
    ax[0, 1].set_xlabel("$t$")
    ax[0, 1].set_ylabel("$p$")

    hx.append(ax[1, 1].plot(t, xdt[:, :3], **kwargs))
    ax[1, 1].grid(visible=grid)
    ax[1, 1].set_xlabel("$t$")
    ax[1, 1].set_ylabel("$\\dot p$")

    hx.append(ax[0, 2].plot(t, xt[:, 3:], **kwargs))
    ax[0, 2].grid(visible=grid)
    ax[0, 2].set_xlabel("$t$")
    ax[0, 2].set_ylabel("$Q$")

    hx.append(ax[1, 2].plot(t, xdt[:, 3:], **kwargs))
    ax[1, 2].grid(visible=grid)
    ax[1, 2].set_xlabel("$t$")
    ax[1, 2].set_ylabel("$\\omega$")

    return hx, ax


def plot_circle(r: float = 1, center: ArrayLike = (0, 0), color: str = "b", linestyle: str = "-", linewidth: float = 2) -> None:
    """
    Plots a circle with a given radius and center using Matplotlib.

    Draw a 2D circle on the current axes using a sampled parametric curve. The radius,
    center, color, line style, and line width can all be customized.

    Parameters
    ----------
    r : float, optional
        Radius of the circle. Defaults to 1.
    center : ArrayLike, optional
        Coordinates of the circle center (x, y). Defaults to (0,0).
    color : str, optional
        Color of the circle. Defaults to 'b' (blue).
    linestyle : str, optional
        Line style for the circle. Defaults to '-'.
    linewidth : float, optional
        Width of the circle line. Defaults to 2.

    Returns
    -------
    None
        This function has no return value. It draws the circle on the current axes.
    """
    theta = np.linspace(0, 2 * np.pi, 300)  # Generate 300 points around the circle
    x = center[0] + r * np.cos(theta)  # X coordinates
    y = center[1] + r * np.sin(theta)  # Y coordinates

    plt.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)


def linkxaxes(ax: Optional[Sequence[plt.Axes]] = None) -> None:
    """
    Share the x-axis between all axes in the list.

    Link the x-limits of multiple subplots so zooming or panning one axis updates the
    others. If no axes are provided, all axes in the current figure are linked.

    Parameters
    ----------
    ax : Sequence[plt.Axes], optional
        List of axes, by default None (link all subplots in current figure).

    Returns
    -------
    None
        This function has no return value. It links the x-axes of the given axes.
    """
    if ax is None:
        ax = plt.gcf().axes
    parent = ax[0]
    for i in range(1, len(ax)):
        ax[i].sharex(parent)


def display_images(images: Sequence[Union[OpenCVIntImageType, NumpyIntImageType, NumpyFloatImageType]], bgr2rgb: bool = False) -> None:
    """
    Display a collection of images in a near-square grid.

    Arrange a list of images into a compact grid for quick visual inspection. The
    layout is chosen automatically to be close to square, and OpenCV-style BGR images
    can optionally be converted to RGB before display.

    The function arranges the given images into a grid whose number of rows
    and columns is chosen to be as close to square as possible. Each image is
    rendered using a grayscale colormap and axes are hidden.

    Parameters
    ----------
    images : Sequence[OpenCVIntImageType | NumpyIntImageType | NumpyFloatImageType]
        A sequence of 2D (H, W) or 3D (H, W, C) NumPy arrays representing images.
        Images are displayed in the order provided.
    bgr2rgb : bool, optional
        Convert 3-channel OpenCV BGR images to RGB before display, by default False.

    Returns
    -------
    None
        This function has no return value. It renders the images using
        Matplotlib.

    Notes
    -----
    The grid dimensions are computed as:

    - ``cols = ceil(sqrt(N))``
    - ``rows = ceil(N / cols)``

    where ``N`` is the number of images.
    """
    N = len(images)
    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    for i, img in enumerate(images):
        if bgr2rgb and img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
