import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

from robotblockset.transformations import map_pose
from robotblockset.tools import rbs_type, isscalar, vector, vecnormalize, gradientPath, gradientCartesianPath


def plotucs(
    x: np.ndarray,
    UCS_length: np.ndarray = np.ones(3),
    UCS_linewidth: float = 1,
    UCS_labels: str = None,
    UCS_handles: list = None,
    ax: list = None,
) -> list:
    """Plot coordinate frame UCS

    Parameters
    ----------
    x : ndarray
        frame pose (position and/or orientation) (7,) or (4, 4) or (3,) or (4,) or (3, 3)
    UCS_length : np.ndarray, optional
        length of UCS axes, by default np.ones(3)
    UCS_linewidth : float, optional
        line width of UCS, by default 1
    UCS_labels : str, optional
        labels for UCS axes, by default None
    UCS_handles : list, optional
        handles used for updateing UCS, by default None
    ax : list, optional
        axes used to plot UCS, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : Axes
        axes where UCS has been ploted

    Raises
    ------
    ValueError
        Wrong input parameter
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


def plotspheregrid(
    radius: float = 1.0,
    alpha: float = 1.0,
    pos: np.ndarray = np.array([0, 0, 0]),
    N: int = 36,
    ax: plt.Axes = None,
) -> list:
    """_summary_

    Parameters
    ----------
    radius : float, optional
        Sphere radius, by default 1.0
    alpha : float, optional
        Sphere transparency, by default 1.0
    pos : np.ndarray, optional
        Sprehe origin, by default np.array([0, 0, 0])
    N : int, optional
        Number of sphere grid lines, by default 36
    ax : plt.Axes, optional
        Axes used to plot sphere, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : Axes
        axes where UCS has been ploted

    Raises
    ------
    ValueError
        Wrong input parameter
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


def plotarrow(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float = 0.02,
    head_length: float = 0.12,
    head_radius: float = 0.04,
    color="k",
    ax=None,
) -> list:
    """Plot arrow from p1 to p2

    Parameters
    ----------
    p1 : ndarray
        initial arrow position (3,)
    p2 : ndarray
        final arrow position (3,)
    radius : float, optional
        arrow linewidth, by default 0.02
    head_length : float, optional
        relative arrow head length, by default 0.12
    head_radius : float, optional
        relative arrow head width, by default 0.04
    color : str, optional
        arrow color, by default "k"
    ax : Axes, optional
        Axes to be used to plot arrow, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : Axes
        axes where UCS has been ploted

    Raises
    ------
    ValueError
        Wrong input parameter
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

    # Compute arrow direction
    direction = dp / dp_norm

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
    t: np.ndarray,
    x: np.ndarray = None,
    T: np.ndarray = None,
    p: np.ndarray = None,
    R: np.ndarray = None,
    Q: np.ndarray = None,
    typ: str = "Pos",
    graph: str = "Time",
    grid: bool = True,
    UCS: bool = False,
    label: bool = False,
    alpha: float = 0.1,
    ori_sel: list = [1, 2],
    fig_num: str = "Cartesian poses",
    ax: plt.Axes = None,
):
    """Plot positions or orientations of Cartesian trajecotry

    Trajectory is defined by one representation x, t, p, R, Q

    Parameters
    ----------
    t : np.ndarray
        time (n,)
    x : np.ndarray, optional
        Cartesian trajectory (n, 7), by default None
    T : np.ndarray, optional
        Cartesian trajectory (n, 4, 4), by default None
    p : np.ndarray, optional
        Cartesian positions (n, 3), by default None
    R : np.ndarray, optional
        Cartesian rotations (n, 3, 3), by default None
    Q : np.ndarray, optional
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
    ori_sel : list, optional
        Selection of two quaternions rotations for 3D plot (1, 2 or 3), by default [1, 2]
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : plt.Axes, optional
        Axes to be used for plot, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : Axes
        axes where UCS has been ploted

    Raises
    ------
    ValueError
        Wrong input parameter
    """
    t = rbs_type(t)
    n = len(t)
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
        ax = fig.add_subplot()
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


def plotcpath(
    s: np.ndarray,
    path: np.ndarray,
    points: np.ndarray = None,
    auxpoints: np.ndarray = None,
    grid: bool = True,
    UCS: bool = True,
    label: bool = True,
    ori_sel: list = [1, 2],
    normscale: float = 1,
    fig_num: str = "Cartesian path",
    **kwargs,
) -> list:
    """Plot positions, orientations, velocities and accelerations for Cartesian path versus path parameter

    Parameters
    ----------
    s : np.ndarray
        path parameter (n,)
    path : np.ndarray
        Catesian path (n, 7)
    points : np.ndarray, optional
        Catesian points used to generate path (m, 7), by default None
    auxpoints : np.ndarray, optional
        Catesian auxilary points used to generate path (k, 7), by default None
    grid : bool, optional
        Grid flag, by default True
    UCS : bool, optional
        Plot UCS flag, by default False
    label : bool, optional
        Plot labels for points in 3D, by default False
    alpha : float, optional
        Transparency of sphere grid for orientations, by default 0.1
    ori_sel : list, optional
        Selection of two quaternions rotations for 3D plot (1, 2 or 3), by default [1, 2]
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : plt.Axes, optional
        Axes to be used for plot, by default None

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : Axes
        axes where UCS has been ploted
    """
    si = rbs_type(s)
    n = len(si)
    xi = rbs_type(path)
    xaux = rbs_type(auxpoints)
    nd = xi.shape[1]

    hx = []
    # 3D positions
    fig_rgt = plt.figure(num=fig_num, figsize=(12, 8))
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
        ax3do = fig_rgt.add_subplot(position=[0.0, 0.0, 0.38, 0.45], projection="3d")
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
    sidd = gradientPath(sid)
    xidd = gradientPath(xid, si)

    axs1 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 3, _dx, _dy])
    hx.append(axs1.plot(si, sid, **kwargs))
    axs1.set_ylim([0, np.max(sid) * 1.1])
    axs1.grid(visible=grid)
    axs1.set_ylabel("$\Delta s$")

    axs2 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 3, _dx, _dy])
    hx.append(axs2.plot(si, nxid, **kwargs))
    axs2.set_ylim([0, np.max(nxid) * 1.1])
    axs2.grid(visible=grid)
    axs2.set_ylabel("$\|\dot x\|$")

    axs3 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 2, _dx, _dy])
    hx.append(axs3.plot(si, xi[:, :3], **kwargs))
    axs3.grid(visible=grid)
    axs3.set_ylabel("$p$")

    axs4 = fig_rgt.add_subplot(position=[_x0, _y0 + _yd * 1, _dx, _dy])
    hx.append(axs4.plot(si, xid[:, :3], **kwargs))
    hx.append(axs4.plot(si, np.linalg.norm(xid[:, :3], axis=1), "k--"))
    axs4.grid(visible=grid)
    axs4.set_ylabel("$\dot p$")

    axs5 = fig_rgt.add_subplot(position=[_x0, _y0, _dx, _dy])
    hx.append(axs5.plot(si, xidd[:, :3], **kwargs))
    axs5.grid(visible=grid)
    axs5.set_ylabel("$\ddot p$")

    if nd == 7:
        axs6 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 2, _dx, _dy])
        hx.append(axs6.plot(si, xi[:, 3:], **kwargs))
        axs6.grid(visible=grid)
        axs6.set_ylabel("$Q$")

        axs7 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0 + _yd * 1, _dx, _dy])
        hx.append(axs7.plot(si, xid[:, 3:], **kwargs))
        hx.append(axs7.plot(si, np.linalg.norm(xid[:, 3:], axis=1), "k--"))
        axs7.grid(visible=grid)
        axs7.set_ylabel("$\omega$")

        axs8 = fig_rgt.add_subplot(position=[_x0 + _xd, _y0, _dx, _dy])
        hx.append(axs8.plot(si, xidd[:, 3:], **kwargs))
        axs8.grid(visible=grid)
        axs8.set_ylabel("$\dot \omega$")

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


def plotctraj(
    t: np.ndarray,
    xt: np.ndarray,
    *args: np.ndarray,
    grid: bool = True,
    fig_num: str = "Cartesian trajectory",
    ax: list = None,
    **kwargs,
) -> list:
    """Plot positions, orientations, velocities and accelerations for Cartesian trajectory

    Parameters
    ----------
    t : np.ndarray
        time (n,)
    xt : np.ndarray
        Catesian position trajectory (n, 7)
    *args : np.ndarray, optional
        Catesian velocity trajectory (n, 6), by default None
    *args : np.ndarray, optional
        Catesian acceleration trajectory (n, 6), by default None
    grid : bool, optional
        Grid flag, by default True
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : list of Axes, optional
        List of axes to be used for plot (3, 2), by default None
    **kwargs : optional
        Optional parameters used in some plot commands

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : list
        axes of subplots

    Raises
    ------
    ValueError
        Wrong input parameter
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
        ax[0, 0].set_xlabel("$t$")
        ax[0, 0].set_ylabel("$p$")

        hx.append(ax[1, 0].plot(t, xdt[:, :3], **kwargs))
        ax[1, 0].grid(visible=grid)
        ax[1, 0].set_xlabel("$t$")
        ax[1, 0].set_ylabel("$\dot p$")

        hx.append(ax[2, 0].plot(t, xddt[:, :3], **kwargs))
        ax[2, 0].grid(visible=grid)
        ax[2, 0].set_xlabel("$t$")
        ax[2, 0].set_ylabel("$\ddot p$")

        hx.append(ax[0, 1].plot(t, xt[:, 3:], **kwargs))
        ax[0, 1].grid(visible=grid)
        ax[0, 1].set_xlabel("$t$")
        ax[0, 1].set_ylabel("$Q$")

        hx.append(ax[1, 1].plot(t, xdt[:, 3:], **kwargs))
        ax[1, 1].grid(visible=grid)
        ax[1, 1].set_xlabel("$t$")
        ax[1, 1].set_ylabel("$\omega$")

        hx.append(ax[2, 1].plot(t, xddt[:, 3:], **kwargs))
        ax[2, 1].grid(visible=grid)
        ax[2, 1].set_xlabel("$t$")
        ax[2, 1].set_ylabel("$\dot \omega$")

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
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$p$")

        hx.append(ax[1].plot(t, xdt[:, :3], **kwargs))
        ax[1].grid(visible=grid)
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$\dot p$")

        hx.append(ax[2].plot(t, xddt[:, :3], **kwargs))
        ax[2].grid(visible=grid)
        ax[2].set_xlabel("$t$")
        ax[2].set_ylabel("$\ddot p$")

    return hx, ax


def plotpathpoints(
    x: np.ndarray = None,
    T: np.ndarray = None,
    p: np.ndarray = None,
    label: bool = False,
    fig_num: str = "Path",
    ax: plt.Axes = None,
    **kwargs,
) -> list:
    """_summary_

    Parameters
    ----------
    x : np.ndarray, optional
        Cartesian trajectory (n, 7), by default None
    T : np.ndarray, optional
        Cartesian trajectory (n, 4, 4), by default None
    p : np.ndarray, optional
        Cartesian positions (n, 3), by default None
    label : bool, optional
        Plot labels for points, by default False
    fig_num : strorint, optional
        Figure identificator, by default "Path"
    ax : Axes, optional
        Axes to be used for plot, by default None

    Returns
    -------
    ax : list
        handles of drawn objects
    hx : list
        axes of subplots

    Raises
    ------
    ValueError
        Wrong input parameter
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


def plotwrench(
    t: np.ndarray,
    FTt: np.ndarray,
    grid: bool = True,
    ax: list = None,
    fig_num="Task forces",
    **kwargs,
):
    """Plot forces and torques signals

    Parameters
    ----------
    t : np.ndarray
        time (n,)
    FTt : np.ndarray
        force and torques signals (n, 6)
    grid : bool, optional
        Grid flag, by default True
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : list of Axes, optional
        List of axes to be used for plot (2,), by default None
    **kwargs : optional
        Optional parameters used in some plot commands

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : list
        axes of subplots

    Raises
    ------
    ValueError
        Wrong input parameter
    """
    t = vector(t)
    FTt = rbs_type(FTt)

    hx = []
    if ax is None:
        fig = plt.figure(num=fig_num)
        fig.clear()
        ax = fig.subplots(1, 1)
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


def plotjtraj(t, qt, *args, grid=True, ax=None, fig_num="Joint trajectory", **kwargs):
    """Plot positions, velocities and accelerations for joint trajectory

    Parameters
    ----------
    t : np.ndarray
        time (n,)
    qt : np.ndarray
        joint position trajectory (n, nj)
    *args : np.ndarray, optional
        joint velocity trajectory (n, nj), by default None
    *args : np.ndarray, optional
        joint acceleration trajectory (n, nj), by default None
    grid : bool, optional
        Grid flag, by default True
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : list of Axes, optional
        List of axes to be used for plot (3,), by default None
    **kwargs : optional
        Optional parameters used in some plot commands

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : list
        axes of subplots

    Raises
    ------
    ValueError
        Wrong input parameter
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
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$q$")

    hx.append(ax[1].plot(t, qdt, **kwargs))
    ax[1].grid(visible=grid)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$\dot q$")

    hx.append(ax[2].plot(t, qddt, **kwargs))
    ax[2].grid(visible=grid)
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$\ddot q$")

    return hx, ax


def plotjctraj(t, qt, xt, *args, grid=True, ax=None, fig_num="Joint and task trajectory", **kwargs):
    """Plot positions, orientations and velocities for joint and task trajectory

    Parameters
    ----------
    t : np.ndarray
        time (n,)
    qt : np.ndarray
        joint position trajectory (n, nj)
    xt : np.ndarray
        Catesian position trajectory (n, nj)
    *args : np.ndarray, optional
        joint velocity trajectory (n, nj), by default None
    *args : np.ndarray, optional
        task  velocity  (n, 6), by default None
    grid : bool, optional
        Grid flag, by default True
    fig_num : str or int, optional
        Figure identifier, by default 1
    ax : list of Axes, optional
        List of axes to be used for plot (3,), by default None
    **kwargs : optional
        Optional parameters used in some plot commands

    Returns
    -------
    hx : list
        handles of drawn objects
    ax : list
        axes of subplots

    Raises
    ------
    ValueError
        Wrong input parameter
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
        if ax.shape != (2,):
            raise ValueError("Axes have to represent (3, ) subplots")

    hx.append(ax[0, 0].plot(t, qt, **kwargs))
    ax[0, 0].grid(visible=grid)
    ax[0, 0].set_xlabel("$t$")
    ax[0, 0].set_ylabel("$q$")

    hx.append(ax[1, 0].plot(t, qdt, **kwargs))
    ax[1, 0].grid(visible=grid)
    ax[1, 0].set_xlabel("$t$")
    ax[1, 0].set_ylabel("$\dot q$")

    hx.append(ax[0, 1].plot(t, xt[:, :3], **kwargs))
    ax[0, 1].grid(visible=grid)
    ax[0, 1].set_xlabel("$t$")
    ax[0, 1].set_ylabel("$p$")

    hx.append(ax[1, 1].plot(t, xdt[:, :3], **kwargs))
    ax[1, 1].grid(visible=grid)
    ax[1, 1].set_xlabel("$t$")
    ax[1, 1].set_ylabel("$\dot p$")

    hx.append(ax[0, 2].plot(t, xt[:, 3:], **kwargs))
    ax[0, 2].grid(visible=grid)
    ax[0, 2].set_xlabel("$t$")
    ax[0, 2].set_ylabel("$Q$")

    hx.append(ax[1, 2].plot(t, xdt[:, 3:], **kwargs))
    ax[1, 2].grid(visible=grid)
    ax[1, 2].set_xlabel("$t$")
    ax[1, 2].set_ylabel("$\omega$")

    return hx, ax


def linkxaxes(ax=None):
    """Share the x-axis between all axes in list

    Parameters
    ----------
    ax : list, optional
        list of axes, by default None (link all subplots in current figure)
    """
    if ax is None:
        ax = plt.gcf().axes
    parent = ax[0]
    for i in range(1, len(ax)):
        ax[i].sharex(parent)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")

    hx = plotucs(
        np.eye(4),
        UCS_length=0.8,
        UCS_linewidth=2,
        UCS_labels=["x", "y"],
    )

    hx1 = plotarrow(
        [0, 0, 0],
        [1, 1, 1],
        color="b",
    )
    hx = hx + hx1

    hx1 = plotspheregrid(pos=[1, 0, 0], radius=0.5, N=36, alpha=0.3)
    hx = hx + hx1

    ax.set_aspect("equal", "box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    # print(hx, ax)
    plt.show()
