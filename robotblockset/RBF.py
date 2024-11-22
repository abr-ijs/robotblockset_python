import numpy as np

from robotblockset.tools import isscalar, isvector


def encodeRBF(
    x: np.ndarray,
    y: np.ndarray,
    N: int = 25,
    c: np.ndarray = None,
    sigma2: np.ndarray = None,
    bc: np.ndarray = None,
    coff: float = 0.1,
    sfac: float = 3.0,
) -> dict:
    """Encode path y(x) with Radial basis functions (RBF) by calculating weights for
    Gaussian kernel functions (GKS) functions

    GKF: exp(-((x-c)**2/(2*sigma2)))
    RBF: sum(w*GKF(x))/sum(GKF(x))

    Optionally initial and final velocity and acceleration boundary conditions can be defined.

    Parameters
    ----------
    x : ndarray
        path parameter (n, )
    y : ndarray
        measured signals (n, m)
    N : int, optional
        number of GKF, by default 25
    c : ndarray, optional
        centers of GKF, by default None (making them equidistant or c=x if n==N)
    sigma2 : ndarray, , optional
        standard deviation of GKF (N,) , by default None (using (diff(c)*0.75)**2)
    bc : ndarray, optional
        boundary conditions [ydot(0), ydot(end), yddot(0) yddot(end)] (4, m), by default None
    coff : float, optional
        relative offset for auxilary GKF centers [0.01 ... 0.05], by default 0.1
    sfac : int, optional
        igma2 factor for auxilary GKF >=1, by default 3

    Returns
    -------
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)

    Raises
    ------
    ValueError
        Input parameters error
    """
    if not isvector(x):
        raise ValueError("Parameter x is not vector")
    x = np.asarray(x, dtype="float")
    y = np.asarray(y, dtype="float")
    n = len(x)
    if y.shape[0] != n:
        raise ValueError(f"Parameter x and y do not have corresponding shapes {x.shape} and {y.shape}")
    m = y.shape[1]

    if c is None:
        if N == n:
            c = x
        else:
            c = np.linspace(np.min(x), np.max(x), N)
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
                f / (h + np.finfo(float).eps)[:, np.newaxis],
                Ad[[0, -1], :],
                Add[[0, -1], :],
            ),
            axis=0,
        )
    else:
        A = f / (h + np.finfo(float).eps)[:, np.newaxis]

    AI = np.linalg.pinv(A)
    RBF["w"] = np.dot(AI, y)

    return RBF


def decodeRBF(x: np.ndarray, RBF: dict, calc_derivative: int = 0) -> np.ndarray:
    """Generate path at points x ecoded by Gaussian radial functions RBF with Gausian kernel functions (GKF)

    GKF: exp(-((x-c)**2/(2*sigma2)))
    RBF: sum(w*GKF(x))/sum(GKF(x))

    Parameters
    ----------
    x : np.ndarray
        path parameter (n,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)
    calc_derivative : int , optional
        selection of maximal derivative order for calculation, by default 0

    Returns
    -------
    y : np.ndarray
        path values (n, m)
    yd : np.ndarray, optional for calc_derivative>=1
        path velocities (n, m)
    ydd : np.ndarray, optional for calc_derivative>=2
        path accelerations (n, m)
    yddd : np.ndarray, optional for calc_derivative==3
        path jerks (n, m)

    Raises
    ------
    ValueError
        Input parameters error
    """
    if not isvector(x):
        raise ValueError("Parameter x is not vector")
    x = np.asarray(x, dtype="float")
    x = x.reshape(-1, 1)
    tmp1 = x - RBF["c"]
    tmp2 = -0.5 * tmp1**2
    tmp3 = tmp2 / RBF["sigma2"]
    f = np.exp(tmp3)
    h = np.sum(f, axis=1)
    A = f / (h + np.finfo(float).eps)[:, np.newaxis]

    y = np.dot(A, RBF["w"])

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


def decodeQuaternionRBF(x: np.ndarray, RBF: dict) -> np.ndarray:
    """Generate quaternion path at points x ecoded by Gaussian radial functions RBF with Gausian kernel functions (GKF)

    GKF: exp(-((x-c)**2/(2*sigma2)))
    RBF: sum(w*GKF(x))/sum(GKF(x))

    Parameters
    ----------
    x : np.ndarray
        path parameter (n,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)

    Returns
    -------
    path: np.ndarray
        path values (n, m)

    Raises
    ------
    ValueError
        Input parameters error
    """
    if not isvector(x):
        raise ValueError("Parameter x is not vector")
    if RBF["w"].shape[1] != 4:
        raise ValueError("RBF is not encoding quaternion path")

    q = decodeRBF(x, RBF, calc_derivative=0)
    qn = np.sqrt(np.sum(q**2, axis=1))
    q = q / qn[:, np.newaxis]
    return q


def decodeCartesianRBF(x: np.ndarray, RBF: dict):
    """Generate Cartesian path at points x ecoded by Gaussian radial functions RBF with Gausian kernel functions (GKF)

    GKF: exp(-((x-c)**2/(2*sigma2)))
    RBF: sum(w*GKF(x))/sum(GKF(x))

    Parameters
    ----------
    x : np.ndarray
        path parameter (n,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)

    Returns
    -------
    path : np.ndarray
        path values (n, m)

    Raises
    ------
    ValueError
        Input parameters error
    """
    if not isvector(x):
        raise ValueError("Parameter x is not vector")
    if RBF["w"].shape[1] != 7:
        raise ValueError("RBF is not encoding Cartesian path")
    x = np.asarray(x, dtype="float")
    y = decodeRBF(x, RBF, calc_derivative=0)
    q = y[:, 3:]
    qn = np.sqrt(np.sum(q**2, axis=1))
    q = q / qn[:, np.newaxis]
    y[:, 3:] = q
    return y


def jacobiRBF(x: np.ndarray, RBF: dict, deps: float = 1e-5) -> np.ndarray:
    """Jacobian and its derivative for RBF encoded path at points x using numeric differenciation

    Parameters
    ----------
    x : float
        path parameter
    yn : ndarray
        measured signals (m,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)
    deps : float, optional
        step for differentiation, by default 1e-5

    Returns
    -------
    J : np.ndarray
        Jacobian (n, m)
    Jd : np.ndarray
        Jacobian derivative (n, m)
    """
    y0 = decodeRBF(x, RBF)
    y1 = decodeRBF(x + deps, RBF)
    J = (y1 - y0) / deps

    y2 = decodeRBF(x - deps, RBF)
    J1 = (y0 - y2) / deps
    Jdot = (J - J1) / deps

    return J, Jdot


def updateRBF(x: np.ndarray, yn: np.ndarray, RBF: dict) -> dict:
    """Update RBF weights using recursive regression

    Parameters
    ----------
    x : float
        path parameter
    yn : ndarray
        measured signals (m,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)

    Returns
    -------
    path : ndarray
        calculated signal at x (m,)
    RBF : dict of RBF parameters:
         RBF["N"]       number of Gaussian kernel functions
         RBF["w"]       weights of GKF (N, m)
         RBF["c"]       centers of GKF (N,)
         RBF["sigma2"]  standard deviation of GKF (N,)

    Raises
    ------
    ValueError
        Input parameters error
    """
    if not isscalar(x):
        raise ValueError("Parameter x must be scalar")
    if not isvector(yn):
        raise ValueError("Parameter yn must be vector")
    x = float(x)
    yn = np.asarray(yn, dtype="float")

    tmp1 = x - RBF["c"]
    tmp2 = -0.5 * tmp1**2
    tmp3 = tmp2 / RBF["sigma2"]
    f = np.exp(tmp3)
    h = np.sum(f, axis=1)
    A = f / (h + np.finfo(float).eps)[:, np.newaxis]
    y = np.dot(A, RBF["w"])

    # Recursive regression
    p = RBF["p"]
    ATA = np.dot(A.T, A)
    p = 1.0 / RBF["lambda"] * (p - np.dot(np.dot(np.dot(p, ATA), p) / (RBF["lambda"] + np.dot(np.dot(A, p), A.T))))
    er = yn - y
    RBF["w"] = RBF["w"] + np.dot(np.dot(p, A.T), er)
    RBF["p"] = p
    y = np.dot(A, RBF["w"])

    return y, RBF


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tools import gradientPath
    from graphics import plotjtraj

    time = np.linspace(0.0, 10.0, 1001)
    time1 = np.linspace(0.0, 10.0, 11)
    pos_ori = np.vstack((np.sin(time), np.cos(2 * time), (time / 5 - 1) ** 2 - 1)).T
    vel_ori = gradientPath(pos_ori, time)
    acc_ori = gradientPath(vel_ori, time)

    RBF = encodeRBF(time, pos_ori)
    y, yd, ydd = decodeRBF(time, RBF, calc_derivative=2)

    hx, ax = plotjtraj(time, pos_ori, vel_ori, acc_ori, fig_num="RBF encoded path")
    plotjtraj(time, y, yd, ydd, ax=ax, color="k", linestyle="--")

    fig, ax = plt.subplots(1, 1, num="RBF encoding error")
    ax.plot(time, pos_ori - y)
    ax.grid("on")
    ax.set_ylabel("$p_{gen}-p$")
    ax.set_xlabel("$t$")

    plt.show()
