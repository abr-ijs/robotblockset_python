"""Example script for testing time-optimal trajectories."""

import numpy as np
import matplotlib.pyplot as plt

from robotblockset.optimal import timeopttraj, timeopt_joint_traj, path_constraints, path_kinematics, plot_path_bounds
from robotblockset.trajectories import cline, gradientPath, gradientCartesianPath, uniqueCartesianPath
from robotblockset.transformations import map_pose, rot_z, prpy2x
from robotblockset.graphics import plotctraj, plotjtraj
from robotblockset.robots import robot
from robotblockset.robot_spec import panda_spec

np.set_printoptions(formatter={"float": "{: 0.4f}".format})


# Robot without a scene!
class panda_test(panda_spec):
    def __init__(self):
        panda_spec.__init__(self)
        robot.__init__(self)
        self.Init()

    def __del__(self):
        self.Message("Robot deleted", 2)


r = panda_test()
TCP = map_pose(Q=rot_z(-np.pi / 4), p=[0, 0, 0.1034])
r.SetTCP(TCP, frame="Flange")
dkin = r.Kinmodel
print(r.q_home)
print(r.TCP)

points6d = np.array(
    [
        [0.2, 0.0, 0.0, 0, 0, 0],
        [0.0, -0.2, -0.2, 0, 0, 0],
    ]
)
points6d_init = [0.3, 0, 0.5, 0, 0, np.pi]
for i in range(points6d.shape[0]):
    points6d[i, :] = points6d[i, :] + points6d_init
points = prpy2x(points6d)

# Path definition
tsamp = 0.01
tt = np.arange(0, 1 + tsamp, tsamp)
path_x = cline(points[0, :], points[1, :], tt)[0]
path_q, _err = r.IKinPath(path_x, r.q_home)

# Path constraints
path_con = path_constraints()
path_con.xdnmax = 1
path_con.xddnmax = 2
path_con.xdmax = np.ones(6) * 1
path_con.xddmax = np.ones(6) * 5
path_con.qdmax = np.ones(7) * 1
path_con.qddmax = np.ones(7) * 2

# Path kinematics
scale = [1, 1]
Cartesian = False
if Cartesian:
    path_kin = path_kinematics(path_x, path_q=path_q, dkin=dkin, Cartesian=True, scale=scale)
else:
    path_kin = path_kinematics(path_q, dkin=dkin, Cartesian=False, scale=scale)

# Bounds in (s, sd) plane
sd_bounds = plot_path_bounds(path_kin, path_con)

T, sp, sv, sa = timeopttraj(path_kin, path_con, tsamp=0.01, plot=True)
if Cartesian:
    path_rx, path_rxd, path_rxdd = path_kin.s2x(sp, sv, sa)
    path_rq, path_rqd, path_rqdd = path_kin.s2q_x(sp, sv, sa)

else:
    # T, path_rq, path_rqd, path_rqdd = timeopt_joint_traj(path_q, path_con, plot=True)
    path_rq, path_rqd, path_rqdd = path_kin.s2x(sp, sv, sa)
    n = path_rq.shape[0]
    path_rx = np.zeros((n, 7))
    for i in range(n):
        path_rx[i, :] = dkin(path_rq[i, :])[0]
    path_rx = uniqueCartesianPath(path_rx)
    path_rxd = gradientCartesianPath(path_rx, T)
    path_rxdd = gradientPath(path_rxd, T)

_, ax = plotjtraj(T, sp, sv, sa, fig_num="Time optimal path trajectory")
ax[0].set_ylabel("$s$")
ax[1].set_ylabel("$\\dot s$")
ax[2].set_ylabel("$\\ddot s$")
plotctraj(T, path_rx, fig_num="Time optimal task trajectory")
plotjtraj(T, path_rq, fig_num="Time optimal joint trajectory")

plt.show()
