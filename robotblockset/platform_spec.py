import numpy as np
from robotblockset.platforms import platform
from robotblockset.transformations import map_pose


class tiagobase_spec(platform):
    def __init__(self):
        self.Name = "Tiagobase:"
        self.nj = 2
        self.q_max = np.array([10000, 10000])  # upper joint limits
        self.q_min = np.array([-10000, -10000])  # lower joint limits
        self.qdot_max = np.array([10, 10])  # maximal joint velocities
        self.v_max = np.array([1, 2])  # maximal task velocities
        self.v_min = np.array([-0.2, -2])  # maximal task velocities

    def Kinmodel(self, *x, out="x"):
        if len(x) > 0:
            _x = x[0]
        else:
            _x = self.x

        wheel_r = 0.0985
        wheel_d = 0.4044
        plate_h = 0.2976
        J = np.zeros((6, 2))
        J[0, :] = np.array([wheel_r / 2, wheel_r / 2])
        J[5, :] = np.array([-wheel_r / wheel_d, wheel_r / wheel_d])
        return map_pose(x=_x, out=out), J


if __name__ == "__main__":

    b = tiagobase_spec()
    print(b.Jacobi)
