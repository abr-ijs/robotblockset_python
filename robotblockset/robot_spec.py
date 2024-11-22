import numpy as np
from robotblockset.robot_models import kinmodel_panda, kinmodel_lwr, kinmodel_iiwa, kinmodel_ur10, kinmodel_ur5
from robotblockset.robots import robot


class panda_spec(robot):
    def __init__(self):
        self.Name = "Panda:"
        self.nj = 7
        self.TCPGripper = np.array([[0.7071, 0.7071, 0, 0], [-0.7071, 0.7071, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
        self.q_home = np.array([0, -0.2, 0, -1.5, 0, 1.5, 0.7854])  # home joint configuration
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])  # upper joint limits
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # lower joint limits
        self.qdot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_panda(qq, tcp=tcp, out=out)


class fr3_spec(robot):
    def __init__(self):
        self.Name = "FR3:"
        self.nj = 7
        self.TCPGripper = np.array([[0.7071, 0.7071, 0, 0], [-0.7071, 0.7071, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
        self.q_home = np.array([0, -0.2, 0, -1.5, 0, 1.5, 0.7854])  # home joint configuration
        self.q_max = np.array([2.8973, 1.8325, 2.8973, -0.1221, 2.8797, 4.6251, 3.0543])  # upper joint limits
        self.q_min = np.array([-2.8973, -1.8325, -2.8973, -3.0717, -2.8797, -0.4363, -3.0543])  # lower joint limits
        self.qdot_max = np.array([2.6179, 2.6179, 2.6179, 2.6179, 4.1713, 4.1713, 4.1713])  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_panda(qq, tcp=tcp, out=out)


class lwr_spec(robot):
    def __init__(self):
        self.Name = "LWR:"
        self.nj = 7
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0, -0.2, 0, 1.3, 0, -0.6, 0])  # home joint configuration
        self.q_max = np.array([170, 120, 170, 120, 170, 120, 170]) * np.pi / 180  # upper joint limits
        self.q_min = -np.array([170, 120, 170, 120, 170, 120, 170]) * np.pi / 180  # lower joint limits
        self.qdot_max = np.array([100, 110, 100, 130, 130, 180, 180]) * np.pi / 180  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_lwr(q, tcp=tcp, out=out)


class iiwa_spec(robot):
    def __init__(self):
        self.Name = "iiwa:"
        self.nj = 7
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0, 0.2, 0, -1.3, 0, -0.6, 0])  # home joint configuration
        self.q_max = np.array([170, 120, 170, 120, 170, 120, 170]) * np.pi / 180  # upper joint limits
        self.q_min = -np.array([170, 120, 170, 120, 170, 120, 170]) * np.pi / 180  # lower joint limits
        self.qdot_max = np.array([100, 110, 100, 130, 130, 180, 180]) * np.pi / 180  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_iiwa(q, tcp=tcp, out=out)


class ur10_spec(robot):
    def __init__(self):
        self.Name = "UR10:"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])  # home joint configuration
        self.q_init = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, 0, +np.pi / 2, 0])  # init work joint configuration
        self.q_max = np.ones(self.nj) * 2 * np.pi  # upper joint limits
        self.q_min = -np.ones(self.nj) * 2 * np.pi  # lower joint limits
        self.qdot_max = np.ones(self.nj) * 2  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur10(qq, tcp=tcp, out=out)


class ur10e_spec(robot):
    def __init__(self):
        self.Name = "UR10:"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])  # home joint configuration
        self.q_init = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])  # init work joint configuration
        self.q_max = np.ones(self.nj) * 2 * np.pi  # upper joint limits
        self.q_min = -np.ones(self.nj) * 2 * np.pi  # lower joint limits
        self.qdot_max = np.ones(self.nj) * 2  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur10e(qq, tcp=tcp, out=out)


class ur5_spec(robot):
    def __init__(self):
        self.Name = "UR5:"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])  # home joint configuration
        self.q_init = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])  # init work joint configuration
        self.q_max = np.ones(self.nj) * 2 * np.pi  # upper joint limits
        self.q_min = -np.ones(self.nj) * 2 * np.pi  # lower joint limits
        self.qdot_max = np.ones(self.nj) * 2  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, *q, tcp=None, out="x"):
        if len(q) > 0:
            qq = q[0]
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur10(q, tcp=tcp, out=out)
