"""Definition of robots' parameters and kinematic models.

This module defines several robot models, including the Panda, FR3, LWR, iiwa, and UR series robots.
Each class represents a specific robot, specifying its kinematic properties such as joint limits,
joint velocities, TCP (Tool Center Point), and home configurations. Each robot also provides a
`Kinmodel` function that returns the robot's kinematic model and Jacobian matrix based on the robot's
current joint configuration and TCP.

Included robot specification classes:

- ``panda_spec``: parameters and kinematics for the Panda robot
- ``fr3_spec``: parameters and kinematics for the FR3 robot
- ``lwr_spec``: parameters and kinematics for the LWR robot
- ``iiwa_spec``: parameters and kinematics for the iiwa robot
- ``ur10_spec``: parameters and kinematics for the UR10 robot
- ``ur10e_spec``: parameters and kinematics for the UR10e robot
- ``ur5_spec``: parameters and kinematics for the UR5 robot
- ``ur5e_spec``: parameters and kinematics for the UR5e robot

Each robot specification class typically defines common attributes such as
``Name``, ``nj``, ``TCPGripper``, ``q_home``, ``q_max``, ``q_min``,
``qdot_max``, and ``v_max``, and provides a ``Kinmodel`` method for
forward-kinematics and Jacobian evaluation.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Optional, Tuple, Union

from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, Pose3DType, TCPType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, JacobianType
from robotblockset.robot_models import kinmodel_panda, invkin_model_panda, invkin_model_panda_valid, kinmodel_lwr, kinmodel_iiwa, kinmodel_ur10, kinmodel_ur10e, kinmodel_ur5, kinmodel_ur5e, kinmodel_crx20, kinmodel_hc20, kinmodel_z1
from robotblockset.robots import robot


class panda_spec(robot):
    def __init__(self) -> None:
        self.Name = "panda"
        self.nj = 7
        self.TCPGripper = np.array([[0.7071, 0.7071, 0.0, 0.0], [-0.7071, 0.7071, 0.0, 0.0], [0.0, 0.0, 1, 0.1034], [0.0, 0.0, 0.0, 1.0]])
        self.q_home = np.array([0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816])  # home joint configuration
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])  # upper joint limits
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # lower joint limits
        self.qdot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])  # maximal joint velocities
        self.v_max = np.array([1.7, 1.7, 1.7, 2.5, 2.5, 2.5])  # maximal task velocities
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            qq = self._actual.q
        else:
            qq = q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_panda(qq, tcp=tcp, out=out)

    def IKin_analytical(self, x: Pose3DType, q7: float = np.pi / 4, q_initial: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, valid: bool = True, closest: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Compute the analytical inverse kinematics for the Panda robot.

        Parameters
        ----------
        x : Pose3DType
            End-effector pose.
        q7 : float, optional
            7th joint value for the elbow rotation.
        q_initial : JointConfigurationType, optional
            Initial joint configuration for solution selection.
        tcp : TCPType, optional
            Tool center point transform or pose.
        valid : bool, optional
            Enforce joint limit validity if True.
        closest : bool, optional
            Select the closest valid solution if True.

        Returns
        -------
        tuple
            Joint solutions as arrays.
        """
        if q_initial is None:
            q_initial = self.q_home
        if tcp is None:
            tcp = self.TCP
        if valid:
            return invkin_model_panda_valid(self, x, q7, q_initial, tcp, closest=closest)
        else:
            return invkin_model_panda(self, x, q7, q_initial, tcp, closest=closest)


class fr3_spec(robot):
    def __init__(self) -> None:
        self.Name = "fr3"
        self.nj = 7
        self.TCPGripper = np.array([[0.7071, 0.7071, 0.0, 0.0], [-0.7071, 0.7071, 0.0, 0.0], [0.0, 0.0, 1, 0.1034], [0.0, 0.0, 0.0, 1.0]])
        self.q_home = np.array([0.0, -0.2, 0.0, -1.5, 0.0, 1.5, 0.7854])  # home joint configuration
        self.q_max = np.array([2.8973, 1.8325, 2.8973, -0.1221, 2.8797, 4.6251, 3.0543])  # upper joint limits
        self.q_min = np.array([-2.8973, -1.8325, -2.8973, -3.0717, -2.8797, -0.4363, -3.0543])  # lower joint limits
        self.qdot_max = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 5.26, 5.26])  # maximal joint velocities
        self.v_max = np.array([3, 3, 3, 2.5, 2.5, 2.5])  # maximal task velocities
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            qq = self._actual.q
        else:
            qq = q
        if tcp is None:
            tcp = self.TCP
        return kinmodel_panda(qq, tcp=tcp, out=out)

    def IKin_analytical(self, x: Pose3DType, q7: float, q_initial: JointConfigurationType, tcp: Optional[TCPType] = None, valid: bool = True, closest: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Compute the analytical inverse kinematics for the FR3 robot.

        Parameters
        ----------
        x : Pose3DType
            End-effector pose.
        q7 : float
            7th joint value for the elbow rotation.
        q_initial : JointConfigurationType
            Initial joint configuration for solution selection.
        tcp : TCPType, optional
            Tool center point transform or pose.
        valid : bool, optional
            Enforce joint limit validity if True.
        closest : bool, optional
            Select the closest valid solution if True.

        Returns
        -------
        tuple
            Joint solutions as arrays.
        """
        if tcp is None:
            tcp = self.TCP
        if valid:
            return invkin_model_panda_valid(self, x, q7, q_initial, tcp, closest=closest)
        else:
            return invkin_model_panda(self, x, q7, q_initial, tcp, closest=closest)


class lwr_spec(robot):
    def __init__(self) -> None:
        self.Name = "LWR"
        self.nj = 7
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -0.2, 0.0, 1.3, 0.0, -0.6, 0.0])  # home joint configuration
        self.q_max = np.array([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 170.0]) * np.pi / 180  # upper joint limits
        self.q_min = -np.array([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 170.0]) * np.pi / 180  # lower joint limits
        self.qdot_max = np.array([100.0, 110.0, 100.0, 130.0, 130.0, 180.0, 180.0]) * np.pi / 180  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_lwr(q, tcp=tcp, out=out)


class iiwa_spec(robot):
    def __init__(self) -> None:
        self.Name = "iiwa"
        self.nj = 7
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -0.2, 0.0, -1.7, 0.0, 0.6, 0.0])  # home joint configuration
        self.q_max = np.array([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175]) * np.pi / 180  # upper joint limits
        self.q_min = -np.array([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175]) * np.pi / 180  # lower joint limits
        self.qdot_max = np.array([85, 85, 100.0, 75, 130.0, 135, 135]) * np.pi / 180  # maximal joint velocities (for AUT mode)
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities
        self.joint_names = [
            "lbr_A1",
            "lbr_A2",
            "lbr_A3",
            "lbr_A4",
            "lbr_A5",
            "lbr_A6",
            "lbr_A7",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_iiwa(q, tcp=tcp, out=out)


class ur10_spec(robot):
    def __init__(self) -> None:
        self.Name = "ur10"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0, +np.pi / 2, 0.0])  # init work joint configuration
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
        self.actuator_names = [
            "shoulder_pan_actuator",
            "shoulder_lift_actuator",
            "elbow_actuator",
            "wrist_1_actuator",
            "wrist_2_actuator",
            "wrist_3_actuator",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur10(q, tcp=tcp, out=out)


class ur10e_spec(robot):
    def __init__(self) -> None:
        self.Name = "ur10e"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0, +np.pi / 2, 0.0])  # init work joint configuration
        self.q_max = np.ones(self.nj) * 2 * np.pi  # upper joint limits
        self.q_min = -np.ones(self.nj) * 2 * np.pi  # lower joint limits
        self.qdot_max = np.array([120, 120, 180, 180, 180, 180]) * np.pi / 180.0  # maximal joint velocities
        self.v_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # maximal task velocities
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.actuator_names = [
            "shoulder_pan_actuator",
            "shoulder_lift_actuator",
            "elbow_actuator",
            "wrist_1_actuator",
            "wrist_2_actuator",
            "wrist_3_actuator",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur10e(q, tcp=tcp, out=out)


class ur5_spec(robot):
    def __init__(self) -> None:
        self.Name = "ur5"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])  # init work joint configuration
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
        self.actuator_names = [
            "shoulder_pan_actuator",
            "shoulder_lift_actuator",
            "elbow_actuator",
            "wrist_1_actuator",
            "wrist_2_actuator",
            "wrist_3_actuator",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur5(q, tcp=tcp, out=out)


class ur5e_spec(robot):
    def __init__(self) -> None:
        self.Name = "ur5e"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])  # init work joint configuration
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
        self.actuator_names = [
            "shoulder_pan_actuator",
            "shoulder_lift_actuator",
            "elbow_actuator",
            "wrist_1_actuator",
            "wrist_2_actuator",
            "wrist_3_actuator",
        ]

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_ur5e(q, tcp=tcp, out=out)


class crx20_spec(robot):
    def __init__(self) -> None:
        self.Name = "CRX20"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # init work joint configuration
        self.q_max = np.ones(self.nj) * np.pi  # upper joint limits
        self.q_min = -np.ones(self.nj) * np.pi  # lower joint limits
        self.qdot_max = np.ones(self.nj) * 2  # maximal joint velocities
        self.v_max = np.array([1.5, 1.5, 1.5, 2, 2, 2])  # maximal task velocities

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : TCPType, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_crx20(q, tcp=tcp, out=out)


class hc20_spec(robot):
    def __init__(self) -> None:
        self.Name = "hc20"
        self.nj = 6
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # home joint configuration
        self.q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # init work joint configuration
        self.q_max = np.array([np.pi, np.pi, 4.31096, 3.66519, np.pi, 3.66519])  # upper joint limits
        self.q_min = np.array([-np.pi, -np.pi, -1.6937, -3.66519, -np.pi, -3.66519])  # lower joint limits
        self.qdot_max = np.array([0.2, 0.15, 0.2, 2.0, 0.8, 2.0])  # maximal joint velocities
        self.v_max = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5])  # maximal task velocities
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

    def Kinmodel(self, q: Optional[ArrayLike] = None, tcp: Optional[ArrayLike] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : ArrayLike, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : ArrayLike, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_hc20(q, tcp=tcp, out=out)


class z1_spec(robot):
    def __init__(self) -> None:
        self.Name = "z1"
        self.nj = 6
        self.TCPGripper = np.array([[1.0, 0.0, 0.0, 0.0085], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1, 0.2], [0.0, 0.0, 0.0, 1.0]])
        self.q_home = np.array([0, 0.785, -0.261, -0.523, 0, 0])  # home joint configuration
        self.q_init = np.array([0, 0.785, -0.261, -0.523, 0, 0])  # init work joint configuration
        self.q_max = np.array([2.61799, 2.96706, 0, 1.51844, 1.3439, 2.79253])  # upper joint limits
        self.q_min = np.array([-2.61799, 0, -2.87979, -1.51844, -1.3439, -2.79253])  # lower joint limits
        self.qdot_max = np.array([np.pi] * 6)  # maximal joint velocities
        self.v_max = np.array([1, 1, 1, 0.5, 0.5, 0.5])  # maximal task velocities
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

    def Kinmodel(self, q: Optional[ArrayLike] = None, tcp: Optional[ArrayLike] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : ArrayLike, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : ArrayLike, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        return kinmodel_z1(q, tcp=tcp, out=out)


class b2_spec(robot):
    def __init__(self) -> None:
        self.Name = "b2"
        self.nj = 12
        self.TCPGripper = np.eye(4)
        self.q_home = np.array([0.0522, 0.27045, -0.45, -0.133625, -0.06735, -0.45, 0.087, 0.240057, -0.709667, 0.0261, 0.4675, -0.459406])  # home joint configuration
        self.q_init = np.array([0.0522, 0.27045, -0.45, -0.133625, -0.06735, -0.45, 0.087, 0.240057, -0.709667, 0.0261, 0.4675, -0.459406])  # init work joint configuration
        self.q_max = np.array([0.8699999, 4.6899998, -0.43, 0.8699999, 4.6899998, -0.43, 0.8699999, 4.6899998, -0.43, 0.8699999, 4.6899998, -0.43])  # upper joint limits
        self.q_min = np.array([-0.8699999, -0.9399999, -2.8199997, -0.8699999, -0.9399999, -2.8199997, -0.8699999, -0.9399999, -2.8199997, -0.8699999, -0.9399999, -2.8199997])  # lower joint limits
        self.qdot_max = np.array([1] * 12)  # maximal joint velocities
        self.v_max = np.array([1, 1, 1, 0.5, 0.5, 0.5])  # maximal task velocities
        self.joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        self.actuator_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]

    def Kinmodel(self, q: Optional[ArrayLike] = None, tcp: Optional[ArrayLike] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute forward kinematics and Jacobian for the robot.

        Parameters
        ----------
        q : ArrayLike, optional
            Joint angles/positions. Uses the current joint state when None.
        tcp : ArrayLike, optional
            Tool center point transform or pose. Uses the current TCP when None.
        out : str, optional
            Output form (e.g., "x", "pR").

        Returns
        -------
        tuple
            Pose representation and JacobianType depending on `out`.
        """
        # self.WarningMessage("Kinmodel not implemented!")
        return np.array([0.0, 0, 0, 1, 0, 0, 0]), np.zeros((6, self.nj))
