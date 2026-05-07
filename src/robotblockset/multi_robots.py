"""Multi-robot system support.

This module defines a `robot` class that represents a multi robotic system with various configurations, transformations, and control mechanisms.
It provides methods for controlling the robot's movement, updating its state, handling the tool center point (TCP), managing sensors and
grippers, as well as various kinematic and control strategies. The class includes support for joint and task space motion, force/torque
sensor integration, and multiple motion control strategies, allowing for flexible and complex robot operations.

Key functionalities include:
- Motion control in joint and task space (Cartesian, Object, Robot, World).
- Force/torque sensor handling.
- Gripper attachment and detachment.
- Asynchronous and synchronous motion control.
- Robot state and pose management.
- Path planning with advanced kinematic computations.
- Trajectory and null-space control.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Optional, Tuple, Any, Union
import numpy as np
from time import sleep
import copy

from robotblockset.robots import robot, MotionResultCodes, CommandModeCodes
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, JointVelocityType, JointTorqueType, TimesType, Pose3DType, Poses3DType, Velocity3DType, Velocities3DType, HomogeneousMatrixType, HomogeneousMatricesType, RotationMatrixType, Vector3DType, TCPType, JacobianType
from robotblockset.tools import load_params, rbs_type, check_option, check_shape, vector, isscalar, isvector, ismatrix, grad, normalize, damped_pinv
from robotblockset.trajectories import ctraj, carctraj, interpPath, interpCartesianPath, gradientPath, gradientCartesianPath, uniqueCartesianPath, slerp
from robotblockset.transformations import map_pose, q2r, r2q, x2x, x2t, t2x, v2s, xerr, qerr, world2frame

np.set_printoptions(formatter={"float": "{: 0.4f}".format})


def join_robot(p1: Vector3DType, R1: RotationMatrixType, J1: JacobianType, p2: Vector3DType, R2: RotationMatrixType, J2: JacobianType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Join two robot kinematic chains: base_1 -> EE_1 -> base_2 -> EE_2.

    Parameters
    ----------
    p1 : Vector3DType
        Position vector of robot 1 end-effector (shape: (3,)).
    R1 : RotationMatrixType
        Rotation matrix of robot 1 end-effector (shape: (3, 3)).
    J1 : JacobianType
        Jacobian matrix of robot 1 (shape: (6, n1)).

    p2 : Vector3DType
        Position vector of robot 2 end-effector (shape: (3,)).
    R2 : RotationMatrixType
        Rotation matrix of robot 2 end-effector (shape: (3, 3)).
    J2 : JacobianType
        Jacobian matrix of robot 2 (shape: (6, n2)).

    Returns
    -------
    x : Vector3DType
        Combined position vector (shape: (3,)).
    R : RotationMatrixType
        Combined rotation matrix (shape: (3, 3)).
    J : JacobianType
        Combined Jacobian matrix (shape: (6, n1 + n2)).
    """
    Jp1 = J1[0:3, :]
    Jr1 = J1[3:6, :]
    Jp2 = J2[0:3, :]
    Jr2 = J2[3:6, :]

    Jp = np.hstack([Jp1 + v2s(R1 @ p2).T @ Jr1, R1 @ Jp2])
    Jr = np.hstack([Jr1, R1 @ Jr2])

    x = p1 + R1 @ p2
    R = R1 @ R2
    J = np.vstack([Jp, Jr])

    return x, R, J


def join_fixed_robot(p1: Vector3DType, R1: RotationMatrixType, p2: Vector3DType, R2: RotationMatrixType, J2: JacobianType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Join a fixed base with a robot: fixed base_1 -> fxed EE_1 -> base_2 -> EE_2.

    Parameters
    ----------
    p1 : Vector3DType
        Fixed robot position (shape: (3,)).
    R1 : RotationMatrixType
        fixed robot orientation (shape: (3, 3)).
    p2 : Vector3DType
        Position of robot 2 relative to its base (shape: (3,)).
    R2 : RotationMatrixType
        Rotation matrix of robot 2 (shape: (3, 3)).
    J2 : JacobianType
        Jacobian of robot 2 (shape: (6, n)).

    Returns
    -------
    x : Vector3DType
        Combined position (shape: (3,)).
    R : RotationMatrixType
        Combined orientation (shape: (3, 3)).
    J : JacobianType
        Combined Jacobian (shape: (6, n)).
    """
    Jp2 = J2[0:3, :]
    Jr2 = J2[3:6, :]

    Jp = R1 @ Jp2
    Jr = R1 @ Jr2

    x = p1 + R1 @ p2
    R = R1 @ R2
    J = np.vstack([Jp, Jr])

    return x, R, J


def join_reverse_robot(p1: Vector3DType, R1: RotationMatrixType, J1: JacobianType, p2: Vector3DType, R2: RotationMatrixType, J2: JacobianType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Join two robots with the structure: EE_1 -> base_1 ->  base_2 -> EE_2.

    Parameters
    ----------
    p1 : Vector3DType
        Position vector of robot 1 end-effector (shape: (3,)).
    R1 : RotationMatrixType
        Rotation matrix of robot 1 end-effector (shape: (3, 3)).
    J1 : JacobianType
        Jacobian matrix of robot 1 (shape: (6, n1)).

    p2 : Vector3DType
        Position vector of robot 2 end-effector (shape: (3,)).
    R2 : RotationMatrixType
        Rotation matrix of robot 2 end-effector (shape: (3, 3)).
    J2 : JacobianType
        Jacobian matrix of robot 2 (shape: (6, n2)).

    Returns
    -------
    x : Vector3DType
        Combined position vector (shape: (3,)).
    R : RotationMatrixType
        Combined orientation matrix (shape: (3, 3)).
    J : JacobianType
        Combined Jacobian matrix (shape: (6, n1 + n2)).
    """
    Jp1 = J1[0:3, :]
    Jr1 = J1[3:6, :]
    Jp2 = J2[0:3, :]
    Jr2 = J2[3:6, :]

    delta_x = -p1 + p2
    Jp = R1.T @ np.hstack([-Jp1 + v2s(delta_x) @ Jr1, Jp2])
    Jr = R1.T @ np.hstack([-Jr1, Jr2])

    x = R1.T @ delta_x
    R = R1.T @ R2
    J = np.vstack([Jp, Jr])

    return x, R, J


def join_robot_fixed(p1: Vector3DType, R1: RotationMatrixType, J1: JacobianType, p2: Vector3DType, R2: RotationMatrixType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Join a robot to a fixed structure: base_1 -> EE_1 > fixed base -> fixed EE.

    Parameters
    ----------
    p1 : Vector3DType
        Position of robot 1 end-effector (shape: (3,)).
    R1 : RotationMatrixType
        Orientation matrix of robot 1 (shape: (3, 3)).
    J1 : JacobianType
        Jacobian of robot 1 (shape: (6, n1)).
    p2 : Vector3DType
        Fixed robot position (shape: (3,)).
    R2 : RotationMatrixType
        Fixed robot rotation (shape: (3, 3)).

    Returns
    -------
    x : Vector3DType
        Combined position (shape: (3,)).
    R : RotationMatrixType
        Combined orientation (shape: (3, 3)).
    J : JacobianType
        Combined Jacobian (shape: (6, n1)).
    """
    Jp1 = J1[0:3, :]
    Jr1 = J1[3:6, :]

    Jp = Jp1 + v2s(R1 @ p2).T @ Jr1
    Jr = Jr1

    x = p1 + R1 @ p2
    R = R1 @ R2
    J = np.vstack([Jp, Jr])

    return x, R, J


def join_robot_reverse(p1: Vector3DType, R1: RotationMatrixType, J1: JacobianType, p2: Vector3DType, R2: RotationMatrixType, J2: JacobianType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Join two robots in reverse: base_1 -> EE_1 -> EE_2 -> base_2.

    Parameters
    ----------
    p1 : Vector3DType
        Position vector of robot 1 end-effector (shape: (3,)).
    R1 : RotationMatrixType
        Rotation matrix of robot 1 end-effector (shape: (3, 3)).
    J1 : JacobianType
        Jacobian matrix of robot 1 (shape: (6, n1)).

    p2 : Vector3DType
        Position vector of robot 2 end-effector (shape: (3,)).
    R2 : RotationMatrixType
        Rotation matrix of robot 2 end-effector (shape: (3, 3)).
    J2 : JacobianType
        Jacobian matrix of robot 2 (shape: (6, n2)).

    Returns
    -------
    x : Vector3DType
        Combined position (shape: (3,)).
    R : RotationMatrixType
        Combined orientation (shape: (3, 3)).
    J : JacobianType
        Combined Jacobian (shape: (6, n1 + n2)).
    """
    Jp1 = J1[0:3, :]
    Jr1 = J1[3:6, :]
    Jp2 = J2[0:3, :]
    Jr2 = J2[3:6, :]

    R2_inv = R2.T
    p2 = -R2_inv @ p2
    Jp2 = -R2_inv @ Jp2
    Jr2 = -R2_inv @ Jr2

    Jp = np.hstack([Jp1 + v2s(R1 @ p2).T @ Jr1, R1 @ Jp2 + R1 @ v2s(p2).T @ Jr2])
    Jr = np.hstack([Jr1, R1 @ Jr2])

    x = p1 + R1 @ p2
    R = R1 @ R2_inv
    J = np.vstack([Jp, Jr])

    return x, R, J


def robot_reverse(p1: Vector3DType, R1: RotationMatrixType, J1: JacobianType) -> Tuple[Vector3DType, RotationMatrixType, JacobianType]:
    """
    Compute the reverse kinematics of a robot (from end-effector to base).

    Parameters
    ----------
    p1 : Vector3DType
        Position of the robot's end-effector (shape: (3,)).
    R1 : RotationMatrixType
        Rotation matrix of the robot's end-effector (shape: (3, 3)).
    J1 : JacobianType
        Jacobian matrix of the robot (shape: (6, n)).

    Returns
    -------
    x : Vector3DType
        Reversed base position (shape: (3,)).
    R : RotationMatrixType
        Reversed orientation matrix (shape: (3, 3)).
    Jc : JacobianType
        Reversed Jacobian matrix (shape: (6, n)).
    """
    Jp1 = J1[0:3, :]
    Jr1 = J1[3:6, :]

    R = R1.T
    x = -R @ p1
    Jp = -R @ Jp1 + R @ v2s(p1).T @ Jr1
    Jr = -R @ Jr1
    Jc = np.vstack([Jp, Jr])

    return x, R, Jc


class multi_robots(robot):
    """
    Represents a robot master class with various configurations, transformations, and control mechanisms.

    Attributes
    ----------
    Name : str
        Name of the robot.
    tsamp : float
        Sampling rate for the robot.
    TCP : np.ndarray
        Transformation matrix for the robot's Tool Center Point (TCP).
    TBase : np.ndarray
        Robot's base pose transformation matrix (4, 4).
    vBase : np.ndarray
        Robot's base velocity twist (6,).
    TObject : np.ndarray
        Transformation matrix for the object the robot manipulates.
    TCPGripper : np.ndarray
        Transformation matrix for the robot's gripper TCP.
    Load : load_params
        Load associated with the robot.
    Gripper : Optional[Any]
        Gripper object attached to the robot, if any.
    FTSensor : Optional[Any]
        Force/Torque sensor attached to the robot, if any.
    FTSensorFrame : np.ndarray
        Transformation matrix of the F/T sensor frame relative to the end-effector.
    Platform : Optional[Any]
        Platform object to which the robot is attached.
    User : Optional[Any]
        User-defined data or object.
    Tag : Optional[Any]
        Tag associated with the robot.
    """

    def __init__(self, robots: Tuple[robot, ...], robot_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initializes combined robot system with default values and optional configurations.

        Parameters
        ----------
        robots : Tuple[robot, ...]
            Tuple[robot, ...]
            Robots in the combined robot system.
        robot_name : str, optional
            Name of the combined robot.
        **kwargs : Any
            Additional keyword arguments for custom configuration or parameters.
        """
        # Initialize parent class
        robot.__init__(self, **kwargs)

        # Definition of combined robot
        if robot_name is None:
            self.Name: str = "MultiRobot"
        else:
            self.Name = robot_name
        self.robots = tuple(robots)  # Robots in combined robot system
        self.nr = len(robots)  # Number of robots in combined robot system

        self._default.TaskDOF = np.array([1] * (6 * self.nr))
        self._default.AddedFT = np.zeros((self.nr, 6))

        self.nj = 0
        _tmpload = []
        for r in robots:
            self.nj += r.nj
            _tmpload.append(load_params())

        self.q_home = np.concatenate([r.q_home for r in robots])  # home joint configuration
        self.q_max = np.concatenate([r.q_max for r in robots])  # upper joint limits
        self.q_min = np.concatenate([r.q_min for r in robots])  # lower joint limits
        self.q_home = np.concatenate([r.q_home for r in robots])  # home joint configuration
        self.qdot_max = np.concatenate([r.qdot_max for r in robots])  # maximal joint velocities
        self.v_max = np.stack([r.v_max for r in robots])  # Maximal task velocity

        self.TCP: np.ndarray = np.tile(np.eye(4), (self.nr, 1, 1))  # Tool Center Point transformation matrix
        self.TCP: np.ndarray = np.tile(np.eye(4), (self.nr, 1, 1))  # Tool Center Point transformation matrix
        self.TBase: np.array = np.tile(np.eye(4), (self.nr, 1, 1))  # Robot base
        self.vBase: np.ndarray = np.zeros((self.nr, 6))  # Robot base velocity
        self.TObject: np.ndarray = np.tile(np.eye(4), (self.nr, 1, 1))  # Object transformation matrix
        self.TCPGripper: np.ndarray = np.tile(np.eye(4), (self.nr, 1, 1))  # Gripper TCP transformation matrix
        self.Load: load_params = _tmpload  # Load object
        self.Gripper: Optional[Any] = [None] * self.nr  # Gripper object attached to robot
        self.FTSensor: Optional[Any] = [None] * self.nr  # Force/Torque sensor attached to robot
        self.FTSensorFrame: np.ndarray = np.tile(np.eye(4), (self.nr, 1, 1))  # F/T sensor transformation matrix

        self.tsamp: float = 0.01  # Sampling rate for the robot
        self.Init()

    def spatial(self, x: ArrayLike, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Validates the shape of the input `x` and returns it in an appropriate format.

        Parameters
        ----------
        x : ArrayLike
            one of the following shapes:
            - (7,) : pose (position and quaternion)
            - (4, 4) : transformation matrix
            - (3,) : position vector
            - (4,) : quaternion
            - (3, 3) : rotation matrix
            - (6,) : twist (linear and angular velocity)
            - (3, 4) : homogeneous matrix without the last row (assumed to be 3x4)

        Returns
        -------
        np.ndarray
            The input `x` in the validated shape, possibly modified if the shape was (3, 4).

        Raises
        ------
        TypeError
            If the input `x` does not have a valid shape.
        """
        x = rbs_type(x)

        # Check for valid shapes
        if shape is None:
            if x.shape == (self.nr, 7) or x.shape == (self.nr, 4, 4) or x.shape == (self.nr, 3) or x.shape == (self.nr, 4) or x.shape == (self.nr, 3, 3) or x.shape == (self.nr, 6):
                return x
            elif x.shape == (self.nr, 3, 4):
                _tmp = np.array([0, 0, 0, 1])
                _tmp_expanded = np.tile(_tmp, (self.nr, 1)).reshape(self.nr, 1, 4)
                x = np.concatenate((x, _tmp_expanded), axis=1)
                return x
            else:
                raise TypeError("Parameter has not proper shape")
        else:
            if x.shape == (self.nr, *shape):
                return x
            else:
                raise TypeError("Parameter has not proper shape")

    # Initialization and update
    def InitObject(self) -> None:
        """
        Initializes the robot's command and actual state with zeros, and sets default values for various attributes.

        This method sets the initial values for joint positions, joint velocities, joint torques, end-effector pose,
        velocities, force/torque sensor data, control inputs, and other state variables.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot object.
        """
        self._command.q = np.zeros(self.nj)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.u = np.zeros(self.nj)
        self._command.ux = np.zeros((self.nr, 6))
        self._command.x = np.zeros((self.nr, 7))
        self._command.rx = np.zeros((self.nr, 7))
        self._command.v = np.zeros((self.nr, 6))
        self._command.rv = np.zeros((self.nr, 6))
        self._command.FT = np.zeros((self.nr, 6))
        self._command.data = None
        self._command.mode = CommandModeCodes.STOP.value

        self._actual.q = np.zeros(self.nj)
        self._actual.qdot = np.zeros(self.nj)
        self._actual.trq = np.zeros(self.nj)
        self._actual.x = np.zeros((self.nr, 7))
        self._actual.v = np.zeros((self.nr, 6))
        self._actual.FT = np.zeros((self.nr, 6))
        self._actual.trqExt = np.zeros(self.nj)

        self._default.AddedTrq = np.zeros(self.nj)

    def GetState(self) -> None:
        """
        Update and synchronize the internal state of the combined robot.

        This method updates the joint positions, velocities, forces/torques, task space position and velocity,
        and the base pose of the dual robot system. It synchronizes the state of all robots and computes the
        combined state. The method handles the relative force computation and updates the base pose
        and velocity for the system.

        The state synchronization occurs if the time since the last update exceeds a certain threshold, determined
        by the sampling rate (`tsamp`).

        Attributes Updated:
        - Joint positions (`self._actual.q`)
        - Joint velocities (`self._actual.qdot`)
        - Joint torques (`self._actual.trq`)
        - Task space position (`self._actual.x`)
        - Task space velocity (`self._actual.v`)
        - Force/Torque sensor data (`self._actual.FT`)

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot object.

        Notes
        -----
        This method sets the following attributes:
        - `_tt`: The current time, can be retrieved using `simtime()`.
        - `_last_update`: The last update time, retrieved using `simtime()`.
        """
        if (self.simtime() - self._last_update) > (self.tsamp * 0.1):
            self._tt = self.simtime()
            for r in self.robots:
                r.GetState()
            self._actual.q = np.concatenate([r._actual.q for r in self.robots])
            self._actual.qdot = np.concatenate([r._actual.qdot for r in self.robots])
            self._actual.trq = np.concatenate([r._actual.trq for r in self.robots])
            self._actual.x = np.vstack([r._actual.x for r in self.robots])
            self._actual.v = np.vstack([r._actual.v for r in self.robots])
            self._actual.FT = np.vstack([r._actual.FT for r in self.robots])

            self._last_update = self.simtime()  # Do not change !

    def ResetCurrentTarget(self, do_move: bool = False, **kwargs: Any) -> int:
        """
        Resets the current target for the robot and optionally moves the robot to the actual configuration.

        This method updates the robot's commanded joint positions, velocities, torques, and end-effector pose to the current actual values.
        It also has an optional move functionality that commands the robot to move to the updated target.

        Parameters
        ----------
        do_move : bool, optional
            If `True`, the robot will move to the actual configuration after resetting the target. Default is `False`.
        **kwargs : Any
            Additional keyword arguments passed to the `GoTo_T` method when `do_move` is `True`.

        Returns
        -------
        int
            A status code:
            - `0` if no movement is performed (i.e., `do_move=False`),
            - `88` if the movement was not executed due to active threads,
            - The result of the `GoTo_q` or `GoTo_T` function otherwise.

        Notes
        -----
        This method modifies the `_command` and `_actual` states of the robot.
        """
        self.GetState()
        self._command.q = copy.deepcopy(self._actual.q)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.rx = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        self._command.rv = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.trq = np.zeros(self.nj)
        sleep(0.1)
        self.Update()
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        if do_move:
            self.Message("Moving to actual configuration", 2)
            if not self.Start():
                return MotionResultCodes.NOT_READY.value

            if self._semaphore._value <= 0:
                self.Message("Not executed due to active threads!")
                return MotionResultCodes.CLOSE_TO_TARGET.value

            self._semaphore.acquire()

            if self._control_strategy.startswith("Joint"):
                self._last_status = self.GoTo_q(self._command.q, np.zeros(self.nj), np.zeros(self.nj), 1, **kwargs)
            else:
                self._last_status = self.GoTo_T(self._command.x, wait=1, **kwargs)

            self.Stop()
            self._semaphore.release()
            return self._last_status
        else:
            return MotionResultCodes.MOTION_SUCCESS.value

    def ResetTaskTarget(self) -> None:
        """
        Resets the task target for the robot based on the current commanded joint positions.

        This method updates the robot's commanded end-effector pose (`_command.x`) and velocity (`_command.v`)
        based on the robot's current joint configuration.

        Returns
        -------
        None
            This method does not return any value but modifies the `_command.x` and `_command.v` attributes.
        """
        _x, _J = self.Kinmodel(self._command.q)
        self._command.x = _x
        self._command.v = (_J @ self._command.qdot).reshape(self.nr, 6)

    def isConnected(self) -> bool:
        """
        Checks if the robot is connected.

        Returns
        -------
        bool
            True if the robot is connected, False otherwise.
        """
        _tmp = True
        for r in self.robots:
            _tmp = _tmp and r.isConnected()
        return _tmp

    def isReady(self) -> bool:
        """
        Check if the robot is ready for operations.

        This method checks the `_connected` attribute to determine if the robot is connected
        and operational.

        Returns
        -------
        bool
            `True` if the robot is connected and ready for operations, otherwise `False`.
        """
        _tmp = True
        for r in self.robots:
            _tmp = _tmp and r.isReady()
        return _tmp

    def isActive(self) -> bool:
        """
        Check if the robot target is active.

        Returns
        -------
        bool
            Indicating if the robot target is active.
        """
        _tmp = True
        for r in self.robots:
            _tmp = _tmp and r.isActive()
        return _tmp

    def inMotion(self) -> bool:
        """
        Check if the robot is in motion.

        Returns
        -------
        bool
            `True` indicating the robot is excetuting motion command.
        """
        _tmp = True
        for r in self.robots:
            _tmp = _tmp and r.inMotion()
        return _tmp

    # Get task space variables
    def GetPose(self, out: str = None, task_space: str = None, kinematics: str = None, state: str = None, refresh: bool = None) -> np.ndarray:
        """
        Get the robot's end-effector pose.

        Parameters
        ----------
        out : str, optional
            Output form for the pose. The default is "x" ("Pose").
        task_space : str, optional
            Task space frame to use. Options are "World", "Object", and "Robot". The default is "World".
        kinematics : str, optional
            The kinematics to use. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot to use. Options are "Actual" or "Commanded". The default is "Actual".
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the pose. Default is `True`.

        Returns
        -------
        np.ndarray
            The end-effector pose in the specified form, shape varies depending on `out` value.

        Raises
        ------
        ValueError
            If the `state`, `kinematics`, or `task_space` options are invalid.
        """
        if out is None:
            out = self._default.TaskPoseForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        if refresh is None:
            refresh = self._default.Refresh

        if refresh:
            self.GetState()

        if check_option(kinematics, "Calculated"):
            _x, _ = self.Kinmodel(self.GetJointPos(state=state))
        elif check_option(kinematics, "Robot"):
            if check_option(state, "Actual"):
                _x = copy.deepcopy(self._actual.x)
            elif check_option(state, "Commanded"):
                _x = copy.deepcopy(self._command.x)
            else:
                raise ValueError(f"State '{state}' not supported in GetPose")
        else:
            raise ValueError(f"Kinematics calculation '{kinematics}' not supported in GetPose")

        if check_option(task_space, "World"):
            _x = self.BaseToWorld(_x)
        elif check_option(task_space, "Object"):
            _x = self.BaseToWorld(_x)
            _x = self.WorldToObject(_x)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")

        return map_pose(x=_x, out=out)

    def GetPos(self, out: str = "p", task_space: str = None, kinematics: str = None, state: str = None) -> np.ndarray:
        """
        Get the robot's end-effector position.

        Parameters
        ----------
        out : str, optional
            Output form for the position. The default is "p" ("Position").
        task_space : str, optional
            Task space frame to use. Options are "World", "Object", and "Robot". The default is "World".
        kinematics : str, optional
            The kinematics to use. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot to use. Options are "Actual" or "Commanded". The default is "Actual".

        Returns
        -------
        np.ndarray
            The end-effector position (3,).

        Raises
        ------
        ValueError
            If the `out` form is not "Position" or "p".
        """
        if out in ["Position", "p"]:
            return self.GetPose(out=out, task_space=task_space, kinematics=kinematics, state=state)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetPos")

    def GetOri(self, out: str = "Q", task_space: str = None, kinematics: str = None, state: str = None) -> np.ndarray:
        """
        Get the robot's end-effector orientation.

        Parameters
        ----------
        out : str, optional
            Output form for the orientation. Options are "Quaternion", "Q", "RotationMatrix", "R". The default is "Q" ("Quaternion").
        task_space : str, optional
            Task space frame to use. Options are "World", "Object", and "Robot". The default is "World".
        kinematics : str, optional
            The kinematics to use. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot to use. Options are "Actual" or "Commanded". The default is "Actual".

        Returns
        -------
        np.ndarray
            The end-effector orientation, either in quaternion (4,) or rotation matrix (3, 3) format.

        Raises
        ------
        ValueError
            If the `out` form is not "Quaternion", "Q", "RotationMatrix", or "R".
        """
        if out in ["Quaternion", "Q", "RotationMatrix", "R"]:
            return self.GetPose(out=out, task_space=task_space, kinematics=kinematics, state=state)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetOri")

    def GetVel(self, out: str = None, task_space: str = None, kinematics: str = None, state: str = None, refresh: bool = None) -> np.ndarray:
        """
        Get robot end-effector velocity.

        Parameters
        ----------
        out : str, optional
            Output form for the velocity. Options are "Twist" (default), "Linear", or "Angular".
        task_space : str, optional
            Task space frame for the velocity. Options are "World", "Object", or "Robot". The default is "World".
        kinematics : str, optional
            The kinematics used for calculation. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot for the calculation. Options are "Actual" or "Commanded". The default is "Actual".
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the velocity. Default is `True`.

        Returns
        -------
        np.ndarray
            The end-effector velocity in the specified output form.
            The shape is either (6,) for the full twist or (3,) for linear or angular components.

        Raises
        ------
        ValueError
            If the `out`, `task_space`, or `state` values are not supported.
        """
        if out is None:
            out = self._default.TaskVelForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        if refresh is None:
            refresh = self._default.Refresh

        if refresh:
            self.GetState()

        if check_option(kinematics, "Calculated"):
            if check_option(state, "Actual") or check_option(state, "Commanded"):
                _qq = self.GetJointPos(state=state)
                _qqdot = self.GetJointVel(state=state)
            else:
                raise ValueError(f"State '{state}' not supported")
            _J = self.Jacobi(_qq)

            if check_option(task_space, "World"):
                _vv = self.BaseToWorld(_J @ _qqdot, typ="Twist")
            elif check_option(task_space, "Object"):
                _vv = self.BaseToWorld(_J @ _qqdot, typ="Twist")
                _vv = self.WorldToObject(_vv)
            elif check_option(task_space, "Robot"):
                _vv = _J @ _qqdot
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        elif check_option(kinematics, "Robot"):
            if check_option(state, "Actual"):
                _vv = copy.deepcopy(self._actual.v)
            elif check_option(state, "Commanded"):
                _vv = copy.deepcopy(self._command.v)
            else:
                raise ValueError(f"State '{state}' not supported")

            if check_option(task_space, "World"):
                _vv = self.BaseToWorld(_vv, typ="Twist")
            elif check_option(task_space, "Object"):
                _vv = self.BaseToWorld(_vv, typ="Twist")
                _vv = self.WorldToObject(_vv)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        else:
            raise ValueError(f"Kinematics calculation '{kinematics}' not supported")

        if check_option(out, "Twist"):
            return _vv
        elif check_option(out, "Linear"):
            return _vv[:3]
        elif check_option(out, "Angular"):
            return _vv[3:]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    def GetFT(self, out: str = None, source: str = None, task_space: str = None, kinematics: str = None, state: str = None, avg_time: int = 0, refresh: bool = None) -> np.ndarray:
        """
        Get force/torque sensor data for the robot.

        Parameters
        ----------
        out : str, optional
            Output form for the force/torque data. Options are "Wrench" (default), "Force", or "Torque".
        source : str, optional
            Source of the force/torque data. Options are "External" or "Robot". The default is "Robot".
        task_space : str, optional
            Task space frame for the force/torque data. Options are "World", "Object", "Robot", or "Tool". The default is "World".
        kinematics : str, optional
            The kinematics used for the calculation. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot for the calculation. Options are "Actual" or "Commanded". The default is "Actual".
        avg_time : int, optional
            Average time for the external force/torque sensor, by default 0.
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the force/torque data. Default is `True`.

        Returns
        -------
        np.ndarray
            The force/torque data in the specified output form. The shape varies depending on the `out` option.

        Raises
        ------
        ValueError
            If the `source`, `task_space`, or `out` values are not supported.
        """
        if out is None:
            out = self._default.TaskFTForm
        if source is None:
            source = self._default.Source
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        if refresh is None:
            refresh = self._default.Refresh

        if refresh:
            self.GetState()

        if check_option(state, "Actual"):
            _R = q2r(self._actual.x[3:])
            if check_option(source, "External"):
                if self.FTSensor:
                    _FT = self.FTSensor.GetFT(avg_time=avg_time)
                    _FT2TCP = np.linalg.pinv(self.FTSensorFrame) @ self.TCP
                    Rsensor = self.R @ _FT2TCP[:3, :3].T
                    _FT -= -(-9.81 * self.FTSensor.Load.mass * np.hstack((Rsensor[2, :], v2s(self.FTSensor.Load.COM) @ Rsensor[2, :])))
                    _FT = world2frame(_FT, _FT2TCP, typ="Wrench")
                else:
                    raise ValueError("No FT sensor assigned to robot")
            elif check_option(source, "Robot"):
                if check_option(kinematics, "Robot"):
                    _FT = self._actual.FT
                elif check_option(kinematics, "Calculated"):
                    _J = self.Jacobi()
                    _FT = np.linalg.pinv(_J.T) @ self._actual.trqExt
                    _FT = np.hstack((_R.T @ _FT[:3], _R.T @ _FT[3:]))
                else:
                    raise ValueError(f"Kinematics calculation '{kinematics}' not supported")
            else:
                raise ValueError(f"Source '{source}' not supported")

            if check_option(task_space, "World"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))
                _FT = self.BaseToWorld(_FT, typ="Wrench")
            elif check_option(task_space, "Object"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))
                _FT = self.BaseToWorld(_FT, typ="Wrench")
                _FT = self.WorldToObject(_FT, typ="Wrench")
            elif check_option(task_space, "Robot"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))
            elif check_option(task_space, "Tool"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        elif check_option(state, "Commanded"):
            _FT = copy.deepcopy(self._command.FT)
            if check_option(task_space, "World"):
                _FT = self.BaseToWorld(_FT, typ="Wrench")
            elif check_option(task_space, "Object"):
                _FT = self.BaseToWorld(_FT, typ="Wrench")
                _FT = self.WorldToObject(_FT, typ="Wrench")
            elif check_option(task_space, "Robot"):
                pass
            elif check_option(task_space, "Tool"):
                _FT = np.hstack((_R.T @ _FT[:3], _R.T @ _FT[3:]))
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        else:
            raise ValueError(f"State '{state}' not supported")

        if check_option(out, "Wrench"):
            return _FT
        elif check_option(out, "Force"):
            return _FT[:3]
        elif check_option(out, "Torque"):
            return _FT[3:]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    # Joint space motion
    def GoTo_q(self, q: JointConfigurationType, qdot: JointVelocityType, trq: JointTorqueType, wait: float, **kwargs: Any) -> int:
        """
        Abstract method to command robots in combined to go to a specific joint configuration.

        It has to be reimplemented in actual robot class!

        This method sets the commanded joint positions (`q`), velocities (`qdot`), and torques (`trq`),
        then sends them to the robot and waits for the specified time (`wait`).

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        qdot : JointVelocityType
            Desired joint velocities (nj,).
        trq : JointTorqueType
            Desired joint torques (nj,).
        wait : float
            Time to wait (in seconds) after commanding the robot to move.

        Returns
        -------
        int
            Status code (0 for success, non-zero for failure).
        """
        self._synchro_control(wait)
        _i = 0
        for r in self.robots:
            _q = q[_i : _i + r.nj]
            _qdot = qdot[_i : _i + r.nj]
            _trq = trq[_i : _i + r.nj]
            _i += r.nj
            self._last_status = r.GoTo_q(_q, _qdot, _trq, 0, **kwargs)
            if self._last_status != MotionResultCodes.MOTION_SUCCESS.value:
                return self._last_status
        self.GetState()
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = (J @ qdot).reshape(self.nr, 6)
        self.Update()
        return self._last_status

    def GoTo_qtraj(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, time: TimesType) -> int:
        """
        Command robots in combined system to follow a joint trajectory.

        It is intended to control the robot to follow a trajectory specified by joint positions (`q`),
        velocities (`qdot`), and accelerations (`qddot`) over a specified time (`time`).

        Parameters
        ----------
        q : np.ndarray
            Desired joint positions for the trajectory (n, nr, nj), where n is the number of trajectory points.
        qdot : np.ndarray
            Desired joint velocities for the trajectory (n, nr, nj), where n is the number of trajectory points.
        qddot : np.ndarray
            Desired joint accelerations for the trajectory (n, nr, nj), where n is the number of trajectory points.
        time : TimesType
            Time points for the trajectory (n,).
        Returns
        -------
        int
            Status code (0 for success, non-zero for failure).
        """
        _i = 0
        for i, r in enumerate(self.robots):
            _q = q[:, i, _i : _i + r.nj]
            _qdot = qdot[_i : _i + r.nj]
            _qddot = qddot[_i : _i + r.nj]
            _i += r.nj
            self._last_status = r.GoTo_qtraj(_q, _qdot, _qddot, time)
        return self._last_status

    # Task space motion
    def GoTo_T(self, x: Union[Poses3DType, HomogeneousMatricesType], v: Optional[Velocities3DType] = None, FT: Optional[np.ndarray] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Move robots to the target pose and velocity in Cartesian space.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., Pose, Transformation matrix).
        v : Velocities3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : np.ndarray, optional
            Target force/torque in Cartesian space. Default is a zero wrench array (nr, 6).
        wait : float, optional
            The time to wait after the movement, by default the sample time (`self.tsamp`).
        **kwargs : Any
            Additional keyword arguments passed to other methods, including `task_space`.

        Returns
        -------
        int
            The status of the move (0 for success, non-zero for error).

        Raises
        ------
        ValueError
            If the provided task space is not supported.

        Notes
        -----
        The method first converts the input `x`, `v`, and `FT` based on the specified task space.
        The robot will be moved using either Cartesian control or a transformation control strategy.
        """
        x = self.spatial(x)
        if v is None:
            v = np.zeros((self.nr, 6))
        else:
            v = self.spatial(v, shape=(6,))
        if FT is None:
            FT = np.zeros((self.nr, 6))
        else:
            FT = self.spatial(FT, shape=(6,))
        if wait is None:
            wait = self.tsamp
        if self._control_strategy.startswith("Cartesian"):
            task_space = kwargs.get("task_space", "World")
            if check_option(task_space, "World"):
                x = self.WorldToBase(x)
                v = self.WorldToBase(v, typ="Twist")
                FT = self.WorldToBase(FT, typ="Wrench")
            elif check_option(task_space, "Robot"):
                pass
            elif check_option(task_space, "Object"):
                x = self.ObjectToWorld(x)
                v = self.ObjectToWorld(v, typ="Twist")
                FT = self.ObjectToWorld(FT, typ="Wrench")
                x = self.WorldToBase(x)
                v = self.WorldToBase(v, typ="Twist")
                FT = self.WorldToBase(FT, typ="Wrench")
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
            self._last_status = self.GoTo_X(x, v, FT, wait, **kwargs)
        else:
            self._command.rx = x
            self._command.rv = v
            self._last_status = self.GoTo_TC(x, v=v, FT=FT, **kwargs)
        return self._last_status

    def GoTo_JT(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        t: TimesType,
        wait: Optional[float] = None,
        traj_samp_fac: float = None,
        max_iterations: int = 1000,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: Optional[str] = None,
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointVelocityType] = None,
        x_opt: Optional[Poses3DType] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
        state: str = "Commanded",
        **kwargs: Any,
    ) -> int:
        """
        Transforms Cartesian space trajectory to joiont space using inverse kinematics and then executes the trajectorj using `GoTo_qtraj`.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            Target end-effector pose in Cartesian space (nr, 7) or (nr, 4, 4).
        t : TimesType
            Time vector for trajectory (n,).
        wait : float, optional
            The time to wait after movement, by default None (using `self.tsamp`).
        traj_samp_fac : float, optional
            The factor for trajectory sampling, by default None (using `self._default.TrajSampTimeFac`).
        max_iterations : int, optional
            Maximum iterations for inverse kinematics, by default 1000.
        pos_err : float, optional
            Position error tolerance, by default None (using `self._default.PosErr`).
        ori_err : float, optional
            Orientation error tolerance, by default None (using `self._default.OriErr`).
        task_space : str, optional
            The task space for the motion, by default None (using `self._default.TaskSpace`).
        task_DOF : ArrayLike, optional
            Task degrees of freedom, by default None (using `self._default.TaskDOF`).
        null_space_task : str, optional
            The null space task for optimization, by default None (using `self._default.NullSpaceTask`).
        task_cont_space : str, optional
            The task control space, by default None (using `self._default.TaskContSpace`).
        q_opt : JointConfigurationType, optional
            Optimal joint configuration, by default None (using `self.q_home`).
        v_ns : Velocity3DType, optional
            Velocity for null space, by default None (using zeros).
        qdot_ns : JointVelocityType, optional
            Joint velocity for null space, by default None (using zeros).
        x_opt : Poses3DType, optional
            Optimal end-effector pose, by default None (calculated using `self.Kinmodel(q_opt)`).
        Kp : float, optional
            Position control gain, by default None (using `self._default.Kp`).
        Kns : float, optional
            Null space gain, by default None (using `self._default.Kns`).
        state : str, optional
            The robot state, by default "Commanded".
        **kwargs : Any
            Additional keyword arguments for customization.

        Returns
        -------
        int
            Status of the operation: 0 for success, non-zero for failure.

        Raises
        ------
        ValueError
            If the task space or kinematics calculation is not supported.

        Notes
        -----
        The method uses inverse kinematics to plan the trajectory from Cartesian to joint space.
        Cartesian movement is converted to joint space via `IKin` and `IKinPath` methods.
        If Cartesian movement is not feasible, a warning message is raised.
        """
        if wait is None:
            wait = self._default.Wait
        if traj_samp_fac is None:
            traj_samp_fac = self._default.TrajSampTimeFac
        else:
            traj_samp_fac = int(traj_samp_fac)
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        if task_cont_space is None:
            task_cont_space = self._default.TaskContSpace
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kns is None:
            Kns = self._default.Kns

        self.Message("Cartesian motion -> joint motion", 2)

        N = len(t)
        q_init = self.GetJointPos(state=state)
        if N == 1:
            rx = x2x(x)
            q_path, self._last_status = self.IKin(
                rx,
                q_init,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_space=task_space,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                task_cont_space=task_cont_space,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
                save_path=True,
            )
            _time = np.arange(0.0, t + self.tsamp, self.tsamp * traj_samp_fac)
            _time[-1] = t
            _s = np.linspace(0, t, q_path.shape[0])
            _s[-1] = t
            qi = interpPath(_s, q_path, _time)
        else:
            if x.ndim == 3:
                rx = uniqueCartesianPath(t2x(x))
            else:
                rx = uniqueCartesianPath(x)
            q_path, self._last_status = self.IKinPath(
                rx,
                q_init,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_space=task_space,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                task_cont_space=task_cont_space,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
            )
            _time = np.arange(0.0, t[-1] + self.tsamp, self.tsamp * traj_samp_fac)
            _time[-1] = t[-1]
            _s = np.linspace(0, t[-1], q_path.shape[0])
            _s[-1] = t[-1]
            qi = interpPath(_s, q_path, _time)

        if self._last_status == MotionResultCodes.MOTION_SUCCESS.value:
            qi[-1, :] = q_path[-1, :]
            qdoti = gradientPath(qi, _time)
            qdoti[-1, :] = qdoti[-1, :] * 0
            self.GoTo_qtraj(qi, qdoti, np.zeros(qi.shape), _time)
        else:
            self.WarningMessage("Cartesian movement not feasible!")

        return self._last_status

    def GoTo_TC(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        v: Optional[Velocities3DType] = None,
        FT: Optional[np.ndarray] = None,
        timeout: Optional[float] = None,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointVelocityType] = None,
        x_opt: Optional[Poses3DType] = None,
        Kp: Optional[float] = None,
        Kff: Optional[float] = None,
        Kns: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> int:
        """
        Kinematic controller for controlling bimanual robot in Cartesian space.

        This function uses inverse kinematics to move the robot's end-effector to a desired
        position while managing joint space motion through various null-space control strategies.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            Combined target end-effector pose (..., 7), expressed in the specified task space.
        v : Velocities3DType, optional
            Combined end-effector velocity (..., 6), by default None (zero velocity).
        FT : np.ndarray, optional
            Combined end-effector Force/Torque  (..., 6), by default None (zero force/torque).
        timeout : float, optional
            Timeout for kinematic controller (by default 0).
        pos_err : float, optional
            Position error tolerance, by default None (using `self._default.PosErr`).
        ori_err : float, optional
            Orientation error tolerance, by default None (using `self._default.OriErr`).
        task_space : str, optional
            The task space for motion, by default None (using `self._default.TaskSpace`).
        task_DOF : ArrayLike, optional
            Task Degrees of Freedom, by default None (using `self._default.TaskDOF`).
        null_space_task : str, optional
            Null-space task for optimization, by default None (using `self._default.NullSpaceTask`).
        task_cont_space : str, optional
            Task control space, by default "Robot".
        q_opt : JointConfigurationType, optional
            Optimal joint configuration for null-space, by default None (using `self.q_home`).
        v_ns : Velocity3DType, optional
            Null-space velocity, by default None (zero velocity).
        qdot_ns : JointVelocityType, optional
            Joint velocity for null-space control, by default None (zero joint velocity).
        x_opt : Poses3DType, optional
            Optimal end-effector pose, by default None (calculated from `self.Kinmodel(q_opt)`).
        Kp : float, optional
            Proportional gain for position control, by default None (using `self._default.Kp`).
        Kns : float, optional
            Null-space gain, by default None (using `self._default.Kns`).
        vel_fac : ArrayLike, optional
            Velocity scaling factor for each joint, by default None.
        **kwargs : Any
            Additional keyword arguments for flexibility.

        Returns
        -------
        int
            Status code (0 for success, non-zero for failure).

        Raises
        ------
        ValueError
            If the task space or null-space task is not supported.

        Notes
        -----
        The method uses inverse kinematics to perform Cartesian motion to joint-space motion.
        Different null-space tasks can be applied for optimization such as manipulability, joint limits, etc.
        """
        if v is None:
            rv = np.zeros((self.nr, 6))
        else:
            rv = self.spatial(v, shape=(6,))
        if FT is None:
            FT = np.zeros((self.nr, 6))
        else:
            FT = self.spatial(FT, shape=(6,))
        if timeout is None:
            timeout = 0
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=self.nr * 6)
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kff is None:
            Kff = self._default.Kff
        if Kns is None:
            Kns = self._default.Kns

        if vel_fac is None:
            vel_fac = self._default.VelocityScaling
        elif not isscalar(vel_fac):
            vel_fac = vector(vel_fac, dim=self.nj)
        _vel = self.qdot_max * vel_fac
        _vel = self.qdot_max * vel_fac

        tx = self.simtime()
        rx = self.spatial(x)
        Sind = np.where(np.asarray(task_DOF) > 0)[0]
        uNS = np.zeros(self.nj)

        if check_option(task_space, "World"):
            rx = self.WorldToBase(rx)
            rv = self.WorldToBase(rv, typ="Twist")
            FT = self.WorldToBase(FT, typ="Wrench")
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            rx = self.ObjectToWorld(rx)
            rv = self.ObjectToWorld(rv, typ="Twist")
            FT = self.ObjectToWorld(FT, typ="Wrench")
            rx = self.WorldToBase(rx)
            rv = self.WorldToBase(rv, typ="Twist")
            FT = self.WorldToBase(FT, typ="Wrench")
        else:
            raise ValueError(f"Task space '{task_space}' not supported")
        self._command.FT = FT

        imode = self._command.mode
        if check_option(null_space_task, "None"):
            self._command.mode = CommandModeCodes.CARTESIAN_NONE.value
        elif check_option(null_space_task, "Manipulability"):
            self._command.mode = CommandModeCodes.CARTESIAN_MANIPULABILITY.value
        elif check_option(null_space_task, "JointLimits"):
            self._command.mode = CommandModeCodes.CARTESIAN_JOINTLIMITS.value
            q_opt = (self.q_max + self.q_min) / 2
        elif check_option(null_space_task, "ConfOptimization"):
            self._command.mode = CommandModeCodes.CARTESIAN_CONFIGURATION.value
            q_opt = vector(q_opt, dim=self.nj)
        elif check_option(null_space_task, "PoseOptimization"):
            self._command.mode = CommandModeCodes.CARTESIAN_POSE.value
            x_opt = x2x(x_opt)
            if check_option(task_space, "World"):
                x_opt = self.WorldToBase(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.ObjectToWorld(x_opt)
                x_opt = self.WorldToBase(x_opt)
        elif check_option(null_space_task, "TaskVelocity"):
            self._command.mode = CommandModeCodes.CARTESIAN_TASKVELOCITY.value
            rv = vector(v_ns, dim=6)
            if check_option(task_space, "World"):
                rv = self.WorldToBase(rv, typ="Twist")
            elif check_option(task_space, "Object"):
                rv = self.ObjectToWorld(rv)
                rv = self.WorldToBase(rv, typ="Twist")
        elif check_option(null_space_task, "JointVelocity"):
            self._command.mode = CommandModeCodes.CARTESIAN_JOINTVELOCITY.value
            rqdn = vector(qdot_ns, dim=self.nj)
        elif check_option(null_space_task, "User"):
            if self._user_null_space_task_callback is not None:
                self._command.mode = CommandModeCodes.CARTESIAN_USER.value
            else:
                null_space_task = "None"
                self._command.mode = CommandModeCodes.CARTESIAN_NONE.value
        else:
            raise ValueError(f"Null-space task '{null_space_task}' not supported")

        self._command.x = rx
        self._command.v = rv

        while True:
            qq = self._command.q

            x, J = self.Kinmodel(qq)
            ee = xerr(rx, x)
            if check_option(task_cont_space, "World"):
                RC = np.block([[self.TBase[i // 2, :3, :3] if i == j else np.zeros((3, 3)) for j in range(self.nr * 2)] for i in range(self.nr * 2)]).T
            elif check_option(task_cont_space, "Robot"):
                RC = np.eye(self.nr * 6)
            elif check_option(task_cont_space, "Tool"):
                R = map_pose(x=x, out="R")
                RC = np.block([[R[i // 2, :3, :3] if i == j else np.zeros((3, 3)) for j in range(self.nr * 2)] for i in range(self.nr * 2)]).T
            elif check_option(task_cont_space, "Object"):
                RC = np.block([[self.TObject[i // 2, :3, :3] if i == j else np.zeros((3, 3)) for j in range(self.nr * 2)] for i in range(self.nr * 2)]).T
            else:
                raise ValueError(f"Task space '{task_cont_space}' not supported")

            eex = RC @ ee.ravel()
            J = RC @ J
            rvx = RC @ rv.ravel()
            ux = Kff * rvx + Kp * eex
            trq = J.T @ FT.ravel()
            self._command.ux = ux
            ux = ux[Sind]
            JJ = J[Sind, :]
            if self._default.DampedPseudoInverseFactor > 0:
                Jp = damped_pinv(JJ, self._default.DampedPseudoInverseFactor)
            else:
                Jp = np.linalg.pinv(JJ)
            NS = np.eye(self.nj) - Jp @ JJ

            if check_option(null_space_task, "None"):
                qdn = np.zeros(self.nj)
            elif check_option(null_space_task, "Manipulability"):
                fun = lambda q: self.Manipulability(q, task_space=task_space, task_DOF=task_DOF)
                qdotn = grad(fun, qq)
                qdn = Kns * qdotn
            elif check_option(null_space_task, "JointLimits"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "ConfOptimization"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "PoseOptimization"):
                qdn = Kns * np.linalg.pinv(J) @ xerr(x_opt, x).ravel()
            elif check_option(null_space_task, "TrackPath"):
                qdn = Kns * np.linalg.pinv(J) @ ee.ravel()
            elif check_option(null_space_task, "TaskVelocity"):
                qdn = np.linalg.pinv(J) @ rv.ravel()
            elif check_option(null_space_task, "JointVelocity"):
                qdn = rqdn
            elif check_option(null_space_task, "User"):
                qdn = self._user_null_space_task_callback(self, **kwargs)
            else:
                qdn = np.zeros(self.nj)

            uq = Jp @ ux
            _fac = np.max(np.abs(uq) / _vel)
            if _fac > 1:
                uq = uq / _fac
            uNS = NS @ qdn
            _i = 0
            for i in range(self.nr):
                _uNS = uNS[_i : _i + self.robots[i].nj]
                _umax = (_vel - uq)[_i : _i + self.robots[i].nj]
                if any(np.abs(_umax) < 1e-5):
                    _f1 = 1e5
                else:
                    _f1 = np.abs(_uNS / _umax)
                    _f1[np.isnan(_f1)] = 1
                    _f1[np.abs(_f1) < 1e-5] = 1
                _umin = (-_vel - uq)[_i : _i + self.robots[i].nj]
                if any(np.abs(_umin) < 1e-5):
                    _f2 = 1e5
                else:
                    _f2 = np.abs(_uNS / _umin)
                    _f2[np.abs(_f2) < 1e-5] = 1
                    _f2[np.isnan(_f2)] = 1
                _fac = max(1, np.max(_f1), np.max(_f2))
                if _fac > 1:
                    _uNS = _uNS / _fac
                    uNS[_i : _i + self.robots[i].nj] = _uNS
                _i += self.robots[i].nj

            u = uq + uNS
            self._command.u = u
            np.clip(u, -self.qdot_max, self.qdot_max)
            rq = qq + u * self.tsamp
            if self.CheckJointLimits(rq):
                self._command.mode = imode
                self._command.qdot = np.zeros(self.nj)
                self._command.v = np.zeros((self.nr, 6))
                self.WarningMessage(f"Joint limits reached: {self.q}")
                return MotionResultCodes.JOINT_LIMITS_REACHED.value

            self._last_status = self.GoTo_q(rq, u, trq, self.tsamp, **kwargs)

            if self.simtime() - tx > timeout or (np.max(np.linalg.norm(ee[:, :3], axis=1)) < pos_err and np.max(np.linalg.norm(ee[:, 3:], axis=1)) < ori_err):
                self._command.mode = imode
                return self._last_status

    def _loop_cartesian_traj(self, xi: np.ndarray, vi: np.ndarray, FT: np.ndarray, time: TimesType, wait: float = 0, **kwargs: Any) -> int:
        """
        Executes a Cartesian trajectory for the robot's end-effector.

        This function controls the robot to follow a given Cartesian trajectory in terms of
        position, velocity, and force/torque while checking for motion errors, abort conditions,
        and motion checks.

        Parameters
        ----------
        xi : np.ndarray
            The target Cartesian positions (n, 7) or (n, 4, 4), where n is the number of trajectory points.
        vi : np.ndarray
            The target Cartesian velocities (n, 6), where n is the number of trajectory points.
        FT : np.ndarray
            The target force/torque sensor data (6,).
        time : TimesType
            Time values corresponding to each point in the trajectory (n,).
        wait : float, optional
            The wait time after motion completion, by default 0.
        **kwargs : Any
            Additional keyword arguments passed to the control methods.

        Returns
        -------
        int
            Status code (0 for success, non-zero for failure).

        Raises
        ------
        ValueError
            If an unsupported control strategy is used.

        Notes
        -----
        The function supports both joint-based and Cartesian-based control strategies for the robot's motion.
        It will continuously monitor the motion status, abort conditions, and motion errors.
        """
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        if self._control_strategy in ["JointPositionTrajectory"]:
            raise ValueError("Trajecotry controller NOT IMPLEMENTED")
            # Joint Position Trajectory control strategy
            self._last_status = self.GoTo_JT(xi, time, wait=wait, **kwargs)
            if self._last_status == MotionResultCodes.MOTION_SUCCESS.value:
                self.Update()
                _t_traj = self.simtime()
                while (self.simtime() - _t_traj) < (time[-1] + wait):
                    self._last_control_time = self.simtime()
                    if self._abort:
                        self.WarningMessage("Motion aborted by user")
                        self.StopMotion()
                        return MotionResultCodes.MOTION_ABORTED.value
                    elif self._do_motion_check and self._motion_check_callback is not None:
                        self._last_status = self._motion_check_callback(self)
                        if self._last_status != 0:
                            self.WarningMessage("Motion aborted")
                            self.StopMotion()
                            self._command.mode = CommandModeCodes.ABORT.value
                        return self._last_status
                    elif (self._motion_error is not None) and (self._motion_error != 0):
                        self.WarningMessage("Motion aborted due to motion controller error")
                        return self._motion_error
                    sleep(self.tsamp)
                    self.Update()
        else:
            # Cartesian-based control strategy
            for xt, vt in zip(xi, vi):
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self.StopMotion()
                    return MotionResultCodes.MOTION_ABORTED.value
                elif self._do_motion_check and self._motion_check_callback is not None:
                    self._last_status = self._motion_check_callback(self)
                    if self._last_status != 0:
                        self._command.qdot = np.zeros(self.nj)
                        self._command.v = np.zeros((self.nr, 6))
                        self.WarningMessage("Motion check stopped motion")
                        self.StopMotion()
                        return self._last_status
                self._last_status = self.GoTo_T(xt, vt, FT, wait=0, **kwargs)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Motion aborted")
                    self.StopMotion()
                    return self._last_status
            self._last_status = self.GoTo_T(xi[-1], np.zeros((self.nr, 6)), FT, wait=wait, **kwargs)
            # self._last_status = self.GoTo_T(xi[-1], np.zeros((self.nr, 6)), FT, wait=self.tsamp, **kwargs)
            # tx = self.simtime()
            # while self.simtime() - tx < wait:
            #     self.Update()
            #     sleep(self.tsamp)
        return self._last_status

    def _CMove(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[np.ndarray] = None,
        state: str = "Commanded",
        min_pos_dist: Optional[float] = None,
        min_ori_dist: Optional[float] = None,
        **kwargs: Any,
    ) -> int:
        """
        Helper method for executing the Cartesian move.

        This method is used internally to carry out the actual motion logic after parameter validation.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            The target Cartesian pose.
        t : float, optional
            The duration for the movement, by default None.
        vel : float, optional
            The velocity at which the end-effector moves, by default None.
        vel_fac : ArrayLike, optional
            A factor to scale the velocity, by default None.
        traj : str, optional
            The trajectory type, by default None.
        short : bool, optional
            Whether to shorten the path, by default None.
        wait : float, optional
            The wait time after the movement, by default None.
        task_space : str, optional
            The task space reference frame, by default None.
        added_FT : np.ndarray, optional
            Additional force/torque to be applied, by default None.
        state : str, optional
            The state of the robot (e.g., "Commanded" or "Actual"), by default "Commanded".
        min_pos_dist : float, optional
            The minimum position distance for stopping, by default None.
        min_ori_dist : float, optional
            The minimum orientation distance for stopping, by default None.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            The status code of the move (0 if successful, non-zero if error occurred).
        """

        if traj is None:
            traj = self._default.Traj
        if short is None:
            short = self._default.RotDirShort
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = self.spatial(added_FT, shape=(6,))
        if min_pos_dist is None:
            min_pos_dist = self._default.MinPosDist
        if min_ori_dist is None:
            min_ori_dist = self._default.MinOriDist

        kwargs.setdefault("kinematics", self._default.Kinematics)

        x = self.spatial(x)
        if wait is None:
            wait = self.tsamp

        rT = np.zeros((self.nr, 4, 4))
        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            for i in range(self.nr):
                if x[i].shape == (4, 4):
                    rT[i] = T0[i] @ x[i]
                elif isvector(x[i], dim=7):
                    rT[i] = T0[i] @ x2t(x[i])
                elif x[i].shape == (3, 3):
                    rT[i] = T0[i] @ map_pose(R=x[i], out="T")
                elif isvector(x[i], dim=3):
                    rT[i] = T0[i] @ map_pose(p=x[i], out="T")
                elif isvector(x[i], dim=4):
                    rT[i] = T0[i] @ map_pose(Q=x[i], out="T")
                else:
                    raise ValueError(f"Parameter shape {x.shape} not supported")
        else:
            for i in range(self.nr):
                if x[i].shape == (4, 4):
                    rT[i] = x[i]
                elif isvector(x[i], dim=7):
                    rT[i] = x2t(x[i])
                else:
                    p0, R0 = self.GetPose(out="pR", state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                    if x[i].shape == (3, 3):
                        rT[i] = map_pose(R=x[i], p=p0, out="T")
                    elif isvector(x[i], dim=4):
                        rT[i] = map_pose(Q=x[i], p=p0, out="T")
                    elif isvector(x, dim=3):
                        rT[i] = map_pose(p=x[i], R=R0, out="T")
                    else:
                        raise ValueError(f"Parameter shape {x.shape} not supported")

        kwargs.setdefault("task_space", task_space)
        rx = np.array(t2x(rT))

        dist = xerr(rx, self._command.x)
        if np.max(np.linalg.norm(dist[:, :3])) < min_pos_dist and np.max(np.linalg.norm(dist[:, 3:])) < min_ori_dist:
            self.Message("CMove not executed - close to target", 2)
            return MotionResultCodes.CLOSE_TO_TARGET.value

        if t is not None:
            if not isscalar(t) or t <= 0:
                raise ValueError("Time must be non-negative scalar")
            elif t <= 10 * self.tsamp:
                t = None
        if t is None:
            _time = np.arange(0.0, 1 + self.tsamp, self.tsamp)
            if vel is None:
                if vel_fac is None:
                    vel_fac = self._default.VelocityScaling
                elif isvector(vel_fac, dim=2):
                    vel_fac = np.tile(np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3))), (self.nr, 1))
                elif not isscalar(vel_fac):
                    vel_fac = self.spatial(vel_fac, shape=(6,))
                _vel = self.v_max * vel_fac
                self.Message(f"CMove started with velocity {100 * np.max(_vel / self.v_max):.1f}% to\n{rx}", 2)
            else:
                if isscalar(vel):
                    _vel = np.hstack((np.abs(normalize(dist[:, :3])) * vel, self.v_max[:, 3:]))
                    self.Message(f"CMove started with velocity {vel:.1f}m/s to \n{rx}", 2)
                elif isvector(vel, dim=2):
                    _vel = np.hstack((np.abs(normalize(dist[:, :3])) * vel[0], np.abs(normalize(dist[:, 3:])) * vel[1]))
                    # _vel = np.zeros((self.nr, 6))
                    # for i in range(self.nr):
                    #     _norm = np.linalg.norm(dist[i, :3])
                    #     if _norm < 1e-3:
                    #         _vel[i, :3] = np.ones(3) * vel[0]
                    #     else:
                    #         _vel[i, :3] = np.abs(dist[i, :3]) / _norm * vel[0]
                    #     _norm = np.linalg.norm(dist[i, 3:])
                    #     if _norm < 1e-3:
                    #         _vel[i, 3:] = np.ones(3) * vel[1]
                    #     else:
                    #         _vel[i, 3:] = np.abs(dist[i, 3:]) / _norm * vel[1]
                    self.Message(f"CMove started with velocity {vel[0]:.1f}m/s and {vel[1]:.1f}rd/s to\n{rx}", 2)
                else:
                    _vel = self.spatial(vel, shape=(6,))
                    self.Message(f"CMove started with velocity {100 * np.max(_vel / self.v_max):.1f}% to\n{rx}", 2)
            _vel = np.clip(_vel, 0, self.v_max)
            _vel[np.where(_vel < 1e-3)] = 1e-3
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CMove started in {_time[-1]:.1f}s to\n{rx}", 2)

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        xi = np.zeros((len(_time), self.nr, 7))
        vi = np.zeros((len(_time), self.nr, 6))
        for i in range(self.nr):
            _xi, _vi, _ = ctraj(x0[i], rx[i], _time, traj=traj, short=short)
            xi[:, i, :] = _xi
            vi[:, i, :] = _vi
        _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _tend = max(_time[-1] * _fac, 10 * self.tsamp) + self.tsamp
            _time = np.arange(0.0, _tend, self.tsamp)
            xi = np.zeros((len(_time), self.nr, 7))
            vi = np.zeros((len(_time), self.nr, 6))
            for i in range(self.nr):
                _xi, _vi, _ = ctraj(x0[i], rx[i], _time, traj=traj, short=short)
                xi[:, i, :] = _xi
                vi[:, i, :] = _vi

        if self._semaphore._value <= 0:
            self.Message("Not executed due to active treads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = 2
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        self._semaphore.acquire()

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CMove finished", 2)
        self._semaphore.release()
        return self._last_status

    def CMoveFor(
        self,
        dx: ArrayLike,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[np.ndarray] = None,
        state: str = "Commanded",
        asynchronous: bool = False,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in Cartesian space based on a displacement vector.

        The robot moves its end-effector from the current position by a given displacement.

        Parameters
        ----------
        dx : ArrayLike
            Displacement in Cartesian space for position or rotation.
        t : float, optional
            Time to complete the movement, by default None.
        vel : float, optional
            Maximum velocity, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Trajectory type, by default None.
        short : bool, optional
            Whether to shorten the trajectory, by default None.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : np.ndarray, optional
            Additional force/torque values, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, executes the motion asynchronously, by default False.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        kwargs.setdefault("kinematics", self._default.Kinematics)
        dx = self.spatial(dx, shape=(3,))
        rT = np.zeros((self.nr, 4, 4))
        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            for i in range(self.nr):
                if isvector(dx[i], dim=3):
                    rT[i] = T0[i] @ map_pose(p=dx[i], out="T")
                elif dx[i].shape == (3, 3):
                    rT[i] = T0[i] @ map_pose(R=dx[i], out="T")
                elif isvector(dx[i], dim=4):
                    rT[i] = T0[i] @ map_pose(Q=dx[i], out="T")
                else:
                    raise ValueError(f"Parameter shape {dx.shape} not supported")
        else:
            rT = self.GetPose(out="T", task_space=task_space, kinematics=kwargs["kinematics"], state=state)
            for i in range(self.nr):
                if isvector(dx[i], dim=3):
                    rT[i, :3, 3] += dx[i]
                elif dx[i].shape == (3, 3):
                    rT[i, :3, :3] = dx[i] @ rT[i, :3, :3]
                elif isvector(dx[i], dim=4):
                    rT[i, :3, :3] = q2r(dx[i]) @ rT[i, :3, :3]
                else:
                    raise ValueError(f"Parameter shape {dx.shape} not supported")
        rx = t2x(rT)
        self.Message("CMoveFor -> CMove", 2)
        self._last_status = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def CApproach(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        dx: ArrayLike,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[np.ndarray] = None,
        state: str = "Commanded",
        asynchronous: bool = False,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot towards a target pose with an offset.

        The robot moves to a position that is offset by a specified displacement from the given target pose.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            The target Cartesian pose.
        dx : ArrayLike
            The positional displacement to move towards.
        t : float, optional
            Time to complete the movement, by default None.
        vel : float, optional
            Maximum velocity, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Trajectory type, by default None.
        short : bool, optional
            Whether to shorten the trajectory, by default None.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : np.ndarray, optional
            Additional force/torque values, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, executes the motion asynchronously, by default False.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        kwargs.setdefault("kinematics", self._default.Kinematics)
        _x = self.spatial(x)
        dx = self.spatial(dx, shape=(3,))
        if check_shape(_x, shape=(4, 4)):
            rx = map_pose(T=_x)
        elif check_shape(_x, shape=7):
            rx = _x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")
        rx[:, :3] += dx
        self.Message("CApproach -> CMove", 2)
        self._last_status = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def _CArc(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        pC: ArrayLike,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[np.ndarray] = None,
        state: str = "Commanded",
        **kwargs: Any,
    ) -> int:
        """
        Execute the internal arc trajectory movement logic in Cartesian space.

        This method computes and executes the arc movement trajectory for the robot's end-effector.

        Parameters
        ----------
        x : Union[Poses3DType, HomogeneousMatricesType]
            Target Cartesian pose.
        pC : ArrayLike
            Center of the arc in Cartesian space.
        t : float, optional
            Time to complete the movement, by default None.
        vel : float, optional
            Maximum velocity, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Trajectory type, by default None.
        short : bool, optional
            Whether to shorten the trajectory, by default None.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : np.ndarray, optional
            Additional force/torque values, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        if traj is None:
            traj = self._default.Traj
        if short is None:
            short = self._default.RotDirShort
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = self.spatial(added_FT, shape=(6,))
        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs.setdefault("task_space", task_space)

        x = self.spatial(x)
        pC = self.spatial(pC, shape=(3,))
        if wait is None:
            wait = self.tsamp
        if FT is None:
            FT = np.zeros((self.nr, 6))
        else:
            FT = self.spatial(FT, shape=(6,))

        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            rpC = T0[:3, :3] @ pC
            if x.shape == (4, 4):
                rT = T0 @ x
            elif isvector(x, dim=7):
                rT = T0 @ x2t(x)
            elif x.shape == (3, 3):
                rT = T0 @ map_pose(R=x, out="T")
            elif isvector(x, dim=3):
                rT = T0 @ map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                rT = T0 @ map_pose(Q=x, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
        else:
            rpC = np.array(pC)
            if x.shape == (4, 4):
                rT = x
            elif isvector(x, dim=7):
                rT = x2t(x)
            elif x.shape == (3, 3):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(R=x, p=p0, out="T")
            elif isvector(x, dim=4):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(Q=x, p=p0, out="T")
            elif isvector(x, dim=3):
                R0 = self.GetOri(state=state, out="R", task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(p=x, R=R0, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        rx = np.array(t2x(rT))

        if t is not None:
            if not isscalar(t) or t <= 0:
                raise ValueError("Time must be non-negative scalar")
            elif t <= 10 * self.tsamp:
                t = None
        if t is None:
            _time = np.arange(0.0, 1 + self.tsamp, self.tsamp)
            if vel is None:
                if vel_fac is None:
                    vel_fac = self._default.VelocityScaling
                elif isvector(vel_fac, dim=2):
                    vel_fac = np.tile(np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3))), (self.nr, 1))
                elif not isscalar(vel_fac):
                    vel_fac = self.spatial(vel_fac)
                _vel = self.v_max * vel_fac
                self.Message(f"CArc started with velocity {100 * np.max(_vel / self.v_max):.1f}% to\n{rx}", 2)
            else:
                if isscalar(vel):
                    _vel = np.ones((self.nr, 6)) * vel
                    self.Message(f"CArc started with velocity {vel:.1f}m/s to \n{rx}\n{rpC}", 2)
                elif isvector(vel, dim=2):
                    _vel = np.tile(np.concatenate((_vel[0] * np.ones(3), _vel[1] * np.ones(3))), (self.nr, 1))
                    self.Message(f"CArc started with velocity {vel[0]:.1f}m/s and {vel[1]:.1f}rd/s to\n{rx}\n{rpC}", 2)
                else:
                    _vel = self.spatial(vel)
                    self.Message(f"CArc started with velocity {100 * np.max(_vel / self.v_max):.1f}% to\n{rx}\n{rpC}", 2)
            _vel = np.clip(_vel, 0, self.v_max)
            _vel[np.where(_vel < 1e-3)[0]] = np.inf
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CArc started in {_time[-1]:.1f}s to\n{rx}\n{rpC}", 2)

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        xi = np.zeros((len(_time), self.nr, 7))
        vi = np.zeros((len(_time), self.nr, 6))
        for i in range(self.nr):
            _xi, _vi, _ = carctraj(x0[i], rx[i], rpC[i], _time, traj=traj, short=short)
            xi[:, i, :] = _xi
            vi[:, i, :] = _vi
        _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _tend = max(_time[-1] * _fac, 10 * self.tsamp) + self.tsamp
            _time = np.arange(0.0, _tend, self.tsamp)
            xi = np.zeros((len(_time), self.nr, 7))
            vi = np.zeros((len(_time), self.nr, 6))
            for i in range(self.nr):
                _xi, _vi, _ = carctraj(x0[i], rx[i], rpC[i], _time, traj=traj, short=short)
                xi[:, i, :] = _xi
                vi[:, i, :] = _vi
        if self._semaphore._value <= 0:
            self.Message("Not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        self._semaphore.acquire()

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CArc finished", 2)
        self._semaphore.release()
        return self._last_status

    def _CPath(self, path: np.ndarray, t: Union[TimesType, float], direction: str = "Forward", wait: Optional[float] = None, task_space: Optional[str] = None, added_FT: Optional[np.ndarray] = None, state: str = "Commanded", **kwargs: Any) -> int:
        """
        Execute the internal path trajectory movement logic in Cartesian space.

        This method computes and executes the path movement trajectory for the robot's end-effector.

        Parameters
        ----------
        path : np.ndarray
            Path in Cartesian space (nr, n, 7).
        t : Union[TimesType, float]
            Time to complete the movement, (n,) or scalar.
        direction : str, optional
            Direction of movement, by default "Forward".
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : np.ndarray, optional
            Additional force/torque values, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        if not check_shape(path, shape=(2, 7)):
            raise ValueError(f"Wrong path size {path.shape}. Must be (...,{self.nr},7)")
        for i in range(self.nr):
            path[:, i, :] = uniqueCartesianPath(path[:, i, :])

        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = self.spatial(added_FT, shape=(6,))

        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs.setdefault("task_space", task_space)

        N = path.shape[0]
        rx_init = self.GetPose(task_space=task_space, state="Commanded", out="x")
        if not isscalar(t) and len(t) == path.shape[0]:
            if t[0] > 0:
                path = np.vstack((rx_init, path))
                t = np.concatenate(([0], t))
            _s = t
            N += 1
            t = max(t)
        else:
            if not isscalar(t):
                t = max(t)
            _s = np.linspace(0, t, N)
        _time = np.arange(0.0, t + self.tsamp, self.tsamp)

        xi = np.zeros((_time.size, self.nr, 7))
        vi = np.zeros((_time.size, self.nr, 6))
        for i in range(self.nr):
            xi[:, i, :] = interpCartesianPath(_s, path[:, i, :], _time)
            vi[:, i, :] = gradientCartesianPath(xi[:, i, :], _time)

        _fac = np.max(np.max(np.abs(vi), axis=0) / self.v_max)
        if _fac > 1:
            _s = np.linspace(0.0, t * _fac, N)
            _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
            xi = np.zeros((_time.size, self.nr, 7))
            vi = np.zeros((_time.size, self.nr, 6))
            for i in range(self.nr):
                xi[:, i, :] = interpCartesianPath(_s, path[:, i, :], _time)
                vi[:, i, :] = gradientCartesianPath(xi[:, i, :], _time)
        N = _time.size

        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        _dist = xerr(path[0, :], rx_init)
        xe = np.amax(np.abs(_dist) / self.v_max) * 2
        if xe > 0.02:
            self.Message(f"Move to path -> CMove ({_dist})", 2)
            self._last_status = self._CMove(path[0], max(xe, 0.2), traj="Poly", shape=True, wait=0, added_FT=FT, **kwargs)
            if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                self.WarningMessage("Robot did not move to path start")
                return self._last_status

        self.Message(f"CPath started {path.shape[0]} points in {t}s", 2)
        if self._semaphore._value <= 0:
            self.Message("Not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value
        self._semaphore.acquire()

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Message("CPath finished", 2)
        self.Stop()
        self._semaphore.release()
        return self._last_status

    def CRBFPath(self, *args: Any, **kwargs: Any) -> int:
        """
        Placeholder for RBF path motion in multi-robot systems.

        Returns
        -------
        int
            Status code indicating the method is not implemented.
        """
        self.WarningMessage("CRBFPath not implemented for multi robot systems!")
        return MotionResultCodes.MOTION_FAILURE.value

    # Transformations
    def BaseToWorld(self, x: np.ndarray, typ: str = None, robot_num: int = None) -> np.ndarray:
        """
        Map from robot base frame to world frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : np.ndarray
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None.
            If "Wrench", the transformation considers the force.
            If "Twist", the transformation considers the velocity.
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        np.ndarray
            Mapped argument in the world frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        if robot_num is None:
            x = np.asarray(x)
            x_out = copy.deepcopy(x)
            for i in range(self.nr):
                R0 = self.TBase[i][:3, :3]
                p0 = self.TBase[i][:3, 3]
                if x.shape == (self.nr * 6, self.nj):
                    _x = x[i * 6 : (i + 1) * 6]
                else:
                    _x = x[i]
                if _x.shape == (4, 4):  # T
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
                elif isvector(_x, dim=7):  # x
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
                elif _x.shape == (3, 3):
                    x_out[i] = R0 @ _x
                elif isvector(_x, dim=4):  # Q
                    x_out[i] = r2q(R0 @ q2r(_x))
                elif isvector(_x, dim=3):  # p
                    x_out[i] = R0 @ _x + p0
                elif isvector(_x, dim=6):  # v, F
                    RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                    if typ == "Twist":  # velocity
                        RRb = np.eye(6)
                        RRb[:3, 3:] = v2s(self.TBase[i][:3, :3] @ self._actual.x[i, :3]).T
                        x_out[i] = RR @ _x + RRb @ self.vBase[i]
                    elif typ == "Wrench":  # wrench (F)
                        RR[3:6, :3] = v2s(p0) @ R0
                    x_out[i] = RR @ _x
                elif _x.shape == (6, self.nj):  # J
                    x_out[i * 6 : (i + 1) * 6] = np.vstack((R0 @ _x[:3], R0 @ _x[3:]))  # TODO: Preveri za premikajočo bazo!
                else:
                    raise ValueError(f"Parameter shape {_x.shape} not supported")
            return x_out
        elif robot_num >= 0 and robot_num < self.nr:
            R0 = self.TBase[robot_num][:3, :3]
            p0 = self.TBase[robot_num][:3, 3]
            x = np.asarray(x)
            if x.shape == (4, 4):  # T
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
            elif isvector(x, dim=7):  # x
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
            elif x.shape == (3, 3):
                return R0 @ x
            elif isvector(x, dim=4):  # Q
                return r2q(R0 @ q2r(x))
            elif isvector(x, dim=3):  # p
                return R0 @ x + p0
            elif isvector(x, dim=6):  # v, F
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Twist":  # velocity
                    RRb = np.eye(6)
                    RRb[:3, 3:] = v2s(self.TBase[robot_num][:3, :3] @ self._actual.x[:3]).T
                    return RR @ x + RRb @ self.vBase[robot_num]
                elif typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                return RR @ x
            elif x.shape == (6, self.nj):  # J
                return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))  # TODO: Preveri za premikajočo bazo!
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

    def WorldToBase(self, x: np.ndarray, typ: str = None, robot_num: int = None) -> np.ndarray:
        """
        Map from world frame to robot base frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : np.ndarray
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None.
            If "Wrench", the transformation considers the force.
            If "Twist", the transformation considers the velocity.
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        np.ndarray
            Mapped argument in the robot base frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        if robot_num is None:
            x = np.asarray(x)
            x_out = copy.deepcopy(x)
            for i in range(self.nr):
                R0 = self.TBase[i][:3, :3].T
                p0 = -R0 @ self.TBase[i][:3, 3]
                if x.shape == (self.nr * 6, self.nj):
                    _x = x[i * 6 : (i + 1) * 6]
                else:
                    _x = x[i]
                if _x.shape == (4, 4):  # T
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
                elif isvector(_x, dim=7):  # x
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
                elif _x.shape == (3, 3):
                    x_out[i] = R0 @ _x
                elif isvector(_x, dim=4):  # Q
                    x_out[i] = r2q(R0 @ q2r(_x))
                elif isvector(_x, dim=3):  # p
                    x_out[i] = R0 @ _x + p0
                elif isvector(_x, dim=6):  # v, F
                    RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                    if typ == "Twist":  # velocity
                        RRb = np.eye(6)
                        RRb[:3, 3:] = v2s(self.TBase[i][:3, :3] @ self._actual.x[i, :3]).T
                        x_out[i] = RR @ _x - RRb @ self.vBase[i]
                    elif typ == "Wrench":  # wrench (F)
                        RR[3:6, :3] = v2s(p0) @ R0
                    x_out[i] = RR @ _x
                elif _x.shape == (6, self.nj):  # J
                    x_out[i * 6 : (i + 1) * 6] = np.vstack((R0 @ _x[:3], R0 @ _x[3:]))  # TODO: Preveri za premikajočo bazo!
                else:
                    raise ValueError(f"Parameter shape {_x.shape} not supported")
            return x_out
        elif robot_num >= 0 and robot_num < self.nr:
            R0 = self.TBase[robot_num][:3, :3]
            p0 = self.TBase[robot_num][:3, 3]
            x = np.asarray(x)
            if x.shape == (4, 4):  # T
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
            elif isvector(x, dim=7):  # x
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
            elif x.shape == (3, 3):
                return R0 @ x
            elif isvector(x, dim=4):  # Q
                return r2q(R0 @ q2r(x))
            elif isvector(x, dim=3):  # p
                return R0 @ x + p0
            elif isvector(x, dim=6):  # v, F
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Twist":  # velocity
                    RRb = np.eye(6)
                    RRb[:3, 3:] = v2s(self.TBase[robot_num][:3, :3] @ self._actual.x[:3]).T
                    return RR @ x + RRb @ self.vBase[robot_num]
                elif typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                return RR @ x
            elif x.shape == (6, self.nj):  # J
                return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))  # TODO: Preveri za premikajočo bazo!
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

    def ObjectToWorld(self, x: np.ndarray, typ: str = None, robot_num: int = None) -> np.ndarray:
        """
        Map from object frame to world frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : np.ndarray
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None. If "Wrench", the transformation considers the force.
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        np.ndarray
            Mapped argument in the world frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        if robot_num is None:
            x = np.asarray(x)
            x_out = copy.deepcopy(x)
            for i in range(self.nr):
                R0 = self.TObject[i][:3, :3]
                p0 = self.TObject[i][:3, 3]
                if x.shape == (self.nr * 6, self.nj):
                    _x = x[i * 6 : (i + 1) * 6]
                else:
                    _x = x[i]
                if _x.shape == (4, 4):  # T
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
                elif isvector(_x, dim=7):  # x
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
                elif _x.shape == (3, 3):
                    x_out[i] = R0 @ _x
                elif isvector(_x, dim=4):  # Q
                    x_out[i] = r2q(R0 @ q2r(_x))
                elif isvector(_x, dim=3):  # p
                    x_out[i] = R0 @ _x + p0
                elif isvector(_x, dim=6):  # v, F
                    RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                    if typ == "Wrench":  # wrench (F)
                        RR[3:6, :3] = v2s(p0) @ R0
                    x_out[i] = RR @ _x
                elif _x.shape == (6, self.nj):  # J
                    x_out[i * 6 : (i + 1) * 6] = np.vstack((R0 @ _x[i * 6 : i * 6 + 3, :], R0 @ _x[i * 6 + 3 : (i + 1) * 6, :]))  # TODO: Preveri za premikajočo bazo!
                else:
                    raise ValueError(f"Parameter shape {_x.shape} not supported")
            return x_out
        elif robot_num >= 0 and robot_num < self.nr:
            R0 = self.TObject[robot_num][:3, :3]
            p0 = self.TObject[robot_num][:3, 3]
            x = np.asarray(x)
            if x.shape == (4, 4):  # T
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
            elif isvector(x, dim=7):  # x
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
            elif x.shape == (3, 3):
                return R0 @ x
            elif isvector(x, dim=4):  # Q
                return r2q(R0 @ q2r(x))
            elif isvector(x, dim=3):  # p
                return R0 @ x + p0
            elif isvector(x, dim=6):  # v, F
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                return RR @ x
            elif x.shape == (6, self.nj):  # J
                return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))  # TODO: Preveri za premikajočo bazo!
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

    def WorldToObject(self, x: np.ndarray, typ: str = None, robot_num: int = None) -> np.ndarray:
        """
        Map from world frame to object frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : np.ndarray
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None. If "Wrench", the transformation considers the force.
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        np.ndarray
            Mapped argument in the object frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        if robot_num is None:
            x = np.asarray(x)
            x_out = copy.deepcopy(x)
            for i in range(self.nr):
                R0 = self.TObject[i][:3, :3].T
                p0 = -R0 @ self.TObject[i][:3, 3]
                if x.shape == (self.nr * 6, self.nj):
                    _x = x[i * 6 : (i + 1) * 6]
                else:
                    _x = x[i]
                if _x.shape == (4, 4):  # T
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
                elif isvector(_x, dim=7):  # x
                    p, R = map_pose(x=_x, out="pR")
                    x_out[i] = map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
                elif _x.shape == (3, 3):
                    x_out[i] = R0 @ _x
                elif isvector(_x, dim=4):  # Q
                    x_out[i] = r2q(R0 @ q2r(_x))
                elif isvector(_x, dim=3):  # p
                    x_out[i] = R0 @ _x + p0
                elif isvector(_x, dim=6):  # v, F
                    RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                    if typ == "Wrench":  # wrench (F)
                        RR[3:6, :3] = v2s(p0) @ R0
                    x_out[i] = RR @ _x
                elif _x.shape == (6, self.nj):  # J
                    x_out[i * 6 : (i + 1) * 6] = np.vstack((R0 @ _x[i * 6 : i * 6 + 3, :], R0 @ _x[i * 6 + 3 : (i + 1) * 6, :]))  # TODO: Preveri za premikajočo bazo!
                else:
                    raise ValueError(f"Parameter shape {_x.shape} not supported")
            return x_out
        elif robot_num >= 0 and robot_num < self.nr:
            R0 = self.TObject[robot_num][:3, :3]
            p0 = self.TObject[robot_num][:3, 3]
            x = np.asarray(x)
            if x.shape == (4, 4):  # T
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
            elif isvector(x, dim=7):  # x
                p, R = map_pose(x=x, out="pR")
                return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
            elif x.shape == (3, 3):
                return R0 @ x
            elif isvector(x, dim=4):  # Q
                return r2q(R0 @ q2r(x))
            elif isvector(x, dim=3):  # p
                return R0 @ x + p0
            elif isvector(x, dim=6):  # v, F
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                if typ == "Wrench":  # wrench (F)
                    RR[3:6, :3] = v2s(p0) @ R0
                return RR @ x
            elif x.shape == (6, self.nj):  # J
                return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))  # TODO: Preveri za premikajočo bazo!
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

    # Kinematic utilities
    def DKinPath(self, path: np.ndarray, out: str = None) -> np.ndarray:
        """
        Compute direct kinematics for a path of joint positions.

        Parameters
        ----------
        path : np.ndarray
            Path in joint space - poses (n, nj).
        out : str, optional
            Output format for the result, by default None (depends on robot settings).

        Returns
        -------
        np.ndarray
            Task positions at target pose (n, 7) or (4, 4, n), depending on the output format.
        """
        if out is None:
            out = self._default.TaskPoseForm

        _path = rbs_type(path)
        _n = np.shape(_path)[0]
        _xpath = np.nan * np.zeros((self.nr, _n, 7))

        for i in range(_n):
            _x = self.DKin(_path[i, :])
            _xpath[:, i, :] = _x

        return map_pose(x=_xpath, out=out)

    def IKin(
        self,
        x: np.ndarray,
        q0: np.ndarray,
        max_iterations: int = 1000,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[np.ndarray] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[np.ndarray] = None,
        v_ns: Optional[np.ndarray] = None,
        qdot_ns: Optional[np.ndarray] = None,
        x_opt: Optional[np.ndarray] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
        save_path: bool = False,
    ) -> np.ndarray:
        """
        Compute inverse kinematics to find joint positions that achieve a target Cartesian pose.

        Parameters
        ----------
        x : np.ndarray
            Target Cartesian pose (7,) or (4, 4).
        q0 : np.ndarray
            Initial joint positions (nj,).
        max_iterations : int, optional
            Maximum number of iterations for the inverse kinematics algorithm, by default 1000.
        pos_err : float, optional
            Position error tolerance, by default None (uses default value).
        ori_err : float, optional
            Orientation error tolerance, by default None (uses default value).
        task_space : str, optional
            Task space in which to perform the inverse kinematics, by default "Robot".
        task_DOF : np.ndarray, optional
            Degrees of freedom of the task, by default None.
        null_space_task : str, optional
            Type of null-space task, by default None.
        task_cont_space : str, optional
            The space in which task continuity is considered, by default "Robot".
        q_opt : np.ndarray, optional
            Optimal joint positions for null space tasks, by default None.
        v_ns : np.ndarray, optional
            Null-space velocity for task-space velocity, by default None.
        qdot_ns : np.ndarray, optional
            Null-space joint velocities, by default None.
        x_opt : np.ndarray, optional
            Optimal Cartesian pose for null-space tasks, by default None.
        Kp : float, optional
            Proportional gain for the controller, by default None.
        Kns : float, optional
            Gain for the null-space task, by default None.
        save_path : bool, optional
            Whether to save the joint positions for the path, by default False.

        Returns
        -------
        np.ndarray
            Joint positions at target pose (nj,).
        """
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kns is None:
            Kns = self._default.Kns

        _max_err = np.ones(6)
        _max_err[:3] = pos_err
        _max_err[3:] = ori_err

        rx = x2x(x)
        q0 = self.jointvar(q0)

        Sind = np.where(np.asarray(task_DOF) > 0)[0]
        uNS = np.zeros(self.nj)

        if check_option(task_space, "World"):
            rx = self.WorldToBase(rx)
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            rx = self.ObjectToWorld(rx)
            rx = self.WorldToBase(rx)
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        if check_option(null_space_task, "None"):
            pass
        elif check_option(null_space_task, "Manipulability"):
            pass
        elif check_option(null_space_task, "JointLimits"):
            q_opt = (self.q_max + self.q_min) / 2
        elif check_option(null_space_task, "ConfOptimization"):
            q_opt = vector(q_opt, dim=self.nj)
        elif check_option(null_space_task, "PoseOptimization"):
            x_opt = self.Kinmodel(self.q_home)
            x_opt = x2x(x_opt)
            if check_option(task_space, "World"):
                x_opt = self.WorldToBase(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.ObjectToWorld(x_opt)
                x_opt = self.WorldToBase(x_opt)
        elif check_option(null_space_task, "TaskVelocity"):
            rv = vector(v_ns, dim=6)
            if check_option(task_space, "World"):
                rv = self.WorldToBase(rv)
            elif check_option(task_space, "Object"):
                rv = self.ObjectToWorld(rv)
                rv = self.WorldToBase(rv)
        elif check_option(null_space_task, "JointVelocity"):
            rqdn = vector(qdot_ns, dim=self.nj)
        else:
            raise ValueError(f"Null-space task '{null_space_task}' not supported")

        rp = copy.deepcopy(rx[:3])
        rR = copy.deepcopy(q2r(rx[3:]))
        _iterations = 0
        qq = q0
        if save_path:
            q_path = q0.reshape((1, self.nj))

        while True:
            p, R, J = self.Kinmodel(qq, out="pR")
            ep = rp - p
            eR = qerr(r2q(rR @ R.T))
            ee = np.hstack((ep, eR))
            if np.all(np.abs(ee[Sind]) < _max_err[Sind]):
                if save_path:
                    return q_path, 0
                else:
                    return qq, 0

            if check_option(task_cont_space, "World"):
                RC = np.kron(np.eye(2), self.TBase[:3, :3]).T
            elif check_option(task_cont_space, "Robot"):
                RC = np.eye(6)
            elif check_option(task_cont_space, "Tool"):
                RC = np.kron(np.eye(2), R).T
            elif check_option(task_cont_space, "Object"):
                RC = np.kron(np.eye(2), self.TObject[:3, :3]).T
            else:
                raise ValueError(f"Task space '{task_cont_space}' not supported")

            ee = RC @ ee
            J = RC @ J
            ux = Kp * ee
            ux = ux[Sind]
            JJ = J[Sind, :]
            Jp = np.linalg.pinv(JJ)
            NS = np.eye(self.nj) - Jp @ JJ

            if check_option(null_space_task, "None"):
                qdn = np.zeros(self.nj)
            elif check_option(null_space_task, "Manipulability"):
                fun = lambda q: self.Manipulability(q, task_space=task_space, task_DOF=task_DOF)
                qdotn = grad(fun, qq)
                qdn = Kns * qdotn
            elif check_option(null_space_task, "JointLimits"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "ConfOptimization"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "PoseOptimization"):
                een = xerr(x_opt, map_pose(p=p, R=R))
                qdn = Kns * np.linalg.pinv(J) @ een
            elif check_option(null_space_task, "TaskVelocity"):
                qdn = np.linalg.pinv(J) @ rv
            elif check_option(null_space_task, "JointVelocity"):
                qdn = rqdn

            uNS = NS @ qdn
            u = Jp @ ux + uNS
            qq = qq + u * self.tsamp

            if save_path:
                q_path = np.vstack((q_path, qq))

            if self.CheckJointLimits(qq):
                self.WarningMessage(f"Joint limits reached: {qq}")
                qq = np.nan * qq
                if save_path:
                    return q_path, 1
                else:
                    return qq, 1

            _iterations += 1
            if _iterations > max_iterations:
                self.WarningMessage(f"No solution found in {_iterations} iterations")
                qq = np.nan * qq
                if save_path:
                    return q_path, 2
                else:
                    return qq, 2

    def IKinPath(
        self,
        path: np.ndarray,
        q0: np.ndarray,
        max_iterations: int = 100,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[np.ndarray] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[np.ndarray] = None,
        v_ns: Optional[np.ndarray] = None,
        qdot_ns: Optional[np.ndarray] = None,
        x_opt: Optional[np.ndarray] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute inverse kinematics for a path of Cartesian poses.

        Parameters
        ----------
        path : np.ndarray
            Path in Cartesian space - poses (n,7) or (n,4,4).
        q0 : np.ndarray
            Initial joint positions (nj,).
        max_iterations : int, optional
            Maximum number of iterations, by default 100.
        pos_err : float, optional
            Position error tolerance, by default None (uses default).
        ori_err : float, optional
            Orientation error tolerance, by default None (uses default).
        task_space : str, optional
            Task space in which to compute the inverse kinematics, by default "Robot".
        task_DOF : np.ndarray, optional
            Degrees of freedom for the task, by default None.
        null_space_task : str, optional
            Type of null-space task, by default None.
        task_cont_space : str, optional
            Task continuity space, by default "Robot".
        q_opt : np.ndarray, optional
            Optimal joint positions for null-space tasks, by default None.
        v_ns : np.ndarray, optional
            Null-space velocity for task-space velocity, by default None.
        qdot_ns : np.ndarray, optional
            Null-space joint velocities, by default None.
        x_opt : np.ndarray, optional
            Optimal Cartesian pose for null-space tasks, by default None.
        Kp : float, optional
            Proportional gain, by default None.
        Kns : float, optional
            Gain for null-space tasks, by default None.

        Returns
        -------
        np.ndarray
            Joint positions at target pose for each point in the path (n, nj).
        """
        if path.ndim == 3:
            _path = uniqueCartesianPath(t2x(path))
        elif ismatrix(path, shape=7):
            _path = uniqueCartesianPath(path)
        else:
            raise ValueError(f"Path shape {path.shape} not supported")

        _n = np.shape(_path)[0]
        _qpath = np.nan * np.zeros((_n, self.nj))
        _q = self.jointvar(q0)
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        for i in range(_n):
            _x = _path[i, :]
            try:
                _q, self._last_status = self.IKin(
                    _x,
                    _q,
                    max_iterations=max_iterations,
                    pos_err=pos_err,
                    ori_err=ori_err,
                    task_space=task_space,
                    task_DOF=task_DOF,
                    null_space_task=null_space_task,
                    task_cont_space=task_cont_space,
                    q_opt=q_opt,
                    v_ns=v_ns,
                    qdot_ns=qdot_ns,
                    x_opt=x_opt,
                    Kp=Kp,
                    Kns=Kns,
                )
                _qpath[i, :] = _q
                if self._last_status != 0:
                    self.Message(f"No IKin solution found for path point sample {i}", 2)
                    return _qpath, self._last_status
            except Exception:
                self.Message(f"No solution found for path point sample {i}", 2)
                self._last_status = 3
                break
        return _qpath, self._last_status

    def Jacobi(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, task_space: str = "Robot") -> JacobianType:
        """
        Compute the Jacobian matrix for the robot given joint positions.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions (nj,). If not provided, the actual joint positions are used.
        tcp : TCPType, optional
            by default the value of `self.TCP` is used.
        task_space : str, optional
            The task space in which to compute the Jacobian, by default "Robot".

        Returns
        -------
        JacobianType
            Jacobian matrix of shape (6, nj) or (6, n), where nj is the number of joints.
        """
        if q is not None:
            qq = self.jointvar(q)
        else:
            qq = self._actual.q
        J = self.Kinmodel(qq, tcp=tcp)[-1]

        # Transform Jacobian to the correct task space
        if check_option(task_space, "World"):
            J = self.WorldToBase(J)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported")
        return J

    def Manipulability(self, q: np.ndarray, task_space: Optional[str] = "Robot", task_DOF: Optional[np.ndarray] = None) -> float:
        """
        Compute the manipulability measure of the robot at a given joint configuration.

        Parameters
        ----------
        q : np.ndarray
            Joint positions (nj,).
        task_space : str, optional
            The task space frame to use for the Jacobian, by default "Robot".
        task_DOF : np.ndarray, optional
            Task Degrees of Freedom (DOF) for the manipulability calculation, by default all 6 DOF are considered.

        Returns
        -------
        float
            Manipulability measure, which is the square root of the determinant of the Jacobian matrix.
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)

        # Get the Jacobian matrix for the given joint positions
        J = self.Jacobi(q, task_space=task_space)

        # Extract the active DOF from the Jacobian
        Sind = np.where(np.asarray(task_DOF) > 0)[0]
        JJ = J[Sind, :]

        # Compute the manipulability as the square root of the determinant of JJ * JJ^T
        return np.sqrt(np.linalg.det(JJ @ JJ.T))

    def TaskDistance(self, x: np.ndarray, out: str = "x", task_space: str = "World", state: str = "Actual", kinematics: str = "Calculated", robot_num: int = None) -> np.ndarray:
        """
        Calculate the distance between the current pose and a target pose.

        Parameters
        ----------
        x : np.ndarray
            The target pose to compare to the current pose.
        out : str, optional
            The output form of the distance, by default "x". Possible values are "x", "p", and "Q".
        task_space : str, optional
            The task space to use for the pose transformation, by default "World".
        state : str, optional
            The state of the current pose, by default "Actual". Other options include "Command" for commanded poses.
        kinematics : str, optional
            The type of kinematics to use, by default "Calculated". Other options might include "Robot".
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        np.ndarray
            The distance between the current and target poses.
        """
        x = np.asarray(x)
        if check_shape(x, (4, 4)):
            rx = t2x(x)
        elif check_shape(x, (7)):
            rx = x
        elif check_shape(x, (3, 3)):
            rx = map_pose(R=x)
            out = "Q"
        elif check_shape(x, (3)):
            rx = map_pose(p=x)
            out = "p"
        elif check_shape(x, (4)):
            rx = map_pose(Q=x)
            out = "Q"
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

        # Compute the difference in pose (task space distance)
        if robot_num is None:
            dx = xerr(rx, self.GetPose(task_space=task_space, state=state, kinematics=kinematics))
        elif robot_num >= 0 and robot_num < self.nr:
            dx = xerr(rx, self.GetPose(task_space=task_space, state=state, kinematics=kinematics)[robot_num])
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

        # Return the appropriate part of the pose distance based on the output form
        if out == "x":
            return dx
        elif out == "Q":
            return dx[..., 3:]
        elif out == "p":
            return dx[..., :3]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    # Load
    def SetLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[np.ndarray] = None, inertia: Optional[np.ndarray] = None, robot_num: Optional[int] = None) -> None:
        """
        Set the load properties of the robot.

        Parameters
        ----------
        load : load_params, optional
            The load object to be assigned, by default None.
        mass : float, optional
            The mass of the load, by default None.
        COM : np.ndarray, optional
            The center of mass of the load, by default None.
        inertia : np.ndarray, optional
            The inertia of the load, by default None.
        robot_num : int, optional
            Number of sub-robot, by default None.
        """
        if robot_num is None:
            raise ValueError("Argument 'robot_num' must be set")
        else:
            if isinstance(load, load_params):
                self.Load[robot_num] = load
            else:
                if mass is not None:
                    self.Load[robot_num].mass = mass
                if COM is not None:
                    self.Load[robot_num].COM = COM
                if inertia is not None:
                    self.Load[robot_num].inertia = inertia

    # TCP
    def SetTCP(self, tcp: Optional[np.ndarray] = None, frame: str = "Gripper", robot_num: Optional[int] = None) -> None:
        """
        Set the Tool Center Point (TCP) of robots in combined system.

        Parameters
        ----------
        tcp : np.ndarray, optional
            The transformation matrix or pose of the TCP. Default is the identity matrix.
        frame : str, optional
            The frame to which the TCP is referenced. Can be "Gripper" or "Flange". Default is "Gripper".
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        None
        """
        if robot_num is None:
            for i in range(self.nr):
                self.robots[i].SetTCP(tcp[i], frame=frame)
        elif robot_num >= 0 and robot_num < self.nr:
            self.robots[robot_num].SetTCP(tcp, frame=frame)
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

        self.TCP: np.ndarray = np.stack([r.TCP for r in self.robots], axis=0)
        rx, rJ = self.Kinmodel(self._command.q)
        self._command.x = self.BaseToWorld(rx)
        self._command.v = self.BaseToWorld((rJ @ self._command.qdot).reshape((self.nr, 6)), typ="Twist")

    def SetTCPGripper(self, tcp: Optional[np.ndarray] = None, robot_num: Optional[int] = None) -> None:
        """
        Set the TCP for the gripper of robots in combined system.

        Parameters
        ----------
        tcp : np.ndarray, optional
            The transformation matrix or pose of the gripper TCP. Default is the identity matrix.
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        None
        """
        if robot_num is None:
            for i in range(self.nr):
                self.robots[i].SetTCPGripper(tcp[i])
        elif robot_num >= 0 and robot_num < self.nr:
            self.robots[robot_num].SetTCPGripper(tcp)
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

        self.TCPGripper: np.ndarray = np.stack([r.TCPGripper for r in self.robots], axis=0)

    def SetObject(self, x: Optional[np.ndarray] = None, robot_num: Optional[int] = None) -> None:
        """
        Set the object pose in the robot's coordinate system for robots in combined system.

        Parameters
        ----------
        x : np.ndarray, optional
            The pose of the object (7,) or (4, 4) (default is None, which sets to the actual object pose).
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        None
        """
        if robot_num is None:
            for i in range(self.nr):
                self.robots[i].SetObject(x[i])
        elif robot_num >= 0 and robot_num < self.nr:
            self.robots[robot_num].SetObject(x)
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

        self.TObject: np.ndarray = np.stack([r.TObject for r in self.robots], axis=0)

    # Base and platform
    def SetBasePose(self, x: np.ndarray, robot_num: int = None) -> None:
        """
        Set the robot base pose for robots in combined system.

        Parameters
        ----------
        x : np.ndarray
            The pose of the base (7,) or (4, 4).
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not recognized.
        """
        if robot_num is None:
            for i in range(self.nr):
                self.robots[i].SetBasePose(x[i])
        elif robot_num >= 0 and robot_num < self.nr:
            self.robots[robot_num].SetBasePose(x)
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

        self.TBase: np.ndarray = np.stack([r.TBase for r in self.robots], axis=0)

    def SetBaseVel(self, v: np.ndarray, robot_num: int = None) -> None:
        """
        Set the robot base velocity for robots in combined system

        Parameters
        ----------
        v : np.ndarray
            The velocity of the base (6,).
        robot_num : int, optional
            Number of sub-robot, by default None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not recognized.
        """
        if robot_num is None:
            if self.spatial(v, shape=(6,)):
                self.vBase = v
                for i in range(self.nr):
                    self.robots[i].vBase(v[i])
            else:
                raise ValueError(f"Base velocity shape {v.shape} not supported")
        elif robot_num >= 0 and robot_num < self.nr:
            if isvector(v, dim=6):
                self.vBase = v
                self.robots[robot_num].vBase(v[robot_num])
            else:
                raise ValueError(f"Base velocity shape {v.shape} not supported")
        else:
            raise ValueError(f"Selected 'robot_num='{robot_num}' is not in range [0, {self.nr - 1}]")

    def UpdateRobotBase(self) -> np.ndarray:
        """
        Update the robot base pose from the base platform, if available.

        Returns
        -------
        np.ndarray
            The updated base pose.
        """
        pass  # TODO Kam je povezana platforma

    # Contacts
    def GetContacts(self, **kwargs) -> Optional[list | np.ndarray]:
        """
        Return contact information.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        list | np.ndarray | None
            Contact information, or ``None`` if no contacts are present.
        """
        _con = None
        for r in self.robots:
            _con_r = r.GetContacts(**kwargs)
            if _con_r is not None:
                if _con is None:
                    _con = []
                _con.append(_con_r)
        return _con

    # Movements
    def Start(self) -> bool:
        """
        Start the robot's motion by setting the control mode to 0.5 and resetting error states.

        Returns
        -------
        None
        """
        for r in self.robots:
            if not r.Start():
                return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.START.value
        self._last_control_time = self.simtime()
        self._abort = False
        self._motion_error = None
        self.Update()
        return True

    def Stop(self) -> None:
        """
        Stop the robot's motion by setting the control mode to 0 and resetting velocities and errors.

        Returns
        -------
        None
        """
        for r in self.robots:
            r.Stop()
        self._command.mode = CommandModeCodes.STOP.value
        self._command.qdot = np.zeros(self.nj)
        self._command.v = np.zeros((self.nr, 6))
        self._abort = False
        self._motion_error = None
        self.reset_threads()
        self.Update()

    def Abort(self, abort: bool = True) -> None:
        """
        Abort the robot's motion by setting the abort flag and changing the control mode to -2.

        Parameters
        ----------
        abort : bool, optional
            Whether to abort the motion. Default is True.

        Returns
        -------
        None
        """
        for r in self.robots:
            r.Abort()
        self.Message("Abort: ", abort)
        self._abort = abort
        self._command.mode = CommandModeCodes.ABORT.value
        self.Update()


class multi_robot(multi_robots):
    """
    Combined dual-robot system

    First task space: robot1
    Second task space: robot2
    """

    def __init__(self, robots: Tuple[robot, ...], robot_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize a combined robot with one task-space output per robot.

        Parameters
        ----------
        robots : Tuple[robot, ...]
            Robots included in the combined robot system.
        robot_name : str, optional
            Name assigned to the combined robot. If omitted, the inherited default is used.
        **kwargs : Any
            Additional keyword arguments forwarded to the parent constructor.

        Returns
        -------
        None
            This constructor initializes the combined robot object in place.
        """
        multi_robots.__init__(self, robots, robot_name=robot_name, **kwargs)
        self.TCP: np.ndarray = np.stack([r.TCP for r in robots], axis=0)  # Tool Center Point transformation matrix
        self.TBase: np.array = np.stack([r.TBase for r in robots], axis=0)  # Robot base
        self.TObject: np.ndarray = np.stack([r.TObject for r in robots], axis=0)  # Object transformation matrix
        self.TCPGripper: np.ndarray = np.stack([r.TCPGripper for r in robots], axis=0)  # Gripper TCP transformation matrix

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[Union[Poses3DType, HomogeneousMatricesType]] = None, out: str = "x") -> Tuple[Poses3DType, JacobianType]:
        """
        Calculate the forward kinematics of the combined dual robot system.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions for both robots (length: robot1.nj + robot2.nj).
            If None, uses the current actual joint state.
        tcp : Union[Poses3DType, HomogeneousMatricesType], optional
            Tool Center Points (TCP) for both robots. If None, uses default TCP.
        out : str, optional
            Output format. Only "x" is supported for combined robots.

        Returns
        -------
        Tuple
            Pose and JacobianType.
        """

        if q is None:
            qq = self._actual.q
        else:
            qq = self.jointvar(q)

        if tcp is None:
            tcp = self.TCP
        else:
            tcp = self.spatial2t(tcp)

        x = np.zeros((self.nr, 7))
        J = np.zeros((self.nr * 6, self.nj))
        _i = 0
        for i in range(self.nr):
            _q = qq[_i : _i + self.robots[i].nj]
            _x, _J = self.robots[i].Kinmodel(_q, tcp=tcp[i], out="x")
            x[i, :] = _x
            J[i * 6 : (i + 1) * 6, _i : _i + self.robots[i].nj] = _J
            _i += self.robots[i].nj

        if out != "x":
            raise ValueError(f"out='{out}' not supported for combined robots")
        else:
            return x, J


class bimanual_robot(multi_robots):
    """
    Combined bimanual-robot system

    Relative task: the base is at the end-effector of the first robot and the
    end-effector is the end-effector of the second robot.

    Absolute task: the base is the base of the first robot and the end-effector
    is the end-effector of the fiirst robot.
    """

    def __init__(self, robots: Tuple[robot, ...], robot_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize a bimanual robot model built from exactly two robots.

        Parameters
        ----------
        robots : Tuple[robot, ...]
            Pair of robots used to form the bimanual system.
        robot_name : str, optional
            Name assigned to the bimanual robot. If omitted, a default name is used.
        **kwargs : Any
            Additional keyword arguments forwarded to the parent constructor.

        Returns
        -------
        None
            This constructor initializes the bimanual robot object in place.
        """
        if len(robots) != 2:
            raise ValueError("Two robots have to be used for a bimanual robot")
        if robot_name is None:
            robot_name = "BimanualRobot"
        multi_robots.__init__(self, robots, robot_name=robot_name, **kwargs)
        self.TBase[1] = self.robots[0].TBase

    def GetState(self) -> None:
        """
        Update and synchronize the internal state of the combined robot.

        This method updates the joint positions, velocities, forces/torques, task space position and velocity,
        and the base pose of the dual robot system. It synchronizes the state of all robots and computes the
        combined state. The method handles the relative force computation and updates the base pose
        and velocity for the system.

        The state synchronization occurs if the time since the last update exceeds a certain threshold, determined
        by the sampling rate (`tsamp`).

        Attributes Updated:
        - Joint positions (`self._actual.q`)
        - Joint velocities (`self._actual.qdot`)
        - Joint torques (`self._actual.trq`)
        - Task space position (`self._actual.x`)
        - Task space velocity (`self._actual.v`)
        - Force/Torque sensor data (`self._actual.FT`)

        Notes
        -----
        - The `self._last_update` attribute is updated to the current simulation time to ensure proper time-based synchronization.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot object.
        """
        if (self.simtime() - self._last_update) > (self.tsamp * 0.1):
            self._tt = self.simtime()
            for r in self.robots:
                r.GetState()
            self._actual.q = np.concatenate([r._actual.q for r in self.robots])
            self._actual.qdot = np.concatenate([r._actual.qdot for r in self.robots])
            self._actual.trq = np.concatenate([r._actual.trq for r in self.robots])

            x, J = self.Kinmodel()
            self._actual.x = x
            self._actual.v = (J @ self._actual.qdot).reshape(self.nr, 6)
            _FT1 = self.robots[0]._actual.FT
            _FT2 = self.robots[1]._actual.FT
            _FTa = _FT1 + _FT2  # Absolute force in world frame

            _R = q2r(x[0, 3:7])
            _FT1 = np.hstack((_R.T @ _FT2[:3], _R.T @ _FT2[3:]))
            _FTr = (_FT2 - _FT1) / 2  # Relative force in robot2 tool frame

            self._actual.FT = np.vstack([_FTr, _FTa])

            # self.TBase[0] = self.robots[0].BaseToWorld(x2t(self.robots[0]._actual.x))
            # self.vBase[0] = self.robots[0].BaseToWorld(self.robots[0]._actual.v, typ="Twist")

            self._last_update = self.simtime()  # Do not change !

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[Union[Pose3DType, HomogeneousMatrixType]] = None, out: str = "x") -> Tuple[Poses3DType, JacobianType]:
        """
        Calculate the forward kinematics of the combined robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions for both robots (length: robot1.nj + robot2.nj).
            If None, uses the current actual joint state.
        tcp : Union[Pose3DType, HomogeneousMatrixType], optional
            Tool Center Point (TCP) of the second robot. If None, uses default TCP.
        out : str, optional
            Output format. Only "x" is supported for combined robots.

        Returns
        -------
        Tuple
            Pose and JacobianType.
        """

        if q is None:
            qq = self._actual.q
        else:
            qq = q
        q1 = qq[: self.robots[0].nj]
        q2 = qq[self.robots[0].nj :]

        if tcp is None:
            tcp = self.TCP[0]
        else:
            tcp = self.robots[0].spatial2t(tcp)

        pb, Rb = map_pose(T=np.linalg.inv(self.robots[0].TBase) @ self.robots[1].TBase, out="pR")

        p1, R1, J1 = self.robots[0].Kinmodel(q1, out="pR")  # robot1
        p1r, R1r, J1r = robot_reverse(p1, R1, J1)  # From EE to base
        p1f, R1f, J1f = join_robot_fixed(p1r, R1r, J1r, pb, Rb)  # From robot1 base to robot2 base
        p2, R2, J2 = self.robots[1].Kinmodel(q2, out="pR")  # robot2
        pr, Rr, Jr = join_robot(p1f, R1f, J1f, p2, R2, J2)  # Added robot2
        xr = map_pose(R=Rr, p=pr, out="x")

        # xa = self.robots[0].BaseToWorld(map_pose(R=R1, p=p1))
        # J1w = self.robots[1].BaseToWorld(J1)
        # Ja = np.hstack((J1w, np.zeros_like(J2)))
        xa, J1w = self.robots[0].Kinmodel(q1, tcp=self.robots[0].TCP @ tcp, out="x")  # robot2
        Ja = np.hstack((J1w, np.zeros_like(J2)))

        x = np.vstack([xr, xa])
        J = np.vstack((Jr, Ja))
        if out == "x":
            return x, J
        else:
            raise ValueError(f"Output form out='{out}' not supported in bimanual robot")

    def SetTCP(self, tcp: Optional[TCPType] = None) -> None:
        """
        Set the Tool Center Point (TCP) of combined robot for absolute pose. This TCP is added to the TCP of the first robot.

        Note that for relative motion only TCP of basic robots in combined system are considered.

        Parameters
        ----------
        tcp : TCPType, optional
            The transformation matrix or pose of the TCP. Default is the identity matrix.

        Returns
        -------
        None
        """
        self.WarningMessage("TCP can not be set!")


class bimanual_robot_mean(multi_robots):
    """
    Combined bimanual-robot system

    Relative task: the base is at the end-effector of the first robot and the
    end-effector is the end-effector of the second robot.

    Absolute task: the base is the base of the first robot and the end-effector
    is the end-effector of the fiirst robot.
    """

    def __init__(self, robots: Tuple[robot, ...], robot_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize a bimanual robot model with a mean absolute task definition.

        Parameters
        ----------
        robots : Tuple[robot, ...]
            Pair of robots used to form the bimanual system.
        robot_name : str, optional
            Name assigned to the bimanual robot. If omitted, a default name is used.
        **kwargs : Any
            Additional keyword arguments forwarded to the parent constructor.

        Returns
        -------
        None
            This constructor initializes the bimanual robot object in place.
        """
        if len(robots) != 2:
            raise ValueError("Two robots have to be used for a bimanual robot")
        if robot_name is None:
            robot_name = "BimanualRobotMean"
        multi_robots.__init__(self, robots, robot_name=robot_name, **kwargs)
        self.TBase[1] = self.robots[0].TBase

    def GetState(self) -> None:
        """
        Update and synchronize the internal state of the combined robot.

        This method updates the joint positions, velocities, forces/torques, task space position and velocity,
        and the base pose of the dual robot system. It synchronizes the state of all robots and computes the
        combined state. The method handles the relative force computation and updates the base pose
        and velocity for the system.

        The state synchronization occurs if the time since the last update exceeds a certain threshold, determined
        by the sampling rate (`tsamp`).

        Attributes Updated:
        - Joint positions (`self._actual.q`)
        - Joint velocities (`self._actual.qdot`)
        - Joint torques (`self._actual.trq`)
        - Task space position (`self._actual.x`)
        - Task space velocity (`self._actual.v`)
        - Force/Torque sensor data (`self._actual.FT`)

        Notes
        -----
        - The `self._last_update` attribute is updated to the current simulation time to ensure proper time-based synchronization.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot object.
        """
        if (self.simtime() - self._last_update) > (self.tsamp * 0.1):
            self._tt = self.simtime()
            for r in self.robots:
                r.GetState()
            self._actual.q = np.concatenate([r._actual.q for r in self.robots])
            self._actual.qdot = np.concatenate([r._actual.qdot for r in self.robots])
            self._actual.trq = np.concatenate([r._actual.trq for r in self.robots])

            x, J = self.Kinmodel()
            self._actual.x = x
            self._actual.v = (J @ self._actual.qdot).reshape(self.nr, 6)
            # _FT1 = self.robots[0]._actual.FT
            # _FT2 = self.robots[1]._actual.FT
            # _FTa = _FT1 + _FT2  # Absolute force in world frame

            # _R = q2r(x[0, 3:7])
            # _FT1 = np.hstack((_R.T @ _FT2[:3], _R.T @ _FT2[3:]))
            # _FTr = (_FT2 - _FT1) / 2  # Relative force in robot2 tool frame

            # self._actual.FT = np.vstack([_FTr, _FTa])

            # self.TBase[0] = self.robots[0].BaseToWorld(x2t(self.robots[0]._actual.x))
            # self.vBase[0] = self.robots[0].BaseToWorld(self.robots[0]._actual.v, typ="Twist")

            self._last_update = self.simtime()  # Do not change !

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[Union[Pose3DType, HomogeneousMatrixType]] = None, out: str = "x") -> Tuple[Poses3DType, JacobianType]:
        """
        Calculate the forward kinematics of the combined robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions for both robots (length: robot1.nj + robot2.nj).
            If None, uses the current actual joint state.
        tcp : Union[Pose3DType, HomogeneousMatrixType], optional
            Tool Center Point (TCP) of the second robot. If None, uses default TCP.
        out : str, optional
            Output format. Only "x" is supported for combined robots.

        Returns
        -------
        Tuple
            Pose and JacobianType.
        """

        if q is None:
            qq = self._actual.q
        else:
            qq = q
        q1 = qq[: self.robots[0].nj]
        q2 = qq[self.robots[0].nj :]

        if tcp is None:
            tcp = self.robots[1].TCP

        pb, Rb = map_pose(T=np.linalg.inv(self.robots[0].TBase) @ self.robots[1].TBase, out="pR")

        p1, R1, J1 = self.robots[0].Kinmodel(q1, out="pR")  # robot1
        p1r, R1r, J1r = robot_reverse(p1, R1, J1)  # From EE to base
        p1f, R1f, J1f = join_robot_fixed(p1r, R1r, J1r, pb, Rb)  # From robot1 base to robot2 base
        p2, R2, J2 = self.robots[1].Kinmodel(q2, tcp, out="pR")  # robot2
        pr, Rr, Jr = join_robot(p1f, R1f, J1f, p2, R2, J2)  # Added robot2
        xr = map_pose(R=Rr, p=pr, out="x")

        x1w = self.robots[0].BaseToWorld(map_pose(R=R1, p=p1))
        J1w = self.robots[0].BaseToWorld(J1)

        x2w = self.robots[1].BaseToWorld(map_pose(R=R2, p=p2))
        J2w = self.robots[1].BaseToWorld(J2)

        # xa = xmean(np.vstack((x2w, x1w)))
        qa = np.squeeze(slerp(x1w[3:], x2w[3:], 0.5))
        xa = np.concatenate(((x2w[:3] + x1w[:3]) / 2, qa))
        Ja = np.hstack((J1w, J2w)) / 2

        x = np.vstack([xr, xa])
        J = np.vstack((Jr, Ja))
        if out == "x":
            return x, J
        else:
            raise ValueError(f"Output form out='{out}' not supported in bimanual robot")

    def SetTCP(self, tcp: Optional[TCPType] = None) -> None:
        """
        Set the Tool Center Point (TCP) of combined robot for absolute pose. This TCP is added to the TCP of the first robot.

        Note that for relative motion only TCP of basic robots in combined system are considered.

        Parameters
        ----------
        tcp : TCPType, optional
            The transformation matrix or pose of the TCP. Default is the identity matrix.

        Returns
        -------
        None
        """
        self.WarningMessage("TCP can not be set!")
