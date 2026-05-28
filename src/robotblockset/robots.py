"""Core robot class implementation.

This module defines a `robot` class that represents a robotic system with various configurations, transformations, and control mechanisms.
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

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Dict, Optional, Tuple, Any, Union
import numpy as np
from time import perf_counter, sleep
from threading import Semaphore, Thread
import platform
import copy
from enum import Enum, IntEnum
from pathlib import Path

from robotblockset.tools import _struct, load_params, rbs_object, rbs_type, check_option, vector, isscalar, isvector, ismatrix, grad, normalize, damped_pinv, tool_params, load_tools_from_yaml
from robotblockset.trajectories import jtraj, ctraj, carctraj, interpPath, interpCartesianPath, gradientPath, gradientCartesianPath, uniqueCartesianPath
from robotblockset.rbf import decodeRBF, decodeCartesianRBF
from robotblockset.transformations import map_pose, q2r, r2q, x2x, x2t, t2x, v2s, xerr, qerr, terr, frame2world, world2frame, spatial2t
from robotblockset.rbs_typing import (
    ArrayLike,
    HomogeneousMatrixArrayType,
    JointConfigurationType,
    JointVelocityType,
    JointAccelerationType,
    JointPathType,
    JointTorqueType,
    Pose3DType,
    Poses3DType,
    QuaternionType,
    RotationMatrixType,
    HomogeneousMatrixType,
    HomogeneousMatricesType,
    Velocity3DType,
    Velocities3DType,
    Vector3DType,
    WrenchType,
    TCPType,
    TimesType,
    JacobianType,
    CartesianPathType,
)

np.set_printoptions(formatter={"float": "{: 0.4f}".format})

flag = True


def _dummy() -> None:
    """Clear the global worker-activity flag."""
    global flag
    flag = False


class MotionResultCodes(IntEnum):
    """
    Result codes returned by robot motion commands.

    Vrednosti
    ---------
    MOTION_SUCCESS : int
        ``0``. Motion completed successfully.
    MOTION_FAILURE : int
        ``1``. Motion failed.
    MOTION_ABORTED : int
        ``2``. Motion was aborted.
    JOINT_LIMITS : int
        ``3``. A joint limit was reached.
    CLOSE_TO_TARGET : int
        ``4``. The robot is already close to the requested target.
    ACTIVE_THREADS : int
        ``5``. Active worker threads prevent execution of the command.
    WRONG_STRATEGY : int
        ``6``. The selected motion strategy is not valid for the requested command.
    NOT_FEASIBLE : int
        ``7``. The requested motion is not feasible.
    NO_ROBOT_ATTACHED : int
        ``8``. No robot is attached or available.
    RTDE_ERROR : int
        ``9``. An RTDE communication error occurred.
    NO_RESPONSE : int
        ``10``. No response was received from the robot or controller.
    NOT_READY : int
        ``11``. The system is not ready to execute motion.
    """

    MOTION_SUCCESS = 0
    MOTION_FAILURE = 1
    MOTION_ABORTED = 2
    JOINT_LIMITS = 3
    CLOSE_TO_TARGET = 4
    ACTIVE_THREADS = 5
    WRONG_STRATEGY = 6
    NOT_FEASIBLE = 7
    NO_ROBOT_ATTACHED = 8
    RTDE_ERROR = 9
    NO_RESPONSE = 10
    NOT_READY = 11


MOTION_RESULT_DESC = {
    MotionResultCodes.MOTION_SUCCESS: "Motion completed successfully",
    MotionResultCodes.MOTION_FAILURE: "Motion failed",
    MotionResultCodes.MOTION_ABORTED: "Motion was aborted",
    MotionResultCodes.JOINT_LIMITS: "Joint limits reached",
    MotionResultCodes.CLOSE_TO_TARGET: "Already close to target",
    MotionResultCodes.ACTIVE_THREADS: "Active threads prevent execution",
    MotionResultCodes.WRONG_STRATEGY: "Wrong motion strategy selected",
    MotionResultCodes.NOT_FEASIBLE: "Requested motion is not feasible",
    MotionResultCodes.NO_ROBOT_ATTACHED: "No robot attached",
    MotionResultCodes.RTDE_ERROR: "RTDE communication error",
    MotionResultCodes.NO_RESPONSE: "No response from robot/controller",
    MotionResultCodes.NOT_READY: "System not ready for motion",
}


def MotionResultStr(code: Union[MotionResultCodes, int]) -> str:
    """
    Convert motion result code to a descriptive human-readable string.

    Parameters
    ----------
    code : Union[MotionResultCodes, int]
        Result code to describe.

    Returns
    -------
    str
        Human-readable description of the result code.
    """
    # Allow raw int inputs
    if not isinstance(code, MotionResultCodes):
        try:
            code = MotionResultCodes(code)
        except ValueError:
            return f"Unknown motion result ({code})"

    return MOTION_RESULT_DESC.get(code, f"Unknown motion result ({code.name} = {code.value})")


class CommandModeCodes(Enum):
    """
    Major command mode codes used by robot motion commands.

    Attributes
    ----------
    ABORT : float
        ``-2``. Abort motion immediately.
    WAIT : float
        ``-1``. Hold position and wait.
    STOP : float
        ``0``. Stop motion and maintain the current position.
    START : float
        ``0.5``. Start motion and enter the preparation phase.
    JOINT : float
        ``1``. Joint-space motion control.
    JOINT_PATH : float
        ``1.1``. Follow a predefined joint-space path.
    JOINT_RBFPATH : float
        ``1.2``. Follow a joint-space path defined by a radial basis function.
    JOINT_TRAJ : float
        ``1.3``. Follow a generated joint-space trajectory.
    CARTESIAN : float
        ``2``. Cartesian motion control.
    CARTESIAN_NONE : float
        ``2.1``. Cartesian motion without null-space optimization.
    CARTESIAN_MANUPULABILITY : float
        ``2.2``. Cartesian motion with manipulability optimization.
    CARTESIAN_JOINTLIMITS : float
        ``2.3``. Cartesian motion with joint-limit avoidance.
    CARTESIAN_CONFIGURATION : float
        ``2.4``. Cartesian motion with configuration optimization.
    CARTESIAN_POSE : float
        ``2.5``. Cartesian pose-based control.
    CARTESIAN_TASKVELOCITY : float
        ``2.6``. Cartesian velocity control in task space.
    CARTESIAN_JOINTVELOCITY : float
        ``2.7``. Cartesian motion with joint-space velocity control.
    CARTESIAN_USER : float
        ``2.9``. Cartesian motion with a user-defined optimization rule.
    PLANAR : float
        ``3``. Generic planar motion control.
    PLANAR_POS : float
        ``3.1``. Planar position control.
    PLANAR_ORI : float
        ``3.2``. Planar orientation control.
    PLANAR_LOCATION : float
        ``3.3``. Planar motion to a specific location.
    PLANAR_FORWARD : float
        ``3.4``. Planar forward motion.
    PLANAR_TURN : float
        ``3.5``. Planar turning motion.
    AUTONOMOUS : float
        ``4``. Autonomous path planning or execution.
    JOGGING : float
        ``5``. Manual jogging or teach-mode movement.
    INTERNAL_JOINT : float
        ``11``. Internal joint-space control.
    INTERNAL_JOINT_CARTESIAN : float
        ``11.1``. Internal joint-space control for task-space motion.
    INTERNAL_CARTESIAN : float
        ``12``. Internal Cartesian control.
    INTERNAL_CARTESIAN_JOINT : float
        ``12.1``. Internal Cartesian control for joint-space motion.
    """

    ABORT = -2
    WAIT = -1
    STOP = 0
    START = 0.5
    JOINT = 1
    JOINT_PATH = 1.1
    JOINT_RBFPATH = 1.2
    JOINT_TRAJ = 1.3
    CARTESIAN = 2
    CARTESIAN_NONE = 2.1
    CARTESIAN_MANUPULABILITY = 2.2
    CARTESIAN_JOINTLIMITS = 2.3
    CARTESIAN_CONFIGURATION = 2.4
    CARTESIAN_POSE = 2.5
    CARTESIAN_TASKVELOCITY = 2.6
    CARTESIAN_JOINTVELOCITY = 2.7
    CARTESIAN_USER = 2.9
    PLANAR = 3
    PLANAR_POS = 3.1
    PLANAR_ORI = 3.2
    PLANAR_LOCATION = 3.3
    PLANAR_FORWARD = 3.4
    PLANAR_TURN = 3.5
    AUTONOMOUS = 4
    JOGGING = 5
    INTERNAL_JOINT = 11
    INTERNAL_JOINT_CARTESIAN = 11.1
    INTERNAL_CARTESIAN = 12
    INTERNAL_CARTESIAN_JOINT = 12.1


COMMAND_MODE_DESC = {
    CommandModeCodes.ABORT: "Abort motion immediately",
    CommandModeCodes.WAIT: "Hold position and wait",
    CommandModeCodes.STOP: "Stop motion and maintain current position",
    CommandModeCodes.START: "Start motion (preparation phase)",
    CommandModeCodes.JOINT: "Joint-space movement control",
    CommandModeCodes.JOINT_PATH: "Follow predefined joint-space path",
    CommandModeCodes.JOINT_RBFPATH: "Joint-space path using Radial Basis Function (smooth trajectory)",
    CommandModeCodes.JOINT_TRAJ: "Joint-space folow trajectory control",
    CommandModeCodes.CARTESIAN: "Cartesian motion control",
    CommandModeCodes.CARTESIAN_NONE: "Cartesian motion without null-space optimization",
    CommandModeCodes.CARTESIAN_MANUPULABILITY: "Cartesian with manipulability optimization",
    CommandModeCodes.CARTESIAN_JOINTLIMITS: "Cartesian with joint limits avoidance",
    CommandModeCodes.CARTESIAN_CONFIGURATION: "Cartesian with configuration (elbow/wrist) optimization",
    CommandModeCodes.CARTESIAN_POSE: "Cartesian pose-based control",
    CommandModeCodes.CARTESIAN_TASKVELOCITY: "Cartesian velocity control in task space",
    CommandModeCodes.CARTESIAN_JOINTVELOCITY: "Cartesian motion with joint-space velocity control",
    CommandModeCodes.CARTESIAN_USER: "Custom user-defined Cartesian optimization",
    CommandModeCodes.PLANAR: "2D planar motion",
    CommandModeCodes.PLANAR_POS: "Planar motion –¸ position only",
    CommandModeCodes.PLANAR_ORI: "Planar motion – orientation control",
    CommandModeCodes.PLANAR_LOCATION: "Planar motion to specific location",
    CommandModeCodes.PLANAR_FORWARD: "Planar forward movement",
    CommandModeCodes.PLANAR_TURN: "Planar turning movement",
    CommandModeCodes.AUTONOMOUS: "Autonomous path planning or execution",
    CommandModeCodes.JOGGING: "Manual jogging (teach-mode movement)",
    CommandModeCodes.INTERNAL_JOINT: "Internal joint-space control ",
    CommandModeCodes.INTERNAL_JOINT_CARTESIAN: "Internal joint-space control - motion in task space",
    CommandModeCodes.INTERNAL_CARTESIAN: "Internal Cartesian control ",
    CommandModeCodes.INTERNAL_CARTESIAN_JOINT: "Internal Cartesian control - motion in joint-space",
}


def CommandModeStr(mode: Union[CommandModeCodes, float, int]) -> str:
    """
    Convert command mode code to a descriptive human-readable string.

    Parameters
    ----------
    mode : Union[CommandModeCodes, float, int]
        Command mode to describe.

    Returns
    -------
    str
        Human-readable description of the command mode.
    """
    # Allow raw float/int inputs
    if not isinstance(mode, CommandModeCodes):
        try:
            mode = CommandModeCodes(mode)
        except ValueError:
            return f"Unknown command mode ({mode})"

    # Use the numeric value in fallback
    return COMMAND_MODE_DESC.get(mode, f"Unknown command mode ({mode.name} = {mode.value})")


class _actual(_struct):
    """
    Represents the actual state of the robot, including joint positions, velocities, torques, and more.

    Attributes
    ----------
    q : Optional[np.ndarray]
        Joint positions, represented as a numpy array (nj,).
    qdot : Optional[np.ndarray]
        Joint velocities, represented as a numpy array (nj,).
    trq : Optional[np.ndarray]
        Joint torques, represented as a numpy array (nj,).
    x : Optional[np.ndarray]
        End-effector pose, represented as a numpy array (7,) or (4, 4).
    v : Optional[np.ndarray]
        End-effector velocity, represented as a numpy array (6,).
    FT : Optional[np.ndarray]
        Force/Torque sensor data (in robot frame), represented as a numpy array (6,).
    trqExt : Optional[np.ndarray]
        External torques, represented as a numpy array (nj,).
    """

    def __init__(self) -> None:
        """
        Initialize the actual robot state container.

        Notes
        -----
        All stored state fields are initialized to ``None`` and are populated
        later by robot-specific state updates.
        """
        self.q: Optional[np.ndarray] = None
        self.qdot: Optional[np.ndarray] = None
        self.trq: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.FT: Optional[np.ndarray] = None
        self.trqExt: Optional[np.ndarray] = None


class _command(_struct):
    """
    Represents the commanded state of the robot, including desired joint positions, velocities, and torques.

    Attributes
    ----------
    q : Optional[np.ndarray]
        Desired joint positions, represented as a numpy array (nj,).
    qdot : Optional[np.ndarray]
        Desired joint velocities, represented as a numpy array (nj,).
    trq : Optional[np.ndarray]
        Desired joint torques, represented as a numpy array (nj,).
    x : Optional[np.ndarray]
        Desired end-effector pose in robot frame, represented as a numpy array (7,) or (4, 4).
    rx : Optional[np.ndarray]
        Desired end-effector pose, represented as a numpy array (7,) or (4, 4).
    v : Optional[np.ndarray]
        Desired end-effector velocity in robot frame, represented as a numpy array (6,).
    rv : Optional[np.ndarray]
        Desired end-effector velocity, represented as a numpy array (6,).
    FT : Optional[np.ndarray]
        Desired force/torque sensor data, represented as a numpy array (6,).
    u : Optional[np.ndarray]
        Task control inputs for kinematic controller. Default is None.
    ux : Optional[np.ndarray]
        Joint control inputs for kinematic controller. Default is None.
    data : Any
        Additional data for the robot. Can be of any type.
    mode : Optional[int]
        Command mode, represented as an integer.
    """

    def __init__(self) -> None:
        """
        Initialize the command robot state container.

        Notes
        -----
        All stored state fields are initialized to ``None`` and are populated
        later by robot-specific state updates.
        """
        self.q: Optional[np.ndarray] = None
        self.qdot: Optional[np.ndarray] = None
        self.trq: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.rx: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.rv: Optional[np.ndarray] = None
        self.FT: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.ux: Optional[np.ndarray] = None
        self.data: Any = None
        self.mode: Optional[int] = None


class _default(_struct):
    """
    Represents the default values of robot parameters used in the robot class.

    Attributes
    ----------
    State : str
        The state of the robot, "Actual" or "Commanded"
    TaskSpace : str
        The space in which the robot operates, "World", "Robot", "Object" or "Tool".
    TaskPoseForm : str
        The form in which task poses are represented, typically "Pose".
    TaskOriForm : str
        The form in which task orientations are represented, typically "Quaternion".
    TaskVelForm : str
        The form in which task velocities are represented, typically "Twist", "Linear" or "Angular".
    TaskFTForm : str
        The form in which force/torque sensor data is represented, typically "Wrench", "Force" or "Torque".
    TaskErrForm : str
        The form in which task errors are represented, typically "Task".
    Kinematics : str
        The kinematics model used, typically "Robot" or "Calculated".
    Refresh : bool
        Whether to refresh the robot's state or not.
    TCPFrame : str
        The tool center point frame, typically "Gripper".
    Source : str
        The source of the robot's configuration, "Robot" or "External".
    Strategy : str
        The robot control strategy, typically "JointPosition" .
    TaskContSpace : str
        The task control space for the robot, "Robot", "World", "Object", or "Tool".
    NullSpaceTask : str
        The task in the null space of the robot, "None", "Manipulability", "JointLimits",
        "ConfOptimization", "PoseOptimization", "TaskVelocity" or "JointVelocity".
    DampedPseudoInverseFactor : float
        Damping factor for the pseudo-inverse computation.
    RotDirShort : bool
        Whether the rotation direction is short or not.
    Traj : str
        The trajectory type used for robot movement, "Jerk", "Poly", "Trap" or "Line".
    TaskDOF : np.ndarray
        The degrees of freedom of the task, represented as an array of 1's and 0's of length 6.
    VelocityScaling : float
        A scaling factor for velocity, typically set to 1.
    MinJointDist : float
        Minimum distance to target joint positions for movement execution
    MinPosDist : float
        The minimum position distance to target positions for movement execution.
    MinOriDist : float
        The minimum distance between actual and target orientation for movement execution.
    PosErr : float
        The position error tolerance.
    OriErr : float
        The orientation error tolerance.
    AddedTrq : Optional[np.ndarray]
        Additional joint torques, represented as a numpy array (nj,). Default is None.
    AddedFT : np.ndarray
        Additional end-effector force/torque sensor data, represented as a numpy array (6,).
    Kp : float
        The proportional gain for the kinematic controller (position).
    Kff : float
        The feed-forward gain for the kinematic controller (velocity).
    Kns : float
        The null-space gain for the kinematic controller.
    Kns0 : float
        The null-space gain for joint limits.
    Wait : float
        The waiting time after robot move operations (in seconds).
    TCTimeout : float
        The timeout for task kinematic controller (in seconds).
    UpdateTime : float
        The time interval between updates (in seconds).
    TrajSampTimeFac : float
        The factor for trajectory sampling time.
    JointVelocity : float
        The maximum joint velocity for the joint motion.
    JointAcceleration : float
        The maximum joint acceleration for the joint motion.
    JointDeceleration : float
        The maximum joint deceleration for stopping the joint motion.
    JointJerk : float
        The maximum joint jerk for the joint motion.
    TaskVelocity : float
        The maximum task velocity for the Cartesian motion.
    TaskAcceleration : float
        The maximum task acceleration for the Cartesian motion.
    TaskDeceleration : float
        The maximum task deceleration for stopping the Cartesian motion.
    TaskJerk : float
        The maximum task jerk for the Cartesian motion.
    """

    def __init__(self) -> None:
        """
        Initializes the default values for the robot parameters.

        Attributes are initialized with typical values for the robot's configuration.
        """
        self.State: str = "Actual"
        self.TaskSpace: str = "World"
        self.TaskPoseForm: str = "Pose"
        self.TaskOriForm: str = "Quaternion"
        self.TaskVelForm: str = "Twist"
        self.TaskFTForm: str = "Wrench"
        self.TaskErrForm: str = "Task"
        self.Kinematics: str = "Robot"
        self.Refresh: bool = True
        self.TCPFrame: str = "Gripper"
        self.Source: str = "Robot"
        self.Strategy: str = "JointPosition"
        self.TaskContSpace: str = "Robot"
        self.NullSpaceTask: str = "JointLimits"
        self.UseInternal: bool = False
        self.DampedPseudoInverseFactor: float = 0.0
        self.RotDirShort: bool = True
        self.Traj: str = "Jerk"
        self.TaskDOF: np.ndarray = np.ones(6)
        self.VelocityScaling: float = 0.25  # Scale factor for velocity
        self.MinJointDist: float = 0
        self.MinPosDist: float = 0
        self.MinOriDist: float = 0
        self.PosErr: float = 0.0001
        self.OriErr: float = 0.001
        self.AddedTrq: Optional[np.ndarray] = None  # Added joint torques (depends on nj, which is not yet defined)
        self.AddedFT: np.ndarray = np.zeros(6)  # Added end-effector FT
        self.Kp: float = 10  # Kinematic controller: position P gain
        self.Kff: float = 1  # Kinematic controller: velocity FF gain
        self.Kns: float = 1  # Kinematic controller: null-space gain
        self.Kns0: float = 0.1  # Kinematic controller: null-space gain for joint limits
        self.Wait: float = 0
        self.TCTimeout: float = 0.5
        self.UpdateTime: float = 1.0
        self.TrajSampTimeFac: float = 5
        self.JointVelocity: float = 1.0
        self.JointAcceleration: float = 5.0
        self.JointDeceleration: float = 5.0
        self.JointJerk: float = 10.0
        self.TaskVelocity: float = 0.25
        self.TaskAcceleration: float = 5
        self.TaskDeceleration: float = 5
        self.TaskJerk: float = 10.0


class robot(rbs_object):
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
        Transformation matrix for the robot's gripper nominal TCP.
    Tool : tool_params
        Parameters of the selected tool.
    Load : load_params
        Load associated with the robot.
    Gripper : Optional[Any]
        Gripper object attached to the robot, if any.
    FTSensor : Optional[Any]
        Force/Torque sensor attached to the robot, if any.
    FTSensorFrame : np.ndarray
        Transformation matrix of the F/T sensor frame relative to the end-effector.
    FTFrame: np.ndarray
        Transformation matrix relative to flange in which forces and torques are expressed
    Platform : Optional[Any]
        Platform object to which the robot is attached.
    User : Optional[Any]
        User-defined data or object.
    Tag : Optional[Any]
        Tag associated with the robot.
    Motion_result_code :  _motion_result_codes
        RBS Result codes reported by motion commands
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the robot with default values and optional configurations.

        Parameters
        ----------
        **kwargs : Any
            Optional arguments for custom configuration or parameters.
        """
        # Initialize parent class
        rbs_object.__init__(self)

        # Default configurations and values
        self.Name: str = "Robot"
        self.tsamp: float = 0.01  # Sampling rate for the robot
        self.TCP: np.ndarray = np.eye(4)  # Tool Center Point transformation matrix
        if not hasattr(self, "TBase"):
            self.TBase: np.ndarray = np.eye(4)  # Robot base transformation matrix
        self.vBase: np.ndarray = np.zeros(6)  # Robot base velocity
        self.TObject: np.ndarray = np.eye(4)  # Object transformation matrix
        if not hasattr(self, "TCPGripper"):
            self.TCPGripper: np.ndarray = np.eye(4)  # Gripper nominal TCP transformation matrix - Has to be defined in robot_spec!
        self.Tool: tool_params = None  # Tool parameters
        self.Load: load_params = load_params()  # Load object
        self.Gripper: Optional[Any] = None  # Gripper object attached to robot
        self.EEFixed: bool = False  # End-effector is fixed to external object
        self.FTSensor: Optional[Any] = None  # Force/Torque sensor attached to robot
        self.FTSensorFrame: np.ndarray = None  # F/T sensor transformation matrix relative to flange
        self.FTFrame: np.ndarray = np.eye(4)  # Transformation matrix relative to flange in which forces and torques are expressed
        self.Camera: Optional[Any] = None  # Camera object attached to robot
        self.CameraFrame: np.ndarray = None  # Camera transformation matrix relative to flange
        self.Platform: Optional[Any] = None  # Platform object to which robot is attached
        self.User: Optional[Any] = None  # User data or object
        self.Tag: Optional[Any] = None  # Tag associated with the robot

        # Time-related attributes
        self._t0: float = 0  # Initial wall time
        self._tt: float = 0  # Actual robot time
        self._tt0: float = 0  # Initial robot time
        self._robottime: float = 0  # Time from simulator
        self._robottime0: float = 0  # Initial time from simulator
        self._connected: bool = False  # Connection status
        self._last_update: float = -100  # Last update timestamp
        self._last_control_time: float = -100  # Last control time timestamp

        # Robot states
        self._command: _command = _command()  # Commanded values
        self._actual: _actual = _actual()  # Measured values
        self._default: _default = _default()  # Default options
        self._do_update: bool = True  # Enables state update
        self._do_capture: bool = False  # Enables callback capture
        self._capture_callback: Optional[Callable] = None  # Callback during capture
        self._do_motion_check: bool = False  # Enables checks during motion
        self._motion_check_callback: Optional[Callable] = None  # Callback during motion checks
        self._motion_error: Optional[Any] = None  # Motion error status
        self._control_strategy: str = "JointPosition"  # Control strategy
        self._user_null_space_task_callback: Optional[Callable] = None  # Callback during GoTo_TC with null-space_task set to 'User'
        self._semaphore: Semaphore = Semaphore(1)  # Semaphore for motion control
        self._threads_active: bool = platform.system() == "Linux"  # Threads active on Linux
        self._abort: bool = False  # Flag to abort current motion
        self._last_status: int = 0  # Last motion command status

    def reset_threads(self) -> None:
        """
        Resets the internal semaphore to its initial state.

        This method reinitializes the semaphore with a value of 1,
        effectively allowing one thread to acquire it.

        Returns
        -------
        None
        """
        self._semaphore = Semaphore(1)
    def jointvar(self, x: ArrayLike) -> JointConfigurationType:
        """
        Validates and returns the input `x` if it has the proper shape for joint positions.

        Parameters
        ----------
        x : ArrayLike
            Input array representing joint positions, which must have the shape (n, nj),
            where n is the number of samples and nj is the number of joints.

        Returns
        -------
        JointConfigurationType
            The input array `x` if it has the correct shape.

        Raises
        ------
        TypeError
            If the input `x` does not have the proper shape.

        Note
        ----
        Assuming 'nj' represents the number of joints, defined elsewhere in the class
        """
        x = np.asarray(x)
        if x.shape[-1] == self.nj:
            return x
        else:
            raise TypeError("Parameter has not proper shape")

    def spatial(self, x: ArrayLike) -> np.ndarray:
        """
        Validates the shape of the input `x` and returns it in an appropriate format.

        Parameters
        ----------
        x : ArrayLike
            The input array representing a spatial quantity, which can be one of the following shapes:

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
        if x.shape == (7,) or x.shape == (4, 4) or x.shape == (3,) or x.shape == (4,) or x.shape == (3, 3) or x.shape == (6,):
            return x
        elif x.shape == (3, 4):
            x = np.vstack((x, np.array([0, 0, 0, 1])))
            return x
        else:
            raise TypeError("Parameter has not proper shape")

    def simtime(self) -> float:
        """
        Returns the current simulation time based on the system's performance counter.

        The `perf_counter` function provides a high-resolution timer that is useful for measuring
        time intervals. It is system-dependent and returns the time as a floating-point value in seconds.

        Returns
        -------
        float
            The current simulation time in seconds since an arbitrary point (usually the start of the program).
        """
        return perf_counter()

    def _sleep(self, time: float) -> None:
        """
        Pause execution for the given duration.

        Parameters
        ----------
        time : float
            Time in seconds to sleep.
        """
        sleep(time)

    def _synchro_control(self, wait: float) -> None:
        """
        Synchronizes the control loop by waiting for the specified time interval to elapse.

        This method calculates the time difference since the last control update, and if the elapsed
        time is less than the specified wait time, it waits for the remaining time. If the system
        it sleeps for half of the remaining wait time to avoid blocking the control loop when threading is active.

        Parameters
        ----------
        wait : float
            The amount of time (in seconds) to wait before proceeding with the next control update.

        Returns
        -------
        None
            This method does not return any value, it only modifies the internal state of the robot.
        """
        remaining = self._last_control_time + wait - self.simtime()
        if remaining > 0:
            self._sleep(remaining)
        self._last_control_time = self.simtime()

    def UseThreads(self, active: bool) -> None:
        """
        Enables or disables the use of threads for robot control.

        This method sets the `_threads_active` attribute, which controls whether the robot's
        control system should use threads for asynchronous operations. When threads are enabled,
        the robot will perform certain operations in parallel to improve performance, especially
        during control updates or simulations.

        Parameters
        ----------
        active : bool
            A boolean value that indicates whether threads should be active.
            If `True`, threads are enabled. If `False`, threads are disabled.

        Returns
        -------
        None
            This method does not return any value.
        """
        self._threads_active = active

    def SetTsamp(self, tsamp: float) -> None:
        """
        Set the sampling time (`tsamp`) for the robot control system.

        This method updates the `tsamp` attribute, which is used to define the time interval
        between control updates or simulations. Additionally, it updates the `Wait` attribute
        in the `_default` object to reflect the new sampling time.

        Parameters
        ----------
        tsamp : float
            The new sampling time in seconds to set for the robot control system.

        Returns
        -------
        None
            This method does not return any value.
        """
        self.tsamp = tsamp
        self._default.Wait = tsamp

    def ResetTime(self) -> None:
        """
        Reset the robot's time to the current simulation time.

        This method updates the internal time attributes `_t0` and `_tt0` to the current
        simulation time and the current robot time, respectively. It also retrieves the robot's
        state and updates any internal state information.

        Returns
        -------
        None
            This method does not return any value.
        """
        self.GetState()
        self._t0 = self.simtime()
        self._tt0 = copy.deepcopy(self._tt)
        self._robottime0 = copy.deepcopy(self._robottime)
        self.Update()

    def isConnected(self) -> bool:
        """
        Checks if the robot is connected.

        Returns
        -------
        bool
            True if the robot is connected, False otherwise.
        """
        return self._connected

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
        return self._connected

    def isActive(self) -> bool:
        """
        Check if the robot is active.

        This method always returns `True`, indicating that the robot is in an active state.
        It can be overridden in subclasses for more complex behavior.

        Returns
        -------
        bool
            Always returns `True`, indicating the robot is active.
        """
        return True

    def inMotion(self) -> bool:
        """
        Check if the robot is in motion.

        Returns
        -------
        bool
            `True` indicating the robot is excetuting motion command.
        """
        return self._command.mode > CommandModeCodes.STOP.value

    def Check(self, silent: bool = False) -> list[str]:
        """
        Checks the status of the robot.

        Parameters
        ----------
        silent : bool, optional
            If `True`, suppress status messages while checking the robot state.

        Returns
        -------
        list[str]
            A list containing the list of non-active status and status description.
        """
        return []

    def HasError(self) -> bool:
        """
        Checks is robot has errors

        Returns
        -------
        --------
        bool
            `True` is robots has erros.
        """
        return len(self.Check(silent=True)) > 0

    @property
    def Time(self) -> float:
        """
        Get the elapsed wall time since the robot was initialized.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        return self.simtime() - self._t0

    @property
    def t(self) -> float:
        """
        Get the elapsed time since the robot was initiated.

        Returns
        -------
        float
            Time difference in seconds.
        """
        return self._tt - self._tt0

    @property
    def trob(self) -> float:
        """
        Get the elapsed robot time since the robot was initiated.

        Returns
        -------
        float
            Time difference in seconds.
        """
        return self._robottime - self._robottime0

    @property
    def command(self) -> _command:
        """
        Get the commanded state of the robot.

        Returns
        -------
        _command
            A copy of the commanded state.
        """
        return copy.deepcopy(self._command)

    @property
    def actual(self) -> _actual:
        """
        Get the actual state of the robot.

        Returns
        -------
        _actual
            A copy of the actual state.
        """
        return copy.deepcopy(self._actual)

    @property
    def q(self) -> JointConfigurationType:
        """
        Get the current joint positions.

        Returns
        -------
        np.ndarray
            Joint positions (nj,).
        """
        return copy.deepcopy(self._actual.q)

    @property
    def qdot(self) -> JointConfigurationType:
        """
        Get the current joint velocities.

        Returns
        -------
        np.ndarray
            Joint velocities (nj,).
        """
        return copy.deepcopy(self._actual.qdot)

    @property
    def trq(self) -> JointConfigurationType:
        """
        Get the current joint torques.

        Returns
        -------
        np.ndarray
            Joint torques (nj,).
        """
        return copy.deepcopy(self._actual.trq)

    @property
    def trqExt(self) -> JointConfigurationType:
        """
        Get the current external joint torques.

        Returns
        -------
        np.ndarray
            External joint torques (nj,).
        """
        return copy.deepcopy(self._actual.trqExt)

    @property
    def x(self) -> Pose3DType:
        """
        Get the current end-effector pose.

        Returns
        -------
        np.ndarray
            End-effector pose (7,) or (..., 7).
        """
        return copy.deepcopy(self.GetPose(state="Actual", out="x"))

    @property
    def p(self) -> Vector3DType:
        """
        Get the current end-effector position.

        Returns
        -------
        np.ndarray
            End-effector position (3,).
        """
        return copy.deepcopy(self.GetPose(state="Actual", out="p"))

    @property
    def Q(self) -> QuaternionType:
        """
        Get the current end-effector quaternion.

        Returns
        -------
        np.ndarray
            End-effector quaternion (4,).
        """
        return copy.deepcopy(self.GetPose(state="Actual", out="Q"))

    @property
    def R(self) -> RotationMatrixType:
        """
        Get the current end-effector rotation matrix.

        Returns
        -------
        np.ndarray
            End-effector rotation matrix (3, 3).
        """
        return copy.deepcopy(self.GetPose(state="Actual", out="R"))

    @property
    def T(self) -> HomogeneousMatrixType:
        """
        Get the current end-effector transformation matrix.

        Returns
        -------
        np.ndarray
            End-effector transformation matrix (4, 4).
        """
        return copy.deepcopy(self.GetPose(state="Actual", out="T"))

    @property
    def v(self) -> Velocity3DType:
        """
        Get the current end-effector velocity.

        Returns
        -------
        np.ndarray
            End-effector velocity (6,).
        """
        return copy.deepcopy(self.GetVel(state="Actual", out="Twist"))

    @property
    def pdot(self) -> Vector3DType:
        """
        Get the current end-effector linear velocity.

        Returns
        -------
        np.ndarray
            End-effector linear velocity (3,).
        """
        return copy.deepcopy(self.GetVel(state="Actual", out="Linear"))

    @property
    def w(self) -> Vector3DType:
        """
        Get the current end-effector angular velocity.

        Returns
        -------
        np.ndarray
            End-effector angular velocity (3,).
        """
        return copy.deepcopy(self.GetVel(state="Actual", out="Angular"))

    @property
    def FT(self) -> WrenchType:
        """
        Get the current force/torque sensor data.

        Returns
        -------
        np.ndarray
            Force/Torque sensor data (6,).
        """
        return copy.deepcopy(self.GetFT(out="Wrench"))

    @property
    def F(self) -> Vector3DType:
        """
        Get the current force sensor data.

        Returns
        -------
        np.ndarray
            Force sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(out="Force"))

    @property
    def Trq(self) -> Vector3DType:
        """
        Get the current torque sensor data.

        Returns
        -------
        np.ndarray
            Torque sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(out="Torque"))

    @property
    def q_ref(self) -> JointConfigurationType:
        """
        Get the desired joint positions.

        Returns
        -------
        np.ndarray
            Desired joint positions (nj,).
        """
        return copy.deepcopy(self._command.q)

    @property
    def qdot_ref(self) -> JointConfigurationType:
        """
        Get the desired joint velocities.

        Returns
        -------
        np.ndarray
            Desired joint velocities (nj,).
        """
        return copy.deepcopy(self._command.qdot)

    @property
    def x_ref(self) -> Pose3DType:
        """
        Get the desired end-effector pose.

        Returns
        -------
        np.ndarray
            Desired end-effector pose (7,).
        """
        return copy.deepcopy(self.GetPose(state="Command", out="x"))

    @property
    def p_ref(self) -> Vector3DType:
        """
        Get the desired end-effector position.

        Returns
        -------
        np.ndarray
            Desired end-effector position (3,).
        """
        return copy.deepcopy(self.GetPose(state="Command", out="p"))

    @property
    def Q_ref(self) -> QuaternionType:
        """
        Get the desired end-effector quaternion.

        Returns
        -------
        np.ndarray
            Desired end-effector quaternion (4,).
        """
        return copy.deepcopy(self.GetPose(state="Command", out="Q"))

    @property
    def R_ref(self) -> RotationMatrixType:
        """
        Get the desired end-effector rotation matrix.

        Returns
        -------
        np.ndarray
            Desired end-effector rotation matrix (3, 3).
        """
        return copy.deepcopy(self.GetPose(state="Command", out="R"))

    @property
    def T_ref(self) -> HomogeneousMatrixType:
        """
        Get the desired end-effector transformation matrix.

        Returns
        -------
        np.ndarray
            Desired end-effector transformation matrix (4, 4).
        """
        return copy.deepcopy(self.GetPose(state="Command", out="T"))

    @property
    def v_ref(self) -> Velocity3DType:
        """
        Get the desired end-effector velocity.

        Returns
        -------
        np.ndarray
            Desired end-effector velocity (6,).
        """
        return copy.deepcopy(self.GetVel(state="Command", out="Twist"))

    @property
    def pdot_ref(self) -> Vector3DType:
        """
        Get the desired end-effector linear velocity.

        Returns
        -------
        np.ndarray
            Desired end-effector linear velocity (3,).
        """
        return copy.deepcopy(self.GetVel(state="Command", out="Linear"))

    @property
    def w_ref(self) -> Vector3DType:
        """
        Get the desired end-effector angular velocity.

        Returns
        -------
        np.ndarray
            Desired end-effector angular velocity (3,).
        """
        return copy.deepcopy(self.GetVel(state="Command", out="Angular"))

    @property
    def FT_ref(self) -> WrenchType:
        """
        Get the desired force/torque sensor data.

        Returns
        -------
        np.ndarray
            Desired force/torque sensor data (6,).
        """
        return copy.deepcopy(self.GetFT(state="Command", out="Wrench"))

    @property
    def F_ref(self) -> Vector3DType:
        """
        Get the desired force sensor data.

        Returns
        -------
        np.ndarray
            Desired force sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(state="Command", out="Force"))

    @property
    def Trq_ref(self) -> Vector3DType:
        """
        Get the desired torque sensor data.

        Returns
        -------
        np.ndarray
            Desired torque sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(state="Command", out="Torque"))

    @property
    def q_err(self) -> JointConfigurationType:
        """
        Get the error in joint positions.

        Returns
        -------
        np.ndarray
            Error in joint positions (nj,).
        """
        return self.q_ref - self.q

    @property
    def qdot_err(self) -> JointConfigurationType:
        """
        Get the error in joint velocities.

        Returns
        -------
        np.ndarray
            Error in joint velocities (nj,).
        """
        return self.qdot_ref - self.qdot

    @property
    def x_err(self) -> Pose3DType:
        """
        Get the error in end-effector pose.

        Returns
        -------
        np.ndarray
            Error in end-effector pose (7,).
        """
        return xerr(self.x_ref, self.x)

    @property
    def p_err(self) -> Vector3DType:
        """
        Get the error in end-effector position.

        Returns
        -------
        np.ndarray
            Error in end-effector position (3,).
        """
        return self.p_ref - self.p

    @property
    def Q_err(self) -> QuaternionType:
        """
        Get the error in end-effector quaternion.

        Returns
        -------
        np.ndarray
            Error in end-effector quaternion (4,).
        """
        return qerr(self.Q_ref, self.Q)

    @property
    def R_err(self) -> RotationMatrixType:
        """
        Get the error in end-effector rotation matrix.

        Returns
        -------
        np.ndarray
            Error in end-effector rotation matrix (3, 3).
        """
        return self.R_ref @ self.R.T

    @property
    def T_err(self) -> HomogeneousMatrixType:
        """
        Get the error in end-effector transformation matrix.

        Returns
        -------
        np.ndarray
            Error in end-effector transformation matrix (4, 4).
        """
        return terr(self.T_ref, self.T)

    @property
    def v_err(self) -> Velocity3DType:
        """
        Get the error in end-effector velocity.

        Returns
        -------
        np.ndarray
            Error in end-effector velocity (6,).
        """
        return self.v_ref - self.v

    @property
    def pdot_err(self) -> Vector3DType:
        """
        Get the error in end-effector linear velocity.

        Returns
        -------
        np.ndarray
            Error in end-effector linear velocity (3,).
        """
        return self.pdot_ref - self.pdot

    @property
    def w_err(self) -> Vector3DType:
        """
        Get the error in end-effector angular velocity.

        Returns
        -------
        np.ndarray
            Error in end-effector angular velocity (3,).
        """
        return self.w_ref - self.w

    @property
    def J(self) -> JacobianType:
        """
        Get the Jacobian matric for current configuration.

        Returns
        -------
        JacobianType
            Jacobian matrix (6, nj).
        """
        return self.Jacobi()

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
        self.FTFrame = self.TCPGripper.copy()
        self._command.q = np.zeros(self.nj)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.u = np.zeros(self.nj)
        self._command.ux = np.zeros(6)
        self._command.x = np.array([0, 0, 0, 1, 0, 0, 0])  # Default pose (identity quaternion)
        self._command.rx = np.array([0, 0, 0, 1, 0, 0, 0])  # Default pose (identity quaternion)
        self._command.v = np.zeros(6)
        self._command.rv = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.data = None
        self._command.mode = CommandModeCodes.STOP.value

        self._actual.q = np.zeros(self.nj)
        self._actual.qdot = np.zeros(self.nj)
        self._actual.trq = np.zeros(self.nj)
        self._actual.x = self.Kinmodel()[0]
        self._actual.v = np.zeros(6)
        self._actual.FT = np.zeros(6)
        self._actual.trqExt = np.zeros(self.nj)

        self._default.AddedTrq = np.zeros(self.nj)

    def Init(self) -> None:
        """
        Initializes the robot by setting up the object, state, current target, time, and logging a message.

        This method calls the following functions to initialize the robot:
        1. `InitObject()` - Initializes the command and actual states with default values.
        2. `GetState()` - Retrieves the current state of the robot.
        3. `ResetCurrentTarget()` - Resets the current target position.
        4. `ResetTime()` - Resets the robot's internal time.

        Returns
        -------
        None
            This method does not return any value. It modifies the robot's internal state and logs a message.
        """
        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    def GetState(self) -> None:
        """
        Abstract method to updates the robot's state.

        It has to be reimplemented in actual robot class!

        It has to set:
        - Actual robot joint and task space states
        - Read all robot sensors.
        Robot states and signal are defined in robot frame except
        the task force/torque sensor data, which is defined in the
        nominal tool frame defned by CPGripper transformation.

        This method sets the following attributes:
        - `_tt`: The current time, can be retrieved using `simtime()`.
        - `_last_update`: The last update time, retrieved using `simtime()`.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot.
        """
        self._tt = self.simtime()
        self._last_update = self.simtime()

    def Update(self) -> None:
        """
        Updates the robot's state and optionally triggers a capture callback.

        This method performs the following actions:

        - If ``_do_update`` is ``True``, it calls ``GetState()`` to update the
          robot's internal state.
        - If ``_do_capture`` is ``True`` and a capture callback function
          (``_capture_callback``) is defined, it calls the callback function,
          passing the robot object as an argument.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot and may
            trigger a callback.
        """
        if self._do_update:
            self.GetState()
            if self._do_capture and self._capture_callback is not None:
                self._capture_callback(self)
        elif (self.simtime() - self._last_update) >= self.tsamp:
            self.GetState()

    def EnableUpdate(self) -> None:
        """
        Enables the update of the robot's internal state.

        This method sets the `_do_update` attribute to `True`, which allows the robot's state to be updated.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot.
        """
        self._do_update = True

    def DisableUpdate(self) -> None:
        """
        Disables the update of the robot's internal state.

        This method sets the `_do_update` attribute to `False`, which prevents the robot's state from being updated.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the robot.
        """
        self._do_update = False

    def GetUpdateStatus(self) -> bool:
        """
        Returns the current status of the update flag.

        This method returns the value of the `_do_update` attribute, which indicates whether the robot's state update is enabled.

        Returns
        -------
        bool
            `True` if updates are enabled, `False` otherwise.
        """
        return self._do_update

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
        self._command.rx = copy.deepcopy(self.BaseToWorld(self._actual.x))
        self._command.v = np.zeros(6)
        self._command.rv = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.trq = np.zeros(self.nj)
        self._sleep(0.1)
        self.Update()
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        if do_move:
            self.Message("Moving to actual configuration", 2)
            if not self.Start():
                return MotionResultCodes.NOT_READY.value

            if not self._semaphore.acquire(blocking=False):
                self.WarningMessage("ResetCurrentTarget: Move not executed due to active threads!")
                return MotionResultCodes.ACTIVE_THREADS.value

            if self._control_strategy.startswith("Joint"):
                self._last_status = self.JMove(self._command.q, wait=1, **kwargs)
            elif self._control_strategy.startswith("Cartesian"):
                self._last_status = self.CMove(self._command.x, wait=1, **kwargs)

            self.Stop()
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
        self._command.v = _J @ self._command.qdot

    # Get joint variables
    def GetJointPos(self, state: Optional[str] = None, refresh: Optional[bool] = None) -> JointConfigurationType:
        """
        Get the joint positions of the robot based on the specified state.

        Parameters
        ----------
        state : str, optional
            The state from which to retrieve joint positions. Can be 'Actual' or 'Commanded'.
            If `None`, the default state defined in the robot class is used.
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving joint positions. Default is `True`.

        Returns
        -------
        np.ndarray
            The joint positions (`q`) from the specified state, copied to prevent external modifications.

        Raises
        ------
        ValueError
            If the `state` is not "Actual" or "Commanded".
        """
        if state is None:
            state = self._default.State
        if refresh is None:
            refresh = self._default.Refresh
        if check_option(state, "Actual"):
            if refresh:
                self.GetState()
            return copy.deepcopy(self._actual.q)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.q)
        else:
            raise ValueError(f"State '{state}' not supported")

    def GetJointVel(self, state: Optional[str] = None, refresh: Optional[bool] = None) -> JointConfigurationType:
        """
        Get the joint velocities of the robot based on the specified state.

        Parameters
        ----------
        state : str, optional
            The state from which to retrieve joint velocities. Can be 'Actual' or 'Commanded'.
            If `None`, the default state defined in the robot class is used.
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving joint velocities. Default is `True`.

        Returns
        -------
        np.ndarray
            The joint velocities (`qdot`) from the specified state, copied to prevent external modifications.

        Raises
        ------
        ValueError
            If the `state` is not "Actual" or "Commanded".
        """
        if state is None:
            state = self._default.State
        if refresh is None:
            refresh = self._default.Refresh
        if check_option(state, "Actual"):
            if refresh:
                self.GetState()
            return copy.deepcopy(self._actual.qdot)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.qdot)
        else:
            raise ValueError(f"State '{state}' not supported")

    def GetJointTrq(self, state: Optional[str] = None, refresh: Optional[bool] = None) -> JointConfigurationType:
        """
        Get the joint torques of the robot based on the specified state.

        Parameters
        ----------
        state : str, optional
            The state from which to retrieve joint torques. Can be 'Actual' or 'Commanded'.
            If `None`, the default state defined in the robot class is used.
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving joint torques. Default is `True`.

        Returns
        -------
        np.ndarray
            The joint torques (`trq`) from the specified state, copied to prevent external modifications.

        Raises
        ------
        ValueError
            If the `state` is not "Actual" or "Commanded".
        """
        if state is None:
            state = self._default.State
        if refresh is None:
            refresh = self._default.Refresh
        if check_option(state, "Actual"):
            if refresh:
                self.GetState()
            return copy.deepcopy(self._actual.trq)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.trq)
        else:
            raise ValueError(f"State '{state}' not supported")

    # Get task space variables
    def GetPose(self, out: str = None, task_space: str = None, kinematics: str = None, state: str = None, refresh: bool = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
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
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
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

        if refresh is None:
            if (self.simtime() - self._last_update) > (self.tsamp * 0.9):
                self.GetState()
        elif refresh:
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

    def GetPos(self, out: str = "p", task_space: str = None, kinematics: str = None, state: str = None) -> Vector3DType:
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
        Vector3DType
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

    def GetOri(self, out: str = "Q", task_space: str = None, kinematics: str = None, state: str = None) -> Union[QuaternionType, RotationMatrixType]:
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
        QuaternionType or RotationMatrixType
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

    def GetVel(self, out: str = None, task_space: str = None, kinematics: str = None, state: str = None, refresh: bool = None) -> Union[Velocity3DType, Vector3DType]:
        """
        Get robot end-effector velocity.

        Parameters
        ----------
        out : str, optional
            Output form for the velocity. Options are "Twist" (default), "Linear", or "Angular".
        task_space : str, optional
            Task space frame for the velocity. Options are "World", "Object", "Tool", or "Robot". The default is "World".
        kinematics : str, optional
            The kinematics used for calculation. Options are "Robot" or "Calculated". The default is "Robot".
        state : str, optional
            The state of the robot for the calculation. Options are "Actual" or "Commanded". The default is "Actual".
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the velocity. Default is `True`.

        Returns
        -------
        Velocity3DType or Vector3DType
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

        if refresh is None:
            if (self.simtime() - self._last_update) > (self.tsamp * 0.9):
                self.GetState()
        elif refresh:
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
            elif check_option(task_space, "Tool"):
                _vv = _J @ _qqdot
                R0 = self.GetOri(state=state, out="R", task_space="Robot", kinematics=kinematics)
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                _vv = RR @ _vv
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
            elif check_option(task_space, "Tool"):
                R0 = self.GetOri(state=state, out="R", task_space="Robot", kinematics=kinematics)
                RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
                _vv = RR @ _vv
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

    def GetFT(
        self,
        out: str = None,
        source: str = None,
        task_space: str = "FTFrame",
        kinematics: str = None,
        avg_time: int = 0,
        user_frame: Optional[TCPType] = None,
        refresh: bool = None,
    ) -> Union[WrenchType, Vector3DType]:
        """
        Get force/torque sensor data for the robot.

        Parameters
        ----------
        out : str, optional
            Output form for the force/torque data. Options are "Wrench" (default), "Force", or "Torque".
        source : str, optional
            Source of the force/torque data. Options are "External" or "Robot". The default is "Robot".
        task_space : str, optional
            Task space frame for the force/torque data. Options are "World", "Object", "Robot", "Tool" or "TCP", "FTFrame", or "User". The default is "Tool".
        kinematics : str, optional
            The kinematics used for the calculation. Options are "Robot" or "Calculated". The default is "Robot".
        avg_time : int, optional
            Average time for the external force/torque sensor, by default 0.
        user_frame : TCPType, optional
            User-defined frame for the force/torque data. Default is None.
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the force/torque data. Default is `True`.

        Returns
        -------
        WrenchType or Vector3DType
            The force/torque data in the specified output form. The shape varies depending on the `out` option.

        Raises
        ------
        ValueError
            If the `source`, `task_space`, or `out` values are not supported.

        Notes
        -----
        User defined frame `user_frame` represents the transforation from the robot flange

        """
        if out is None:
            out = self._default.TaskFTForm
        if source is None:
            source = self._default.Source
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        if check_option(task_space, "User"):
            if user_frame is None:
                raise ValueError("User frame must be provided when task_space is 'User'")
            else:
                user_frame = spatial2t(user_frame)
        if refresh is None:
            refresh = self._default.Refresh

        if refresh is None:
            if (self.simtime() - self._last_update) > (self.tsamp * 0.9):
                self.GetState()
        elif refresh:
            self.GetState()

        # First we get the force/torque in the F/T frame
        if check_option(source, "External"):
            if self.FTSensor:
                _FT = self.FTSensor.GetFT(avg_time=avg_time)
                if self.FTSensorFrame is None:
                    _frame = self.FTFrame
                else:
                    _frame = self.FTSensorFrame
                _Rsensor = q2r(self._actual.x[3:]) @ self.TCP[:3, :3].T @ _frame[:3, :3]  # Rotation of sensor frame
                _sensor2FTFrame = np.linalg.pinv(_frame) @ self.FTFrame  # Transformation from sensor frame to F/T frame
                _FT -= -(-9.81 * self.FTSensor.Load.mass * np.hstack((_Rsensor[2, :], v2s(self.FTSensor.Load.COM) @ _Rsensor[2, :])))
                _FT = world2frame(_FT, _sensor2FTFrame, typ="Wrench")
            else:
                raise ValueError("No FT sensor assigned to robot")
        elif check_option(source, "Robot"):
            if check_option(kinematics, "Robot"):
                _FT = self._actual.FT
                _frame = self.FTFrame
                _Rsensor = q2r(self._actual.x[3:]) @ self.TCP[:3, :3].T @ _frame[:3, :3]  # Rotation of F/T frame
                _sensor2FTFrame = np.linalg.pinv(_frame) @ self.FTFrame  # Transformation from sensor frame to F/T frame
                _FT -= -(-9.81 * self.Load.mass * np.hstack((_Rsensor[2, :], v2s(self.Load.COM) @ _Rsensor[2, :])))
                _FT = world2frame(_FT, _sensor2FTFrame, typ="Wrench")
            elif check_option(kinematics, "Calculated"):
                # Calculate F/T in TCP from external torques using Jacobian transpose
                _J = self.Jacobi()
                _FT = np.linalg.pinv(_J.T) @ self._actual.trqExt  # F/T in TCP frame
                _Rsensor = q2r(self._actual.x[3:]) @ self.TCP[:3, :3].T @ self.FTFrame[:3, :3]  # Rotation of F/T frame
                _sensor2FTFrame = np.linalg.pinv(self.TCP) @ self.FTFrame  # Transformation from TCP frame to F/T frame
                _FT = world2frame(_FT, _sensor2FTFrame, typ="Wrench")
            else:
                raise ValueError(f"Kinematics calculation '{kinematics}' not supported")
        else:
            raise ValueError(f"Source '{source}' not supported")

        # Transform  F/T in FTFrame  to selected frame
        _F = _FT.copy()
        if check_option(task_space, "World"):
            _T = np.linalg.pinv(self.FTFrame) @ self.TCP  # Transformation from F/T frame to TCP frame
            _FTtool = world2frame(_F, _T, typ="Wrench")  # F/T in TCP frame
            _Tw = np.linalg.pinv(self.GetPose(task_space="World", out="T"))  # Transformation from TCP frame to world frame
            _F = world2frame(_FTtool, _Tw, typ="Wrench")  # F/T in world frame
        elif check_option(task_space, "Object"):
            _T = np.linalg.pinv(self.FTFrame) @ self.TCP  # Transformation from F/T frame to TCP frame
            _FTtool = world2frame(_F, _T, typ="Wrench")  # F/T in TCP frame
            _Tw = np.linalg.pinv(self.GetPose(task_space="World", out="T"))  # Transformation from TCP frame to world frame
            _FTw = world2frame(_FTtool, _Tw, typ="Wrench")  # F/T in world frame
            _F = self.WorldToObject(_FTw, typ="Wrench")
        elif check_option(task_space, "Robot"):
            _T = np.linalg.pinv(self.FTFrame) @ self.TCP  # Transformation from F/T frame to TCP frame
            _FTtool = world2frame(_F, _T, typ="Wrench")  # F/T in TCP frame
            _Tr = np.linalg.pinv(self.GetPose(task_space="Robot", out="T"))  # Transformation from TCP frame to robotframe
            _F = world2frame(_FTtool, _Tr, typ="Wrench")  # F/T in robot frame
        elif check_option(task_space, "Tool") or check_option(task_space, "TCP"):
            _T = np.linalg.pinv(self.FTFrame) @ self.TCP  # Transformation from F/T frame to TCP frame
            _F = world2frame(_F, _T, typ="Wrench")  # F/T in TCP frame
        elif check_option(task_space, "FTFrame"):
            pass
        elif check_option(task_space, "User"):
            _T = np.linalg.pinv(self.FTFrame) @ user_frame  # Transformation from F/T frame to TCP frame
            _F = world2frame(_F, _T, typ="Wrench")  # F/T in TCP frame
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        if check_option(out, "Wrench"):
            return _F
        elif check_option(out, "Force"):
            return _F[:3]
        elif check_option(out, "Torque"):
            return _F[3:]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    # Joint space motion
    def GoToActual(self, **kwargs: Any) -> None:
        """
        Resets the current target and moves the robot to the actual configuration.

        This method internally calls `ResetCurrentTarget` with `do_move=True`, which resets
        the current target configuration and moves the robot to the actual configuration.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to `ResetCurrentTarget` method.
        """
        self.ResetCurrentTarget(do_move=True, **kwargs)

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> None:
        """
        Abstract method to command the robot to go to a specific joint configuration.

        It has to be reimplemented in actual robot class!

        This method sets the commanded joint positions (`q`), velocities (`qdot`), and torques (`trq`),
        then sends them to the robot and waits for the specified time (`wait`).

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        qdot : JointVelocityType, optional
            Desired joint velocities (nj,).
        trq : JointTorqueType, optional
            Desired joint torques (nj,).
        wait : float, optional
            Time to wait (in seconds) after commanding the robot to move.
        """
        q = vector(q, dim=self.nj)
        if qdot is None:
            qdot = np.zeros(self.nj)
        else:
            qdot = vector(qdot, dim=self.nj)
        if trq is None:
            trq = np.zeros(self.nj)
        else:
            trq = vector(trq, dim=self.nj)
        if wait is None:
            wait = self.tsamp
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = J @ qdot

        self._actual.q = q
        self._actual.qdot = qdot
        self._actual.trq = trq
        self._actual.x = x
        self._actual.v = J @ qdot
        self._sleep(wait)
        raise NotImplementedError("Joint position controller not implemented!")

    def GoTo_qtraj(self, q: JointConfigurationType, qdot: JointVelocityType, qddotrq: JointAccelerationType, time: TimesType) -> None:
        """
        Command the robot to follow a joint trajectory.

        It has to be reimplemented in actual robot class!

        It is intended to control the robot to follow a trajectory specified by joint positions (`q`),
        velocities (`qdot`), and accelerations (`qddot`) over a specified time (`time`).

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points.
        qdot : JointVelocityType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points.
        qddotrq : JointAccelerationType
            Desired joint accelerations for the trajectory (n, nj), where n is the number of trajectory points.
        time : TimesType
            Time points for the trajectory (n,).
        """
        self._command.q = q[-1, :]
        self._command.qdot = np.zeros(self.nj)
        x = self.Kinmodel(self._command.q)[0]
        self._command.x = x
        self._command.v = np.zeros(6)
        raise NotImplementedError("Joint trajectory controller not implemented!")

    def _loop_joint_traj(self, qi: JointPathType, qdoti: JointPathType, trq: JointPathType, time: TimesType, wait: float = 0, **kwargs: Any) -> int:
        """
        Executes a joint trajectory motion for the robot.

        This method controls the robot to follow a joint trajectory using the given joint positions
        (`qi`), velocities (`qdoti`), torques (`trq`), and time intervals (`time`). It handles motion
        control logic, including checking for abort conditions and monitoring the motion error.

        If the control strategy is "JointPositionTrajectory", it uses the `GoTo_qtraj` method to command
        the robot to follow the trajectory, and if `_do_motion_check` True, it waits for the motion
        to complete, and checks for errors.

        If the control strategy is different, it iterates through each point of the trajectory and moves
        the robot to each joint configuration using `GoTo_q`.

        Parameters
        ----------
        qi : JointPathType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points
            and nj is the number of joints.
        qdoti : JointPathType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points
            and nj is the number of joints.
        trq : JointPathType
            Desired joint torques (nj,).
        time : TimesType
            Time intervals for the trajectory (n,).
        wait : float, optional
            Additional wait time after the trajectory execution, by default 0.
        **kwargs : Any
            Additional keyword arguments passed to the control methods.

        Returns
        -------
        int
            Return code indicating the result of the motion:
            - 0: Motion completed successfully.
            - 99: Motion aborted by the user.
            - Non-zero: Motion aborted due to motion controller error.

        Notes
        -----
        If the control strategy is "JointPositionTrajectory", the method will call `GoTo_qtraj` to command
        the trajectory and wait for its completion. If an error occurs during the motion, the method will
        return a non-zero value indicating the error. The motion can be aborted by setting the `_abort` flag.
        """
        if self._control_strategy in ["JointPositionTrajectory"]:
            _t_traj = self.simtime()
            qdoti[-1, :] = np.zeros(self.nj)
            self._last_status = self.GoTo_qtraj(qi, qdoti, np.zeros(qi.shape), time)
            self.Update()
            if self._last_status != MotionResultCodes.MOTION_SUCCESS.value:
                return self._last_status

            if self._do_motion_check:
                while (self._motion_error is None) and ((self.simtime() - _t_traj) < time[-1]):
                    if self._abort:
                        self.WarningMessage("Motion aborted by user")
                        self.StopMotion()
                        return MotionResultCodes.MOTION_ABORTED.value
                    elif self._do_motion_check and self._motion_check_callback is not None:
                        self._last_status = self._motion_check_callback(self)
                        if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                            self.WarningMessage("Motion aborted")
                            self.StopMotion()
                        return self._last_status
                    self._sleep(self.tsamp)
                    self.Update()

            _t_traj = self.simtime()
            while (self._motion_error is None) and ((self.simtime() - _t_traj) < wait):
                self._sleep(self.tsamp)
                self.Update()

            if (self._motion_error is not None) and (self._motion_error != 0):
                self.WarningMessage(f"Motion aborted due to motion controller error ({self._motion_error})")
                return self._motion_error
            else:
                # print("End status:", self._motion_error, (self.simtime() - _t_traj), time[-1])
                return MotionResultCodes.MOTION_SUCCESS.value
        elif self._control_strategy.startswith("Joint"):
            for qt, qdt in zip(qi, qdoti):
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self.StopMotion()
                    return MotionResultCodes.MOTION_ABORTED.value
                elif self._do_motion_check and self._motion_check_callback is not None:
                    self._last_status = self._motion_check_callback(self)
                    if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                        self.WarningMessage("Motion aborted")
                        self.StopMotion()
                        return self._last_status
                self._last_status = self.GoTo_q(qt, qdt, trq, self.tsamp, **kwargs)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Motion aborted")
                    self.StopMotion()
                    return self._last_status

            # Last sample
            self._last_status = self.GoTo_q(qi[-1, :], np.zeros(self.nj), trq, self.tsamp, **kwargs)
            # # Update task space values
            # x = self.Kinmodel(self._command.q)[0]
            # self._command.x = x
            # self._command.v = np.zeros(6)

            tx = self.simtime()
            while self.simtime() - tx < wait:
                self.Update()
                self._sleep(self.tsamp)

            return self._last_status
        else:
            raise NotImplementedError("Joint trajectory controller not implemented!")

    def JMove(
        self,
        q: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        wait: Optional[float] = None,
        traj: Optional[str] = None,
        added_trq: Optional[JointTorqueType] = None,
        min_joint_dist: Optional[float] = None,
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> Thread:
        """
        Moves the robot to a specified joint position with specified velocity and trajectory.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        t : float, optional
            Time for the movement, by default None.
        vel : float, optional
            Desired velocity for the movement, by default None.
        vel_fac : JointConfigurationType, optional
            Velocity scaling factor for each joint, by default None.
        wait : float, optional
            Time to wait after movement is completed, by default None.
        traj : str, optional
            Trajectory type, by default None.
        added_trq : JointTorqueType, optional
            Additional joint torques, by default None.
        min_joint_dist : float, optional
            Minimum distance to target joint positions for movement execution, by default None.
        asynchronous : bool, optional
            If True, executes the movement asynchronously, by default False.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        Thread
            The thread executing the asynchronous movement, if `asynchronous` is True.
        int
            The status of the move (0 for success, 99 for abort) if `asynchronous` is False.

        Notes
        -----
        - If `asynchronous` is set to True, the movement will be executed in a separate thread.
        - The control strategy should be set to "Joint" for this function to work.
        """
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JMove not executed")
            return MotionResultCodes.WRONG_STRATEGY.value
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC JMove", 2)
            _th = Thread(target=self._JMove, args=(q,), kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "wait": wait, "traj": traj, "added_trq": added_trq, "min_joint_dist": min_joint_dist, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JMove(q, t=t, vel=vel, vel_fac=vel_fac, wait=wait, traj=traj, added_trq=added_trq, min_joint_dist=min_joint_dist, **kwargs)

    def _JMove(
        self,
        q: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        wait: Optional[float] = None,
        traj: Optional[str] = None,
        added_trq: Optional[JointTorqueType] = None,
        min_joint_dist: Optional[float] = None,
        **kwargs: Any,
    ) -> int:
        """
        Executes the joint movement command to a specified target position, considering additional parameters
        such as velocity and trajectory. This function can either execute the movement synchronously or asynchronously.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        t : float, optional
            Time for the movement, by default None.
        vel : float, optional
            Desired velocity for the movement, by default None.
        vel_fac : JointConfigurationType, optional
            Velocity scaling factor for each joint, by default None.
        wait : float, optional
            Time to wait after movement is completed, by default None.
        traj : str, optional
            Trajectory type, by default None.
        added_trq : JointTorqueType, optional
            Additional joint torques, by default None.
        min_joint_dist : float, optional
            Minimum distance to target joint positions for movement execution, by default None.

        Returns
        -------
        int
            The status of the move (0 for success, 99 for abort) or an error code if movement is unsuccessful.

        Raises
        ------
        ValueError
            If the time `t` is not a non-negative scalar.
        ValueError
            If the joint positions `q` are out of range.
        """
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(added_trq, dim=self.nj)
        if min_joint_dist is None:
            min_joint_dist = self._default.MinJointDist

        q = self.jointvar(q)
        if self.CheckJointLimits(q):
            raise ValueError("Joint positions out of range")

        dist = np.abs(q - self._command.q)
        if np.all(np.abs(dist) < min_joint_dist):
            self.Message("JMove not executed - close to target", 2)
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
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=self.nj)
                _vel = self.qdot_max * vel_fac
                self.Message(f"JMove started: {q} with velocity {100 * np.max(_vel / self.qdot_max):.1f}%", 2)
            else:
                if isscalar(vel):
                    _vel = np.ones(self.nj) * vel
                else:
                    _vel = vector(vel, dim=self.nj)
                self.Message(f"JMove started: {q} with velocity {np.max(_vel):.1f}rd/s", 2)
            _vel = np.clip(np.abs(_vel), 0, self.qdot_max)
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.qdot_max
            self.Message(f"JMove started to: {q} in {_time[-1]:.1f}s", 2)

        q0 = self.GetJointPos(state="Commanded")
        if check_option(traj, "Jerk"):
            qd_max = kwargs.get('qd_max', _vel)
            qdd_max = kwargs.get('qdd_max', self._default.JointAcceleration)
            jerk_max = kwargs.get('jerk_max', self._default.JointJerk)
            if t is None:    
                qi, qdoti, _ = jtraj(q0, q, [0, self.tsamp], traj=traj, qd_max=qd_max, qdd_max=qdd_max, jerk_max=jerk_max)
                _time =np.arange(qi.shape[0]) * self.tsamp
            else:
                qi, qdoti, _ = jtraj(q0, q, _time, traj=traj, qd_max=qd_max, qdd_max=qdd_max, jerk_max=jerk_max)
                if qi.shape[0] > _time.shape[0]:
                    _time =np.arange(qi.shape[0]) * self.tsamp
                    if t < _time[-1]:
                        self.Message(f"Execution time will be prolonged due to bounded joint velocities or accelerations form {t:.1f}s to {_time[-1]:.1f}s.", 2)
        else:
            qi, qdoti, _ = jtraj(q0, q, _time, traj=traj)
            _fac = np.max(np.max(np.abs(qdoti), axis=0) / _vel)
            if (_fac > 1) or (t is None):
                _tend = max(_time[-1] * _fac, 0.5) + self.tsamp
                self.Message(f"Execution time will be prolonged due to bounded joint velocities form {_time[-1]:.1f}s to {_tend:.1f}s.", 2)
                _time = np.arange(0.0, _tend, self.tsamp)
                qi, qdoti, _ = jtraj(q0, q, _time, traj=traj)

        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("JMove not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.JOINT.value

        self._last_status = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("JMove finished", 2)
        return self._last_status

    def JMoveFor(
        self,
        dq: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        state: str = "Commanded",
        traj: Optional[str] = None,
        wait: Optional[float] = None,
        added_trq: Optional[JointTorqueType] = None,
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Moves the robot by a specified joint displacement (dq) relative to the current joint positions.

        Parameters
        ----------
        dq : JointConfigurationType
            Joint displacement (nj,). The relative change in joint positions.
        t : float, optional
            Time for the movement, by default None.
        vel : float, optional
            Desired velocity for the movement, by default None.
        vel_fac : JointConfigurationType, optional
            Velocity scaling factor for each joint, by default None.
        state : str, optional
            State of the joint positions to reference, by default "Commanded".
        traj : str, optional
            Trajectory type, by default None.
        wait : float, optional
            Time to wait after movement is completed, by default None.
        added_trq : JointTorqueType, optional
            Additional joint torques, by default None.
        asynchronous : bool, optional
            If True, executes the movement asynchronously, by default False.

        Returns
        -------
        int
            The status of the move (0 for success, 99 for abort).

        Notes
        -----
        - This function performs a joint move by adding the displacement (`dq`) to the current joint positions and then calls `JMove` to execute the movement.
        - If `asynchronous` is set to True, the movement will be executed in a separate thread.
        """
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JMoveFor not executed")
            return MotionResultCodes.WRONG_STRATEGY.value
        dq = self.jointvar(dq)
        q0 = self.GetJointPos(state=state)
        q = q0 + dq
        self.Message("JMoveFor -> JMove", 2)
        self._last_status = self.JMove(q, t=t, vel=vel, vel_fac=vel_fac, traj=traj, wait=wait, added_trq=added_trq, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def JLine(
        self,
        q: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        state: str = "Commanded",
        wait: Optional[float] = None,
        added_trq: Optional[JointTorqueType] = None,
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Performs a linear joint trajectory move to the specified joint positions.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        t : float, optional
            Time for the movement, by default None.
        vel : float, optional
            Desired velocity for the movement, by default None.
        vel_fac : JointConfigurationType, optional
            Velocity scaling factor for each joint, by default None.
        state : str, optional
            State of the joint positions to reference, by default "Commanded".
        wait : float, optional
            Time to wait after movement is completed, by default None.
        added_trq : JointTorqueType, optional
            Additional joint torques, by default None.
        asynchronous : bool, optional
            If True, executes the movement asynchronously, by default False.

        Returns
        -------
        int
            The status of the move (0 for success, 99 for abort).

        Notes
        -----
        - This function performs a joint move using a trapezoidal trajectory (`"Trap"`) to the desired joint positions (`q`).
        - If `asynchronous` is set to True, the movement will be executed in a separate thread.
        """
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JLine not executed")
            return MotionResultCodes.WRONG_STRATEGY.value
        self.Message("JLine -> JMove", 2)
        self._last_status = self.JMove(q, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", wait=wait, added_trq=added_trq, asynchronous=asynchronous)
        return self._last_status

    def JPath(self, path: JointPathType, t: TimesType, wait: Optional[float] = None, traj: Optional[str] = None, added_trq: Optional[JointTorqueType] = None, asynchronous: Optional[bool] = None, **kwargs: Any) -> int:
        """
        Execute a joint path trajectory move.

        Parameters
        ----------
        path : JointPathType
            Joint path ``(n, nj)`` containing the waypoints to execute.
        t : TimesType
            Time values associated with the path trajectory.
        wait : float, optional
            Time to wait after movement. Default is ``None``.
        traj : str, optional
            Trajectory type. Default is ``None``.
        added_trq : JointTorqueType, optional
            Additional torques to apply during movement. Default is ``None``.
        asynchronous : bool, optional
            If ``True``, execute the move asynchronously in a separate thread.
            Default is ``False``.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move.

        Notes
        -----
        This method moves the robot along a path defined by joint positions.
        If ``asynchronous`` is ``True``, the movement is executed in a new
        thread.
        """
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JPath not executed")
            return MotionResultCodes.WRONG_STRATEGY.value
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            _th = Thread(
                target=self._JPath,
                args=(
                    path,
                    t,
                ),
                kwargs={"wait": wait, "traj": traj, "added_trq": added_trq, **kwargs},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._JPath(path, t, wait=wait, traj=traj, added_trq=added_trq, **kwargs)

    def _JPath(self, path: JointPathType, t: TimesType, wait: Optional[float] = None, traj: Optional[str] = None, added_trq: Optional[JointTorqueType] = None, **kwargs: Any) -> int:
        """
        Executes a joint path trajectory move in a blocking manner.

        Parameters
        ----------
        path : JointPathType
            A 2D array (n, nj) representing the path of joint positions to move through, where `n` is the number of waypoints and `nj` is the number of joints.
        t : TimesType
            A 1D array of time values for the path trajectory.
        wait : float, optional
            The time to wait after movement, by default None.
        traj : str, optional
            The trajectory type, by default None.
        added_trq : JointTorqueType, optional
            Additional torques to apply during movement, by default None.

        Returns
        -------
        int
            The status of the move (0 for success, 99 for aborted).

        Raises
        ------
        ValueError
            If time `t` is not a scalar or is negative.
            If the path shape is incompatible with expected dimensions.
        """
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(added_trq, dim=self.nj)

        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        if isscalar(t) or t[0] == 0:
            _dist = np.abs(path[0, :] - self.q_ref)
            _qerr = np.max(_dist / self.qdot_max) * 2
            if _qerr > 0.01:
                self.Message(f"Move to path -> JPath ({_dist})", 2)
                self._last_status = self._JMove(path[0, :], max(_qerr, 0.2), wait=0, added_trq=added_trq)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Robot did not move to path start")
                    return self._last_status

        _n = np.shape(path)[0]
        if not isscalar(t) and len(t) == _n:
            if self._control_strategy in ["JointPositionTrajectory"]:
                _time = t
                qi = path
                qdoti = gradientPath(qi, _time)
            else:
                if t[0] > 0:
                    path = np.vstack((self.q_ref, path))
                    t = np.concatenate(([0], t))
                _time = np.arange(0.0, np.max(t) + self.tsamp, self.tsamp)
                qi = interpPath(t, path, _time)
                qdoti = gradientPath(qi, _time)
        else:
            if not isscalar(t):
                t = max(t)
            _s = np.linspace(0, t, _n)
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            qi = interpPath(_s, path, _time)
            qdoti = gradientPath(qi, _time)
            _fac = np.max(np.max(np.abs(qdoti), axis=0) / self.qdot_max)
            if _fac > 1:
                _s = np.linspace(0, t * _fac + self.tsamp, _n)
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                self.Message(f"Execution time will be prolonged due to bounded joint velocities form {t:.1f}s to {_time:.1f}s.", 2)
                qi = interpPath(_s, path, _time)
                qdoti = gradientPath(qi, _time)

        self.Message(f"JPath started: {path.shape[0]} points in {np.max(t)}s ", 2)
        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("JPath not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.JOINT_PATH.value

        self._last_status = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("JPath finished", 2)
        return self._last_status

    def JRBFPath(self, pathRBF: Dict[str, Any], t: float, direction: str = "Forward", wait: Optional[float] = None, traj: Optional[str] = None, added_trq: Optional[JointTorqueType] = None, asynchronous: Optional[bool] = None, **kwargs: Any) -> int:
        """
        Execute a joint path generated from radial basis function interpolation.

        Parameters
        ----------
        pathRBF : Dict[str, Any]
            Dictionary containing the RBF parameters and control points.
        t : float
            Time used to execute the path.
        direction : str, optional
            Motion direction, either ``"Forward"`` or ``"Backward"``.
            Default is ``"Forward"``.
        wait : float, optional
            Time to wait after the movement finishes. Default is ``None``.
        traj : str, optional
            Trajectory type. Default is ``None``.
        added_trq : JointTorqueType, optional
            Additional torques to apply during movement. Default is ``None``.
        asynchronous : bool, optional
            If ``True``, execute the move in a separate thread. Default is
            ``False``.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move.

        Notes
        -----
        If ``asynchronous`` is ``True``, the motion is executed in a new
        thread. The joint trajectory is computed from the control points stored
        in ``pathRBF``.
        """
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JRBFPath not executed")
            return MotionResultCodes.WRONG_STRATEGY.value
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            _th = Thread(
                target=self._JRBFPath,
                args=(
                    pathRBF,
                    t,
                ),
                kwargs={"direction": direction, "wait": wait, "traj": traj, "added_trq": added_trq, **kwargs},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._JRBFPath(pathRBF, t, direction=direction, wait=wait, traj=traj, added_trq=added_trq, **kwargs)

    def _JRBFPath(self, pathRBF: Dict[str, Any], t: float, direction: str = "Forward", wait: Optional[float] = None, traj: Optional[str] = None, added_trq: Optional[JointTorqueType] = None, **kwargs: Any) -> int:
        """
            Executes a joint trajectory path based on Radial Basis Function (RBF) interpolation in a blocking manner.

            Parameters
            ----------
            pathRBF : Dict[str, Any]
                A dictionary containing the RBF parameters and control points (e.g., "c" for control points).
            t : float
                The time to execute the path.
            direction : str, optional
                The direction of motion, either "Forward" or "Backward", by default "Forward".
            wait : float, optional
                The time to wait after movement, by default None.
            traj : str, optional
                The trajectory type, by default None.
        added_trq : JointTorqueType, optional
                Additional torques to apply during movement, by default None.

            Returns
            -------
            int
                The status of the move (0 for success, 99 for aborted).

            Raises
            ------
            ValueError
                If `t` is not a non-negative scalar.

            Notes
            -----
            - This method is executed in a blocking manner and will not return until the motion is complete.
            - The trajectory is calculated using Radial Basis Function (RBF) interpolation.
        """
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(added_trq, dim=self.nj)

        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        if not isscalar(t) or t <= 0:
            raise ValueError("Time must be non-negative scalar")

        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        _n = len(_time)
        _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
        qi = decodeRBF(_s, pathRBF)
        qdoti = gradientPath(qi, _time)
        _fac = np.max(np.max(np.abs(qdoti), axis=0) / self.qdot_max)
        if _fac > 1:
            _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
            self.Message(f"Execution time will be prolonged due to bounded joint velocities form {t:.1f}s to {_time:.1f}s.", 2)
            _n = len(_time)
            _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
            qi = decodeRBF(_s, pathRBF)
            qdoti = gradientPath(qi, _time)

        self.Message("JRBFPath started", 2)
        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("JRBFPath not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.JOINT_RBFPATH.value

        if direction == "Backward":
            self._last_status = self._loop_joint_traj(qi[::-1, :], qdoti[::-1, :], trq, _time, wait=wait, **kwargs)
        else:
            self._last_status = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("JRBFPath finished", 2)
        return self._last_status

    # Task space motion
    def GoTo_T(self, x: Union[Pose3DType, HomogeneousMatrixType], v: Optional[Velocity3DType] = None, FT: Optional[WrenchType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Move the robot to the target pose and velocity in Cartesian space.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., pose or transformation matrix).
        v : Velocity3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : WrenchType, optional
            Target force/torque in Cartesian space. Default is a zero wrench vector (6,).
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
        x = x2x(x)
        if v is None:
            v = np.zeros(6)
        else:
            v = vector(v, dim=6)
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)
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
                v = self.ObjectToWorld(v)
                FT = self.ObjectToWorld(FT)
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

    def GoTo_X(self, x: Union[Pose3DType, HomogeneousMatrixType], v: Optional[Velocity3DType] = None, FT: Optional[WrenchType] = None, wait: Optional[float] = None, impedance: Optional[ArrayLike] = None, **kwargs: Any) -> int:
        """Update task pose using cartesian space controller and wait

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., Pose, Transformation matrix).
        v : Velocity3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : WrenchType, optional
            WrenchType, optional, NOT USED!
            Target force/torque in Cartesian space. Default is a zero wrench vector (6,).
        wait : float, optional
            The time to wait after the movement, by default the sample time (`self.tsamp`).
        impedance : ArrayLike, optional
            The Cartesian impedance to set during the movement.
        **kwargs : dict
            Additional keyword arguments for special use.

        The robot will be moved using Cartesian control.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        raise NotImplementedError("Cartesian space controller not implemented!")

    def GoTo_JT(
        self,
        x: Union[Poses3DType, HomogeneousMatricesType],
        t: TimesType,
        wait: Optional[float] = None,
        traj_samp_fac: Optional[float] = None,
        max_iterations: int = 1000,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: Optional[str] = None,
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointConfigurationType] = None,
        x_opt: Optional[Pose3DType] = None,
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
            Target end-effector poses in Cartesian space.
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
            Task velocity for null space, by default None (using zeros).
        qdot_ns : JointConfigurationType, optional
            Joint velocity for null space, by default None (using zeros).
        x_opt : Pose3DType, optional
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
        else:
            q_opt = vector(q_opt, dim=self.nj)
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

        self.Message("Cartesian motion -> joint motion transformation ...", 2)

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

        self.Message(f"Cartesian motion -> joint motion transformation: Done ({qi.shape[0]} points)", 2)

        if self._last_status == MotionResultCodes.MOTION_SUCCESS.value:
            qi[-1, :] = q_path[-1, :]
            qdoti = gradientPath(qi, _time)
            qdoti[-1, :] = qdoti[-1, :] * 0

            _fac = np.max(np.abs(qdoti) / self.qdot_max, axis=1)
            if np.max(_fac) > 1:
                for i in range(1, _time.shape[0]):
                    if _fac[i] > 1:
                        _time[i:] += (_time[i] - _time[i - 1]) * (_fac[i] - 1)
                qdoti = gradientPath(qi, _time)
                qdoti[-1, :] = qdoti[-1, :] * 0
                self.Message(f"Execution time will be prolonged due to bounded joint velocities form {t[-1]:.1f}s to {_time[-1]:.1f}s.", 2)

            _t_traj = self.simtime()
            self._last_status = self.GoTo_qtraj(qi, qdoti, np.zeros(qi.shape), _time)
            if self._last_status == MotionResultCodes.MOTION_SUCCESS.value:
                self.Update()
                if self._do_motion_check:
                    while (self.simtime() - _t_traj) < (_time[-1] + wait):
                        if self._abort:
                            self.WarningMessage("Motion aborted by user")
                            self.StopMotion()
                            return MotionResultCodes.MOTION_ABORTED.value
                        elif self._do_motion_check and self._motion_check_callback is not None:
                            self._last_status = self._motion_check_callback(self)
                            if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                                self.WarningMessage("Motion aborted")
                                self.StopMotion()
                                self._command.mode = CommandModeCodes.ABORT.value
                            return self._last_status
                        elif (self._motion_error is not None) and (self._motion_error != 0):
                            self.WarningMessage("Motion aborted due to motion controller error")
                            return self._motion_error
                        self._sleep(self.tsamp)
                        self.Update()
        else:
            self.WarningMessage("Cartesian movement not feasible!")

        return self._last_status

    def SetUserNSTaskCallback(self, fun: Callable[..., Any]) -> None:
        """
        Set the user callback function for null-space task.

        Parameters
        ----------
        fun : Callable[..., Any]
            The callback function to be called in GoTo_TC when null_space_task is set to 'User'

        Returns
        -------
        None
        """
        self._user_null_space_task_callback = fun

    def GoTo_TC(
        self,
        x: Union[Pose3DType, HomogeneousMatrixType],
        v: Optional[Velocity3DType] = None,
        FT: Optional[WrenchType] = None,
        timeout: Optional[float] = None,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointConfigurationType] = None,
        x_opt: Optional[Pose3DType] = None,
        Kp: Optional[float] = None,
        Kff: Optional[float] = None,
        Kns: Optional[float] = None,
        vel_fac: Optional[float] = None,
        **kwargs: Any,
    ) -> int:
        """
        Kinematic controller for controlling robot end-effector in Cartesian space.

        This function uses inverse kinematics to move the robot's end-effector to a desired
        position while managing joint space motion through various null-space control strategies.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose, expressed in the specified task space.
        v : Velocity3DType, optional
            End-effector velocity (6,), by default None (zero velocity).
        FT : WrenchType, optional
            End-effector Force/Torque (6,), by default None (zero force/torque).
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
            Task-space velocity, by default None (zero velocity).
        qdot_ns : JointConfigurationType, optional
            Joint velocity for null-space control, by default None (zero joint velocity).
        x_opt : Pose3DType, optional
            Optimal end-effector pose, by default None (calculated from `self.Kinmodel(q_opt)`).
        Kp : float, optional
            Proportional gain for position control, by default None (using `self._default.Kp`).
        Kns : float, optional
            Null-space gain, by default None (using `self._default.Kns`).
        vel_fac : float, optional
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
            rv = np.zeros(6)
        else:
            rv = vector(v, dim=6)
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)
        if timeout is None:
            timeout = 0.0
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
        else:
            q_opt = vector(q_opt, dim=self.nj)

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
        rx = x2x(x)
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
            rv = self.ObjectToWorld(rv)
            FT = self.ObjectToWorld(FT)
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

        rp = copy.deepcopy(rx[:3])
        rR = copy.deepcopy(q2r(rx[3:]))
        self._command.x = rx
        self._command.v = rv

        while True:
            qq = self._command.q
            p, R, J = self.Kinmodel(qq, out="pR")
            ep = rp - p
            eR = qerr(r2q(rR @ R.T))
            ee = np.hstack((ep, eR))

            if check_option(task_cont_space, "World"):
                RC = np.block([[self.TBase[:3, :3] if i == j else np.zeros((3, 3)) for j in range(2)] for i in range(2)]).T
            elif check_option(task_cont_space, "Robot"):
                RC = np.eye(6)
            elif check_option(task_cont_space, "Tool"):
                RC = np.block([[R if i == j else np.zeros((3, 3)) for j in range(2)] for i in range(2)]).T
            elif check_option(task_cont_space, "Object"):
                RC = np.block([[self.TObject[:3, :3] if i == j else np.zeros((3, 3)) for j in range(2)] for i in range(2)]).T
            else:
                raise ValueError(f"Task space '{task_cont_space}' not supported")

            ee = RC @ ee
            J = RC @ J
            rv = RC @ rv
            ux = Kff * rv + Kp * ee
            trq = J.T @ FT
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
                een = xerr(x_opt, map_pose(p=p, R=R))
                qdn = Kns * np.linalg.pinv(J) @ een
            elif check_option(null_space_task, "TrackPath"):
                qdn = Kns * np.linalg.pinv(J) @ ee
            elif check_option(null_space_task, "TaskVelocity"):
                qdn = np.linalg.pinv(J) @ rv
            elif check_option(null_space_task, "JointVelocity"):
                qdn = rqdn
            elif check_option(null_space_task, "User"):
                qdn = self._user_null_space_task_callback(self, **kwargs)

            uq = Jp @ ux
            _fac = np.max(np.abs(uq) / _vel)
            if _fac > 1:
                uq = uq / _fac
            uNS = NS @ qdn
            if any(abs(_vel - uq) < 1e-3) or any(abs(-_vel - uq) < 1e-3):
                uNS = uNS * 0
            else:
                _fac = max(1, np.max(uNS / (_vel - uq)), np.max(uNS / (-_vel - uq)))
                if _fac > 1:
                    uNS = uNS / _fac
            u = uq + uNS
            rq = qq + u * self.tsamp
            if self.CheckJointLimits(rq):
                self._command.mode = imode
                self._command.qdot = np.zeros(self.nj)
                self._command.v = np.zeros(6)
                self.WarningMessage(f"Joint limits reached: {self.q}")
                return MotionResultCodes.JOINT_LIMITS.value

            self._last_status = self.GoTo_q(rq, u, trq, self.tsamp, **kwargs)
            if self.simtime() - tx > timeout or (np.linalg.norm(ep) < pos_err and np.linalg.norm(eR) < ori_err):
                self._command.mode = imode
                return self._last_status

    def Check_CPath(
        self,
        x: CartesianPathType,
        t: TimesType,
        q_init: JointConfigurationType,
        wait: Optional[float] = None,
        traj_samp_fac: Optional[float] = None,
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
        x_opt: Optional[Pose3DType] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
        state: str = "Commanded",
        **kwargs: Any,
    ) -> int:
        """
        Transforms Cartesian space trajectory to joiont space using inverse kinematics and then check if feasible.

        Parameters
        ----------
        x : CartesianPathType
            Target end-effector pose path in Cartesian space (n, 7) or (n, 4, 4).
        t : TimesType
            Time vector for trajectory (n,).
        q_init : JointConfigurationType
            initial joint configuration, by default None (using current configuration).
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
        x_opt : Pose3DType, optional
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
        list
            int
                Status of the operation: 0 for success, non-zero for failure.
            t_new
                New time vesctor considering joint velocity limits

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
        if q_init is None:
            q_init = self.GetJointPos(state=state)
        else:
            q_init = vector(q_init, dim=self.nj)
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
        else:
            q_opt = vector(q_opt, dim=self.nj)
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

        N = len(t)
        _time = t.copy()
        if N == 1:
            rx = x2x(x)
            q_path, _status = self.IKin(
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
            if _status == 0:
                _vel = (q_path - q_init) / _time
                _fac = np.max(_vel / self.qdot_max)
                if _fac > 1:
                    _time = _time * _fac
                    return _status, _time
        else:
            if x.ndim == 3:
                rx = uniqueCartesianPath(t2x(x))
            else:
                rx = uniqueCartesianPath(x)
            q_path, _status = self.IKinPath(
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
            if _status == 0:
                qi = q_path.copy()
                qdoti = gradientPath(qi, _time)

                _fac = np.max(np.abs(qdoti) / self.qdot_max, axis=1)
                if np.max(_fac) > 1:
                    for i in range(1, _time.shape[0]):
                        if _fac[i] > 1:
                            _time[i:] += (_time[i] - _time[i - 1]) * (_fac[i] - 1)
                    return _status, _time

        return _status, None

    def _loop_cartesian_traj(self, xi: Union[Poses3DType, HomogeneousMatricesType], vi: Velocities3DType, FT: Optional[WrenchType], time: TimesType, wait: float = 0, check_feasibility: bool = False, **kwargs: Any) -> int:
        """
        Executes a Cartesian trajectory for the robot's end-effector.

        This function controls the robot to follow a given Cartesian trajectory in terms of
        position, velocity, and force/torque while checking for motion errors, abort conditions,
        and motion checks.

        Parameters
        ----------
        xi : Union[Poses3DType, HomogeneousMatricesType]
            The target Cartesian poses for the trajectory.
        vi : Velocities3DType
            The target Cartesian velocities for the trajectory.
        FT : WrenchType, optional
            The target force/torque sensor data (6,).
        time : TimesType
            Time values corresponding to each point in the trajectory (n,).
        wait : float, optional
            The wait time after motion completion, by default 0.
        check_feasibility : bool, optional
            Whether to check the feasibility of the trajectory (joint limits), by default False.
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
        if check_feasibility:
            _q = self.GetJointPos()
            status, new_time = self.Check_CPath(xi, time, q_init=_q)
            if status:
                self.WarningMessage("Trajectory not feasible - joint limits or singularity reached")
                return MotionResultCodes.NOT_FEASIBLE.value
            elif new_time is not None and new_time[-1] > time[-1]:
                _time = np.arange(0.0, new_time[-1] + self.tsamp, self.tsamp)
                _time[-1] = new_time[-1]
                self.Message(f"Execution time will be prolonged due to bounded joint velocities form {time[-1]:.1f}s to {_time[-1]:.1f}s.", 2)
                _xi = interpCartesianPath(new_time, xi, _time)
                xi = _xi.copy()
                vi = gradientCartesianPath(xi, _time)
                time = _time.copy()

        if self._control_strategy in ["JointPositionTrajectory"]:
            # Joint Position Trajectory control strategy
            self._last_status = self.GoTo_JT(xi, time, wait=wait, **kwargs)
        else:
            # Cartesian-based control strategy
            for xt, vt in zip(xi, vi):
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self.StopMotion()
                    return MotionResultCodes.MOTION_ABORTED.value
                elif self._do_motion_check and self._motion_check_callback is not None:
                    self._last_status = self._motion_check_callback(self)
                    if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                        self._command.qdot = np.zeros(self.nj)
                        self._command.v = np.zeros(6)
                        self.WarningMessage("Motion check stopped motion")
                        self.StopMotion()
                        return self._last_status
                self._last_status = self.GoTo_T(xt, vt, FT, wait=self.tsamp, **kwargs)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Motion aborted")
                    self.StopMotion()
                    return self._last_status

            # Last sample
            self._last_status = self.GoTo_T(xi[-1, :], np.zeros(6), FT, wait=self.tsamp, timeout=kwargs.get("TC_timeout", self._default.TCTimeout), **kwargs)

            tx = self.simtime()
            while self.simtime() - tx < wait:
                self.Update()
                self._sleep(self.tsamp)

            # Update joint space values if not updated in GoTo_T
            _max_err = np.ones(6)
            _max_err[:3] = self._default.PosErr
            _max_err[3:] = self._default.OriErr
            if np.any(abs(xerr(self._command.x, self.Kinmodel(self._command.q)[0])) > _max_err):
                _qx, _stat = self.IKin(self.x_ref, self.q, Kp=10)
                if _stat == MotionResultCodes.MOTION_SUCCESS.value:
                    self._command.q = _qx
                self._command.qdot = np.zeros(self.nj)

        return self._last_status

    def CMove(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[Velocity3DType] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        min_pos_dist: Optional[float] = None,
        min_ori_dist: Optional[float] = None,
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Executes a Cartesian move with specified target position, velocity, and trajectory.

        The robot moves its end-effector to a target position with optional velocity, trajectory type,
        and additional force/torque settings.

        Parameters
        ----------
        x : Pose3DType
            The target Cartesian pose (7,).
        t : float, optional
            The duration for the movement, by default None.
        vel : float, optional
            The velocity at which the end-effector moves, by default None.
        vel_fac : Velocity3DType, optional
            A factor to scale the velocity, by default None.
        traj : str, optional
            The trajectory type, by default None.
        short : bool, optional
            Whether to shorten the path, by default None.
        wait : float, optional
            The wait time after the movement, by default None.
        task_space : str, optional
            The task space reference frame, by default None.
        added_FT : WrenchType, optional
            Additional force/torque to be applied, by default None.
        state : str, optional
            The state of the robot (e.g., "Commanded" or "Actual"), by default "Commanded".
        min_pos_dist : float, optional
            The minimum position distance to target positions for movement execution, by default None.
        min_ori_dist : float, optional
            The minimum distance between actual and target orientation for movement execution, by default None.
        asynchronous : bool, optional
            Whether the motion should be performed asynchronously, by default False.
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        int
            The status code of the move (0 if successful, non-zero if error occurred).

        Raises
        ------
        ValueError
            If an unsupported task space or parameter shape is provided.
        """
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC CMove", 2)
            _th = Thread(
                target=self._CMove,
                args=(x,),
                kwargs={
                    "t": t,
                    "vel": vel,
                    "vel_fac": vel_fac,
                    "traj": traj,
                    "short": short,
                    "wait": wait,
                    "task_space": task_space,
                    "added_FT": added_FT,
                    "state": state,
                    "min_pos_dist": min_pos_dist,
                    "min_ori_dist": min_ori_dist,
                    **kwargs,
                },
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, min_pos_dist=min_pos_dist, min_ori_dist=min_ori_dist, **kwargs)

    def _CMove(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[Velocity3DType] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
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
        x : Pose3DType
            The target Cartesian pose (7,).
        t : float, optional
            The duration for the movement, by default None.
        vel : float, optional
            The velocity at which the end-effector moves, by default None.
        vel_fac : Velocity3DType, optional
            A factor to scale the velocity, by default None.
        traj : str, optional
            The trajectory type, by default None.
        short : bool, optional
            Whether to shorten the path, by default None.
        wait : float, optional
            The wait time after the movement, by default None.
        task_space : str, optional
            The task space reference frame, by default None.
        added_FT : WrenchType, optional
            Additional force/torque to be applied, by default None.
        state : str, optional
            The state of the robot (e.g., "Commanded" or "Actual"), by default "Commanded".
        min_pos_dist : float, optional
            The minimum position distance to target positions for movement execution, by default None.
        min_ori_dist : float, optional
            The minimum distance between actual and target orientation for movement execution, by default None.
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
            FT = vector(added_FT, dim=6)
        if min_pos_dist is None:
            min_pos_dist = self._default.MinPosDist
        if min_ori_dist is None:
            min_ori_dist = self._default.MinOriDist

        kwargs.setdefault("kinematics", self._default.Kinematics)

        x = self.spatial(x)
        if wait is None:
            wait = self.tsamp

        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
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

        kwargs.setdefault("task_space", task_space)
        rx = np.array(t2x(rT))

        dist = xerr(rx, self._command.x)
        if np.linalg.norm(dist[:3]) < min_pos_dist and np.linalg.norm(dist[3:]) < min_ori_dist:
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
                    vel_fac = np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3)))
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=6)
                _vel = self.v_max * vel_fac
                self.Message(f"CMove started: {rx} with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
            else:
                if isscalar(vel):
                    # _vel = np.ones(6) * vel
                    _vel = np.concatenate((normalize(dist[:3]) * vel, self.v_max[3:]))
                    self.Message(f"CMove started: {rx} in {task_space} space  with velocity {vel:.1f}m/s", 2)
                elif isvector(vel, dim=2):
                    # _vel = np.concatenate((vel[0] * np.ones(3), vel[1] * np.ones(3)))
                    _norm = np.linalg.norm(dist[:3])
                    if _norm < 1e-3:
                        _dp = np.ones(3)
                    else:
                        _dp = dist[:3] / _norm
                    _norm = np.linalg.norm(dist[3:])
                    if _norm < 1e-3:
                        _dr = np.ones(3)
                    else:
                        _dr = dist[3:] / _norm
                    _vel = np.concatenate((_dp * vel[0], _dr * vel[1]))
                    self.Message(f"CMove started: {rx} in {task_space} space with velocity {vel[0]:.1f}m/s and {vel[1]:.1f}rd/s ", 2)
                else:
                    _vel = vector(vel, dim=6)
                    self.Message(f"CMove started: {rx} in {task_space} space with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
            _vel = np.clip(np.abs(_vel), 0, self.v_max)
            _vel[np.where(_vel < 1e-3)[0]] = np.inf
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CMove started: {rx} in {task_space} space in {_time[-1]:.1f}s", 2)

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        if check_option(traj, "Jerk"):
            v_max = kwargs.get('v_max', _vel)
            a_max = kwargs.get('a_max', self._default.TaskAcceleration)
            jerk_max = kwargs.get('jerk_max', self._default.TaskJerk)
            if t is None:
                xi, vi, _ = ctraj(x0, rx, [0, self.tsamp], traj=traj, short=short, v_max=v_max, a_max=a_max, jerk_max=jerk_max)
                _time = np.arange(xi.shape[0]) * self.tsamp
            else:
                xi, vi, _ = ctraj(x0, rx, _time, traj=traj, short=short, v_max=v_max, a_max=a_max, jerk_max=jerk_max)
                if xi.shape[0] > _time.shape[0]:
                    _time = np.arange(xi.shape[0]) * self.tsamp
                    self.Message(f"Execution time will be prolonged due to bounded task velocities or accelerations from {t:.1f}s to {_time[-1]:.1f}s.", 2)
        else:
            xi, vi, _ = ctraj(x0, rx, _time, traj=traj, short=short)
            _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
            if (_fac > 1) or (t is None):
                _tend = max(_time[-1] * _fac, 0.5) + self.tsamp
                if _tend > _time[-1]:
                    self.Message(f"Execution time will be prolonged due to bounded task velocities form {_time[-1]:.1f}s to {_tend:.1f}s.", 2)
                _time = np.arange(0.0, _tend, self.tsamp)
                xi, vi, _ = ctraj(x0, rx, _time, traj=traj, short=short)

        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("CMove not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CMove finished", 2)
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
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in Cartesian space based on a displacement vector.

        The robot moves its end-effector from the current position by a given displacement.

        Parameters
        ----------
        dx : ArrayLike
            Displacement in Cartesian space (3, ) for position or (3, 3) for rotation or (4, ) for quaternion.
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
        added_FT : WrenchType, optional
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
        dx = self.spatial(dx)
        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            if isvector(dx, dim=3):
                rT = T0 @ map_pose(p=dx, out="T")
            elif dx.shape == (3, 3):
                rT = T0 @ map_pose(R=dx, out="T")
            elif isvector(dx, dim=4):
                rT = T0 @ map_pose(Q=dx, out="T")
            else:
                raise ValueError(f"Parameter shape {dx.shape} not supported")
        else:
            rT = self.GetPose(out="T", task_space=task_space, kinematics=kwargs["kinematics"], state=state)
            if isvector(dx, dim=3):
                rT[:3, 3] += dx
            elif dx.shape == (3, 3):
                rT[:3, :3] = dx @ rT[:3, :3]
            elif isvector(dx, dim=4):
                rT[:3, :3] = q2r(dx) @ rT[:3, :3]
            else:
                raise ValueError(f"Parameter shape {dx.shape} not supported")
        rx = t2x(rT)
        self.Message("CMoveFor -> CMove", 2)
        self._last_status = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def CApproach(
        self,
        x: Pose3DType,
        dx: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot towards a target pose with an offset.

        The robot moves to a position that is offset by a specified displacement from the given target pose.

        Parameters
        ----------
        x : Pose3DType
            The target pose (7, ) or (4, 4).
        dx : Vector3DType
            The displacement to move towards (3, ).
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
        added_FT : WrenchType, optional
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
        dx = vector(dx, dim=3)
        if _x.shape == (4, 4):
            rx = map_pose(T=_x)
        elif isvector(x, dim=7):
            rx = _x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")
        rx[:3] += dx
        self.Message("CApproach -> CMove", 2)
        self._last_status = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def CLine(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[Velocity3DType] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Execute a linear move to a target Cartesian position.

        The robot moves to the target position using a trapezoidal velocity profile.

        Parameters
        ----------
        x : Pose3DType
            The target pose (7,).
        t : float, optional
            Time to complete the movement, by default None.
        vel : float, optional
            Maximum velocity, by default None.
        vel_fac : Velocity3DType, optional
            Velocity scaling factor, by default None.
        short : bool, optional
            Whether to shorten the trajectory, by default None.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        self.Message("CLine -> CMove", 2)
        self._last_status = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def CLineFor(
        self,
        dx: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[Velocity3DType] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Execute a linear move based on a displacement vector.

        The robot moves from its current position by the given displacement using a trapezoidal velocity profile.

        Parameters
        ----------
        dx : Pose3DType
            Displacement pose (7,).
        t : float, optional
            Time to complete the movement, by default None.
        vel : float, optional
            Maximum velocity, by default None.
        vel_fac : Velocity3DType, optional
            Velocity scaling factor, by default None.
        short : bool, optional
            Whether to shorten the trajectory, by default None.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        self.Message("CLineFor -> CMoveFor", 2)
        self._last_status = self.CMoveFor(dx, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def CArc(
        self,
        x: Pose3DType,
        pC: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Execute an arc trajectory movement in Cartesian space.

        The robot moves its end-effector along an arc defined by a target pose and a center position.

        Parameters
        ----------
        x : Pose3DType
            Target position or pose (7, ) or (4, 4).
        pC : Vector3DType
            Center of the arc in Cartesian space (3, ).
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
        added_FT : WrenchType, optional
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
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC CArc", 2)
            _th = Thread(
                target=self._CArc,
                args=(x, pC),
                kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "traj": traj, "short": short, "wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CArc(x, pC, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

    def _CArc(
        self,
        x: Pose3DType,
        pC: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        **kwargs: Any,
    ) -> int:
        """
        Execute the internal arc trajectory movement logic in Cartesian space.

        This method computes and executes the arc movement trajectory for the robot's end-effector.

        Parameters
        ----------
        x : Pose3DType
            Target position or pose (7, ) or (4, 4).
        pC : Vector3DType
            Center of the arc in Cartesian space (3, ).
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
        added_FT : WrenchType, optional
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
            FT = vector(added_FT, dim=6)
        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs.setdefault("task_space", task_space)

        x = self.spatial(x)
        pC = vector(pC, dim=3)
        if wait is None:
            wait = self.tsamp
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)

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
                    vel_fac = np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3)))
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=6)
                _vel = self.v_max * vel_fac
            else:
                if isscalar(vel):
                    _vel = np.ones(6) * vel
                elif isvector(vel, dim=2):
                    _vel = np.concatenate((vel[0] * np.ones(3), vel[1] * np.ones(3)))
                else:
                    _vel = vector(vel, dim=6)
            _vel = np.clip(np.abs(_vel), 0, self.v_max)
            self.Message(f"CArc started: {rx}/{rpC} in {task_space} with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CArc started: {rx}/{rpC} in {task_space} in {_time[-1]:.1f}s", 2)

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        xi, vi, _ = carctraj(x0, rx, rpC, _time, traj=traj, short=short)
        _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _tend = max(_time[-1] * _fac, 0.5) + self.tsamp
            self.Message(f"Execution time will be prolonged due to bounded task velocities form {_time[-1]:.1f}s to {_tend:.1f}s.", 2)
            _time = np.arange(0.0, _tend, self.tsamp)
            xi, vi, _ = carctraj(x0, rx, rpC, _time, traj=traj, short=short)

        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("CArc not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CArc finished", 2)
        return self._last_status

    def CPath(
        self,
        path: CartesianPathType,
        t: Union[TimesType, float],
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Execute a path trajectory movement in Cartesian space.

        The robot moves its end-effector along a defined Cartesian path.

        Parameters
        ----------
        path : CartesianPathType
            Path in Cartesian space (n, 7) or (n, 4, 4) representing positions or poses.
        t : Union[TimesType, float]
            Time to complete the movement, (n,) or scalar.
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC CPath", 2)
            _th = Thread(
                target=self._CPath,
                args=(path, t),
                kwargs={"wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CPath(path, t, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

    def _CPath(self, path: CartesianPathType, t: Union[TimesType, float], direction: str = "Forward", wait: Optional[float] = None, task_space: Optional[str] = None, added_FT: Optional[WrenchType] = None, state: str = "Commanded", **kwargs: Any) -> int:
        """
        Execute the internal path trajectory movement logic in Cartesian space.

        This method computes and executes the path movement trajectory for the robot's end-effector.

        Parameters
        ----------
        path : CartesianPathType
            Path in Cartesian space (n, 7) or (n, 4, 4).
        t : Union[TimesType, float]
            Time to complete the movement, (n,) or scalar.
        direction : str, optional
            Direction of movement, by default "Forward".
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        if path.ndim == 3:
            path = uniqueCartesianPath(t2x(path))
        else:
            path = uniqueCartesianPath(path)

        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)

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
            t = np.max(t)
        else:
            if not isscalar(t):
                t = np.max(t)
            _s = np.linspace(0, t, N)
        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        xi = interpCartesianPath(_s, path, _time)
        vi = gradientCartesianPath(xi, _time)
        _fac = np.max(np.max(np.abs(vi), axis=0) / self.v_max)
        if _fac > 1:
            _s = np.linspace(0.0, t * _fac, N)
            _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
            self.Message(f"Execution time will be prolonged due to bounded task velocities form {t:.1f}s to {_time[-1]:.1f}s.", 2)
            xi = interpCartesianPath(_s, path, _time)
            vi = gradientCartesianPath(xi, _time)
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

        self.Message(f"CPath started: {path.shape[0]} points in {np.max(t)}s", 2)
        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("CPath not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Message("CPath finished", 2)
        self.Stop()
        return self._last_status

    def CRBFPath(
        self,
        pathRBF: Dict[str, Any],
        t: float,
        direction: str = "Forward",
        wait: float = None,
        task_space: str = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Execute a path trajectory using Radial Basis Function (RBF) interpolation.

        The robot moves along a path defined by a Radial Basis Function (RBF), with the option to move in the forward or backward direction.

        Parameters
        ----------
        pathRBF : Dict[str, Any]
            RBF path data containing control points and weights. The dictionary must include "c" for control points and "w" for weights.
        t : float
            Total time for completing the path.
        direction : str, optional
            Direction of motion, either "Forward" or "Backward", by default "Forward".
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC CRBFPath", 2)
            _th = Thread(
                target=self._CRBFPath,
                args=(pathRBF, t),
                kwargs={"direction": direction, "wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CRBFPath(pathRBF, t, direction=direction, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

    def _CRBFPath(
        self,
        pathRBF: Dict[str, Any],
        t: float,
        direction: str = "Forward",
        wait: float = None,
        task_space: str = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        **kwargs: Any,
    ) -> int:
        """
        Internal method to compute and execute the RBF path trajectory in Cartesian space.

        Parameters
        ----------
        pathRBF : Dict[str, Any]
            RBF path data containing control points and weights. The dictionary must include "c" for control points and "w" for weights.
        t : float
            Total time for completing the path.
        direction : str, optional
            Direction of motion, either "Forward" or "Backward", by default "Forward".
        wait : float, optional
            Wait time after movement, by default None.
        task_space : str, optional
            The task space for the movement, by default None.
        added_FT : WrenchType, optional
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
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)

        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs.setdefault("task_space", task_space)

        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        if not isscalar(t) or t <= 0:
            raise ValueError("Time must be non-negative scalar")

        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        _n = len(_time)
        _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
        if pathRBF["w"].shape[1] == 3:
            pi = decodeRBF(_s, pathRBF)
            pdi = gradientPath(pi, _time)
            _fac = np.max(np.max(np.abs(pdi), axis=0) / self.v_max[:3])
            if _fac > 1:
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                self.Message(f"Execution time will be prolonged due to bounded task velocities form {t:.1f}s to {_time[-1]:.1f}s.", 2)
                _n = len(_time)
                _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
                xi = decodeRBF(_s, pathRBF)
                xdi = np.hstack((gradientPath(xi, _time), np.zeros((_n, 3))))
            vi = np.hstack((xdi, np.zeros((_n, 3))))
        elif pathRBF["w"].shape[1] == 7:
            xi = decodeCartesianRBF(_s, pathRBF)
            vi = gradientCartesianPath(xi, _time)
            _fac = np.max(np.max(np.abs(vi), axis=0) / self.v_max)
            if _fac > 1:
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                self.Message(f"Execution time will be prolonged due to bounded task velocities form {t:.1f}s to {_time[-1]:.1f}s.", 2)
                _n = len(_time)
                _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
                xi = decodeCartesianRBF(_s, pathRBF)
                vi = gradientCartesianPath(xi, _time)
        else:
            raise ValueError(f"Wrong RBF path size {pathRBF['w'].shape[1]}. Must be 3 or 7.")

        if direction == "Backward":
            _initial_x = xi[-1, :]
        else:
            _initial_x = xi[0, :]

        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        xe = np.amax(np.abs(self.TaskDistance(_initial_x)) / self.v_max) * 2
        if xe > 0.02:
            self.Message("Move to path -> CMove", 2)
            self._last_status = self._CMove(_initial_x, max(xe, 0.2), traj="Poly", short=True, wait=0, added_FT=FT, **kwargs)
            if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                self.WarningMessage("Robot did not move to path start")
                return self._last_status

        self.Message("CRBFPath started", 2)
        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("CRBFPath not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value
        if not self.Start():
            self._semaphore.release()
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.CARTESIAN.value

        if direction == "Backward":
            self._last_status = self._loop_cartesian_traj(xi[::-1, :], vi[::-1, :], FT, _time, wait=wait, **kwargs)
        else:
            self._last_status = self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CRBFPath finished", 2)
        return self._last_status

    def CJogging(self, scale: Optional[ArrayLike] = None, null_space_task: Optional[str] = None) -> int:
        """
        Jogging function for the robot using a space navigator.

        This function allows the robot to jog (move) using a space navigator (3D input device).
        The robot's movement is controlled based on the space navigator's input, with scaling factors
        and different velocity modes. The user can toggle between position and orientation control, as
        well as change the velocity factor. The function terminates when both buttons on the space
        navigator are pressed simultaneously.

        Parameters
        ----------
        scale : ArrayLike, optional
            A scaling factor to adjust the velocity. If not provided, a default scaling factor of 10% of the robot's
            maximum velocity is used. If a 2D vector is provided, it will be expanded into a 6D vector
            for position and orientation scaling. If a scalar is provided, it will be converted into a 6D vector.

        null_space_task : str, optional
            The name of the null space task to be used by the robot. If not provided, the robot's default null space task
            is used.

        Returns
        -------
        int
            Motion result code after jogging stops.

        Notes
        -----
        The jogging functionality is toggled between position (translation) and orientation (rotation) by pressing
        the first button on the space navigator. The velocity factor can be toggled by pressing the second button.

        The jogging mode is terminated when both buttons on the space navigator are pressed simultaneously.
        """
        from robotblockset.spacenavigator import spacenavigator

        if scale is None:
            scale = self.v_max * 0.1  # Default scale for jogging
        elif isvector(scale, dim=2):
            scale = np.concatenate((scale[0] * np.ones(3), scale[1] * np.ones(3)))
        elif not isscalar(scale):
            scale = vector(scale, dim=6)
        vel_fac = np.clip(scale, 0, self.v_max / 2)

        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask

        _pos = [1, 1, 0]
        _ori = [1, 0, 1]
        _im = 0
        _fac = [1, 0.2]
        _iv = 0
        ginp = spacenavigator()
        _last_buttons = ginp.buttons.copy()
        self.Message("Jogging started", 2)
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.JOGGING.value
        while True:
            if _last_buttons[0] == 1 and ginp.buttons[0] == 0:
                _im = (_im + 1) % 3
                self.Message(f"Jogging mode: {_im} (0: all, 1: pos, 2: ori)", 2)
            if _last_buttons[1] == 1 and ginp.buttons[1] == 0:
                _iv = (_iv + 1) % 2
                self.Message(f"Jogging velocity factor: {_fac[_iv]}", 2)
            _last_buttons = ginp.buttons.copy()
            xprpy = map_pose(x=self.x_ref, out="pRPY")
            v = ginp.x * vel_fac * _fac[_iv] * np.concatenate((_pos[_im] * np.ones(3), _ori[_im] * np.ones(3)))
            xprpy += v * self.tsamp
            x = map_pose(pRPY=xprpy, out="x")
            self.GoTo_T(x, v=v, null_space_task=null_space_task)
            self.Update()
            if ginp.buttons[0] and ginp.buttons[1]:
                break

        self.Message("Jogging stopped", 2)
        self.Stop()
        ginp._stop_event.set()
        ginp = None
        del ginp
        return MotionResultCodes.MOTION_SUCCESS.value

    # Movements in Tool frame
    def TMove(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in tool space.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in tool space (7,) or (4, 4).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Type of trajectory to use, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("TMove -> CMove", 2)
        self._last_status = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Tool", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def TLine(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in tool space along a line.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in tool space (7,) or (4, 4).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("TLine -> CLine", 2)
        self._last_status = self.CLine(x, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Tool", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    # Movements in object frame
    def OMove(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in object space.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in object space (7,) or (4, 4).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Type of trajectory to use, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OMove -> CMove", 2)
        self._last_status = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OMoveFor(
        self,
        dx: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in object space by a given displacement.

        Parameters
        ----------
        dx : Vector3DType
            The displacement vector in object space (3,).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Type of trajectory to use, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OMoveFor -> CMoveFor", 2)
        self._last_status = self.CMoveFor(dx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OApproach(
        self,
        x: Pose3DType,
        dx: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Approach the target in object space.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in object space (7,) or (4, 4).
        dx : Vector3DType
            The displacement vector in object space (3,).
        t : float, optional
            Time to complete the approach, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Type of trajectory to use, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after approach, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the approach is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the approach (0 if successful, non-zero if failed).
        """
        self.Message("OApproach -> CApproach", 2)
        self._last_status = self.CApproach(x, dx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OLine(
        self,
        x: Pose3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in object space along a line.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in object space (7,) or (4, 4).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OLine -> CLine", 2)
        self._last_status = self.CLine(x, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OLineFor(
        self,
        x: Pose3DType,
        dx: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in object space by a given displacement along a line.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in object space (7,) or (4, 4).
        dx : Vector3DType
            The displacement vector in object space (3,).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OLineFor -> CLineFor", 2)
        self._last_status = self.CLineFor(x, dx, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OArc(
        self,
        x: Pose3DType,
        pC: Vector3DType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[ArrayLike] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot in object space along an arc.

        Parameters
        ----------
        x : Pose3DType
            The target position/pose in object space (7,) or (4, 4).
        pC : Vector3DType
            The center of the arc (3,).
        t : float, optional
            Time to complete the move, by default None.
        vel : float, optional
            Linear velocity in m/s, by default None.
        vel_fac : ArrayLike, optional
            Velocity scaling factor, by default None.
        traj : str, optional
            Type of trajectory to use, by default None.
        short : bool, optional
            If True, use a shorter trajectory, by default None.
        wait : float, optional
            Wait time after move, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OArc -> CArc", 2)
        self._last_status = self.CArc(x, pC, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def OPath(self, path: CartesianPathType, t: float, wait: Optional[float] = None, task_space: Optional[str] = None, added_FT: Optional[WrenchType] = None, state: str = "Commanded", asynchronous: Optional[bool] = None, **kwargs: Any) -> int:
        """
        Move the robot along a path in object space.

        Parameters
        ----------
        path : CartesianPathType
            The path to follow in object space (n, 7) or (n, 4, 4).
        t : float
            Time to complete the path move.
        wait : float, optional
            Wait time after move, by default None.
        task_space : str, optional
            Task space frame, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("OPath -> CPath", 2)
        self._last_status = self.CPath(path, t, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    def ORBFPath(
        self,
        pathRBF: Dict[str, Any],
        t: float,
        direction: str = "Forward",
        wait: float = None,
        task_space: str = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        asynchronous: Optional[bool] = None,
        **kwargs: Any,
    ) -> int:
        """
        Move the robot along a Radial Basis Function (RBF) path in object space.

        Parameters
        ----------
        pathRBF : Dict[str, Any]
            The RBF path data, including control points and weights.
        t : float
            Time to complete the path move.
        direction : str, optional
            Direction of the path, "Forward" or "Backward", by default "Forward".
        wait : float, optional
            Wait time after move, by default None.
        task_space : str, optional
            Task space frame, by default None.
        added_FT : WrenchType, optional
            Additional force/torque, by default None.
        state : str, optional
            The robot state ("Commanded" or "Actual"), by default "Commanded".
        asynchronous : bool, optional
            If True, the move is executed asynchronously, by default False.
        **kwargs : Any
            Additional arguments passed to internal methods.

        Returns
        -------
        int
            Status code of the move (0 if successful, non-zero if failed).
        """
        self.Message("ORBFPath -> CRBFPath", 2)
        self._last_status = self.CBFPath(pathRBF, t, direction=direction, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return self._last_status

    # Control strategy
    def AvailableStrategies(self) -> list[str]:
        """
        Get the available control strategies for the robot.

        Returns
        -------
        list[str]
            List of available control strategies, by default includes the current control strategy.
        """
        return [self._control_strategy]

    def SetStrategy(self, strategy: str) -> None:
        """
        Set the control strategy for the robot.

        Parameters
        ----------
        strategy : str
            The control strategy to set for the robot.
        """
        raise NotImplementedError("Setting control strategy is not supported")

    def GetStrategy(self) -> str:
        """
        Get the current control strategy of the robot.

        Returns
        -------
        str
            The current control strategy of the robot.
        """
        return self._control_strategy

    def isStrategy(self, strategy: str) -> bool:
        """
        Check if the current control strategy matches the provided strategy.

        Parameters
        ----------
        strategy : str
            The control strategy to check against the current strategy.

        Returns
        -------
        bool
            True if the current strategy matches the provided strategy, False otherwise.
        """
        return self._control_strategy.lower() == strategy.lower()

    def GetJointStiffness(self) -> JointConfigurationType:
        """
        Get the joint stiffness of the robot.

        Returns
        -------
        JointConfigurationType
            An array representing the joint stiffness, defaulting to a high stiffness value.
        """
        self.WarningMessage("Compliance not supported")
        return np.ones(self.nj) * 100000

    def SetJointStiffness(self, stiffness: JointConfigurationType, **kwargs: Any) -> None:
        """
        Set the joint stiffness for the robot.

        Parameters
        ----------
        stiffness : JointConfigurationType
            The stiffness values for the robot's joints.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def GetJointDamping(self) -> JointConfigurationType:
        """
        Get the joint damping of the robot.

        Returns
        -------
        JointConfigurationType
            An array representing the joint damping, defaulting to a value of 1 for each joint.
        """
        self.WarningMessage("Compliance not supported")
        return np.ones(self.nj)

    def SetJointDamping(self, damping: JointConfigurationType, **kwargs: Any) -> None:
        """
        Set the joint damping for the robot.

        Parameters
        ----------
        damping : JointConfigurationType
            The damping values for the robot's joints.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def SetJointSoft(self, softness: float, **kwargs: Any) -> None:
        """
        Set the joint softness for the robot.

        Parameters
        ----------
        softness : float
            The softness value for the robot's joints.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def SetJointStiff(self) -> None:
        """
        Set the joints to be stiff (high stiffness).

        This is a shorthand for setting joint stiffness to 1.0.
        """
        self.SetJointSoft(1.0)

    def SetJointCompliant(self) -> None:
        """
        Set the joints to be compliant (low stiffness).

        This is a shorthand for setting joint stiffness to 0.0.
        """
        self.SetJointSoft(0.0)

    def GetCartesianStiffness(self) -> Velocity3DType:
        """
        Get the Cartesian stiffness of the robot.

        Returns
        -------
        Velocity3DType
            An array representing the Cartesian stiffness, defaulting to a high stiffness value.
        """
        self.WarningMessage("Compliance not supported")
        return np.ones(6) * 100000

    def SetCartesianStiffness(self, stiffness: Velocity3DType, **kwargs: Any) -> None:
        """
        Set the Cartesian stiffness for the robot.

        Parameters
        ----------
        stiffness : Velocity3DType
            The stiffness values for the robot's Cartesian space.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def GetCartesianDamping(self) -> Velocity3DType:
        """
        Get the Cartesian damping of the robot.

        Returns
        -------
        Velocity3DType
            An array representing the Cartesian damping, defaulting to a value of 1 for each component.
        """
        self.WarningMessage("Compliance not supported")
        return np.ones(6)

    def SetCartesianDamping(self, damping: Velocity3DType, **kwargs: Any) -> None:
        """
        Set the Cartesian damping for the robot.

        Parameters
        ----------
        damping : Velocity3DType
            The damping values for the robot's Cartesian space.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def SetCartesianSoft(self, softness: float, **kwargs: Any) -> None:
        """
        Set the Cartesian softness for the robot.

        Parameters
        ----------
        softness : float
            The softness value for the robot's Cartesian space.
        **kwargs : Any
            Additional arguments passed to the method.
        """
        raise NotImplementedError("Compliance not supported")

    def SoftSetCartesianStiffness(self, stiffness: ArrayLike, t: float, dt: float = None, **kwargs: Any) -> None:
        """
        Smoothly ramp Cartesian stiffness/compliance over ``t`` seconds.

        Interpolates from current Cartesian stiffness to target ``stiffness`` in ``N`` steps,
        where ``N = max(1, round(t / dt))``. If ``dt`` is not specified, ``self.tsamp`` is used.

        Parameters
        ----------
        stiffness : ArrayLike
            Target Cartesian stiffness (skalar, 2D ali 6D vektor).
        t : float
            Desired ramp duration in seconds.
        dt : float, optional
            Step duration (če None, uporabi self.tsamp).
        **kwargs : Any
            Dodatni argumenti za SetCartesianStiffness.
        """
        if not isscalar(t) or t < 0:
            raise ValueError("Time must be a non-negative scalar")
        if dt is None:
            dt = self.tsamp
        # Expand target stiffness to 6D
        if isscalar(stiffness):
            stiffness_target = np.ones(6) * float(stiffness)
        elif isvector(stiffness, dim=2):
            # 2D: [trans, rot] -> [trans, trans, trans, rot, rot, rot]
            stiffness_target = np.hstack([np.ones(3) * float(stiffness[0]), np.ones(3) * float(stiffness[1])])
        else:
            stiffness_target = vector(stiffness, dim=6)

        stiffness_now = vector(self.GetCartesianStiffness(), dim=6)
        N = max(1, int(np.round(t / dt)))
        d_stiff = (stiffness_target - stiffness_now) / N
        for i in range(N):
            stiffness_now = stiffness_now + d_stiff
            self.SetCartesianStiffness(stiffness_now, **kwargs)
            if i < N - 1 and t > 0:
                sleep(dt)

    def SetCartesianStiff(self) -> None:
        """
        Set the robot's Cartesian space to be stiff (high stiffness).

        This is a shorthand for setting Cartesian stiffness to 1.0.
        """
        self.SetCartesianSoft(1.0)

    def SetCartesianCompliant(self) -> None:
        """
        Set the robot's Cartesian space to be compliant (low stiffness).

        This is a shorthand for setting Cartesian stiffness to 0.0.
        """
        self.SetCartesianSoft(0.0)

    def SetTeachMode(
        self,
    ) -> None:
        """
        Set the robot control to Tesch mode.

        BY default is sets the compliance to 0
        """
        if self.GetStrategy().lower().startswith("cart"):
            self.SetCartesianSoft(0)
        else:
            self.SetJointSoft(0)
        self.Message("Robot is entering Teach mode.", 2)

    def EndTeachMode(
        self,
    ) -> None:
        """
        End the robot control to Tesch mode.

        By default is sets the compliance to default value
        """
        self.ResetCurrentTarget()
        if self.GetStrategy().lower().startswith("cart"):
            self.SetCartesianSoft(1)
        else:
            self.SetJointSoft(1)
        self.Message("Robot is ending Teach mode.", 2)

    # Transformations
    def BaseToWorld(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """
        Map from robot base frame to world frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : ArrayLike
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

        Returns
        -------
        np.ndarray
            Mapped argument in the world frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        R0 = self.TBase[:3, :3]
        p0 = self.TBase[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):  # T
            p, R = map_pose(T=x, out="pR")
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
                # RR[:3, 3:] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                RRb = np.eye(6)
                RRb[:3, 3:] = v2s(self.TBase[:3, :3] @ self._actual.x[:3]).T
                return RR @ x + RRb @ self.vBase
            elif typ == "Wrench":  # wrench (F)
                # RR[3:6, :3] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                pass
            else:  # twist (v)
                # RR[:3, 3:] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                pass
            return RR @ x
        elif x.shape == (6, self.nj):  # J
            # RR = np.block([[R0, v2s(p0) @ R0], [np.zeros((3, 3)), R0]])
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            return RR @ x  # TODO: Preveri za premikajočo bazo!
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToBase(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """
        Map from world frame to robot base frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : ArrayLike
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

        Returns
        -------
        np.ndarray
            Mapped argument in the robot base frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        R0 = self.TBase[:3, :3].T
        p0 = -R0 @ self.TBase[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):  # T
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
        elif isvector(x, dim=7):  # x
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):  # R
            return R0 @ x
        elif isvector(x, dim=4):  # Q
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):  # p
            return R0 @ x + p0
        elif isvector(x, dim=6):  # v, F
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Twist":  # velocity
                # RR[:3, 3:] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                RRb = np.eye(6)
                RRb[:3, 3:] = v2s(self.TBase[:3, :3] @ self._actual.x[:3]).T
                return RR @ (x - RRb @ self.vBase)
            elif typ == "Wrench":  # wrench (F)
                # RR[3:6, :3] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                pass
            else:  # twist (v)
                # RR[:3, 3:] = v2s(p0) @ R0  # Tega se ne sme uporabiti
                pass
            return RR @ x
        elif x.shape == (6, self.nj):  # J
            # RR = np.block([[R0, v2s(p0) @ R0], [np.zeros((3, 3)), R0]])
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            return RR @ x  # TODO: Preveri za premikajočo bazo!
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def ObjectToWorld(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """
        Map from object frame to world frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : ArrayLike
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None. If "Wrench", the transformation considers the force.
            If "Twist", the transformation considers the velocities.

        Returns
        -------
        np.ndarray
            Mapped argument in the world frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        R0 = self.TObject[:3, :3]
        p0 = self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            # if typ == "Wrench":  # wrench (F)
            #     RR[3:6, :3] = v2s(p0) @ R0
            # elif typ == "Twist":  # twist (v)
            #     RR[:3, 3:] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            # RR = np.block([[R0, v2s(p0) @ R0], [np.zeros((3, 3)), R0]])
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            return RR @ x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToObject(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """
        Map from world frame to object frame.

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj).

        Parameters
        ----------
        x : ArrayLike
            Argument to map. It can be one of the following shapes:
            - pose (7,) or (4, 4)
            - position (3,)
            - orientation (4,) or (3, 3)
            - velocity or force (6,)
            - JacobianType (6, nj)
        typ : str, optional
            Transformation type, by default None. If "Wrench", the transformation considers the force.
            If "Twist", the transformation considers the velocities.

        Returns
        -------
        np.ndarray
            Mapped argument in the object frame.

        Raises
        ------
        ValueError
            If the parameter shape is not supported.
        """
        R0 = self.TObject[:3, :3].T
        p0 = -R0 @ self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="T")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            # if typ == "Wrench":  # wrench (F)
            #     RR[3:6, :3] = v2s(p0) @ R0
            # elif typ == "Twist":  # twist (v)
            #     RR[:3, 3:] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            # RR = np.block([[R0, v2s(p0) @ R0], [np.zeros((3, 3)), R0]])
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            return RR @ x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def CameraToBase(self, x: ArrayLike, typ: Optional[str] = None, kinematics: Optional[str] = None, state: Optional[str] = None, refresh: Optional[bool] = None) -> np.ndarray:
        """
        Map a variable from the camera frame to the robot base frame.

        The camera pose in the robot base frame is computed as
        ``Tflange @ CameraFrame``, where ``Tflange`` is the current robot flange
        pose in the robot base frame and ``CameraFrame`` is the camera pose
        relative to the flange.

        Parameters
        ----------
        x : ArrayLike
            Variable expressed in the camera frame. Supported shapes follow
            :func:`robotblockset.transformations.frame2world`, including poses,
            homogeneous transforms, positions, rotations, twists, and wrenches.
        typ : str, optional
            Transformation type, by default None. Use ``"Twist"`` or
            ``"Wrench"`` for spatial velocity or force/torque variables.
        kinematics : str, optional
            Kinematics source for the current robot pose, by default None.
        state : str, optional
            Robot state for the current robot pose, by default None.
        refresh : bool, optional
            Whether to refresh robot state before reading the pose, by default None.

        Returns
        -------
        np.ndarray
            Mapped variable in the robot base frame.

        Raises
        ------
        ValueError
            If ``CameraFrame`` is not set.
        """
        if self.CameraFrame is None:
            raise ValueError("CameraFrame is not set. Use SetCameraFrame first.")
        Tflange = self.GetPose(out="T", task_space="Robot", kinematics=kinematics, state=state, refresh=refresh) @ np.linalg.inv(self.TCP)
        Tcamera = Tflange @ self.CameraFrame
        return frame2world(x, Tcamera, typ=typ)

    def BaseToCamera(self, x: ArrayLike, typ: Optional[str] = None, kinematics: Optional[str] = None, state: Optional[str] = None, refresh: Optional[bool] = None) -> np.ndarray:
        """
        Map a variable from the robot base frame to the camera frame.

        The camera pose in the robot base frame is computed as
        ``Tflange @ CameraFrame``, where ``Tflange`` is the current robot flange
        pose in the robot base frame and ``CameraFrame`` is the camera pose
        relative to the flange.

        Parameters
        ----------
        x : ArrayLike
            Variable expressed in the robot base frame. Supported shapes follow
            :func:`robotblockset.transformations.world2frame`, including poses,
            homogeneous transforms, positions, rotations, twists, and wrenches.
        typ : str, optional
            Transformation type, by default None. Use ``"Twist"`` or
            ``"Wrench"`` for spatial velocity or force/torque variables.
        kinematics : str, optional
            Kinematics source for the current robot pose, by default None.
        state : str, optional
            Robot state for the current robot pose, by default None.
        refresh : bool, optional
            Whether to refresh robot state before reading the pose, by default None.

        Returns
        -------
        np.ndarray
            Mapped variable in the camera frame.

        Raises
        ------
        ValueError
            If ``CameraFrame`` is not set.
        """
        if self.CameraFrame is None:
            raise ValueError("CameraFrame is not set. Use SetCameraFrame first.")
        Tflange = self.GetPose(out="T", task_space="Robot", kinematics=kinematics, state=state, refresh=refresh) @ np.linalg.inv(self.TCP)
        Tcamera = Tflange @ self.CameraFrame
        return world2frame(x, Tcamera, typ=typ)

    # Kinematic utilities
    @abstractmethod
    def Kinmodel(self, *q: JointConfigurationType, tcp: Optional[TCPType] = None, out: str = "x") -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Abstract method to compute the forward kinematics of the robot.

        Parameters
        ----------
        *q : tuple
            Joint angles as input to the kinematic model.
        tcp : TCPType, optional
            Tool Center Point (TCP) pose or transformation, by default None.
        out : str, optional
            Output format for the result (pose, position, etc.), by default "x".

        Returns
        -------
        tuple
            Pose representation and JacobianType, depending on `out`.
        """
        pass

    def DKin(
        self,
        q: Optional[JointConfigurationType] = None,
        out: Optional[str] = None,
        task_space: Optional[str] = None,
    ) -> Union[Pose3DType, HomogeneousMatrixType]:
        """
        Compute the direct kinematics (position and orientation) for given joint angles.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions (nj,).
        out : str, optional
            Output format for the result, by default None (depends on robot settings).
        task_space : str, optional
            The task space frame to use for the result. Can be "World", "Object", or "Robot", by default None.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType
            The computed task position (7,) or transformation matrix (4, 4), depending on the output format.
        """
        if q is not None:
            _q = self.jointvar(q)
        else:
            _q = self._actual.q

        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace

        _x = self.Kinmodel(_q)[0]

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

    def DKinPath(self, path: JointPathType, out: Optional[str] = None) -> Union[Poses3DType, HomogeneousMatrixArrayType]:
        """
        Compute direct kinematics for a path of joint positions.

        Parameters
        ----------
        path : JointPathType
            Path in joint space - poses (n, nj).
        out : str, optional
            Output format for the result, by default None (depends on robot settings).

        Returns
        -------
        Poses3DType or HomogeneousMatrixArrayType
            Task positions at target pose (n, 7) or (4, 4, n), depending on the output format.
        """
        if out is None:
            out = self._default.TaskPoseForm

        _path = rbs_type(path)
        _n = np.shape(_path)[0]
        _xpath = np.nan * np.zeros((_n, 7))

        for i in range(_n):
            _x = self.DKin(_path[i, :])
            _xpath[i, :] = _x

        return map_pose(x=_xpath, out=out)

    def IKin(
        self,
        x: Union[Pose3DType, HomogeneousMatrixType],
        q0: Optional[JointConfigurationType] = None,
        max_iterations: int = 1000,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointVelocityType] = None,
        x_opt: Optional[Pose3DType] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
        save_path: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[JointConfigurationType, int], Tuple[JointPathType, int]]:
        """
        Compute inverse kinematics to find joint positions that achieve a target Cartesian pose.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target Cartesian pose (7,) or (4, 4).
        q0 : JointConfigurationType, optional
            Initial joint positions (nj,). Defaults to current joint positions if not provided.
        max_iterations : int, optional
            Maximum number of iterations for the inverse kinematics algorithm, by default 1000.
        pos_err : float, optional
            Position error tolerance, by default None (uses default value).
        ori_err : float, optional
            Orientation error tolerance, by default None (uses default value).
        task_space : str, optional
            Task space in which to perform the inverse kinematics, by default "Robot".
        task_DOF : ArrayLike, optional
            Degrees of freedom of the task, by default None.
        null_space_task : str, optional
            Type of null-space task, by default None.
        task_cont_space : str, optional
            The space in which task continuity is considered, by default "Robot".
        q_opt : JointConfigurationType, optional
            Optimal joint positions for null space tasks, by default None.
        v_ns : Velocity3DType, optional
            Null-space velocity for task-space velocity, by default None.
        qdot_ns : JointVelocityType, optional
            Null-space joint velocities, by default None.
        x_opt : Pose3DType, optional
            Optimal Cartesian pose for null-space tasks, by default None.
        Kp : float, optional
            Proportional gain for the controller, by default None.
        Kns : float, optional
            Gain for the null-space task, by default None.
        save_path : bool, optional
            Whether to save the joint positions for the path, by default False.

        Returns
        -------
        tuple
            Joint positions at target pose (nj,) or joint path when `save_path` is True, and a status code.
        """
        if q0 is None:
            q0 = self._actual.q
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
            u_path = q0.reshape((1, self.nj))
            ee_path = np.zeros((1, 6))

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

            uq = Jp @ ux
            _fac = np.max(np.abs(uq) / self.qdot_max)
            if _fac > 1:
                uq = uq / _fac
            uNS = NS @ qdn
            if any(abs(self.qdot_max - uq) < 1e-3) or any(abs(-self.qdot_max - uq) < 1e-3):
                uNS = uNS * 0
            else:
                _fac = max(1, np.max(uNS / (self.qdot_max - uq)), np.max(uNS / (-self.qdot_max - uq)))
            if _fac > 1:
                uNS = uNS / _fac
            u = uq + uNS

            u = uq + uNS
            qq = qq + u * self.tsamp

            if save_path:
                q_path = np.vstack((q_path, qq))
                u_path = np.vstack((u_path, u))
                ee_path = np.vstack((ee_path, ee))

            if self.CheckJointLimits(qq):
                self.WarningMessage(f"Joint limits reached: {qq}")
                qq = np.nan * qq
                if save_path:
                    return q_path, MotionResultCodes.JOINT_LIMITS.value
                else:
                    return qq, MotionResultCodes.JOINT_LIMITS.value

            _iterations += 1
            if _iterations > max_iterations:
                self.WarningMessage(f"No close solution found in {_iterations} iterations, err: {ee}")
                # qq = np.nan * qq
                if save_path:
                    return q_path, MotionResultCodes.NOT_FEASIBLE.value
                else:
                    return qq, MotionResultCodes.NOT_FEASIBLE.value

    def IKinPath(
        self,
        path: CartesianPathType,
        q0: JointConfigurationType,
        max_iterations: int = 100,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        task_space: Optional[str] = None,
        task_DOF: Optional[ArrayLike] = None,
        null_space_task: Optional[str] = None,
        task_cont_space: str = "Robot",
        q_opt: Optional[JointConfigurationType] = None,
        v_ns: Optional[Velocity3DType] = None,
        qdot_ns: Optional[JointVelocityType] = None,
        x_opt: Optional[Pose3DType] = None,
        Kp: Optional[float] = None,
        Kns: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[JointPathType, int]:
        """
        Compute inverse kinematics for a path of Cartesian poses.

        Parameters
        ----------
        path : CartesianPathType
            Path in Cartesian space - poses (n,7) or (n,4,4).
        q0 : JointConfigurationType
            Initial joint positions (nj,).
        max_iterations : int, optional
            Maximum number of iterations, by default 100.
        pos_err : float, optional
            Position error tolerance, by default None (uses default).
        ori_err : float, optional
            Orientation error tolerance, by default None (uses default).
        task_space : str, optional
            Task space in which to compute the inverse kinematics, by default "Robot".
        task_DOF : ArrayLike, optional
            Degrees of freedom for the task, by default None.
        null_space_task : str, optional
            Type of null-space task, by default None.
        task_cont_space : str, optional
            Task continuity space, by default "Robot".
        q_opt : JointConfigurationType, optional
            Optimal joint positions for null-space tasks, by default None.
        v_ns : Velocity3DType, optional
            Null-space velocity for task-space velocity, by default None.
        qdot_ns : JointVelocityType, optional
            Null-space joint velocities, by default None.
        x_opt : Pose3DType, optional
            Optimal Cartesian pose for null-space tasks, by default None.
        Kp : float, optional
            Proportional gain, by default None.
        Kns : float, optional
            Gain for null-space tasks, by default None.

        Returns
        -------
        tuple
            Joint positions at target pose for each point in the path (n, nj) and a status code.
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
                    self.Message(f"No IKin solution found for path point sample {i}", 0)
                    return _qpath, self._last_status
            except Exception:
                self.Message(f"No solution found for path point sample {i}", 0)
                self._last_status = MotionResultCodes.NOT_FEASIBLE.value
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
        if tcp is None:
            tcp = self.TCP
        J = self.Kinmodel(qq, tcp=tcp)[-1]

        # Transform Jacobian to the correct task space
        if check_option(task_space, "World"):
            J = self.WorldToBase(J)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported")
        return J

    def Manipulability(self, q: Optional[JointConfigurationType] = None, task_space: Optional[str] = "Robot", task_DOF: Optional[ArrayLike] = None) -> float:
        """
        Compute the manipulability measure of the robot at a given joint configuration.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions (nj,).
        task_space : str, optional
            The task space frame to use for the Jacobian, by default "Robot".
        task_DOF : ArrayLike, optional
            Task Degrees of Freedom (DOF) for the manipulability calculation, by default all 6 DOF are considered.

        Returns
        -------
        float
            Manipulability measure, which is the square root of the determinant of the Jacobian matrix.
        """
        if q is not None:
            qq = self.jointvar(q)
        else:
            qq = self._actual.q
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)

        # Get the Jacobian matrix for the given joint positions
        J = self.Jacobi(qq, task_space=task_space)

        # Extract the active DOF from the Jacobian
        Sind = np.where(np.asarray(task_DOF) > 0)[0]
        JJ = J[Sind, :]

        # Compute the manipulability as the square root of the determinant of JJ * JJ^T
        return np.sqrt(np.linalg.det(JJ @ JJ.T))

    def JointDistance(self, q: JointConfigurationType, state: str = "Actual") -> JointConfigurationType:
        """
        Calculate the distance between the current joint position and a target joint position.

        Parameters
        ----------
        q : JointConfigurationType
            Joint positions (nj,) to calculate the distance from.
        state : str, optional
            The state of the joint positions, by default "Actual". Other options are "Command" for commanded joint positions.

        Returns
        -------
        JointConfigurationType
            Distance (nj,) between the current joint position and the target joint position `q`.
        """
        q = self.jointvar(q)
        return q - self.GetJointPos(state=state)

    def TaskDistance(
        self,
        x: Pose3DType,
        out: str = "x",
        task_space: str = "World",
        state: str = "Actual",
        kinematics: str = "Calculated",
    ) -> Union[Pose3DType, Vector3DType, QuaternionType]:
        """
        Calculate the distance between the current pose and a target pose.

        Parameters
        ----------
        x : Pose3DType
            The target pose to compare to the current pose.
        out : str, optional
            The output form of the distance, by default "x". Possible values are "x", "p", and "Q".
        task_space : str, optional
            The task space to use for the pose transformation, by default "World".
        state : str, optional
            The state of the current pose, by default "Actual". Other options include "Command" for commanded poses.
        kinematics : str, optional
            The type of kinematics to use, by default "Calculated". Other options might include "Robot".

        Returns
        -------
        Pose3DType or Vector3DType or QuaternionType
            The distance between the current and target poses.
        """
        x = self.spatial(x)

        # Handle different shape formats for pose
        if x.shape == (4, 4):
            rx = t2x(x)
        elif isvector(x, dim=7):
            rx = x
        elif x.shape == (3, 3):
            rx = map_pose(R=x)
            out = "Q"
        elif isvector(x, dim=3):
            rx = map_pose(p=x)
            out = "p"
        elif isvector(x, dim=4):
            rx = map_pose(Q=x)
            out = "Q"
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

        # Compute the difference in pose (task space distance)
        dx = xerr(rx, self.GetPose(task_space=task_space, state=state, kinematics=kinematics))

        # Return the appropriate part of the pose distance based on the output form
        if out == "x":
            return dx
        elif out == "Q":
            return dx[3:]
        elif out == "p":
            return dx[:3]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    def CheckJointLimits(self, q: JointConfigurationType) -> bool:
        """
        Check if the joint positions `q` are within the joint range.

        Parameters
        ----------
        q : JointConfigurationType
            Joint position (nj,).

        Returns
        -------
        bool
            True if one or more joints are out of the limits, False otherwise.
        """
        return np.any(self.q_max - q < 0) or np.any(q - self.q_min < 0)

    def DistToJointLimits(self, q: Optional[JointConfigurationType] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the distance to the joint limits.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint position (nj,), by default None. If None, uses the current joint positions.

        Returns
        -------
        tuple
            - Minimal distance to joint limits (nj,).
            - Distance to lower joint limits (nj,).
            - Distance to upper joint limits (nj,).
        """
        if q is None:
            q = self._actual.q
        else:
            q = self.jointvar(q)
        dqUp = self.q_max - q
        dqLow = q - self.q_min
        dq = np.fmin(dqLow, dqUp)
        return dq, dqLow, dqUp

    # Gripper
    def SetGripper(self, gripper: Optional[Any] = None) -> None:
        """
        Attach or detach a gripper to the robot.

        Parameters
        ----------
        gripper : Any, optional
            The gripper to be attached to the robot, by default None. If None, the current gripper will be detached.
        """
        if self.Gripper is not None:
            self.Gripper.Detach()
        self.Gripper = gripper
        if gripper is not None:
            gripper.AttachTo(self)

    def GetGripper(self) -> list[Any]:
        """
        Get the current gripper and its name.

        Returns
        -------
        list[Any]
            A list where the first element is the current gripper (None if not set) and
            the second element is the gripper's name, or "None" if no gripper is set.
        """
        if self.Gripper is None:
            return [None, "None"]
        else:
            return [self.Gripper, self.Gripper.Name]

    # Camera
    def SetCamera(self, camera: Optional[Any] = None) -> None:
        """
        Attach or detach a camera to the robot.

        Parameters
        ----------
        camera : Any, optional
            The camera to be attached to the robot, by default None. If None, the current camera will be detached.
        """
        if self.Camera is not None:
            self.Camera.Detach()
        self.Camera = camera
        if camera is not None:
            self.Camera.AttachTo(self)

    def GetCamera(self) -> list[Any]:
        """
        Get the current camera and its name.

        Returns
        -------
        list[Any]
            A list where the first element is the current camera (None if not set) and
            the second element is the camera's name, or "None" if no camera is set.
        """
        if self.Camera is None:
            return [None, "None"]
        else:
            return [self.Camera, self.Camera.Name]

    def SetCameraFrame(self, x: TCPType) -> None:
        """
        Set the transformation matrix or pose of the camera frame in robot flange frame.

        Parameters
        ----------
        x : TCPType
            The pose or transformation of the camera frame. It can be a 4x4 homogeneous matrix, a 3x3 rotation matrix,
            or any compatible representation.
        """
        x = self.spatial(x)
        if x.shape == (4, 4):
            _T = x
        elif x.shape == (3, 3):
            _T = map_pose(R=x, out="T")
        elif isvector(x, dim=7):
            _T = map_pose(x=x, out="T")
        elif isvector(x, dim=3):
            _T = map_pose(p=x, out="T")
        elif isvector(x, dim=4):
            _T = map_pose(Q=x, out="T")
        else:
            raise ValueError(f"Camera frame shape {x.shape} not supported")
        self.CameraFrame = _T

    def GetCameraFrame(self, out: Optional[str] = None) -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]]:
        """Get the current camera frame in the specified output format.

        Parameters
        ----------
        out : str, optional
            The output form, by default None. The format can be "x", "p", "Q", or other valid forms.

        Returns
        -------
        Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]]
            The current camera frame in the specified output format.
        """
        if self.CameraFrame is None:
            return None
        if out is None:
            out = self._default.TaskPoseForm
        return map_pose(T=self.CameraFrame, out=out)

    # F/T sensor
    def SetFTFrame(self, FTFrame: Optional[TCPType] = None) -> None:
        """
        Set the F/T frame.

        Parameters
        ----------
        FTFrame : TCPType, optional
            The transformation matrix or pose of the F/T frame. Default is the identity matrix.

        Returns
        -------
        None
        """
        if FTFrame is not None:
            _T = spatial2t(FTFrame)
        else:
            _T = np.eye(4)

        self.FTFrame = _T

    def GetFTFrame(self) -> HomogeneousMatrixType:
        """
        Get the current FT frame

        Parameters
        ----------
        out : str, optional
            The output form, by default None. The format can be "x", "p", "Q", or other valid forms.

        Returns
        -------
        HomogeneousMatrixType
            The current FT frame.
        """
        return self.FTFrame

    def GetFTFramePose(self, out: Optional[str] = None, task_space: Optional[str] = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the current FT frame pose in the specified task space and output format.

        Parameters
        ----------
        out : str, optional
            The output form, by default None. The format can be "x", "p", "Q", or other valid forms.
        task_space : str, optional
            The task space for the pose transformation, by default "World". Other options can be
            "Object", "Robot", or "Tool".

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The current FT sensor pose in the specified task space and output form.
        """
        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace
        _T = self.GetPose(state="Actual", task_space="World", out="T") @ np.linalg.inv(self.TCP) @ self.FTFrame
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Robot"):
            _T = self.WorldToBase(_T)
        elif check_option(task_space, "Tool"):
            _T = np.linalg.inv(self.TCP) @ self.FTFrame
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(T=_T, out=out)

    def SetFTSensor(self, FTsensor: Optional[Any] = None) -> None:
        """
        Attach or detach a force/torque (FT) sensor to the robot.

        Parameters
        ----------
        FTsensor : Any, optional
            The FT sensor to be attached to the robot, by default None. If None, the current FT sensor will be detached.
        """
        if self.FTSensor is not None:
            self.FTSensor.Detach()
        self.FTSensor = FTsensor
        if FTsensor is not None:
            self.FTSensor.AttachTo(self)

    def GetFTSensor(self) -> list[Any]:
        """
        Get the current FT sensor and its name.

        Returns
        -------
        list[Any]
            A list where the first element is the current FT sensor (None if not set) and
            the second element is the FT sensor's name, or "None" if no FT sensor is set.
        """
        if self.FTSensor is None:
            return [None, "None"]
        else:
            return [self.FTSensor, self.FTSensor.Name]

    def SetFTSensorFrame(self, x: TCPType) -> None:
        """
        Set the transformation matrix or pose of the FT sensor frame in robot flange frame.

        Parameters
        ----------
        x : TCPType
            The pose or transformation of the FT sensor frame. It can be a 4x4 homogeneous matrix, a 3x3 rotation matrix,
            or any compatible representation.
        """
        x = self.spatial(x)
        if x.shape == (4, 4):
            _T = x
        elif x.shape == (3, 3):
            _T = map_pose(R=x, out="T")
        elif isvector(x, dim=7):
            _T = map_pose(x=x, out="T")
        elif isvector(x, dim=3):
            _T = map_pose(p=x, out="T")
        elif isvector(x, dim=4):
            _T = map_pose(Q=x, out="T")
        else:
            raise ValueError(f"FT sensor frame shape {x.shape} not supported")
        self.FTSensorFrame = _T

    def GetFTSensorFrame(self, out: Optional[str] = None) -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]]:
        """
        Get the current FT sensor frame in the specified output format.

        Parameters
        ----------
        out : str, optional
            The output form, by default None. The format can be "x", "p", "Q", or other valid forms.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType, optional
            The current FT sensor frame in the specified output form.
        """
        if self.FTSensorFrame is None:
            return None
        if out is None:
            out = self._default.TaskPoseForm
        return map_pose(T=self.FTSensorFrame, out=out)

    def GetFTSensorPose(self, out: Optional[str] = None, task_space: Optional[str] = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the current FT sensor pose in the specified task space and output format.

        Parameters
        ----------
        out : str, optional
            The output form, by default None. The format can be "x", "p", "Q", or other valid forms.
        task_space : str, optional
            The task space for the pose transformation, by default "World". Other options can be "Object" or "Robot".

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The current FT sensor pose in the specified task space and output form.
        """
        if self.FTSensorFrame is None:
            _frame = self.TCP
        else:
            _frame = self.FTSensorFrame
        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace
        _T = self.GetPose(state="Actual", out="T") @ np.linalg.inv(self.TCP) @ _frame
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Robot"):
            _T = self.WorldToBase(_T)
        elif check_option(task_space, "Tool"):
            _T = np.linalg.inv(self.TCP) @ _frame
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(T=_T, out=out)

    def SetFTSensorLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[Vector3DType] = None, inertia: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None) -> None:
        """
        Set the load properties of the FT sensor.

        Parameters
        ----------
        load : load_params, optional
            The load object, by default None.
        mass : float, optional
            The mass of the load, by default None.
        COM : Vector3DType, optional
            The center of mass of the load, by default None.
        inertia : np.ndarray, optional
            The inertia of the load, by default None.
        offset : np.ndarray, optional
            The offset of the load, by default None.
        """
        if self.FTSensor is not None:
            self.FTSensor.SetLoad(load=load, mass=mass, COM=COM, inertia=inertia, offset=offset)

    def GetFTSensorLoad(self) -> Optional[load_params]:
        """
        Get the load properties of the FT sensor.

        Returns
        -------
        Load or None
            The current load properties of the FT sensor, or None if no load is set.
        """
        if self.FTSensor is None:
            return None
        else:
            return self.FTSensor.GetLoad()

    # Contacts
    def GetContacts(self, **kwargs) -> Optional[np.ndarray]:
        """
        Return contact forces.

        THis method is intended to be overridden by derived classes that implement specific contact sensing capabilities. In the base class, it simply returns None.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        None
            What is returned depends on the implementation. In the base class, this method returns None, but in derived classes it may return contact information (force of something else)
        """
        return None

    # Load
    def SetLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[Vector3DType] = None, inertia: Optional[np.ndarray] = None) -> None:
        """
        Set the load properties of the robot.

        Parameters
        ----------
        load : load_params, optional
            The load object to be assigned, by default None.
        mass : float, optional
            The mass of the load, by default None.
        COM : Vector3DType, optional
            The center of mass of the load, by default None.
        inertia : np.ndarray, optional
            The inertia of the load, by default None.
        """
        if isinstance(load, load_params):
            self.Load = load
        else:
            if mass is not None:
                if mass < 0:
                    raise ValueError("Load mass cannot be negative")
                self.Load.mass = mass
            if COM is not None:
                if not isvector(COM, 3):
                    raise ValueError("Load COM must be a vector of shape (3,)")
                self.Load.COM = COM
            if inertia is not None:
                if not ismatrix(inertia, (3, 3)):
                    raise ValueError("Load inertia must be a 3x3 matrix")
                self.Load.inertia = inertia

    def GetLoad(self) -> Optional[load_params]:
        """
        Get the current load properties of the robot.

        Returns
        -------
        load_params or None
            The current load object, or None if no load is set.
        """
        return self.Load

    def SelectToolFromYAML(self, tool_name: Optional[str] = None, tool_yaml_file: Optional[Union[str, Path]] = "tools.yaml") -> None:
        """
        Apply tool TCP and load parameters to a robot instance.

        Parameters
        ----------
        tool_name : str, optional
            Tool name.
        tool_yaml_file : Union[str, Path], optional
            str | Path, optional
            Path to the YAML file containing tool definitions. Defauls to "tools.yaml"

        Returns
        -------
        None
        """
        tools, default_tool_name = load_tools_from_yaml(tool_yaml_file)

        if tool_name is None:
            tool_name = default_tool_name
        if tool_name not in tools.keys():
            self.Message(f"Tool not available!\n Available tools: {list(tools.keys())}")
            return

        self.Tool = tools[tool_name]
        self.SetTCP(map_pose(p=self.Tool.tcp_position, Q=self.Tool.tcp_orientation), frame=self.Tool.mounted_on)
        self.SetLoad(mass=self.Tool.load.mass, COM=self.Tool.load.COM, inertia=self.Tool.load.inertia)

    # TCP
    def SetTCP(self, tcp: Optional[TCPType] = None, frame: str = "Gripper") -> None:
        """
        Set the Tool Center Point (TCP) of the robot.

        Parameters
        ----------
        tcp : TCPType, optional
            The transformation matrix or pose of the TCP. Default is the identity matrix.
        frame : str, optional
            The frame to which the TCP is referenced. Can be "Gripper" or "Flange". Default is "Gripper".

        Returns
        -------
        None
        """
        if tcp is not None:
            _tcp = spatial2t(tcp)
        else:
            _tcp = np.eye(4)

        if check_option(frame, "Flange"):
            newTCP = _tcp
        elif check_option(frame, "Gripper"):
            newTCP = self.TCPGripper @ _tcp
        else:
            raise ValueError(f"Frame '{frame}' not supported")

        self.TCP = newTCP
        rx, rJ = self.Kinmodel(self._command.q)
        self._command.x = rx
        self._command.v = rJ @ self._command.qdot

    def GetTCP(self, out: str = "T") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the Tool Center Point (TCP) of the robot.

        Parameters
        ----------
        out : str, optional
            The output format of the TCP. Default is "T" (transformation matrix).

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The TCP in the specified output format.
        """
        return map_pose(T=self.TCP, out=out)

    def SetTCPGripper(self, tcp: Optional[TCPType] = None) -> None:
        """
        Set the TCP for the gripper.

        Parameters
        ----------
        tcp : TCPType, optional
            The transformation matrix or pose of the gripper TCP. Default is the identity matrix.

        Returns
        -------
        None
        """
        if tcp is not None:
            _tcp = spatial2t(tcp)
        else:
            _tcp = np.eye(4)

        self.TCPGripper = _tcp

    def GetTCPGripper(self, out: str = "T") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the Tool Center Point (TCP) of the gripper.

        Parameters
        ----------
        out : str, optional
            The output format of the gripper TCP. Default is "T" (transformation matrix).

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The gripper TCP in the specified output format.
        """
        return map_pose(T=self.TCPGripper, out=out)

    # Object
    def SetObject(self, x: Optional[Union[Pose3DType, HomogeneousMatrixType]] = None) -> None:
        """
        Set the object pose in the robot's coordinate system.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType], optional
            The pose of the object (7,) or (4, 4) (default is None, which sets to the actual object pose).

        Returns
        -------
        None
        """
        if x is None:
            _x = self.BaseToWorld(self._actual.x)
        else:
            _x = self.spatial(x)

        if _x.shape == (4, 4):
            _T = _x
        elif isvector(_x, dim=7):
            _T = x2t(_x)
        else:
            raise ValueError(f"Object pose shape {_x.shape} not supported")

        self.TObject = _T

    def GetObject(self, out: str = "T", task_space: Optional[str] = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the object pose in the specified task space.

        Parameters
        ----------
        out : str, optional
            The output format of the object pose. Default is "T" (transformation matrix).
        task_space : str, optional
            The task space for the pose transformation. Can be "World", "Object", or "Robot". Default is None.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The object pose in the specified output format.

        Raises
        ------
        ValueError
            If the task space is not recognized or the output format is not supported.
        """
        if task_space is None:
            task_space = self._default.TaskSpace

        _T = self.TObject

        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Robot"):
            _T = self.WorldToBase(_T)
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetObject")

        return map_pose(T=_T, out=out)

    # Base and platform
    def SetBasePose(self, x: Union[Pose3DType, HomogeneousMatrixType]) -> None:
        """
        Set the robot base pose.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            The pose of the base (7,) or (4, 4).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not recognized.
        """
        self.TBase = spatial2t(x)

    def GetBasePose(self, out: str = "T") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the robot base pose.

        Parameters
        ----------
        out : str, optional
            The output format of the base pose. Default is "T" (transformation matrix).

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The base pose in the specified output format.
        """
        _T = self.TBase
        return map_pose(T=_T, out=out)

    def SetBaseVel(self, v: Velocity3DType) -> None:
        """
        Set the robot base velocity.

        Parameters
        ----------
        v : Velocity3DType
            The velocity of the base (6,).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not recognized.
        """
        if isvector(v, dim=6):
            self.vBase = v
        else:
            raise ValueError(f"Base velocity shape {v.shape} not supported")

    def GetBaseVel(self) -> Velocity3DType:
        """
        Get the robot base velocity.

        Returns
        -------
        Velocity3DType
            The base velocity.
        """
        return self.vBase

    def SetBasePlatform(self, platform: Any, x: Optional[Union[Pose3DType, HomogeneousMatrixType]] = None) -> None:
        """
        Set the base platform and optionally set its pose.

        Parameters
        ----------
        platform : Any
            The platform to attach to the robot.
        x : Union[Pose3DType, HomogeneousMatrixType], optional
            The pose to set for the base platform (7,) or (4, 4).

        Returns
        -------
        None
        """
        if platform is None:
            self.Platform.Detach()
            self.Platform = None
        else:
            self.Platform = platform
            self.Platform.AttachTo(self)
            if x is not None:
                self.Platform.SetRobotBase(x)

    def GetBasePlatform(self) -> list[Any]:
        """
        Get the current base platform attached to the robot.

        Returns
        -------
        list[Any]
            A list containing the platform object and its name. If no platform is attached, returns `[None, "None"]`.
        """
        if self.Platform is None:
            return [None, "None"]
        else:
            return [self.Platform, self.Platform.Name]

    def UpdateRobotBase(self) -> HomogeneousMatrixType:
        """
        Update the robot base pose from the base platform, if available.

        Returns
        -------
        HomogeneousMatrixType
            The updated base pose.
        """
        if self.Platform is not None:
            self.TBase = self.Platform.GetRobotBasePose(out="T")
        return self.TBase

    # Movements
    def Start(self) -> bool:
        """
        Start the robot's motion by setting the control mode to 0.5 and resetting error states.

        Returns
        -------
        bool
            True if the robot is active and motion can start, False otherwise.
        """
        if self.HasError():
            self.WarningMessage("Robot in error mode. Can not start!")
            return False
        if self.isActive():
            self._abort = False
        else:
            self._abort = True
            self.WarningMessage("Not started due to inactive robot!")
            return False

        self._command.mode = CommandModeCodes.START.value
        self._last_control_time = self.simtime()
        self._motion_error = None
        self.Update()
        return True

    def Stop(self) -> None:
        """
        Stop the robot's motion by setting the control mode to 0 and resetting velocities, errors and threads.

        Returns
        -------
        None
        """
        self._command.mode = CommandModeCodes.STOP.value
        self._command.qdot = np.zeros(self.nj)
        self._command.v = np.zeros(6)
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
        if abort and self._command.mode <= CommandModeCodes.STOP.value:
            self.Message(f"Motiom can not be aborted. Current motion state: {CommandModeStr(self._command.mode)}", 2)
        else:
            self.Message(f"Abort: {abort}", 2)
            self._abort = abort
            self._command.mode = CommandModeCodes.ABORT.value
        self.Update()

    def StopMotion(self) -> None:
        """
        Stop the robot's motion by executing a trajectory stop command and setting the robot to stop mode.

        Returns
        -------
        None
        """
        if self._control_strategy in ["JointPositionTrajectory"]:
            self.GoTo_qtraj(self.q, np.zeros(self.nj), np.zeros(self.nj), self.tsamp)
        self.Stop()

    def WaitUntilStopped(self, eps: float = 0.001) -> None:
        """
        Wait until the robot's joint velocities are below a specified threshold.

        Parameters
        ----------
        eps : float, optional
            The velocity threshold to stop waiting. Default is 0.001.

        Returns
        -------
        None
        """
        self.GetState()
        while np.linalg.norm(self._actual.qdot) > eps:
            self.GetState()

    def WaitUntilDone(self, timeout: Optional[float] = None) -> int:
        """
        Blocks execution until the motion completion or an optional timeout occurs.

        Parameters
        ----------
        dt : float, optional
            Delay between consecutive checks in seconds. Default is ``0.01``.
        timeout : float, optional
            Maximum time to wait in seconds. If ``None``, the function waits indefinitely.
            Default is ``None``.

        Returns
        -------
        int
            A result/error code.

        Notes
        -----
        - This method can be reimplemented in robot subclass
        - This function is **blocking** and should not be called from a
          realtime control loop or high-frequency callback.
        - If motion finishes before ``timeout`` (if set), the function exits early.
        - If motion never completes and no timeout is specified, this method blocks indefinitely.
        """
        if self._command.mode > CommandModeCodes.STOP.value:
            start_time = perf_counter()
            while self._command.mode > CommandModeCodes.STOP.value:
                if timeout is not None and (perf_counter() - start_time) >= timeout:
                    return MotionResultCodes.NO_RESPONSE.value  # Timed out
                sleep(self.tsamp)
                self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value  # No motion

    def Wait(self, wait: float, dt: Optional[float] = None) -> None:
        """
        Wait for a specified duration by updating the robot state and pausing execution.

        Parameters
        ----------
        wait : float
            The duration to wait in seconds.
        dt : float, optional
            The time step between state updates. Default is `tsamp`.

        Returns
        -------
        None
        """
        if not self._semaphore.acquire(blocking=False):
            self.WarningMessage("Wait not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self.Message(f"Wait for {wait:.3f}s", 2)
        if dt is None:
            dt = self.tsamp
        tx = self.simtime()
        imode = self._command.mode
        self._command.mode = CommandModeCodes.WAIT.value
        while self.simtime() - tx < wait:
            self._sleep(dt)
            self.GetState()
            self.Update()
        self._command.mode = imode
        self._semaphore.release()

    def Restart(self) -> None:
        """
        Stop and then start the robot to restart its motion.

        Returns
        -------
        None
        """
        self.Stop()
        self.Start()

    def SetMotionCheckCallback(self, fun: Callable[..., Any]) -> None:
        """
        Set a callback function to be called for motion checks.

        Parameters
        ----------
        fun : Callable[..., Any]
            The callback function to check motion status.

        Returns
        -------
        None
        """
        self._motion_check_callback = fun

    def EnableMotionCheck(self, check: bool = True) -> None:
        """
        Enable or disable the motion check callback.

        Parameters
        ----------
        check : bool, optional
            Whether to enable motion check. Default is True.

        Returns
        -------
        None
        """
        self._do_motion_check = check

    def DisableMotionCheck(self) -> None:
        """
        Disable the motion check callback.

        Returns
        -------
        None
        """
        self._do_motion_check = False

    # Utilities
    def SetCaptureCallback(self, fun: Callable[..., Any]) -> None:
        """
        Set the callback function for capture events.

        Parameters
        ----------
        fun : Callable[..., Any]
            The callback function to be called when a capture event occurs.

        Returns
        -------
        None
        """
        self._capture_callback = fun

    def StartCapture(self) -> None:
        """
        Start the capture process, ensuring that the update is enabled.

        Returns
        -------
        None
        """
        if not self._do_update:
            self.WarningMessage("Update is not enabled")
        self._do_capture = True
        self.Message("Capture started", 2)
        self.Update()

    def StopCapture(self) -> None:
        """
        Stop the capture process.

        Returns
        -------
        None
        """
        self.Message("Capture stopped", 2)
        self._do_capture = False

    def SetUserData(self, data: Optional[Any]) -> None:
        """
        Set the user data to be used for commands.

        Parameters
        ----------
        data : Any, optional
            The user data to be set for commands.

        Returns
        -------
        None
        """
        self._command.data = data
        self.Message(f"User data: {data}", 2)
        self.Update()

    def GetUserdata(self) -> Optional[Any]:
        """
        Get the user data associated with the command.

        Returns
        -------
        Any, optional
            The current user data set for the command.
        """
        return self._command.data

    @staticmethod
    def MotionResultStr(code: int) -> str:
        """
        Convert motion result status code to a human-readable explanation.

        Parameters
        ----------
        code : int
            Motion result code returned by the controller.

        Returns
        -------
        str
            Explanation of the motion result (no code name included).
        """

        STATUS_MAP = {
            0: "Motion completed successfully.",
            1: "Motion failed due to an unspecified error.",
            2: "Motion was aborted before completion.",
            3: "Motion exceeded joint limits or was infeasible given joint constraints.",
            4: "Motion stopped because the robot is already close enough to the target.",
            5: "Another motion thread is active; motion could not start.",
            6: "The selected control strategy is incorrect or incompatible.",
            7: "Motion is not feasible due to kinematic or planning constraints.",
            8: "No robot is attached to the controller; motion cannot be executed.",
            9: "RTDE communication failure (likely data exchange issue).",
            10: "Timeout (no response in expected time).",
        }

        return STATUS_MAP.get(code, f"Unknown motion result code: {code}.")


def isrobot(obj: object) -> bool:
    """
    Check if the given object is an instance of the `robot` class.

    Parameters
    ----------
    obj : object
        The object to be checked.

    Returns
    -------
    bool
        Returns `True` if the object is an instance of the `robot` class, otherwise `False`.
    """
    return isinstance(obj, robot)


def manipulability(J: JacobianType) -> float:
    """
    Calculate the manipulability of a robot based on its Jacobian matrix.

    The manipulability is calculated as the square root of the determinant of the
    product of the Jacobian matrix and its transpose.

    Parameters
    ----------
    J : JacobianType
        The Jacobian matrix of the robot. It is expected to have shape (m, n),
        where m is the number of task space dimensions and n is the number of
        degrees of freedom.

    Returns
    -------
    float
        The manipulability measure, which is a scalar value representing the
        robot's ability to move in all directions within the task space.
    """
    return np.sqrt(np.linalg.det(J @ J.T))


def dkin(
    q: JointConfigurationType,
    kinmodel: Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]],
    tcp: TCPType = np.eye(4),
    out: str = "x",
) -> Union[Pose3DType, HomogeneousMatrixType]:
    """
    Direct kinematics.

    This function computes the task position for a given joint position using the provided kinematics model.

    Parameters
    ----------
    q : JointConfigurationType
        Joint positions (nj,). An array of joint values.
    kinmodel : Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]]
        The direct kinematics function. This function takes joint positions and TCP as input and returns the task position.
    tcp : TCPType, optional
        Tool center point pose (7,) or (4, 4). Default is the identity matrix `np.eye(4)`.
    out : str, optional
        Specifies the output form. Default is "x" (task positions). Other options can be defined depending on the kinematics model.

    Returns
    -------
    Pose3DType or HomogeneousMatrixType
        Task pose in the representation requested by `out`.

    """
    _q = rbs_type(q)
    return kinmodel(_q, tcp=tcp, out=out)[0]


def dkinpath(
    path: JointPathType,
    kinmodel: Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]],
    tcp: TCPType = np.eye(4),
    out: str = "x",
) -> Union[Poses3DType, HomogeneousMatrixArrayType]:
    """
    Direct kinematics for a path.

    This function computes the task positions for a series of joint positions (path) using the provided kinematics model.

    Parameters
    ----------
    path : JointPathType
        Path in joint space - poses (n, nj). A 2D array of joint positions for multiple configurations.
    kinmodel : Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]]
        The direct kinematics function. This function takes joint positions and TCP as input and returns the task position.
    tcp : TCPType, optional
        Tool center point pose (7,) or (4, 4). Default is the identity matrix `np.eye(4)`.
    out : str, optional
        Specifies the output form. Default is "x" (task positions). Other options can be defined depending on the kinematics model.

    Returns
    -------
    Poses3DType or HomogeneousMatrixArrayType
        Task poses corresponding to each joint configuration in the path.
    """
    _path = rbs_type(path)
    _n = np.shape(_path)[0]
    _xpath = np.nan * np.zeros((_n, 7))
    for i in range(_n):
        _x = dkin(_path[i, :], kinmodel, tcp=tcp, out="x")
        _xpath[i, :] = _x
    return _xpath


def ikin(
    x: Pose3DType,
    q0: JointConfigurationType,
    kinmodel: Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]],
    tcp: TCPType = np.eye(4),
    tsamp: float = 0.01,
    max_iterations: int = 1000,
    pos_err: float = 0.0001,
    ori_err: float = 0.001,
    task_DOF: ArrayLike = np.array([1, 1, 1, 1, 1, 1]),
    null_space_task: str = "None",
    q_min: Optional[JointConfigurationType] = None,
    q_max: Optional[JointConfigurationType] = None,
    q_opt: Optional[JointConfigurationType] = None,
    v_ns: Optional[JointConfigurationType] = None,
    qdot_ns: Optional[JointConfigurationType] = None,
    x_opt: Optional[Pose3DType] = None,
    Kp: float = 10,
    Kns: float = 1,
    save_path: bool = False,
) -> Union[Tuple[JointConfigurationType, int], Tuple[JointPathType, int]]:
    """
    Inverse kinematics.

    This function computes the joint positions for a given target Cartesian pose using inverse kinematics
    based on kinematic controller.

    Parameters
    ----------
    x : Pose3DType
        Target Cartesian pose (7,).
    q0 : JointConfigurationType
        Initial joint positions (nj,).
    kinmodel : Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]]
        Direct kinematics function that computes the task pose given joint positions.
    tcp : TCPType, optional
        Tool center point pose. Default is the identity matrix `np.eye(4)`.
    tsamp : float, optional
        Sampling time. Default is 0.01.
    max_iterations : int, optional
        Maximum number of iterations. Default is 1000.
    pos_err : float, optional
        Position error tolerance. Default is 0.0001.
    ori_err : float, optional
        Orientation error tolerance. Default is 0.001.
    task_DOF : ArrayLike, optional
        Degrees of freedom for the task (6,). Default is all 6 DOFs.
    null_space_task : str, optional
        Type of null-space task. Default is "None". Other options include "Manipulability", "JointLimits", etc.
    q_min : JointConfigurationType, optional
        Joint limits for the minimum joint positions (nj,). Default is None.
    q_max : JointConfigurationType, optional
        Joint limits for the maximum joint positions (nj,). Default is None.
    q_opt : JointConfigurationType, optional
        Optimal joint configuration for certain null-space tasks. Default is None.
    v_ns : JointConfigurationType, optional
        Null-space task velocity vector (6,). Default is None.
    qdot_ns : JointConfigurationType, optional
        Joint velocity for the null-space task (nj,). Default is None.
    x_opt : Pose3DType, optional
        Optimal task pose for pose optimization tasks. Default is None.
    Kp : float, optional
        Proportional gain for task-space control. Default is 10.
    Kns : float, optional
        Proportional gain for null-space task. Default is 1.
    save_path : bool, optional
        Flag to save the joint path during the iterations. Default is False.

    Returns
    -------
    tuple[JointConfigurationType, int] or tuple[JointPathType, int]
        Joint positions and status code, or the full joint path and status code
        when `save_path=True`.
    """
    rx = x2x(x)
    q0 = rbs_type(q0)
    nj = q0.shape[0]

    # Initialize null-space task values if not provided
    if v_ns is None:
        v_ns = np.zeros(6)
    if qdot_ns is None:
        qdot_ns = np.zeros(nj)

    # Ensure joint limits are provided
    if q_min is None or q_max is None:
        raise ValueError("Joint limits q_min and q_max have to be defined")

    task_DOF = vector(task_DOF, dim=6)

    # Handle different null-space tasks
    if check_option(null_space_task, "None"):
        pass
    elif check_option(null_space_task, "Manipulability"):
        pass
    elif check_option(null_space_task, "JointLimits"):
        q_opt = (q_max + q_min) / 2
    elif check_option(null_space_task, "ConfOptimization"):
        if q_opt is None:
            raise ValueError("Optimal joint configuration q_opt has to be defined")
        q_opt = vector(q_opt, dim=nj)
    elif check_option(null_space_task, "PoseOptimization"):
        if x_opt is None:
            raise ValueError("Optimal task pose x_opt has to be defined")
        x_opt = x2x(x_opt)
    elif check_option(null_space_task, "TaskVelocity"):
        rv = vector(v_ns, dim=6)
    elif check_option(null_space_task, "JointVelocity"):
        rqdn = vector(qdot_ns, dim=nj)
    else:
        raise ValueError(f"Null-space task '{null_space_task}' not supported")

    # Initialize error tolerances
    _max_err = np.ones(6)
    _max_err[:3] = pos_err
    _max_err[3:] = ori_err

    rp = copy.deepcopy(rx[:3])
    rR = copy.deepcopy(q2r(rx[3:]))
    _iterations = 0
    qq = q0

    # Optionally save joint path
    if save_path:
        q_path = q0.reshape((1, nj))

    Sind = np.where(np.asarray(task_DOF) > 0)[0]
    uNS = np.zeros(nj)

    while True:
        p, R, J = kinmodel(qq, tcp=tcp, out="pR")
        ep = rp - p
        eR = qerr(r2q(rR @ R.T))
        ee = np.hstack((ep, eR))

        # Check for convergence
        if np.all(np.abs(ee) < _max_err):
            if save_path:
                return q_path, 0
            else:
                return qq, 0

        ux = Kp * ee
        ux = ux[Sind]
        JJ = J[Sind, :]
        Jp = np.linalg.pinv(JJ)
        NS = np.eye(nj) - Jp @ JJ

        # Compute null-space velocity based on selected task
        if check_option(null_space_task, "None"):
            qdn = np.zeros(nj)
        elif check_option(null_space_task, "Manipulability"):
            fun = lambda q: manipulability(kinmodel(q, tcp=tcp)[1])
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

        # Update joint velocities and positions
        uNS = NS @ qdn
        u = Jp @ ux + uNS
        qq = qq + u * tsamp

        if save_path:
            q_path = np.vstack((q_path, qq))

        # Check for joint limits
        if np.any(q_max - qq < 0) or np.any(qq - q_min < 0):
            print(f"Joint limits reached: {qq}")
            qq = np.nan * qq
            if save_path:
                return q_path, MotionResultCodes.JOINT_LIMITS.value
            else:
                return qq, MotionResultCodes.JOINT_LIMITS.value

        _iterations += 1
        if _iterations > max_iterations:
            print(f"No solution found in {_iterations} iterations")
            qq = np.nan * qq
            if save_path:
                return q_path, MotionResultCodes.NOT_FEASIBLE.value
            else:
                return qq, MotionResultCodes.NOT_FEASIBLE.value


def ikinpath(
    path: Poses3DType,
    q0: JointConfigurationType,
    kinmodel: Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]],
    tcp: TCPType = np.eye(4),
    tsamp: float = 0.01,
    max_iterations: int = 1000,
    pos_err: float = 0.001,
    ori_err: float = 0.001,
    task_DOF: ArrayLike = np.array([1, 1, 1, 1, 1, 1]),
    null_space_task: str = "None",
    q_min: Optional[JointConfigurationType] = None,
    q_max: Optional[JointConfigurationType] = None,
    q_opt: Optional[JointConfigurationType] = None,
    v_ns: Optional[JointConfigurationType] = None,
    qdot_ns: Optional[JointConfigurationType] = None,
    x_opt: Optional[Pose3DType] = None,
    Kp: float = 10,
    Kns: float = 1,
) -> Tuple[JointPathType, int]:
    """
    Inverse kinematics for a path.

    This function computes joint positions for a given path of target Cartesian poses using inverse kinematics.

    Parameters
    ----------
    path : Poses3DType
        Path in Cartesian space as pose array.
    q0 : JointConfigurationType
        Initial joint positions (nj,).
    kinmodel : Callable[..., Tuple[Union[Pose3DType, HomogeneousMatrixType], JacobianType]]
        Direct kinematics function that computes the task pose given joint positions.
    tcp : TCPType, optional
        Tool center point pose. Default is the identity matrix `np.eye(4)`.
    tsamp : float, optional
        Sampling time. Default is 0.01.
    max_iterations : int, optional
        Maximum number of iterations. Default is 1000.
    pos_err : float, optional
        Position error tolerance. Default is 0.001.
    ori_err : float, optional
        Orientation error tolerance. Default is 0.001.
    task_DOF : ArrayLike, optional
        Degrees of freedom for the task (6,). Default is all 6 DOFs.
    null_space_task : str, optional
        Type of null-space task. Default is "None". Other options include "Manipulability", "JointLimits", etc.
    q_min : JointConfigurationType, optional
        Joint limits for the minimum joint positions (nj,). Default is None.
    q_max : JointConfigurationType, optional
        Joint limits for the maximum joint positions (nj,). Default is None.
    q_opt : JointConfigurationType, optional
        Optimal joint configuration for certain null-space tasks. Default is None.
    v_ns : JointConfigurationType, optional
        Null-space task velocity vector (6,). Default is None.
    qdot_ns : JointConfigurationType, optional
        Joint velocity for the null-space task (nj,). Default is None.
    x_opt : Pose3DType, optional
        Optimal task pose for pose optimization tasks. Default is None.
    Kp : float, optional
        Proportional gain for task-space control. Default is 10.
    Kns : float, optional
        Proportional gain for null-space task. Default is 1.

    Returns
    -------
    tuple[JointPathType, int]
        Joint positions for each target pose together with the status code.
    """
    # Handle different input shapes for path
    if path.ndim == 3:
        _path = uniqueCartesianPath(t2x(path))
    elif ismatrix(path, shape=7):
        _path = uniqueCartesianPath(path)
    else:
        raise ValueError(f"Path shape {path.shape} not supported")

    _q = rbs_type(q0)
    nj = _q.shape[0]
    _n = np.shape(_path)[0]
    _qpath = np.nan * np.zeros((_n, nj))
    _tmperr = 0

    # Loop over path to compute joint positions
    for i in range(_n):
        _x = _path[i, :]
        try:
            # Compute inverse kinematics for each path point
            _q, _tmperr = ikin(
                _x,
                _q,
                kinmodel,
                tcp=tcp,
                q_min=q_min,
                q_max=q_max,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
            )
            # If error occurs, return the current path and error code
            if _tmperr != 0:
                return _qpath, _tmperr
            _qpath[i, :] = _q
        except Exception:
            print(f"No solution found for path point sample {i}")
            _tmperr = MotionResultCodes.NOT_FEASIBLE.value
            break

    return _qpath, _tmperr
