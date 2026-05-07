"""Platform base classes and utilities.

This module defines the common platform abstraction used in robotblockset for
mobile bases and related motion-control components. It provides state and
command containers, default motion parameters, motion result codes, and the
main `platform` base class with support for coordinate transforms, motion
commands, attached robots, and asynchronous execution helpers.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Any, Tuple, Union
from time import perf_counter, sleep
from threading import Semaphore, Thread
import platform as platform_os
import copy
from enum import Enum

from robotblockset.tools import rbs_object, _eps, rbs_type, check_option, vector, isvector, isscalar, wrap_to_pi
from robotblockset.transformations import map_pose, q2rpy, x2t, t2x, r2q, v2s, xerr, terr, qerr, rot_z, q2r
from robotblockset.trajectories import jtraj
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, JointVelocityType, JointTorqueType, Pose3DType, QuaternionType, RotationMatrixType, HomogeneousMatrixType, Velocity3DType, Vector2DType, Vector3DType, WrenchType, JacobianType
from robotblockset.robots import robot, CommandModeCodes


flag = True


def _dummy():
    global flag
    flag = False


class MotionResultCodes(Enum):
    """
    Result codes returned by platform motion commands.

    Attributes
    ----------
    MOTION_SUCCESS : int
        ``0``. Motion completed successfully.
    MOTION_FAILURE : int
        ``1``. Motion failed.
    MOTION_ABORTED : int
        ``2``. Motion was aborted.
    JOINT_LIMITS : int
        ``3``. A joint or configuration limit was reached.
    CLOSE_TO_TARGET : int
        ``4``. The platform is already close to the requested target.
    ACTIVE_THREADS : int
        ``5``. Active worker threads prevent execution of the command.
    WRONG_STRATEGY : int
        ``6``. The selected motion strategy is not valid for the requested command.
    NOT_FEASIBLE : int
        ``7``. The requested motion is not feasible.
    NO_ROBOT_ATTACHED : int
        ``8``. No robot is attached to the platform.
    RTDE_ERROR : int
        ``9``. A communication error occurred in the backend interface.
    NO_RESPONSE : int
        ``10``. No response was received from the platform or controller.
    NOT_READY : int
        ``11``. The platform is not ready to execute motion.
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


class _actual:
    """
    Represents the actual state of the platform, including planar pose, velocity, and wrench data.

    Attributes
    ----------
    q : Optional[JointConfigurationType]
        Platform configuration variables.
    qdot : Optional[JointVelocityType]
        Platform configuration velocities.
    trq : Optional[JointTorqueType]
        Platform generalized torques or effort values.
    x : Optional[Pose3DType]
        Actual platform pose.
    v : Optional[Velocity3DType]
        Actual platform spatial velocity.
    FT : Optional[WrenchType]
        Actual force/torque data associated with the platform.
    """

    def __init__(self):
        """
        Initialize the container for the measured platform state.

        Returns
        -------
        None
            This constructor initializes the actual-state container in place.
        """
        self.q: Optional[JointConfigurationType] = None
        self.qdot: Optional[JointVelocityType] = None
        self.trq: Optional[JointTorqueType] = None
        self.x: Optional[Pose3DType] = None
        self.v: Optional[Velocity3DType] = None
        self.FT: Optional[WrenchType] = None


class _command:
    """
    Represents the commanded state of the platform, including desired motion, wrench, and control inputs.

    Attributes
    ----------
    q : Optional[JointConfigurationType]
        Commanded platform configuration variables.
    qdot : Optional[JointVelocityType]
        Commanded platform configuration velocities.
    trq : Optional[JointTorqueType]
        Commanded platform generalized torques or effort values.
    x : Optional[Pose3DType]
        Commanded platform pose.
    v : Optional[Velocity3DType]
        Commanded platform spatial velocity.
    FT : Optional[WrenchType]
        Commanded force/torque data.
    u : Optional[Velocity3DType]
        Control input used by platform controllers.
    ux : Optional[Velocity3DType]
        Additional control input in task or Cartesian space.
    data : Optional[Any]
        User-defined command data associated with the platform.
    mode : Optional[float]
        Command mode identifier.
    """

    def __init__(self):
        """
        Initialize the container for commanded platform values.

        Returns
        -------
        None
            This constructor initializes the command-state container in place.
        """
        self.q: Optional[JointConfigurationType] = None  # Commanded joint positions
        self.qdot: Optional[JointVelocityType] = None  # Commanded joint velocities
        self.trq: Optional[JointTorqueType] = None  # Commanded joint torques
        self.x: Optional[Pose3DType] = None  # Commanded Cartesian pose
        self.v: Optional[Velocity3DType] = None  # Commanded Cartesian velocities
        self.FT: Optional[WrenchType] = None  # Commanded force/torque data
        self.u: Optional[Velocity3DType] = None  # Control input
        self.ux: Optional[Velocity3DType] = None  # Control input for Cartesian space
        self.data: Optional[Any] = None  # User-defined data
        self.mode: Optional[float] = None  # Control mode


class _default:
    """
    Class to store default parameters for robot behavior.

    Attributes
    ----------
    State : str
        Default state for robot (Actual/Commanded)
    TaskSpace : str
        Default task space
    TaskPoseForm : str
        Default pose form
    TaskOriForm : str
        Default orientation form
    TaskVelForm : str
        Default velocity form
    TaskFTForm : str
        Default force/torque form
    Refresh : bool
        Whether to refresh the robot's state or not.
    Traj : str
        Default trajectory type
    VelFac : float
        Default velocity scaling factor
    PosErr : float
        Default position error tolerance
    OriErr : float
        Default orientation error tolerance
    LaserAngleRange : float
        Default laser angle range
    CheckObstacles : bool
        Flag to check obstacles
    ObstacleMaxDist : float
        Maximum distance for obstacles
    ObstacleMinDist : float
        Minimum distance for obstacles
    ObstaclesForPlatform : bool
        Flag to include obstacles for platform
    MinVel: float
        Minimal velocity (used in stop platform when close to obstacle)
    ApproachDist : float
        Approach distance
    Krot : float
        Rotation motion controller constant
    Kdist : float
        Linear motion controller constant
    Kdir : float
        Direction motion controller constant
    Wait : float
        Default wait time
    UpdateTime : float
        Update time interval
    """

    def __init__(self):
        """
        Initialize the default platform configuration parameters.

        Returns
        -------
        None
            This constructor initializes the default-parameter container in place.
        """
        self.State: str = "Actual"  # Default state for robot (Actual/Commanded)
        self.TaskSpace: str = "World"  # Default task space
        self.TaskPoseForm: str = "2d"  # Default pose form
        self.TaskOriForm: str = "Theta"  # Default orientation form
        self.TaskVelForm: str = "Twist"  # Default velocity form
        self.TaskFTForm: str = "Wrench"  # Default force/torque form
        self.Refresh: bool = True  # Whether to refresh the robot's state or not
        self.Traj: str = "poly"  # Default trajectory type
        self.VelFac: float = 0.25  # Default velocity scaling factor
        self.PosErr: float = 0.01  # Default position error tolerance
        self.OriErr: float = 0.01  # Default orientation error tolerance
        self.LaserAngleRange: float = np.pi / 4  # Default laser angle range
        self.CheckObstacles: bool = True  # Flag to check obstacles
        self.ObstacleMaxDist: float = 1  # Maximum distance for obstacles
        self.ObstacleMinDist: float = 0.4  # Minimum distance for obstacles
        self.ObstaclesForPlatform: bool = True  # Flag to include obstacles for platform
        self.MinVel: float = 0.001  # Minimum velocity
        self.ApproachDist: float = 0.5  # Approach distance
        self.Krot: float = 10  # Rotation motion controller constant
        self.Kdist: float = 4  # Linear motion controller constant
        self.Kdir: float = 20  # Direction motion controller constant
        self.Wait: float = 0.1  # Default wait time
        self.UpdateTime: float = 1.0  # Update time interval


class platform(rbs_object):
    """
    Represents a mobile platform base class with state handling, frame transforms, and motion-control utilities.

    Attributes
    ----------
    Name : str
        Name of the platform.
    tsamp : float
        Sampling rate for the platform.
    TRobotBase : HomogeneousMatrixType
        Transformation matrix of the attached robot base relative to the platform.
    TObject : HomogeneousMatrixType
        Transformation matrix of the tracked or manipulated object.
    Robot : Optional[robot]
        Robot attached to the platform, if any.
    User : Optional[Any]
        User-defined data or object associated with the platform.
    Tag : Optional[str]
        Tag associated with the platform.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the platform with default values and optional configuration arguments.

        Parameters
        ----------
        **kwargs : Any
            Optional arguments for custom configuration or parameters.
        """
        rbs_object.__init__(self)
        self.Name: str = "Platform"  # Platform name
        self.tsamp: float = 0.01  # Sampling rate
        self.TRobotBase: HomogeneousMatrixType = np.eye(4)  # Robot base transformation matrix
        self.TObject: HomogeneousMatrixType = np.eye(4)  # Object transformation matrix
        self.Robot: Optional[robot] = None  # Robot attached to platform
        self.User: Optional[Any] = None  # User data or object
        self.Tag: Optional[str] = None  # Tag for platform

        self._t0: float = 0  # Initial time
        self._tt: float = 0  # Actual robot time
        self._tt0: float = 0  # Initial robot time
        self._robottime: float = 0  # Time from simulator
        self._last_update: float = -100  # Last update time
        self._last_control_time: float = -100  # Last control time
        self._command: _command = _command()  # Commanded values
        self._actual: _actual = _actual()  # Measured values
        self._default: _default = _default()  # Default options
        self._do_update: bool = True  # Flag to enable state update
        self._do_capture: bool = False  # Flag to enable callback capture
        self._capture_callback: Optional[Any] = None  # Callback function in Update
        self._do_motion_check: bool = False  # Flag to enable motion checks
        self._motion_check_callback: Optional[Any] = None  # Callback during motion
        self._last_status: int = 0  # Last motion command status
        self._platform_autonomous_motion: Optional[Any] = None  # Autonomous motion callback
        self._abort_autonomous_motion: bool = False  # Flag to abort autonomous motion
        self._control_strategy: str = "CartesianVelocity"  # Control strategy
        self._semaphore: Semaphore = Semaphore(1)  # Semaphore for asynchronous motion
        self._threads_active: bool = platform_os.system() == "Linux"  # Flag for threads on Linux
        self._abort_motion: bool = False  # Flag to abort current motion
        self._connected: bool = False  # Connection status
        self._verbose: int = 1  # Verbosity level

    def reset_threads(self) -> None:
        """
        Resets the internal semaphore to its initial state.

        This method reinitializes the semaphore with a value of 1,
        effectively allowing one thread to acquire it.

        Returns
        -------
        None
        """
        self._semaphore._value = 1

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
        supports threading, it sleeps for half of the remaining wait time to avoid blocking the control loop.

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
        Enable or disable worker threads for platform control.

        This method sets the `_threads_active` attribute, which controls whether the platform
        control system should use threads for asynchronous operations.

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
        Set the sampling time for the platform control system.

        This method updates the `tsamp` attribute and synchronizes the default wait interval.

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
        Reset the platform timing reference to the current simulation time.

        This method updates the internal time attributes `_t0` and `_tt0` to the current
        simulation time and current platform time, respectively.

        Returns
        -------
        None
            This method does not return any value.
        """
        self.GetState()
        self._t0 = self.simtime()
        self._tt0 = copy.deepcopy(self._tt)
        self.Update()

    def isReady(self) -> bool:
        """
        Check if the platform is ready for operation.

        This method checks the `_connected` attribute to determine if the platform is connected
        and operational.

        Returns
        -------
        bool
            `True` if the robot is connected and ready for operations, otherwise `False`.
        """
        return self._connected

    def isActive(self) -> bool:
        """
        Check if the platform target is active.

        This method always returns `True`, indicating that the platform target is in an active state.
        It can be overridden in subclasses for more complex behavior.

        Returns
        -------
        bool
            Always returns `True`, indicating the platform target is active.
        """
        return True

    def inMotion(self) -> bool:
        """
        Check if the platform is in motion.

        Returns
        -------
        bool
            `True` indicating the platform is excetuting motion command.
        """
        return self._command.mode > CommandModeCodes.STOP.value

    def Check(self, silent: bool = False) -> list[str]:
        """
        Check the status of the platform.

        Parameters
        ----------
        silent : bool, optional
            If `True`, suppress status messages while checking the platform state.

        Returns
        -------
        list[str]
            A list containing non-active status entries and descriptions.
        """
        return []

    @property
    def Time(self) -> float:
        """
        Get the elapsed wall time since the platform was initialized.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        return self.simtime() - self._t0

    @property
    def t(self) -> float:
        """
        Get the elapsed platform time.

        Returns
        -------
        float
            Time difference in seconds.
        """
        return self._tt - self._tt0

    @property
    def command(self) -> _command:
        """
        Get the commanded state of the platform.

        Returns
        -------
        _command
            A copy of the commanded state.
        """
        return copy.deepcopy(self._command)

    @property
    def actual(self) -> _actual:
        """
        Get the actual state of the platform.

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
    def qdot(self) -> JointVelocityType:
        """
        Get the current joint velocities.

        Returns
        -------
        np.ndarray
            Joint velocities (nj,).
        """
        return copy.deepcopy(self._actual.qdot)

    @property
    def trq(self) -> JointTorqueType:
        """
        Get the current joint torques.

        Returns
        -------
        np.ndarray
            Joint torques (nj,).
        """
        return copy.deepcopy(self._actual.trq)

    @property
    def x(self) -> Pose3DType:
        """
        Get the current platform pose.

        Returns
        -------
        np.ndarray
            Platform pose (7,).
        """
        return copy.deepcopy(self._actual.x)

    @property
    def p(self) -> Vector2DType:
        """
        Get the current planar platform position.

        Returns
        -------
        np.ndarray
            Platform position (2,).
        """
        return copy.deepcopy(self._actual.x[:2])

    @property
    def Q(self) -> QuaternionType:
        """
        Get the current platform orientation as a quaternion.

        Returns
        -------
        np.ndarray
            Platform quaternion (4,).
        """
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="Q"))

    @property
    def R(self) -> RotationMatrixType:
        """
        Get the current platform orientation as a rotation matrix.

        Returns
        -------
        np.ndarray
            Platform rotation matrix (3, 3).
        """
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="R"))

    @property
    def T(self) -> HomogeneousMatrixType:
        """
        Get the current platform pose as a homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            Platform transformation matrix (4, 4).
        """
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="T"))

    @property
    def theta(self) -> float:
        """
        Get the current platform yaw angle.

        Returns
        -------
        float
            The rotation about the world z-axis extracted from the platform orientation.
        """
        return q2rpy(copy.deepcopy(self._actual.x[3:]))[0]

    @property
    def v(self) -> Velocity3DType:
        """
        Get the current platform spatial velocity.

        Returns
        -------
        np.ndarray
            Platform velocity (6,).
        """
        return copy.deepcopy(self.GetVel(state="Actual", task_space="World", out="Twist"))

    @property
    def pdot(self) -> Vector2DType:
        """
        Get the current planar platform linear velocity.

        Returns
        -------
        np.ndarray
            Platform linear velocity (2,).
        """
        return copy.deepcopy(self._actual.v[:2])

    @property
    def w(self) -> Vector3DType:
        """
        Get the current platform angular velocity.

        Returns
        -------
        np.ndarray
            Platform angular velocity (3,).
        """
        return copy.deepcopy(self._actual.v[3:])

    @property
    def FT(self) -> WrenchType:
        """
        Get the current force/torque sensor data.

        Returns
        -------
        np.ndarray
            Force/Torque sensor data (6,).
        """
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Wrench"))

    @property
    def F(self) -> Vector3DType:
        """
        Get the current force sensor data.

        Returns
        -------
        np.ndarray
            Force sensor data (3,) or (..., 3).
        """
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Force"))

    @property
    def Trq(self) -> Vector3DType:
        """
        Get the current torque sensor data.

        Returns
        -------
        np.ndarray
            Torque sensor data (3,) or (..., 3).
        """
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Torque"))

    @property
    def q_ref(self) -> JointConfigurationType:
        """
        Get the commanded platform configuration.

        Returns
        -------
        np.ndarray
            Desired joint positions (nj,).
        """
        return copy.deepcopy(self._command.q)

    @property
    def qdot_ref(self) -> JointVelocityType:
        """
        Get the commanded platform configuration velocities.

        Returns
        -------
        np.ndarray
            Desired joint velocities (nj,).
        """
        return copy.deepcopy(self._command.qdot)

    @property
    def x_ref(self) -> Pose3DType:
        """
        Get the commanded platform pose.

        Returns
        -------
        np.ndarray
            Desired end-effector pose (7,).
        """
        return copy.deepcopy(self._command.x)

    @property
    def p_ref(self) -> Vector2DType:
        """
        Get the commanded planar platform position.

        Returns
        -------
        np.ndarray
            Desired end-effector position (2,).
        """
        return copy.deepcopy(self._command.x[:2])

    @property
    def Q_ref(self) -> QuaternionType:
        """
        Get the commanded platform orientation as a quaternion.

        Returns
        -------
        np.ndarray
            Desired end-effector quaternion (4,).
        """
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="Q"))

    @property
    def R_ref(self) -> RotationMatrixType:
        """
        Get the commanded platform orientation as a rotation matrix.

        Returns
        -------
        np.ndarray
            Desired end-effector rotation matrix (3, 3).
        """
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="R"))

    @property
    def T_ref(self) -> HomogeneousMatrixType:
        """
        Get the commanded platform pose as a homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            Desired end-effector transformation matrix (4, 4).
        """
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="T"))

    @property
    def theta_ref(self) -> float:
        """
        Get the commanded platform yaw angle.

        Returns
        -------
        float
            The rotation about the world z-axis extracted from the commanded orientation.
        """
        return q2rpy(copy.deepcopy(self._command.x[3:]))[0]

    @property
    def v_ref(self) -> Velocity3DType:
        """
        Get the commanded platform spatial velocity.

        Returns
        -------
        np.ndarray
            Commanded platform velocity (6,).
        """
        return q2rpy(copy.deepcopy(self._command.v))

    @property
    def pdot_ref(self) -> Vector2DType:
        """
        Get the commanded planar platform linear velocity.

        Returns
        -------
        np.ndarray
            Commanded platform linear velocity (2,).
        """
        return q2rpy(copy.deepcopy(self._command.v[:2]))

    @property
    def w_ref(self) -> Vector3DType:
        """
        Get the commanded platform angular velocity.

        Returns
        -------
        np.ndarray
            Desired end-effector angular velocity (3,) or (..., 3).
        """
        return q2rpy(copy.deepcopy(self._command.v[3:]))

    @property
    def FT_ref(self) -> WrenchType:
        """
        Get the desired force/torque sensor data.

        Returns
        -------
        np.ndarray
            Desired force/torque sensor data (6,).
        """
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Wrench"))

    @property
    def F_ref(self) -> Vector3DType:
        """
        Get the desired force sensor data.

        Returns
        -------
        np.ndarray
            Desired force sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Force"))

    @property
    def Trq_ref(self) -> Vector3DType:
        """
        Get the desired torque sensor data.

        Returns
        -------
        np.ndarray
            Desired torque sensor data (3,).
        """
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Torque"))

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
    def qdot_err(self) -> JointVelocityType:
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
    def p_err(self) -> Vector2DType:
        """
        Get the error in end-effector position.

        Returns
        -------
        np.ndarray
            Error in end-effector position (2,).
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
    def theta_err(self) -> float:
        """
        Property to calculate the difference between the reference orientation and the current orientation along z-axis.

        Returns
        -------
        float
            The difference along z-axis between the reference and current orientation.
        """
        return self.theta_ref - self.theta

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
    def pdot_err(self) -> Vector2DType:
        """
        Get the error in end-effector linear velocity.

        Returns
        -------
        np.ndarray
            Error in end-effector linear velocity (2,).
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

    # Initialization and update
    def InitObject(self) -> None:
        """
        Initializes the platforms's command and actual state with zeros, and sets default values for various attributes.

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
        self._command.x = np.zeros(7)
        self._command.v = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.ux = np.zeros(2)
        self._command.data = None
        self._command.mode = CommandModeCodes.STOP.value
        self._actual.q = np.zeros(self.nj)
        self._actual.qdot = np.zeros(self.nj)
        self._actual.trq = np.zeros(self.nj)
        self._actual.x = np.zeros(7)
        self._actual.v = np.zeros(6)
        self._actual.FT = np.zeros(6)
        self.Js = self.Kinmodel()[-1][[0, 5], :]
        self.pJs = np.linalg.pinv(self.Js)

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
        Abstract method to updates the platforms's state.

        It has to be reimplemented in actual platforms class!

        It has to set:
        - Actual platforms joint and task space states
        - Read all platforms sensors

        This method sets the following attributes:
        - `_tt`: The current time, can be retrieved using `simtime()`.
        - `_last_update`: The last update time, retrieved using `simtime()`.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the platforms.
        """
        self._tt = self.simtime()
        self._last_update = self.simtime()
        self.WarningMessage("Not implemented!")

    def Update(self) -> None:
        """
        Updates the platform's state and optionally triggers a capture callback.

        This method performs the following actions:

        - If ``_do_update`` is ``True``, it calls ``GetState()`` to update the
          platform's internal state.
        - If ``_do_capture`` is ``True`` and a capture callback function
          (``_capture_callback``) is defined, it calls the callback function,
          passing the platform object as an argument.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the platform and may
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
        Enables the update of the platform's internal state.

        This method sets the `_do_update` attribute to `True`, which allows the platform's state to be updated.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the platform.
        """
        self._do_update = True

    def DisableUpdate(self) -> None:
        """
        Disables the update of the platform's internal state.

        This method sets the `_do_update` attribute to `False`, which prevents the platform's state from being updated.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the platform.
        """
        self._do_update = False

    def GetUpdateStatus(self) -> bool:
        """
        Returns the current status of the update flag.

        This method returns the value of the `_do_update` attribute, which indicates whether the platform's state update is enabled.

        Returns
        -------
        bool
            `True` if updates are enabled, `False` otherwise.
        """
        return self._do_update

    def ResetCurrentTarget(self) -> None:
        """
        Resets the current target to the actual values of joint positions, velocities, torques, and other state variables.

        Returns
        -------
        None
        """
        self.GetState()
        self._command.q = copy.deepcopy(self._actual.q)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.trq = np.zeros(self.nj)
        self._last_control_time = self.simtime()
        if self.Robot is not None:
            self.Robot.ResetCurrentTarget()
        self._sleep(0.1)
        self.Update()

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

    def GetJointVel(self, state: Optional[str] = None, refresh: Optional[bool] = None) -> JointVelocityType:
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

    def GetJointTrq(self, state: Optional[str] = None, refresh: Optional[bool] = None) -> JointTorqueType:
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
    def GetPose(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[str] = None, refresh: Optional[bool] = None) -> Pose3DType:
        """Get platform pose

        Parameters
        ----------
        out : str, optional
            Output form, by default "x" ("Pose")
        task_space : str, optional
            Task space frame, by default "World"
        state : str, optional
            Variable state, by default "Actual"
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the pose. Default is `True`.

        Returns
        -------
        array of floats
            Platform pose
        """
        if out is None:
            out = self._default.TaskPoseForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if refresh is None:
            refresh = self._default.Refresh

        if refresh:
            self.GetState()

        if check_option(state, "Actual"):
            self.GetState()
            _x = copy.deepcopy(self._actual.x)
        elif check_option(state, "Commanded"):
            _x = copy.deepcopy(self._command.x)
        else:
            raise ValueError(f"State {state} not supported in GetPose")
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _x = self.WorldToObject(_x)
        elif check_option(task_space, "Platform"):
            _x = self.WorldToPlatform(_x)
        else:
            raise ValueError(f"Task space {task_space} not supported in GetPose")
        return map_pose(x=_x, out=out)

    def GetPos(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[str] = None) -> Vector3DType:
        """Get platform position

        Parameters
        ----------
        out : str, optional
            Output form, by default "p" ("Position")
        task_space : str, optional
            Task space frame, by default "World"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            platform position (3,)
        """
        if out is None:
            out = self._default.TaskPoseForm
        if check_option(out, "2d"):
            out = "XY"
        if out in ["Position", "p", "XY"]:
            return self.GetPose(out=out, task_space=task_space, state=state)
        else:
            raise ValueError(f"Output form {out} not supported in GetPos")

    def GetOri(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[str] = None) -> Union[QuaternionType, RotationMatrixType]:
        """Get platform orientation

        Parameters
        ----------
        out : str, optional
            Output form, by default "Q" ("Quaternion")
        task_space : str, optional
            Task space frame, by default "World"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            platform orientation (4,) or (3,3)
        """
        if out is None:
            out = self._default.TaskOriForm
        if out in ["2d", "theta"]:
            _x = self.GetPose(out="Q", task_space=task_space, state=state)
            return q2rpy(_x)[0]
        elif out in ["Quaternion", "Q", "RotationMatrix", "R", "Angle"]:
            return self.GetPose(out=out, task_space=task_space, state=state)
        else:
            raise ValueError(f"Output form {out} not supported in GetOri")

    def GetVel(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[str] = None, refresh: Optional[bool] = None) -> Velocity3DType:
        """Get platform velocity

        Parameters
        ----------
        out : str, optional
            Output form, by default "Twist"
        task_space : str, optional
            Task space frame, by default "World"
        state : str, optional
            Variable state, by default "Actual"
        refresh : bool, optional
            If `True`, the robot's state is updated before retrieving the velocity. Default is `True`.

        Returns
        -------
        array of floats
            Platform velocity (6,) or (3,)
        """
        if out is None:
            out = self._default.TaskVelForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if refresh is None:
            refresh = self._default.Refresh
        if check_option(state, "Actual"):
            self.GetState()
            _vv = copy.deepcopy(self._actual.v)
        elif check_option(state, "Commanded"):
            _vv = copy.deepcopy(self._command.v)
        else:
            raise ValueError(f"State {state} not supported")
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _vv = self.WorldToObject(_vv)
        elif check_option(task_space, "Platform"):
            _vv = self.WorldToPlatform(_vv)
        else:
            raise ValueError(f"Task space {task_space} not supported in GetVel")

        if check_option(out, "Twist"):
            return _vv
        elif check_option(out, "Linear"):
            return _vv[:3]
        elif check_option(out, "Angular"):
            return _vv[3:]
        elif check_option(out, "2d"):
            return _vv[[0, 1, 5]]
        else:
            raise ValueError(f"Output form {out} not supported")

    def GetFT(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[str] = None, avg_time: float = 0, refresh: Optional[bool] = None) -> WrenchType:
        """
        Get the platforms's end-effector pose.

        Parameters
        ----------
        out : str, optional
            Output form for the pose. The default is "x" ("Pose").
        task_space : str, optional
            Task space frame to use. Options are "World", "Object", and "Platform". The default is "World".
        state : str, optional
            The state of the robot to use. Options are "Actual" or "Commanded". The default is "Actual".
        refresh : bool, optional
            If `True`, the platforms's state is updated before retrieving the pose. Default is `True`.

        Returns
        -------
        np.ndarray
            The end-effector pose in the specified form, shape varies depending on `out` value.

        Raises
        ------
        ValueError
            If the `state`, or `task_space` options are invalid.
        """
        if out is None:
            out = self._default.TaskFTForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if refresh is None:
            refresh = self._default.Refresh
        if check_option(state, "Actual"):
            self.GetState()
            _FT = self._actual.FT  # in EE (tool) CS
        elif check_option(state, "Commanded"):
            _FT = self._command.FT  # in robot CS
        else:
            raise ValueError(f"State {state} not supported")

        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _FT = self.WorldToObject(_FT, typ="Wrench")
        elif check_option(task_space, "Platform"):
            _FT = self.WorldToPlatform(_FT, typ="Wrench")
        else:
            raise ValueError(f"Task space {state} not supported in GetFT")

        if check_option(out, "Wrench"):
            return _FT
        elif check_option(out, "Force"):
            return _FT[:3]
        elif check_option(out, "Torque"):
            return _FT[3:]
        elif check_option(out, "2d"):
            return _FT[[0, 1, 5]]
        else:
            raise ValueError(f"Output form {out} not supported")

    # Task space motion
    def Set_vel(self, v: ArrayLike, wait: Optional[float] = None) -> int:
        if wait is None:
            wait = self.tsamp
        v = vector(v, dim=2)
        self._command.u = self.pJs @ v
        self._command.ux = v
        self._command.v = np.array([v[0], 0.0, 0.0, 0.0, 0.0, v[1]])
        self._sleep(wait)
        return MotionResultCodes.MOTION_SUCCESS.value

    def CMoveToOri(self, rtheta: float, task_space: Optional[str] = None, wait: Optional[float] = None, vel_fac: Optional[float] = None, ori_err: Optional[float] = None, k_rot: Optional[float] = None, asynchronous: bool = False) -> Union[Thread, int]:
        if asynchronous:
            self.Message("ASYNC CMoveToOri", 2)
            _th = Thread(target=self._CMoveToOri, args=(rtheta,), kwargs={"task_space": task_space, "wait": wait, "vel_fac": vel_fac, "ori_err": ori_err, "k_rot": k_rot}, daemon=True)
            _th.start()
            return _th
        else:
            return self._CMoveToOri(rtheta, task_space=task_space, wait=wait, vel_fac=vel_fac, ori_err=ori_err, k_rot=k_rot)

    def _CMoveToOri(self, rtheta: float, task_space: Optional[str] = None, wait: Optional[float] = None, vel_fac: Optional[float] = None, ori_err: Optional[float] = None, k_rot: Optional[float] = None) -> int:
        if self._semaphore._value <= 0:
            self.WarningMessage("CMoveToOri not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        if task_space is None:
            task_space = self._default.TaskSpace
        if wait is None:
            wait = self._default.Wait
        if vel_fac is None:
            vel_fac = self._default.VelFac
        if ori_err is None:
            ori_err = self._default.OriErr
        if k_rot is None:
            k_rot = self._default.Krot

        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            rtheta = self.ObjectToWorld(rtheta)
        elif check_option(task_space, "Platform"):
            rtheta = self.PlatformToWorld(rtheta)
        else:
            raise ValueError(f"Task space {task_space} not supported in CMoveToOri")

        if wait == self._default.Wait:
            _eo = rtheta - self.theta
            wait = max(_eo / self.v_max[1], _eo / self.v_min[1]) * 10

        self.Message(f"CMoveToOri started: theta={rtheta:.3f}", 2)
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.PLANAR_ORI.value
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        self._command.x[3:] = rot_z(rtheta)
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value
        _tstart = self.simtime()
        while (self.simtime() - _tstart) < wait:
            _eo = wrap_to_pi(rtheta - self.theta)
            _u1 = 0
            _u2 = k_rot * _eo
            _ux = np.array([_u1, _u2])
            _fac = max(max(_ux / self.v_max), max(_ux / self.v_min), 1)
            _ux = _ux / _fac * vel_fac
            self._command.v[:2] = q2r(self.x[3:])[:2, :2] @ np.array([_ux[0], 0.0])
            self._command.v[5] = _ux[1]
            self.Set_vel(_ux)
            if abs(_eo) < ori_err:
                break

        self.Stop()
        self.Message("CMoveToOri finished", 2)
        return self._last_status

    def CMoveToLocation(
        self,
        rp: Vector2DType,
        rtheta: Optional[float] = None,
        task_space: Optional[str] = None,
        robot_as_a_sensor: bool = False,
        min_dist: Optional[float] = None,
        approach_dist: Optional[float] = None,
        final_orientation_correction: bool = True,
        wait: Optional[float] = None,
        vel_fac: Optional[float] = None,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        k_dist: Optional[float] = None,
        k_dir: Optional[float] = None,
        asynchronous: bool = False,
        allow_backward: bool = False,
        reach_check_fn: Optional[Callable[..., bool]] = None,
        **kwargs: Any,
    ) -> Union[Thread, int]:
        rp = vector(rp)
        if isvector(rp, dim=2):
            pass
        elif isvector(rp, dim=3):
            rtheta = rp[2] if rtheta is None else rtheta
            rp = rp[:2]
        if asynchronous:
            self.Message("ASYNC CMoveToLocation", 2)
            _th = Thread(
                target=self._CMoveToLocation,
                args=(rp,),
                kwargs={"rtheta": rtheta, "task_space": task_space, "min_dist": min_dist, "approach_dist": approach_dist, "wait": wait, "vel_fac": vel_fac, "pos_err": pos_err, "ori_err": ori_err, "k_dist": k_dist, "k_dir": k_dir},
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CMoveToLocation(
                rp,
                rtheta=rtheta,
                task_space=task_space,
                robot_as_a_sensor=robot_as_a_sensor,
                min_dist=min_dist,
                approach_dist=approach_dist,
                final_orientation_correction=final_orientation_correction,
                wait=wait,
                vel_fac=vel_fac,
                pos_err=pos_err,
                ori_err=ori_err,
                k_dist=k_dist,
                k_dir=k_dir,
                allow_backward=allow_backward,
                reach_check_fn=reach_check_fn,
                **kwargs,
            )

    def _CMoveToLocation(
        self,
        rp: Vector2DType,
        rtheta: Optional[float] = None,
        task_space: Optional[str] = None,
        robot_as_a_sensor: bool = False,
        min_dist: Optional[float] = None,
        approach_dist: Optional[float] = None,
        final_orientation_correction: bool = True,
        wait: Optional[float] = None,
        vel_fac: Optional[float] = None,
        pos_err: Optional[float] = None,
        ori_err: Optional[float] = None,
        k_dist: Optional[float] = None,
        k_dir: Optional[float] = None,
        min_vel: float = 0.02,
        allow_backward: bool = False,
        reach_check_fn: Optional[Callable[..., bool]] = None,
        **kwargs: Any,
    ) -> int:
        if self._semaphore._value <= 0:
            self.WarningMessage("CMoveToLocation not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        if task_space is None:
            task_space = self._default.TaskSpace
        if wait is None:
            wait = self._default.Wait
        if min_dist is None:
            min_dist = np.inf
        if approach_dist is None:
            approach_dist = self._default.ApproachDist
        if vel_fac is None:
            vel_fac = self._default.VelFac
        if pos_err is None:
            pos_err = self._default.PosErr
        if k_dist is None:
            k_dist = self._default.Kdist
        if k_dir is None:
            k_dir = self._default.Kdir

        rp = vector(rp, dim=2)
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            rp = self.ObjectToWorld(rp)
            if rtheta is not None:
                rtheta = self.ObjectToWorld(rtheta)
        elif check_option(task_space, "Platform"):
            rp = self.PlatformToWorld(rp)
            if rtheta is not None:
                rtheta = self.PlatformToWorld(rtheta)
        else:
            raise ValueError(f"Task space {task_space} not supported in CMoveToLocation")

        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        if rtheta is not None:
            self._command.x[3:] = rot_z(rtheta)
        self._command.x[:2] = rp

        _ee = rp - self._actual.x[:2]
        if approach_dist == 0:
            _beta = np.arctan2(_ee[1], _ee[0])
            self.Message("CMoveToLocation -> CMoveToOri", 2)
            self._semaphore.release()
            self._CMoveToOri(_beta, ori_err=ori_err)
            if self._semaphore._value <= 0:
                self.WarningMessage("CMoveToLocation not executed due to active threads!")
                return MotionResultCodes.ACTIVE_THREADS.value
            self._semaphore.acquire()

        if (wait == self._default.Wait) and (pos_err < np.inf):
            wait = 100 * max(np.linalg.norm(_ee) / self.v_max[0] / vel_fac, 1)

        _ang0 = None
        _in_range = False

        # self.Message(f"CMoveToLocation started: xy={rp} theta={rtheta:.3f}", 2) TODO
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.PLANAR_LOCATION.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        _tstart = self.simtime()
        _distance_to_target = np.linalg.norm(rp - self.p[:2])

        # Procedure for cheking if we are in backward case
        if robot_as_a_sensor:
            _xrObj = np.linalg.inv(x2t(self.Robot._actual.x)) @ np.linalg.inv(self.TRobotBase)
            _xaWorld = self.ObjectToWorld(t2x(_xrObj))
            _xa = _xaWorld[:2]
            _theta = np.arctan2(x2t(_xaWorld)[1, 0], x2t(_xaWorld)[0, 0])
        else:
            _xa = self.p[:2]
            _theta = self.theta
        vec_to_goal = rp - _xa  # vector from robot to goal
        head = np.array([np.cos(_theta), np.sin(_theta)])  # heading of robot
        forward_dot = np.dot(vec_to_goal, head)  # dot product: positive = front hemisphere, negative = rear hemisphere
        backward_case = allow_backward and (forward_dot < 0)  # enable backward if goal in rear hemisphere

        while ((self.simtime() - _tstart) < wait) and not self._abort_motion:
            if robot_as_a_sensor:
                _xrObj = np.linalg.inv(x2t(self.Robot._actual.x)) @ np.linalg.inv(self.TRobotBase)
                _xaWorld = self.ObjectToWorld(t2x(_xrObj))
                _xa = _xaWorld[:2]
                _theta = np.arctan2(x2t(_xaWorld)[1, 0], x2t(_xaWorld)[0, 0])
            else:
                _xa = self.p[:2]
                _theta = self.theta

            if backward_case:
                _theta = wrap_to_pi(_theta + np.pi)  # "virtual" angle for backward motion

            if rtheta is None:
                self._command.x[3:] = copy.deepcopy(self._actual.x[3:])
                _xref = rp
            else:
                if backward_case:
                    _rtheta = wrap_to_pi(rtheta + np.pi)  # "virtual" angle for appcoach
                else:
                    _rtheta = rtheta
                _dx = rp - _xa
                _ndx = np.linalg.norm(_dx)
                if _ndx < _eps:
                    _xref = rp
                elif (_ndx < min_dist and approach_dist > 0) or _in_range:
                    _in_range = True
                    _beta = np.arctan2(_dx[1], _dx[0])
                    _gamma = min(max(wrap_to_pi(_beta - _rtheta), -np.pi / 4), np.pi / 4)
                    _phi = _beta + min(approach_dist / _ndx, 1) * _gamma
                    _xref = _xa + _ndx * np.array([np.cos(_phi), np.sin(_phi)])
                else:
                    _xref = rp

            _ee = _xref - _xa
            _dist = np.linalg.norm(_ee)
            _ang = np.arctan2(_ee[1], _ee[0])
            if _ang0 is not None:
                _dang = _ang - _ang0
                if _dang > np.pi:
                    _ang -= 2 * np.pi
                elif _dang < -np.pi:
                    _ang += 2 * np.pi
            _ang0 = _ang
            _alpha = wrap_to_pi(_ang - _theta)
            _scale = min(_dist / min(pos_err, 0.1), 1.0)

            _u1 = k_dist * max(np.cos(2 * _alpha), 0) * _scale * _dist
            _u2 = k_dir * _scale * _alpha
            if backward_case:
                _u1 = -_u1  # physically we are moving backward

            _ux = np.array([_u1, _u2])
            _fac = max(max(_ux / self.v_max), max(_ux / self.v_min), 1)
            _ux = _ux / _fac * vel_fac
            self._command.v[:2] = q2r(self.x[3:])[:2, :2] @ np.array([_ux[0], 0.0])
            self._command.v[5] = _ux[1]
            self.Set_vel(_ux)
            if (_dist < pos_err) or ((np.linalg.norm(_ee) < 0.1 * _distance_to_target) and (np.linalg.norm(self._actual.v[:2]) < min_vel)):
                if rtheta is not None and final_orientation_correction:
                    self.Message("CMoveToLocation -> CMoveToOri", 2)
                    self._semaphore.release()
                    self._CMoveToOri(rtheta, ori_err=ori_err)
                    if self._semaphore._value <= 0:
                        self.WarningMessage("CMoveToLocation not executed due to active threads!")
                        return MotionResultCodes.ACTIVE_THREADS.value

                    self._semaphore.acquire()
                break
            if robot_as_a_sensor and reach_check_fn is not None:
                if reach_check_fn(self.Robot, **kwargs):
                    self.Message("CMoveToLocation aborted by reach_check_fn", 2)
                    break

        self.Stop()
        self.Message("CMoveToLocation finished", 2)
        return self._last_status

    def PForward(self, d: float, t: float = 1, traj: Optional[str] = None, asynchronous: bool = False) -> Union[Thread, int]:
        if asynchronous:
            self.Message("ASYNC PForward", 2)
            _th = Thread(target=self._PForward, args=(d,), kwargs={"t": t, "traj": traj}, daemon=True)
            _th.start()
            return _th
        else:
            return self._PForward(d, t, traj=traj)

    def _PForward(self, d: float, t: float = 1, traj: Optional[str] = None) -> int:
        if self._semaphore._value <= 0:
            self.WarningMessage("PForward not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        if traj is None:
            traj = self._default.Traj

        self.Message("PForward started", 2)
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.PLANAR_FORWARD.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        time = np.arange(self.tsamp, t + self.tsamp, self.tsamp)
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        p0 = self.p_ref[:2]
        R0 = self.R_ref[:2, :2]
        _di, _vi, _ = jtraj(0, d, time, traj=traj)
        _fac = max(np.max(_vi, axis=0) / self.v_max[0], np.min(_vi, axis=0) / self.v_min[0])
        if _fac > 1000:
            self.Stop()
            self.Message("PForward not possible", 0)
            return
        elif _fac > 1:
            time = np.arange(self.tsamp, t * _fac + self.tsamp, self.tsamp)
            _di, _vi, _ = jtraj(0, d, time, traj=traj)

        for _xt, _vt in zip(_di, _vi):
            if self._do_motion_check and self._motion_check_callback is not None:
                self._last_status = self._motion_check_callback(self)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Motion abborted")
                    self._semaphore.release()
                    return self._last_status
            self._command.x[:2] = p0 + R0 @ np.array([_xt, 0.0])
            self._command.v[:2] = R0 @ np.array([_vt, 0.0])
            self.Set_vel([_vt, 0])
            if self._abort_motion:
                break

        self.Stop()
        self.Message("PForward finished", 2)
        return self._last_status

    def PTurn(self, ang: float, t: float = 1, traj: Optional[str] = None, asynchronous: bool = False) -> Union[Thread, int]:
        if asynchronous:
            self.Message("ASYNC PTurn", 2)
            _th = Thread(target=self._PTurn, args=(ang,), kwargs={"t": t, "traj": traj}, daemon=True)
            _th.start()
            return _th
        else:
            return self._PTurn(ang, t, traj=traj)

    def _PTurn(self, ang: float, t: float = 1, traj: Optional[str] = None) -> int:
        if self._semaphore._value <= 0:
            self.WarningMessage("PTurn not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        if traj is None:
            traj = self._default.Traj

        self.Message("PTurn started", 2)
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.PLANAR_TURN.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        time = np.arange(self.tsamp, t + self.tsamp, self.tsamp)
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        theta0 = self.theta
        _ai, _wi, _ = jtraj(0, ang, time, traj=traj)
        _fac = max(np.max(_wi, axis=0) / self.v_max[1], np.min(_wi, axis=0) / self.v_min[1])
        if _fac > 1000:
            self.Stop()
            self.Message("PTurn not possible", 0)
            return
        elif _fac > 1:
            time = np.arange(self.tsamp, t * _fac + self.tsamp, self.tsamp)
            _ai, _wi, _ = jtraj(0, ang, time, traj=traj)

        for _xt, _vt in zip(_ai, _wi):
            if self._do_motion_check and self._motion_check_callback is not None:
                self._last_status = self._motion_check_callback(self)
                if self._last_status > MotionResultCodes.MOTION_SUCCESS.value:
                    self.WarningMessage("Motion abborted")
                    self._semaphore.release()
                    return self._last_status
            self._command.x[3:] = rot_z(theta0 + _xt)
            self._command.v[5] = _vt
            self.Set_vel([0, _vt])

        self.Stop()
        self.Message("PTurn finished", 2)
        return self._last_status

    def AutonomousMotion(self, callback: Optional[Callable[..., int]] = None, asynchronous: bool = True, **kwargs: Any) -> Union[Thread, int]:
        if callback is None:
            self.Message("No callback for autonomous motion defined!", 0)
            return
        elif not callable(callback):
            self.Message("Parameter for autonomous motion is not a function!", 0)
            return
        elif self.Robot is None:
            self.Message("No robot attached to platform!", 0)
            return
        self._abort_autonomous_motion = False

        if asynchronous:
            self.Message("ASYNC AutonmousMotion", 2)
            _th = Thread(target=self._AutonomousMotion, args=(), kwargs={"callback": callback, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._AutonomousMotion(callback=callback, **kwargs)

    def _AutonomousMotion(self, callback: Optional[Callable[..., int]] = None, **kwargs: Any) -> int:
        if self._semaphore._value <= 0:
            self.WarningMessage("AutonomousMotion not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        self.Message("AutonmousMotion started", 2)
        if not self.Start():
            return MotionResultCodes.NOT_READY.value
        self._command.mode = CommandModeCodes.AUTONOMOUS.value
        self._last_status = MotionResultCodes.MOTION_SUCCESS.value

        self._last_status = callback(**kwargs)

        self.Stop()
        self.Message("AutonmousMotion finished", 2)
        return self._last_status

    def AbortAutonomousMotion(self) -> None:
        self._abort_autonomous_motion = True

    # Transformations
    def PlatformToWorld(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """Map from platform base frame to world frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj)

        Parameters
        ----------
        x : ArrayLike
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - JacobianType (6, nj)
            - 2D position (2,)
            - rotation (1,)
        typ : str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported

        Note
        ----
        2D position and scalar rotation can be used only if `z`-axis of both frames are colinear
        """
        R0 = q2r(self._actual.x[3:])
        p0 = self._actual.x[:3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=2):
            return R0[:2, :2] @ x + p0[:2]
        elif isscalar(x):
            return x + self.theta
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToPlatform(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """Map from world frame to robot base frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj)

        Parameters
        ----------
        x : ArrayLike
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - JacobianType (6, nj)
            - 2D position (2,)
            - rotation (1,)
        typ : str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported

        Note
        ----
        2D position and scalar rotation can be used only if `z`-axis of both frames are colinear
        """
        R0 = q2r(self._actual.x[3:]).T
        p0 = -R0 @ self._actual.x[:3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
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
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=2):
            return R0[:2, :2] @ x + p0[:2]
        elif isscalar(x):
            return x - self.theta
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def ObjectToWorld(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """Map from object frame to world frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj)

        Parameters
        ----------
        x : ArrayLike
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - JacobianType (6, nj)
            - 2D position (2,)
            - rotation (1,)
        typ : str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported

        Note
        ----
        2D position and scalar rotation can be used only if `z`-axis of both frames are colinear
        """
        R0 = self.TObject[:3, :3]
        p0 = self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            # if typ == "Wrench":  # wrench (F)
            #     RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=2):
            return R0[:2, :2] @ x + p0[:2]
        elif isscalar(x):
            R_obj = rot_z(x, out="R")
            R_world = R0 @ R_obj
            theta_world = np.arctan2(R_world[1, 0], R_world[0, 0])
            return theta_world
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToObject(self, x: ArrayLike, typ: Optional[str] = None) -> np.ndarray:
        """Map from world frame to object frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and JacobianType (6, nj)

        Parameters
        ----------
        x : ArrayLike
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - JacobianType (6, nj)
            - 2D position (2,)
            - rotation (1,)
        typ : str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported

        Note
        ----
        2D position and scalar rotation can be used only if `z`-axis of both frames are colinear
        """
        R0 = self.TObject[:3, :3].T
        p0 = -R0 @ self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            # if typ == "Wrench":  # wrench (F)
            #     RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=2):
            return R0[:2, :2] @ x + p0[:2]
        elif isscalar(x):
            return x + self.theta
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    # Kinematic utilities
    def Kinmodel(self, q: Optional[ArrayLike] = None, out: str = "x") -> Tuple[np.ndarray, JacobianType]:
        if q is None:
            _q = self.x
        else:
            _q = q
        J = np.zeros((6, 2))
        J[0, 0] = 1
        J[0, 1] = 1
        J[5, 0] = -1
        J[5, 1] = 1
        return map_pose(x=_q, out=out), J

    def Jacobi(self) -> JacobianType:
        km = self.Kinmodel()
        return km[-1]

    # Object
    def SetObject(self, x: Optional[Union[Pose3DType, HomogeneousMatrixType]] = None) -> None:
        """
        Set the object pose in the platform coordinate system.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType], optional
            The object pose as a pose vector ``(7,)`` or homogeneous matrix ``(4, 4)``.
            If ``None``, the current actual platform pose is used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the pose shape is not supported or if the input pose z-axis is not aligned
            with ``[0, 0, 1]``.
        """
        if x is None:
            _x = self._actual.x
        else:
            _x = self.spatial(x)
        if _x.shape == (4, 4):
            _T = _x
        elif isvector(_x, dim=7):
            _T = x2t(_x)
        else:
            raise ValueError(f"Object pose shape {_x.shape} not supported")

        _z_axis = _T[:3, 2]
        if not np.allclose(_z_axis, np.array([0.0, 0.0, 1.0]), atol=_eps, rtol=0.0):
            raise ValueError(f"Object pose z-axis must equal [0, 0, 1], got {_z_axis}")
        self.TObject = _T

    def GetObject(self, out: str = "T", task_space: Optional[str] = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the object pose in the specified task space.

        Parameters
        ----------
        out : str, optional
            The output format of the object pose. Default is ``"T"``.
        task_space : str, optional
            The task space for the pose transformation. Supported values are
            ``"World"``, ``"Object"``, and ``"Platform"``.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The object pose in the requested output format.

        Raises
        ------
        ValueError
            If the task space is not supported.
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        _T = self.TObject
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Platform"):
            _T = self.WorldToPlatform(_T)
        else:
            raise ValueError(f"Task space {task_space} not supported in GetObject")
        return map_pose(T=_T, out=out)

    # Robot base
    def AttachTo(self, robot: robot) -> None:
        """Attach a robot instance to the platform."""
        self.Robot = robot

    def Detach(self) -> None:
        """Detach the currently attached robot from the platform."""
        self.Robot = None

    def GetAttachedRobot(self) -> Tuple[Optional[robot], str]:
        """
        Get the attached robot and its name.

        Returns
        -------
        Tuple[Optional[robot], str]
            The attached robot instance and its name, or ``(None, "None")``.
        """
        if self.Robot is None:
            return None, "None"
        else:
            return self.Robot, self.Robot.Name

    def SetRobotBase(self, x: ArrayLike) -> None:
        """
        Set the robot base pose relative to the platform.

        Parameters
        ----------
        x : ArrayLike
            The robot base pose as a homogeneous matrix, pose vector, position,
            rotation matrix, or quaternion.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not supported.
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
            raise ValueError(f"Robot base frame shape {x.shape} not supported")
        self.TRobotBase = _T

    def GetRobotBase(self, out: str = "T") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the robot base pose relative to the platform.

        Parameters
        ----------
        out : str, optional
            The output format of the base pose. Default is ``"T"``.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The robot base pose in the requested output format.
        """
        if out is None:
            out = self._default.TaskPoseForm
        return map_pose(T=self.TRobotBase, out=out)

    def GetRobotBasePose(self, out: Optional[str] = None) -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, QuaternionType, RotationMatrixType]:
        """
        Get the attached robot base pose in the world frame.

        Parameters
        ----------
        out : str, optional
            The output format of the base pose. If ``None``, the default task pose
            format is used.

        Returns
        -------
        Pose3DType or HomogeneousMatrixType or Vector3DType or QuaternionType or RotationMatrixType
            The robot base pose in the world frame.
        """
        if out is None:
            out = self._default.TaskPoseForm
        self.GetState()
        return map_pose(T=self.T @ self.TRobotBase, out=out)

    # Movements
    def Start(self) -> bool:
        self._command.mode = CommandModeCodes.START.value
        self._last_control_time = self.simtime()
        self._abort_motion = False
        self.Update()
        return True

    def Stop(self) -> None:
        self._command.mode = CommandModeCodes.STOP.value
        self._command.qdot = np.zeros(self.nj)
        self._command.v = np.zeros(6)
        self.reset_threads()
        self.Update()

    def WaitUntilStopped(self, eps: float = 0.001) -> None:
        self.GetState()
        while np.linalg.norm(self._actual.qdot) > eps:
            self.GetState()

    def Wait(self, wait: float, dt: Optional[float] = None) -> Optional[int]:
        if self._semaphore._value <= 0:
            self.WarningMessage("Wait not executed due to active threads!")
            return MotionResultCodes.ACTIVE_THREADS.value

        self._semaphore.acquire()
        self.Message(f"Wait for {wait} s", 2)
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
        self.Stop()
        self.Start()

    def SetMotionCheckCallback(self, fun: Callable[..., int]) -> None:
        """Set the motion-check callback."""
        self._motion_check_callback = fun

    def EnableMotionCheck(self, check: bool = True) -> None:
        """Enable motion-check callbacks."""
        self._do_motion_check = check

    def DisableMotionCheck(self) -> None:
        """Disable motion-check callbacks."""
        self._do_motion_check = False

    # Utilities
    def SetCaptureCallback(self, fun: Callable[..., None]) -> None:
        """Set the callback used during capture updates."""
        self._capture_callback = fun

    def StartCapture(self) -> None:
        """Enable capture callbacks during state updates."""
        if not self._do_update:
            self.WarningMessage("Update is not enabled")
        # self._t0 = self._tt
        self._do_capture = True
        self.Update()

    def StopCapture(self) -> None:
        """Disable capture callbacks during state updates."""
        self._do_capture = False

    def SetUserData(self, data: Optional[Any]) -> None:
        """Store user-defined data in the command state."""
        self._command.data = data
        self.Update()

    def GetUserData(self) -> Optional[Any]:
        """Get user-defined data stored in the command state."""
        return self._command.data


def isplatform(obj: object) -> bool:
    """
    Check whether an object is a platform instance.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of :class:`platform`, otherwise ``False``.
    """
    return isinstance(obj, platform)
