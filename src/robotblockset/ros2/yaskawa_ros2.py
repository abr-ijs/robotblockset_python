"""Yaskawa robots ROS 2 interface implementations.

This module defines ROS 2-backed robot wrappers used by RobotBlockSet robot
interfaces. It provides communication plumbing, controller integration, state
feedback handling, and motion-execution helpers for multiple robot platforms.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Optional, Tuple, Union
import numpy as np
from time import sleep
from copy import deepcopy
from pathlib import Path

from robotblockset.robots import robot
from robotblockset.trajectories import interpPath
from robotblockset.rbs_typing import JointPathType, TimesType

from robotblockset.ros2.controllers_ros2 import JointTrajectoryControllerInterface
from robotblockset.ros2.robots_ros2 import robot_ros2
from robotblockset.robot_spec import hc20_spec, hc30_spec

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from std_msgs.msg import Bool
    from std_srvs.srv import Trigger
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if all are installed.")


# Check for Yaskawa specific message imports
try:
    from industrial_msgs.msg import RobotStatus
    from motoros2_interfaces.srv import StartTrajMode, ResetError, SelectMotionTool
except ImportError:
    RobotStatus = None
    StartTrajMode = None
    ResetError = None


# Yaskawa robots
class yaskawa_ros2(robot_ros2, robot):
    """
    Yaskawa collaborative robot class using ROS2 and MotoROS2 interface.

    This class provides a ROS2-based interface to a Yaskawa robot
    using the MotoROS2 driver. It initializes subscriptions to robot status,
    service clients to control trajectory execution modes, and sets up a
    joint trajectory controller interface.

    Unlike generic robot interfaces, **control strategy changes are not allowed**.
    The robot only supports *JointPositionTrajectory* control via
    ``follow_joint_trajectory`` action.

    Attributes
    ----------
    robot_mode : RobotStatus | None
        Last robot-status message received from MotoROS2.
    last_robot_status_callback_time : float
        Robot time of the last robot-status callback.
    start_traj_mode_client : Any
        Client for the ``start_traj_mode`` service.
    stop_traj_mode_client : Any
        Client for the ``stop_traj_mode`` service.
    reset_error_client : Any
        Client for the ``reset_error`` service.
    select_tool_client : Any
        Client for the ``select_motion_tool`` service.
    """

    def __init__(self, name: str = "hc20", namespace: str = "yaskawa", tof_topic: Optional[str] = None) -> None:
        """
        Initialize the HC20 ROS 2 interface.

        Parameters
        ----------
        name : str, optional
            Name of the ROS2 node (default: ``"hc20"``).
        namespace : str, optional
            Namespace for all robot topics and services. Determines ROS namespace prefix.
            Default is ``"yaskawa"``.
        tof_topic : str, optional
            Optional topic used to monitor the TOF breach state.

        Notes
        -----
        This wrapper always uses the MotoROS2 ``follow_joint_trajectory``
        action and therefore does not support controller strategy switching.
        """
        robot_ros2.__init__(self, name=name, namespace=namespace, strategy_to_controller_interface_mapping=None, joint_states_topic="joint_states", control_strategy=None)

        if RobotStatus is None or StartTrajMode is None:
            raise ImportError("ROS2 `industrial_msgs` or `motoros2_interfaces` package not found. Robot in hc20_ros2 class will not be available.")

        # Subscribers
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5, durability=DurabilityPolicy.SYSTEM_DEFAULT)
        self.robot_status_subscription = self._node.create_subscription(msg_type=RobotStatus, topic=f"{self._namespace}/robot_status", callback=self._robot_status_callback, qos_profile=qos)
        self._node.get_logger().info(f"Subscribing to robot state topic: {self._namespace}/robot_status")
        self.robot_mode = None
        self.last_robot_status_callback_time = 0

        # Initialize ROS2 subscription to TFO state
        if tof_topic is not None:
            topic_full_name = f"{self._namespace}/{tof_topic.strip('/')}"
            self._node.get_logger().info(f"Subscribing to TOF camera state topic: {topic_full_name}")
            self._tof_sub = self._node.create_subscription(msg_type=Bool, topic=topic_full_name, callback=self._tof_callback, qos_profile=10)

        # Services
        self.start_traj_mode_client = self._node.create_client(StartTrajMode, f"{self._namespace}/start_traj_mode")
        self._node.get_logger().info(f"Client for service {self._namespace}/start_traj_mode")
        self.stop_traj_mode_client = self._node.create_client(Trigger, f"{self._namespace}/stop_traj_mode")
        self._node.get_logger().info(f"Client for service {self._namespace}/stop_traj_mode")
        self.reset_error_client = self._node.create_client(ResetError, f"{self._namespace}/reset_error")
        self._node.get_logger().info(f"Client for service {self._namespace}/reset_error")
        self.select_tool_client = self._node.create_client(SelectMotionTool, f"{self._namespace}/select_motion_tool")
        self._node.get_logger().info(f"Client for service {self._namespace}/select_motion_tool")

        # Start spinning only after all publishers/subscribers/clients are created
        self._start_spinning(wait_for_state=True)

        self.controller = JointTrajectoryControllerInterface(namespace=namespace, ros_plugin_name=None, action="follow_joint_trajectory", topic="")
        self.controller.Activate(robot=self, node=self._node)
        self._control_strategy = "JointPositionTrajectory"

        self.Init()
        self.Message("Initialized", 2)

    # States
    def _robot_status_callback(self, msg: RobotStatus) -> None:  # pyright: ignore[reportInvalidTypeForm]
        """
        Store the latest robot-status message.

        Parameters
        ----------
        msg : RobotStatus
            Robot-status message received from MotoROS2.
        """
        self.robot_mode = msg
        self.last_robot_status_callback_time = self.simtime()

    def _tof_callback(self, msg: Bool) -> None:
        """
        Store the latest TOF breach-state message.

        Parameters
        ----------
        msg : Bool
            TOF breach-state message.
        """
        # print(f"Status: {msg.data}")
        self.state = deepcopy(msg)
        self._last_state_callback_time = self.simtime()

    # Strategies
    def SetStrategy(self, new_strategy: str) -> None:
        """
        Reject attempts to change the control strategy.

        Parameters
        ----------
        new_strategy : str
            Requested control strategy.

        Notes
        -----
        The HC20 wrapper always uses the MotoROS2 trajectory interface and
        therefore ignores strategy changes.
        """
        self.WarningMessage("Strategy can not be set or changed!")

    # Movements
    def GoTo_qtraj(self, q: JointPathType, qdot: JointPathType, qddot: JointPathType, time: TimesType) -> int:
        """
        Command the Yaskawa robot to follow a joint trajectory.

        It is intended to control the robot to follow a trajectory specified by joint positions (`q`),
        velocities (`qdot`), and accelerations (`qddot`) over a specified time (`time`).

        Due to controller resource constraints and implementation details of micro-ROS, MotoROS2
        imposes an upper limit on the number of JointTrajectoryPoints in a JointTrajectorys submitted
        as part of control_msgs/FollowJointTrajectory action goals.

        This maximum number of points in a single trajectory is currently 200.

        Parameters
        ----------
        q : JointPathType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points.
        qdot : JointPathType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points.
        qddot : JointPathType
            Desired joint accelerations for the trajectory (n, nj). Not used in MotoROS2 controller.
        time : TimesType
            Time points for the trajectory (n,).

        Returns
        -------
        int
            Controller status code (0 for success).

        Raises
        ------
        ValueError
            If control is not implemented in controller.
        """
        if hasattr(self.controller, "GoTo_qtraj"):
            q[0] = self.q.copy()
            if q.shape[0] >= 200:
                # Resample trajectory to 200 samples based on time
                _time = np.linspace(time[0], time[-1], 200)
                _q = interpPath(time, q, _time)
                _qdot = interpPath(time, qdot, _time)
                _stat = self.controller.GoTo_qtraj(_q, _qdot, qddot, _time)
                # self._command.q = _q[-1]
            else:
                _stat = self.controller.GoTo_qtraj(q, qdot, qddot, time)
                # self._command.q = q[-1]
            # self._command.qdot = np.zeros(self.nj)
            # x = self.Kinmodel(self._command.q)[0]
            # self._command.x = x
            # self._command.v = np.zeros(6)
            self.ResetCurrentTarget()  # HC20 must start from current configuration!s
            return _stat
        else:
            raise NotImplementedError("GoTo_qtraj_pub method not implemented for the current controller")

    def StartTrajectoryMode(self, timeout_sec: float = 4.0) -> Optional[int]:
        """
        Enable MotoROS2 trajectory mode.

        This service engages servo drives and activates the trajectory execution
        mode required by ``follow_joint_trajectory`` actions.

        Parameters
        ----------
        timeout_sec : float, optional
            Maximum time (in seconds) to wait for service availability and response.
            Default is ``4.0``.

        Returns
        -------
        int | None
            Result code returned by ``StartTrajMode``, or ``None`` if the
            service is unavailable, fails, or times out.

        Notes
        -----
        - If activation fails, use :meth:`ResetError` first and inspect the
          ``robot_status`` topic to identify the issue.
        - Extended MotoROS2 error codes may be returned (e.g. ``111`` → “PFL event”).
        - Motion cannot be executed unless this mode is successfully activated.

        """
        if not self.start_traj_mode_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("Service StartTrajectoryMode not available")
            return None
        request = StartTrajMode.Request()
        future = self.start_traj_mode_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self._node.get_clock().now()
        while rclpy.ok() and not future.done():
            sleep(0.01)
            if timeout_sec is not None:
                elapsed = (self._node.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self._node.get_logger().error("Service StartTrajectoryMode call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self._node.get_logger().info(f"Service StartTrajectoryMode: {response.message}")
            # motoros2_interfaces.srv.StartTrajMode_Response(result_code=motoros2_interfaces.msg.MotionReadyEnum(value=111), message='A PFL event has occurred. Please return the robot to a safe state')
        except Exception as e:
            self._node.get_logger().error(f"Service StartTrajectoryMode call failed: {e}")
            return None
        self.ResetCurrentTarget()
        return response.result_code.value

    def StopTrajectoryMode(self, timeout_sec: float = 2.0) -> Optional[bool]:
        """
        Disable trajectory mode on the robot.

        Attempts to deactivate trajectory execution mode. Servo drives remain enabled
        if they were active prior to deactivation.

        Parameters
        ----------
        timeout_sec : float, optional
            Maximum time to wait for service availability and response.
            Default is ``2.0``.

        Returns
        -------
        bool or None
            ``True`` if the service call succeeded and the mode was deactivated.
            ``False`` if deactivation failed.
            ``None`` if the service is unavailable, fails, or times out.

        Notes
        -----
        - This service **fails if motion is actively being executed**.
        - To stop an active motion, cancel the running FollowJointTrajectory action goal instead.
        - This is typically used after motion is complete.
        - Use :func:`MOTOROS2_error_str` to interpret numeric result codes.
        """
        if not self.stop_traj_mode_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("Service StopTrajectoryMode not available")
            return None
        request = Trigger.Request()
        future = self.stop_traj_mode_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self._node.get_clock().now()
        while rclpy.ok() and not future.done():
            sleep(0.01)
            if timeout_sec is not None:
                elapsed = (self._node.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self._node.get_logger().error("Service StopTrajectoryMode call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self._node.get_logger().info(f"Service StopTrajectoryMode: {response.message}")
        except Exception as e:
            self._node.get_logger().error(f"Service StopTrajectoryMode call failed: {e}")
            return None
        return response.success

    # Load
    def SelectToolFromYAML(self, tool_name: Optional[str] = None, tool_yaml_file: Optional[Union[str, Path]] = "tools.yaml") -> None:
        """
        Load tool data from YAML and select the corresponding MotoROS2 tool.

        Parameters
        ----------
        tool_name : str, optional
            Tool name to load from the YAML file.
        tool_yaml_file : str | Path, optional
            Path to the YAML file containing tool definitions.

        Notes
        -----
        After loading RobotBlockSet tool parameters, the selected tool index is
        propagated to MotoROS2 with :meth:`SelectMotionTool`.
        """
        robot.SelectToolFromYAML(self, tool_name=tool_name, tool_yaml_file=tool_yaml_file)
        self.SelectMotionTool(tool=self.Tool.id)

    # Utils
    def ResetError(self, timeout_sec: float = 2.0) -> Optional[int]:
        """
        Attempt to clear robot error and alarm conditions.

        Calls the MotoROS2 ``reset_error`` service to clear recoverable errors.
        Errors requiring physical operator intervention (e.g., e-stop) cannot be reset
        using this method.

        Parameters
        ----------
        timeout_sec : float, optional
            Maximum time to wait for service availability and response.
            Default is ``2.0``.

        Returns
        -------
        int or None
            Returns the `result_code` from :class:`motoros2_interfaces.srv.ResetError`.
            A value of ``1`` typically indicates success (may depend on configuration).
            Returns ``None`` if the service is unavailable, fails, or times out.

        Notes
        -----
        - Use this method after a failed trajectory mode activation.
        - Always check current status via the ``robot_status`` topic after reset.
        - Some errors may persist until manually acknowledged on the teach pendant.
        """
        if not self.reset_error_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("Service ResetError not available")
            return None
        request = ResetError.Request()
        future = self.reset_error_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self._node.get_clock().now()
        while rclpy.ok() and not future.done():
            sleep(0.01)
            if timeout_sec is not None:
                elapsed = (self._node.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self._node.get_logger().error("Service ResetError call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self._node.get_logger().info(f"Service ResetError: {response.message}")
            # motoros2_interfaces.srv.ResetError_Response(result_code=motoros2_interfaces.msg.MotionReadyEnum(value=1), message='success')
        except Exception as e:
            self._node.get_logger().error(f"Service ResetError call failed: {e}")
            return None
        return response.result_code.value

    def SelectMotionTool(self, group: int = 0, tool: int = 0, check_motion: bool = True, timeout_sec: float = 2.0) -> Optional[Tuple[int, bool]]:
        """
        Select the active motion tool on the MotoROS2 controller.

        This method calls the ROS2 service ``SelectMotionTool`` to update the tool
        definition used during robot motion. Tool changes affect safety systems
        (FSU/PFL), payload characteristics (mass, CoG, inertia), and tool interference
        models but **do not change kinematic execution** of joint-space trajectories.

        If ``check_motion`` is enabled, the method waits until the robot has come
        to a complete stop (i.e., no queued motion segments) before invoking the
        service — recommended behavior based on MotoROS2 specifications.

        Parameters
        ----------
        group : int, optional
            motion groups. Default is ``0``.
        tool : int, optional
            Tool selection index (0–63). Default is ``0``.
        check_motion : bool, optional
            If ``True``, waits until the robot is stationary before selecting a tool.
            This ensures that new trajectories are executed using the selected tool.
            Default is ``True``.
        timeout_sec : float, optional
            Timeout for the service request in seconds. Default is ``2.0``.

        Returns
        -------
        tuple of (int, bool) or None
            Returns ``(result_code, success)`` where:
            - ``result_code`` : ``int`` — Selection result (see
            ``SelectionResultCodes``; ``0`` indicates success).
            - ``success`` : ``bool`` — ``True`` if the service invocation succeeded
            on the MotoROS controller, ``False`` otherwise.
            - Returns ``None`` if the service is unavailable, times out, or fails.

        Notes
        -----
        - Tool selection is applied **after currently queued motion completes**.
        - Tool does *not* affect Cartesian execution of joint-space trajectories.
        - Use TF frames to define tool poses for motion planning.
        - Use :func:`MOTOROS2_error_str` to decode ``result_code``.

        Raises
        ------
        ValueError
            If ``group`` or ``tool`` indices are negative or out of range.
        """
        if group < 0:
            self.WarningMessage(f"Group number must be between 0 and total number of defined groups ({group})!")
        if tool < 0 or tool > 63:
            self.WarningMessage(f"Tool number must be between 0 and 63 ({tool})!")
        if not self.select_tool_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("Service SelectMotionTool not available")
            return None

        if check_motion and self.inMotion():
            self.Message("Waiting for motion to be finished before invoking tool selection ...", 2)
            while self.inMotion():
                sleep(0.01)

        request = SelectMotionTool.Request()
        request.group_number = group
        request.tool_number = tool
        future = self.select_tool_client.call_async(request)

        start_time = self._node.get_clock().now()
        while rclpy.ok() and not future.done():
            sleep(0.01)
            if timeout_sec is not None:
                elapsed = (self._node.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self._node.get_logger().error("Service SelectMotionTool call timed out")
                    return None

        try:
            response = future.result()
            self._node.get_logger().info(f"Service SelectMotionTool: {response.message}")
        except Exception as e:
            self._node.get_logger().error(f"Service SelectMotionTool call failed: {e}")
            return None

        return response.result_code.value, response.success

    # Status
    def isReady(self) -> bool:
        """
        Check whether the robot is ready for motion.

        Returns
        -------
        bool
            `True` if the robot is connected and ready for operations, otherwise `False`.
        """
        return self.robot_mode is not None and self.robot_mode.drives_powered.val == 1 and self.robot_mode.in_error.val == 0

    def isActive(self) -> bool:
        """
        Check whether motion is currently possible.

        Returns
        -------
        bool
            `True` if the robot is active.
        """
        return self.robot_mode is not None and self.robot_mode.motion_possible.val == 1

    def inMotion(self) -> bool:
        """
        Check if the robot is in motion.

        Returns
        -------
        bool
            `True` if the robot is in motion.
        """
        return self.robot_mode is not None and self.robot_mode.in_motion.val == 1

    def Check(self, silent: bool = False) -> list[str]:
        """
        Check the current robot status and return active error codes.

        Parameters
        ----------
        silent : bool, optional
            If ``True``, suppress status messages while checking the robot state.

        Returns
        -------
        list[str]
            List of active MotoROS2 error codes converted to strings.

        Notes
        -----
        When ``silent`` is ``False``, the method also reports a detailed
        multi-line robot status through :meth:`Message`.
        """
        _stat = []
        _tmp = f"\nRobot mode: {self.RobotModeStr()}"
        _stat = _stat + [_tmp]
        _tmp = f"Emergency stop: {'ACTIVE' if self.robot_mode.e_stopped.val == 1 else 'INACTIVE'}"
        _stat = _stat + [_tmp]
        _tmp = f"Drives power: {'ON' if self.robot_mode.drives_powered.val == 1 else 'OFF'}"
        _stat = _stat + [_tmp]
        _tmp = f"Motion: {'POSSIBLE' if self.robot_mode.motion_possible.val == 1 else 'NOT POSSIBLE'}"
        _stat = _stat + [_tmp]
        _tmp = f"In motion: {'YES' if self.robot_mode.in_motion.val == 1 else 'NO'}"
        _stat = _stat + [_tmp]
        _tmp = f"Errors: {'YES' if self.robot_mode.in_error.val == 1 else 'NO'}"
        _stat = _stat + [_tmp]
        _str = "\n".join(_stat)
        if not silent:
            self.Message(f"Robot status:{_str}", 1)
        _err = [str(err) for err in self.robot_mode.error_codes.tolist()]
        if not silent and self.robot_mode.in_motion.val == 1:
            self.Message(f"Robot error codes:{_err}", 1)
        return _err

    def HasError(self) -> bool:
        """
        Check whether the robot reports an error state.

        Returns
        -------
        bool
            ``True`` if the robot reports an active error.
        """
        return self.robot_mode.in_error.val != 0

    def RobotModeStr(self) -> str:
        """
        Convert the current robot-mode code to text.

        Returns
        -------
        str
            Human-readable description of the robot mode.
        """
        return self._motoros2_robot_mode_to_str(self.robot_mode.mode.val)

    @staticmethod
    def _motoros2_robot_mode_to_str(mode: int) -> str:
        """
        Convert RobotStatus `mode` value to a readable string.

        Parameters
        ----------
        mode : int
            The RobotMode message `mode` value.

        Returns
        -------
        str
            A human-readable robot mode string.
        """
        MOTOROS2_MODE_MAP = {
            -1: "Unknown",
            1: "Manual",
            2: "Auto",
        }
        return MOTOROS2_MODE_MAP.get(mode, f"Unkonwn mode ({mode})")

    @staticmethod
    def _motoros2_motion_not_ready_code_str(status_code: int) -> str:
        """
        Convert MOTOROS2 FollowJointTrajectory result code to a readable string.

        Parameters
        ----------
        status_code : int
            Error code from trajectory failure.

        Returns
        -------
        str
            Human-readable failure description.
        """
        MOTOROS2_MOTION_NOT_READY_CODE_MAP = {
            1: "Ready",
            100: "Unspecified",
            101: "The controller has an active Alarm",
            102: "The controller has an active Error",
            103: "The controller is in E-Stop",
            104: "The teach pendant must not be in TEACH mode",
            105: "The teach pendant must be in REMOTE mode",
            106: "Servo power is OFF. Please call reset_error or start_point_queue_mode service",
            107: "The controller is in HOLD",
            108: "The INIT_ROS job has not started. Please call start_traj_mode or start_point_queue_mode service",
            109: "INFORM job is not on a WAIT command.SelectMotionTool the format of INIT_ROS",
            111: "A PFL event has occurred. Please return reset_errorobot to a safe state",
            112: "There was an error with the internal motion API",
            113: "Controller is running another job",
            114: "Another trajectory mode is already active. Please call stop_traj_mode service",
        }
        return MOTOROS2_MOTION_NOT_READY_CODE_MAP.get(status_code, f"Unknown status code: {status_code}")

    @staticmethod
    def _motoros2_init_trajectory_status_str(status_code: int) -> str:
        """
        Convert MotoROS2 trajectory-initialization status codes to text.

        Parameters
        ----------
        status_code : int
            MotoROS2 trajectory-initialization status code.

        Returns
        -------
        str
            Human-readable description of the initialization status.
        """
        INIT_TRAJ_STATUS_MAP = {
            0: "OK",
            200: "Unspecified failure",
            201: "Trajectory must contain at least two points.",
            202: "Trajectory contains too many points (Not enough memory).",
            203: "Already running a trajectory.",
            204: "The first point must match the robot's current position.",
            205: "The commanded velocity is too high.",
            206: "Invalid joint name specified. Check motoros2_config.yaml.",
            207: "Trajectory must contain data for all joints.",
            208: "Invalid time in trajectory.",
            209: "Must call start_traj_mode service",
            210: "Trajectory message contains waypoints that are not strictly increasing in time.",
            211: "Trajectory did not contain position data for all axes.",
            212: "Trajectory did not contain velocity data for all axes.",
            213: "The final point in the trajectory must have zero velocity.",
            214: "The final point in the trajectory must have zero acceleration.",
            215: "The trajectory contains duplicate joint names.",
        }

        return INIT_TRAJ_STATUS_MAP.get(status_code, f"Unknown init trajectory status code: {status_code}")

    @staticmethod
    def _motoros2_failed_trajectory_status_str(error_code: int) -> str:
        """
        Convert MOTOROS2 Failed_Trajectory_Status error code to a readable string.

        Parameters
        ----------
        error_code : int
            Error code from trajectory failure.

        Returns
        -------
        str
            Human-readable failure description.
        """
        FAILED_TRAJECTORY_STATUS = {
            300: "Goal was cancelled by the user.",
            301: "Final position was outside tolerance. Check robot safety-limits that could be inhibiting motion.",
            302: "Execution time was outside tolerance.",
            303: "Goal was aborted due to an alarm or error (check RobotStatus messages)",
            304: "Parsing goal_tolerance failed. Failing trajectory execution.",
        }

        return FAILED_TRAJECTORY_STATUS.get(error_code, f"Unknown trajectory failure code: {error_code}")

    @staticmethod
    def _motoros2_selection_result_code_to_str(code: int) -> str:
        """
        Convert SelectionResultCodes result value to a human-readable string.

        Parameters
        ----------
        code : int
            `motoros2_interfaces/msg/SelectionResultCodes`.

        Returns
        -------
        str
            Readable description of the result code. Returns an informative
            fallback message if the code is not recognized.

        Notes
        -----
        Reference values (from SelectionResultCodes.msg):
            - OK = 0 → Request completed successfully.
            - INVALID_CONTROLLER_STATE = 400 → Pendant not in REMOTE mode.
            - INVALID_CONTROL_GROUP = 401 → Invalid control group.
            - INVALID_SELECTION_INDEX = 402 → Invalid tool selection.
        """
        SELECTION_RESULT_MAP = {
            0: "OK — request completed successfully.",
            400: "Invalid controller state — Robot teach pendant is not in REMOTE mode.",
            401: "Invalid control group — Provided group ID is unsupported.",
            402: "Invalid selection index — Tool or configuration selection is incorrect.",
        }

        return SELECTION_RESULT_MAP.get(code, f"Unknown selection result code ({code})")

    @staticmethod
    def MOTOROS2_error_str(code: int) -> str:
        """
        Convert a MotoROS2 status code to a human-readable description.

        Parameters
        ----------
        code : int
            The `error code` value.

        Returns
        -------
        str
            Human-readable description of the status code.
        """
        if code is None:
            return "No code!"

        # Handle success case
        if code == 0:
            return "Trajectory execution successful"

        if code < 200:
            _msg = hc20._motoros2_motion_not_ready_code_str(code)
        elif code < 300:
            _msg = hc20._motoros2_init_trajectory_status_str(code)
        elif code < 400:
            _msg = hc20._motoros2_failed_trajectory_status_str(code)
        else:
            _msg = hc20._motoros2_selection_result_code_to_str(code)

        return _msg

    @staticmethod
    def MotionResultStr(result: int) -> str:
        """
        Convert an extended MotoROS2 trajectory result code to text.

        Extended format: -ECCCCC
        E (first digit)     = standard ROS FollowJointTrajectory result code
        CCCCC (next digits) = MOTOROS2 extended status code

        Parameters
        ----------
        result : int
            The FollowJointTrajectory message `result code` value.

        Returns
        -------
        str
            A human-readable result code string.
        """
        if result is None:
            return "No result!"

        # Handle success case
        if result == 0:
            return "Trajectory execution successful"

        if result > 0:
            return robot.MotionResultStr(result)

        # Decode -ECCCCC → split ROS2 code (E) and MOTORS2 code (CCCCC)
        standard_code = -(abs(result) // 100000)  # integer division
        extended_code = abs(result) % 100000  # last 5 digits

        standard_msg = JointTrajectoryControllerInterface.FollowJointTrajectory_error_str(standard_code)
        if extended_code < 200:
            extended_msg = hc20._motoros2_motion_not_ready_code_str(extended_code)
        elif extended_code < 300:
            extended_msg = hc20._motoros2_init_trajectory_status_str(extended_code)
        elif extended_code < 400:
            extended_msg = hc20._motoros2_failed_trajectory_status_str(extended_code)
        else:
            extended_msg = hc20._motoros2_selection_result_code_to_str(extended_code)

        return f"{standard_msg} | MOTOROS2: {extended_msg} (raw={result})"

class hc20(yaskawa_ros2, hc20_spec):
    """Robot wrapper for the Yaskawa HC20 manipulator."""

    def __init__(self, robot_name: str = "hc20", namespace: str = "yaskawa", tof_topic: Optional[str] = None) -> None:
        """Create an HC20 robot in ROS2.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in ROS2.
        namespace : str, optional
            Namespace for all robot topics and services. Determines ROS namespace prefix.
            Default is ``"yaskawa"``.
        tof_topic : str, optional
            Optional topic used to monitor the TOF breach state.
        **kwargs : Any
            Additional keyword arguments passed to `yaskawa_ros2`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        hc20_spec.__init__(self)
        yaskawa_ros2.__init__(self, robot_name, namespace=namespace, tof_topic=tof_topic)


class hc30(yaskawa_ros2, hc30_spec):
    """Robot wrapper for the Yaskawa HC30 manipulator."""

    def __init__(self, robot_name: str = "hc30", namespace: str = "yaskawa", tof_topic: Optional[str] = None) -> None:
        """Create an HC30 robot in ROS2.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in ROS2.
        namespace : str, optional
            Namespace for all robot topics and services. Determines ROS namespace prefix.
            Default is ``"yaskawa"``.
        tof_topic : str, optional
            Optional topic used to monitor the TOF breach state.
        **kwargs : Any
            Additional keyword arguments passed to `yaskawa_ros2`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        hc30_spec.__init__(self)
        yaskawa_ros2.__init__(self, robot_name, namespace=namespace, tof_topic=tof_topic)


