"""Universal Robots ROS 2 interface implementations.

This module defines Universal Robots ROS 2-backed robot wrappers used by RobotBlockSet robot
interfaces. It provides communication plumbing, controller integration, state
feedback handling, and motion-execution helpers for multiple robot platforms.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import List

from robotblockset.ros2.controllers_ros2 import JointTrajectoryControllerInterface
from robotblockset.ros2.robots_ros2 import robot_ros2
from robotblockset.robot_spec import ur10_spec, ur10e_spec

try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from geometry_msgs.msg import WrenchStamped
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if all are installed.")


# Check for UR specific message imports
try:
    from ur_dashboard_msgs.msg import RobotMode, SafetyMode
except ImportError:
    RobotMode = None
    SafetyMode = None


class ur_ros2(robot_ros2):
    """
    Generic Universal Robots ROS 2 interface.

    Attributes
    ----------
    robot_mode : RobotMode | None
        Last robot-mode message received from the dashboard interface.
    safety_mode : SafetyMode | None
        Last safety-mode message received from the dashboard interface.
    """

    def __init__(self, name: str = "ur10", namespace: str = "", control_strategy: str = "JointPositionTrajectory") -> None:
        """
        Initialize a ROS 2 Universal Robots interface.

        Parameters
        ----------
        name : str, optional
            Name of the robot node.
        namespace : str, optional
            Namespace for the robot topics, services, and actions.
        control_strategy : str, optional
            Initial RobotBlockSet control strategy.

        Notes
        -----
        The constructor configures the joint-trajectory controller interface,
        subscribes to wrench, robot-mode, and safety-mode topics, starts the
        ROS 2 executor thread, and initializes the robot state.
        """
        ur10_spec.__init__(self)
        # Initialize robot base class
        joint_position_trajectory_controller = JointTrajectoryControllerInterface(namespace=namespace, ros_plugin_name="scaled_joint_trajectory_controller", topic="joint_trajectory", action="follow_joint_trajectory")
        robot_ros2.__init__(
            self,
            name=name,
            namespace=namespace,
            strategy_to_controller_interface_mapping={"JointPositionTrajectory": joint_position_trajectory_controller},
            joint_states_topic=f"{namespace}/joint_states",
            control_strategy=control_strategy,
        )

        if RobotMode is None or SafetyMode is None:
            raise ImportError("ur_dashboard_msgs package not found. Robot in ur10 ROS2 class will not be available.")

        # Initialize ROS2 subscribers
        self.force_state_subscription = self._node.create_subscription(WrenchStamped, f"{self._namespace}/force_torque_sensor_broadcaster/wrench", self._force_state_callback, qos_profile_sensor_data)
        self._node.get_logger().info(f"Subscribing to topic: {self._namespace}/force_torque_sensor_broadcaster/wrench")

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.robot_mode_subscription = self._node.create_subscription(RobotMode, f"{self._namespace}/io_and_status_controller/robot_mode", self._robot_mode_callback, qos)
        self._node.get_logger().info(f"Subscribing to topic: {self._namespace}/io_and_status_controller/robot_mode")
        self.robot_mode = None
        self.last_robot_mode_callback_time = 0
        self.safety_mode_subscription = self._node.create_subscription(SafetyMode, f"{self._namespace}/io_and_status_controller/safety_mode", self._safety_mode_callback, qos)
        self._node.get_logger().info(f"Subscribing to topic: {self._namespace}/io_and_status_controller/safety_mode")
        self.safety_mode = None
        self.last_safety_mode_callback_time = 0

        # Start spinning only after all publishers/subscribers/clients are created
        self._start_spinning(wait_for_state=True)

        self.Init()
        self.Message("Initialized", 2)

    # States
    def _robot_mode_callback(self, msg: RobotMode) -> None:  # pyright: ignore[reportInvalidTypeForm]
        """
        Store the latest robot-mode message.

        Parameters
        ----------
        msg : RobotMode
            Robot mode message received from the dashboard topic.
        """
        self.robot_mode = msg
        self.last_robot_mode_callback_time = self.simtime()

    def _safety_mode_callback(self, msg: SafetyMode) -> None:  # pyright: ignore[reportInvalidTypeForm]
        """
        Store the latest safety-mode message.

        Parameters
        ----------
        msg : SafetyMode
            Safety mode message received from the dashboard topic.
        """
        self.safety_mode = msg
        self.last_safety_mode_callback_time = self.simtime()

    # Status
    def isReady(self) -> bool:
        """
        Check whether the robot is ready for motion.

        Returns
        -------
        bool
            ``True`` if the safety mode is ``NORMAL``, otherwise ``False``.

        Notes
        -----
        This readiness check reflects the reported safety mode and does not
        guarantee that the motion controller is active.
        """
        return self.safety_mode.mode == 1

    def inMotion(self) -> bool:
        """
        Check whether the robot reports running motion.

        Returns
        -------
        bool
            ``True`` if the robot-mode state is ``RUNNING``.
        """
        return self.robot_mode.mode == 7

    def Check(self, silent: bool = False) -> List[str]:
        """
        Check the robot status and return active issues.

        Parameters
        ----------
        silent : bool, optional
            If ``True``, suppress status messages while checking the robot state.

        Returns
        -------
        list[str]
            Status strings describing conditions that prevent normal operation.

        Notes
        -----
        The returned list is empty when the robot is ready and active.
        """
        _err = []
        if not self.isReady():
            _err = [f"\nSafety mode {self.SafetyModeStr()}"]
        if not self.isActive():
            _err = _err + [f"\nRobot mode {self.RobotModeStr()}"]
        if len(_err) > 0:
            if not silent:
                self.Message("Robot not ready/active: " + ", ".join(_err), 0)
        elif not silent:
            self.Message("Robot ready and active", 2)
        return _err

    def RobotModeStr(self) -> str:
        """
        Convert the current UR robot-mode code to text.

        Returns
        -------
        str
            Human-readable description of the robot mode.
        """
        mode_table = {
            -1: "NO_CONTROLLER",
            0: "DISCONNECTED",
            1: "CONFIRM_SAFETY",
            2: "BOOTING",
            3: "POWER_OFF",
            7: "RUNNING",
            8: "UPDATING_FIRMWARE",
        }

        mode = self.robot_mode.mode if self.robot_mode else None
        return mode_table.get(mode, f"UNKNOWN_MODE ({mode})")

    def SafetyModeStr(self) -> str:
        """
        Convert the current UR safety-mode code to text.

        Returns
        -------
        str
            Human-readable description of the safety mode.
        """
        safety_table = {
            1: "NORMAL",
            2: "REDUCED",
            3: "PROTECTIVE_STOP",
            4: "RECOVERY",
            5: "SAFEGUARD_STOP",
            6: "SYSTEM_EMERGENCY_STOP",
            7: "ROBOT_EMERGENCY_STOP",
            8: "VIOLATION",
            9: "FAULT",
            10: "VALIDATE_JOINT_ID",
            11: "UNDEFINED_SAFETY_MODE",
            12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
            13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
        }

        safety_mode = self.safety_mode.mode if self.safety_mode else None
        return safety_table.get(safety_mode, f"UNKNOWN_SAFETY_MODE ({safety_mode})")


class ur10e(ur_ros2, ur10e_spec):
    """
    ROS 2 interface for the Universal Robots UR10e platform.

    Attributes
    ----------
    All attributes from :class:`ur_ros2` and :class:`ur10e_spec`.
    """

    def __init__(self, name: str = "ur10e", namespace: str = "", control_strategy: str = "JointPositionTrajectory") -> None:
        """
        Initialize the ROS 2 UR10e robot wrapper.

        Parameters
        ----------
        name : str, optional
            Name of the robot node.
        namespace : str, optional
            Namespace for the robot topics, services, and actions.
        control_strategy : str, optional
            Initial RobotBlockSet control strategy.
        """
        ur10e_spec.__init__(self)
        # Initialize robot base class
        ur_ros2.__init__(self, name=name, namespace=namespace, control_strategy=control_strategy)


class ur10(ur_ros2, ur10e_spec):
    """
    ROS 2 interface for the Universal Robots UR10 platform.

    Attributes
    ----------
    All attributes from :class:`ur_ros2` and the selected robot specification.
    """

    def __init__(self, name: str = "ur10e", namespace: str = "", control_strategy: str = "JointPositionTrajectory") -> None:
        """
        Initialize the ROS 2 UR10 robot wrapper.

        Parameters
        ----------
        name : str, optional
            Name of the robot node.
        namespace : str, optional
            Namespace for the robot topics, services, and actions.
        control_strategy : str, optional
            Initial RobotBlockSet control strategy.
        """
        ur10e_spec.__init__(self)
        # Initialize robot base class
        ur_ros2.__init__(self, name=name, namespace=namespace, control_strategy=control_strategy)
