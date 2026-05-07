"""ROS 2 KUKA iiwa robot interfaces.

This module defines ROS 2-backed interfaces for KUKA iiwa robots. It provides
support for Cartesian impedance control, joint-trajectory control, and wrench
feedback integration through ROS 2 controllers and topics.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic, Leon Zlajpah
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np

try:
    from rclpy.qos import qos_profile_sensor_data
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

from robotblockset.robot_spec import iiwa_spec
from robotblockset.ros2.controllers_ros2 import (
    CsfCartesianImpedanceControllerInterface,
    JointTrajectoryControllerInterface,
    JointPositionControllerInterface,
)
from robotblockset.ros2.robots_ros2 import robot_ros2

try:
    from geometry_msgs.msg import WrenchStamped
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if all are installed.")


class iiwa(robot_ros2, iiwa_spec):
    def __init__(
        self,
        name: str = "iiwa",
        ns: str = "lbr",
        control_strategy: str = "JointPositionTrajectory",
    ) -> None:
        """Initialize the ROS 2 iiwa7 robot wrapper."""
        # Initialize specification (kinematics, joint names, limits, ...)
        iiwa_spec.__init__(self)

        # Build the controller interfaces for the supported control strategies. These are passed
        # to the robot_ros2 base class, which will use them to  create the controller_manager_helper
        # and to route strategy changes to the correct controller interface.
        cartesian_impedance_controller = CsfCartesianImpedanceControllerInterface(
            ros_plugin_name="cartesian_impedance_controller",
            topic="cartesian_command",
            namespace=ns,
            Kp=np.array([200.0, 200.0, 200.0]),
            Kr=np.array([10.0, 10.0, 10.0]),
            R=np.eye(3),
            D=2.0,
        )

        joint_position_trajectory_controller = JointTrajectoryControllerInterface(
            ros_plugin_name="joint_trajectory_controller",
            topic="joint_trajectory",
            action="follow_joint_trajectory",
            namespace=ns,
        )

        joint_position_controller = JointPositionControllerInterface(
            ros_plugin_name="lbr_joint_position_command_controller",
            topic="joint_position",
            namespace=ns,
        )

        strategy_mapping = {
            "CartesianImpedance": cartesian_impedance_controller,
            "JointPositionTrajectory": joint_position_trajectory_controller,
            "JointPosition": joint_position_controller,
        }

        robot_ros2.__init__(
            self,
            name=name,
            namespace=ns,
            strategy_to_controller_interface_mapping=strategy_mapping,
            joint_states_topic="joint_states",
            control_strategy=control_strategy,
        )

        # Optional wrench state subscription
        self.force_state_subscription = self._node.create_subscription(
            msg_type=WrenchStamped,
            topic=f"{self._namespace}/force_torque_sensor_broadcaster/wrench",
            callback=self._force_state_callback,
            qos_profile=qos_profile_sensor_data,
        )

        # Start spinning only after all publishers/subscribers/clients are created.
        # _start_spinning will call SetStrategy(control_strategy) using the correct helper.
        self._start_spinning(wait_for_state=True)

        # Finalize robot state
        self.Init()

        self.Message("Initialized", 2)

    def Check(self, silent: bool = False) -> list:
        """Return the current list of detected robot issues."""
        return []

    def shutdown(self) -> None:
        """Shut down the ROS 2 robot wrapper and its background spinner."""
        self.Shutdown()
