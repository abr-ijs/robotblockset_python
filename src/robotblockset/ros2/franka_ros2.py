"""ROS 2 Franka robot interfaces.

This module defines ROS 2-backed interfaces for Franka robots. It provides
support for joint-trajectory control, Cartesian impedance control, wrench
feedback, load configuration, and TCP updates for real or simulated systems.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic.
"""

from __future__ import annotations

import numpy as np
from time import sleep
from typing import Any

from robotblockset.tools import check_option, isscalar, isvector, vector, matrix
import rclpy
from robotblockset.robot_spec import fr3_spec
from robotblockset.rbs_typing import ArrayLike
from robotblockset.transformations import map_pose
from geometry_msgs.msg import WrenchStamped
from robotblockset.ros2.controllers_ros2 import CsfCartesianImpedanceControllerInterface, JointPositionControllerInterface, JointTrajectoryControllerInterface
from robotblockset.ros2.robots_ros2 import robot_ros2
from franka_msgs.srv import SetLoad, SetTCPFrame, SetFullCollisionBehavior

from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from franka_msgs.action import ErrorRecovery as ErrorRecoveryAction
import time


class fr3(robot_ros2, fr3_spec):
    def __init__(self, name: str = "fr3", ns: str = "", control_strategy: str = "CartesianImpedance", SIM: bool = False) -> None:
        """Initialize the ROS 2 Franka FR3 robot wrapper."""
        # Initialize specification (kinematics, joint names, etc.)
        fr3_spec.__init__(self)
        self.joint_names = [f"{name}_joint{i+1}" for i in range(self.nj)]

        # Initialize interfaces for available control strategies
        cartesian_impedance_controller = CsfCartesianImpedanceControllerInterface(
            ros_plugin_name="cartesian_impedance_controller",
            topic="cartesian_command",
            namespace=ns,
            Kp=np.array([2000.0, 2000.0, 2000.0]),
            Kr=np.array([30.0, 30.0, 30.0]),
            R=np.eye(3),
            D=2.0,
            command_frame=f"{name}_link8",
        )

        joint_position_trajectory_controller = JointTrajectoryControllerInterface(ros_plugin_name="fr3_arm_controller", topic="joint_trajectory", action="follow_joint_trajectory", namespace=ns)
        joint_position_controller = JointPositionControllerInterface(ros_plugin_name="fr3_arm_controller", topic="joint_trajectory", namespace=ns)

        # Initialize robot base class
        robot_ros2.__init__(
            self,
            name=name,
            namespace=ns,
            strategy_to_controller_interface_mapping={
                "CartesianImpedance": cartesian_impedance_controller,
                "JointPositionTrajectory": joint_position_trajectory_controller,
                "JointPosition": joint_position_controller,
            },
            joint_states_topic=f"{ns}/joint_states",
            control_strategy=control_strategy,
        )

        self.SIM = SIM
        
        # Initialize collision behavior tracking
        self._collision_behavior = None

        if not self.SIM:
            # Add wrench state subscription
            self.force_state_subscription = self._node.create_subscription(msg_type=WrenchStamped, topic=f"{self._namespace}/franka_robot_state_broadcaster/external_wrench_in_base_frame", callback=self._force_state_callback, qos_profile=qos_profile_sensor_data)
            # Create EE load service client
            self._set_load_client = self._node.create_client(SetLoad, f"{self._namespace}/service_server/set_load")
            while not self._set_load_client.wait_for_service(timeout_sec=1.0):
                self.Message(f"Service {self._namespace}/service_server/set_load not available, waiting...", 1)

            # Set TCP frame service client
            self._set_tcp_frame_client = self._node.create_client(SetTCPFrame, f"{self._namespace}/service_server/set_tcp_frame")
            while not self._set_tcp_frame_client.wait_for_service(timeout_sec=1.0):
                self.Message(f"Service {self._namespace}/service_server/set_tcp_frame not available, waiting...", 1)

            # Collision behavior service client
            self._set_collision_behavior_client = self._node.create_client(SetFullCollisionBehavior, f"{self._namespace}/service_server/set_full_collision_behavior")
            while not self._set_collision_behavior_client.wait_for_service(timeout_sec=1.0):
                self.Message(f"Service {self._namespace}/service_server/set_full_collision_behavior not available, waiting...", 1)

            # Error recovery action client
            self._error_recovery_client = ActionClient(self._node, ErrorRecoveryAction, f"{self._namespace}/action_server/error_recovery")
            while not self._error_recovery_client.wait_for_server(timeout_sec=1.0):
                self.Message(f"Action server {self._namespace}/action_server/error_recovery not available, waiting...", 1)

        # Start spinning only after all publishers/subscribers/clients are created
        self._start_spinning(wait_for_state=True)

        # Control strategy (if provided) is applied by _start_spinning via _desired_strategy

        # Finalize robot state
        self.Init()
        self.Message("Initialized", 2)

    def Check(self, silent: bool = False) -> list:
        """Return the current list of detected robot issues."""
        return []

    def shutdown(self) -> None:
        """Shut down the ROS 2 robot wrapper and its background spinner."""
        self.Shutdown()

    def GetFullCollisionBehavior(self) -> dict | None:
        """Get the current collision behavior thresholds tracked internally.

        Returns the collision behavior that was set via ``SetFullCollisionBehavior``.
        Does not query the robot, so only returns values that have been set through
        this interface. If collision behavior has not been set internally, returns ``None``.

        Returns
        -------
        dict or None
            Dictionary with keys:
            - ``'lower_torque_acc'``: Contact torque thresholds during acceleration (list of 7 floats)
            - ``'upper_torque_acc'``: Collision torque thresholds during acceleration (list of 7 floats)
            - ``'lower_torque_nom'``: Contact torque thresholds during constant-velocity (list of 7 floats)
            - ``'upper_torque_nom'``: Collision torque thresholds during constant-velocity (list of 7 floats)
            - ``'lower_force_acc'``: Contact force/torque thresholds during acceleration (list of 6 floats)
            - ``'upper_force_acc'``: Collision force/torque thresholds during acceleration (list of 6 floats)
            - ``'lower_force_nom'``: Contact force/torque thresholds during constant-velocity (list of 6 floats)
            - ``'upper_force_nom'``: Collision force/torque thresholds during constant-velocity (list of 6 floats)
            
            Returns ``None`` if collision behavior has not been set via ``SetFullCollisionBehavior``.
            Note: This only tracks values set through this interface; values set externally via ROS
            are not reflected here.
        """
        if self._collision_behavior is None:
            self.Message("GetFullCollisionBehavior: No collision behavior set internally", 1)
            return None
        return self._collision_behavior

    def SetFullCollisionBehavior(
        self,
        *,
        lower_torque_acc: ArrayLike = (25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0),
        upper_torque_acc: ArrayLike = (35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0),
        lower_torque_nom: ArrayLike = (25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0),
        upper_torque_nom: ArrayLike = (35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0),
        lower_force_acc: ArrayLike = (30.0, 30.0, 30.0, 25.0, 25.0, 25.0),
        upper_force_acc: ArrayLike = (40.0, 40.0, 40.0, 35.0, 35.0, 35.0),
        lower_force_nom: ArrayLike = (30.0, 30.0, 30.0, 25.0, 25.0, 25.0),
        upper_force_nom: ArrayLike = (40.0, 40.0, 40.0, 35.0, 35.0, 35.0),
    ) -> int:
        """Set separate torque and force collision thresholds for acceleration and nominal phases.

        Wraps ``franka::Robot::setCollisionBehavior`` (8-argument overload) via the
        ROS 2 service ``/service_server/set_full_collision_behavior``.

        Forces or torques between the *lower* and *upper* threshold are reported as
        **contacts** in ``RobotState``.  Values that exceed the *upper* threshold are
        registered as a **collision** and cause the robot to stop moving.
        Call ``automaticErrorRecovery`` (or the equivalent ROS 2 service) to resume
        after a collision.

        Parameters
        ----------
        lower_torque_acc : ArrayLike, length 7
            Contact torque thresholds during acceleration/deceleration for each
            joint in Nm.  Default: ``(25, 25, 22, 20, 19, 17, 14)``.
        upper_torque_acc : ArrayLike, length 7
            Collision torque thresholds during acceleration/deceleration for each
            joint in Nm.  Default: ``(35, 35, 32, 30, 29, 27, 24)``.
        lower_torque_nom : ArrayLike, length 7
            Contact torque thresholds during constant-velocity motion for each
            joint in Nm.  Default: ``(25, 25, 22, 20, 19, 17, 14)``.
        upper_torque_nom : ArrayLike, length 7
            Collision torque thresholds during constant-velocity motion for each
            joint in Nm.  Default: ``(35, 35, 32, 30, 29, 27, 24)``.
        lower_force_acc : ArrayLike, length 6
            Contact force/torque thresholds during acceleration/deceleration for
            ``(x, y, z)`` in N and ``(R, P, Y)`` in Nm.
            Default: ``(30, 30, 30, 25, 25, 25)``.
        upper_force_acc : ArrayLike, length 6
            Collision force/torque thresholds during acceleration/deceleration for
            ``(x, y, z)`` in N and ``(R, P, Y)`` in Nm.
            Default: ``(40, 40, 40, 35, 35, 35)``.
        lower_force_nom : ArrayLike, length 6
            Contact force/torque thresholds during constant-velocity motion for
            ``(x, y, z)`` in N and ``(R, P, Y)`` in Nm.
            Default: ``(30, 30, 30, 25, 25, 25)``.
        upper_force_nom : ArrayLike, length 6
            Collision force/torque thresholds during constant-velocity motion for
            ``(x, y, z)`` in N and ``(R, P, Y)`` in Nm.
            Default: ``(40, 40, 40, 35, 35, 35)``.

        Returns
        -------
        int
            ``0`` on success, ``-1`` on failure.

        Raises
        ------
        RuntimeError
            If called in simulation mode.
        ValueError
            If a threshold array does not have the required length.

        See Also
        --------
        https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html
        """

        if self.SIM:
            self.Message("SetFullCollisionBehavior: Not available in SIM mode", 1)
            return 0

        def _AsFloatList(x: ArrayLike, n: int, name: str) -> list[float]:
            vals = list(x)
            if len(vals) != n:
                raise ValueError(f"{name} must have length {n}, got {len(vals)}")
            return [float(v) for v in vals]

        request = SetFullCollisionBehavior.Request()
        request.lower_torque_thresholds_acceleration = _AsFloatList(lower_torque_acc, 7, "lower_torque_acc")
        request.upper_torque_thresholds_acceleration = _AsFloatList(upper_torque_acc, 7, "upper_torque_acc")
        request.lower_torque_thresholds_nominal = _AsFloatList(lower_torque_nom, 7, "lower_torque_nom")
        request.upper_torque_thresholds_nominal = _AsFloatList(upper_torque_nom, 7, "upper_torque_nom")
        request.lower_force_thresholds_acceleration = _AsFloatList(lower_force_acc, 6, "lower_force_acc")
        request.upper_force_thresholds_acceleration = _AsFloatList(upper_force_acc, 6, "upper_force_acc")
        request.lower_force_thresholds_nominal = _AsFloatList(lower_force_nom, 6, "lower_force_nom")
        request.upper_force_thresholds_nominal = _AsFloatList(upper_force_nom, 6, "upper_force_nom")

        self._control_helper.deactivate(self.controller._ros_plugin_name)
        future = self._set_collision_behavior_client.call_async(request)
        while rclpy.ok() and not future.done():
            sleep(0.01)
        self._control_helper.activate(self.controller._ros_plugin_name)
        if future.done() and future.result() is not None:
            # Store the collision behavior state internally
            self._collision_behavior = {
                'lower_torque_acc': request.lower_torque_thresholds_acceleration,
                'upper_torque_acc': request.upper_torque_thresholds_acceleration,
                'lower_torque_nom': request.lower_torque_thresholds_nominal,
                'upper_torque_nom': request.upper_torque_thresholds_nominal,
                'lower_force_acc': request.lower_force_thresholds_acceleration,
                'upper_force_acc': request.upper_force_thresholds_acceleration,
                'lower_force_nom': request.lower_force_thresholds_nominal,
                'upper_force_nom': request.upper_force_thresholds_nominal,
            }
            self.Message("SetFullCollisionBehavior: Collision behavior updated", 2)
            return 0
        else:
            self.Message(f"SetFullCollisionBehavior: Service call failed {future.exception()}", 0)
            return -1

    def ErrorRecovery(self) -> int:
        """Trigger automatic error recovery on the robot.

        Sends an ``ErrorRecovery`` goal to ``/action_server/error_recovery``
        to reset the robot after a collision or error has been detected.
        Equivalent to ``franka::Robot::automaticErrorRecovery``.

        Returns
        -------
        int
            ``0`` on success, ``-1`` on failure.
        """
        if self.SIM:
            self.Message("ErrorRecovery: Not available in SIM mode", 1)
            return -1
        
        goal_handle_future = self._error_recovery_client.send_goal_async(ErrorRecoveryAction.Goal())
        while rclpy.ok() and not goal_handle_future.done():
            sleep(0.01)
        if not goal_handle_future.done() or goal_handle_future.result() is None:
            self.Message("ErrorRecovery: Goal rejected or server unavailable", 0)
            return -1
        goal_handle = goal_handle_future.result()
        if not goal_handle.accepted:
            self.Message("ErrorRecovery: Goal rejected by action server", 0)
            return -1
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            sleep(0.01)
        if result_future.done() and result_future.result() is not None:
            self.Message("ErrorRecovery: Robot error recovered", 2)
            
            # Hardware may be in unconfigured state after error recovery.
            # Properly configure and activate the hardware interface.
            if not self._control_helper.configure_hardware("FrankaHardwareInterface"):
                self.Message("ErrorRecovery: Failed to configure hardware", 0)
                return -1
            self.Message("ErrorRecovery: Hardware configured", 2)
            
            if not self._control_helper.activate_hardware("FrankaHardwareInterface"):
                self.Message("ErrorRecovery: Failed to activate hardware", 0)
                return -1
            self.Message("ErrorRecovery: Hardware activated", 2)

            # Ensure controller is operational after hardware reactivation.
            ctrl_name = self.controller._ros_plugin_name
            ctrl_state = self._control_helper.get_state(ctrl_name)
            if ctrl_state == "active":
                # Force a restart cycle now that command interfaces are available.
                if not self._control_helper.deactivate(ctrl_name):
                    self.Message("ErrorRecovery: Failed to deactivate controller after hardware activation", 0)
                    return -1
                if not self._control_helper.activate(ctrl_name):
                    self.Message("ErrorRecovery: Failed to reactivate controller after restart", 0)
                    return -1
            else:
                if not self._control_helper.activate(ctrl_name):
                    self.Message("ErrorRecovery: Failed to activate controller", 0)
                    return -1
            self.Message("ErrorRecovery: Controller ready", 2)
            
            self.Message("ErrorRecovery: Recovery complete", 2)
            return 0
        else:
            self.Message(f"ErrorRecovery: Failed {result_future.exception()}", 0)
            return -1

    def SetLoad(self, mass: float, COM: tuple = [0, 0, 0], inertia: tuple = None) -> int:
        """Update the load configuration on the physical robot."""

        if self.SIM:
            self.Message("SetLoad: Not available in SIM mode", 1)
            return 0

        if (not isscalar(mass)) and (mass <= 0):
            raise ValueError("Mass must be scalar > 0")
        COM = vector(COM, dim=3)
        inertia = matrix(inertia, shape=(3, 3))

        request = SetLoad.Request()
        request.mass = mass
        request.center_of_mass = COM
        # request.load_inertia = inertia.flatten(order='F').tolist()  # column-major flatten

        self._control_helper.deactivate(self.controller._ros_plugin_name)
        future = self._set_load_client.call_async(request)
        while rclpy.ok() and not future.done():
            sleep(0.01)
        self._control_helper.activate(self.controller._ros_plugin_name)
        if future.done() and future.result() is not None:
            self.Message(f"SetLoad: Load set to mass={mass}, COM={COM}, inertia={inertia}", 2)
            return 0
        else:
            self.Message(f"SetLoad: Service call failed {future.exception()}", 0)
            return -1

    def SetTCP(self, *tcp: np.ndarray, frame: str = "Gripper", send_to_robot: bool = True, EE_frame: str = "Flage") -> int:
        """Set the TCP locally and optionally forward it to the robot service."""
        if len(tcp) > 0:
            x = self.spatial(tcp[0])
            if x.shape == (4, 4):
                _tcp = x
            elif x.shape == (3, 3):
                _tcp = map_pose(R=x, out="T")
            elif isvector(x, dim=7):
                _tcp = map_pose(x=x, out="T")
            elif isvector(x, dim=3):
                _tcp = map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                _tcp = map_pose(Q=x, out="T")
            else:
                raise ValueError(f"TCP shape {x.shape} not supported")
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
        self._command.v = srJ @ self._command.qdot

        if send_to_robot:
            return self._set_tcp_frame(newTCP)

        self.GetState()
        self.Update()
        return 0

    def _set_tcp_frame(self, transformation: np.ndarray = np.eye(4)) -> int:
        """Send a TCP frame update request to the physical robot."""

        if self.SIM:
            self.Message("SetTCPFrame: Not available in SIM mode", 1)
            return 0

        transformation = matrix(transformation, shape=(4, 4))

        request = SetTCPFrame.Request()
        request.tcp_frame = transformation.flatten(order="F").tolist()  # column-major flatten

        self._control_helper.deactivate(self.controller._ros_plugin_name)
        future = self._set_tcp_frame_client.call_async(request)
        while rclpy.ok() and not future.done():
            sleep(0.01)
        self._control_helper.activate(self.controller._ros_plugin_name)
        if future.done() and future.result() is not None:
            self.Message(f"SetTCPFrame: TCP frame set to \n{transformation}", 2)
            return 0
        else:
            self.Message(f"SetTCPFrame: Service call failed {future.exception()}", 0)
            return 1


if __name__ == "__main__":

    # Example usage of the fr3 robot class with ROS2 Cartesian controller and joint trajectory controller.
    rclpy.init()

    r = fr3(SIM=True)

    print("Robot:", r.Name)
    print("q: ", r.q)
    print("x: ", r.x)

    print("Change to joint-space control")
    r.SetStrategy("JointPositionTrajectory")
    r.JMove(r.q_home)

    start_time = time.time()
    r.CMove(r.x, 2)
    print("command duration: {:.2f} seconds".format(time.time() - start_time))
    print("q: ", r.q)
    print("x: ", r.x)

    start_time = time.time()
    r.CMoveFor([0, 0, -0.05], 2)
    print("command duration: {:.2f} seconds".format(time.time() - start_time))

    r.SetStrategy("CartesianImpedance")
    start_time = time.time()
    r.CMoveFor([0, 0, 0.05], 2)
    print("command duration: {:.2f} seconds".format(time.time() - start_time))

    print("Returned to original position.")

    r.shutdown()
    rclpy.shutdown()
