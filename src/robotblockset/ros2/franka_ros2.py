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

from robotblockset.tools import check_option, isscalar, isvector, vector, matrix
import rclpy
from robotblockset.robot_spec import fr3_spec
from robotblockset.transformations import map_pose
from geometry_msgs.msg import WrenchStamped
from robotblockset.ros2.controllers_ros2 import CsfCartesianImpedanceControllerInterface, JointPositionControllerInterface, JointTrajectoryControllerInterface
from robotblockset.ros2.robots_ros2 import robot_ros2
from franka_msgs.srv import SetLoad, SetTCPFrame

from rclpy.qos import qos_profile_sensor_data
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
        # Avoid nested spinning (we already have a background spinner); wait until done
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
        self._command.x = self.BaseToWorld(rx)
        self._command.v = self.BaseToWorld(rJ @ self._command.qdot)

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
        # Avoid nested spinning (we already have a background spinner); wait until done
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
