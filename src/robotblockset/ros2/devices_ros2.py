"""ROS 2 device interface implementations.

This module defines ROS 2-backed device wrappers used by RobotBlockSet device
interfaces. It provides node lifecycle helpers, state subscriptions, and
service-based control for fixtures, feeders, tool changers, and related tools.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah, Mihael Simonic.
"""

from time import sleep
from copy import deepcopy

from threading import Thread
from typing import Any, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from std_msgs.msg import String, Bool
    from std_srvs.srv import Trigger, SetBool
    from diagnostic_msgs.msg import DiagnosticArray
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if all are installed.")

from robotblockset.tools import rbs_object


class device_ros2(Node, rbs_object):
    """
    Base class for ROS2 devices

    Parameters
    ----------
    name : str
        Name of the device.
    ns : str, optional
        Namespace for the device, by default "".
    """

    def __init__(self, name: str = "device", namespace: str = "") -> None:
        """Initialize the node.

        Parameters
        ----------
        name : str, optional
            Name of the ROS 2 node.
        ns : str, optional
            Namespace for the node, by default "" (no namespace).
        """
        super().__init__(name)
        rbs_object.__init__(self)
        if namespace == "":
            self._namespace = ""
        else:
            self._namespace = "/" + str(namespace).strip("/")

        # Device name
        self.Name = name

        # Create the executor (can be multi-threaded or single-threaded)
        self.main_executor = MultiThreadedExecutor()
        self.main_executor.add_node(self)
        self.get_logger().info(f"{self.Name} object is being created...")

        # Spin control
        self._spinning = False
        self._spin_thread = None

        self.Message("Initialized", 2)

    def __del__(self) -> None:
        """Clean up resources on object destruction.

        Destructor automatically called when the object is garbage-collected.

        Notes
        -----
        - Python may call `__del__` during interpreter shutdown, when logging
            and ROS resources may already be partially destroyed.
        - All actions are inside try/except blocks for safety.
        - Calls `_shutdown()` to stop executor, join the thread, and destroy the node.
        """
        try:
            # Logging may fail during interpreter shutdown
            try:
                self.get_logger().info(f"{self.Name} object is being deleted...")
            except Exception:
                pass

            # Perform full shutdown of executors, thread, and node
            try:
                self._shutdown()
            except Exception:
                pass

            try:
                self.get_logger().info(f"{self.Name} deletion complete.")
            except Exception:
                pass

        except Exception:
            # Never allow destructor exceptions to propagate
            pass

    # ------------------------------------------------------------------
    # Spinning
    # ------------------------------------------------------------------
    def _run(self) -> None:
        """
        Background spin loop to process ROS2 callbacks for this node.

        Returns
        -------
        None
        """
        try:
            while rclpy.ok() and getattr(self, "_spinning", False):
                # Process any pending callbacks; timeout keeps loop responsive for shutdown
                # Use the executor instead of spinning the node directly to avoid wait set races
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            # Non-fatal: just report and exit spin loop
            try:
                self.get_logger().error(f"Spin loop stopped: {e}")
            except Exception:
                pass

    def _start_spinning(self, wait_for_state: bool = True, timeout_sec: float = 10.0) -> None:
        """
        Start executor spin thread and optionally wait for first state.

        Parameters
        ----------
        wait_for_state : bool, optional
            Whether to wait for the first state message to arrive.
        timeout_sec : float, optional
            Timeout in seconds when waiting for the first state.
        Returns
        -------
        None
        """
        if self._spinning:
            return
        self._spinning = True
        self._spin_thread = Thread(target=self._run, args=(), kwargs={}, daemon=True)
        self._spin_thread.start()

        # Optionally wait for first state message to arrive
        if wait_for_state:
            waited = 0.0
            step = 0.05
            while self.state is None and waited < timeout_sec:
                sleep(step)
                waited += step
            if self.state is None:
                self.get_logger().warning("No state received within timeout; continuing without connection confirmation.")
                self._connected = False
            else:
                self._connected = True
                self.get_logger().info("Connected")

        # Apply desired strategy now that spinning is active
        if self._control_strategy is not None:
            self.SetStrategy(self._control_strategy)
        else:
            self.controller = None
            self._control_strategy = None
            self.get_logger().warning("Initializing robot object without a controller.")

    def _shutdown(self, join_timeout: float = 2.0) -> None:
        """
        Cleanly stop spinning, stop the executor,
        and destroy the node.

        Parameters
        ----------
        join_timeout : float, optional
            Seconds to wait for the spin thread to finish.

        Returns
        -------
        None
        """
        # Stop spin loop and join background thread
        try:
            self._spinning = False
            if self._spin_thread is not None and self._spin_thread.is_alive():
                self.get_logger().info("Waiting for ROS2 spin thread to finish...")
                self._spin_thread.join(timeout=join_timeout)
                if self._spin_thread.is_alive():
                    self.get_logger().warning("ROS2 spin thread did not exit within timeout")
        except Exception as e:
            try:
                self.get_logger().error(f"Error while stopping spin thread: {e}")
            except Exception:
                pass

        # Remove node from executor and shut it down
        try:
            if hasattr(self, "_executor") and self._executor is not None:
                self.get_logger().info("Removing node from executor...")
                try:
                    self._executor.remove_node(self)
                except Exception:
                    # It's fine if the node was already removed
                    pass

                self.get_logger().info("Shutting down executor...")
                try:
                    self._executor.shutdown()
                except Exception as e:
                    self.get_logger().error(f"Error while shutting down executor: {e}")
        except Exception:
            # Again, avoid raising during shutdown
            pass

        # Destroy node
        try:
            self.get_logger().info("Destroying node...")
            self.destroy_node()
        except Exception as e:
            try:
                self.get_logger().error(f"Error while destroying node: {e}")
            except Exception:
                pass


# TOF camera
class tof_ros2(device_ros2):
    """
    TOF camera bclass for ROS2.
    """

    def __init__(self, name: str = "tof", namespace: str = "", state_topic: Optional[str] = "/tof_monitor/breached") -> None:
        """
        TOF camera class for ROS2.

        Parameters
        ----------
        name : str, optional
            Name of the device, by default "device".
        ns : str, optional
            Namespace for the device, by default "fixturing".
        state_topic : str, optional
            Topic name for state subscription.
        """
        device_ros2.__init__(self, name=name, namespace=namespace)

        self.state = None
        self._last_state_callback_time = None

        # Initialize ROS2 subscription to TFO state
        if state_topic is not None:
            topic_full_name = f"{self._namespace}/{state_topic.strip('/')}"
            self.get_logger().info(f"Subscribing to state topic: {topic_full_name}")
            self._states_sub = self.create_subscription(msg_type=Bool, topic=topic_full_name, callback=self._state_callback, qos_profile=10)

    # States
    def _state_callback(self, msg: Bool) -> None:
        """Callback function for state subscription."""
        # print(f"Status: {msg.data}")
        self.state = deepcopy(msg)
        self._last_state_callback_time = self.simtime()


# Fixture
class fixture_ros2(device_ros2):
    """
    Fixture class for ROS2.
    """

    def __init__(self, name: str = "fixture", namespace: str = "fixturing", state_topic: Optional[str] = None, service: Optional[str] = None) -> None:
        """
        Fixture class for ROS2.

        Parameters
        ----------
        name : str, optional
            Name of the device, by default "device".
        ns : str, optional
            Namespace for the device, by default "fixturing".
        state_topic : str, optional
            Topic name for state subscription, by default None.
        service : str, optional
            Service name for controlling the device, by default None.
        """
        device_ros2.__init__(self, name=name, namespace=namespace)

        self.state = None
        self._last_state_callback_time = None

        # Initialize ROS2 subscription to joint states
        # if state_topic is not None:
        #     topic_full_name = f"{self._namespace}/{state_topic.strip('/')}"
        #     self.get_logger().info(f"Subscribing to state topic: {topic_full_name}")
        #     self._joint_states_sub = self.create_subscription(msg_type=String, topic=topic_full_name, callback=self._joint_state_callback, qos_profile=10)

        # ROS2 service clients
        if service is not None:
            service_full_name = f"{self._namespace}/{service.strip('/')}"
            self.lock_client = self.create_client(SetBool, service_full_name)
            self.get_logger().info(f"Client for service '{service_full_name}'")

    # States
    def _state_callback(self, msg: String) -> None:
        """Callback function for state subscription."""
        # print(f"Status: {msg.data}")
        self.state = deepcopy(msg)
        self._last_state_callback_time = self.simtime()

    def Lock(self, lock: bool, timeout_sec: float = 1.0) -> Optional[bool]:
        """
        Send a request to the service and wait for the response.

        Parameters
        ----------
        lock : bool
            Lock fixture.
        timeout_sec : float, optional
            for the service response, by default 1.0.

        Returns
        -------
        int or None
            service succeeds, otherwise ``None``.

        Notes
        -----
        - If the service is not available within ``timeout_sec``,
          a warning is logged and ``None`` is returned.
        - If the service call fails or no result is received within
          ``timeout_sec``, an error is logged and ``None`` is returned.
        """
        if not self.lock_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("Service not available")
            return None

        request = SetBool.Request()
        request.data = lock
        future = self.lock_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            # Small sleep to avoid busy-waiting
            sleep(0.01)

            # Optional timeout handling
            if timeout_sec is not None:
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self.get_logger().error("Service call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self.get_logger().info(f"Service Lock: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service Lock call failed: {e}")
            return None

        return response.success


# Screw feeder
class screw_feeder_ros2(device_ros2):
    """
    Screw feeder class for ROS2.
    """

    def __init__(self, name: str = "screw", namespace: str = "screw_feeder", state_topic: Optional[str] = None) -> None:
        """
        Screw feeder class for ROS2.

        Parameters
        ----------
        name : str, optional
            Name of the device, by default "device".
        ns : str, optional
            Namespace for the device, by default "screw_feeder".
        state_topic : str, optional
            Topic name for state subscription, by default None.
        service : str, optional
            Service name for controlling the device, by default None.
        """
        device_ros2.__init__(self, name=name, namespace=namespace)

        self.state = None
        self._last_state_callback_time = None

        # Initialize ROS2 subscription to joint states
        # if state_topic is not None:
        #     topic_full_name = f"{self._namespace}/{state_topic.strip('/')}"
        #     self.get_logger().info(f"Subscribing to state topic: {topic_full_name}")
        #     self._joint_states_sub = self.create_subscription(msg_type=String, topic=topic_full_name, callback=self._joint_state_callback, qos_profile=10)

        # ROS2 service clients
        service_full_name = f"{self._namespace}/cycle_start"
        self.cycle_start_client = self.create_client(SetBool, service_full_name)
        self.get_logger().info(f"Client for service '{service_full_name}'")

        service_full_name = f"{self._namespace}/send_screw"
        self.send_screw_client = self.create_client(SetBool, service_full_name)
        self.get_logger().info(f"Client for service '{service_full_name}'")

    # States
    def _state_callback(self, msg: String) -> None:
        """Callback function for state subscription."""
        # print(f"Status: {msg.data}")
        self.state = deepcopy(msg)
        self._last_state_callback_time = self.simtime()

    def CycleStart(self, start: bool, timeout_sec: float = 1.0) -> Optional[bool]:
        """
        Send a request to the service and wait for the response.

        Parameters
        ----------
        start : bool
            Start cycle.
        timeout_sec : float, optional
            for the service response, by default 1.0.

        Returns
        -------
        int or None
            Service succeeds, otherwise ``None``.

        Notes
        -----
        - If the service is not available within ``timeout_sec``,
          a warning is logged and ``None`` is returned.
        - If the service call fails or no result is received within
          ``timeout_sec``, an error is logged and ``None`` is returned.
        """
        if not self.cycle_start_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("Service not available")
            return None

        request = SetBool.Request()
        request.data = start
        future = self.cycle_start_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            # Small sleep to avoid busy-waiting
            sleep(0.01)

            # Optional timeout handling
            if timeout_sec is not None:
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self.get_logger().error("Service call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self.get_logger().info(f"Service Lock: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service Lock call failed: {e}")
            return None

        return response.success

    def SendScrew(self, timeout_sec: float = 1.0) -> Optional[bool]:
        """
        Send a request to the service and wait for the response.

        Parameters
        ----------
        timeout_sec : float, optional
            for the service response, by default 1.0.

        Returns
        -------
        int or None
            Service succeeds, otherwise ``None``.

        Notes
        -----
        - If the service is not available within ``timeout_sec``,
          a warning is logged and ``None`` is returned.
        - If the service call fails or no result is received within
          ``timeout_sec``, an error is logged and ``None`` is returned.
        """
        if not self.send_screw_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("Service not available")
            return None

        request = Trigger.Request()
        future = self.send_screw_client.call_async(request)

        # Wait until the future is done (executor is running in another thread)
        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            # Small sleep to avoid busy-waiting
            sleep(0.01)

            # Optional timeout handling
            if timeout_sec is not None:
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self.get_logger().error("Service call timed out")
                    return None
        # Handle result / errors
        try:
            response = future.result()
            self.get_logger().info(f"Service Lock: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service Lock call failed: {e}")
            return None

        return response.success


########################################


class ScrewFeeder(device_ros2):
    """
    RAPID step screw feeder interface
    """

    def __init__(self, node: Any, ns: str = "/screw_feeder") -> None:
        """Initialize the high-level ROS2 screw-feeder wrapper."""
        super().__init__(node, name="screw_feeder")

        self.ns = ns

        # Services
        self._cycle_start = self._create_service_client(SetBool, f"{ns}/cycle_start")
        self._send_screw = self._create_service_client(Trigger, f"{ns}/send_screw")

        # State
        self.state = None
        self.sensors = {}

        self._state_sub = node.create_subscription(String, f"{ns}/feeder_state", self._on_state, 10)
        self._sensor_sub = node.create_subscription(DiagnosticArray, f"{ns}/sensor_states", self._on_sensors, 10)

    def _on_state(self, msg: String) -> None:
        """Store the latest high-level feeder state string."""
        self.state = msg.data

    def _on_sensors(self, msg: DiagnosticArray) -> None:
        """Store the latest feeder sensor values as a name-to-bool mapping."""
        if not msg.status:
            return
        values = msg.status[0].values
        self.sensors = {v.key: v.value == "true" for v in values}

    # High-level API

    def enable_cycle(self, enable: bool = True, timeout: float = 2.0) -> Any:
        """Enable or disable screw-feeder cycling."""
        return self._call_setbool(self._cycle_start, enable, timeout)

    def request_screw(self, timeout: float = 5.0) -> Any:
        """Request the feeder to dispense one screw."""
        return self._call_trigger(self._send_screw, timeout)

    def is_idle(self) -> bool:
        """Return whether the feeder reports the `IDLE` state."""
        return self.state == "IDLE"


class ScrewDriver(device_ros2):
    """
    Desoutter CVI-R2 screw driver
    """

    def __init__(self, node: Any, ns: str = "/screw_driver") -> None:
        """Initialize the high-level ROS2 screwdriver wrapper."""
        super().__init__(node, name="screw_driver")

        self.ns = ns

        self._c1 = self._create_service_client(Trigger, f"{ns}/CVIR_C1")
        self._c2 = self._create_service_client(Trigger, f"{ns}/CVIR_C2")
        self._c4 = self._create_service_client(Trigger, f"{ns}/CVIR_C4")

        self.last_result = None
        self._result_sub = node.create_subscription(DiagnosticArray, f"{ns}/cvir_results", self._on_result, 10)

    def _on_result(self, msg: DiagnosticArray) -> None:
        """Store the latest screwdriver diagnostic result message."""
        self.last_result = msg

    # High-level API

    def cycle_1(self, timeout: float = 10.0) -> Any:
        """Run screwdriver cycle 1."""
        return self._call_trigger(self._c1, timeout)

    def cycle_2(self, timeout: float = 10.0) -> Any:
        """Run screwdriver cycle 2."""
        return self._call_trigger(self._c2, timeout)

    def cycle_4(self, timeout: float = 10.0) -> Any:
        """Run screwdriver cycle 4."""
        return self._call_trigger(self._c4, timeout)


class ToolChangerRobot(device_ros2):
    """
    Toolchanger on robot flange
    """

    def __init__(self, node: Any, ns: str = "/workholding") -> None:
        """Initialize the robot-side tool changer wrapper."""
        super().__init__(node, name="tc_robot")

        self._open = self._create_service_client(SetBool, f"{ns}/open_tc_robot")

    def open(self, timeout: float = 2.0) -> Any:
        """Open the robot-side tool changer."""
        return self._call_setbool(self._open, True, timeout)

    def close(self, timeout: float = 2.0) -> Any:
        """Close the robot-side tool changer."""
        return self._call_setbool(self._open, False, timeout)


class ToolChangerTable(device_ros2):
    """
    Toolchanger on table
    """

    def __init__(self, node: Any, ns: str = "/workholding") -> None:
        """Initialize the table-side tool changer wrapper."""
        super().__init__(node, name="tc_table")

        self._open = self._create_service_client(SetBool, f"{ns}/open_tc_table")

    def open(self, timeout: float = 2.0) -> Any:
        """Open the table-side tool changer."""
        return self._call_setbool(self._open, True, timeout)

    def close(self, timeout: float = 2.0) -> Any:
        """Close the table-side tool changer."""
        return self._call_setbool(self._open, False, timeout)


class Chuck(device_ros2):
    """
    Workholding chuck mounted on table
    """

    def __init__(self, node: Any, ns: str = "/workholding") -> None:
        """Initialize the workholding chuck wrapper."""
        super().__init__(node, name="chuck")

        self._open = self._create_service_client(SetBool, f"{ns}/open_chuck")
        self._disable = self._create_service_client(SetBool, f"{ns}/disable_chuck")

    def open(self, timeout: float = 2.0) -> Any:
        """Open the chuck."""
        return self._call_setbool(self._open, True, timeout)

    def close(self, timeout: float = 2.0) -> Any:
        """Close the chuck."""
        return self._call_setbool(self._open, False, timeout)

    def disable(self, timeout: float = 2.0) -> Any:
        """Disable the chuck output."""
        return self._call_setbool(self._disable, True, timeout)
