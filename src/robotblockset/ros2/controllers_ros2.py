"""ROS 2 controller interfaces and helpers.

This module defines ROS 2-backed controller wrappers used by RobotBlockSet
robot interfaces. It provides controller-manager helpers, joint-trajectory
interfaces, and Cartesian impedance command interfaces through ROS 2 topics,
actions, and services.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah, Mihael Simonic.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from abc import abstractmethod
from typing import Optional, Iterable, List, Any


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.task import Future
    from rclpy.action import ActionClient
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from builtin_interfaces.msg import Duration
    from controller_manager_msgs.srv import ListControllers, SwitchController, LoadController, UnloadController, ConfigureController
    from controller_manager_msgs.msg import ControllerState
    from control_msgs.action import FollowJointTrajectory
    from action_msgs.msg import GoalStatus
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except Exception as e:
    raise e from RuntimeError("Problems with importin ROS2 messages. Check if all are installed.")

from robotblockset.tools import rbs_object, matrix, vector, isvector, ismatrix, isscalar, rbs_type
from robotblockset.robots import robot, MotionResultCodes, CommandModeCodes
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, JointPathType, JointVelocityType, Pose3DType, TimesType, Velocity3DType, WrenchType, JointTorqueType

import numpy as np
from time import monotonic, sleep

try:
    from compliant_controllers_msgs.msg import CartesianCommand
except ImportError:
    CartesianCommand = None


class controller_manager_helper:
    """
    Helper for ROS 2 controller-manager lifecycle operations.

    High-level lifecycle transitions:
        load(name)       -> ensure controller plugin instance exists (UNCONFIGURED)
        configure(name)  -> UNCONFIGURED -> INACTIVE
        activate(name)   -> INACTIVE -> ACTIVE (via switch_controllers)
        deactivate(name) -> ACTIVE -> INACTIVE
        unload(name)     -> remove controller instance (must not be ACTIVE)

    Legacy wrappers:
      load_controller(name) -> load + configure
      start_controller(name) -> activate
      stop_controller(name) -> deactivate
      switch_controllers(start, stop) -> activate/deactivate

    Utilities:
        list_controllers(), get_state(name), is_active(name)

    Attributes
    ----------
    cm_name : str
        Name of the controller-manager namespace.
    list_client : Any
        Service client for listing controllers.
    switch_client : Any
        Service client for switching controllers.
    load_client : Any
        Service client for loading controllers.
    unload_client : Any
        Service client for unloading controllers.
    configure_client : Any
        Service client for configuring controllers.
    """

    def __init__(self, node: Node, namespace: str = "", controller_manager_name: Optional[str] = "controller_manager") -> None:
        """
        Initialize the controller-manager helper.

        Parameters
        ----------
        node : Node
            The ROS 2 node used to create service clients. The caller is
            responsible for spinning this node.
        namespace : str, optional
            Namespace for the controller manager.
        controller_manager_name : str, optional
            Name of the ROS 2 controller manager.
        """
        self._node = node
        if namespace is None or namespace == "":
            self._namespace = ""
        else:
            self._namespace = "/" + str(namespace).strip("/")
            
        self.cm_name = f"{self._namespace}/{controller_manager_name('/')}"
        # Core service clients
        self.list_client = self._node.create_client(ListControllers, f"{self.cm_name}/list_controllers")
        self.switch_client = self._node.create_client(SwitchController, f"{self.cm_name}/switch_controller")
        self.load_client = self._node.create_client(LoadController, f"{self.cm_name}/load_controller")
        self.unload_client = self._node.create_client(UnloadController, f"{self.cm_name}/unload_controller")
        self.configure_client = self._node.create_client(ConfigureController, f"{self.cm_name}/configure_controller")

        self._node.get_logger().info("Waiting for controller_manager lifecycle services ...")
        for c in [self.list_client, self.switch_client, self.load_client, self.unload_client, self.configure_client]:
            c.wait_for_service()
        self._node.get_logger().info("Lifecycle helper connected.")

    # ----------------------------- Query utilities -----------------------------
    def list_controllers(self) -> List[ControllerState]:
        """
        Query the controller_manager for all available controllers.

        Returns
        -------
        list of controller_manager_msgs.msg.ControllerState
            A list of controller state objects. If the service call fails,
            an empty list is returned and an error is logged.
        """
        req = ListControllers.Request()
        result = self.list_client.call(req)
        if result is not None:
            return result.controller
        self._node.get_logger().error("Failed to list controllers")
        return []

    def get_state(self, name: str) -> Optional[str]:
        """
        Get the lifecycle state of a specific controller.

        Parameters
        ----------
        name : str
            Name of the controller.

        Returns
        -------
        str or None
            The controller state string (e.g. ``"active"``, ``"inactive"``),
            or ``None`` if the controller is not found.
        """
        for c in self.list_controllers():
            if c.name == name:
                return c.state
        return None

    def is_active(self, name: str) -> bool:
        """
        Check whether a controller is in the ``"active"`` state.

        Parameters
        ----------
        name : str
            Name of the controller.

        Returns
        -------
        bool
            ``True`` if the controller exists and is active, otherwise ``False``.
        """
        return self.get_state(name) == "active"

    def _wait_for_state(self, name: str, states: Iterable[Optional[str]], timeout: float = 2.0, period: float = 0.05) -> Optional[str]:
        """
        Wait briefly for controller-manager state to match one of ``states``.

        ROS 2 controller-manager lifecycle service replies can be visible before
        ``list_controllers`` has settled, and some managers may complete the
        requested transition even when the immediate response is not successful
        from the client's point of view.  The helper methods are intentionally
        idempotent, so confirming the eventual state avoids spurious first-call
        failures during strategy switches.
        """
        states = set(states)
        deadline = monotonic() + timeout
        while True:
            state = self.get_state(name)
            if state in states:
                return state
            if monotonic() >= deadline:
                return state
            sleep(period)

    # ----------------------------- Lifecycle ops -----------------------------
    def load(self, name: str) -> bool:
        """
        Load a controller by name.

        This requests that the controller_manager loads the specified controller
        plugin so that it can be configured and activated later.

        Parameters
        ----------
        name : str
            Name of the controller to load.

        Returns
        -------
        bool
            ``True`` if the controller is already loaded or was loaded successfully,
            ``False`` if the load request failed.
        """
        state = self.get_state(name)
        if state is not None:
            return True  # already loaded

        req = LoadController.Request()
        req.name = name
        result = self.load_client.call(req)
        ok = bool(result and result.ok)
        state = self._wait_for_state(name, states=["unconfigured", "inactive", "active"])
        if state is not None:
            return True
        if not ok:
            self._node.get_logger().error(f"Load failed: {name}")
        return ok

    def configure(self, name: str) -> bool:
        """
        Configure a controller, loading it first if necessary.

        Parameters
        ----------
        name : str
            Name of the controller to configure.

        Returns
        -------
        bool
            ``True`` if the controller is already configured (``"inactive"`` or
            ``"active"``) or was configured successfully, otherwise ``False``.
        """
        state = self.get_state(name)
        if state is None:
            if not self.load(name):
                return False
            state = self.get_state(name)
        if state in ["inactive", "active"]:
            return True  # already configured
        req = ConfigureController.Request()
        req.name = name
        result = self.configure_client.call(req)
        ok = bool(result and result.ok)
        state = self._wait_for_state(name, states=["inactive", "active"])
        if state in ["inactive", "active"]:
            return True
        if not ok:
            self._node.get_logger().error(f"Configure failed: {name}")
        return False

    def activate(self, name: str) -> bool:
        """
        Activate a controller.

        Ensures the controller is loaded and configured, then requests the
        controller_manager to start (activate) it.

        Parameters
        ----------
        name : str
            Name of the controller to activate.

        Returns
        -------
        bool
            ``True`` if the controller is already active or was activated successfully,
            ``False`` otherwise.
        """
        if self.is_active(name):
            return True
        if not self.configure(name):
            return False
        req = SwitchController.Request()
        req.start_controllers = [name]
        req.stop_controllers = []
        req.strictness = SwitchController.Request.STRICT
        req.start_asap = False
        req.activate_asap = True
        req.timeout = Duration(sec=5)
        result = self.switch_client.call(req)
        ok = bool(result and result.ok)
        if self._wait_for_state(name, states=["active"]) == "active":
            return True
        if not ok:
            self._node.get_logger().error(f"Activate failed: {name}")
        return False

    def deactivate(self, name: str) -> bool:
        """
        Deactivate (stop) a controller.

        Parameters
        ----------
        name : str
            Name of the controller to deactivate.

        Returns
        -------
        bool
            ``True`` if the controller is already not active or was deactivated successfully,
            ``False`` otherwise.
        """
        if self.get_state(name) != "active":
            return True  # already not active
        req = SwitchController.Request()
        req.start_controllers = []
        req.stop_controllers = [name]
        req.strictness = SwitchController.Request.STRICT
        req.start_asap = False
        req.activate_asap = False
        req.timeout = Duration(sec=5)
        result = self.switch_client.call(req)
        ok = bool(result and result.ok)
        if not ok:
            self._node.get_logger().error(f"Deactivate failed: {name}")
        return ok

    def unload(self, name: str) -> bool:
        """
        Unload a controller.

        The controller must not be active when unloading. If it is active,
        this function first attempts to deactivate it.

        Parameters
        ----------
        name : str
            Name of the controller to unload.

        Returns
        -------
        bool
            ``True`` if the controller is already absent or was unloaded successfully,
            ``False`` if deactivation or unload failed.
        """
        # Must be inactive to unload
        state = self.get_state(name)
        if state is None:
            return True  # already gone
        if state == "active" and not self.deactivate(name):
            return False
        req = UnloadController.Request()
        req.name = name
        result = self.unload_client.call(req)
        ok = bool(result and result.ok)
        if not ok:
            self._node.get_logger().error(f"Unload failed: {name}")
        return ok

    # Backward compatibility wrappers
    def start_controller(self, name: str) -> bool:
        """
        Backwards-compatible wrapper to start (activate) a controller.

        Parameters
        ----------
        name : str
            Name of the controller to start.

        Returns
        -------
        bool
            ``True`` if activation succeeded or the controller is already active,
            ``False`` otherwise.
        """
        return self.activate(name)

    def load_controller(self, name: str) -> bool:
        """
        Backwards-compatible wrapper to load and configure a controller.

        Parameters
        ----------
        name : str
            Name of the controller to load and configure.

        Returns
        -------
        bool
            ``True`` if the controller is loaded and configured, ``False`` otherwise.
        """
        return self.load(name) and self.configure(name)

    def stop_controller(self, name: str) -> bool:
        """
        Backwards-compatible wrapper to stop (deactivate) a controller.

        Parameters
        ----------
        name : str
            Name of the controller to stop.

        Returns
        -------
        bool
            ``True`` if the controller is not active or was deactivated successfully,
            ``False`` otherwise.
        """
        return self.deactivate(name)

    def switch_controllers(self, start_list: Iterable[str], stop_list: Iterable[str]) -> bool:
        """
        Switch sets of controllers in a single operation.

        Parameters
        ----------
        start_list : Iterable[str]
            Names of controllers to activate.
        stop_list : Iterable[str]
            Names of controllers to deactivate.

        Returns
        -------
        bool
            ``True`` if the switch request succeeded, ``False`` otherwise.
        """
        start_list = list(start_list)
        stop_list = list(stop_list)
        req = SwitchController.Request()
        req.activate_controllers = start_list
        req.deactivate_controllers = stop_list
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.start_asap = False
        req.activate_asap = True
        req.timeout = Duration(sec=5)
        result = self.switch_client.call(req)
        ok = bool(result and result.ok)
        if start_list or stop_list:
            deadline = monotonic() + 2.0
            while True:
                controllers = {c.name: c.state for c in self.list_controllers()}
                started = all(controllers.get(name) == "active" for name in start_list)
                stopped = all(controllers.get(name) != "active" for name in stop_list)
                if started and stopped:
                    return True
                if monotonic() >= deadline:
                    break
                sleep(0.05)
        return ok

    def get_active_controller(self, controller_names: Optional[Iterable[str]] = None) -> Optional[str]:
        """
        Get the name of the first active controller from a candidate list.

        Parameters
        ----------
        controller_names : Iterable[str], optional
            Iterable of controller names to check. If ``None``, all known controllers
            are considered.

        Returns
        -------
        str or None
            Name of the first active controller from the list, or ``None`` if none
            of them are active.
        """
        # Use provided list of controller names or call list_controllers() to get all controllers
        if controller_names is None:
            controller_names = [c.name for c in self.list_controllers()]

        # Get the full list of controllers
        controllers = self.list_controllers()

        # Check for active controller in the provided list of names
        for c in controllers:
            if c.name in controller_names and c.state == "active":
                return c.name  # Return the name of the first active controller

        # If no active controller is found in the provided list, return None
        return None

    def _shutdown(self) -> None:
        """
        Gracefully destroy the service clients owned by this helper.

        The underlying node is owned by the caller and is not destroyed here.

        Returns
        -------
        None
        """
        # Destroy service clients only; the node is owned by the caller.
        for attr in [
            "list_client",
            "switch_client",
            "load_client",
            "unload_client",
            "configure_client",
        ]:
            if hasattr(self, attr):
                try:
                    client = getattr(self, attr)
                    if client is not None:
                        client.destroy()
                except Exception:
                    pass


# Controllers
class RosControllerInterface(rbs_object):
    """
    Generic interface for a ROS2-based controller.

    This class defines a skeleton for ROS2 controller interfaces used within
    Robotics Behavior Specification (RBS) systems. Derived classes must implement
    the `_init_ros_interfaces` method to create all ROS-specific communication
    elements such as publishers, subscribers, actions, or services.

    Attributes
    ----------
    motion_result : int | None
        Last motion result code reported by the controller interface.
    motion_done : bool | None
        Completion flag for the active motion command.
    _robot : robot
        RobotBlockSet robot instance bound during activation.
    _node : Node
        ROS 2 node used to create communication entities.
    """

    def __init__(self) -> None:
        """
        Initialize a `RosControllerInterface` instance.

        Notes
        -----
        - This method only initializes the `rbs_object` base class.
        - ROS interfaces are initialized later via `Activate()`.
        """
        rbs_object.__init__(self)
        self.motion_result = None
        self.motion_done = None

    def Activate(self, robot: robot, node: Node) -> None:
        """
        Activate the controller using the specified robot and ROS node.

        This method initializes the controller with data from the robot, registers
        ROS interfaces, and marks the controller as active.

        Parameters
        ----------
        robot : robot
            The robot instance providing joint and state data.
        node : Node
            The ROS2 node used to create publishers, subscribers, or action clients.

        Returns
        -------
        None

        Notes
        -----
        - This method should be called before issuing any control commands.
        - Derived classes typically call `_init_ros_interfaces()` to build ROS2 entities.
        """
        self._robot = robot
        self._node = node
        self.nj = robot.nj
        self._init_ros_interfaces()
        self._registered = True
        self._last_control_time = robot._last_control_time
        self._node.get_logger().info("Activated RosControllerInterface")

    # Verbosity level is proxied to the robot so changes propagate automatically after Activate()
    # integers are immutable so setting self._verbose = robot._verbose won't keep them in sync
    @property
    def _verbose(self) -> int:
        """Return the effective verbosity level, proxied from the active robot when available."""
        if hasattr(self, "_robot") and self._robot is not None:
            return getattr(self._robot, "_verbose", 1)
        return getattr(self, "__local_verbose", 1)

    @_verbose.setter
    def _verbose(self, value: int) -> None:
        """Set the effective verbosity level on the robot or locally before activation."""
        if hasattr(self, "_robot") and self._robot is not None:
            self._robot._verbose = value
        else:
            self.__local_verbose = value

    @abstractmethod
    def _init_ros_interfaces(self) -> None:
        """
        Initialize ROS2 communication interfaces.

        This method must be implemented by derived classes to set up all ROS
        interfaces needed by the controller, such as:
        - publishers
        - subscribers
        - action clients
        - service clients

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        pass


class CsfCartesianImpedanceControllerInterface(RosControllerInterface):
    """
    Interface for a Cartesian impedance controller.

    Attributes
    ----------
    Kp : np.ndarray
        Cartesian translational stiffness vector.
    Kr : np.ndarray
        Cartesian rotational stiffness vector.
    R : np.ndarray
        Rotation matrix used to orient the translational stiffness frame.
    D : float
        Damping scaling factor.
    """

    def __init__(
        self,
        robot: Optional[robot] = None,
        ros_plugin_name: str = "cartesian_impedance_controller",
        namespace: str = "",
        topic: str = "cartesian_command",
        Kp: np.ndarray = np.array([2000.0, 2000.0, 2000.0]),
        Kr: np.ndarray = np.array([30.0, 30.0, 30.0]),
        R: np.ndarray = np.eye(3),
        D: float = 2.0,
        command_frame: str = "",
    ) -> None:
        """
        Initialize the Cartesian impedance controller interface.

        Parameters
        ----------
        robot : robot, optional
            Reserved robot argument kept for compatibility.
        ros_plugin_name : str, optional
            Name of the ROS 2 controller plugin.
        namespace : str, optional
            Namespace used for the controller topics.
        topic : str, optional
            Cartesian command topic name.
        Kp : np.ndarray, optional
            Translational stiffness gains.
        Kr : np.ndarray, optional
            Rotational stiffness gains.
        R : np.ndarray, optional
            Rotation matrix of the translational stiffness frame.
        D : float, optional
            Damping scaling factor.
        """
        RosControllerInterface.__init__(self)

        if CartesianCommand is None:
            raise ImportError("compliant_controllers_msgs package not found. CsfCartesianImpedanceControllerInterface will not be available.")

        self._ros_plugin_name = ros_plugin_name
        self.Name = "CsfCartesianImpedanceControllerInterface"
        self._namespace = namespace
        self._topic = topic
        self.Kp = vector(Kp, dim=3)
        self.Kr = vector(Kr, dim=3)
        self.R = matrix(R, shape=(3, 3))
        self.D = float(D)
        self._default_Kp = vector(Kp, dim=3)
        self._default_Kr = vector(Kr, dim=3)
        self._default_R = matrix(R, shape=(3, 3))
        self._default_D = float(D)
        self.command_frame = command_frame
        self._registered = False

    def _normalise_compliance(
        self,
        Kp: Optional[ArrayLike] = None,
        Kr: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        D: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return validated Cartesian impedance parameters."""
        if Kp is None:
            Kp = self.Kp
        elif isscalar(Kp):
            Kp = np.ones(3) * Kp
        else:
            Kp = vector(Kp, dim=3)

        if Kr is None:
            Kr = self.Kr
        elif isscalar(Kr):
            Kr = np.ones(3) * Kr
        else:
            Kr = vector(Kr, dim=3)

        if R is None:
            R = self.R
        else:
            R = matrix(R, shape=(3, 3))

        if D is None:
            D = self.D
        elif not isscalar(D):
            raise ValueError("Damping 'D' is not scalar")

        D = float(D)
        if D < 0 or D > 2:
            raise ValueError("Damping 'D' must be in range [0, 2]")

        return np.asarray(Kp, dtype=float), np.asarray(Kr, dtype=float), np.asarray(R, dtype=float), D

    def GetCartesianCompliance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return the active Cartesian impedance parameters."""
        return self.Kp, self.Kr, self.R, self.D

    def GetCartesianStiffness(self) -> tuple[np.ndarray, np.ndarray]:
        """Return translational and rotational Cartesian stiffness."""
        return self.Kp, self.Kr

    def GetCartesianDamping(self) -> float:
        """Return the Cartesian damping scaling factor."""
        return self.D

    def SetCartesianCompliance(
        self,
        Kp: Optional[ArrayLike] = None,
        Kr: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        D: Optional[float] = None,
        hold_pose: bool = True,
    ) -> int:
        """Update Cartesian impedance parameters for subsequent CSF commands."""
        Kp, Kr, R, D = self._normalise_compliance(Kp=Kp, Kr=Kr, R=R, D=D)
        self.Kp = Kp
        self.Kr = Kr
        self.R = R
        self.D = D

        if hold_pose and getattr(self, "_registered", False) and getattr(self, "_robot", None) is not None:
            x = getattr(self._robot._command, "x", None)
            if x is not None:
                trq = getattr(self._robot._command, "FT", None)
                if trq is None:
                    trq = np.zeros(6)
                self.GoTo_X(x, np.zeros(6), trq, 0.0, Kp=Kp, Kr=Kr, R=R, D=D)

        return MotionResultCodes.MOTION_SUCCESS.value

    def SetCartesianStiffness(self, Kp: Optional[ArrayLike] = None, Kr: Optional[ArrayLike] = None, hold_pose: bool = True) -> int:
        """Update Cartesian stiffness while preserving orientation frame and damping."""
        return self.SetCartesianCompliance(Kp=Kp, Kr=Kr, R=self.R, D=self.D, hold_pose=hold_pose)

    def SetCartesianDamping(self, D: Optional[float] = None, hold_pose: bool = True) -> int:
        """Update Cartesian damping while preserving stiffness."""
        return self.SetCartesianCompliance(Kp=self.Kp, Kr=self.Kr, R=self.R, D=D, hold_pose=hold_pose)

    def SetCartesianSoft(self, stiffness: ArrayLike, hold_pose: bool = True) -> int:
        """Scale Cartesian stiffness relative to this controller's configured defaults."""
        if isscalar(stiffness):
            fac_p = np.ones(3) * stiffness
            fac_r = fac_p
        elif isvector(stiffness, dim=3):
            fac_p = vector(stiffness, dim=3)
            fac_r = fac_p
        else:
            fac = vector(stiffness, dim=6)
            fac_p = fac[:3]
            fac_r = fac[3:]

        fac_p = np.clip(fac_p, 0.0, 1.0)
        fac_r = np.clip(fac_r, 0.0, 1.0)
        fac = float(np.max(np.hstack((fac_p, fac_r))))

        return self.SetCartesianCompliance(
            Kp=self._default_Kp * fac_p,
            Kr=self._default_Kr * fac_r,
            R=self._default_R,
            D=self._default_D * fac,
            hold_pose=hold_pose,
        )

    def SetCartesianStiff(self, hold_pose: bool = True) -> int:
        """Set Cartesian impedance to the configured stiff defaults."""
        return self.SetCartesianSoft(1.0, hold_pose=hold_pose)

    def SetCartesianCompliant(self, hold_pose: bool = True) -> int:
        """Set Cartesian impedance to zero stiffness."""
        return self.SetCartesianSoft(0.0, hold_pose=hold_pose)

    def _init_ros_interfaces(self) -> None:
        """Create ROS 2 publishers used by the Cartesian impedance controller."""
        self._cartesian_command_publisher = self._node.create_publisher(CartesianCommand, f"{self._namespace}/{str(self._topic).strip('/')}", 1)

    def _unregister_interface(self) -> None:
        """Destroy ROS 2 entities owned by the Cartesian impedance controller."""
        if hasattr(self, "_cartesian_command_publisher"):
            try:
                self._node.destroy_publisher(self._cartesian_command_publisher)
            except Exception:
                pass
            del self._cartesian_command_publisher
        self._registered = False
        self._robot = None

    def GoTo_X(
        self,
        x: Pose3DType,
        xdot: Velocity3DType,
        trq: WrenchType,
        wait: TimesType,
        Kp: Optional[ArrayLike] = None,
        Kr: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        D: Optional[float] = None,
        **kwargs: Any,
    ) -> int:
        """
        Publish a Cartesian impedance command.

        Parameters
        ----------
        x : Pose3DType
            Desired task pose represented as position and quaternion.
        xdot : Velocity3DType
            Desired task twist.
        trq : WrenchType
            Desired feed-forward wrench.
        wait : TimesType
            Kept for API compatibility; not used by the ROS 2 publisher.
        Kp : ArrayLike, optional
            Translational stiffness vector.
        Kr : ArrayLike, optional
            Rotational stiffness vector.
        R : ArrayLike, optional
            Rotation matrix of the translational stiffness frame.
        D : float, optional
            Damping scaling factor.

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        The method updates the stored compliance parameters with the values
        used for the published command.
        """

        Kp, Kr, R, D = self._normalise_compliance(Kp=Kp, Kr=Kr, R=R, D=D)

        command_x = np.asarray(x, dtype=float)
        command_xdot = np.asarray(xdot, dtype=float)
        command_trq = np.asarray(trq, dtype=float)

        # Build CartesianCommand message
        cmd = CartesianCommand()
        cmd.header.stamp = self._node.get_clock().now().to_msg()
        cmd.header.frame_id = self.command_frame

        # Pose
        cmd.pose.position.x = command_x[0]
        cmd.pose.position.y = command_x[1]
        cmd.pose.position.z = command_x[2]
        cmd.pose.orientation.x = command_x[4]
        cmd.pose.orientation.y = command_x[5]
        cmd.pose.orientation.z = command_x[6]
        cmd.pose.orientation.w = command_x[3]

        # Twist
        cmd.velocity.linear.x = command_xdot[0]
        cmd.velocity.linear.y = command_xdot[1]
        cmd.velocity.linear.z = command_xdot[2]
        cmd.velocity.angular.x = command_xdot[3]
        cmd.velocity.angular.y = command_xdot[4]
        cmd.velocity.angular.z = command_xdot[5]

        # Wrench
        cmd.wrench_ff.force.x = command_trq[0]
        cmd.wrench_ff.force.y = command_trq[1]
        cmd.wrench_ff.force.z = command_trq[2]
        cmd.wrench_ff.torque.x = command_trq[3]
        cmd.wrench_ff.torque.y = command_trq[4]
        cmd.wrench_ff.torque.z = command_trq[5]

        # Stiffness and damping
        trM = np.diag(Kp)
        rotM = np.diag(Kr)
        trK = R @ trM @ R.T
        rotK = rotM
        trD = R @ (2 * np.sqrt(trM)) @ R.T
        rotD = D * np.sqrt(rotM)

        # Check if any is NaN
        if np.isnan(trM).any() or np.isnan(rotM).any() or np.isnan(trK).any() or np.isnan(rotK).any() or np.isnan(trD).any() or np.isnan(rotD).any():
            raise Exception("%s: GoTo_x: trM or rotM or trK or rotK or trD or rotD - NaN error" % self.Name)

        cmd.stiffness_pos = trK.flatten(order="C").tolist()
        cmd.stiffness_ori = rotK.flatten(order="C").tolist()
        cmd.damping_pos = trD.flatten(order="C").tolist()
        cmd.damping_ori = rotD.flatten(order="C").tolist()

        # Keep configured values for next call
        self.Kp = Kp
        self.Kr = Kr
        self.R = R
        self.D = D

        # Publish command
        self._cartesian_command_publisher.publish(cmd)
        return MotionResultCodes.MOTION_SUCCESS.value


class JointPositionControllerInterface(RosControllerInterface):
    """
    Interface for a joint-position controller.

    Attributes
    ----------
    _namespace : str
        Namespace used for the controller topics.
    _ros_plugin_name : str
        Name of the ROS 2 controller plugin.
    _topic : str
        Topic used to publish joint commands.
    _use_publisher : bool
        Indicates whether a publisher interface is active.
    """

    def __init__(self, namespace: str = "", ros_plugin_name: str = "command", topic: str = "joint_position") -> None:
        """
        Initialize the joint-position controller interface.

        This class provides a thin wrapper around a ROS 2 controller that
        accepts single-step joint commands through a `trajectory_msgs/JointTrajectory`
        topic. It is used for direct joint position setpoints without action
        goal management.

        Parameters
        ----------
        namespace : str, optional
            Namespace prefix for the controller (for example ``"arm"``).
            If empty, no namespace prefix is used.
        ros_plugin_name : str, optional
            Name of the controller plugin (e.g. ``"command"``).
            If empty or ``None``, the controller is assumed to be directly
            under the namespace (no extra name component is added).
        topic : str, optional
            Topic name used when publishing joint-position commands.

        Notes
        -----
        - Effective topic names are constructed as
          ``<namespace>/<ros_plugin_name>/<topic>`` or ``<namespace>/<topic>``.
        - The actual ROS node is provided later through
          :meth:`RosControllerInterface.Activate`.
        """
        RosControllerInterface.__init__(self)
        self.Name = "JointPositionControllerInterface"
        self._namespace = namespace
        if ros_plugin_name is None or ros_plugin_name == "":
            self._ros_plugin_name = ""
        else:
            self._ros_plugin_name = str(ros_plugin_name).strip("/")
        self._topic = str(topic).strip("/")
        self._use_publisher = False

    def _init_ros_interfaces(self) -> None:
        """
        Initialize ROS2 communication interfaces for the controller.

        Depending on configuration, this will either create:

        - a `JointTrajectory` publisher used for single-point commands

        The chosen interface is stored in:

        - ``self.command_publisher`` if using topics.

        Raises
        ------
        ValueError
            If topic name is not configured.

        Notes
        -----
        - This method assumes that `self._node` has been set by
          :meth:`RosControllerInterface.Activate`.
        """
        self._use_publisher = True

        if self._topic is not None:
            if self._ros_plugin_name == "":
                topic_full_name = f"{self._namespace}/{self._topic}"
            else:
                topic_full_name = f"{self._namespace}/{self._ros_plugin_name}/{self._topic}"

            self._node.get_logger().info(f"Publishing to topic: {topic_full_name}")
            self.command_publisher = self._node.create_publisher(JointTrajectory, topic_full_name, 10)
            self._use_publisher = True
        else:
            raise ValueError("Topic for joint position controller must be defined.")

    def _unregister_interface(self) -> None:
        """
        Tear down ROS2 interfaces and reset controller bindings.

        Destroys any created publishers or action clients and clears references
        to the associated robot and command/state buffers. After this call,
        the interface is no longer considered registered.

        Returns
        -------
        None

        Notes
        -----
        - This is typically called as part of a shutdown sequence or when
          switching to a different controller interface.
        - Any exceptions during destruction are caught and ignored.
        """
        if hasattr(self, "command_publisher"):
            try:
                self._node.destroy_publisher(self.command_publisher)
            except Exception:
                pass
            del self.command_publisher
        self.motion_done = None
        self._registered = False
        self._robot = None
        self._use_publisher = False

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Publish a joint-position command.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        qdot : JointVelocityType, optional
            Desired joint velocities (nj,).
        trq : JointTorqueType, optional
            Desired joint torques (nj,).
        wait : float, optional
            Kept for API compatibility; not used by the publisher interface.

        Returns
        -------
        int
            Motion result code.

        Raises
        ------
        ValueError
            If control is not implemented in controller.
        """
        if self._use_publisher:
            q = vector(q, dim=self.nj)
            if qdot is None:
                qdot = np.zeros(self.nj)
            else:
                qdot = vector(qdot, dim=self.nj)

            if wait is None:
                wait = float(getattr(self._robot, "tsamp", 0.01))

            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = list(self._robot.joint_names)

            point_msg = JointTrajectoryPoint()
            point_msg.positions = q.astype(float).tolist()
            point_msg.velocities = qdot.astype(float).tolist()
            point_msg.time_from_start.sec = int(wait)
            point_msg.time_from_start.nanosec = int((wait - int(wait)) * 1e9)
            trajectory_msg.points.append(point_msg)

            self.command_publisher.publish(trajectory_msg)
            self.motion_done = True
            self.motion_result = MotionResultCodes.MOTION_SUCCESS.value
            self._robot._command.mode = CommandModeCodes.JOINT.value
            self._robot._command.q = q
            self._robot._command.qdot = qdot
            x = self._robot.Kinmodel(q)[0]
            self._robot._command.x = x
            self._robot._command.v = np.zeros(6)
            return MotionResultCodes.MOTION_SUCCESS.value
        else:
            raise ValueError("Joint position controller is not ready.")


class JointTrajectoryControllerInterface(RosControllerInterface):
    """
    Interface for a joint-trajectory controller.

    Attributes
    ----------
    _namespace : str
        Namespace used for the controller topics and actions.
    _ros_plugin_name : str
        Name of the ROS 2 controller plugin.
    _topic : str
        Topic used for trajectory publishing.
    _action : str
        Action used for trajectory execution.
    _use_publisher : bool
        Indicates whether the publisher path is active.
    _use_action : bool
        Indicates whether the action-client path is active.
    """

    def __init__(self, namespace: str = "", ros_plugin_name: str = "joint_trajectory_controller", topic: str = "joint_trajectory", action: str = "follow_joint_trajectory") -> None:
        """
        Initialize the joint-trajectory controller interface.

        This class provides a thin wrapper around a ROS 2 controller manager
        plugin exposing a `control_msgs/FollowJointTrajectory` action or a
        `trajectory_msgs/JointTrajectory` topic. It can be configured to
        either communicate via an action client or a plain publisher.

        Parameters
        ----------
        namespace : str, optional
            Namespace prefix for the controller (for example ``"arm"``).
            If empty, no namespace prefix is used.
        ros_plugin_name : str, optional
            Name of the controller plugin (e.g. ``"joint_trajectory_controller"``).
            If empty or ``None``, the controller is assumed to be directly
            under the namespace (no extra name component is added).
        topic : str, optional
            Topic name used when publishing ``JointTrajectory`` commands.
        action : str, optional
            Action name used when creating a ``FollowJointTrajectory`` action
            client. If not ``None``, this path is preferred over publishing.

        Notes
        -----
        - Effective topic/action names are constructed as:

          - ``<namespace>/<ros_plugin_name>/<action>`` or ``<namespace>/<action>``
          - ``<namespace>/<ros_plugin_name>/<topic>`` or ``<namespace>/<topic>``

        - The actual ROS node is provided later through
          :meth:`RosControllerInterface.Activate`.
        """
        RosControllerInterface.__init__(self)
        self.Name = "JointTrajectoryControllerInterface"
        self._namespace = namespace
        if ros_plugin_name is None or ros_plugin_name == "":
            self._ros_plugin_name = ""
        else:
            self._ros_plugin_name = str(ros_plugin_name).strip("/")
        self._topic = str(topic).strip("/")
        self._action = str(action).strip("/")
        self._use_publisher = False
        self._use_action = True

    def _init_ros_interfaces(self) -> None:
        """
        Initialize ROS2 communication interfaces for the controller.

        Depending on configuration, this will either create:

        - a `FollowJointTrajectory` :class:`ActionClient` (preferred), or
        - a `JointTrajectory` publisher, if `action` is ``None`` and `topic`
          is provided.

        The chosen interface is stored in:

        - ``self.command_action_client`` if using actions.
        - ``self.command_publisher`` if using topics.

        Raises
        ------
        ValueError
            If neither an action name nor a topic name is configured.

        Notes
        -----
        - This method assumes that `self._node` has been set by
          :meth:`RosControllerInterface.Activate`.
        - If the action server is not available within a short timeout,
          a warning is logged and `_use_action` remains ``False``.
        """
        self._use_publisher = False
        self._use_action = False

        # Prefer action client if configured
        if self._action is not None:
            if self._ros_plugin_name == "":
                topic_full_name = f"{self._namespace}/{self._action}"
            else:
                topic_full_name = f"{self._namespace}/{self._ros_plugin_name}/{self._action}"

            self._node.get_logger().info(f"Action topic: {topic_full_name}")
            self.command_action_client: ActionClient = ActionClient(self._node, FollowJointTrajectory, topic_full_name)
            self._node.get_logger().info(f"Waiting for '{topic_full_name}' action server ...")

            if not self.command_action_client.wait_for_server(timeout_sec=2.0):
                self.WarningMessage(f"Action {topic_full_name} not available")
            else:
                self._use_action = True

        elif self._topic is not None:
            if self._ros_plugin_name == "":
                topic_full_name = f"{self._namespace}/{self._topic}"
            else:
                topic_full_name = f"{self._namespace}/{self._ros_plugin_name}/{self._topic}"

            self._node.get_logger().info(f"Publishing to topic: {topic_full_name}")
            self.command_publisher = self._node.create_publisher(JointTrajectory, topic_full_name, 10)
            self._use_publisher = True
        else:
            raise ValueError("Topic or action for joint trajectory controller must be defined.")

    def _unregister_interface(self) -> None:
        """
        Tear down ROS2 interfaces and reset controller bindings.

        Destroys any created publishers or action clients and clears references
        to the associated robot and command/state buffers. After this call,
        the interface is no longer considered registered.

        Returns
        -------
        None

        Notes
        -----
        - This is typically called as part of a shutdown sequence or when
          switching to a different controller interface.
        - Any exceptions during destruction are caught and ignored.
        """
        if hasattr(self, "command_publisher"):
            try:
                self._node.destroy_publisher(self.command_publisher)
            except Exception:
                pass
            del self.command_publisher
        if hasattr(self, "command_action_client"):
            try:
                self._node.remove_waitable(self.command_action_client)
                self.command_action_client.destroy()
            except Exception:
                pass
            del self.command_action_client

        self.motion_done = None
        self._registered = False
        self._robot = None
        self._use_publisher = False
        self._use_action = False

    def GoTo_qtraj(self, q: JointPathType, qdot: JointPathType, qddot: JointPathType, time: TimesType) -> int:
        """
        Command the robot to follow a joint trajectory.

        Parameters
        ----------
        q : JointPathType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points.
        qdot : JointPathType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points.
        qddot : JointPathType
            Desired joint accelerations for the trajectory. The current
            interfaces do not use these values.
        time : TimesType
            Time points for the trajectory (n,).

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        The command is routed to either the publisher-based or action-based
        trajectory backend depending on the active ROS 2 interface.
        """
        if self._use_publisher:
            return self.GoTo_qtraj_pub(q, qdot, qddot, rbs_type(time).flatten())
        elif self._use_action:
            return self.GoTo_qtraj_act(q, qdot, qddot, rbs_type(time).flatten())
        else:
            raise ValueError("Joint trajectory controller is not ready.")

    def GoTo_qtraj_pub(self, q: JointPathType, qdot: JointPathType, qddot: JointPathType, time: TimesType) -> int:
        """
        Publish a joint trajectory through the ROS 2 topic interface.

        Parameters
        ----------
        q : JointPathType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points.
        qdot : JointPathType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points.
        qddot : JointPathType
            Desired joint accelerations for the trajectory. The publisher path
            does not use these values.
        time : TimesType
            Time points for the trajectory (n,).

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        On success, the final trajectory point is copied into the robot command
        state and the motion is marked complete immediately after publishing.
        """
        if isvector(q, dim=self.nj):
            q = rbs_type(q).reshape((1, self.nj))
            qdot = rbs_type(qdot).reshape((1, self.nj))
            if np.isscalar(time):
                time = np.array([float(time)])

        if ismatrix(q, shape=self.nj) and ismatrix(qdot, shape=self.nj) and isvector(time, dim=q.shape[0]):
            _trajectory_msg = JointTrajectory()
            _trajectory_msg.joint_names = self._robot.joint_names
            for qt, qdt, t in zip(q, qdot, time):
                _point_msg = JointTrajectoryPoint()
                _point_msg.positions = qt.astype(float).tolist()
                _point_msg.velocities = qdt.astype(float).tolist()
                _point_msg.time_from_start.sec = int(t)
                _point_msg.time_from_start.nanosec = int((t - int(t)) * 1e9)
                _trajectory_msg.points.append(_point_msg)
            self.command_publisher.publish(_trajectory_msg)
            self._robot._command.mode = CommandModeCodes.JOINT_TRAJ.value
            self.motion_done = True
            self._robot._command.q = q[-1]
            self._robot._command.qdot = np.zeros(self.nj)
            x = self._robot.Kinmodel(self._robot._command.q)[0]
            self._robot._command.x = x
            self._robot._command.v = np.zeros(6)
            return MotionResultCodes.MOTION_SUCCESS.value
        else:
            raise ValueError("Invalid joint trajectory data")

    def GoTo_qtraj_act(self, q: JointPathType, qdot: JointPathType, qddot: JointPathType, time: TimesType) -> int:
        """
        Send a joint trajectory through the ROS 2 action interface.

        Parameters
        ----------
        q : JointPathType
            Desired joint positions for the trajectory (n, nj), where n is the number of trajectory points.
        qdot : JointPathType
            Desired joint velocities for the trajectory (n, nj), where n is the number of trajectory points.
        qddot : JointPathType
            Desired joint accelerations for the trajectory. The action path
            does not use these values.
        time : TimesType
            Time points for the trajectory (n,).

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        The method waits for the action response and, when enabled, for the
        final motion-completion callback.
        """
        if isvector(q, dim=self.nj):
            q = rbs_type(q).reshape((1, self.nj))
            qdot = rbs_type(qdot).reshape((1, self.nj))
            if np.isscalar(time):
                time = np.array([float(time)])

        if ismatrix(q, shape=self.nj) and ismatrix(qdot, shape=self.nj) and isvector(time, dim=q.shape[0]):
            _trajectory_msg = JointTrajectory()
            _trajectory_msg.joint_names = self._robot.joint_names
            # Do NOT stamp header to avoid sim-time vs wall-time mismatch; let controller start immediately
            for qt, qdt, t in zip(q, qdot, time):
                _point_msg = JointTrajectoryPoint()
                _point_msg.positions = qt.astype(float).tolist()
                _point_msg.velocities = qdt.astype(float).tolist()
                _point_msg.time_from_start.sec = int(t)
                _point_msg.time_from_start.nanosec = int((t - int(t)) * 1e9)
                _trajectory_msg.points.append(_point_msg)
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = _trajectory_msg
            self.motion_done = False
            self.motion_result = None
            self._node.get_logger().info(f"FollowJointTrajectory goal: points={len(_trajectory_msg.points)} duration={time[-1]:.3f}s")
            try:
                qtraj_send_goal_future = self.command_action_client.send_goal_async(goal, feedback_callback=self.qtraj_feedback_callback)
                qtraj_send_goal_future.add_done_callback(self.qtraj_goal_response_callback)
            except Exception as e:
                self._node.get_logger().warning(f"FollowJointTrajectory send_goal_async exception: {e}")
                return GoalStatus.STATUS_CANCELED

            # Waiting for motion result (timeout expected motion time)
            self._node.get_logger().debug("Waiting for FollowJointTrajectory motion result ...")
            waited = 0
            while self.motion_result is None:  # and waited < np.max(t):
                sleep(self._robot.tsamp)
                waited += self._robot.tsamp
                self._robot.Update()
                if waited > np.max(time):
                    self._node.get_logger().debug("Timeout for FollowJointTrajectory motion result!")
                    return MotionResultCodes.NO_RESPONSE.value

            # Waiting further for motion done reported (maximal 2 x expected motion time)
            if self._robot._wait_for_action_server:
                self._node.get_logger().debug("Waiting for FollowJointTrajectory motion done ...")
                while not self.motion_done and waited < 2 * np.max(time):  # and waited < np.max(t):
                    sleep(self._robot.tsamp)
                    waited += self._robot.tsamp
                    self._robot.Update()

            return self.motion_result
        else:
            raise ValueError("Invalid joint trajectory data")

    def qtraj_goal_response_callback(self, future: Future) -> None:
        """
        Callback executed when the action server responds to a trajectory goal request.

        Parameters
        ----------
        future : Future
            Future containing the ``ClientGoalHandle`` result from the ``send_goal_async`` call.

        Notes
        -----
        - If the goal is rejected, ``motion_result`` is set to
          ``INVALID_GOAL``.
        - If the goal is accepted, the completion callback is registered
          through ``get_result_async``.
        """
        self._node.get_logger().debug("goal_response_callback invoked")

        # print type of self
        self._node.get_logger().debug(f"type(self) = {type(self)}")
        # print if self has message method
        self._node.get_logger().debug(f"hasattr(self, 'Message') = {hasattr(self, 'Message')}")
        # print verbositz level of self if hasattr(self, "_verbose"):
        self._node.get_logger().debug("GoTo_qtraj: Goal response received")

        self._goal_handle = future.result()
        if not self._goal_handle or not self._goal_handle.accepted:
            self._node.get_logger().error("FollowJointTrajectory response: Goal rejected!")
            self.motion_result = FollowJointTrajectory.Result.INVALID_GOAL
        else:
            self._node.get_logger().info("FollowJointTrajectory response: Goal accepted")
            self.motion_result = FollowJointTrajectory.Result.SUCCESSFUL
            self._robot._command.mode = CommandModeCodes.JOINT_TRAJ.value
            self._goal_handle.get_result_async().add_done_callback(self.qtraj_get_result_callback)

    def qtraj_get_result_callback(self, future: Future) -> None:
        """
        Callback executed when the trajectory has finished execution.

        Parameters
        ----------
        future : Future
            Future containing the result of the action execution.

        Notes
        -----
        - Updates ``motion_result`` with the final action result error code.
        - Sets ``motion_done`` to ``True`` indicating motion completion.
        - Logs success or error based on trajectory outcome.
        - If motion was aborted, sets ``_abort = True`` on the robot instance (if available).

        Returns
        -------
        None
        """
        self._node.get_logger().debug("Trajecotry result callback invoked")
        wrapped = future.result()
        status = wrapped.status
        result = wrapped.result
        code = result.error_code
        if code == 0:
            self._node.get_logger().info(f"FollowJointTrajectory result: {result.error_string}")
        else:
            self._node.get_logger().error(f"FollowJointTrajectory result: {result.error_string}")
        self.motion_result = code
        self.motion_done = True
        self._robot._command.mode = CommandModeCodes.JOINT_TRAJ.value
        if status == GoalStatus.STATUS_ABORTED:
            if hasattr(self, "_robot"):
                setattr(self._robot, "_abort", True)

    def qtraj_feedback_callback(self, feedback_msg: FollowJointTrajectory.Feedback) -> None:
        """
        Callback executed when feedback is received during trajectory execution.

        Parameters
        ----------
        feedback_msg : FollowJointTrajectory.Feedback
            Feedback message containing intermediate trajectory updates.

        Notes
        -----
        - Stores feedback data in ``last_feedback``.
        - Feedback may include joint states and progress updates.
        - Logging is optional and typically kept disabled for performance reasons.

        Returns
        -------
        None
        """
        # self._node.get_logger().debug("Trajectory feedback callback invoked")
        q = np.array(feedback_msg.feedback.desired.positions)
        qd = np.array(feedback_msg.feedback.desired.velocities)
        self._robot._command.q = q
        x, J = self._robot.Kinmodel(q)
        self._robot._command.x = x
        if len(qd) > 0:
            self._robot._command.qd = qd
            try:
                self._robot._command.v = J @ qd
            except Exception as e:
                self._node.get_logger().error(f"FollowJointTrajectory feedback: {e}\nq  = {q}\nqd = {qd}\nJ=\n{J}")

        self.last_feedback = feedback_msg.feedback

    def abort_motion(self) -> int | None:
        """
        Attempt to abort the current trajectory motion by canceling the active FollowJointTrajectory goal.

        This method sends a cancel request to the ROS2 action server for the active trajectory goal.
        If no goal is currently active, a warning is logged and the method exits without performing any action.

        Notes
        -----
        - This does not immediately stop the robot; it requests the controller
          to perform a controlled stop.
        - After the cancel request is sent, the result is processed in
          :meth:`qtraj_cancel_done_callback`.

        Returns
        -------
        int | None
            Motion result code of the cancel operation, or ``None`` if no
            active goal exists.
        """
        if not hasattr(self, "_goal_handle") or self._goal_handle is None:
            self._node.get_logger().warn("No active FollowJointTrajectory goal to abort.")
            return

        self.cancel_response = None
        self._node.get_logger().info("FollowJointTrajectory goal canceling ...")
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.qtraj_cancel_done_callback)

        waited = 0
        while self.cancel_response is None:  # and waited < np.max(t):
            sleep(self._robot.tsamp)
            waited += self._robot.tsamp
            self._robot.Update()
            # Timeout 2s
            if waited > np.max(2.0):
                self._node.get_logger().debug("Timeout for FollowJointTrajectory cancel motion result!")
                return MotionResultCodes.NO_RESPONSE.value

        return self.cancel_response

    def qtraj_cancel_done_callback(self, future: Future) -> None:
        """
        Process the result of a cancel request for an active trajectory goal.

        Parameters
        ----------
        future : Future
            Future object returned from ``cancel_goal_async`` containing the cancel response.

        Notes
        -----
        - Logs whether cancellation was successful.
        - Does not raise exceptions; any failure is logged.
        - This callback does not wait for the robot to fully stop; it only
          indicates whether the cancel request was accepted.
        """
        self.motion_done = True
        _result = future.result()
        if _result and len(_result.goals_canceling) > 0:
            self._node.get_logger().info("FollowJointTrajectory goal successfully canceled.")
            self.cancel_response = MotionResultCodes.MOTION_ABORTED.value
            self._robot._command.mode = CommandModeCodes.JOINT_TRAJ.value
        else:
            self._node.get_logger().warn("Failed to cancel FollowJointTrajectory goal.")
            self.cancel_response = MotionResultCodes.MOTION_FAILURE.value

    @staticmethod
    def FollowJointTrajectory_error_str(error_code: int) -> str:
        """
        Convert a ``FollowJointTrajectory`` result code to text.

        Parameters
        ----------
        error_code : int
            Error code from ``FollowJointTrajectory`` action result.

        Returns
        -------
        str
            Human-readable explanation of the failure cause.
        """
        FOLLOWJOINTTRAJECTORY_STATUS_MAP = {
            FollowJointTrajectory.Result.SUCCESSFUL: "Trajectory executed successfully.",
            FollowJointTrajectory.Result.INVALID_GOAL: "The trajectory goal is invalid (e.g., unreachable target or invalid constraints).",
            FollowJointTrajectory.Result.INVALID_JOINTS: "The trajectory references unknown or mismatched joint names.",
            FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: "The command timestamp is older than the robot’s current time.",
            FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: "The robot deviated from the desired trajectory beyond allowed path tolerance.",
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: "The final joint or pose error exceeded the goal tolerance.",
        }
        return FOLLOWJOINTTRAJECTORY_STATUS_MAP.get(error_code, f"Unknown FollowJointTrajectory error code: {error_code}")

    @staticmethod
    def FollowJointTrajectory_goal_status_str(status_code: int) -> str:
        """
        Convert an action goal status code to text.

        Parameters
        ----------
        status_code : int
            Action goal execution status.

        Returns
        -------
        str
            Human-readable status description.
        """
        STATUS_MAP = {
            GoalStatus.STATUS_UNKNOWN: "Goal status unknown (controller did not report a valid state).",
            GoalStatus.STATUS_ACCEPTED: "Goal accepted and queued for execution.",
            GoalStatus.STATUS_EXECUTING: "Goal is currently being executed.",
            GoalStatus.STATUS_CANCELING: "Goal is being canceled.",
            GoalStatus.STATUS_SUCCEEDED: "Goal execution finished successfully.",
            GoalStatus.STATUS_CANCELED: "Goal was canceled before finishing.",
            GoalStatus.STATUS_ABORTED: "Goal execution was aborted due to an error or external interruption.",
        }
        return STATUS_MAP.get(status_code, f"Unknown FollowJointTrajectory goal status: {status_code}")
