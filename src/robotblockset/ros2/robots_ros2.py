"""ROS 2 robot interface implementations.

This module defines ROS 2-backed robot wrappers used by RobotBlockSet robot
interfaces. It provides communication plumbing, controller integration, state
feedback handling, and motion-execution helpers for multiple robot platforms.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah, Mihael Simonic.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Optional, Any, Dict, List
import numpy as np
from time import sleep, time
from copy import deepcopy
from threading import Thread

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import WrenchStamped
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if all are installed.")

from robotblockset.robots import robot, MotionResultCodes
from robotblockset.tools import rbs_type, isvector, vector
from robotblockset.ros2.controllers_ros2 import controller_manager_helper
from robotblockset.ros2.controllers_ros2 import JointTrajectoryControllerInterface, RosControllerInterface
from robotblockset.rbs_typing import JointConfigurationType, JointVelocityType, JointTorqueType, Pose3DType, Velocity3DType, WrenchType, JointPathType, TimesType


class robot_ros2(robot):
    """
    Base class for ROS 2-backed robot interfaces.

    Attributes
    ----------
    Name : str
        Name of the ROS 2 node and RobotBlockSet robot instance.
    _namespace : str
        Normalized ROS namespace prefix used for topics, actions, and services.
    controller : RosControllerInterface | None
        Active controller interface bound to the robot.
    joint_state : JointState | None
        Last received joint-state message.
    force_state : WrenchStamped | None
        Last received force/torque message.
    """

    def __init__(self, name: str, namespace: str = "", joint_states_topic: Optional[str] = "joint_states", control_strategy: Optional[str] = None, strategy_to_controller_interface_mapping: Optional[Dict[str, RosControllerInterface]] = None, node: Optional[Node] = None, control_manager_name: Optional[str] = "controller_manager") -> None:
        """
        Initialize a ROS 2 robot interface.

        Parameters
        ----------
        name : str
            Name of the robot node.
        namespace : str, optional
            Namespace for the robot topics, actions, and services.
        joint_states_topic : str | None, optional
            Relative topic used to subscribe to joint states. If ``None``, no
            joint-state subscription is created.
        control_strategy : str | None, optional
            Initial RobotBlockSet control strategy.
        strategy_to_controller_interface_mapping : dict[str, RosControllerInterface] | None, optional
            Mapping from strategy names to controller interface instances.
        node : Node | None, optional
            Optional externally created ROS 2 node to use. If ``None``, a new node is created with the given name.
        control_manager_name : str, optional
            Name of the ROS 2 controller manager.

        Notes
        -----
        Subclasses may create additional subscriptions, clients, and publishers
        before calling :meth:`_start_spinning`.
        """
        robot.__init__(self)
        if node is None:
            self._node = Node(name)
            self._owns_node = True
        else:
            self._node = node
            self._owns_node = False
        if namespace is None or namespace == "":
            self._namespace = ""
        else:
            self._namespace = "/" + str(namespace).strip("/")

        # Robot name for logging
        self.Name = name

        # Mapping of RBS strategies to ros_control controller interfaces
        self._strategy_to_controller_interface_mapping = strategy_to_controller_interface_mapping or {}
        self._control_strategy = control_strategy
        self.controller = None

        # Create the executor
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        # Initialize the control helper
        if self._strategy_to_controller_interface_mapping == {}:
            self._node.get_logger().info("No Strategy<->Control_interface mapping  and control manager is not used")
            self._control_helper = None
        else:
            self._control_helper = controller_manager_helper(self._node, namespace=namespace, controller_manager_name=control_manager_name)

        # Spin control
        self._spinning = False
        self._spin_thread = None

        # Connection and state variables
        self._connected = False

        self.joint_state = None
        self._last_joint_state_callback_time = 0
        self.force_state = None
        self._last_force_state_callback_time = 0

        self._reorder = None
        self._wait_for_action_server = True  # Motion command wait to get Action server response

        # Initialize ROS2 subscription to joint states
        if joint_states_topic is not None:
            topic_full_name = f"{self._namespace}/{joint_states_topic.strip('/')}"
            self._node.get_logger().info(f"Subscribing to joint states topic: {topic_full_name}")
            self._joint_states_sub = self._node.create_subscription(msg_type=JointState, topic=topic_full_name, callback=self._joint_state_callback, qos_profile=qos_profile_sensor_data)

        # We don't start spinning here yet. Subclasses may add more subs/clients first to avoid
        # concurrent entity mutations during wait set building. They should call
        # self._start_spinning(wait_for_state=True) when ready.

        # Subclasses should also SetStrategy after spinning starts.

    def __del__(self) -> None:
        """
        Clean up ROS 2 resources during object destruction.

        Notes
        -----
        Python may call ``__del__`` during interpreter shutdown when logging and
        ROS resources are already partially destroyed. All shutdown actions are
        therefore wrapped in ``try`` blocks.
        """
        try:
            # Logging may fail during interpreter shutdown
            try:
                if hasattr(self, "_node") and self._node is not None:
                    self._node.get_logger().info(f"{self.Name} object is being deleted...")
            except Exception:
                pass

            # Perform full shutdown of executors, thread, and node
            try:
                self._shutdown()
            except Exception:
                pass

            try:
                if hasattr(self, "_node") and self._node is not None:
                    self._node.get_logger().info(f"{self.Name} deletion complete.")
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
        Execute the background ROS 2 spin loop.

        Notes
        -----
        The loop repeatedly calls ``spin_once`` on the shared executor so that
        callback processing remains responsive to shutdown requests.
        Transient errors from entity destruction (e.g. when switching controllers)
        are caught per-iteration so the loop continues rather than exiting.
        """
        while rclpy.ok() and getattr(self, "_spinning", False):
            self._executor.spin_once(timeout_sec=0.1)

    def _start_spinning(self, wait_for_state: bool = True, timeout_sec: float = 10.0) -> None:
        """
        Start the executor thread and optionally wait for initial state feedback.

        Parameters
        ----------
        wait_for_state : bool, optional
            Whether to wait for the first joint state message to arrive.
        timeout_sec : float, optional
            Timeout in seconds when waiting for the first joint state.
        Returns
        -------
        None

        Notes
        -----
        If ``wait_for_state`` is enabled and no joint state arrives within the
        timeout, the robot remains marked as disconnected.
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
            while self.joint_state is None and waited < timeout_sec:
                sleep(step)
                waited += step
            if self.joint_state is None:
                self._node.get_logger().warning("No joint state received within timeout; continuing without connection confirmation.")
                self._connected = False
            else:
                self._connected = True
                self._node.get_logger().info("Connected")

        # Apply desired strategy now that spinning is active
        if self._control_strategy is not None:
            self.SetStrategy(self._control_strategy)
        else:
            self.controller = None
            self._control_strategy = None
            self._node.get_logger().warning("Initializing robot object without a controller.")

    def _shutdown(self, join_timeout: float = 2.0) -> None:
        """
        Stop spinning and release ROS 2 resources.

        Parameters
        ----------
        join_timeout : float, optional
            Seconds to wait for the spin thread to finish.

        Returns
        -------
        None

        Notes
        -----
        This method stops the background thread, shuts down the controller
        helper, removes the node from the executor, and destroys the node.
        """
        # Stop spin loop and join background thread
        try:
            self._spinning = False
            if self._spin_thread is not None and self._spin_thread.is_alive():
                self._node.get_logger().info("Waiting for ROS2 spin thread to finish...")
                self._spin_thread.join(timeout=join_timeout)
                if self._spin_thread.is_alive():
                    self._node.get_logger().warning("ROS2 spin thread did not exit within timeout")
        except Exception as e:
            try:
                self._node.get_logger().error(f"Error while stopping spin thread: {e}")
            except Exception:
                pass

        # Shut down control helper
        try:
            if hasattr(self, "_control_helper") and self._control_helper is not None:
                self._node.get_logger().info("Shutting down control helper...")
                try:
                    self._control_helper._shutdown()
                except Exception as e:
                    self._node.get_logger().error(f"Error while shutting down control helper: {e}")
        except Exception:
            # Swallow any logging issues
            pass

        # Remove node from executor and shut it down
        try:
            if hasattr(self, "_executor") and self._executor is not None:
                self._node.get_logger().info("Removing node from executor...")
                try:
                    self._executor.remove_node(self._node)
                except Exception:
                    # It's fine if the node was already removed
                    pass

                self._node.get_logger().info("Shutting down executor...")
                try:
                    self._executor.shutdown()
                except Exception as e:
                    self._node.get_logger().error(f"Error while shutting down executor: {e}")
        except Exception:
            # Again, avoid raising during shutdown
            pass

        # Destroy node
        try:
            if hasattr(self, "_node") and self._node is not None:
                self._node.get_logger().info("Destroying node...")
                if self._owns_node:
                    self._node.destroy_node()
        except Exception as e:
            try:
                if hasattr(self, "_node") and self._node is not None:
                    self._node.get_logger().error(f"Error while destroying node: {e}")
            except Exception:
                pass

    # States
    def _joint_state_callback(self, data: JointState) -> None:
        """
        Store the latest joint-state message.

        Parameters
        ----------
        data : JointState
            Joint-state message received from ROS 2.
        """
        self.joint_state = deepcopy(data)
        self._last_joint_state_callback_time = self.simtime()

    def _force_state_callback(self, data: WrenchStamped) -> None:
        """
        Store the latest force/torque message.

        Parameters
        ----------
        data : WrenchStamped
            Force/torque message received from ROS 2.
        """
        self.force_state = data
        self._last_force_state_callback_time = self.simtime()

    def GetState(self) -> None:
        """
        Update the RobotBlockSet state from the latest ROS 2 messages.

        Notes
        -----
        Joint state, force/torque data, and controller reference state are
        copied into the RobotBlockSet ``_actual`` and ``_command`` containers.
        """
        if self.joint_state is not None:
            if self._reorder is None:
                self._reorder = [self.joint_state.name.index(a) for a in self.joint_names if a in self.joint_state.name]

            pos = [self.joint_state.position[i] for i in self._reorder]
            vel = [self.joint_state.velocity[i] for i in self._reorder]
            trq = [self.joint_state.effort[i] for i in self._reorder]

            self._tt = self.simtime()
            self._actual.q = rbs_type(pos)
            self._actual.qdot = rbs_type(vel)
            self._actual.trq = rbs_type(trq)

            x, J = self.Kinmodel()
            self._actual.x = x
            self._actual.v = J @ self._actual.qdot

        if self.force_state:
            _f = self.force_state.wrench.force
            _t = self.force_state.wrench.torque
            self._actual.FT = rbs_type([_f.x, _f.y, _f.z, _t.x, _t.y, _t.z])

        self._last_update = self.simtime()  # Do not change !

    # Strategies
    def AvailableStrategies(self) -> List[str]:
        """
        Return the available control strategies for the robot.

        Returns
        -------
        List[str]
            A list of available control strategies.
        """
        return list(self._strategy_to_controller_interface_mapping.keys())

    def SetStrategy(self, new_strategy: str) -> None:
        """
        Switch the active control strategy.

        Parameters
        ----------
        new_strategy : str
            Name of the control strategy to activate.

        Returns
        -------
        None

        Notes
        -----
        The current controller is stopped and unregistered before the new ROS 2
        controller is loaded, switched, and activated.

        Raises
        ------
        ValueError
            If the specified strategy is not supported.
        """
        if new_strategy not in self.AvailableStrategies():
            raise ValueError(f"Strategy '{new_strategy}' not supported")

        # Check if any of the possible controllers is already active
        ci = self._strategy_to_controller_interface_mapping
        possible_controllers_names = [ci[s]._ros_plugin_name for s in self.AvailableStrategies()]
        loaded_controllers = self._control_helper.list_controllers()

        # Identify active controllers that claim actuation interfaces.
        # Note: ROS allows multiple controllers to run concurrently (e.g., state broadcasters),
        # and multiple controllers may claim different types of interfaces for the same joint.
        # However, actuation interfaces (e.g., joint_x) must be exclusively owned — only one controller
        # can claim a given actuation interface at a time.
        # Other interfaces may still be claimed by other controllers without conflict.

        active_controller_names = [c.name for c in loaded_controllers if c.name in possible_controllers_names and c.state == "active" and len(c.claimed_interfaces) > 0]

        # ... but we don't handle this yet
        if len(active_controller_names) > 1:
            raise RuntimeError("Undefined behavior: Multiple active controllers claiming actuation interfaces detected! " f"Active controllers: {active_controller_names}")

        # Make reverse mapping from controller name to strategy
        if self._control_strategy is not None and self.controller is not None:
            # Make sure internal state is consistent with the actual active controller
            if len(active_controller_names) > 0:
                # ToDo : poglej:self._control_strategy = active_controller_names[0]
                active_controller_name = active_controller_names[0]
                reverse_map = {iface._ros_plugin_name: strategy for strategy, iface in ci.items()}
                self._control_strategy = reverse_map.get(active_controller_name, "")
            else:
                self._control_strategy = ""

            # Check if already using the desired strategy.
            # Also verify the controller object matches, because two strategies may share
            # the same ros_plugin_name (e.g. "JointPosition" and "JointPositionTrajectory"
            # both use "fr3_arm_controller") — the reverse_map would collapse them to one
            # entry, so we must confirm the actual interface object is correct too.
            if self._control_strategy == new_strategy and self.controller is not None and self.controller is ci.get(new_strategy):
                self.Message(f"Not switching; already using '{new_strategy}'", 2)
                return

        # Stop motion before switching.  For action-backed controllers, first
        # abort goals.
        try:
            if self._control_strategy is not None:
                self.Abort()
                self.Stop()
        except Exception:
            # Non-critical if semaphore not yet initialized.
            pass

        # Snapshot the feedback state as the new command target before the old
        # interface is torn down.
        self.ResetCurrentTarget(do_move=False)

        # Safely unload current controller interface
        if self.controller is not None and self.controller._registered:
            self.controller._unregister_interface()
            self.controller = None

        # Check if the target controller is loaded
        target_controller_name = ci.get(new_strategy)._ros_plugin_name
        loaded_controllers = self._control_helper.list_controllers()
        if target_controller_name not in [c.name for c in loaded_controllers]:
            if not self._control_helper.load_controller(target_controller_name):
                raise ValueError(f"ROS2 controller '{target_controller_name}' can not be loaded!")

        # Prepare switch request
        if target_controller_name not in active_controller_names:
            if not self._control_helper.switch_controllers(stop_list=active_controller_names, start_list=[target_controller_name]):
                self.Message("Switching failed. Check ROS2 logs!", 0)
                return False

        # Activate controller interface
        controller_instance = ci.get(new_strategy)
        self.controller = controller_instance
        self.controller.Activate(robot=self, node=self._node)
        self._control_strategy = new_strategy
        self.Message(f"Strategy set to '{new_strategy}'", 2)

    # Status
    def isConnected(self) -> bool:
        """
        Check whether the robot has received initial state feedback.

        Returns
        -------
        bool
            ``True`` if the robot is connected, otherwise ``False``.
        """
        return self._connected

    def Shutdown(self, join_timeout: float = 2.0) -> None:
        """
        Shut down the robot node and its background executor.

        Parameters
        ----------
        join_timeout : float, optional
            Seconds to wait for the background spin thread to stop.
        """
        self._shutdown(join_timeout=join_timeout)

    # Movements
    def StopMotion(self) -> Optional[int]:
        """
        Stop the current motion and reset the local target state.

        Returns
        -------
        int | None
            Result code from the controller's abort call, if available.

        Notes
        -----
        The robot state is switched to stop mode and the current RobotBlockSet
        target is reset even when the controller does not provide an explicit
        abort method.
        """
        _res = None
        if hasattr(self.controller, "abort_motion"):
            _res = self.controller.abort_motion()
        self.Stop()
        self.ResetCurrentTarget()
        return _res

    def EnableWaitForActionServer(self, check: bool = True) -> None:
        """
        Enable waiting for action completion callbacks.

        Parameters
        ----------
        check : bool, optional
            Whether to enable check. Default is True.

        Returns
        -------
        None
        """
        self._wait_for_action_server = check

    def DisableWaitForActionServer(self) -> None:
        """
        Disable waiting for action completion callbacks.
        """
        self._wait_for_action_server = False

    def WaitUntilDone(self, timeout: Optional[float] = None) -> int:
        """
        Wait until the controller reports motion completion.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds. If ``None``, wait indefinitely.

        Returns
        -------
        int
            Motion result code reported by the controller.

        Notes
        -----
        This method polls ``self.controller.motion_done`` and updates the robot
        state while waiting. It is a blocking helper and should not be used from
        real-time or callback threads.
        """
        # If no motion_done attribute, assume completed
        if not hasattr(self.controller, "motion_done") or self.controller.motion_done is None:
            return MotionResultCodes.MOTION_SUCCESS.value
        elif self.controller.motion_done:
            return self.controller.motion_result

        start_time = time()

        # Loop until motion completes or timeout expires
        while not self.controller.motion_done:
            if timeout is not None and (time() - start_time) >= timeout:
                return MotionResultCodes.NO_RESPONSE.value  # Timed out
            sleep(self.tsamp)
            self.Update()

        return self.controller.motion_result  # Completed

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Command the robot to move to a joint configuration.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        qdot : JointVelocityType, optional
            Desired joint velocities (nj,).
        trq : JointTorqueType, optional
            Desired joint torques (nj,).
        wait : float, optional
            Time step used for synchronization and command execution. If
            ``None``, ``self.tsamp`` is used.

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        If the active controller does not implement direct joint commands, the
        method falls back to a two-point joint trajectory.

        Raises
        ------
        ValueError
            If control is not implemented in controller.
        """
        self._synchro_control(wait)
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

        self._synchro_control(wait)
        if hasattr(self.controller, "GoTo_q"):
            res = self.controller.GoTo_q(q, qdot, trq, wait)
        elif hasattr(self.controller, "GoTo_qtraj"):
            # Two points are needed. We use last commanded q ar first point
            qq = np.vstack((self._actual.q, q))
            qqdot = np.vstack((qdot, 0 * qdot))
            time = np.hstack((0, wait))
            res = self.controller.GoTo_qtraj(qq, qqdot, np.zeros(q.shape), time)
        else:
            raise NotImplementedError("GoTo_q method not implemented for the current controller")
        if res == 0:
            self._command.q = q
            self._command.qdot = qdot if qdot is not None else np.zeros(self.nj)
            x, J = self.Kinmodel(q)
            self._command.x = x
            self._command.v = J @ self._command.qdot
        self.Update()
        return res

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
            Desired joint accelerations for the trajectory (n, nj). Not used in ROS2 controller.
        time : TimesType
            Time points for the trajectory (n,).

        Returns
        -------
        int
            Controller status code (0 for success).

        Notes
        -----
        On success, the final trajectory point is copied into the RobotBlockSet
        command state.

        Raises
        ------
        ValueError
            If control is not implemented in controller.
        """
        if hasattr(self.controller, "GoTo_qtraj"):
            _stat = self.controller.GoTo_qtraj(q, qdot, qddot, time)
            if _stat == 0:
                self._command.q = q[-1]
                self._command.qdot = np.zeros(self.nj)
                x = self.Kinmodel(self._command.q)[0]
                self._command.x = x
                self._command.v = np.zeros(6)

            return _stat
        else:
            raise NotImplementedError("GoTo_qtraj_pub method not implemented for the current controller")

    def GoTo_X(self, x: Pose3DType, xdot: Velocity3DType, trq: WrenchType, wait: float, **kwargs: Any) -> int:
        """
        Command the robot in Cartesian space.

        Parameters
        ----------
        x : Pose3DType
            Desired end-effector pose.
        xdot : Velocity3DType
            Desired end-effector twist.
        trq : WrenchType
            Desired end-effector wrench.
        wait : float
            Time step used for controller synchronization.
        **kwargs : Any
            Additional controller-specific keyword arguments.

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        Successful commands update the RobotBlockSet Cartesian command state.

        Raises
        ------
        ValueError
            If control is not implemented in controller.
        """
        if hasattr(self.controller, "GoTo_X"):
            if not isvector(x, dim=7):
                raise Exception("%s: GoTo_x: NAN x value" % self.Name)
            if not isvector(xdot, dim=6):
                raise Exception("%s: GoTo_x: NAN xdot value" % self.Name)
            if not isvector(trq, dim=6):
                raise Exception("%s: GoTo_x: NAN trq value" % self.Name)
            self.Message(f"pose: {x}, vel: {xdot}, trq: {trq}", 4)

            res = self.controller.GoTo_X(x, xdot, trq, wait, **kwargs)
            if res == MotionResultCodes.MOTION_SUCCESS.value:
                self._command.x = x
                self._command.v = xdot
                self._command.FT = trq
            
            self.Update()
            #self._command.q = self._actual.q  # Keep current joint state as target
            #self._command.qdot = np.zeros(self.nj)  # Clear joint velocity command
            self._synchro_control(self.tsamp)
            return res
        else:
            raise NotImplementedError("GoTo_X method not implemented for the current controller")

    def GetCartesianCompliance(self) -> Any:
        """Return Cartesian compliance parameters from the active controller."""
        if hasattr(self.controller, "GetCartesianCompliance"):
            return self.controller.GetCartesianCompliance()
        raise NotImplementedError("Cartesian compliance not implemented for the current controller")

    def SetCartesianCompliance(
        self,
        Kp: Optional[Any] = None,
        Kr: Optional[Any] = None,
        R: Optional[Any] = None,
        D: Optional[float] = None,
        hold_pose: bool = True,
    ) -> int:
        """Update Cartesian compliance through the active controller."""
        if hasattr(self.controller, "SetCartesianCompliance"):
            return self.controller.SetCartesianCompliance(Kp=Kp, Kr=Kr, R=R, D=D, hold_pose=hold_pose)
        raise NotImplementedError("Cartesian compliance not implemented for the current controller")

    def GetCartesianStiffness(self) -> Any:
        """Return Cartesian stiffness parameters from the active controller."""
        if hasattr(self.controller, "GetCartesianStiffness"):
            return self.controller.GetCartesianStiffness()
        raise NotImplementedError("Cartesian stiffness not implemented for the current controller")

    def SetCartesianStiffness(self, stiffness: Any, hold_pose: bool = True, **kwargs: Any) -> int:
        """Update Cartesian stiffness through the active controller.

        Additional keyword arguments are forwarded to the controller
        (for example ``R`` and ``D`` for Cartesian impedance controllers).
        """
        if hasattr(self.controller, "SetCartesianStiffness"):
            return self.controller.SetCartesianStiffness(stiffness=stiffness, hold_pose=hold_pose, **kwargs)
        raise NotImplementedError("Cartesian stiffness not implemented for the current controller")

    def GetCartesianDamping(self) -> Any:
        """Return Cartesian damping parameters from the active controller."""
        if hasattr(self.controller, "GetCartesianDamping"):
            return self.controller.GetCartesianDamping()
        raise NotImplementedError("Cartesian damping not implemented for the current controller")

    def SetCartesianDamping(self, D: Optional[float] = None, hold_pose: bool = True) -> int:
        """Update Cartesian damping through the active controller."""
        if hasattr(self.controller, "SetCartesianDamping"):
            return self.controller.SetCartesianDamping(D=D, hold_pose=hold_pose)
        raise NotImplementedError("Cartesian damping not implemented for the current controller")

    def SetCartesianSoft(self, stiffness: Any, hold_pose: bool = True) -> int:
        """Scale Cartesian compliance through the active controller."""
        if hasattr(self.controller, "SetCartesianSoft"):
            return self.controller.SetCartesianSoft(stiffness, hold_pose=hold_pose)
        raise NotImplementedError("Cartesian softness not implemented for the current controller")

    def SetCartesianStiff(self, hold_pose: bool = True) -> int:
        """Set Cartesian stiffness to the active controller's stiff defaults."""
        if hasattr(self.controller, "SetCartesianStiff"):
            return self.controller.SetCartesianStiff(hold_pose=hold_pose)
        return self.SetCartesianSoft(1.0, hold_pose=hold_pose)

    def SetCartesianCompliant(self, hold_pose: bool = True) -> int:
        """Set Cartesian stiffness to compliant mode on the active controller."""
        if hasattr(self.controller, "SetCartesianCompliant"):
            return self.controller.SetCartesianCompliant(hold_pose=hold_pose)
        return self.SetCartesianSoft(0.0, hold_pose=hold_pose)

    @staticmethod
    def MotionResultStr(code: int) -> str:
        """
        Convert a motion result code to a human-readable description.

        Parameters
        ----------
        code : int
            The  `result code` value.

        Returns
        -------
        str
            A human-readable result code string.
        """
        # Handle success case
        if code == 0:
            return "Trajectory execution successful"

        if code < 0:
            _msg = JointTrajectoryControllerInterface.FollowJointTrajectory_error_str(code)
        else:
            _msg = robot.MotionResultStr(code)

        return _msg
