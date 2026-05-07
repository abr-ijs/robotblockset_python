"""ROS 2 gripper interface implementations.

This module defines ROS 2-backed gripper wrappers used by RobotBlockSet robot
interfaces.

Authors: Mihael Simonic
"""

from time import sleep
from threading import Thread
from typing import Any, Optional

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed or sourced.")

try:
    from franka_msgs.action import Grasp, Homing, Move
    from std_srvs.srv import Trigger
    from sensor_msgs.msg import JointState
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 messages. Check if franka_msgs and std_srvs are installed.")

from robotblockset.grippers import gripper
from robotblockset.robots import robot


class FrankaGripper(gripper):
    """
    ROS 2 action-based interface for the Franka gripper (Panda / FR3).

    Uses ``franka_msgs`` action servers for homing, moving, and grasping,
    and a ``std_srvs/Trigger`` service for stopping.

    Can be used standalone (``robot=None``) or attached to an existing
    ``robot`` instance.

    Attributes
    ----------
    Name : str
        Identifier for the gripper instance.
    GripperTagNames : str
        Tag used for identifying the gripper in logs or systems.
    Robot : robot or None
        The robot associated with the gripper, or ``None`` if standalone.
    """

    def __init__(self, robot: Optional[robot] = None, namespace: str = "franka_gripper", **kwargs: Any) -> None:
        """
        Initialize the Franka gripper ROS 2 wrapper.

        Parameters
        ----------
        robot : robot, optional
            Robot instance to which the gripper is attached.  Must be an
            ``robot`` subclass  instance.
            If ``None``, a minimal internal node is created and spun in a
            background thread so the gripper can be used standalone.
        namespace : str, optional
            ROS 2 namespace the ``franka_gripper_node`` runs in.
            The node is always named ``franka_gripper``, so topics resolve to
            ``/{namespace}/franka_gripper/{action}``.  Defaults to
            ``"franka_gripper"`` which matches the standard launch setup.
        **kwargs : Any
            Additional keyword arguments for future extensions or configuration.
        """
        self.Name = "Franka:Gripper:ROS2"
        self.GripperTagNames = "gripper"
        self.Robot = robot
        self._namespace = namespace.strip("/")
        self._width_grasp = 0
        self._width = 0
        self._width_max = 0.08
        self._speed = 0.0
        self._speed_max = 0.5
        self._verbose = 1
        self._state = -1
        self._own_node = False  # True when we created the node ourselves

        # Resolve the ROS 2 node used for creating clients.
        if robot is not None and hasattr(robot, "_node"):
            self._node: Node = robot._node
        elif robot is not None and isinstance(robot, Node):
            self._node = robot
        elif robot is None:
            self._node = Node("franka_gripper_client")
            self._own_node = True
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()
        else:
            raise TypeError("robot must be a robot_ros2 instance or None.")

        # Build namespaced topic prefix.
        # The franka_gripper_node is always named 'franka_gripper', so
        # topics resolve to /{namespace}/franka_gripper/{action}.
        _ns = f"/{self._namespace}" if self._namespace else ""
        self._topic_homing = f"{_ns}/franka_gripper/homing"
        self._topic_grasp = f"{_ns}/franka_gripper/grasp"
        self._topic_move = f"{_ns}/franka_gripper/move"
        self._topic_stop = f"{_ns}/franka_gripper/stop"
        self._topic_joint_states = f"{_ns}/franka_gripper/joint_states"

        # Action clients
        self._client_homing = ActionClient(self._node, Homing, self._topic_homing)
        self._client_grasp = ActionClient(self._node, Grasp, self._topic_grasp)
        self._client_move = ActionClient(self._node, Move, self._topic_move)

        # Stop service client
        self._client_stop = self._node.create_client(Trigger, self._topic_stop)

        # Joint-state subscriber for finger width feedback
        self._joint_states_sub = self._node.create_subscription(
            JointState,
            self._topic_joint_states,
            self._joint_state_callback,
            10,
        )

        self.Message("Created", 2)

    def __del__(self) -> None:
        """Shut down the internal node and executor when used standalone."""
        if getattr(self, "_own_node", False):
            try:
                self._executor.shutdown(timeout_sec=1.0)
                self._node.destroy_node()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState) -> None:
        """
        Store the latest joint-state message received from the gripper.

        Parameters
        ----------
        msg : JointState
            Joint-state message received from ROS 2.
        """
        if msg.position:
            self._width = sum(msg.position)

    def _wait_future(self, future: Any, timeout_sec: float) -> bool:
        """Poll a future until done or timeout. Returns True if done."""
        elapsed = 0.0
        while not future.done() and elapsed < timeout_sec:
            sleep(0.05)
            elapsed += 0.05
        return future.done()

    def _send_goal_sync(self, client: ActionClient, goal: Any, timeout_sec: float = 10.0) -> Optional[Any]:
        """
        Send an action goal and block until the result is available.

        The robot's background executor processes the futures; this method
        polls them without stalling that thread.

        Returns the action result, or ``None`` on timeout or rejection.
        """
        if not client.wait_for_server(timeout_sec=2.0):
            self._node.get_logger().warning(f"Action server not available: {client._action_name}")
            return None

        send_future = client.send_goal_async(goal)
        if not self._wait_future(send_future, timeout_sec):
            self._node.get_logger().error("Timeout waiting for goal acceptance.")
            return None

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self._node.get_logger().warning("Goal was rejected by action server.")
            return None

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout_sec):
            self._node.get_logger().error("Timeout waiting for action result.")
            return None

        return result_future.result().result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """
        Get the last known gripper width from joint-state feedback.

        Returns
        -------
        float
            Current gripper finger width in metres.
        """
        return self._width

    def GetState(self) -> str:
        """
        Return the current state of the gripper as a human-readable string.

        Returns
        -------
        str
            ``"Opened"``, ``"Closed"``, or ``"Undefined"``.
        """
        if self._state == 0:
            return "Opened"
        elif self._state == 1:
            return "Closed"
        else:
            return "Undefined"

    def Grasp(self, width: float, speed: float = 0.1, force: float = 5.0, eps: float = 0.005) -> bool:
        """
        Grasp an object with the gripper.

        Parameters
        ----------
        width : float
            Target grasp width in metres.
        speed : float, optional
            Closing speed in m/s (default 0.1).
        force : float, optional
            Grasp force in N (default 5.0).
        eps : float, optional
            Inner and outer epsilon tolerance in metres (default 0.005).

        Returns
        -------
        bool
            ``True`` if the grasp was reported as successful, ``False`` otherwise.
        """
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        goal.epsilon.inner = float(eps)
        goal.epsilon.outer = float(eps)

        result = self._send_goal_sync(self._client_grasp, goal)
        success = result.success if result is not None else False
        if success:
            self._state = 1
        return success

    def Move(self, width: float, speed: float = 0.1) -> bool:
        """
        Move the gripper to a specified width.

        Parameters
        ----------
        width : float
            Target width in metres.
        speed : float, optional
            Desired speed in m/s (default 0.1).

        Returns
        -------
        bool
            ``True`` if the move was reported as successful, ``False`` otherwise.
        """
        self._width = abs(min(width, self._width_max))
        self._speed = abs(min(speed, self._speed_max))
        self._state = -1

        goal = Move.Goal()
        goal.width = self._width
        goal.speed = self._speed

        result = self._send_goal_sync(self._client_move, goal)
        return result.success if result is not None else False

    def Homing(self) -> bool:
        """
        Home the gripper and open it fully.

        Returns
        -------
        bool
            ``True`` if homing was reported as successful, ``False`` otherwise.
        """
        goal = Homing.Goal()
        result = self._send_goal_sync(self._client_homing, goal)
        success = result.success if result is not None else False
        if success:
            self._state = 0
        return success

    def Stop(self, timeout_sec: float = 1.0) -> Optional[bool]:
        """
        Stop any ongoing gripper action.

        Parameters
        ----------
        timeout_sec : float, optional
            Timeout in seconds for the service call (default 1.0).

        Returns
        -------
        bool or None
            ``True`` if the stop service call succeeded, ``None`` if it
            was unavailable or timed out.
        """
        if not self._client_stop.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().warning("Stop service not available.")
            return None

        request = Trigger.Request()
        future = self._client_stop.call_async(request)

        elapsed = 0.0
        step = 0.05
        while rclpy.ok() and not future.done() and elapsed < timeout_sec:
            sleep(step)
            elapsed += step

        try:
            response = future.result()
            return response.success
        except Exception as e:
            self._node.get_logger().error(f"Stop service call failed: {e}")
            return None


if __name__ == "__main__":
    import time

    rclpy.init()

    # Standalone usage — no robot needed
    gripper = FrankaGripper()

    print(f"Gripper state: {gripper.GetState()}")
    print(f"Gripper width: {gripper.width:.4f} m")

    print("Homing...")
    ok = gripper.Homing()
    print(f"  Homing result: {ok}  state: {gripper.GetState()}")

    time.sleep(0.5)

    print("Moving to 0.04 m...")
    ok = gripper.Move(0.04)
    print(f"  Move result: {ok}  width: {gripper.width:.4f} m")

    time.sleep(0.5)

    print("Grasping at 0.00 m (force=10 N)...")
    ok = gripper.Grasp(0.00, force=10.0)
    print(f"  Grasp result: {ok}  state: {gripper.GetState()}")

    time.sleep(0.5)

    print("Stop to release...")
    gripper.Stop()
    print(f"  Final state: {gripper.GetState()}")

    del gripper
    rclpy.shutdown()
