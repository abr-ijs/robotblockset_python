"""ROS joint trajectory action helpers.

This module provides helpers for ROS FollowJointTrajectory action clients used
by RobotBlockSet robot interfaces. It manages trajectory points, goal sending,
status monitoring, and result handling for joint-space motion commands.

based on: http://sdk.rethinkrobotics.com/wiki/Simple_Joint_trajectory_example

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah, Mihael Simonic.
"""

from typing import Any, Callable, Optional

from copy import copy

import rospy
import time
import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState

from actionlib_msgs.msg import GoalStatusArray, GoalStatus, GoalID
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

from robotblockset.rbs_typing import JointAccelerationType, JointConfigurationType, JointVelocityType


class JointTrajectory(object):
    """
    Helper for ROS ``FollowJointTrajectory`` action clients.

    Attributes
    ----------
    robot : Any
        Robot object passed to the motion-check callback.
    generic_state : GoalStatus
        Reusable goal-status message used for status comparisons.
    state : int
        Cached action state.
    READY : int
        Readiness flag indicating whether a new trajectory can be sent.
    """

    def __init__(self, controller_ns: str, do_motion_check: bool = False, motion_check_callback: Optional[Callable[[Any], bool]] = None) -> None:
        """
        Initialize a ROS joint-trajectory action client.

        Parameters
        ----------
        controller_ns : str
            Controller namespace used to resolve action and status topics.
        do_motion_check : bool, optional
            If ``True``, use the feedback callback to monitor motion execution.
        motion_check_callback : callable, optional
            Callback used to decide whether motion should be interrupted.

        Notes
        -----
        The constructor subscribes to the controller status topic, creates the
        action client, queries the joint names from ROS parameters, and resets
        the internal goal state.
        """

        self._do_motion_check = do_motion_check
        self._motion_check_callback = motion_check_callback
        self.robot = None

        self.generic_state = GoalStatus()
        self.status_listener = rospy.Subscriber("/%s/position_joint_trajectory_controller/follow_joint_trajectory/status" % controller_ns, GoalStatusArray, self.handle_status_callback)
        time.sleep(0.2)
        self.state = self.generic_state.SUCCEEDED  # 3 means ready
        self.READY = 0

        self._client = actionlib.SimpleActionClient("/%s/position_joint_trajectory_controller/follow_joint_trajectory" % controller_ns, FollowJointTrajectoryAction)

        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.5)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(0.1))
        # if not server_up:
        #    rospy.logerr("Timed out waiting for Joint Trajectory"
        #                 " Action Server to connect. Start the action server"
        #                 " before running example.")
        #    rospy.signal_shutdown("Timed out waiting for Action Server")
        #    sys.exit(1)

        self._joint_names = list(rospy.get_param("/%s/position_joint_trajectory_controller/joints" % controller_ns))

        self.clear()

    def handle_status_callback(self, data: GoalStatusArray) -> None:
        """
        Update controller readiness from the latest action-status message.

        Parameters
        ----------
        data : GoalStatusArray
            Action-status message received from the controller.

        Notes
        -----
        The helper marks the controller as ready only when all reported goal
        statuses are in the ``SUCCEEDED`` state.
        """

        self.READY = 1

        # Get all statuses. If any of them is not SUCCEEDED, forbid sending goals until all of them are succeeded. COULD BE BUGGY.
        statuses = data.status_list
        for st in statuses:
            state = int(st.status)
            if state != self.generic_state.SUCCEEDED:
                self.READY = 0

        # status = data.status_list[0].goal_id

    def add_point(self, positions: JointConfigurationType, velocities: JointVelocityType, accelerations: Optional[JointAccelerationType] = None, delta_time: float = 0.01) -> None:
        """
        Append a trajectory point to the active goal.

        Parameters
        ----------
        positions : JointConfigurationType
            Joint positions of the trajectory point.
        velocities : JointVelocityType
            Joint velocities of the trajectory point.
        accelerations : JointAccelerationType, optional
            Joint accelerations of the trajectory point.
        delta_time : float, optional
            Time offset from the previous point in seconds.
        """
        self._point_msg.positions = positions
        if velocities is not None:
            self._point_msg.velocities = velocities
        if accelerations is not None:
            0  # self._point_msg.accelerations = accelerations
        self._time_sum += delta_time
        self._point_msg.time_from_start = rospy.Duration(self._time_sum)
        self._goal.trajectory.points.append(copy(self._point_msg))

    def start(self, max_wait_until_controller_ready: float = 10) -> None:
        """
        Send the currently prepared trajectory goal.

        Parameters
        ----------
        max_wait_until_controller_ready : float, optional
            Maximum time to wait for the controller to become ready before the
            goal is sent.

        Notes
        -----
        The trajectory header timestamp is forced to zero as a workaround for
        controllers that may drop the first points when they are stamped in the
        near future.
        """
        start_t = time.time()
        while self.READY != 1:

            t = time.time()
            if (t - start_t) > max_wait_until_controller_ready:
                raise Exception("Joint Traj controller - cant run trajectory, controller was not ready within the timeout period {} s.".format(max_wait_until_controller_ready))

        # HACK: workaround for dropping first X points out of X as they occur before the current time
        self._goal.trajectory.header.stamp = rospy.Duration(0)  # rospy.Time.now()+rospy.Duration(0.7)

        if not self._do_motion_check:
            self._client.send_goal(self._goal)
        else:
            print("using callback")
            self._client.send_goal(self._goal, feedback_cb=self.callback_wrapper)

    def callback_wrapper(self, gh: Any) -> None:
        """
        Wrap the user motion-check callback for action feedback handling.

        Parameters
        ----------
        gh : Any
            Action feedback payload forwarded by the ROS action client.

        Notes
        -----
        The feedback object is not used directly. The wrapper forwards the
        stored robot instance to ``_motion_check_callback``.
        """
        if self._motion_check_callback(self.robot):
            self.stop()
        else:
            print("ok")

    def stop(self) -> None:
        """
        Cancel the currently active trajectory goal.
        """
        print("Stop")
        self._client.cancel_goal()

    def wait(self, timeout: float = 10.0) -> None:
        """
        Wait until the active trajectory goal finishes or times out.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.
        """
        self._client.wait_for_result(timeout=rospy.Duration(timeout + 0.4))

    # def wait(self, timeout=10.0, refresh_dt = 0.05):
    #    """An attempt to make a non-blocking wait but i guess it's blocking anyway. """
    #
    #    #self._client.wait_for_result(timeout=rospy.Duration(timeout+0.4))
    #    ct = time.time()
    #    last_upd = time.time()
    #    while self._client.get_state() not in [3,9]:
    #        #print(self._client.get_state())
    #        time.sleep(refresh_dt)

    def result(self) -> Any:
        """
        Return the action result for the current trajectory goal.

        Returns
        -------
        Any
            Result object returned by the ROS action client.
        """
        return self._client.get_result()

    def clear(self) -> None:
        """
        Reset the current trajectory goal and point accumulator.

        Notes
        -----
        The method creates a fresh goal, restores the joint-name list, resets
        the time tolerance, and clears the accumulated trajectory duration.
        """
        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = self._joint_names
        self._point_msg = JointTrajectoryPoint()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._time_sum = 0
