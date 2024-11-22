#!/usr/bin/env python

#based on: http://sdk.rethinkrobotics.com/wiki/Simple_Joint_trajectory_example

import sys

from copy import copy

import rospy
import time
import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)

from actionlib_msgs.msg import GoalStatusArray, GoalStatus, GoalID
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

class JointTrajectory(object):
    def __init__(self, controller_ns,do_motion_check=False,motion_check_callback=None):

        self._do_motion_check = do_motion_check
        self._motion_check_callback = motion_check_callback
        self.robot = None

        self.generic_state = GoalStatus()
        self.status_listener = rospy.Subscriber('/%s/position_joint_trajectory_controller/follow_joint_trajectory/status'%controller_ns, GoalStatusArray, self.handle_status_callback);time.sleep(0.2)
        self.state = self.generic_state.SUCCEEDED # 3 means ready
        self.READY = 0

        self._client = actionlib.SimpleActionClient('/%s/position_joint_trajectory_controller/follow_joint_trajectory'%controller_ns,
            FollowJointTrajectoryAction)

        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.5)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(0.1))
        #if not server_up:
        #    rospy.logerr("Timed out waiting for Joint Trajectory"
        #                 " Action Server to connect. Start the action server"
        #                 " before running example.")
        #    rospy.signal_shutdown("Timed out waiting for Action Server")
        #    sys.exit(1)

        self._joint_names = list(rospy.get_param("/%s/position_joint_trajectory_controller/joints"%controller_ns))

        self.clear()

    def handle_status_callback(self, data):
        
        self.READY = 1
        
        # Get all statuses. If any of them is not SUCCEEDED, forbid sending goals until all of them are succeeded. COULD BE BUGGY.
        statuses = data.status_list
        for st in statuses:
            state = int(st.status)
            if state != self.generic_state.SUCCEEDED: 
                self.READY = 0
            
        #status = data.status_list[0].goal_id
        
    def add_point(self, positions, velocities, accelerations=None, delta_time=0.01):
        self._point_msg.positions = positions
        if velocities is not None:
            self._point_msg.velocities = velocities
        if accelerations is not None:
            0#self._point_msg.accelerations = accelerations
        self._time_sum += delta_time  
        self._point_msg.time_from_start = rospy.Duration(self._time_sum)
        self._goal.trajectory.points.append(copy(self._point_msg))

    def start(self, max_wait_until_controller_ready = 10):
        """ ARGS: 
            max_wait_until_controller_ready : if controller is not ready(still executing some other goal, 
                                               this is the timeout duration"""
        start_t = time.time()
        while self.READY != 1:
            
            t = time.time()
            if (t-start_t) > max_wait_until_controller_ready:
                raise Exception('Joint Traj controller - cant run trajectory, controller was not ready within the timeout period {} s.'.format(max_wait_until_controller_ready))
        
        #HACK: workaround for dropping first X points out of X as they occur before the current time
        self._goal.trajectory.header.stamp = rospy.Duration(0)  #rospy.Time.now()+rospy.Duration(0.7)

        if not self._do_motion_check:
            self._client.send_goal(self._goal)
        else:
            print("using callback")
            self._client.send_goal(self._goal,feedback_cb=self.callback_wrapper)


    def callback_wrapper(self, gh):
        if self._motion_check_callback(self.robot):
            self.stop()
        else:
            print("ok")
    

    def stop(self):
        print("Stop")
        self._client.cancel_goal()

    def wait(self, timeout=10.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout+0.4))
        
    #def wait(self, timeout=10.0, refresh_dt = 0.05):
    #    """An attempt to make a non-blocking wait but i guess it's blocking anyway. """
    #    
    #    #self._client.wait_for_result(timeout=rospy.Duration(timeout+0.4))
    #    ct = time.time()
    #    last_upd = time.time()
    #    while self._client.get_state() not in [3,9]:
    #        #print(self._client.get_state())
    #        time.sleep(refresh_dt)
        
    def result(self):
        return self._client.get_result()

    def clear(self):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = self._joint_names
        self._point_msg = JointTrajectoryPoint()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._time_sum = 0


if __name__ == "__main__":
    """
    Creates a client of the Joint Trajectory Action Server
    to send commands of standard action type,
    control_msgs/FollowJointTrajectoryAction.
    """

    rospy.init_node('example', anonymous=True)

    from limb_interface import LimbInterface
    left_arm = LimbInterface("/left_arm_controller")
    p0 = left_arm.query_actual_positions()
    
    
    traj = JointTrajectory("/left_arm_controller")

    rospy.on_shutdown(traj.stop)

    n_sec = 1.0
    traj.add_point(p0, n_sec)

    p1 = list(p0) 
    p1[1] = -0.1
    n_sec += 5.0
    traj.add_point(p1, n_sec)

    traj.start()
    traj.wait(n_sec)
