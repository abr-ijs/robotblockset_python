import rospy
import actionlib
import franka_gripper.msg
from sensor_msgs.msg import JointState

from robotblockset.grippers import gripper
from robotblockset.robots import robot


class PandaGripper(gripper):
    def __init__(self,
                 robot: robot,
                 namespace: str = '',
                 **kwargs):
        self.Name = 'Panda:Gripper:ROS'
        self.GripperTagNames = 'gripper'
        self._robot = robot
        self._namespace = namespace
        self._width_grasp = 0
        self._width = 0
        self._width_max = 0.077
        self._speed = 0.0
        self._speed_max = 0.5
        self._verbose = 1
        self._state = -1

        self._topic_homing = f'{self._namespace}/franka_gripper/homing'
        self._topic_grasp = f'{self._namespace}/franka_gripper/grasp'
        self._topic_move = f'{self._namespace}/franka_gripper/move'
        self._topic_stop = f'{self._namespace}/franka_gripper/stop'
        self._topic_gripper_action = f'{self._namespace}/franka_gripper/gripper_action'

        self._client_homing = actionlib.SimpleActionClient(self._topic_homing, franka_gripper.msg.HomingAction)
        self._client_grasp = actionlib.SimpleActionClient(self._topic_grasp, franka_gripper.msg.GraspAction)
        self._client_move = actionlib.SimpleActionClient(self._topic_move, franka_gripper.msg.MoveAction)
        self._client_stop = actionlib.SimpleActionClient(self._topic_stop, franka_gripper.msg.StopAction)
        # self._client_gripper_action = actionlib.SimpleActionClient(self._topic_gripper_action, GripperCommandAction)

        self._topic_state = f'{self._namespace}/franka_gripper/joint_states'
        self._state_subscriber = rospy.Subscriber(self._topic_state, JointState, self.val)

        self.Message('Created')

    def val(self, msg):
        self.position = msg.position
        # msg = receive(gripper.width_sub,1);
        # val = sum(msg.Position);

    def Open(self):
        self.Move(self._width_max)
        self._state = 0
        return 1

    def Close(self):
        self.Move(0)
        self._state = 1
        return 1

    def Grasp(self, width: float, speed: float = 0.1, eps: float = 0.005, force: int = 5):
        # Waits until the action server has started up and started
        # listening for goals.
        self._client_grasp.wait_for_server()

        # Creates a goal to send to the action server.
        grasp_goal = franka_gripper.msg.GraspGoal()
        grasp_goal.width = width
        grasp_goal.epsilon.inner = eps
        grasp_goal.epsilon.outer = eps
        grasp_goal.speed = speed
        grasp_goal.force = force

        # Sends the goal to the action server.
        self._client_grasp.send_goal(grasp_goal)

        # Waits for the server to finish performing the action.
        self._client_grasp.wait_for_result()

        # Prints out the result of executing the action
        return self._client_grasp.get_result()  # A GraspResult

    def Move(self, width, speed: float = 0.1):
        # Check and enforce width and speed limits
        self._width = abs(min(width, self._width_max))
        self._speed = abs(min(speed, self._speed_max))

        # Create goal to send to action server
        move_goal = franka_gripper.msg.MoveGoal()
        move_goal.width = self._width
        move_goal.speed = self._speed

        # Send the goal to action server
        self._client_move.send_goal(move_goal)

        # Waits for the server to finish performing the action.
        self._client_move.wait_for_result()

        # Prints out the result of executing the action
        return self._client_move.get_result()  # A MoveResult

    def Homing(self):
        homing_goal = franka_gripper.msg.HomingAction.action_goal
        self._client_homing.send_goal(homing_goal)

        # Waits for the server to finish performing the action
        self._client_homing.wait_for_result()

        # Prints out the result of executing the action
        self._client_homing.get_result()

        return self.Open()
