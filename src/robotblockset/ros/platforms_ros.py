"""ROS platform interface implementations.

This module defines ROS-backed platform wrappers used by RobotBlockSet mobile
platform interfaces. It provides localization, navigation, obstacle sensing,
map visualization, and base-motion helpers through ROS topics and services.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Any, Optional, Tuple

import numpy as np
from scipy.linalg import block_diag
from time import perf_counter, sleep
from threading import Thread

from roscpp.srv import SetLoggerLevel, SetLoggerLevelRequest
import rospy
from sensor_msgs.msg import JointState, LaserScan, Range
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Empty, Float32MultiArray, Bool

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from std_srvs.srv import Trigger
from pal_navigation_msgs.srv import Acknowledgment

from robotblockset.platforms import platform
from robotblockset.robots import CommandModeCodes
from robotblockset.platform_spec import tiagobase_spec
from robotblockset.optitrack.optitrack_localization import optitrack_localization

from robotblockset.transformations import map_pose, t2x, x2t, q2r, r2q, rot_z, world2frame, world2frame2d, frame2world2d, v2s
from robotblockset.tools import rbs_type, vector, check_option, smoothstep, isvector, isscalar
from robotblockset.graphics import plot_circle

import matplotlib.pyplot as plt


def plot_map(platform_pose: Optional[np.ndarray] = None, scan_msg: Optional[LaserScan] = None, map_msg: Optional[Any] = None, map_frame: Optional[np.ndarray] = None, in_map_frame: bool = False) -> None:
    """
    Plot the map, robot pose, and laser scan points.

    Parameters
    ----------
    platform_pose : np.ndarray, optional
        The 2D pose of the platform in the format (x, y, theta). Defaults to (0, 0, 0).
    scan_msg : LaserScan, optional
        The received laser scan message.
    map_msg : OccupancyGrid, optional
        The received occupancy grid message.
    map_frame : np.ndarray, optional
        The origin of the map in the world frame as (x, y, theta). Defaults to (0, 0, 0).
    in_map_frame : bool, optional
        If True, display the scene in the map frame instead of the world frame. Defaults to False.

    Returns
    -------
    None
        This function renders the map and sensor data using Matplotlib.

    Notes
    -----
    The plot is intended for debugging and visualization and does not modify
    the platform state.
    """

    if platform_pose is None:
        platform_pose = np.zeros(3)

    if map_frame is None:
        map_frame = np.zeros(3)

    plt.figure(figsize=(8, 8))

    if map_msg is not None:
        # Extract occupancy grid metadata
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        grid = np.array(map_msg.data).reshape((height, width))
        grid = np.where(grid == -1, 50, grid)  # Convert unknown (-1) to gray

        if in_map_frame:
            extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]  # X limits  # Y limits
            plt.imshow(grid, origin="lower", cmap="gray", extent=extent)
        else:
            map_x, map_y = np.meshgrid(np.linspace(origin_x, origin_x + width * resolution, width), np.linspace(origin_y, origin_y + height * resolution, height))
            map_p = np.column_stack((map_x.flatten(), map_y.flatten()))
            grid_values = grid.flatten()
            map_p = frame2world2d(map_p, map_frame)
            plt.scatter(map_p[:, 0], map_p[:, 1], c=grid_values, cmap="gray", s=2, label="Occupancy grid")

    if scan_msg is not None:
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        valid = np.isfinite(ranges)
        ranges, angles = ranges[valid], angles[valid]
        scan_p = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        scan_p = frame2world2d(scan_p, platform_pose)
        if in_map_frame:
            scan_p = world2frame2d(scan_p, map_frame)

        plt.scatter(scan_p[:, 0], scan_p[:, 1], s=5, c="red", label="Laser scan")

    if in_map_frame:
        _p = world2frame2d(platform_pose, map_frame)
    else:
        _p = platform_pose
    plt.plot(_p[0], _p[1], "bo", markersize=10, label="Robot")
    arrow_length = 0.5
    arrow_dx = arrow_length * np.cos(_p[2])
    arrow_dy = arrow_length * np.sin(_p[2])
    plt.arrow(_p[0], _p[1], arrow_dx, arrow_dy, head_width=0.2, head_length=0.2, fc="green", ec="green", label="Robot direction")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Scene")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.axis("equal")  # Keep the aspect ratio square
    plt.show()


class platform_ros(platform):
    """
    Base class for ROS-backed mobile platform interfaces.

    Attributes
    ----------
    _namespace : str
        ROS namespace prefix used by the platform wrapper.
    """

    def __init__(self, name: str = "", ns: str = "", init_node: bool = True, multi_node: bool = False) -> None:
        """
        Initialize the generic ROS platform interface.

        Parameters
        ----------
        name : str, optional
            Name of the platform.
        ns : str, optional
            ROS namespace of the platform.
        init_node : bool, optional
            If ``True``, initialize a ROS node for the wrapper.
        multi_node : bool, optional
            If ``True``, initialize the ROS node in anonymous mode.
        """
        platform.__init__(self)

        if ns == "":
            self._namespace = ""
        else:
            self._namespace = "/" + ns

        self._active = False

        if init_node:
            self._init_ros_node(multi_node)
        else:
            print("Make sure that ROS node is initialized outside")

    def is_ros_ready(self) -> bool:
        """
        Return whether the ROS node for the platform is active.

        Returns
        -------
        bool
            ``True`` if the ROS node is active.
        """
        return self._active

    def _init_ros_node(self, anonymous: bool = False) -> None:
        """
        Initialize the ROS node used by the platform wrapper.

        Parameters
        ----------
        anonymous : bool, optional
            If ``True``, initialize the node in anonymous mode.
        """
        try:
            rospy.init_node("Platform_{}".format(self.Name), anonymous=anonymous)
            self._active = True
        except rospy.ROSException as e:
            self.Message("Skipping node init because of exception: {}".format(e), 0)

    # ROS subsciber callbacks
    def _joint_states_callback(self, data: JointState) -> None:
        """
        Store the latest wheel or joint-state message.

        Parameters
        ----------
        data : JointState
            Joint-state message received from ROS.
        """
        self._joint_states_msg = data
        # self._joint_states_msg = copy.deepcopy(data)
        self._last_joint_states_callback_time = self.simtime()

    def _robot_pose_callback(self, data: PoseWithCovarianceStamped) -> None:
        """
        Store the latest robot pose estimate.

        Parameters
        ----------
        data : PoseWithCovarianceStamped
            Pose estimate received from ROS localization.
        """
        self._robot_pose_msg = data
        # self._robot_pose_msg = copy.deepcopy(data)
        self._last_robot_pose_callback_time = self.simtime()

    def _laser_scan_callback(self, data: LaserScan) -> None:
        """
        Store the latest laser-scan message.

        Parameters
        ----------
        data : LaserScan
            Laser-scan message received from ROS.
        """
        self._laser_scan_msg = data
        # self._laser_scan_msg = copy.deepcopy(data)
        self._last_laser_scan_callback_time = self.simtime()

    def _sonar_callback(self, data: Range) -> None:
        """
        Store the latest sonar-range message.

        Parameters
        ----------
        data : Range
            Sonar range message received from ROS.
        """
        self._sonar_msg = data
        # self._sonar_msg = copy.deepcopy(data)
        self._last_sonar_callback_time = self.simtime()

    # Navigation location
    def SetLocation(self, x_init: Any, theta_init: Optional[float] = None, cov_init: Optional[np.ndarray] = None) -> None:
        """
        Sets the platform's initial location in the map frame.

        This function updates the platform's pose and publishes it to the localization topic.

        Parameters
        ----------
        x_init : array-like
            The initial position (x, y) or (x, y, theta) of the platform.
        theta_init : float, optional
            The initial orientation (theta) of the platform. If None, it is extracted from x_init.
        cov_init : np.ndarray, optional
            The covariance matrix of the initial pose. Defaults to a diagonal matrix with small uncertainty.

        Returns
        -------
        None
            This method publishes a new initial pose for localization.
        """
        x_init = vector(x_init)
        if isvector(x_init, dim=2):
            pass
        elif isvector(x_init, dim=3):
            theta_init = x_init[2] if theta_init is None else theta_init
            x_init = x_init[:2]

        if cov_init is None:
            cov_init = np.diag([1, 1, 0, 0, 0, 1]) * 0.5

        self._loc_msg.header.frame_id = "map"
        self._loc_msg.pose.pose.position.x = x_init[0]
        self._loc_msg.pose.pose.position.y = x_init[1]
        _Q = rot_z(theta_init)
        self._loc_msg.pose.pose.orientation.w = _Q[0]
        self._loc_msg.pose.pose.orientation.x = _Q[1]
        self._loc_msg.pose.pose.orientation.y = _Q[2]
        self._loc_msg.pose.pose.orientation.z = _Q[3]
        self._loc_msg.pose.covariance = cov_init.flatten()
        self._loc_pub.publish(self._loc_msg)

    def SetNavigationMode(self, mode: str = "MAP") -> Tuple[bool, str]:
        """
        Sets the navigation mode of the platform.

        This function switches the navigation mode between map-based (MAP) and localization-based (LOC).

        Parameters
        ----------
        mode : str, optional
            The navigation mode, either "MAP" or "LOC". Defaults to "MAP".

        Returns
        -------
        tuple (bool, str)
            A boolean indicating success and an error message if applicable.
        """
        if mode in ["MAP", "LOC"]:
            rospy.wait_for_service("/pal_navigation_sm")
            _resp = self._nav_service(mode)
            return _resp.output, _resp.error
        else:
            self.Message(f"Invalid mode {mode}! Use 'MAP' or 'LOC'.", 0)
            return False, "Invalid mode"

    # AMCL Movements

    def GetSLAMLocation(self, out: Optional[str] = None, task_space: Optional[str] = None, state: Optional[Any] = None) -> np.ndarray:
        """
        Retrieves the current SLAM-based location of the platform.

        This function fetches the latest SLAM pose information, checks its validity,
        and transforms it into the desired output format.

        Parameters
        ----------
        out : str, optional
            The format of the output location. Defaults to the platform's task pose form.
        task_space : str, optional
            Defines the task space in which the pose should be returned. (To be implemented).
        state : any, optional
            Additional state information (currently not used).

        Returns
        -------
        np.ndarray
            The transformed SLAM pose in the requested format.
            If SLAM data is unavailable, returns an array of NaN values.
        """
        if out is None:
            out = self._default.TaskPoseForm

        if self._robot_pose_msg is not None:
            if (self.simtime() - self._last_robot_pose_callback_time) > 2 * self.tsamp:
                self.WarningMessage("SLAM pose is not updated")
            _msg = self._robot_pose_msg
            _pose = _msg.pose.pose
            _x = rbs_type([_pose.position.x, _pose.position.y, _pose.position.z, _pose.orientation.w, _pose.orientation.x, _pose.orientation.y, _pose.orientation.z])
        else:
            _x = np.full((7,), np.nan)
        # ToDo:  task_space, state
        return map_pose(x=_x, out=out)

    def CMoveToLocAMCL(self, rp: Any, rtheta: Optional[float] = None, task_space: str = "World") -> int:
        """
        Moves the platform to a specified location using AMCL localization.

        Parameters
        ----------
        rp : array-like
            The target position (x, y) or (x, y, theta) in the specified task space.
        rtheta : float, optional
            The target orientation. If None, it is extracted from rp.
        task_space : str, optional
            The task space for the movement. Options: "World", "Object", "Platform". Defaults to "World".

        Returns
        -------
        int
            Error status (0 if successful, nonzero if failed).
        """
        rp = vector(rp)
        if isvector(rp, dim=2):
            pass
        elif isvector(rp, dim=3):
            rtheta = rp[2] if rtheta is None else rtheta
            rp = rp[:2]

        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            rp = self.ObjectToWorld(rp)
            if rtheta is not None:
                rtheta = self.ObjectToWorld(rtheta)
        elif check_option(task_space, "Platform"):
            rp = self.PlatformToWorld(rp)
            if rtheta is not None:
                rtheta = self.PlatformToWorld(rtheta)
        else:
            raise ValueError(f"Task space {task_space} not supported in CMoveToLocAMCL")

        self.Message("CMoveToLocAMCL started", 2)
        self.Start()
        self._command.mode = CommandModeCodes.PLANAR.value
        tmperr = 0

        self._move_pose_goal_msg.Header.FrameId = "map"
        self._move_pose_goal_msg.pose.pose.position.x = rp[0]
        self._move_pose_goal_msg.pose.pose.position.y = rp[1]
        _Q = self.rot_z(rtheta)
        self._move_pose_goal_msg.pose.pose.orientation.w = _Q[0]
        self._move_pose_goal_msg.pose.pose.orientation.x = _Q[1]
        self._move_pose_goal_msg.pose.pose.orientation.y = _Q[2]
        self._move_pose_goal_msg.pose.pose.orientation.z = _Q[3]

        _, state, _ = self._move_pose_action.sendGoalAndWait(self._move_pose_goal_msg)
        _ = self.FinalGoalStates.index(state) if state in self.FinalGoalStates else -1
        self.Stop()
        self.Message("CMoveToLocAMCL finished", 2)
        return tmperr

    def OMoveToLocAMCL(self, rp: Any, rtheta: float) -> int:
        """
        Moves the platform to an object-relative location using AMCL localization.

        Returns
        -------
        int
            Motion status code returned by `CMoveToLocAMCL`.
        """
        self.Message("OMoveToLocAMCL -> CMoveToLocAMCL", 2)
        return self.CMoveToLocAMCL(rp, rtheta, task_space="Object")

    def PMoveToLocAMCL(self, rp: Any, rtheta: float) -> int:
        """
        Moves the platform to a platform-relative location using AMCL localization.

        Returns
        -------
        int
            Motion status code returned by `CMoveToLocAMCL`.
        """
        self.Message("PMoveToLocAMCL -> CMoveToLocAMCL", 2)
        return self.CMoveToLocAMCL(rp, rtheta, task_space="Platform")

    # ROS sensors
    def GetClosestFrontObstacle(self, ang_range: Optional[Any] = None, for_platform: Optional[bool] = None) -> Tuple[float, float]:
        """
        Finds the closest obstacle in the front laser scan within a specified angular range.

        Parameters
        ----------
        ang_range : float or array-like, optional
            If None, uses the default laser angle range. If scalar, it defines a symmetric range around zero.
            If a 2-element array, it specifies the range directly. Defaults to None.
        for_platform : bool, optional
            If True, calculates distance of platform in forward direction . If False, uses raw laser sensor data. Defaults to True.

        Returns
        -------
        tuple (float, float)
            The minimum detected distance and its corresponding angle.
        """
        if self._laser_scan_msg is None:
            _dist = np.inf
            _ang = 0.0
        else:
            if ang_range is None:
                ang_range = np.array([-1, 1]) * self._default.LaserAngleRange
            elif isscalar(ang_range):
                ang_range = np.array([-1, 1]) * abs(ang_range)
            elif isvector(ang_range, dim=2):
                pass
            else:
                raise ValueError("Wrong input parameter")

            if for_platform is None:
                for_platform = self._default.ObstaclesForPlatform

            _ang = np.arange(self._laser_scan_msg.angle_min, self._laser_scan_msg.angle_max, self._laser_scan_msg.angle_increment)
            _i1 = np.searchsorted(_ang, ang_range[0], side="left")
            _i2 = np.searchsorted(_ang, ang_range[1], side="right") - 1
            _ang = _ang[_i1 : _i2 + 1]
            _ranges = rbs_type(self._laser_scan_msg.ranges[_i1 : _i2 + 1])

            # Find minimum distance and corresponding angle
            valid = np.isfinite(_ranges)
            _ranges, _ang = _ranges[valid], _ang[valid]
            if for_platform:
                _scan_p = np.column_stack((self.laser_offset + _ranges * np.cos(_ang), _ranges * np.sin(_ang)))
                in_range = (_scan_p[:, 1] <= self.r_platform) & (_scan_p[:, 1] >= -self.r_platform)
                _ranges, _ang, _scan_p = _ranges[in_range], _ang[in_range], _scan_p[in_range]

                _ang = np.arcsin(_scan_p[:, 1] / self.r_platform)
                _ranges = _scan_p[:, 0] - np.cos(_ang) * self.r_platform
            _dist = np.min(_ranges)
            _ii = np.argmin(_ranges)
            _ang = _ang[_ii]
        return _dist, _ang

    def GetClosestRearObstacle(self) -> float:
        """
        Finds the closest obstacle detected by the rear sonar sensor.

        Returns
        -------
        float
            The measured distance to the closest rear obstacle. Returns infinity if no reading is available.
        """
        if self._sonar_msg is None:
            return np.inf
        else:
            return self._sonar_msg.range

    def GetClosestObstacle(self, ang_range: Any = [-np.pi, np.pi]) -> float:
        """
        Finds the closest obstacle detected by either the front laser scanner or the rear sonar sensor.

        Parameters
        ----------
        ang_range : array-like, optional
            The angular range for front obstacle detection. Defaults to [-pi, pi].

        Returns
        -------
        float
            The minimum detected obstacle distance from either the front or rear sensor.
        """
        return min(self.GetClosestFrontObstacle(ang_range=ang_range), self.GetClosestRearObstacle())

    def PlotObstacles(self, ang_range: Optional[Any] = None) -> None:
        """Plot laser-scan obstacles and closest platform distances."""
        if self._laser_scan_msg is not None:
            _ang = np.arange(self._laser_scan_msg.angle_min, self._laser_scan_msg.angle_max + self._laser_scan_msg.angle_increment / 10, self._laser_scan_msg.angle_increment)
            _ranges = rbs_type(self._laser_scan_msg.ranges)
            valid = np.isfinite(_ranges)
            _ranges, _ang = _ranges[valid], _ang[valid]
            _scan_p = np.column_stack((self.laser_offset + _ranges * np.cos(_ang), _ranges * np.sin(_ang)))
            if ang_range is not None:
                if isscalar(ang_range):
                    ang_range = np.array([-1, 1]) * abs(ang_range)
                _i1 = np.searchsorted(_ang, ang_range[0], side="left")
                _i2 = np.searchsorted(_ang, ang_range[1], side="right")
                _ang = _ang[_i1:_i2]
                _ranges = rbs_type(self._laser_scan_msg.ranges[_i1:_i2])
                _scan_p = _scan_p[_i1:_i2, :]
                plt.scatter(_scan_p[:, 0], _scan_p[:, 1], s=5, c="red", label="Laser scan")

            # Obstacles in front of platform
            in_range = (_scan_p[:, 1] <= self.r_platform) & (_scan_p[:, 1] >= -self.r_platform)
            _ranges, _ang, _scan_p = _ranges[in_range], _ang[in_range], _scan_p[in_range]
            _ii = np.argmin(_ranges)

            # Platform distance to obstacle in forward direction
            _ang_p = np.arcsin(_scan_p[:, 1] / self.r_platform)
            _ranges_p = _scan_p[:, 0] - np.cos(_ang_p) * self.r_platform
            _ii_p = np.argmin(_ranges_p)

            if ang_range is None:
                plt.scatter(_scan_p[:, 0], _scan_p[:, 1], s=5, c="red", label="Laser scan")
            plt.plot(_scan_p[_ii, 0], _scan_p[_ii, 1], "kx", markersize=10, label="Closest to sensor")
            plt.plot(_scan_p[_ii_p, 0], _scan_p[_ii_p, 1], "bx", markersize=10, label="Closest to platform")
            plt.plot(0, 0, "go", markersize=10, label="Platform")
            plt.plot(self.laser_offset, 0, "yo", markersize=5, label="Sensor")
            plot_circle(r=self.r_platform, color="green", linestyle="--", linewidth=1)
            plt.xlim(-self.r_platform - 0.1, np.max(_scan_p[:, 0]) + 0.1)
            plt.ylim(-self.r_platform - 0.1, self.r_platform + 0.1)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.show(block=False)

    # Status
    def Check(self, silent: bool = False) -> list[str]:
        """
        Check the current platform status.

        Parameters
        ----------
        silent : bool, optional
            If ``True``, suppress status messages while checking the platform.

        Returns
        -------
        list[str]
            List of active platform errors.
        """
        return []

    def HasError(self) -> bool:
        """Return whether the platform currently reports any errors."""
        return len(self.Check(silent=True)) > 0


class tiagobase(tiagobase_spec, platform_ros):
    """
    ROS interface for the Tiago base platform.

    Attributes
    ----------
    localization : str
        Active localization mode, for example ``"SLAM"`` or ``"OPTI"``.
    opti : optitrack_localization | None
        OptiTrack localization helper.
    TOPTIFrame : np.ndarray
        Transform defining the OptiTrack world frame.
    """

    def __init__(self, name: str = "TiagoBase", ns: str = "", localization: Optional[str] = None, natnet_platform_rigidbody_id: int = 1, natnet_host_ip: str = "127.0.0.1", natnet_client_ip: str = "127.0.0.1", natnet_multicast_ip: str = "239.255.42.99") -> None:
        """
        Initialize the Tiago base ROS interface.

        Parameters
        ----------
        name : str, optional
            Name of the platform.
        ns : str, optional
            ROS namespace of the platform.
        localization : str, optional
            Initial localization mode.
        natnet_platform_rigidbody_id : int, optional
            NatNet rigid-body identifier used by OptiTrack.
        natnet_host_ip : str, optional
            NatNet host IP address.
        natnet_client_ip : str, optional
            NatNet client IP address.
        natnet_multicast_ip : str, optional
            NatNet multicast IP address.
        """
        platform_ros.__init__(self, name=name, ns=ns)
        tiagobase_spec.__init__(self)

        self.Name = name

        # Initialize localization
        self.localization = "SLAM"
        self.xSLAM = np.zeros(3)  # transformation between SLAM frame (x y theta) and world frame
        self.opti = None
        self._natnet_host_ip = natnet_host_ip
        self._natnet_client_ip = natnet_client_ip
        self._natnet_multicast_ip = natnet_multicast_ip
        self._natnet_platform_rigidbody_id = natnet_platform_rigidbody_id
        if localization is not None:
            self.SetLocalization(localization=localization)
        self._actual_SLAM_x = None
        self._actual_OPTI_x = None
        self.TOPTIFrame = np.eye(4)
        self._T_SLAM_to_OPTI = None

        # Initialize ROS subscribers, publishers and action clients
        self._last_joint_states_callback_time = self.simtime()
        self._joint_states_msg = None
        self._joint_states_subscriber = rospy.Subscriber(f"{self._namespace}/joint_states", JointState, self._joint_states_callback)

        self._last_robot_pose_callback_time = self.simtime()
        self._robot_pose_msg = None
        self._robot_pose_subscriber = rospy.Subscriber(f"{self._namespace}/robot_pose", PoseWithCovarianceStamped, self._robot_pose_callback)

        self._last_laser_scan_callback_time = self.simtime()
        self._laser_scan_msg = None
        self._laser_scan_subscriber = rospy.Subscriber(f"{self._namespace}/scan", LaserScan, self._laser_scan_callback)

        self._last_sonar_callback_time = self.simtime()
        self._sonar_msg = None
        self._sonar_subscriber = rospy.Subscriber(f"{self._namespace}/sonar_base", Range, self._sonar_callback)

        self._cmd_pub = rospy.Publisher(f"{self._namespace}/mobile_base_controller/cmd_vel", Twist, queue_size=10)
        self._cmd_msg = Twist()

        self._nav_service = rospy.ServiceProxy(f"{self._namespace}/pal_navigation_sm", Acknowledgment)
        self._loc_pub = rospy.Publisher(f"{self._namespace}/initialpose", PoseWithCovarianceStamped, queue_size=10)
        self._loc_msg = PoseWithCovarianceStamped()

        self._move_pose_action = actionlib.SimpleActionClient(f"{self._namespace}/move_base", MoveBaseAction)
        self._move_pose_goal_msg = MoveBaseGoal()

        self._last_closest_distance = np.inf

        while (self._joint_states_msg is None) or (self._robot_pose_msg is None):
            self.Message("Waiting for ROS callbacks ...", 1)
            sleep(1)

        if check_option(self.localization, "OPTI"):
            while self.opti.frame is None:
                self.Message("Waiting for OPTI callbacks ...", 1)
                self.opti.GetFrame()
                sleep(1)
        self.Message("ROS ready", 1)

        self.Init()
        self._connected = True

    def SetOPTIFrame(self, x: Any) -> None:
        """
        Set the OptiTrack world frame used by the platform.

        Parameters
        ----------
        x : Any
            Pose specification accepted by :meth:`spatial`.
        """
        x = self.spatial(x)
        if x.shape == (4, 4):
            _T = x
        elif x.shape == (3, 3):
            _T = map_pose(R=x, out="T")
        elif isvector(x, dim=7):
            _T = map_pose(x=x, out="T")
        elif isvector(x, dim=3):
            _T = map_pose(p=x, out="T")
        elif isvector(x, dim=4):
            _T = map_pose(Q=x, out="T")
        else:
            raise ValueError(f"Opti frame shape {x.shape} not supported")
        self.TOPTIFrame = _T

    def GetOPTIFrame(self, out: Optional[str] = None) -> Any:
        """
        Return the configured OptiTrack world frame.

        Parameters
        ----------
        out : str, optional
            Requested output pose format.

        Returns
        -------
        Any
            OptiTrack world frame in the requested format.
        """
        if out is None:
            out = self._default.TaskPoseForm
        return map_pose(T=self.TOPTIFrame, out=out)

    # -----------------------------------------
    def GetState(self) -> None:
        """
        Update the platform state from ROS localization and sensor topics.

        Notes
        -----
        The method updates wheel state, localization, obstacle sensors, and the
        base pose of an attached robot, if present.
        """

        t = self.simtime()
        if self._joint_states_msg is not None:
            if (t - self._last_joint_states_callback_time) > 10 * self.tsamp:
                self.WarningMessage("Joint_state is not updated")
            else:
                self._actual.q = rbs_type(self._joint_states_msg.position[:2])
                self._actual.qdot = rbs_type(self._joint_states_msg.velocity[:2])
                self._actual.trq = rbs_type(self._joint_states_msg.effort[:2])

        if self._robot_pose_msg is not None:
            if (t - self._last_robot_pose_callback_time) > 10 * self.tsamp:
                self.WarningMessage("SLAM pose is not updated")
            else:
                _msg = self._robot_pose_msg
                _pose = _msg.pose.pose
                self._actual_SLAM_x = rbs_type([_pose.position.x, _pose.position.y, _pose.position.z, _pose.orientation.w, _pose.orientation.x, _pose.orientation.y, _pose.orientation.z])
                if check_option(self.localization, "SLAM"):
                    self._actual.x = self._actual_SLAM_x

        if check_option(self.localization, "OPTI"):
            self.opti.GetFrame()
            if not any(np.isnan(self.opti.x)):
                if (t - self.opti.GetFrameTime()) > 2 * self.tsamp:
                    self.WarningMessage("OPTI pose is not updated")
                    if self._T_SLAM_to_OPTI is not None:
                        self._actual.x = t2x(self._T_SLAM_to_OPTI @ x2t(self._actual_SLAM_x))
                else:
                    self._actual_OPTI_x = world2frame(rbs_type(self.opti.x), self.TOPTIFrame)
                    self._actual.x = self._actual_OPTI_x
                    if self._actual_SLAM_x is not None:
                        self._T_SLAM_to_OPTI = x2t(self._actual_OPTI_x) @ np.linalg.inv(x2t(self._actual_SLAM_x))  # Transformation between SLAM and OPTI frame

        _J = self.Jacobi()
        _R = q2r(self._actual.x[3:])
        self._actual.v = block_diag(_R, _R) @ _J @ self._actual.qdot

        if (t - self._last_laser_scan_callback_time) > 50 * self.tsamp:
            self.Message("No ROS laser scan msg received!", 2)

        if t - self._last_sonar_callback_time > 50 * self.tsamp:
            self.Message("No ROS sonar msg received!", 2)

        if self.Robot is not None:
            _tmp = x2t(self._actual.x) @ self.TRobotBase
            self.Robot.SetBasePose(_tmp)
            RR = np.eye(6)
            RR[:3, 3:] = v2s(q2r(self._actual.x[3:]) @ self.TRobotBase[:3, 3]).T
            self.Robot.SetBaseVel(RR @ self.actual.v)
            if self.Robot.EEFixed:
                self.TObject = map_pose(x=self.Robot.BaseToWorld(self.Robot._actual.x), out="T")

        self._tt = self.simtime()
        self._last_update = self.simtime()  # Do not change !

    # Movements
    def Set_vel(self, v: Any, wait: Optional[float] = None) -> None:
        """
        Command planar platform velocity and wait.

        Parameters
        ----------
        v : Any
            Desired planar velocity ``(forward, turn)``.
        wait : float, optional
            Duration for which the command should be maintained.

        Returns
        -------
        None
            This method publishes a planar velocity command.
        """
        v = vector(v, dim=2)
        if wait is None:
            wait = self.tsamp
        self._synchro_control(wait)

        _t0 = self.simtime()

        _fac = max(max(v / self.v_max), max(v / self.v_min), 1)
        v = v / _fac

        if self._default.CheckObstacles:
            if v[0] > 0:
                d, _ = self.GetClosestFrontObstacle(np.array([-1, 1]) * self._default.LaserAngleRange)
                d = d if d > 0 else self._last_closest_distance
                self._last_closest_distance = d

                if abs(d) < 0.5 * self._default.ObstacleMinDist:
                    self.Message("Very close to front obstacle", 2)
                elif abs(d) < 0.75 * self._default.ObstacleMinDist:
                    self.Message("Close to front obstacle", 2)
                elif abs(d) < self._default.ObstacleMinDist:
                    self.Message("Reducing velocity due to fron obstacle", 2)
                v[0] *= smoothstep(d, 0.2, 1)
                if (abs(d) < self._default.ObstacleMinDist) and (v[0] < self._default.MinVel):
                    self.Message("Stopping due to front obstacle", 2)
                    self._abort_motion = True
                    v[0] = 0
            elif v[0] < 0:
                d = self.GetClosestRearObstacle()
                if abs(d) < 0.5:
                    self.Message("Very close to rear obstacle", 2)
                elif abs(d) < 0.8:
                    self.Message("Close to rear obstacle", 2)
                elif abs(d) < 1:
                    self.Message("Reducing velocity due to rear obstacle", 2)
                v[0] *= smoothstep(d, 0.4, 1)
                if (abs(d) < self._default.ObstacleMinDist) and (-v[0] < self._default.MinVel):
                    self.Message("Stopping due to rear obstacle", 1)
                    self._abort_motion = True
                    v[0] = 0

        self._cmd_msg.linear.x = v[0]
        self._cmd_msg.angular.z = v[1]
        self._command.ux = v
        self._cmd_pub.publish(self._cmd_msg)
        self.GetState()
        self.Update()

        while self.simtime() - _t0 < wait - self.tsamp:
            self.send(self.cmd_pub, self.cmd_msg)
            sleep(self.tsamp)
            self.GetState()
            self.Update()

    # Localization (SLAM or OPTI)
    def SetLocalization(self, localization: str = "SLAM", natnet_host_ip: Optional[str] = None, natnet_client_ip: Optional[str] = None, natnet_multicast_ip: Optional[str] = None) -> None:
        """
        Select SLAM or OptiTrack localization for the platform.

        Parameters
        ----------
        localization : str, optional
            Requested localization mode.
        natnet_host_ip : str, optional
            Optional NatNet host IP override.
        natnet_client_ip : str, optional
            Optional NatNet client IP override.
        natnet_multicast_ip : str, optional
            Optional NatNet multicast IP override.
        """
        if check_option(localization, "SLAM"):
            self.localization = "SLAM"
        elif check_option(localization, "OPTI"):
            if self.opti is None:
                if natnet_host_ip is not None:
                    self._natnet_host_ip = natnet_host_ip
                if natnet_client_ip is not None:
                    self._natnet_client_ip = natnet_client_ip
                if natnet_multicast_ip is not None:
                    self._natnet_multicast_ip = natnet_multicast_ip
                self.opti = optitrack_localization(host_ip=self._natnet_host_ip, client_ip=self._natnet_client_ip, multicast_ip=self._natnet_multicast_ip)
                self.opti.SetBodyID(id=self._natnet_platform_rigidbody_id)
            self.localization = "OPTI"
        else:
            self.Message(f"Invalid selected localization  '{localization}'! Use 'SLAM' or 'OPTI'.", 0)

    def GetMapLocation(self, out: str = "2d") -> Any:
        """
        Retrieves the current location the map frame.

        This function computes the transformation between the SLAM and OPTI frames
        and returns the mapped pose of the map.

        Parameters
        ----------
        out : str, optional
            The format of the output location, default is "2d".

        Returns
        -------
        np.ndarray or float
            If localization data is available, returns the transformed pose in the desired format.
            Otherwise, returns NaN.
        """
        if self._actual_SLAM_x is not None:
            if self._actual_OPTI_x is not None:
                _tmp = x2t(self._actual_OPTI_x) @ np.linalg.inv(x2t(self._actual_SLAM_x))
                return map_pose(T=_tmp, out=out)
            else:
                return map_pose(p=[0, 0, 0], out=out)
        else:
            return np.nan

    def SetSLAMFrame(self, x: Optional[Any] = None) -> None:
        """
        Set the SLAM frame transform used by the platform wrapper.

        Parameters
        ----------
        x : Any, optional
            Pose specification of the SLAM frame transform.
        """
        if x is None:
            pass  # ToDo: Preberi trenutno lokacijo SLAM
        else:
            _x = x
        self.xSLAM = _x


if __name__ == "__main__":
    # Run platform
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    b = tiagobase()
    print("Robot:", b.Name)
    print("Pos: ", b.p)
