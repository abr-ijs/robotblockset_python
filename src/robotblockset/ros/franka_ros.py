"""ROS Franka robot interfaces.

This module defines ROS-backed interfaces for Franka Panda and FR3 robots. It
provides state handling, controller integration, compliance management, load
configuration, collision-behavior services, and tool-center-point updates.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic.
"""

import math
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import rospy
import numpy as np
import json
import copy
import actionlib

from franka_msgs.srv import (
    SetJointImpedance,
    SetJointImpedanceRequest,
    SetCartesianImpedance,
    SetKFrame,
    SetKFrameRequest,
    SetForceTorqueCollisionBehavior,
    SetForceTorqueCollisionBehaviorRequest,
    SetFullCollisionBehavior,
    SetFullCollisionBehaviorRequest,
    SetEEFrame,
    SetEEFrameRequest,
    SetLoad,
    SetLoadRequest,
)
from franka_msgs.msg import (
    FrankaState,
    ErrorRecoveryAction,
)

from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray, Bool

from roscpp.srv import SetLoggerLevel, SetLoggerLevelRequest


from robotblockset.transformations import map_pose, t2x, frame2world
from robotblockset.robots import robot, MotionResultCodes
from robotblockset.robot_spec import panda_spec, fr3_spec
from robotblockset.tools import _struct, rbs_type, isscalar, vector, isvector, matrix, ismatrix, check_option


from robotblockset.ros.controllers_ros import joint_impedance_controller, cartesian_impedance_controller
from robotblockset.ros.joint_trajectory_interface import JointTrajectory
from robotblockset.ros.robots_ros import robot_ros
from robotblockset.rbs_typing import HomogeneousMatrixType, JointConfigurationType, JointPathType, JointTorqueType, Pose3DType, RotationMatrixType, Vector3DType, WrenchType


class JointCompliance(_struct):
    """
    Container for Franka joint-compliance parameters.

    Attributes
    ----------
    K : np.ndarray | None
        Joint stiffness vector.
    D : np.ndarray | None
        Joint damping vector.
    """

    def __init__(self) -> None:
        """
        Initialize joint stiffness and damping storage.

        Notes
        -----
        Values are initialized to ``None`` and are populated later from the
        Franka defaults or explicit user settings.
        """
        self.K = None
        self.D = None


class CartesianCompliance(_struct):
    """
    Container for Franka Cartesian-compliance parameters.

    Attributes
    ----------
    Kp : np.ndarray | None
        Translational stiffness vector.
    Kr : np.ndarray | None
        Rotational stiffness vector.
    R : np.ndarray | None
        Rotation matrix of the translational stiffness frame.
    D : float | None
        Damping scaling factor.
    """

    def __init__(self) -> None:
        """
        Initialize Cartesian stiffness, orientation, and damping storage.

        Notes
        -----
        Values are initialized to ``None`` and are populated later from the
        Franka defaults or explicit user settings.
        """
        self.Kp = None
        self.Kr = None
        self.R = None
        self.D = None


class FrankaCollisionBehaviour(_struct):
    """
    Container for Franka collision-threshold parameters.

    Attributes
    ----------
    lower_torque_thresholds_acceleration : np.ndarray | None
        Lower joint-torque thresholds in acceleration mode.
    upper_torque_thresholds_acceleration : np.ndarray | None
        Upper joint-torque thresholds in acceleration mode.
    lower_torque_thresholds_nominal : np.ndarray | None
        Lower joint-torque thresholds in nominal mode.
    upper_torque_thresholds_nominal : np.ndarray | None
        Upper joint-torque thresholds in nominal mode.
    lower_force_thresholds_acceleration : np.ndarray | None
        Lower Cartesian force thresholds in acceleration mode.
    upper_force_thresholds_acceleration : np.ndarray | None
        Upper Cartesian force thresholds in acceleration mode.
    lower_force_thresholds_nominal : np.ndarray | None
        Lower Cartesian force thresholds in nominal mode.
    upper_force_thresholds_nominal : np.ndarray | None
        Upper Cartesian force thresholds in nominal mode.
    """

    def __init__(self) -> None:
        """Initialize storage for Franka collision-threshold parameters."""
        self.lower_torque_thresholds_acceleration = None
        self.upper_torque_thresholds_acceleration = None
        self.lower_torque_thresholds_nominal = None
        self.upper_torque_thresholds_nominal = None
        self.lower_force_thresholds_acceleration = None
        self.upper_force_thresholds_acceleration = None
        self.lower_force_thresholds_nominal = None
        self.upper_force_thresholds_nominal = None


class FrankaDefaults(_struct):
    """
    Container for default Franka controller and safety parameters.

    Attributes
    ----------
    InternalController : str
        Default internal Franka controller mode.
    JointCompliance : JointCompliance
        Default joint-compliance parameters.
    CartesianCompliance : CartesianCompliance
        Default Cartesian-compliance parameters.
    CollisionBehavior : FrankaCollisionBehaviour
        Default collision-threshold parameters.
    """

    def __init__(self) -> None:
        """Initialize default Franka compliance and collision parameters."""
        self.InternalController = "joint_impedance"
        self.InternalJointCompliance = np.array([3500, 3500, 3500, 3000, 2500, 2000, 1000])
        self.InternalCartesianCompliance = np.array([2500, 2500, 2500, 250, 250, 250])
        self.Max_InternalJointCompliance = np.array([6000, 6000, 6000, 6000, 6000, 6000, 6000])
        self.Max_InternalCartesianCompliance = np.array([3000, 3000, 3000, 300, 300, 300])
        self.Min_InternalJointCompliance = np.array([0, 0, 0, 0, 0, 0, 0])
        self.Min_InternalCartesianCompliance = np.array([10, 10, 10, 1, 1, 1])
        self.CartesianStiffnessFrame = np.eye(4)
        self.JointCompliance = JointCompliance()
        self.JointCompliance.K = np.array([1200, 1200, 1200, 1200, 250, 250, 100])
        self.JointCompliance.D = np.array([25, 25, 25, 25, 10, 10, 10])
        self.CartesianCompliance = CartesianCompliance()
        self.CartesianCompliance.Kp = np.array([2000, 2000, 2000])
        self.CartesianCompliance.Kr = np.array([30, 30, 30])
        self.CartesianCompliance.R = np.eye(3)
        self.CartesianCompliance.D = 2.0
        self.MinSoftnessForMotion = 0.005  # Expected minimal compliance which can allow motion
        self.CollisionBehavior = FrankaCollisionBehaviour()
        self.CollisionBehavior.lower_torque_thresholds_acceleration = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.upper_torque_thresholds_acceleration = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.lower_torque_thresholds_nominal = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.upper_torque_thresholds_nominal = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.lower_force_thresholds_acceleration = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])
        self.CollisionBehavior.upper_force_thresholds_acceleration = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])
        self.CollisionBehavior.lower_force_thresholds_nominal = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])
        self.CollisionBehavior.upper_force_thresholds_nominal = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])


class panda_ros(panda_spec, fr3_spec, robot_ros):
    """
    ROS interface for Franka Panda and FR3 robots.

    Attributes
    ----------
    collision_behavior : FrankaCollisionBehaviour
        Active collision-threshold configuration.
    joint_compliance : JointCompliance
        Active joint-compliance configuration.
    cartesian_compliance : CartesianCompliance
        Active Cartesian-compliance configuration.
    """

    def __init__(self, ns: str = "", model: str = "panda", init_node: bool = True, multi_node: bool = False, control_strategy: str = "JointImpedance") -> None:
        """
        Initialize a ROS-backed Franka Panda or FR3 wrapper.

        Parameters
        ----------
        ns : str, optional
            ROS namespace of the robot.
        model : str, optional
            Robot model, typically ``"panda"`` or ``"fr3"``.
        init_node : bool, optional
            If ``True``, initialize a ROS node for the wrapper.
        multi_node : bool, optional
            If ``True``, initialize the ROS node in anonymous mode.
        control_strategy : str, optional
            Initial RobotBlockSet control strategy.
        """

        strategy_controller_mapping = {
            # "RbsName": ["ros_controller_name", rbs_controller_support_class_name]
            "JointImpedance": ["joint_impedance_controller", joint_impedance_controller],
            "CartesianImpedance": ["cartesian_impedance_controller", cartesian_impedance_controller],
            # "JointPositionTrajectory": "position_joint_trajectory_controller",
            # "JointDMP": "joint_dmp_controller",
        }

        robot_ros.__init__(self, ns=ns, init_node=init_node, multi_node=multi_node, control_strategy=control_strategy, strategy_controller_mapping=strategy_controller_mapping)

        if model == "panda":
            panda_spec.__init__(self)

        if model == "fr3":
            fr3_spec.__init__(self)
        self.joint_names = [f"{model}_joint{i+1}" for i in range(self.nj)]

        self.tsamp = 1.0 / 300.0
        self._active = False
        self._connected = False

        # Initialize franka parameters
        self._franka_default = FrankaDefaults()
        self.collision_behavior = copy.deepcopy(self._franka_default.CollisionBehavior)
        self._init_franka_ros_interfaces()

        # Initialize panda specific parameters for controllers
        self.joint_compliance = copy.deepcopy(self._franka_default.JointCompliance)
        self.cartesian_compliance = copy.deepcopy(self._franka_default.CartesianCompliance)

        rospy.wait_for_message(self.franka_state_topic, FrankaState, 1)
        self.Init()
        self._update_tcp_from_franka_state()

        self._connected = True
        self.Message("Initialized", 1)

    # ROS stuff
    def _init_franka_ros_interfaces(self) -> None:
        """
        Initialize ROS publishers, subscribers, and services for Franka control.

        Notes
        -----
        State subscribers and common service proxies are created only once when
        the wrapper is first initialized.
        """

        franka_control_topic_ns = f"{self._namespace}/franka_control"
        if not self._connected:
            # Franka state
            self.joint_state = None  # If the controller is dead, there will be no state callback
            self._last_joint_state_callback_time = None  # Last time we got FrankaState (inside GetStateCallback() )
            self._last_update = self.simtime()  # Last time we did GetState()
            self.franka_state_topic = f"{self._namespace}/franka_state_controller/franka_states"
            self.franka_state_subscriber = rospy.Subscriber(self.franka_state_topic, FrankaState, self.GetStateCallback)

            # Franka internal states
            self.joint_impedance_franka_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_joint_impedance", SetJointImpedance)
            self.cart_impedance_franka_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_cartesian_impedance", SetCartesianImpedance)
            self.K_frame_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_K_frame", SetKFrame)
            self.force_torque_limits_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_force_torque_collision_behavior", SetForceTorqueCollisionBehavior)
            self.full_force_torque_limits_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_full_collision_behavior", SetFullCollisionBehavior)
            self.EE_frame_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_EE_frame", SetEEFrame)
            self.EE_load_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_load", SetLoad)
            self.error_recovery_action_client = actionlib.SimpleActionClient(f"{franka_control_topic_ns}/error_recovery", ErrorRecoveryAction)
            self.logger_svc_proxy = rospy.ServiceProxy(f"{franka_control_topic_ns}/set_logger_level", SetLoggerLevel)

    def ResetCurrentTarget(self, do_move: bool = False, **kwargs: Any) -> None:
        """
        Reset the current target and optionally synchronize the active controller target.

        Parameters
        ----------
        do_move : bool, optional
            If ``True``, propagate the current measured state to the active
            controller target before resetting the RobotBlockSet state.
        **kwargs : Any
            Additional keyword arguments forwarded to the base implementation.
        """

        if self._control_strategy in ["CartesianImpedance", "JointImpedance"] and do_move == True:
            # call internal controller strategy to set controller target values to measured
            self._controller.ResetCurrentTarget()
            # update internal RBS states, but don't send GoTo command to the robot
            robot.ResetCurrentTarget(self, do_move=False)

        robot.ResetCurrentTarget(self, do_move, **kwargs)

    def GetStateCallback(self, data: FrankaState) -> None:
        """
        Store the latest Franka state message.

        Parameters
        ----------
        data : FrankaState
            Franka state message received from ROS.
        """
        self.joint_state = data
        self._last_joint_state_callback_time = self.simtime()

    def GetState(self) -> Optional[FrankaState]:
        """
        Update the RobotBlockSet state from the latest Franka state message.

        Returns
        -------
        FrankaState | None
            Latest Franka state message, if available.
        """

        _state = self.joint_state
        t = self.simtime()
        if _state is not None:
            self._tt = self.simtime()
            self._actual.q = rbs_type(_state.q)
            self._actual.qdot = rbs_type(_state.dq)
            self._actual.trq = rbs_type(_state.tau_J)

            if self._control_strategy in ["CartesianVelocity"]:
                self._command.q = rbs_type(_state.q_d)
                self._command.qdot = rbs_type(_state.dq_d)
                T_D = np.reshape(_state.O_T_EE_d, (4, 4), order="F")
                self._command.x = t2x(T_D)

            T = np.reshape(_state.O_T_EE, (4, 4), order="F")
            self._actual.x = t2x(T)
            self._actual.v = self.Jacobi(_state.q) @ _state.dq

            # Get safety status
            self.joint_contacts = _state.joint_contact
            self.joint_collisions = _state.joint_collision
            self.cartesian_contacts = _state.cartesian_contact
            self.cartesian_collisions = _state.cartesian_collision

            self._actual.FT = _state.K_F_ext_hat_K

            self.Tstiff = np.reshape(_state.EE_T_K, (4, 4), order="F")
            self._actual.FT = frame2world(_state.K_F_ext_hat_K, self.Tstiff, 1)  # external EE wrench in tool CS (considering EETK)
            self._actual.F = self._actual.FT[0:2]
            self._actual.T = self._actual.FT[2:]
            self._actual.trqExt = _state.tau_ext_hat_filtered

            if self.EEFixed:
                self.TObject = map_pose(x=self.BaseToWorld(self._actual.x), out="T")

            self._last_update = self.simtime()  # Do not change !

        return _state

    # Behaviour and collisions
    def GetCollisions(self) -> Tuple[Any, Any]:
        """Return the latest joint and Cartesian collision indicators."""
        _qcol = self.joint_collisions
        _xcol = self.cartesian_collisions
        return _qcol, _xcol

    def GetContacts(self) -> Tuple[Any, Any]:
        """Return the latest joint and Cartesian contact indicators."""
        _qcon = self.joint_contacts
        _xcon = self.cartesian_contacts
        return _qcon, _xcon

    def GetCollisionBehaviour(self) -> Dict[str, Any]:
        """Return the configured Franka collision-threshold parameters."""
        return self.collision_behavior.asdict()

    def SetCollisionBehavior(self, F: WrenchType, tq: JointTorqueType, F_low: Optional[WrenchType] = None, tq_low: Optional[JointTorqueType] = None, restart: bool = False) -> Optional[int]:
        """
        Set nominal Franka collision thresholds.

        Parameters
        ----------
        F : WrenchType
            Nominal Cartesian force and torque thresholds.
        tq : JointTorqueType
            Nominal joint-torque thresholds.
        F_low : WrenchType, optional
            Lower Cartesian thresholds. If ``None``, half of ``F`` is used.
        tq_low : JointTorqueType, optional
            Lower joint-torque thresholds. If ``None``, half of ``tq`` is used.
        restart : bool, optional
            If ``True``, restart the active controller around the update.

        Returns
        -------
        int | None
            ``1`` on exception, otherwise ``None``.
        """
        if isscalar(F):
            F = np.ones(6) * F
        else:
            F = vector(F, dim=6)

        if isscalar(tq):
            tq = np.ones(7) * tq
        else:
            tq = vector(tq, dim=7)

        if F_low is None:
            F_low = F * 0.5
        elif isscalar(F_low):
            F_low = np.ones(6) * F_low
        else:
            F_low = vector(F_low, dim=6)

        if tq_low is None:
            tq_low = tq * 0.5
        elif isscalar(tq_low):
            tq_low = np.ones(6) * tq_low
        else:
            tq_low = vector(tq_low, dim=7)

        collision_behavior = SetForceTorqueCollisionBehaviorRequest()
        collision_behavior.upper_force_thresholds_nominal = F
        collision_behavior.lower_force_thresholds_nominal = F_low
        collision_behavior.upper_torque_thresholds_nominal = tq
        collision_behavior.lower_torque_thresholds_nominal = tq_low

        self.collision_behavior.upper_force_thresholds_nominal = F
        self.collision_behavior.lower_force_thresholds_nominal = F_low
        self.collision_behavior.upper_torque_thresholds_nominal = tq
        self.collision_behavior.lower_torque_thresholds_nominal = tq_low

        # Stop controller
        if restart:
            self.controller_helper.stop_active_controller()

        try:
            success = self.force_torque_limits_proxy.call(collision_behavior)
            if success.success == True:
                self.Message("ColThr call: {0}".format(success.success))
                self.Message(f"Collision behavior changed: \nF {F}[N]  T {tq}[Nm]")
            elif success.success == False:
                self.Message("Successful: {0}".format(success.success))
        except Exception as e:
            self.Message("ColThr exception: {}".format(e))
            return 1

        sleep(0.3)
        if restart:
            self.controller_helper.start_last_controller()
        self.GetState()
        self.Update()

    def SetFullCollisionBehavior(self, F: WrenchType, tq: JointTorqueType, F_acc: Optional[WrenchType] = None, tq_acc: Optional[JointTorqueType] = None, F_low: Optional[WrenchType] = None, tq_low: Optional[JointTorqueType] = None, F_acc_low: Optional[WrenchType] = None, tq_acc_low: Optional[JointTorqueType] = None, restart: bool = False) -> Optional[int]:
        """
        Set nominal and acceleration Franka collision thresholds.

        Parameters
        ----------
        F : WrenchType
            Nominal Cartesian force and torque thresholds.
        tq : JointTorqueType
            Nominal joint-torque thresholds.
        F_acc : WrenchType, optional
            Acceleration-mode Cartesian thresholds.
        tq_acc : JointTorqueType, optional
            Acceleration-mode joint-torque thresholds.
        F_low : WrenchType, optional
            Lower nominal Cartesian thresholds.
        tq_low : JointTorqueType, optional
            Lower nominal joint-torque thresholds.
        F_acc_low : WrenchType, optional
            Lower acceleration Cartesian thresholds.
        tq_acc_low : JointTorqueType, optional
            Lower acceleration joint-torque thresholds.
        restart : bool, optional
            If ``True``, restart the active controller around the update.

        Returns
        -------
        int | None
            ``1`` on exception, otherwise ``None``.
        """
        if isscalar(F):
            F = np.ones(6) * F
        else:
            F = vector(F, dim=6)

        if isscalar(tq):
            tq = np.ones(7) * tq
        else:
            tq = vector(tq, dim=7)

        if F_low is None:
            F_low = F * 0.5
        elif isscalar(F_low):
            F_low = np.ones(6) * F_low
        else:
            F_low = vector(F_low, dim=6)

        if tq_low is None:
            tq_low = tq * 0.5
        elif isscalar(tq_low):
            tq_low = np.ones(6) * tq_low
        else:
            tq_low = vector(tq_low, dim=7)

        if F_acc is None:
            F_acc = F
        elif isscalar(F_acc):
            F_acc = np.ones(6) * F_acc
        else:
            F_acc = vector(F_acc, dim=6)

        if tq_acc is None:
            tq_acc = tq
        elif isscalar(tq_acc):
            tq_acc = np.ones(7) * tq_acc
        else:
            tq_acc = vector(tq_acc, dim=7)

        if F_acc_low is None:
            F_acc_low = F_low * 0.5
        elif isscalar(F_acc_low):
            F_acc_low = np.ones(6) * F_acc_low
        else:
            F_acc_low = vector(F_acc_low, dim=6)

        if tq_acc_low is None:
            tq_acc_low = tq_acc * 0.5
        elif isscalar(tq_acc_low):
            tq_acc_low = np.ones(6) * tq_acc_low
        else:
            tq_acc_low = vector(tq_acc_low, dim=7)

        collision_behavior = SetFullCollisionBehaviorRequest()
        collision_behavior.upper_force_thresholds_nominal = F
        collision_behavior.lower_force_thresholds_nominal = F_low
        collision_behavior.upper_torque_thresholds_nominal = tq
        collision_behavior.lower_torque_thresholds_nominal = tq_low
        collision_behavior.upper_force_thresholds_acceleration = F_acc
        collision_behavior.lower_force_thresholds_acceleration = F_acc_low
        collision_behavior.upper_torque_thresholds_acceleration = tq_acc
        collision_behavior.lower_torque_thresholds_acceleration = tq_acc_low

        self.collision_behavior.upper_force_thresholds_nominal = F
        self.collision_behavior.lower_force_thresholds_nominal = F_low
        self.collision_behavior.upper_torque_thresholds_nominal = tq
        self.collision_behavior.lower_torque_thresholds_nominal = tq_low
        self.collision_behavior.upper_force_thresholds_acceleration = F_acc
        self.collision_behavior.lower_force_thresholds_acceleration = F_acc_low
        self.collision_behavior.upper_torque_thresholds_acceleration = tq_acc
        self.collision_behavior.lower_torque_thresholds_acceleration = tq_acc_low

        # Stop controller
        if restart:
            self.controller_helper.stop_active_controller()

        # call(robot.set_force_torque_collision_behavior,collision_behavior,'Timeout',2);
        # self.Message("Calling ")
        # self.force_torque_limits_proxy.call(collision_behavior)

        try:
            if 1:
                # if self.force_torque_limits_proxy.is_available(self.force_torque_limits_topic, timeout=2):
                success = self.full_force_torque_limits_proxy.call(collision_behavior)
                if success.success == True:
                    self.Message("FullColThr call: {0}".format(success.success))
                    self.Message("FullCollision behavior changed: \nF_acc   [N]: {0} \nF_nom [N]: {1} \ntq_acc [Nm]: {2}\ntq_nom [Nm]: {3}".format(F_acc, F, tq_acc, tq))
                elif success.success == False:
                    self.Message("Successful: {0}".format(success.success))
        except Exception as e:
            # except (CommandException, NetworkException) as e:
            self.Message("FullCollisionThresholds exception: {}".format(e))
            return 1

        sleep(0.3)
        self.controller_helper.start_last_controller()
        self.GetState()
        self.Update()

    # Cartesian compliance
    def GetCartesianCompliance(self) -> Tuple[Any, Any, Any, Any]:
        """Return the active Cartesian compliance parameters."""
        return self.cartesian_compliance.Kp, self.cartesian_compliance.Kr, self.cartesian_compliance.R, self.cartesian_compliance.D

    def SetCartesianCompliance(self, Kp: Optional[Vector3DType] = None, Kr: Optional[Vector3DType] = None, R: Optional[RotationMatrixType] = None, D: Optional[float] = None, hold_pose: bool = True) -> int:
        """
        Update Cartesian compliance parameters through the active controller.

        Parameters
        ----------
        Kp : Vector3DType, optional
            Translational stiffness values.
        Kr : Vector3DType, optional
            Rotational stiffness values.
        R : RotationMatrixType, optional
            Rotation matrix of the translational stiffness frame.
        D : float, optional
            Damping scaling factor.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating compliance.

        Returns
        -------
        int
            Result code returned by the active controller.
        """
        return self.controller.SetCartesianCompliance(Kp=Kp, Kr=Kr, R=R, D=D, hold_pose=hold_pose)

    def GetCartesianStiffness(self) -> Tuple[Any, Any]:
        """Return the active Cartesian stiffness parameters."""
        return self.cartesian_compliance.Kp, self.cartesian_compliance.Kr

    def SetCartesianStiffness(self, Kp: Optional[Vector3DType] = None, Kr: Optional[Vector3DType] = None, hold_pose: bool = True) -> int:
        """
        Update Cartesian stiffness while preserving the remaining compliance settings.

        Parameters
        ----------
        Kp : Vector3DType, optional
            Translational stiffness values.
        Kr : Vector3DType, optional
            Rotational stiffness values.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating stiffness.

        Returns
        -------
        int
            Result code returned by :meth:`SetCartesianCompliance`.
        """
        return self.SetCartesianCompliance(Kp=Kp, Kr=Kr, R=self.cartesian_compliance.R, D=self.cartesian_compliance.D, hold_pose=hold_pose)
        pass

    def GetCartesianDamping(self) -> Any:
        """Return the active Cartesian damping parameter."""
        return self.cartesian_compliance.D

    def SetCartesianDamping(self, D: Optional[float] = None, hold_pose: bool = True) -> int:
        """
        Update Cartesian damping while preserving the remaining compliance settings.

        Parameters
        ----------
        D : float, optional
            Damping scaling factor.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating damping.

        Returns
        -------
        int
            Result code returned by :meth:`SetCartesianCompliance`.
        """
        return self.SetCartesianCompliance(Kp=self.cartesian_compliance.Kp, Kr=self.cartesian_compliance.Kr, R=self.cartesian_compliance.R, D=D, hold_pose=hold_pose)

    def SetCartesianSoft(self, stiffness: Any, hold_pose: bool = True) -> int:
        """
        Scale Cartesian compliance relative to the Franka default values.

        Parameters
        ----------
        stiffness : Any
            Scalar or vector compliance scaling factor.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating compliance.

        Returns
        -------
        int
            ``0`` when the scaling request is applied.
        """
        if isscalar(stiffness):
            fac_p = np.ones(3) * stiffness
            fac_r = fac_p
        elif isvector(stiffness, dim=3):
            fac_p = stiffness
            fac_r = stiffness
        else:
            fac = vector(stiffness, dim=6)
            fac_p = fac[:3]
            fac_r = fac[3:]

        fac_p = np.clip(fac_p, 0.0, 1.0)
        fac_r = np.clip(fac_r, 0.0, 1.0)
        fac = np.max(np.hstack((fac_p, fac_r)))
        self.SetCartesianCompliance(
            Kp=self._franka_default.CartesianCompliance.Kp * fac_p,
            Kr=self._franka_default.CartesianCompliance.Kr * fac_r,
            R=self._franka_default.CartesianCompliance.R,
            D=self._franka_default.CartesianCompliance.D * fac,
            hold_pose=hold_pose,
        )
        return 0

    # Joint compliance
    def GetJointCompliance(self) -> Tuple[Any, Any]:
        """Return the active joint compliance parameters."""
        return self.joint_compliance.K, self.joint_compliance.D

    def SetJointCompliance(self, K: Optional[JointTorqueType] = None, D: Optional[JointTorqueType] = None, hold_pose: bool = True) -> None:
        """
        Update joint compliance parameters through the active controller.

        Parameters
        ----------
        K : JointTorqueType, optional
            Joint stiffness values.
        D : JointTorqueType, optional
            Joint damping values.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating compliance.
        """
        self.controller.SetJointCompliance(K, D, hold_pose)

    def GetJointStiffness(self) -> Any:
        """Return the active joint stiffness values."""
        return self.joint_compliance.K

    def SetJointStiffness(self, K: Optional[JointTorqueType] = None, hold_pose: bool = True) -> None:
        """
        Update joint stiffness while preserving the remaining compliance settings.

        Parameters
        ----------
        K : JointTorqueType, optional
            Joint stiffness values.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating stiffness.
        """
        self.SetJointCompliance(K=K, D=self.joint_compliance.D, hold_pose=hold_pose)

    def GetJointDamping(self) -> Any:
        """Return the active joint damping values."""
        return self.joint_compliance.D

    def SetJointDamping(self, D: Optional[JointTorqueType] = None, hold_pose: bool = True) -> None:
        """
        Update joint damping while preserving the remaining compliance settings.

        Parameters
        ----------
        D : JointTorqueType, optional
            Joint damping values.
        hold_pose : bool, optional
            If ``True``, hold the current target pose while updating damping.
        """
        self.SetJointCompliance(K=self.joint_compliance.K, D=D, hold_pose=hold_pose)

    def SetJointSoft(self, stiffness: Any, hold_pose: bool = True) -> None:
        """Scale joint compliance relative to the Franka default values."""
        if isscalar(stiffness):
            fac = np.ones(7) * stiffness
        else:
            fac = vector(stiffness, dim=7)
        fac = np.clip(fac, 0.0, 1.0)
        self.SetJointCompliance(self._franka_default.JointCompliance.K * fac, self._franka_default.JointCompliance.D * fac, hold_pose=hold_pose)

    # Impedance parameters used for franka's internal motion generators
    def SetCartesianImpedanceFranka(self, stiffness: Any) -> int:
        """Set Franka Cartesian impedance.

        Sets the cartesian impedance of the internal franka controller.
                These values only have effect when franka's internal motion generators are used."""
        request = SetCartesianImpedance()
        request.cartesian_stiffness = stiffness
        try:
            success = self.cart_impedance_franka_proxy.call(request)
            if success.success == True:
                self.Message("Franka cart imp. successful: {0}".format(success.success))
                return 0
            elif success.success == False:
                self.Message("Franka cart imp. successful: {0}".format(success.success))
                return 1
        except Exception as e:
            self.Message(e)
            return 1
        return 0

    def SetJointImpedanceFranka(self, stiffness: Any, restart: bool = False) -> int:
        """Sets the joint impedance of the internal franka controller
        Parameters
        ----------
        ------------------
        restart : bool, optional
            if True, then  controller will first be stopped, impedance changed, then the controller will be started."""

        if restart:
            self.controller_helper.stop_active_controller()

        # Max value is around 14200
        MAX_VALUE = 14000
        for _x in stiffness:
            if _x <= MAX_VALUE:
                raise ValueError("Stiffness value {stiffnes} too high, max value: {MAX_VALUE}")

        request = SetJointImpedanceRequest()
        request.joint_stiffness = stiffness
        try:
            if 1:
                # if self.K_frame_proxy.is_available(self.K_frame_topic):
                success = self.joint_impedance_franka_proxy.call(request)
                if success.success == True:
                    self.Message("Franka joint imp. successful: {0}".format(success.success))
                    return 0
                elif success.success == False:
                    self.Message("Franka joint imp. successful: {0}".format(success.success))
                    return 1
        except Exception as e:
            # except (CommandException, NetworkException) as e:
            self.Message(e)
            return 1

        if restart:
            self.controller_helper.start_last_controller()
        return 0

    # TCP
    def _update_tcp_from_franka_state(self) -> None:
        """Refresh the TCP from Franka Desk state.

        Reads the currently set TCP in franka desk and sets it as default inside the panda_ros object"""
        frankadesk_TCP = self.joint_state.F_T_EE
        self.TCPGripper = np.reshape(frankadesk_TCP, (4, 4), order="f")
        self.SetTCP()  # Set TCP to none, then it reads self.TCPGripper

    def SetTCP(self, *tcp: Any, frame: str = "Gripper", send_to_robot: bool = False, EE_frame: str = "Flage") -> None:
        """Set the TCP locally and optionally propagate it to the robot."""
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
        if check_option(frame, "Robot"):
            newTCP = _tcp
        elif check_option(frame, "Gripper"):
            newTCP = self.TCPGripper @ _tcp
        else:
            raise ValueError(f"Frame '{frame}' not supported")
        self.TCP = newTCP
        rx, rJ = self.Kinmodel(self._command.q)
        self._command.x = rx
        self._command.v = rJ @ self._command.qdot

        if send_to_robot:
            # First stop currently running controller
            self.controller_helper.stop_active_controller()
            set_ee_request_msg = SetEEFrameRequest()
            if check_option(EE_frame, "Nominal"):
                NE_TCP = self.TCPGripper / newTCP
            elif check_option(EE_frame, "Flange"):
                NE_TCP = newTCP
            else:
                raise ValueError(f"EE_frame {EE_frame} not supported")

            set_ee_request_msg.NE_T_EE = np.reshape(NE_TCP, 16, order="A")
            response = self.EE_frame_proxy.call(set_ee_request_msg)
            self.Message("Response to SetTCP:", response)

            self.controller_helper.start_last_controller()

        self.GetState()
        self.Update()

    # x# ToDo
    def SetKFrame(self, EE_T_K: Any) -> int:
        """Set the EE-to-stiffness-frame transform.

        Sets the transformation ``EE_T_K`` from end-effector frame to stiffness frame.
                The transformation matrix is represented as a vectorized 4x4 matrix in column-major format.

                Parameters as follow
                -- EE_T_K           float[16]   Vectorized EE-to-K transformation matrix , column-major.
        """
        request = SetKFrameRequest()
        request.EE_T_K = EE_T_K
        try:
            if 1:
                # if self.K_frame_proxy.is_available(self.K_frame_topic):
                success = self.K_frame_proxy.call(request)
                if success.success:
                    self.Message("Successful: {0}".format(success.success))
                    return 0
                else:
                    self.Message("Successful: {0}".format(success.success))
                    return 1
        except Exception as e:
            self.Message(e)
            return 1
        return 0

    def GetStiffnessFrame(self) -> np.ndarray:
        """Return the current EE-to-stiffness-frame transform."""
        Tx = self.joint_state.EE_T_K
        T = np.reshape(Tx, (4, 4))
        return T

    # x# ToDo
    def SetStiffnessFrame(self, T: Optional[Any] = None) -> None:
        """Set the stiffness frame.

        SetStiffnessFrame Sets the stiffness frame (EETK) relative to EE frame
                (controller is temporary stopped!)"""
        if T is None:
            newT = np.eye(4)
        else:
            0
        self.Message("SetStiffnessFrame UNFINISHED")

    def SetLoad(self, mass: float, COM: Optional[tuple] = None, inertia: Optional[tuple] = None) -> Any:
        """Update the configured end-effector load on the robot."""

        if (not isscalar(mass)) and (mass <= 0):
            raise ValueError("Mass must be scalar > 0")
        COM = vector(COM, dim=3)
        inertia = matrix(inertia, shape=(3, 3))

        request = SetLoadRequest()
        request.mass = mass
        request.F_x_center_load = COM
        request.load_inertia = inertia.flatten()

        try:
            if 1:
                success = self.EE_load_proxy.call(request)
                if success.success == True:
                    self.Message("Successful: Success = {0}".format(success.success))
                    return success
                elif success.success == False:
                    self.Message("Unuccessful: Success ={0}".format(success.success))
                    return success
        except Exception as e:
            # except (CommandException, NetworkException) as e:
            self.Message(e)
            return 1
        return 0

    def SetEEFrame(self, NE_T_EE: Any) -> int:
        """
        Sets the transformation ``NE_T_EE`` from nominal end effector to end effector frame.
        The transformation matrix is represented as a vectorized 4x4 matrix in column-major format.

        Parameters as follow
        -- NE_T_EE          float[16]   4x4 matrix -> Vectorized NE-to-EE transformation matrix , column-major.
        """
        request = SetEEFrameRequest()
        request.NE_T_EE = NE_T_EE

        try:
            if 1:
                # if self.EE_frame_proxy.is_available(self.K_frame_topic):
                success = self.EE_frame_proxy.call(request)
                if success.success == True:
                    self.Message("Successful: Success = {0}".format(success.success))
                    return 1
                elif success.success == False:
                    self.Message("Unsuccessful: Success = {0}".format(success.success))
                    return 0
        except Exception as e:
            # except (CommandException, NetworkException) as e:
            self.Message(e)
            return 1
        return 0

    # Status
    def isConnected(self) -> bool:
        """Return whether the ROS Franka interface has been connected."""
        return self._connected

    def isReady(self) -> bool:
        """Return whether the ROS Franka interface is ready for commands."""
        return self._connected

    def isActive(self) -> bool:
        """Return whether the robot is in an active motion-capable state."""
        return self._active

    def Check(self, silent: bool = False) -> list:
        """
        Check the current Franka error state.

        Parameters
        ----------
        silent : bool, optional
            If ``True``, suppress status messages while checking the robot.

        Returns
        -------
        list[str]
            List of active Franka error flags.
        """
        self.GetState()
        _err = []
        error_list = self.joint_state.current_errors
        # prev_error_list = self.joint_state.last_motion_errors

        _attrs = dir(error_list)

        for attr in _attrs:
            # Attributes starting with _ are internal, for example __hash__. We dont care about them
            if attr[0] != "_":
                value = getattr(error_list, attr)
                if value is True:
                    _err += [attr]
        return _err

    def SetNewEEConfig(self, tool_name: str, restart: bool = True) -> int:
        """
        Load a new end-effector configuration from ROS parameters.

        Parameters
        ----------
        tool_name : str
            Name of the tool configuration on the ROS parameter server.
        restart : bool, optional
            If ``True``, restart the active controller after reconfiguration.

        Returns
        -------
        int
            ``0`` on success, otherwise a nonzero error code.
        """
        config = rospy.get_param(f"{self._namespace}/{tool_name}")
        tool_mass = config["mass"]
        tool_COM = config["centerOfMass"]
        tool_trans = config["transformation"]
        tool_inertia = config["inertia"]

        try:
            self.controller_helper.stop_active_controller()
        except Exception as e:
            self.Message(e)
            return 1
        try:
            success = self.SetEEFrame(NE_T_EE=tool_trans)
            if success:
                self.Message(f"Successful: Success = {success}")
            else:
                self.Message(f"Unsuccesful: Success = {success}")
                return success
        except Exception as e:
            self.Message(e)
            return 1
        try:
            success = self.SetLoad(mass=tool_mass, COM=tool_COM, inertia=tool_inertia)
            if success:
                self.Message(f"Successful: Success = {success}")
            else:
                self.Message(f"Unsuccesful: Success = {success}")
                return success
        except Exception as e:
            self.Message(e)
            return 1

        if restart:
            try:
                self.controller_helper.start_last_controller()
            except Exception as e:
                self.Message(e)
                return 1
        return 0

    def ErrorRecovery(self, enforce: bool = False, reset_target: bool = True) -> int:
        """
        Recover the robot from an error state.

        Parameters
        ----------
        enforce : bool, optional
            If ``True``, skip state checks and send the recovery request
            immediately.
        reset_target : bool, optional
            If ``True``, reset the current RobotBlockSet target before recovery.

        Returns
        -------
        int
            ``0`` when recovery handling completes.
        """

        # Check if we are in an error state at all
        try:
            self.GetState()
            if reset_target:
                self.ResetCurrentTarget()
        except:
            pass

        # If controller has failed in the beginning, we cannot call GetState since state is not set at all.
        if enforce:
            recovery_goal = ErrorRecoveryAction()
            self.error_recovery_action_client.send_goal(recovery_goal)
            sleep(0.1)
            return 0

        error_detected = False
        controller_failed = False
        if self.joint_state is None:
            error_detected = True
            controller_failed = True
        else:
            err_list = self.Check()
            if err_list:
                error_detected = True
                self.Message(f"Recovering from errors:\n {err_list}")

            if self.joint_state.robot_mode in [FrankaState.ROBOT_MODE_IDLE, FrankaState.ROBOT_MODE_REFLEX, FrankaState.ROBOT_MODE_OTHER]:
                error_detected = True
                self.Message(f"Attempting to change robot mode from {self.joint_state.robot_mode}")
            if self.joint_state.robot_mode == FrankaState.ROBOT_MODE_USER_STOPPED:
                self.Message("Error recovery is not possible: User Stop is pressed!")
                raise Exception("Error recovery not possible: User Stop is pressed!")

        if error_detected:
            if not controller_failed:
                self.GetState()
                self.ResetCurrentTarget()
            # if self._control_strategy == "CartesianImpedance":
            #    neutral_q = self.q_home
            #    self.controller.SetCartImpContNullspace(q=neutral_q, k=np.zeros(7))

            recovery_goal = ErrorRecoveryAction()
            self.error_recovery_action_client.send_goal(recovery_goal)
            sleep(0.1)

            self.error_recovery_action_client.wait_for_result(rospy.Duration(3))
            self.WarningMessage(f"Got error recovery result. Waiting for robot mode to be {FrankaState.ROBOT_MODE_MOVE} (i.e. frankaState.ROBOT_MODE_MOVE)")

            # Wait for controller to be up and running.
            if self._control_strategy is not None:
                while self.joint_state.robot_mode not in [FrankaState.ROBOT_MODE_MOVE]:
                    sleep(self.tsamp)
                    self.GetState()
                self.WarningMessage(f"Robot mode is {FrankaState.ROBOT_MODE_MOVE}")

        self._active = False
        sleep(0.01)
        return 0

    # Movements
    def Start(self) -> Any:
        """
        Activate the current control mode and prepare for motion.

        Returns
        -------
        Any
            Result returned by the base :meth:`robot.Start` implementation.
        """
        if self.HasError():
            raise Exception("Robot in error mode. Can not start!")

        if self._control_strategy in ["JointImpedance"]:
            if np.min(self.joint_compliance.K / self._franka_default.JointCompliance.K) < self._franka_default.MinSoftnessForMotion:
                self.WarningMessage("Robot is to compliant and will probably not move")
            self._active = True
        elif not self.isActive() and (self._control_strategy in ["CartesianImpedance"]):
            if min(np.min(self.cartesian_compliance.Kp / self._franka_default.CartesianCompliance.Kp), np.min(self.cartesian_compliance.Kr / self._franka_default.CartesianCompliance.Kr)) < self._franka_default.MinSoftnessForMotion:
                self.WarningMessage("Robot is to compliant and will probably not move")
            self.controller.ActivateController()
            self._active = True
        elif self._control_strategy in ["JointPositionTrajectory"]:
            # self.Message("Start() running")
            self.joint_trajectory_interface.clear()
            self.joint_trajectory_start_time = self.simtime()  # Keep this for logging how long calculations take.

        return robot.Start(self)

    def Stop(self) -> None:
        """
        Stop the active motion mode through the base robot API.
        """
        robot.Stop(self)

    def CheckContacts(self) -> int:
        """
        Return whether any Cartesian contact is currently detected.

        Returns
        -------
        int
            ``1`` if contact is detected, otherwise ``0``.
        """
        # Checks if a contact is ocurring (which contact level is activated).
        # if contact_values are all zero, there is no contact

        for v in self.joint_state.cartesian_contact:
            if v > 0:
                return 1
        # If we get to here, no collisions are detected
        return 0

    def GoTo_qtraj(self, qt: JointPathType, qdott: JointPathType, qddott: JointPathType, time: Any, wait: float) -> int:
        """
        Execute a joint trajectory through the ROS trajectory interface.

        Parameters
        ----------
        qt : JointPathType
            Joint-position trajectory.
        qdott : JointPathType
            Joint-velocity trajectory.
        qddott : JointPathType
            Joint-acceleration trajectory.
        time : Any
            Time vector for the trajectory points.
        wait : float
            Additional wait time after the trajectory finishes.

        Returns
        -------
        int
            Motion result code.
        """
        self.joint_trajectory_interface.clear()
        self.joint_trajectory_interface.add_points(positions=qt, velocities=qdott, accelerations=qddott, time=time)
        self.joint_trajectory_interface.start()
        self.joint_trajectory_interface.wait(self.joint_trajectory_interface._time_sum + wait)
        self.Update()
        return 0


class panda_ros_reconcycle(panda_ros):
    """
    ReconCycle-specific Franka Panda ROS wrapper.

    Attributes
    ----------
    Base_link_name : str
        Name of the robot base link used by the ReconCycle setup.
    """

    def __init__(self) -> None:
        """
        Initialize the ReconCycle-specific Panda wrapper.
        """
        panda_ros.__init__(self)
        self.Base_link_name = self.Name + "_link0"

    def shutdown_hook(self) -> None:
        """
        Handle ROS shutdown for the ReconCycle wrapper.
        """
        self.Message("{0} shutting down".format(self.Name))

    # legacy aliases
    def ResetCurrentTarget(self, send_msg: bool = True) -> Any:
        """
        Legacy alias for resetting the current target.

        Parameters
        ----------
        send_msg : bool, optional
            Legacy flag mapped to ``do_move`` of :meth:`panda_ros.ResetCurrentTarget`.

        Returns
        -------
        Any
            Result returned by :meth:`panda_ros.ResetCurrentTarget`.
        """
        return self.ResetCurrentTarget(self, do_move=send_msg)

    def check_contact(self) -> int:
        """
        Legacy alias for contact detection.

        Returns
        -------
        int
            Result returned by :meth:`CheckContacts`.
        """
        return self.CheckContacts()

    # x# ToDo
    def on_shutdown(self) -> None:
        """
        Register the ReconCycle shutdown hook with ROS.
        """
        rospy.on_shutdown(self.shutdown_hook)

    def SetNewEEConfig(self, tool_name: str, mass: Optional[float] = None, COM: Optional[list] = None, trans_mat: Optional[list] = None, inertia: Optional[list] = None, restart: bool = True) -> None:
        """
        Apply a new ReconCycle end-effector configuration.

        Parameters
        ----------
        tool_name : str
            Name of the JSON configuration file.
        mass : float, optional
            Tool mass.
        COM : list, optional
            Center of mass relative to the robot flange.
        trans_mat : list, optional
            Flattened transformation matrix from flange to end effector.
        inertia : list, optional
            Flattened inertia matrix.
        restart : bool, optional
            If ``True``, restart the active controller around the update.
        """
        if tool_name is not None:
            # Read JSON and load mass, COM, T_flange_to_EE
            GripperFile = open("/devel_ws/src/disassembly_pipeline/disassembly_pipeline/robot_ee_settings/" + tool_name)
            GripperSon = json.load(GripperFile)
            # print(f"{Fore.LIGHTMAGENTA_EX}{json.dumps(GripperSon, indent=1)}")
            GripperFile.close()
            Mass = GripperSon["mass"]
            self.Message("New load mass: {}".format(Mass))
            CenterOfMass = GripperSon["centerOfMass"]
            self.Message("New center of mass: {}".format(CenterOfMass))
            Transformation = GripperSon["transformation"]
            self.Message("New transformation matrix: {}".format(Transformation))
            Inertia = GripperSon["inertia"]
            0
        else:
            Mass = mass
            CenterOfMass = COM
            Transformation = trans_mat
            Inertia = inertia

        # Stop controller
        if restart:
            self.controller_helper.stop_active_controller()

        # (wait?)
        outFrame = self.SetEEFrame(NE_T_EE=Transformation)
        if outFrame:
            self.Message("New frame set: {0}".format(outFrame))
        else:
            self.Message("Failed new frame set: {0}".format(outFrame))

        outLoad = self.SetEELoad(mass=Mass, inertia=Inertia, center_of_mass=CenterOfMass)
        if outLoad:
            self.Message("New load set: {0}".format(outLoad))
        else:
            self.Message("New load set: {0}".format(outLoad))

        # sleep(0.5)
        # self.GetState();sleep(0.2)
        # self._update_tcp_from_franka_state() # Set robotblockset internal TCP
        # Read JSON of tool
        # Set EE mass and COM
        # self.SetLoadAndCOM()
        # Set EE T frame
        # self.SetEEFrame()

        rospy.set_param(self._namespace + "/current_tool", tool_name)

        # Start controller
        if restart:
            self.controller_helper.start_last_controller()
            sleep(1)
        self.GetState()
        self._update_tcp_from_franka_state()

    # def save_ros_parameters(self):
    #     # Save parameters in ROS Parameter Server
    #     params = dict()
    #     params["joint_compliance"] = dict(self.joint_compliance)
    #     params["cartesian_compliance"] = dict(self.cartesian_compliance)
    #     params["collision_thresholds"] = dict(self.collision_thresholds)
    #     rospy.set_param(self.Name + "/rbs_params", self.params)

    # def load_ros_parameters(self):
    #     # Load parameters from ROS Parameter Server and update internal state
    #     params = rospy.get_param(self.Name + "/rbs_params")
    #     self.joint_compliance.from_dict(params["joint_compliance"])
    #     self.cartesian_compliance.from_dict(params["cartesian_compliance"])
    #     self.collision_thresholds.from_dict(params["collision_thresholds"])


if __name__ == "__main__":
    from transformations import rot_x

    # Run robot
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    r = panda_ros(ns="pingvin")
    print("Robot:", r.Name)
    print("q: ", r.q)
    print("x: ", r.x)

    x = r.Kinmodel(out="x")[0]
    print("Robot pose:\n ", x)
    J = r.Jacobi()
    print("Robot Jacobian:\n ", J)

    print("Strategy:", r.GetStrategy())
    # r.SetCartesianSoft(0.2)
    print("Set strategy:", r.SetStrategy("JointImpedance"))

    r.ErrorRecovery()
    r.JMove(r.q_home, 2)

    r.SetJointSoft(1)
    print("Soft")
    r.SetJointSoft(0.1)
    print("Stiff")
    r.SetJointSoft(1)
    r.Wait(1)
    print("GetPose(task_space='Robot',kinematics='Robot','State','Commanded'): ", r.GetPose(task_space="Robot", kinematics="Robot", state="Commanded"))
    print("GetPose(task_space='Robot',kinematics='Calculated','State','Commanded'): ", r.GetPose(task_space="Robot", kinematics="Calculated", state="Commanded"))
    print("GetPose(task_space='Robot',kinematics='Robot'): ", r.GetPose(task_space="Robot", kinematics="Robot"))
    print("GetPose(task_space='Robot',kinematics='Calculated'): ", r.GetPose(task_space="Robot", kinematics="Calculated"))
    print("GetPose(task_space='World',kinematics='Robot','State','Commanded'): ", r.GetPose(task_space="World", kinematics="Robot", state="Commanded"))
    print("GetPose(task_space='World',kinematics='Calculated','State','Commanded'): ", r.GetPose(task_space="World", kinematics="Calculated", state="Commanded"))
    print("GetPose(task_space='World',kinematics='Robot'): ", r.GetPose(task_space="World", kinematics="Robot"))
    print("GetPose(task_space='World',kinematics='Calculated'): ", r.GetPose(task_space="World", kinematics="Calculated"))

    print("IKin:", r.IKin(map_pose(p=[0.5, 0.2, 0.5], Q=rot_x(np.pi)), r.q_home))

    print("Pose: ", r.GetPose())

    r.GetVel()

    print("FT: ", r.GetFT())
