#!/usr/bin/env python

import math
from time import sleep

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
from robotblockset.robots import robot
from robotblockset.robot_spec import panda_spec, fr3_spec
from robotblockset.tools import _struct, rbs_type, isscalar, vector, isvector, matrix, ismatrix, check_option


from robotblockset.ros.controllers_ros import joint_impedance_controller, cartesian_impedance_controller
from robotblockset.ros.joint_trajectory_interface import JointTrajectory
from robotblockset.ros.robots_ros import robot_ros

class JointCompliance(_struct):
    def __init__(self):
        self.K = None
        self.D = None

class CartesianCompliance(_struct):
    def __init__(self):
        self.Kp = None
        self.Kr = None
        self.R = None
        self.D = None
class FrankaCollisionBehaviour(_struct):
    def __init__(self):
        self.lower_torque_thresholds_acceleration = None
        self.upper_torque_thresholds_acceleration = None
        self.lower_torque_thresholds_nominal = None
        self.upper_torque_thresholds_nominal = None
        self.lower_force_thresholds_acceleration = None
        self.upper_force_thresholds_acceleration = None
        self.lower_force_thresholds_nominal = None
        self.upper_force_thresholds_nominal = None

class FrankaDefaults(_struct):
    def __init__(self):
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
    def __init__(self, ns="", model="panda", init_node=True, multi_node=False, control_strategy="JointImpedance"):

        strategy_controller_mapping = {
            # "RbsName": ["ros_controller_name", rbs_controller_support_class_name]
            "JointImpedance": ["joint_impedance_controller", joint_impedance_controller],
            "CartesianImpedance": ["cartesian_impedance_controller", cartesian_impedance_controller],
            #"JointPositionTrajectory": "position_joint_trajectory_controller",
            #"JointDMP": "joint_dmp_controller",
        }

        robot_ros.__init__(self,
                           ns=ns,
                           init_node=init_node,
                           multi_node=multi_node,
                           control_strategy=control_strategy,
                           strategy_controller_mapping=strategy_controller_mapping)

        if model == "panda":
            panda_spec.__init__(self)

        if model == "fr3":
            fr3_spec.__init__(self)

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

        self.Init()

        rospy.wait_for_message(self.franka_state_topic,FrankaState,1)
        self._update_tcp_from_franka_state()

        self._connected = True
        self.Message('Initialized',1)

    # ROS stuff
    def _init_franka_ros_interfaces(self):
        """
        Initialize all ROS publishers, subscribers, and services
        regarding the current selected ROS controller.

        Note: state subscriber and common services are initialized only when called first (in __init__)
        """

        franka_control_topic_ns = f"{self._namespace}/franka_control"
        if not self._connected:
            # Franka state
            self.state = None  # If the controller is dead, there will be no state callback
            self._last_state_callback_time = None  # Last time we got FrankaState (inside GetStateCallback() )
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

    # States
    def ResetCurrentTarget(self, do_move=True):
        # do_move syncs actual to commended in the controller as well!
        if do_move:
            _c = self._control_strategy
            if _c in ["CartesianImpedance", "JointImpedance"]:
                msg = Empty()
                # self.reset_target_svc.publish(msg)
            elif _c in ["JointImpedance"]:
                msg = Empty()
                self.joint_impedance_reset_client.publish(msg)
            elif _c in ["JointPositionTrajectory"]:
                self.Message("Target reset msg not sent - %s strategy does not support it" % _c)
            else:
                self.Message("Target reset msg not sent! Strategy %s not supported" % _c)
        sleep(0.1)
        self.GetState()
        robot.ResetCurrentTarget(self)

    def GetStateCallback(self, data):
        self.state = data
        # self.state = copy.deepcopy(data)
        self._last_state_callback_time = self.simtime()

    def GetState(self):
        """Update  Reads robot state"""

        # st = copy.deepcopy(self.state)
        _state = self.state
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

            self._last_update = self.simtime()  # Do not change !

        return _state


    # Behaviour and collisions
    def GetCollisions(self):
        _qcol = self.joint_collisions
        _xcol = self.cartesian_collisions
        return _qcol, _xcol

    def GetContacts(self):
        _qcon = self.joint_contacts
        _xcon = self.cartesian_contacts
        return _qcon, _xcon

    def GetCollisionBehaviour(self):
        return self.collision_behavior.asdict()

    def SetCollisionBehavior(self, F, tq, F_low=None, tq_low=None, restart: bool = False):
        """Set nominal collision thresholds

        Input:
          F       collision task force treshold [N,Nm]
          tq      collision joint torque treshold [Nm]
          restart if yes, stop and then start the controller
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
                self.Message("Collision behavior changed: \n\t\t F [N]: {0} \n\t\t T [Nm]: {1} \n\t\t Joint T [Nm]: {2}".format(F, tq))
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

    def SetFullCollisionBehavior(self, F, tq, F_acc=None, tq_acc=None, F_low=None, tq_low=None, F_acc_low=None, tq_acc_low=None, restart: bool = False):
        """Set nominal and acceleration collision thresholds

        % Input:
        %   F       collision task force treshold [N]
        %   T       collision task torques treshold [Nm]
        %   tq      collision joint torque treshold [Nm]
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
    def GetCartesianCompliance(self):
        return self.cartesian_compliance.Kp, self.cartesian_compliance.Kr, self.cartesian_compliance.R, self.cartesian_compliance.D

    def SetCartesianCompliance(self, Kp=None, Kr=None, R=None, D=None, hold_pose=True):
        return self.controller.SetCartesianCompliance(Kp=Kp,Kr=Kr,R=R,D=D,hold_pose=hold_pose)

    def GetCartesianStiffness(self):
        return self.cartesian_compliance.Kp, self.cartesian_compliance.Kr

    def SetCartesianStiffness(self, Kp=None, Kr=None, hold_pose=True):
        return self.SetCartesianCompliance(Kp=Kp, Kr=Kr, R=self.cartesian_compliance.R, D=self.cartesian_compliance.D, hold_pose=hold_pose)
        pass

    def GetCartesianDamping(self):
        return self.cartesian_compliance.D

    def SetCartesianDamping(self, D=None, hold_pose=True):
        return self.SetCartesianCompliance(Kp=self.cartesian_compliance.Kp, Kr=self.cartesian_compliance.Kr, R=self.cartesian_compliance.R, D=D, hold_pose=hold_pose)

    def SetCartesianSoft(self, stiffness, hold_pose=True):
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
    def GetJointCompliance(self):
        return self.joint_compliance.K, self.joint_compliance.D

    def SetJointCompliance(self, K=None, D=None, hold_pose=True):
         self.controller.SetJointCompliance(K,D,hold_pose)
    
    def GetJointStiffness(self):
        return self.joint_compliance.K

    def SetJointStiffness(self, K=None, hold_pose=True):
        self.SetJointCompliance(K=K, D=self.joint_compliance.D, hold_pose=hold_pose)

    def GetJointDamping(self):
        return self.joint_compliance.D

    def SetJointDamping(self, D=None, hold_pose=True):
        self.SetJointCompliance(K=self.joint_compliance.K, D=D, hold_pose=hold_pose)

    def SetJointSoft(self, stiffness, hold_pose=True):
        if isscalar(stiffness):
            fac = np.ones(7) * stiffness
        else:
            fac = vector(stiffness, dim=7)
        fac = np.clip(fac, 0.0, 1.0)
        self.SetJointCompliance(self._franka_default.JointCompliance.K * fac, self._franka_default.JointCompliance.D * fac, hold_pose=hold_pose)

    # Impedance parameters used for franka's internal motion generators
    def SetCartesianImpedanceFranka(self, stiffness):
        """Sets the cartesian impedance of the internal franka controller.
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
    
    def SetJointImpedanceFranka(self, stiffness, restart: bool = False):
        """Sets the joint impedance of the internal franka controller
        Args:
        ------------------
        restart: if True, then  controller will first be stopped, impedance changed, then the controller will be started."""

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
    def _update_tcp_from_franka_state(self):
        """Reads the currently set TCP in franka desk and sets it as default inside the panda_ros object"""
        frankadesk_TCP = self.state.F_T_EE
        self.TCPGripper = np.reshape(frankadesk_TCP, newshape=(4, 4), order="f")
        self.SetTCP()  # Set TCP to none, then it reads self.TCPGripper

    def SetTCP(self, *tcp, frame="Gripper", send_to_robot=False, EE_frame="Flage"):
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
        self._command.x = self.BaseToWorld(rx)
        self._command.v = self.BaseToWorld(rJ @ self._command.qdot)

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

            set_ee_request_msg.NE_T_EE = np.reshape(NE_TCP, newshape=16, order="A")
            response = self.EE_frame_proxy.call(set_ee_request_msg)
            self.Message("Response to SetTCP:", response)

            self.controller_helper.start_last_controller()

        self.GetState()
        self.Update()

    # x# ToDo
    def SetKFrame(self, EE_T_K):
        """Sets the transformation \(^{EE}T_K\) from end effector frame to stiffness frame.
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

    def GetStiffnessFrame(self):
        Tx = self.state.EE_T_K
        T = np.reshape(Tx, (4, 4))
        return T

    # x# ToDo
    def SetStiffnessFrame(self, T=None):
        """SetStiffnessFrame Sets the stiffness frame (EETK) relative to EE frame
        (controller is temporary stopped!)"""
        if T is None:
            newT = np.eye(4)
        else:
            0
        self.Message("SetStiffnessFrame UNFINISHED")

    def SetLoad(self, mass: float, COM: tuple = None, inertia: tuple = None):

        if (not isscalar(mass)) and (mass <= 0):
            raise ValueError("Mass must be scalar > 0")
        COM = vector(COM, dim=3)
        inertia = matrix(inertia, sahpe=(3, 3))

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

    def SetEEFrame(self, NE_T_EE):
        """
        Sets the transformation \(^{NE}T_{EE}\) from nominal end effector to end effector frame.
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
    def isConnected(self):
        return self._connected

    def isReady(self):
        return self._connected

    def isActive(self):
        return self._active

    def Check(self):
        self.GetState()
        _i = 0
        _err = []
        error_list = self.state.current_errors 
        #prev_error_list = self.state.last_motion_errors
        
        _attrs = dir(error_list)
    
        for attr in _attrs:
            # Attributes starting with _ are internal, for example __hash__. We dont care about them
            if attr[0] != '_':
                value = getattr(error_list, attr)
                if value is True:
                    _i += 1
                    _err += [attr]
        return _i, _err
    
    def HasError(self):
        return self.Check()[0] > 0

    def ErrorRecovery(self, enforce=False, reset_target=True):
        """Recover Panda robot from error state (for example if emergency stop was pressed

        args:
        enforce: If true, all checks will be skipped and error recovery message will be sent. However the robot state will not be reset and the robot may make a jumping move"""

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
        if self.state is None:
            error_detected = True
            controller_failed = True
        else:
            errors, err_list = self.Check()
            if errors > 0:
                error_detected = True
                self.Message(f"Recovering from errors:\n {err_list}")

            if self.state.robot_mode in [FrankaState.ROBOT_MODE_IDLE, FrankaState.ROBOT_MODE_REFLEX, FrankaState.ROBOT_MODE_OTHER]:
                error_detected = True
                self.Message(f"Attempting to change robot mode from {self.state.robot_mode}")
            if self.state.robot_mode == FrankaState.ROBOT_MODE_USER_STOPPED:
                self.Message("Error recovery is not possible: User Stop is pressed!")
                raise Exception("Error recovery not possible: User Stop is pressed!")

        if error_detected:
            if not controller_failed:
                self.GetState()
                self.ResetCurrentTarget()
            #if self._control_strategy == "CartesianImpedance":
            #    neutral_q = self.q_home
            #    self.controller.SetCartImpContNullspace(q=neutral_q, k=np.zeros(7))

            recovery_goal = ErrorRecoveryAction()
            self.error_recovery_action_client.send_goal(recovery_goal)
            sleep(0.1)

            self.error_recovery_action_client.wait_for_result(rospy.Duration(3))
            self.WarningMessage(f"Got error recovery result. Waiting for robot mode to be {FrankaState.ROBOT_MODE_MOVE} (i.e. frankaState.ROBOT_MODE_MOVE)")

            # Wait for controller to be up and running.
            if self._control_strategy is not None:
                while self.state.robot_mode not in [FrankaState.ROBOT_MODE_MOVE]:
                    sleep(self.tsamp)
                    self.GetState()
                self.WarningMessage(f"Robot mode is {FrankaState.ROBOT_MODE_MOVE}")

        self._active = False
        sleep(0.01)
        return 0

    # Movements
    def Start(self):
        if self.HasError():
            raise Exception("Robot in error mode. Can not start!")

        if self._control_strategy in ["JointImpedance"]:
            if np.min(self.joint_compliance.K / self._franka_default.JointCompliance.K) < self._franka_default.MinSoftnessForMotion:
                self.WarningMessage("Robot is to compliant and will probably not move")
        elif not self.isActive() and (self._control_strategy in ["CartesianImpedance"]):
            if min(np.min(self.cartesian_compliance.Kp / self._franka_default.CartesianCompliance.Kp), np.min(self.cartesian_compliance.Kr / self._franka_default.CartesianCompliance.Kr)) < self._franka_default.MinSoftnessForMotion:
                self.WarningMessage("Robot is to compliant and will probably not move")
            self.controller.ActivateController()
            self._active = True
        elif self._control_strategy in ["JointPositionTrajectory"]:
            # self.Message("Start() running")
            self.joint_trajectory_interface.clear()
            self.joint_trajectory_start_time = self.simtime()  # Keep this for logging how long calculations take.

        robot.Start(self)

    def Stop(self):
        self._active = False
        robot.Stop(self)

    def CheckContacts(self):
        # Checks if a contact is ocurring (which contact level is activated).
        # if contact_values are all zero, there is no contact

        for v in self.state.cartesian_contact:
            if v > 0:
                return 1
        # If we get to here, no collisions are detected
        return 0

    def GoTo_qtraj(self,qt,qdott,qddott,time,wait):
        self.joint_trajectory_interface.clear()
        self.joint_trajectory_interface.add_points(positions=qt, velocities=qdott, accelerations=qddott, time=time)
        self.joint_trajectory_interface.start()
        self.joint_trajectory_interface.wait(self.joint_trajectory_interface._time_sum+wait)
        self.Update()
        return 0

class panda_ros_reconcycle(panda_ros):
    def __init__(self):
        panda_ros.__init__(self)
        self.Base_link_name = self.Name + "_link0"

    def shutdown_hook(self):
        self.Message("{0} shutting down".format(self.Name))

    # legacy aliases
    def ResetCurrentTarget(self, send_msg=True):
        return self.ResetCurrentTarget(self,do_move=send_msg)

    def check_contact(self):
        return self.CheckContacts()


    # x# ToDo
    def on_shutdown(self):
        rospy.on_shutdown(self.shutdown_hook)

    def SetNewEEConfig(self, tool_name: str, mass: float = None, COM: list = None, trans_mat: list = None, inertia: list = None, restart: bool = True):
        """
        This function switches the end effector configuration file, allowing the user to
        use toolchangers in an intuitive way through Python using the ROSPy API

        Args
        ----
            tool_name(str): name of the configuration JSON file

            mass(float): mass of the tool combinedwith the tool-side toolchanger

            COM(list[3]): location of the center of mass in respect to the robot flange [X, Y, Z]

            trans_mat(list[16]): A flattened transformation matrix (4X4 -> 1X16) from the robot flange to the end effector

            inertia(list[9]): Inertial profile of the tool, flattened matrix (3X3 -> 1X9)

            restart(bool): Stop and start controller

        Usage
        -----
            The user needs to specify either:

                tool_name -> Grab all the parameters from the tool's configuration JSON file

                everything else -> Specify every parameter by hand when using a ne unconfigured tool

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
            
        rospy.set_param(self._namespace+"/current_tool", tool_name)

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
    #r.SetCartesianSoft(0.2)
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
