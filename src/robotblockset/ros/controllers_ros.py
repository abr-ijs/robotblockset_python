
"""ROS controller interface implementations.

This module defines ROS-backed controller wrappers used by RobotBlockSet robot
interfaces. It provides helpers for controller-manager interaction, joint-space
impedance commands, and Cartesian impedance commands through ROS topics,
actions, and service proxies.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic.
"""

from typing import Any, List, Optional, Tuple

from robotblockset.controllers import joint_controller_type, cartesian_controller_type, compliant_controller_type
from robotblockset.tools import rbs_object, vector, isvector, ismatrix, isscalar, matrix
from abc import abstractmethod
from robot_module_msgs.msg import ImpedanceParameters, CartesianCommand, JointCommand, JointTrapVelAction, JointTrapVelGoal
from std_msgs.msg import Empty, Float32MultiArray, Bool
import numpy as np
import rospy
import actionlib
import controller_manager_msgs.srv as cm_srv

from roscpp.srv import SetLoggerLevel, SetLoggerLevelRequest
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, RotationMatrixType, Vector3DType, Velocity3DType, WrenchType


class ros_controller_interface(rbs_object):
    """
    Base class for ROS-backed controller interfaces.

    Attributes
    ----------
    _ros_controller_name : str
        Name of the ROS controller resource managed by the interface.
    _ros_logger_name : str
        Name of the ROS logger associated with the controller.
    """

    def __init__(self, ros_controller_name: str, ros_logger_name: str) -> None:
        """
        Initialize the ROS controller interface base.

        Parameters
        ----------
        ros_controller_name : str
            Name of the ROS controller resource.
        ros_logger_name : str
            Name of the ROS logger associated with the controller.

        Returns
        -------
        None
            This constructor initializes the ROS controller interface in place.
        """
        self._ros_controller_name = ros_controller_name  # name in ros_control json
        self._ros_logger_name = ros_logger_name
        self._register_interface()
        self._preload_ros_messages()

    @abstractmethod
    def _register_interface(self) -> None:
        """
        Register ROS communication entities for the controller interface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _preload_ros_messages(self) -> None:
        """
        Pre-create ROS message instances reused by the controller.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def __del__(self) -> None:
        """
        Release ROS resources owned by the controller interface.

        Returns
        -------
        None
        """
        pass


class controller_manager_proxy(object):
    """
    Proxy around ROS controller-manager services.

    Attributes
    ----------
    robot_resource_name : str
        Resource name used to identify controllers that belong to the robot.
    controller_manager_node_location : str
        ROS namespace of the controller-manager node.
    current_controller : str
        Name of the currently active controller, if known.
    _last_controller : str
        Name of the last active controller remembered for restart.
    """

    def __init__(self, controller_manager_node_location: str = "controller_manager", robot_resource_name: str = "panda") -> None:
        """
        Create a proxy for ROS controller-manager services.

        Parameters
        ----------
        controller_manager_node_location : str, optional
            ROS namespace of the controller manager.
        robot_resource_name : str, optional
            Resource name used to identify robot-specific controllers.
        """
        self.robot_resource_name = robot_resource_name
        self.controller_manager_node_location = controller_manager_node_location
        self.current_controller = ""
        self._last_controller = ""
        self.reconnect()
        self.update_active_controller()

    def reconnect(self) -> None:
        """
        Reconnect all controller-manager service proxies.

        Returns
        -------
        None
        """
        rospy.wait_for_service(self.controller_manager_node_location + "/switch_controller", 0.1)
        rospy.wait_for_service(self.controller_manager_node_location + "/list_controllers", 0.1)
        rospy.wait_for_service(self.controller_manager_node_location + "/load_controller", 0.1)
        try:
            # Create ROS service proxies for the controller_manager ROS nodes
            self.switcher_proxy = rospy.ServiceProxy(self.controller_manager_node_location + "/switch_controller", cm_srv.SwitchController)
            self.lister_proxy = rospy.ServiceProxy(self.controller_manager_node_location + "/list_controllers", cm_srv.ListControllers)
            self.loader_proxy = rospy.ServiceProxy(self.controller_manager_node_location + "/load_controller", cm_srv.LoadController)
            self.unloader_proxy = rospy.ServiceProxy(self.controller_manager_node_location + "/unload_controller", cm_srv.UnloadController)
        except Exception as e:
            print(e)

    def update_active_controller(self) -> str:
        """
        Query the active controller that claims the configured robot resource.

        Returns
        -------
        str
            Name of the active controller, or an empty string if none was found.
        """

        controllers = []
        found = False

        try:
            # Get the list of controllers from the controller_manager ROS node
            controllers = self.lister_proxy().controller
        except Exception as e:
            # Log an error if the switch_controller service fails
            rospy.logerr("ControllerHelper list_controllers error: {}".format(e))

        # Iterate through the list of controllers
        for controller in controllers:
            # Check if the controller is running
            if controller.state != "running":
                continue

            # Get the list of resources claimed by the controller
            resources = [item for claimed in controller.claimed_resources for item in claimed.resources]

            # Check if the robot resource is claimed by the controller
            if len(list(resource for resource in resources if resource.startswith(self.robot_resource_name))):
                self.current_controller = controller.name
                found = True
                break

        if not found:
            # rospy.logwarn('No controller is currently running.')
            self.current_controller = ""

        # also retrun
        return self.current_controller

    def list_loaded_controllers(self) -> List[str]:
        """
        List names of controllers known to the controller manager.

        Returns
        -------
        list[str]
            Names of all controllers reported by the controller manager.
        """
        controllers = self.lister_proxy().controller
        controller_names = [controller.name for controller in controllers]
        return controller_names

    def switch_controller(self, start_controllers: List[str], stop_controllers: List[str], strictness: int = 1) -> bool:
        """
        Switch the active controller set.

        Parameters
        ----------
        start_controllers : list[str]
            Names of controllers to start.
        stop_controllers : list[str]
            Names of controllers to stop.
        strictness : int, optional
            ROS controller-manager strictness level. ``1`` means best effort and
            ``2`` means strict.

        Returns
        -------
        bool
            ``True`` if the switch request succeeded, otherwise ``False``.
        """

        # Check if the start_controllers and stop_controllers are lists
        assert isinstance(start_controllers, list)
        assert isinstance(stop_controllers, list)
        # Check if the strictness is either 1 or 2
        assert strictness in [1, 2]
        try:
            # Call the switch_controller service
            response = self.switcher_proxy(start_controllers=start_controllers, stop_controllers=stop_controllers, strictness=strictness)
            # Return the response
            return response.ok
        except Exception as e:
            # Log an error if the switch_controller service fails
            rospy.logerr("ControllerHelper switch_controller error: {}".format(e))
            # Return False
            return False

    def load_controller(self, controller_name: str) -> bool:
        """
        Load a controller through the controller manager.

        Parameters
        ----------
        controller_name : str
            Name of the controller to load.

        Returns
        -------
        bool
            ``True`` if the controller was loaded successfully, otherwise ``False``.
        """
        try:
            # Call the load_controller service
            response = self.loader_proxy(name=controller_name)
            # Return the response
            return response.ok
        except Exception as e:
            # Log an error if the load_controller service fails
            rospy.logerr("ControllerHelper load_controller error: {}".format(e))
            import traceback

            traceback.print_exc()
            # Return False
            return False

    def unload_controller(self, controller_name: str) -> bool:
        """
        Unload a controller through the controller manager.

        Parameters
        ----------
        controller_name : str
            Name of the controller to unload.

        Returns
        -------
        bool
            ``True`` if the controller was unloaded successfully, otherwise ``False``.
        """
        try:
            # Call the unload_controller service
            response = self.unloader_proxy(name=controller_name)
            # Return the response
            return response.ok
        except Exception as e:
            # Log an error if the unload_controller service fails
            rospy.logerr("ControllerHelper unload_controller error: {}".format(e))
            # Return False
            return False

    def stop_active_controller(self) -> bool:
        """
        Stop the currently active controller and remember it for restart.

        Returns
        -------
        bool
            ``True`` if the stop request succeeded, otherwise ``False``.
        """
        self._last_controller = self.update_active_controller()
        return self.switch_controller(start_controllers=[""], stop_controllers=[self.current_controller], strictness=1)

    def start_last_controller(self) -> bool:
        """
        Restart the last controller stopped by `stop_active_controller`.

        Returns
        -------
        bool
            ``True`` if the start request succeeded, otherwise ``False``.
        """
        resp = self.switch_controller(start_controllers=[self._last_controller], stop_controllers=[""], strictness=1)
        self._last_controller = ""
        return resp

    def stop_controller(self, controller: str) -> bool:
        """
        Stop the named ROS controller.

        Parameters
        ----------
        controller : str
            Name of the controller to stop.

        Returns
        -------
        bool
            ``True`` if the stop request succeeded, otherwise ``False``.
        """
        return self.switch_controller(start_controllers=[""], stop_controllers=[controller], strictness=1)

    def start_controller(self, controller: str) -> None:
        """
        Start the named ROS controller.

        Parameters
        ----------
        controller : str
            Name of the controller to start.

        Returns
        -------
        None
        """
        res = self.switch_controller(start_controllers=[controller], stop_controllers=[""], strictness=1)
        if res:
            self.current_controller = controller


class joint_impedance_controller(ros_controller_interface, joint_controller_type):
    """
    ROS joint-impedance controller wrapper.

    Attributes
    ----------
    _ns : str
        Fully qualified ROS namespace for the controller topics.
    _robot : Any
        Robot instance associated with the controller.
    joint_command_publisher : rospy.Publisher
        Publisher used to send joint command messages.
    joint_action_client : actionlib.SimpleActionClient
        Action client used for trapezoidal-velocity joint moves.
    joint_impedance_reset_client : rospy.Publisher
        Publisher used to reset the controller target.
    _joint_command_msg : JointCommand
        Reusable command message instance.
    """

    def __init__(self, robot: Optional[Any] = None, namespace: Optional[str] = None, ros_controller_name: str = "joint_impedance_controller", ros_logger_name: str = "ros.ijs_controllers") -> None:
        """
        Initialize the ROS joint-impedance controller wrapper.

        Parameters
        ----------
        robot : Any, optional
            Robot instance associated with the controller.
        namespace : str, optional
            Namespace prefix used to resolve controller topics.
        ros_controller_name : str, optional
            Name of the ROS controller resource.
        ros_logger_name : str, optional
            Name of the ROS logger associated with the controller.
        """
        self._ns = f"{namespace}/{ros_controller_name}"
        self._robot = robot
        joint_controller_type.__init__(self)
        ros_controller_interface.__init__(self, ros_controller_name=ros_controller_name, ros_logger_name=ros_logger_name)

    def _register_interface(self) -> None:
        """
        Create ROS publishers and action clients for joint control.

        Returns
        -------
        None
        """
        self.joint_command_publisher = rospy.Publisher(f"{self._ns}/command", JointCommand, queue_size=1)
        self.joint_action_client = actionlib.SimpleActionClient(f"{self._ns}/move_joint_trap", JointTrapVelAction)
        self.joint_impedance_reset_client = rospy.Publisher(f"{self._ns}/reset_target", Empty, queue_size=1)

    def _preload_ros_messages(self) -> None:
        """
        Create reusable joint-command ROS messages.

        Returns
        -------
        None
        """
        self._joint_command_msg = JointCommand()

    def __del__(self) -> None:
        """
        Unregister ROS entities owned by the joint controller wrapper.

        Returns
        -------
        None
        """
        self.Message(f"Shutting down controller interface: {self._ros_controller_name}")
        self.joint_command_publisher.unregister()
        self.joint_impedance_reset_client.unregister()
        self.joint_action_client.cancel_all_goals()

    def GoTo_q(self, q: JointConfigurationType, qdot: JointVelocityType, trq: JointTorqueType, **kwargs: Any) -> None:
        """
        Publish a joint-space command to the impedance controller.

        Parameters
        ----------
        q : JointConfigurationType
            Joint-position target.
        qdot : JointVelocityType
            Joint-velocity target.
        trq : JointTorqueType
            Joint-torque feedforward command.

        Returns
        -------
        None
        """

        q = vector(q, dim=7)
        qdot = vector(qdot, dim=7)
        trq = vector(trq, dim=7)

        cmd_msg = self._joint_command_msg
        cmd_msg.pos = q
        cmd_msg.vel = qdot
        cmd_msg.trq = trq
        cmd_msg.impedance.n = self._robot.nj
        cmd_msg.impedance.k = self._robot.joint_compliance.K
        cmd_msg.impedance.d = self._robot.joint_compliance.D
        self.joint_command_publisher.publish(cmd_msg)

    def SetJointCompliance(self, K: Optional[ArrayLike] = None, D: Optional[ArrayLike] = None, hold_pose: bool = True) -> None:
        """
        Update the commanded joint impedance parameters.

        Parameters
        ----------
        K : ArrayLike, optional
            Desired joint stiffness values. Scalars are broadcast to all joints.
        D : ArrayLike, optional
            Desired joint damping values. Scalars are broadcast to all joints.
        hold_pose : bool, optional
            If ``True``, reset the current target and apply the new compliance at
            the current commanded pose.

        Returns
        -------
        None

        Notes
        -----
        When ``hold_pose`` is ``False``, the stiffness is ramped in steps to avoid
        abrupt changes in commanded impedance.
        """
        if K is None:
            K = self._robot._franka_default.JointCompliance.K
        elif isscalar(K):
            K = np.ones(7) * K
        else:
            K = vector(K, dim=7)
        if D is None:
            D = self._robot._franka_default.JointCompliance.D
        elif isscalar(D):
            D = np.ones(7) * D
        else:
            D = vector(D, dim=7)

        cmd_msg = self._joint_command_msg
        if hold_pose:
            self.ResetCurrentTarget()
            cmd_msg.pos = self._robot._command.q
            cmd_msg.vel = np.zeros(7)
            cmd_msg.trq = np.zeros(7)
            cmd_msg.impedance.n = self._robot.nj
            cmd_msg.impedance.k = K
            cmd_msg.impedance.d = D
            self.joint_command_publisher.publish(cmd_msg)
        else:
            _KK = np.copy(self._robot.joint_compliance.K)
            _KK[(_KK < 0.001).nonzero()] = 0.001
            while np.amax(np.log2(K / _KK)) > 0.3:
                _KK = 2 ** np.clip(np.log2(K * 1.2 / _KK), 0.1, np.inf) * _KK
                cmd_msg.pos = self._robot._command.q
                cmd_msg.vel = np.zeros(7)
                cmd_msg.trq = np.zeros(7)
                cmd_msg.impedance.n = self._robot.nj
                cmd_msg.impedance.k = _KK
                cmd_msg.impedance.d = D
                self.joint_command_publisher.publish(cmd_msg)
                self.Wait(self.tsamp)
                if not np.all(K == _KK):
                    cmd_msg.pos = self._command.q
                    cmd_msg.vel = np.zeros(7)
                    cmd_msg.trq = np.zeros(7)
                    cmd_msg.impedance.n = self._robot.nj
                    cmd_msg.impedance.k = K
                    cmd_msg.impedance.d = D
                    self.joint_command_publisher.publish(cmd_msg)
        self._robot.joint_compliance.K = K
        self._robot.joint_compliance.D = D
        self.Message(f"Joint compliance changed \nStiff:{K}\nDamp:{D}")

    def ResetCurrentTarget(self) -> None:
        """
        Reset the current target of the joint-impedance controller.

        Returns
        -------
        None
        """
        msg = Empty()
        self.joint_impedance_reset_client.publish(msg)


# cartesian impedance controller
class cartesian_impedance_controller(ros_controller_interface, cartesian_controller_type):
    """
    ROS Cartesian-impedance controller wrapper.

    Attributes
    ----------
    _ns : str
        Fully qualified ROS namespace for the controller topics.
    _robot : Any
        Robot instance associated with the controller.
    cart_activate_publisher : rospy.Publisher
        Publisher used to enable the Cartesian controller.
    cartesian_command_publisher : rospy.Publisher
        Publisher used to send Cartesian command messages.
    cart_null_q_publisher : rospy.Publisher
        Publisher used to send null-space joint targets.
    cart_null_k_publisher : rospy.Publisher
        Publisher used to send null-space stiffness values.
    reset_target_svc : rospy.Publisher
        Publisher used to reset the controller target.
    cart_stiff_publisher : rospy.Publisher
        Publisher used to send stiffness-only updates.
    _cartesian_command_msg : CartesianCommand
        Reusable Cartesian command message instance.
    _impedance_parameters_msg : ImpedanceParameters
        Reusable impedance message instance.
    """

    def __init__(self, robot: Optional[Any] = None, namespace: Optional[str] = None, ros_controller_name: str = "cartesian_impedance_controller", ros_logger_name: str = "ros.ijs_controllers") -> None:
        """
        Initialize the ROS Cartesian-impedance controller wrapper.

        Parameters
        ----------
        robot : Any, optional
            Robot instance associated with the controller.
        namespace : str, optional
            Namespace prefix used to resolve controller topics.
        ros_controller_name : str, optional
            Name of the ROS controller resource.
        ros_logger_name : str, optional
            Name of the ROS logger associated with the controller.
        """
        self._ns = f"{namespace}/{ros_controller_name}"
        self._robot = robot
        self._robot._active = False
        joint_controller_type.__init__(self)
        ros_controller_interface.__init__(self, ros_controller_name=ros_controller_name, ros_logger_name=ros_logger_name)

    def _register_interface(self) -> None:
        """
        Create ROS publishers used by the Cartesian impedance controller.

        Returns
        -------
        None
        """
        self.cart_activate_publisher = rospy.Publisher(f"{self._ns}/activate", Bool, queue_size=1)
        self.cartesian_command_publisher = rospy.Publisher(f"{self._ns}/command", CartesianCommand, queue_size=1)
        self.cart_null_q_publisher = rospy.Publisher(f"{self._ns}/nullspace_q", Float32MultiArray, queue_size=1, latch=False)
        self.cart_null_k_publisher = rospy.Publisher(f"{self._ns}/nullspace_stiff", Float32MultiArray, queue_size=1, latch=False)
        self.reset_target_svc = rospy.Publisher(f"{self._ns}/reset_target", Empty, queue_size=1)
        self.cart_stiff_publisher = rospy.Publisher(f"{self._ns}/stiffness", ImpedanceParameters, queue_size=1)

    def _preload_ros_messages(self) -> None:
        """
        Create reusable Cartesian-command ROS messages.

        Returns
        -------
        None
        """
        self._cartesian_command_msg = CartesianCommand()
        self._impedance_parameters_msg = ImpedanceParameters()

    def __del__(self) -> None:
        """
        Unregister ROS entities owned by the Cartesian controller wrapper.

        Returns
        -------
        None
        """
        self.Message(f"Shutting down controller interface: {self._ros_controller_name}")
        self.cart_activate_publisher.unregister()
        self.cartesian_command_publisher.unregister()
        self.cart_null_q_publisher.unregister()
        self.cart_null_k_publisher.unregister()
        self.reset_target_svc.unregister()
        self.cart_stiff_publisher.unregister()

    def ActivateController(self) -> None:
        """
        Activate the Cartesian controller.

        Returns
        -------
        None
        """
        msg = Bool()
        msg.data = True
        self.cart_activate_publisher.publish(msg)

    def GoTo_X(self, x: Pose3DType, xdot: Velocity3DType, trq: WrenchType, wait: float, **kwargs: Any) -> None:
        """
        Publish a Cartesian command to the impedance controller.

        Parameters
        ----------
        x : Pose3DType
            Cartesian pose target expressed as position and quaternion.
        xdot : Velocity3DType
            Cartesian twist command.
        trq : WrenchType
            End-effector wrench command.
        wait : float
            Wait time associated with the command.
        **kwargs : Any
            Optional Cartesian compliance overrides:
            ``Kp``, ``Kr``, ``R``, and ``D``.

        Returns
        -------
        None
        """

        kwargs.setdefault("R", self._robot.cartesian_compliance.R)
        kwargs.setdefault("D", self._robot.cartesian_compliance.D)
        kwargs.setdefault("Kp", self._robot.cartesian_compliance.Kp)
        kwargs.setdefault("Kr", self._robot.cartesian_compliance.Kr)

        R = matrix(kwargs["R"], shape=(3, 3))
        D = kwargs["D"]
        Kp = vector(kwargs["Kp"], dim=3)
        Kr = vector(kwargs["Kr"], dim=3)

        if not (D >= 0 and D <= 2):
            raise Exception("%s: GoTo_x: D out of bounds" % self.Name)

        cmd_msg = self._cartesian_command_msg
        cmd_msg.pose.position.x = x[0]
        cmd_msg.pose.position.y = x[1]
        cmd_msg.pose.position.z = x[2]

        cmd_msg.pose.orientation.w = x[3]
        cmd_msg.pose.orientation.x = x[4]
        cmd_msg.pose.orientation.y = x[5]
        cmd_msg.pose.orientation.z = x[6]

        # TODO: moznost izklopa, npr. robot.User.SEND_VELOCITY = 1
        cmd_msg.vel.linear.x = xdot[0]
        cmd_msg.vel.linear.y = xdot[1]
        cmd_msg.vel.linear.z = xdot[2]
        cmd_msg.vel.angular.x = xdot[3]
        cmd_msg.vel.angular.y = xdot[4]
        cmd_msg.vel.angular.z = xdot[5]

        cmd_msg.ft.force.x = trq[0]
        cmd_msg.ft.force.y = trq[1]
        cmd_msg.ft.force.z = trq[2]

        cmd_msg.ft.torque.x = trq[3]
        cmd_msg.ft.torque.y = trq[4]
        cmd_msg.ft.torque.z = trq[5]

        # Calculate stiffness matrix
        trM = np.diag(Kp)
        rotM = np.diag(Kr)
        # Rotate
        trK = R * trM * np.transpose(R)
        rotK = rotM
        # Damping
        trD = R * 2 * np.sqrt(trM) * np.transpose(R)
        rotD = D * np.sqrt(rotM)

        # Check if any is NaN
        if np.isnan(trM).any() or np.isnan(rotM).any() or np.isnan(trK).any() or np.isnan(rotK).any() or np.isnan(trD).any() or np.isnan(rotD).any():
            raise Exception("%s: GoTo_x: trM or rotM or trK or rotK or trD or rotD - NaN error" % self.Name)

        stiffness = self._impedance_parameters_msg
        stiffness.n = 9
        stiffness.k = np.concatenate([np.reshape(trK, (9, 1)), np.reshape(rotK, (9, 1))])
        # self.Message("{0}".format(stiffness.k))
        stiffness.d = np.concatenate([np.reshape(trD, (9, 1)), np.reshape(rotD, (9, 1))])

        cmd_msg.impedance = stiffness

        # Update values to reflect current values of stiffness and positions.
        self._robot.cartesian_compliance.R = R
        self._robot.cartesian_compliance.D = D
        self._robot.cartesian_compliance.Kp = Kp
        self._robot.cartesian_compliance.Kr = Kr
        self.cartesian_command_publisher.publish(cmd_msg)

    # Utils
    # x# ToDo
    def SetCartImpContNullspace(self, q: JointConfigurationType, k: JointConfigurationType) -> None:
        """
        Set null-space target and stiffness for the Cartesian controller.

        Parameters
        ----------
        q : JointConfigurationType
            Null-space joint target.
        k : JointConfigurationType
            Null-space stiffness values.

        Returns
        -------
        None
        """
        if type(q) in [list, tuple]:
            q = np.array(q)
        if type(k) in [list, tuple]:
            k = np.array(k)

        assert q.shape[0] == 7
        assert k.shape[0] == 7

        if self._control_strategy == "CartesianImpedance":
            q_msg = Float32MultiArray()
            k_msg = Float32MultiArray()

            q_msg.data = q
            k_msg.data = k

            self.cart_null_q_publisher.publish(q_msg)
            self.cart_null_k_publisher.publish(k_msg)

            self.cart_null_q = q
            self.cart_null_k = k
            self.Message("Cart imp nullspace is set. {}\n {}".format(q, k))

        else:
            self.WarningMessage("SetCartesianNullspace: Strategy not supported. {}".format(self.Name))

    # x# ToDo
    def GetCartImpContNullspace(self) -> Tuple[JointConfigurationType, JointConfigurationType]:
        """
        Return the Cartesian-controller null-space target and stiffness.

        Returns
        -------
        tuple[JointConfigurationType, JointConfigurationType]
            Tuple containing the stored null-space target and stiffness vectors.
        """
        return self.cart_null_q, self.cart_null_k

    def SetCartesianCompliance(self, Kp: Optional[ArrayLike] = None, Kr: Optional[ArrayLike] = None, R: Optional[RotationMatrixType] = None, D: Optional[float] = None, hold_pose: bool = True) -> int:
        """
        Update the commanded Cartesian impedance parameters.

        Parameters
        ----------
        Kp : ArrayLike, optional
            Translational stiffness values. Scalars are broadcast to three axes.
        Kr : ArrayLike, optional
            Rotational stiffness values. Scalars are broadcast to three axes.
        R : RotationMatrixType, optional
            Rotation matrix used to orient translational stiffness.
        D : float, optional
            Damping factor in the interval ``[0, 2]``.
        hold_pose : bool, optional
            If ``True``, update compliance while holding the current target pose.

        Returns
        -------
        int
            ``0`` when the compliance update was accepted.

        Notes
        -----
        The method updates both the outgoing ROS impedance message and the cached
        compliance values stored on the attached robot object.
        """
        if Kp is None:
            Kp = self._franka_default.CartesianCompliance.Kp
        elif isscalar(Kp):
            Kp = np.ones(3) * Kp
        else:
            Kp = vector(Kp, dim=3)
        if Kr is None:
            Kr = self._franka_default.CartesianCompliance.Kr
        elif isscalar(Kr):
            Kr = np.ones(3) * Kr
        else:
            Kr = vector(Kr, dim=3)
        if R is None:
            R = self._franka_default.CartesianCompliance.R
        elif not ismatrix(R, shape=(3, 3)):
            raise ValueError("Rotational matrix 'R' is not 3 x 3")
        if D is None:
            D = self._franka_default.CartesianCompliance.D
        elif not isscalar(D):
            raise ValueError("Damping 'D' is not scalar")
        elif (D < 0) or (D > 2):
            raise ValueError("Damping 'D' must be in range [0,2]")

        # Calculate stiff matrix
        trM = np.diag(Kp)
        rotM = np.diag(Kr)

        # Rotate
        trK = R * trM * np.transpose(R)
        rotM = np.diag(Kr)

        # Damping
        trD = R * 2 * np.sqrt(trM) * np.transpose(R)
        rotD = D * np.sqrt(rotM)

        # Check for NaN
        if np.isnan(trD).any() or np.isnan(rotD).any() or np.isnan(trK).any() or np.isnan(rotM).any():
            raise ValueError("NaNs present in compliance parameters")

        stiffness = self._impedance_parameters_msg
        stiffness.n = 9
        stiffness.k = np.concatenate((np.reshape(trK, (9, 1)), np.reshape(rotM, (9, 1))))
        stiffness.d = np.concatenate((np.reshape(trD, (9, 1)), np.reshape(rotD, (9, 1))))

        if not hold_pose:
            self.cart_stiff_publisher.publish(stiffness)
        else:
            cmd_msg = self._cartesian_command_msg
            cmd_msg.impedance = stiffness

            self.ResetCurrentTarget()
            cmd_msg.pose.position.x = self._robot._command.x[0]
            cmd_msg.pose.position.y = self._robot._command.x[1]
            cmd_msg.pose.position.z = self._robot._command.x[2]
            cmd_msg.pose.orientation.w = self._robot._command.x[3]
            cmd_msg.pose.orientation.x = self._robot._command.x[4]
            cmd_msg.pose.orientation.y = self._robot._command.x[5]
            cmd_msg.pose.orientation.z = self._robot._command.x[6]
            cmd_msg.time_from_start = rospy.Duration(0, 0)

            # Send command message
            self.cartesian_command_publisher.publish(cmd_msg)

        # Handle (update) internal cartesian impedance values
        self._robot.cartesian_compliance.Kp = Kp
        self._robot.cartesian_compliance.Kr = Kr
        self._robot.cartesian_compliance.R = R
        self._robot.cartesian_compliance.D = D
        return 0

    def ResetCurrentTarget(self) -> None:
        """
        Reset the current target of the Cartesian-impedance controller.

        Returns
        -------
        None
        """
        msg = Empty()
        self.reset_target_svc.publish(msg)
