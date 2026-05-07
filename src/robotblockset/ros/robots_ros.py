"""ROS robot interface implementations.

This module defines ROS-backed robot wrappers used by RobotBlockSet robot
interfaces. It provides strategy-to-controller mapping, controller switching,
logger configuration, and joint- and Cartesian-command forwarding.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic, Leon Zlajpah.
"""

from typing import Any, Dict, List, Optional, Type

from robotblockset.robots import robot, MotionResultCodes
from robotblockset.ros.controllers_ros import joint_impedance_controller, controller_manager_proxy
from robotblockset.tools import isvector, ismatrix, check_option
from roscpp.srv import SetLoggerLevel, SetLoggerLevelRequest
from robotblockset.rbs_typing import JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, Velocity3DType, WrenchType

import rospy


class robot_ros(robot):
    """
    Base class for ROS-backed robot interfaces.

    Attributes
    ----------
    _namespace : str
        ROS namespace prefix used by the robot wrapper.
    controller_helper : controller_manager_proxy
        Helper object for loading and switching ROS controllers.
    controller : Any
        Active RobotBlockSet controller wrapper.
    """

    def __init__(self, ns: str = "", init_node: bool = True, multi_node: bool = False, control_strategy: Optional[str] = None, strategy_controller_mapping: Optional[dict] = None) -> None:
        """
        Initialize the generic ROS robot interface.

        Parameters
        ----------
        ns : str, optional
            ROS namespace of the robot.
        init_node : bool, optional
            If ``True``, initialize a ROS node for the wrapper.
        multi_node : bool, optional
            If ``True``, initialize the ROS node in anonymous mode.
        control_strategy : str, optional
            Initial RobotBlockSet control strategy.
        strategy_controller_mapping : dict[str, tuple[str, Type[Any]]] | None, optional
            Mapping from RobotBlockSet strategy names to pairs of ROS controller
            names and controller wrapper classes.

        Notes
        -----
        The constructor initializes the controller-manager helper and optionally
        activates the requested control strategy.
        """
        robot.__init__(self)

        if ns == "":
            self._namespace = ""
            self.Name = "robot_ros"
        else:
            self._namespace = "/" + ns
            self.Name = ns

        if init_node:
            self._init_ros_node(multi_node)
        else:
            if not rospy.core.is_initialized():
                self.WarningMessage("The ROS node must be initialized externally before proceeding.")

        # Mapping of RBS strategies into ros_control controller names
        self._controller_to_strategy_mapping = {v[0]: k for k, v in strategy_controller_mapping.items()}
        self._strategy_to_controller_mapping = {k: v[0] for k, v in strategy_controller_mapping.items()}
        self._strategy_to_class_mapping = {k: v[1] for k, v in strategy_controller_mapping.items()}

        # Controller helper
        self.controller_helper = controller_manager_proxy(controller_manager_node_location=self._namespace + "/controller_manager", robot_resource_name=self.Name)

        if control_strategy is not None:
            self.SetStrategy(control_strategy)
        else:
            self.controller = None
            self._control_strategy = None
            self.Message("Initializing robot object without a controller. You will only be able to read robot state.", 1)

    def _init_ros_node(self, anonymous: bool = False) -> None:
        """
        Initialize the ROS node used by the robot wrapper.

        Parameters
        ----------
        anonymous : bool, optional
            If ``True``, initialize the ROS node in anonymous mode.
        """
        try:
            rospy.init_node("FrankaHandler_{}".format(self.Name), anonymous=anonymous)
        except rospy.ROSException as e:
            self.Message("Skipping node init because of exception: {}".format(e), 0)

    # @abstractmethod
    # def save_ros_parameters(self):
    #    pass

    # @abstractmethod
    # def load_ros_parameters(self):
    #    pass

    # @abstractmethod
    # def _preload_ros_messages(self):
    #    pass

    def _get_controller_from_strategy(self, strategy: str) -> Optional[str]:
        """
        Return the ROS controller mapped to a RobotBlockSet strategy.

        Parameters
        ----------
        strategy : str
            RobotBlockSet control strategy name.

        Returns
        -------
        str | None
            ROS controller name mapped to the strategy.
        """
        return self._strategy_to_controller_mapping.get(strategy)

    def _get_strategy_from_controller(self, controller: str) -> Optional[str]:
        """
        Return the RobotBlockSet strategy mapped to a ROS controller.

        Parameters
        ----------
        controller : str
            ROS controller name.

        Returns
        -------
        str | None
            RobotBlockSet strategy mapped to the controller.
        """
        return self._controller_to_strategy_mapping.get(controller)

    def _cleanup_ros_topic_interface(self, interface_name: str) -> None:
        """
        Unregister and clear a ROS topic interface attribute when present.

        Parameters
        ----------
        interface_name : str
            Name of the attribute holding the ROS topic interface.
        """
        if hasattr(self, interface_name):
            interface = getattr(self, interface_name, None)
            if interface:
                interface.unregister()
                setattr(self, interface_name, None)

    # Strategies
    def GetStrategy(self) -> Optional[str]:
        """
        Return the currently selected RobotBlockSet control strategy.

        Returns
        -------
        str | None
            Name of the active control strategy.
        """
        return self._control_strategy
        # self.controller_helper.update_active_controller()

    def AvailableStrategies(self) -> List[str]:
        """
        List the control strategies supported by the ROS robot wrapper.

        Returns
        -------
        list[str]
            Available RobotBlockSet control strategies.
        """
        return list(self._strategy_to_controller_mapping.keys())

    def SetStrategy(self, new_strategy: str) -> Optional[bool]:
        """
        Switch the active ROS controller to match a RobotBlockSet strategy.

        Parameters
        ----------
        new_strategy : str
            Requested RobotBlockSet control strategy.

        Returns
        -------
        bool | None
            ``False`` if no switch was needed or if switching failed. Returns
            ``None`` when the switch succeeds.

        Notes
        -----
        Existing motion is stopped before switching controllers.
        """

        if new_strategy in self.AvailableStrategies():

            # First check if strategy was changed elsewhere
            active_controller = self.controller_helper.update_active_controller()
            self._control_strategy = self._get_strategy_from_controller(active_controller)

            # Check if object is initialized
            if self.isReady():
                if self._control_strategy == new_strategy:
                    self.Message(f"Not switching because already using '{new_strategy}'", 2)
                    return False

                # Stop any existing movements
                self.Stop()
                self._semaphore.release()

            # Check if controller is loaded
            ros_controller = self._get_controller_from_strategy(new_strategy)
            if ros_controller not in self.controller_helper.list_loaded_controllers():
                self.controller_helper.load_controller(ros_controller)

            # Prepare switch request
            if hasattr(self, "controller"):
                stop_controller = [] if self.controller is None else [self.controller._ros_controller_name]
            else:
                self.controller_helper.stop_active_controller()
                stop_controller = []

            start_controller = [self._get_controller_from_strategy(new_strategy)]
            resp = self.controller_helper.switch_controller(stop_controllers=stop_controller, start_controllers=start_controller)

            if resp:
                self._control_strategy = new_strategy
                self.controller = self._strategy_to_class_mapping[new_strategy](self, self._namespace)
            else:
                self.Message("Switching failed. Check ros logs!", 0)
                return False

        else:
            raise ValueError(f"Strategy '{new_strategy}' not supported")

    def SetLoggerLevel(self, level: str = "info", logger: Optional[str] = None) -> None:
        """
        Set the ROS logger verbosity level for the active controller.

        Parameters
        ----------
        level : str, optional
            Target logger level.
        logger : str, optional
            Explicit logger name. If ``None``, use the active controller logger.
        """
        if level in ["debug", "info", "warn"]:
            log_msg = SetLoggerLevelRequest()
            log_msg.logger = logger if logger is not None else self.controller._ros_logger_name
            log_msg.level = level
            self.logger_svc_proxy.call(log_msg)
            if check_option(level, "debug"):
                self.verbose = 3
            elif check_option(level, "info"):
                self.verbose = 1
        else:
            raise ValueError("unsupported ")

    def GoTo_q(self, q: JointConfigurationType, qdot: JointVelocityType, trq: JointTorqueType, wait: float, **kwargs: Any) -> int:
        """
        Command the robot in joint space through the active ROS controller.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions.
        qdot : JointVelocityType
            Desired joint velocities.
        trq : JointTorqueType
            Desired joint torques.
        wait : float
            Synchronization time after the command is sent.

        Returns
        -------
        int
            Motion result code.
        """
        self.Message(f"position: {q}, vel: {qdot}, trq: {trq}", 4)
        self.controller.GoTo_q(q, qdot, trq)

        self._synchro_control(wait)
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = J @ qdot
        self.Update()
        self._last_control_time = self.simtime()
        return MotionResultCodes.MOTION_SUCCESS.value

    def GoTo_X(self, x: Pose3DType, xdot: Velocity3DType, trq: WrenchType, wait: float, do_not_publish_msg: bool = False, **kwargs: Any) -> int:
        """
        Command the robot in Cartesian space through the active ROS controller.

        Parameters
        ----------
        x : Pose3DType
            Desired Cartesian pose.
        xdot : Velocity3DType
            Desired Cartesian twist.
        trq : WrenchType
            Desired end-effector wrench.
        wait : float
            Synchronization time after the command is sent.
        do_not_publish_msg : bool, optional
            Legacy flag kept for compatibility.
        **kwargs : Any
            Additional controller-specific keyword arguments.

        Returns
        -------
        int
            Motion result code.
        """

        if not isvector(x, dim=7):
            raise Exception("%s: GoTo_x: NAN x value" % self.Name)
        if not isvector(xdot, dim=6):
            raise Exception("%s: GoTo_x: NAN xdot value" % self.Name)
        if not isvector(trq, dim=6):
            raise Exception("%s: GoTo_x: NAN trq value" % self.Name)

        if do_not_publish_msg:
            raise Exception("Not supported anymore. Use GoTo_Xtraj instead.")

        self.Message(f"pose: {x}, vel: {xdot}, trq: {trq}", 4)
        self.controller.GoTo_X(x, xdot, trq, wait, **kwargs)
        self._command.x = x
        self._command.v = xdot
        self._command.FT = trq
        self.Update()
        self._synchro_control(wait)
        self._last_control_time = self.simtime()
        return 0
