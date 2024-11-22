from robotblockset.robots import robot
from robotblockset.ros.controllers_ros import joint_impedance_controller, controller_manager_proxy
from robotblockset.tools import isvector, ismatrix, check_option
from roscpp.srv import SetLoggerLevel, SetLoggerLevelRequest

import rospy

class robot_ros(robot):
    def __init__(self, ns="", init_node=True, multi_node=False, control_strategy=None, strategy_controller_mapping = None):
        robot.__init__(self)

        if ns == "":
            self._namespace = "/"
            self.Name="robot_ros"
        else:
            self._namespace = "/" + ns
            self.Name=ns

        if init_node:
            self._init_ros_node(multi_node)
        else:
            print("Make sure that ROS node is initialized outside")

        # Mapping of RBS strategies into ros_control controller names
        self._controller_to_strategy_mapping = {v[0]: k for k, v in strategy_controller_mapping.items()}
        self._strategy_to_controller_mapping = {k: v[0] for k, v in strategy_controller_mapping.items()} 
        self._strategy_to_class_mapping = {k: v[1] for k, v in strategy_controller_mapping.items()}

        # Controller helper
        self.controller_helper = controller_manager_proxy(controller_manager_node_location=self._namespace+"/controller_manager", robot_resource_name=self.Name)
          
        if control_strategy is not None:
            self.SetStrategy(control_strategy)
        else:
            self.controller = None
            self._control_strategy = None
            self.Message("Initializing robot object without a controller. You will only be able to read robot state.")


    def _init_ros_node(self, anonymous=False):
        try:
            rospy.init_node("FrankaHandler_{}".format(self.Name), anonymous=anonymous)
        except rospy.ROSException as e:
            self.Message("Skipping node init because of exception: {}".format(e))

    #@abstractmethod
    #def save_ros_parameters(self):
    #    pass

    #@abstractmethod
    #def load_ros_parameters(self):
    #    pass

    #@abstractmethod
    #def _preload_ros_messages(self):
    #    pass

    def _get_controller_from_strategy(self, strategy):
        return self._strategy_to_controller_mapping.get(strategy)

    def _get_strategy_from_controller(self, controller):
        return self._controller_to_strategy_mapping.get(controller)
    
    def _cleanup_ros_topic_interface(self, interface_name):
        if hasattr(self, interface_name):
            interface = getattr(self, interface_name, None)
            if interface:
                interface.unregister()
                setattr(self, interface_name, None)

    # Strategies
    def GetStrategy(self):
        return self._control_strategy
        #self.controller_helper.update_active_controller()

    def AvailableStrategies(self):
        return list(self._strategy_to_controller_mapping.keys())

    def SetStrategy(self, new_strategy):

        if new_strategy in self.AvailableStrategies():
            
            # First check if strategy was changed elsewhere
            active_controller = self.controller_helper.update_active_controller()
            self._control_strategy = self._get_strategy_from_controller(active_controller)

            # Check if object is initialized
            if self.isReady():
                if self._control_strategy == new_strategy:
                    self.Message(f"Not switching because already using '{new_strategy}'")
                    return False
            
                # Stop any existing movements
                self.Stop()
                self._semaphore.release()
                
            # Check if controller is loaded
            ros_controller = self._get_controller_from_strategy(new_strategy)
            if ros_controller not in self.controller_helper.list_loaded_controllers():
                self.controller_helper.load_controller(ros_controller)

            # Prepare switch request
            if hasattr(self,"controller"):
                stop_controller = [] if self.controller is None else [self.controller._ros_controller_name]
            else:
                self.controller_helper.stop_active_controller()
                stop_controller = []

            start_controller = [self._get_controller_from_strategy(new_strategy)]
            resp = self.controller_helper.switch_controller(stop_controllers=stop_controller,start_controllers=start_controller)

            if resp:
                self._control_strategy = new_strategy
                self.controller = self._strategy_to_class_mapping[new_strategy](self,self._namespace)
            else:
                self.Message("Switching failed. Check ros logs!")
                return False

        else:
            raise ValueError(f"Strategy '{new_strategy}' not supported")

    def SetLoggerLevel(self, level="info", logger=None):
        if (level in ["debug", "info", "warn"]):
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
        

    def GoTo_q(self, q, qdot, trq, wait):
        self.controller.GoTo_q(q,qdot,trq)
        
        self._sinhro_control(wait)
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = J @ qdot
        self.Update()
        self._last_control_time = self.simtime()

    def GoTo_X(self, x, xdot, trq, wait, do_not_publish_msg=False, **kwargs):
        """GoTo_X Update task position and orientation for task controller and wait for time t
        Inputs:
        x        task position,quaternion (1 x 7) (in robot CS)
        xdot   task twist (6 x 1) (in robot CS)
        trq      EE wrench (6 x 1) (in robot CS)
        wait    wait time after update

        """

        if not isvector(x, dim=7):
            raise Exception("%s: GoTo_x: NAN x value" % self.Name)
        if not isvector(xdot, dim=6):
            raise Exception("%s: GoTo_x: NAN xdot value" % self.Name)
        if not isvector(trq, dim=6):
            raise Exception("%s: GoTo_x: NAN trq value" % self.Name)

        if do_not_publish_msg:
            raise Exception("Not supported anymore. Use GoTo_Xtraj instead.")

        self.controller.GoTo_X(x,xdot,trq,wait,**kwargs)
        self._command.x = x 
        self._command.v = xdot
        self._command.FT = trq
        self.Update()
        #while (self.simtime() - self.last_control_time) < wait:
        #    0
        self._last_control_time = self.simtime()
        return 0
