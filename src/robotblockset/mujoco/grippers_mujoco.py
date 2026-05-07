"""MuJoCo Grippers Module.

This module provides gripper implementations for the MuJoCo `simmujoco` simulator.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from time import sleep
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from robotblockset.grippers import gripper
from robotblockset.mujoco.mujoco_api import mjInterface

if TYPE_CHECKING:
    from robotblockset.robots import robot


class gripper_mujoco(gripper):
    """MuJoCo-backed gripper interface using the `mjInterface` API."""

    def __init__(self, name: str, actuator_name: str, width_max: float, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, ctrl_sign: float = 1.0, **kwargs: Any) -> None:
        """
        Initialize a MuJoCo gripper interface.

        Parameters
        ----------
        name : str
            Gripper instance name.
        actuator_name : str
            MuJoCo actuator name used to command the gripper.
        width_max : float
            Maximum allowed opening width.
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        port : int, optional
            Port number for the MuJoCo simulator connection.
        ctrl_sign : float, optional
            Sign applied to actuator control commands.
        **kwargs : Any
            Additional keyword arguments passed to `gripper.__init__`.

        Raises
        ------
        Exception
            If the configured actuator name does not resolve in the MJCF model.

        Returns
        -------
        None
            This constructor initializes the MuJoCo gripper object in place.
        """
        gripper.__init__(self, **kwargs)
        self.Name = name
        self._ctrl_sign = ctrl_sign

        if scene is None:
            self.scene = mjInterface(host=host, port=port)
            self._connected = False
        else:
            self.scene = scene
            self._connected = False

        if self.scene.mj_connected() == 0:
            if self.scene.mj_connect() == 0:
                self._connected = True
            else:
                raise Exception("Connection to MuJoCo simulator failed")
        else:
            self._connected = True
        self.Message("Gripper connected to MuJoCo", 1)

        self.tsamp = 0.02
        self._width_grasp = 0
        self._width = 0
        self._width_max = width_max
        self._state = -1
        self._ActuatorNames = [actuator_name]
        self._last_update = -100
        self._info = self.scene.mj_info()
        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo actuator handles.

        Returns
        -------
        None
            This method caches the actuator handles.
        """
        self._ActuatorHandles = self.scene.mj_name2id("actuator", self._ActuatorNames[0])
        if self._ActuatorHandles == -1:
            raise Exception("Check naming of actuators in MJCF model")

    def GetState(self) -> str:
        """
        Return the gripper state as a human-readable string.

        Returns
        -------
        str
            "Opened", "Closed", "Undefined", or "Unknown".
        """
        if self._connected and ((self.simtime() - self._last_update) > (self.tsamp * 0.0001)):
            ctrl = self.scene.mj_get_control()
            self._width = self._ctrl_sign * np.take(ctrl.ctrl, self._ActuatorHandles)
            self._last_update = self.simtime()
            if self._state == 0:
                return "Opened"
            if self._state == 1:
                return "Closed"
            return "Undefined"
        return "Unknown"

    def Move(self, width: float, speed: float = 0.1, timeout: float = 1, check: bool = True, eps: float = 0.001) -> bool:
        """
        Move the gripper to a target width.

        Parameters
        ----------
        width : float
            Target gripper opening width.
        speed : float, optional
            Motion speed (unused in MuJoCo backend).
        timeout : float, optional
            Maximum time to wait for completion.
        check : bool, optional
            If True, wait until the target is reached.
        eps : float, optional
            Tolerance for reaching the target width.

        Returns
        -------
        bool
            `True` if the target width is reached, `False` on timeout.
        """
        del speed
        target_width = max(min(width, self._width_max), 0.0)
        self.SendCmd(target_width)
        success = True
        if timeout > 0:
            sleep(timeout)
        if check:
            t0 = self.simtime()
            while np.abs(self.GetWidth() - target_width) > eps:
                if (self.simtime() - t0) > max(timeout, 2.0):
                    self.Message("Gripper move goal not reached", 2)
                    success = False
                    break
        self.Message("Gripper moveed", 2)
        self._state = -1
        return success

    def GetWidth(self) -> float:
        """
        Return the current gripper width.

        Returns
        -------
        float
            Current gripper opening width.
        """
        self.GetState()
        return self._width

    def AttachTo(self, robot: "robot") -> None:
        """
        Attach the gripper to a robot and reuse its scene.

        Parameters
        ----------
        robot : 'robot'
            Robot instance providing the MuJoCo scene.

        Returns
        -------
        None
            This method stores the robot reference and reuses its scene.
        """
        self.Robot = robot
        self.scene = robot.scene

    def SendCmd(self, u: float) -> None:
        """
        Send a control command to the gripper actuator.

        Parameters
        ----------
        u : float
            Target control value for the gripper actuator.

        Returns
        -------
        None
            This method updates the actuator control signal in the MuJoCo scene.
        """
        if self._connected:
            ctrl = self.scene.mj_get_control()
            ctrl.ctrl[self._ActuatorHandles] = self._ctrl_sign * u
            self.scene.mj_set_control(ctrl)


class panda_gripper(gripper_mujoco):
    """MuJoCo gripper wrapper for the Franka Panda hand."""

    def __init__(self, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, actuator_name: str = "panda_gripper", **kwargs: Any) -> None:
        """
        Initialize a MuJoCo Panda gripper interface.

        Parameters
        ----------
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        actuator_name : str, optional
            Name of the MuJoCo actuator for the gripper, default is "panda_gripper".
        **kwargs : Any
            Additional keyword arguments passed to `gripper_mujoco.__init__`.

        Returns
        -------
        None
            This constructor initializes the Panda MuJoCo gripper in place.
        """
        super().__init__(
            name="Panda_Gripper_MuJoCo",
            actuator_name=actuator_name,
            width_max=0.077,
            scene=scene,
            host=host,
            port=port,
            **kwargs,
        )


class robotiq_gripper(gripper_mujoco):
    """MuJoCo gripper wrapper for the Robotiq 2f85 gripper."""

    def __init__(self, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, actuator_name: str = "robotiq_gripper", **kwargs: Any) -> None:
        """
        Initialize a MuJoCo Robotiq gripper interface.

        Parameters
        ----------
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        actuator_name : str, optional
            Name of the MuJoCo actuator for the gripper, default is "robotiq_gripper".
        **kwargs : Any
            Additional keyword arguments passed to `gripper_mujoco.__init__`.

        Returns
        -------
        None
            This constructor initializes the Panda MuJoCo gripper in place.
        """
        super().__init__(
            name="Robotiq_Gripper_MuJoCo",
            actuator_name=actuator_name,
            width_max=0.077,
            scene=scene,
            host=host,
            port=port,
            **kwargs,
        )


class z1_gripper(gripper_mujoco):
    """MuJoCo gripper wrapper for the Unitree Z1 gripper."""

    def __init__(self, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, actuator_name: str = "z1_gripper", **kwargs: Any) -> None:
        """
        Initialize a MuJoCo Unitree Z1 gripper interface.

        Parameters
        ----------
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        actuator_name : str, optional
            Name of the MuJoCo actuator for the gripper, default is "z1_gripper".
        **kwargs : Any
            Additional keyword arguments passed to `gripper_mujoco.__init__`.

        Note
        ----
        The joint position and control command are reversed.

        Returns
        -------
        None
            This constructor initializes the Z1 MuJoCo gripper in place.
        """
        super().__init__(
            name="Z1_Gripper_MuJoCo",
            actuator_name=actuator_name,
            width_max=1.51844,
            scene=scene,
            host=host,
            port=port,
            ctrl_sign=-1.0,
            **kwargs,
        )
