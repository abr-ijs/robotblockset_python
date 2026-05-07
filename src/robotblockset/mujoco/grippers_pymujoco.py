"""PyMuJoCo Grippers Module.

This module provides gripper implementations for the Python MuJoCo simulator.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from time import sleep
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from robotblockset.grippers import gripper
from robotblockset.mujoco.scene_pymujoco import mujoco_scene

if TYPE_CHECKING:
    from robotblockset.robots import robot


class gripper_pymujoco(gripper):
    """PyMuJoCo gripper interface operating on a :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene` instance."""

    def __init__(self, name: str, actuator_name: str, width_max: float, scene: Optional[mujoco_scene] = None, ctrl_sign: float = 1.0, **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo gripper interface.

        Parameters
        ----------
        name : str
            Gripper instance name.
        actuator_name : str
            MuJoCo actuator name used to command the gripper.
        width_max : float
            Maximum allowed opening width.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            MuJoCo scene instance to control. Must be provided.
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
            This constructor initializes the PyMuJoCo gripper object in place.
        """
        gripper.__init__(self, **kwargs)
        self.Name = name
        self._ctrl_sign = ctrl_sign

        if scene is None:
            raise ValueError("MuJoCo scene is not defined")
        self.scene = scene
        self.Message("Gripper connected to PyMuJoCo", 1)

        self.tsamp = 0.01
        self._width_grasp = 0
        self._width = 0
        self._width_max = width_max
        self._state = -1
        self._ActuatorNames = [actuator_name]
        self._last_update = -100
        self.Init()

    def Init(self) -> None:
        """
        Initialize pyMuJoCo actuator handles.

        Returns
        -------
        None
            This method caches the actuator handles.
        """
        actuator_names = [self.scene.model.actuator(i).name for i in range(self.scene.model.nu)]
        actuator_handle = self.scene.model.actuator(self._ActuatorNames[0]).id if self._ActuatorNames[0] in actuator_names else -1
        self._ActuatorHandles = np.array(actuator_handle, dtype=np.int32)
        if int(self._ActuatorHandles) == -1:
            raise Exception("Check naming of actuators in MJCF model")

    def GetState(self) -> str:
        """
        Return the gripper state as a human-readable string.

        Returns
        -------
        str
            "Opened", "Closed", "Undefined", or "Unknown".
        """
        if (self.simtime() - self._last_update) > (self.tsamp * 0.9):
            self._width = self._ctrl_sign * np.take(self.scene.data.ctrl.copy(), self._ActuatorHandles)
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
            Motion speed (unused in PyMuJoCo backend).
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
        ctrl = self.scene.data.ctrl.copy()
        ctrl[self._ActuatorHandles] = self._ctrl_sign * u
        self.scene.data.ctrl = ctrl


class panda_gripper(gripper_pymujoco):
    """PyMuJoCo gripper wrapper for the Franka Panda hand."""

    def __init__(self, scene: Optional[mujoco_scene] = None, actuator_name: str = "panda_gripper", **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo Panda gripper interface.

        Parameters
        ----------
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            MuJoCo scene instance to control. Must be provided.
        actuator_name : str, optional
            Name of the MuJoCo actuator for the gripper, default is "panda_gripper".
        **kwargs : Any
            Additional keyword arguments passed to `gripper_pymujoco.__init__`.

        Returns
        -------
        None
            This constructor initializes the Panda PyMuJoCo gripper in place.
        """
        super().__init__(
            name="Panda_Gripper_PyMuJoCo",
            actuator_name=actuator_name,
            width_max=0.077,
            scene=scene,
            **kwargs,
        )


class robotiq_gripper(gripper_pymujoco):
    """PyMuJoCo gripper wrapper for the Robotiq 2f85 gripper."""

    def __init__(self, scene: Optional[mujoco_scene] = None, actuator_name: str = "robotiq_gripper", **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo Robotiq gripper interface.

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
            **kwargs,
        )


class z1_gripper(gripper_pymujoco):
    """PyMuJoCo gripper wrapper for the Unitree Z1 gripper."""

    def __init__(self, scene: Optional[mujoco_scene] = None, actuator_name: str = "z1_gripper", **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo Unitree Z1 gripper interface.

        Parameters
        ----------
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            MuJoCo scene instance to control. Must be provided.
        actuator_name : str, optional
            Name of the MuJoCo actuator for the gripper, default is "z1_gripper".
        **kwargs : Any
            Additional keyword arguments passed to `gripper_pymujoco.__init__`.

        Note
        ----
        The joint position and control command are reversed.

        Returns
        -------
        None
            This constructor initializes the Z1 PyMuJoCo gripper in place.
        """
        super().__init__(
            name="Z1_Gripper_PyMuJoCo",
            actuator_name=actuator_name,
            width_max=1.51844,
            scene=scene,
            qpos_sign=-1.0,
            ctrl_sign=-1.0,
            **kwargs,
        )
