"""Franka gripper interface via panda_py.

High-level interface for controlling gripper for Franka Emika Panda / FR3 robots via panda_py.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah
"""

try:
    from panda_py import libfranka  # pyright: ignore[reportMissingImports]
except Exception as e:
    raise e from RuntimeError(" Python bindings for the Panda not installed. \nYou can install them through pip:\n  pip install panda-python")

from typing import Any, Optional

from robotblockset.grippers import gripper
from robotblockset.robots import robot


class panda_gripper(gripper):
    """PandaPy-backed interface for controlling the Franka Emika Panda gripper.

    Attributes
    ----------
    Name : str
        Identifier string for the gripper instance.
    Robot : robot or None
        Robot instance this gripper is attached to, if available.
    hostname : str
        Hostname or IP address of the Panda/FR3 controller used by the gripper.
    """

    def __init__(self, robot: Optional[robot], name: str = "panda_gripper", hostname: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initializes the gripper object with default attributes.

        Parameters
        ----------
        robot : robot, optional
            robot, optional
            An instance of the robot class that the gripper is attached to.
        name : str, optional
            Name identifier for the gripper instance (default is 'panda_gripper').
        hostname : str, optional
            IP address or hostname of the Panda / FR3 robot. Has to be defined if no robot is selected.
        **kwargs : Any
            Additional keyword arguments for future extensions or configuration.

        Returns
        -------
        None
            This constructor initializes the Panda gripper interface in place.
        """
        self.Name = name
        self.Robot = robot
        if robot is None:
            self.hostname = hostname
        else:
            self.hostname = robot.hostname
        self.gripper = libfranka.Gripper(self.hostname)
        self.gripper_state = self.gripper.read_once()
        self._width_grasp = 0
        self._width = self.gripper_state.width
        self._width_max = self.gripper_state.max_width
        self._state = self.gripper_state.is_grasped
        self._speed = 0.0
        self._speed_max = 0.5
        self._verbose = 1

        self.Message("Created", 2)

    @property
    def width(self) -> float:
        """
        Get the current width of the gripper.

        Returns
        -------
        float
            Current gripper width.
        """
        self.gripper_state = self.gripper.read_once()
        return self.gripper_state.width

    def is_grasped(self) -> bool:
        """
        Check if the gripper is currently grasping an object.

        Returns
        -------
        bool
            True if the gripper is grasping an object, False otherwise.
        """
        self.gripper_state = self.gripper.read_once()
        return self.gripper_state.is_grasped

    def GetState(self) -> str:
        """
        Returns the current state of the gripper.

        Returns
        -------
        str
            The state of the gripper, either "Opened", "Closed", or "Undefined".
        """
        self.gripper_state = self.gripper.read_once()
        if self._state == 0:
            return "Opened"
        elif self._state == 1:
            return "Closed"
        else:
            return "Undefined"

    def Grasp(self, width: float, speed: float = 0.1, force: int = 5, eps: float = 0.005) -> bool:
        """
        Grasps an object with the gripper at a specified width.

        Parameters
        ----------
        width : float
            The width to which the gripper should close to grasp the object.
        speed : float, optional
            The speed at which the gripper should move (default is 0.1).
        force : int, optional
            The force to apply during the grasp (default is 5).
        eps : float, optional
            The tolerance for the grasping width (default is 0.005).
        Returns
        -------
        bool
            True if the gripper successfully grasps, False otherwise.
        """

        # Sends the goal to the action server.
        _success = self.gripper.grasp(width, speed, force, epsilon_inner=eps, epsilon_outer=eps)
        if _success:
            self._state = 1
        return _success

    def Move(self, width: float, speed: float = 0.1) -> bool:
        """
        Moves the gripper to a specified width.
        Parameters
        ----------
        width : float
            The width to which the gripper should move.
        speed : float, optional
            The speed at which the gripper should move (default is 0.1).

        Returns
        -------
        bool
            True if the gripper successfully moves, False otherwise.

        """
        self._state = -1
        return self.gripper.move(width, speed)

    def Homing(self) -> bool:
        """
        Resets the gripper to an open state.

        Returns
        -------
        bool
            True if the gripper successfully homes, False otherwise.
        """
        self._state = 0
        return self.gripper.homing()
