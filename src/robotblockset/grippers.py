"""Grippers Module.

This module provides the implementation of different gripper types and their functionality.

The `gripper` class provides a common interface for real or simulated grippers, while the `dummygripper`
class offers a mock implementation for scenarios where no physical hardware is involved.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from abc import abstractmethod
from time import perf_counter
from typing import Any, Optional, Tuple

from robotblockset.tools import rbs_object
from robotblockset.robots import robot


class gripper(rbs_object):
    """
    A class representing a gripper attached to a robot, allowing control over the gripper's state and actions.

    Attributes
    ----------
    Name : str
        Name of the gripper, default is "Gripper".
    Robot : robot, optional
        The robot to which the gripper is attached, by default None.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the gripper object with default attributes.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to initialize the gripper object.
        """
        rbs_object.__init__(self)
        self.Name = "Gripper"
        self._verbose = 1  # verbose level
        self._state = -1  # gripper state
        self.Robot = None  # robot to which gripper is attached

    def __del__(self) -> None:
        """
        Destructor for the gripper object. Detaches the gripper from the robot, if attached.
        """
        if self.Robot is not None:
            self.Robot.Gripper = None

    def simtime(self) -> float:
        """
        Returns the current simulation time.

        Returns
        -------
        float
            The current simulation time.
        """
        return perf_counter()

    @abstractmethod
    def Move(self, width: float, **kwargs: Any) -> bool:
        """
        Moves the gripper to a specified width. This is an abstract method that must be implemented.

        Parameters
        ----------
        width : float
            The width to which the gripper should move.
        **kwargs : dict
            Additional arguments for the move operation.

        Returns
        -------
        bool
            True if the gripper successfully moves, False otherwise.
        """
        pass

    def Open(self, **kwargs: Any) -> bool:
        """
        Opens the gripper to its maximum width.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the open operation.

        Returns
        -------
        bool
            True if the gripper successfully opens, False otherwise.
        """
        _succx = self.Move(self._width_max, **kwargs)
        if _succx:
            self._state = 0
        else:
            self._state = -1
        return _succx

    def Close(self, **kwargs: Any) -> bool:
        """
        Closes the gripper to a width of 0.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the close operation.

        Returns
        -------
        bool
            True if the gripper successfully closes, False otherwise.
        """
        _succx = self.Move(0, **kwargs)
        if _succx:
            self._state = 1
        else:
            self._state = -1
        return _succx

    def Grasp(self, **kwargs: Any) -> bool:
        """
        Grasps an object with the gripper at a specified width.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the grasp operation, may include a "width" key.

        Returns
        -------
        bool
            True if the gripper successfully grasps, False otherwise.
        """
        if "width" in kwargs:
            _width = max(min(kwargs["width"], self._width_max), 0)
            del kwargs["width"]
        else:
            _width = 0
        _succx = self.Move(_width, **kwargs)
        if _succx:
            self._state = 1
        else:
            self._state = -1
        return _succx

    def Homing(self, **kwargs: Any) -> bool:
        """
        Resets the gripper to an open state.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the homing operation.

        Returns
        -------
        bool
            True if the gripper successfully homes, False otherwise.
        """
        kwargs.setdefault("check", True)
        return self.Open(**kwargs)

    def isOpened(self) -> bool:
        """
        Returns True if the gripper is open.

        Returns
        -------
        bool
            True if the gripper is open, False otherwise.
        """
        return self._state == 0

    def isClosed(self) -> bool:
        """
        Returns True if the gripper is closed.

        Returns
        -------
        bool
            True if the gripper is closed, False otherwise.
        """
        return self._state == 1

    def GetState(self) -> str:
        """
        Returns the current state of the gripper.

        Returns
        -------
        str
            The state of the gripper, either "Opened", "Closed", or "Undefined".
        """
        if self._state == 0:
            return "Opened"
        elif self._state == 1:
            return "Closed"
        else:
            return "Undefined"

    def AttachTo(self, robot: robot) -> None:
        """
        Attaches the gripper to a robot.

        Parameters
        ----------
        robot : robot
            robot
            The robot to which the gripper should be attached.
        """
        self.Robot = robot

    def Detach(self) -> None:
        """
        Detaches the gripper from its current robot.
        """
        self.Robot = None

    def GetAttachedRobot(self) -> Tuple[Optional[robot], str]:
        """
        Returns the robot and its name, or (None, "None") if no robot is attached.

        Returns
        -------
        tuple
            A tuple containing the robot and its name, or (None, "None") if no robot is attached.
        """
        if self.Robot is None:
            return None, "None"
        else:
            return self.Robot, self.Robot.Name


class dummygripper(gripper):
    """
    A dummy gripper class that simulates a gripper's basic functionality without actually controlling hardware.
    This class is a subclass of the `gripper` class and provides mock implementations for gripper actions.

    Attributes
    ----------
    Name : str
        Name of the gripper, default is "DummyGripper".
    Robot : robot, optional
        The robot to which the gripper is attached, by default None.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the dummy gripper object with default attributes.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for initializing the dummy gripper object.
        """
        gripper.__init__(self, **kwargs)
        self.Name = "DummyGripper"
        self._state = -1

    def Open(self, **kwargs: Any) -> bool:
        """
        Simulates opening the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the open operation.

        Returns
        -------
        bool
            Always returns True to simulate a successful operation.
        """
        self._state = 0
        return True

    def Close(self, **kwargs: Any) -> bool:
        """
        Simulates closing the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the close operation.

        Returns
        -------
        bool
            Always returns True to simulate a successful operation.
        """
        self._state = 1
        return True

    def Grasp(self, **kwargs: Any) -> bool:
        """
        Simulates grasping an object with the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the grasp operation.

        Returns
        -------
        bool
            Always returns True to simulate a successful operation.
        """
        self._state = 0
        return True

    def Move(self, width: float, **kwargs: Any) -> bool:
        """
        Simulates moving the gripper to a specified width.

        Parameters
        ----------
        width : float
            The width to which the gripper should move.
        **kwargs : dict
            Additional arguments for the move operation.

        Returns
        -------
        bool
            Always returns True to simulate a successful operation.
        """
        self._state = -1
        return True

    def Homing(self, **kwargs: Any) -> bool:
        """
        Simulates resetting the gripper to an open state.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the homing operation.

        Returns
        -------
        bool
            Always returns True to simulate a successful operation.
        """
        self._state = 0
        return True


def isgripper(obj: object) -> bool:
    """
    Checks if the given object is an instance of the `gripper` class.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is an instance of the `gripper` class, otherwise False.
    """
    return isinstance(obj, gripper)
