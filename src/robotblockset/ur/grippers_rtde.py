"""UR RTDE Grippers Module.

This module provides a simple gripper interface for Universal Robots using
RTDE digital outputs. The implementation follows the `gripper` base class API
from `robotblockset.grippers`.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Any

from robotblockset.grippers import gripper
from robotblockset.robots import robot


class ur_gripper(gripper):
    """UR gripper interface controlled through the RTDE robot connection.

    Attributes
    ----------
    Name : str
        Identifier string for the gripper instance.
    GripperTagNames : str
        Tag name used for identifying the gripper in logs or configurations.
    Robot : robot
        Robot instance this gripper is associated with.
    control_signal : int
        Digital output index used to command the gripper.
    """

    def __init__(self, robot: robot, control_signal: int = 0, **kwargs: Any) -> None:
        """
        Initialize a UR gripper controlled via RTDE interface.

        Parameters
        ----------
        robot : robot
            robot
            An instance of the robot class that this gripper is attached to.
        control_signal : int, optional
            The number of digital output signal used to command the gripper (default is 0).
        **kwargs : Any
            Additional keyword arguments for extended configuration.

        Returns
        -------
        None
            This constructor initializes the RTDE gripper interface in place.
        """
        self.Name = "UR:Gripper:RTDE"
        self.GripperTagNames = "gripper"
        if not hasattr(robot, "rtde_i"):
            raise ValueError(f"Asociated robot {robot.Name} does NOT support RTDE.")
        self.Robot = robot
        self.control_signal = control_signal
        self._verbose = 1
        self._state = -1

        self.Message("Created")

    def Open(self) -> bool:
        """
        Opens the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the open operation.

        Returns
        -------
        bool
            True if the gripper successfully opens, False otherwise.
        """
        _tmp = self.Robot.rtde_i.setStandardDigitalOut(self.control_signal, False)
        if _tmp:
            self._state = 0
        return _tmp

    def Close(self) -> bool:
        """
        Closes the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the close operation.

        Returns
        -------
        bool
            True if the gripper successfully closes, False otherwise.
        """
        _tmp = self.Robot.rtde_i.setStandardDigitalOut(self.control_signal, True)
        if _tmp:
            self._state = 1
        return _tmp

    def Grasp(self, **kwargs) -> bool:
        """
        Closes the gripper.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the grasp operation, may include a "width" key.

        Returns
        -------
        bool
            True if the gripper successfully grasps, False otherwise.
        """
        return self.Close()

    def Move(self, width: float, **kwargs: Any) -> bool:
        """
        Opens or closes the gripper based on the specified width.
        Parameters
        ----------
        width : float
            If width>0 gripper opens else it closes.
        **kwargs : dict
            Additional arguments for the grasp operation, may include a "speed" key.

        Returns
        -------
        bool
            True if the gripper successfully moves, False otherwise.
        """
        if width == 0:
            return self.Close()
        else:
            return self.Open()

    def isOpened(self) -> bool:
        """
        Returns True if the gripper is open.

        Returns
        -------
        bool
            True if the gripper is open, False otherwise.
        """
        self._state = self.Robot.rtde_r.getDigitalOutState(self.control_signal)
        return self._state

    def isClosed(self) -> bool:
        """
        Returns True if the gripper is closed.

        Returns
        -------
        bool
            True if the gripper is closed, False otherwise.
        """
        self._state = self.Robot.rtde_r.getDigitalOutState(self.control_signal)
        return self._state

    def GetState(self) -> str:
        """
        Returns the current state of the gripper.

        Returns
        -------
        str
            The state of the gripper, either "Opened", "Closed", or "Undefined".
        """
        self._state = self.Robot.rtde_r.getDigitalOutState(self.control_signal)
        if self._state == 0:
            return "Opened"
        elif self._state == 1:
            return "Closed"
        else:
            return "Undefined"

    def Homing(self) -> bool:
        """
        Reset the gripper to an open state.

        Returns
        -------
        bool
            True if the gripper successfully homes, False otherwise.
        """
        return self.Open()
