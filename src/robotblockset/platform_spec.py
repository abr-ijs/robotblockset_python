"""Platforms' Parameters and Kinematic Models Module.

This module defines the platform parameters and kinematic models for different mobile robots.
It includes specific models for platforms like Tiago and MiR, which are commonly used for mobile robotics.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Optional, Tuple

from robotblockset.platforms import platform
from robotblockset.transformations import map_pose
from robotblockset.rbs_typing import ArrayLike


class tiagobase_spec(platform):
    """
    Defines the parameters and kinematic model for the Tiago mobile robot platform.

    Attributes
    ----------
    Name : str
        The name of the platform ("Tiagobase").
    nj : int
        The number of joints (2).
    q_max : numpy.ndarray
        The upper joint limits.
    q_min : numpy.ndarray
        The lower joint limits.
    qdot_max : numpy.ndarray
        The maximal joint velocities.
    v_max : numpy.ndarray
        The maximal task velocities.
    v_min : numpy.ndarray
        The minimal task velocities.
    r_platform : float
        The radius of the platform.
    laser_offset : float
        The distance from the platform center to the laser scanner.
    """

    def __init__(self) -> None:
        """
        Initializes the Tiago platform with specific parameters such as joint limits, velocities,
        platform radius, and laser offset.
        """
        self.Name = "tiagobase"
        self.nj = 2
        self.q_max = np.array([10000, 10000])  # upper joint limits
        self.q_min = np.array([-10000, -10000])  # lower joint limits
        self.qdot_max = np.array([10, 10])  # maximal joint velocities
        self.v_max = np.array([1, 2])  # maximal task velocities
        self.v_min = np.array([-0.2, -2])  # maximal task velocities
        self.r_platform = 0.52 / 2  # platform radius
        self.laser_offset = self.r_platform - 0.04  # distance from the platform center to the laser scanner

    def Kinmodel(self, x: Optional[ArrayLike] = None, out: str = "x") -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the kinematic model and Jacobian of the Tiago platform.

        Parameters
        ----------
        x : ArrayLike, optional
            The configuration state of the platform (position and/or orientation).
        out : str, optional
            The desired output format (default is "x").

        Returns
        -------
        tuple
            A tuple containing the kinematic model and Jacobian matrix.
        """
        if x is None:
            _x = x
        else:
            _x = np.copy(self._actual.x)

        wheel_r = 0.0985
        wheel_d = 0.4044
        # plate_h = 0.2976

        J = np.zeros((6, 2))
        J[0, :] = np.array([wheel_r / 2, wheel_r / 2])
        J[5, :] = np.array([-wheel_r / wheel_d, wheel_r / wheel_d])
        return map_pose(x=_x, out=out), J


class mir100_spec(platform):
    """
    Defines the parameters and kinematic model for the MiR 100 mobile robot platform.

    Attributes
    ----------
    Name : str
        The name of the platform ("MiR").
    nj : int
        The number of joints (2).
    q_max : numpy.ndarray
        The upper joint limits.
    q_min : numpy.ndarray
        The lower joint limits.
    qdot_max : numpy.ndarray
        The maximal joint velocities.
    v_max : numpy.ndarray
        The maximal task velocities.
    v_min : numpy.ndarray
        The minimal task velocities.
    """

    def __init__(self) -> None:
        """
        Initializes the MiR platform with specific parameters such as joint limits, velocities,
        and other platform characteristics.
        """
        self.Name = "mir"
        self.nj = 2
        self.q_max = np.array([10000, 10000])  # upper joint limits
        self.q_min = np.array([-10000, -10000])  # lower joint limits
        self.qdot_max = np.array([10, 10])  # maximal joint velocities
        self.v_max = np.array([1.5, 3])  # maximal task velocities
        self.v_min = np.array([-0.3, -3])  # minimal task velocities

    def Kinmodel(self, x: Optional[ArrayLike] = None, out: str = "x") -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the kinematic model and Jacobian of the MiR platform.

        Parameters
        ----------
        x : ArrayLike, optional
            The configuration state of the platform (position and/or orientation).
        out : str, optional
            The desired output format (default is "x").

        Returns
        -------
        tuple
            A tuple containing the kinematic model and Jacobian matrix.
        """
        if x is None:
            _x = x
        else:
            _x = np.copy(self._actual.x)

        wheel_r = 0.0625
        wheel_d = 0.222604 * 2
        # plate_h = 0.354

        J = np.zeros((6, 2))
        J[0, :] = np.array([wheel_r / 2, wheel_r / 2])
        J[5, :] = np.array([-wheel_r / wheel_d, wheel_r / wheel_d])
        return map_pose(x=_x, out=out), J
