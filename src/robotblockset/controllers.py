"""Controller base classes and compliance-control defaults.

Copyright (c) 2024- Jozef Stefan Institute

Author: Mihael Simonic
"""

from abc import abstractmethod
from typing import Any

import numpy as np

from robotblockset.tools import rbs_object
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType


class rbs_controller_type(rbs_object):
    """Base interface for RobotBlockSet controllers.

    This base class defines the motion and compliance methods expected from
    controller implementations. Subclasses can override the compliance-related
    methods when the underlying backend supports configurable stiffness or
    damping.
    """

    @abstractmethod
    def GoTo_q(self) -> None:
        """Move in joint space.

        Notes
        -----
        This method must be implemented by subclasses that support
        joint-space commands.
        """
        pass

    @abstractmethod
    def GoTo_X(self) -> None:
        """Move in Cartesian space.

        Notes
        -----
        This method must be implemented by subclasses that support
        Cartesian-space commands.
        """
        pass

    def GetJointStiffness(self) -> JointConfigurationType:
        """Return joint stiffness values.

        Returns
        -------
        JointConfigurationType
            Joint stiffness values. A default high-stiffness vector is returned
            when the backend does not support explicit stiffness queries.
        """
        self.Message("Compliance not supported", 3)
        return np.ones(self.nj) * 100000

    def SetJointStiffness(self, stiffness: ArrayLike, **kwargs: Any) -> None:
        """Set joint stiffness.

        Parameters
        ----------
        stiffness : ArrayLike
            Desired joint stiffness values.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def GetJointDamping(self) -> JointConfigurationType:
        """Return joint damping values.

        Returns
        -------
        JointConfigurationType
            Joint damping values. A unit vector is returned when the backend
            does not support explicit damping queries.
        """
        self.Message("Compliance not supported", 3)
        return np.ones(self.nj)

    def SetJointDamping(self, damping: ArrayLike, **kwargs: Any) -> None:
        """Set joint damping.

        Parameters
        ----------
        damping : ArrayLike
            Desired joint damping values.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def SetJointSoft(self, softness: float, **kwargs: Any) -> None:
        """Set joint compliance scalar.

        Parameters
        ----------
        softness : float
            compliant behavior.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def SetJointStiff(self) -> None:
        """Set joints to stiff mode.

        Notes
        -----
        This is a convenience wrapper around :meth:`SetJointSoft` with a
        softness of ``1.0``.
        """
        self.SetJointSoft(1.0)

    def SetJointCompliant(self) -> None:
        """Set joints to compliant mode.

        Notes
        -----
        This is a convenience wrapper around :meth:`SetJointSoft` with a
        softness of ``0.0``.
        """
        self.SetJointSoft(0.0)

    def GetCartesianStiffness(self) -> np.ndarray:
        """Return Cartesian stiffness values.

        Returns
        -------
        np.ndarray
            Cartesian stiffness values for the six task-space degrees of
            freedom. A default high-stiffness vector is returned when the
            backend does not support explicit stiffness queries.
        """
        self.Message("Compliance not supported", 3)
        return np.ones(6) * 100000

    def SetCartesianStiffness(self, stiffness: ArrayLike, **kwargs: Any) -> None:
        """Set Cartesian stiffness.

        Parameters
        ----------
        stiffness : ArrayLike
            Desired Cartesian stiffness values.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def GetCartesianDamping(self) -> np.ndarray:
        """Return Cartesian damping values.

        Returns
        -------
        np.ndarray
            Cartesian damping values for the six task-space degrees of freedom.
            A unit vector is returned when the backend does not support explicit
            damping queries.
        """
        self.Message("Compliance not supported", 3)
        return np.ones(6)

    def SetCartesianDamping(self, damping: ArrayLike, **kwargs: Any) -> None:
        """Set Cartesian damping.

        Parameters
        ----------
        damping : ArrayLike
            Desired Cartesian damping values.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def SetCartesianSoft(self, softness: float, **kwargs: Any) -> None:
        """Set Cartesian compliance scalar.

        Parameters
        ----------
        softness : float
            Scalar compliance factor for task-space behavior.
        **kwargs : Any
            Backend-specific options accepted by subclasses.

        Notes
        -----
        The default implementation only reports that compliance control is not
        supported.
        """
        self.Message("Compliance not supported", 3)

    def SetCartesianStiff(self) -> None:
        """Set Cartesian space to stiff mode.

        Notes
        -----
        This is a convenience wrapper around :meth:`SetCartesianSoft` with a
        softness of ``1.0``.
        """
        self.SetCartesianSoft(1.0)

    def SetCartesianCompliant(self) -> None:
        """Set Cartesian space to compliant mode.

        Notes
        -----
        This is a convenience wrapper around :meth:`SetCartesianSoft` with a
        softness of ``0.0``.
        """
        self.SetCartesianSoft(0.0)


class joint_controller_type(rbs_controller_type):
    """Base class for controllers that only accept joint commands."""

    def GoTo_X(self) -> None:
        """Reject Cartesian commands.

        Raises
        ------
        TypeError
            Always raised because joint controllers do not accept Cartesian
            motion commands.
        """
        raise TypeError("cartesian commands should not be sent to joint controller")


class cartesian_controller_type(rbs_controller_type):
    """Base class for controllers that only accept Cartesian commands."""

    def GoTo_q(self) -> None:
        """Reject joint commands.

        Raises
        ------
        TypeError
            Always raised because Cartesian controllers do not accept
            joint-space motion commands.
        """
        raise TypeError("joint commands should not be sent to cartesian controller")


class compliant_controller_type(rbs_controller_type):
    """Marker base class for controllers with compliance support."""

    pass
