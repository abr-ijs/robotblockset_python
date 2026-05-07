"""ROS sensor interface implementations.

This module defines ROS-backed sensor wrappers used by RobotBlockSet sensor
interfaces. It provides force-torque state subscriptions and raw wrench access
through ROS topics.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Mihael Simonic, Leon Zlajpah.
"""

from typing import Any

import rospy
from geometry_msgs.msg import Wrench

from robotblockset.sensors import force_torque_sensor
from robotblockset.rbs_typing import WrenchType


class ati_ft_ros(force_torque_sensor):
    """
    ROS interface for an ATI force-torque sensor.

    Attributes
    ----------
    _namespace : str
        ROS namespace prefix used to resolve the sensor topic.
    _topic_FT : str
        Topic used to subscribe to raw wrench data.
    """

    def __init__(self, namespace: str = "", **kwargs: Any) -> None:
        """
        Initialize the ATI force-torque sensor ROS wrapper.

        Parameters
        ----------
        namespace : str, optional
            ROS namespace prefix used to resolve the sensor topic.
        **kwargs : Any
            Additional keyword arguments reserved for future use.
        """
        self.Name = "ATI_FT:ROS"
        self._namespace = namespace
        self._verbose = 1

        self._topic_FT = f"{self._namespace}/netft_data"

        self._FT_subscriber = rospy.Subscriber(self._topic_FT, Wrench, self._joint_states_callback)

        self.Message("Created", 2)

    def _joint_states_callback(self, data: Wrench) -> None:
        """
        Store the most recent wrench message from the ROS topic.

        Parameters
        ----------
        data : Wrench
            Wrench message received from ROS.
        """
        self._FT_msg = data
        # self._FT_msg = copy.deepcopy(data)
        self._last_FT_callback_time = self.simtime()

    def GetRawFT(self) -> WrenchType:
        """
        Return the latest raw force-torque measurement.

        Returns
        -------
        WrenchType
            Latest raw force-torque sample.

        Notes
        -----
        The most recent ROS wrench message is converted into the RobotBlockSet
        sensor-data format.
        """
        t = self.simtime()
        if self._FT_msg is not None:
            if (t - self._FT_callback_time) > 10 * self.tsamp:
                self.WarningMessage("FT_state is not updated")
            else:
                _F = self._FT_msg.wrench.force
                _M = self._FT_msg.wrench.torque
                self.SensorData = [_F.x, _F.y, _F.z, _M.x, _M.y, _M.z]
        return self.SensorData
