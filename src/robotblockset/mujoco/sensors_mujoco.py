"""MuJoCo Sensors Module.

This module provides force-torque sensor implementations for the MuJoCo simulator.
It mirrors the base sensor interfaces defined in `robotblockset.sensors` and exposes
MuJoCo-backed sensor data through the same API.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Any, Optional, Sequence

from robotblockset.sensors import force_torque_sensor
from robotblockset.mujoco.mujoco_api import mjInterface


class ft_sensor_mujoco(force_torque_sensor):
    """MuJoCo-backed force-torque sensor interface using the socket-based server API."""

    def __init__(self, scene: Optional[mjInterface] = None, host: str = "localhost", sensor_force_name: Optional[Sequence[str]] = None, sensor_torque_name: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """
        Initialize a MuJoCo force-torque sensor.

        Parameters
        ----------
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        sensor_force_name : Sequence[str], optional
            Name(s) of the force sensor in MuJoCo.
        sensor_torque_name : Sequence[str], optional
            Name(s) of the torque sensor in MuJoCo.
        **kwargs : dict
            Additional keyword arguments for the base force-torque sensor class.
        """
        force_torque_sensor.__init__(self, **kwargs)
        self.Name = "FTSensor_MuJoCo"
        if scene is None:
            self.scene = mjInterface(host=host)
            self._connected = False
        else:
            self.scene = scene
        if self.scene.mj_connected() == 0:
            if self.scene.mj_connect() == 0:
                self._connected = True
            else:
                raise Exception("Connection to MuJoCo simulator failed")
        else:
            self._connected = True
        self.Message("FT sensor connected to MuJoCo", 2)

        if sensor_force_name is None:
            sensor_force_name = ["force"]
        if sensor_torque_name is None:
            sensor_torque_name = ["torque"]
        self.tsamp = 0.01
        self.SensorForceName = sensor_force_name
        self.SensorTorqueName = sensor_torque_name
        self._info = self.scene.mj_info()
        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo sensor handles.

        Returns
        -------
        None
        """
        idx = [self.scene.mj_name2id("sensor", self.SensorForceName)]
        idx.append(self.scene.mj_name2id("sensor", self.SensorTorqueName))
        if np.all(np.array(idx) >= 0):
            hh = []
            for i in range(2):
                adr = self._info.sensor_adr[idx[i]]
                dim = self._info.sensor_dim[idx[i]]
                hh += list(range(adr, adr + dim))
            self._SensorHandles = hh
        else:
            self._SensorHandles = None
        self.GetState()
        self.Message("Initialized", 2)

    def GetRawFT(self) -> np.ndarray:
        """
        Read raw force-torque data from MuJoCo sensors.

        Returns
        -------
        np.ndarray
            Force-torque vector (6,) or NaNs if unavailable.
        """
        if self._info.nsensor > 0 and self._SensorHandles is not None:
            _sensor = self.scene.mj_get_sensor()
            self.SensorData = np.take(_sensor.sensordata, self._SensorHandles)
        else:
            self.SensorData = np.full(6, np.nan)
        return self.SensorData
