"""PyMuJoCo Sensors Module.

This module provides force-torque sensor implementations for the Python MuJoCo simulator.
It mirrors the base sensor interfaces defined in `robotblockset.sensors` and exposes
PyMuJoCo-backed sensor data through the same API.

Copyright (c) 2025- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Any, Optional

from robotblockset.sensors import force_torque_sensor
from robotblockset.mujoco.scene_pymujoco import mujoco_scene


class ft_sensor_pymujoco(force_torque_sensor):
    """PyMuJoCo-backed force-torque sensor interface operating on a :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`."""

    def __init__(self, sensor_name: str = "FTSensor", scene: Optional[mujoco_scene] = None, sensor_force_name: Optional[str] = None, sensor_torque_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo force-torque sensor.

        Parameters
        ----------
        sensor_name : str, optional
            Sensor name prefix used for this instance.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            MuJoCo scene instance to read sensors from. Must be provided.
        sensor_force_name : str, optional
            Name of the force sensor in MuJoCo.
        sensor_torque_name : str, optional
            Name of the torque sensor in MuJoCo.
        **kwargs : dict
            Additional keyword arguments for the base force-torque sensor class.
        """
        force_torque_sensor.__init__(self, **kwargs)
        self.Name = sensor_name + ":PyMuJoCo"
        if scene is None:
            raise ValueError("MuJoCo scene is not defined")
        self.scene = scene
        print(f"Sensors {sensor_name} connected to PyMuJoCo")

        if sensor_force_name is None:
            sensor_force_name = ["force"]
        if sensor_torque_name is None:
            sensor_torque_name = ["torque"]
        self.tsamp = 0.01
        self.SensorForceName = sensor_force_name
        self.SensorTorqueName = sensor_torque_name
        self._SensorNamesList = [self.scene.model.sensor(i).name for i in range(self.scene.model.nsensor)]
        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo sensor handles.

        Returns
        -------
        None
        """
        if self.SensorForceName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorForceName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorHandles = list(range(adr, adr + dim))
        else:
            self._SensoreHandles = None

        if self.SensorTorqueName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorTorqueName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorHandles.append(list(range(adr, adr + dim)))
        else:
            self._SensorHandles = None

        self.GetState()
        self.Message("Initialized", 2)

    def GetRawFT(self) -> np.ndarray:
        """
        Read raw force-torque data from PyMuJoCo sensors.

        Returns
        -------
        np.ndarray
            Force-torque vector (6,) or NaNs if unavailable.
        """
        if self.SensorForceName in self._SensorNamesList and self.SensorTorqueName in self._SensorNamesList:
            self.SensorData = np.concatenate((self.scene.data.sensor(self.SensorForceName).data, self.scene.data.sensor(self.SensorTorqueName).data))
        else:
            self.SensorData = np.full(6, np.nan)
        return self.SensorData
