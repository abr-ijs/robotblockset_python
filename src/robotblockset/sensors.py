"""Sensors module.

This module defines a flexible and extensible framework for integrating sensors into robotic platforms.
It includes an abstract `sensor` base class, a `force_torque_sensor` interface for handling 6-axis force-torque data,
and concrete implementations such as `ati_ft_sensor` for ATI hardware and `dummysensor` for simulation or testing.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from abc import abstractmethod
import numpy as np
from time import perf_counter, sleep, time
import copy
import socket
import struct
from typing import Any, Optional, Tuple

from robotblockset.tools import load_params, rbs_object, vector
from robotblockset.robots import robot
from robotblockset.rbs_typing import ArrayLike, Vector3DType, WrenchType


class sensor(rbs_object):
    """
    Abstract base class for a sensor attached to a robot.

    This class represents a sensor in the context of a robotic system. It includes methods for attaching
    the sensor to a robot, updating the sensor state, and simulating time. Derived classes must implement
    the `GetState` method to retrieve the sensor's state.

    Attributes
    ----------
    Name : str
        The name of the sensor (default is "Sensor").
    Robot : robot or None
        The robot to which the sensor is attached. Default is None.

    Notes
    -----
    This is an abstract class, and `GetState()` must be implemented in subclasses to define the behavior
    for retrieving the sensor's state.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the sensor object.

        Parameters
        ----------
        **kwargs : Any
            Additional parameters to customize the sensor. If ``robot`` is provided,
            the sensor is attached to that robot via `AttachTo`.

        Returns
        -------
        None
            This constructor initializes the sensor object in place.
        """
        attached_robot = kwargs.pop("robot", None)
        rbs_object.__init__(self)
        self.Name = "Sensor"
        self.Robot = None  # robot to which sensor is attached
        self.tsamp: float = 0.01  # Sampling rate for the sensor
        self._verbose = 1  # verbose level
        self._last_update: float = -100
        if attached_robot is not None:
            self.AttachTo(attached_robot)

    def simtime(self) -> float:
        """
        Get the current simulation time.

        Returns
        -------
        float
            The current simulation time, as returned by `perf_counter`.
        """
        return perf_counter()

    @abstractmethod
    def GetState(self) -> None:
        """
        Abstract method to retrieve the sensor's state.

        Derived classes must implement this method to return the state of the sensor.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        pass

    def SetTsamp(self, tsamp: float) -> None:
        """
        Set the sensor's sampling time.

        Parameters
        ----------
        tsamp : float
            Sampling period of the sensor in seconds.

        Returns
        -------
        None
            This method updates the sampling time in place.
        """
        self.tsamp = tsamp

    def Update(self) -> None:
        """
        Update the sensor's state by calling the `GetState` method.

        Returns
        -------
        None
            This method refreshes the current sensor state in place.
        """
        self.GetState()

    def AttachTo(self, robot: robot) -> None:
        """
        Attach the sensor to a robot.

        Parameters
        ----------
        robot : robot
            Robot instance to which the sensor is attached.

        Returns
        -------
        None
            This method stores the robot reference in the sensor.
        """
        self.Robot = robot

    def Detach(self) -> None:
        """
        Detach the sensor from the robot.

        Returns
        -------
        None
            This method clears the stored robot reference.
        """
        self.Robot = None

    def GetAttachedRobot(self) -> Tuple[Optional[robot], str]:
        """
        Get the robot to which the sensor is attached.

        Returns
        -------
        tuple
            A tuple containing the robot object and its name, or (None, "None") if not attached.
        """
        if self.Robot is None:
            return None, "None"
        else:
            return self.Robot, self.Robot.Name


class force_torque_sensor(sensor):
    """
    Class for a force-torque sensor attached to a robot.

    This class extends the `sensor` class to represent a force-torque sensor. The sensor measures forces
    and torques in six degrees of freedom (3 forces and 3 torques) and provides methods for getting
    the raw data, updating the sensor state, zeroing the sensor, and setting or updating the load attached to the sensor.

    Attributes
    ----------
    SensorData : np.ndarray
        A 1D array containing the sensor's data (6 values: 3 forces and 3 torques).
    Load : load_params
        The load object associated with the sensor, containing mass, center of mass, and inertia properties.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the force-torque sensor with default parameters.

        Parameters
        ----------
        **kwargs : Any
            Additional parameters forwarded to `sensor.__init__`. If ``robot`` is
            provided, the sensor is attached to that robot via `AttachTo`.

        Returns
        -------
        None
            This constructor initializes the force-torque sensor object in place.
        """
        sensor.__init__(self, **kwargs)
        self.SensorData: np.ndarray = np.zeros(6)
        self.Load: load_params = load_params()
        self._offset: np.ndarray = np.zeros(6)

    def __del__(self) -> None:
        """
        Cleans up the sensor when it is deleted.

        If the sensor is attached to a robot, it removes the sensor from the robot's attributes.
        """
        if self.Robot is not None:
            self.Robot.FTSensor = None

    @property
    def FT(self) -> WrenchType:
        """
        Returns the 6D force-torque data as a deep copy.

        Returns
        -------
        np.ndarray
            A deep copy of the 6D force-torque data (3 forces and 3 torques).
        """
        return copy.deepcopy(self.SensorData)

    @property
    def F(self) -> Vector3DType:
        """
        Returns the 3D force values as a deep copy.

        Returns
        -------
        np.ndarray
            A deep copy of the 3D force values (the first three values of SensorData).
        """
        return copy.deepcopy(self.SensorData[:3])

    @property
    def Trq(self) -> Vector3DType:
        """
        Returns the 3D torque values as a deep copy.

        Returns
        -------
        np.ndarray
            A deep copy of the 3D torque values (the last three values of SensorData).
        """
        return copy.deepcopy(self.SensorData[3:])

    @abstractmethod
    def GetRawFT(self) -> WrenchType:
        """
        Abstract method to retrieve the raw force-torque data from the sensor.

        This method should be implemented by subclasses to obtain the raw data from the sensor hardware.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    def GetState(self) -> None:
        """
        Retrieves the current state of the sensor by calling `GetFT()`.

        Returns
        -------
        None
            This method updates the cached force-torque data in place.
        """
        self.GetFT()

    def GetFT(self, avg_time: float = 0) -> Optional[WrenchType]:
        """
        Retrieves the force-torque data, optionally averaging over a period of time.

        Parameters
        ----------
        avg_time : float, optional
            The time period over which to average the data. Default is 0.

        Returns
        -------
        np.ndarray or None
            The 6D force-torque data after averaging and applying the offset, or None if no update.
        """
        if avg_time > self.tsamp:
            _n = avg_time // self.tsamp - 1
        else:
            _n = 1
        if (self.simtime() - self._last_update) > (self.tsamp * 0.9):
            _FT = np.zeros(6)
            for i in range(int(_n)):
                _FT += self.GetRawFT()
                if _n > 1:
                    sleep(self.tsamp)
            self.SensorData = _FT / _n - self._offset
            self._last_update = self.simtime()
            return self.SensorData

    def ZeroingFT(self, time: float = 0) -> None:
        """
        Zeroes the force-torque sensor by setting the offset based on the current measurement.

        Parameters
        ----------
        time : float, optional
            Averaging interval in seconds used to estimate the zero offset.

        Returns
        -------
        None
            This method updates the internal sensor offset.
        """
        self._offset = 0 * self._offset
        self._offset = self.GetFT(time)

    def SetLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[Vector3DType] = None, inertia: Optional[ArrayLike] = None, offset: Optional[ArrayLike] = None) -> None:
        """
        Sets or updates the load properties (mass, COM, inertia, and offset).

        Parameters
        ----------
        load : load_params, optional
            Load description object to assign directly.
        mass : float, optional
            Load mass in kilograms.
        COM : Vector3DType, optional
            Load center of mass expressed in the sensor frame.
        inertia : ArrayLike, optional
            Load inertia tensor or inertia parameters.
        offset : ArrayLike, optional
            Sensor offset to apply to the measured wrench.

        Returns
        -------
        None
            This method updates the stored load parameters and optional offset.
        """
        if isinstance(load, load_params):
            self.Load = load
        else:
            if mass is not None:
                self.Load.mass = mass
            if COM is not None:
                self.Load.COM = COM
            if inertia is not None:
                self.Load.inertia = inertia
        if offset is not None:
            _off = vector(offset, dim=6)
            self._offset = _off

    def GetLoad(self) -> load_params:
        """
        Retrieves the current load associated with the sensor.

        Returns
        -------
        load_params
            The current load object attached to the sensor.
        """
        return self.Load

    def SetOffset(self, offset: ArrayLike) -> None:
        """
        Sets the offset for the sensor.

        Parameters
        ----------
        offset : ArrayLike
            Offset wrench applied to subsequent sensor readings.

        Returns
        -------
        None
            This method replaces the current offset.
        """
        _off = vector(offset, dim=6)
        self._offset = _off

    def UpdateOffset(self, offset: ArrayLike) -> None:
        """
        Updates the offset for the sensor by subtracting the provided offset.

        Parameters
        ----------
        offset : ArrayLike
            Offset wrench to subtract from the current offset.

        Returns
        -------
        None
            This method modifies the existing offset in place.
        """
        _off = vector(offset, dim=6)
        self._offset -= _off


class ati_ft_sensor(force_torque_sensor):
    """
    Force-torque sensor interface for ATI sensors using UDP communication.

    This class provides communication with ATI Net F/T sensors over a UDP network connection.

    Attributes
    ----------
    host : str
        Resolved IP address of the sensor.
    port_send : int
        UDP port used to send data to the sensor (default is 49152).
    port_recv : int
        UDP port used to receive data from the sensor (based on the host IP).
    sock : socket.socket
        UDP socket object for communication.
    command : bytes
        Command packet to request force-torque data from the sensor.
    """

    def __init__(self, host: str = "192.168.1.100", **kwargs: Any) -> None:
        """
        Initialize the sensor either by host IP or robot name.

        Parameters
        ----------
        host : str, optional
            The IP address of the ATI sensor.
        **kwargs : Any
            Additional keyword arguments passed to `force_torque_sensor.__init__`.
            If ``robot`` is provided, the sensor is attached to that robot via
            the inherited `AttachTo` logic.

        Returns
        -------
        None
            This constructor initializes the ATI force-torque sensor object in place.
        """
        super().__init__(**kwargs)
        self.host = host

        self.port_send = 49152
        self.port_recv = 10000 + int(self.host.split(".")[-1])
        self.Name = "ATI_FT"

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port_recv))
        self.sock.settimeout(1.0)

        self.command = struct.pack(">HHI", 0x1234, 0x0002, 1)

    def GetRawFT(self) -> np.ndarray:
        """
        Retrieve raw force-torque data from the ATI sensor.

        Returns
        -------
        np.ndarray
            A 6-element numpy array containing the measured forces (Fx, Fy, Fz)
            and torques (Tx, Ty, Tz) in Newtons and Newton-meters respectively.

        Raises
        ------
        TimeoutError
            If the sensor does not respond within the timeout period.
        """
        try:
            # Send request
            self.sock.sendto(self.command, (self.host, self.port_send))

            # Wait for response
            start = time()
            while True:
                try:
                    data, _ = self.sock.recvfrom(36)
                    if len(data) == 36:
                        break
                except socket.timeout:
                    raise TimeoutError("Timeout waiting for ATI sensor response")
                if time() - start > 1.0:
                    raise TimeoutError("No data received from ATI sensor within timeout")

            # Parse 6 signed 32-bit integers from bytes 12 to 36
            ft_raw = struct.unpack(">6i", data[12:36])
            return np.array(ft_raw, dtype=np.float64) / 1e6

        except Exception as e:
            print(f"[{self.Name}] Error reading data: {e}")
            return np.full(6, np.nan)

    def __del__(self) -> None:
        """
        Close the UDP socket owned by the ATI sensor.

        Returns
        -------
        None
            This destructor releases the socket if it was created.
        """
        if hasattr(self, "sock"):
            self.sock.close()


class dummysensor(sensor):
    """
    A dummy sensor class for testing or simulation purposes.

    This class extends the `sensor` class and provides a simple implementation of the `GetRawFT` method.
    It returns a force-torque data vector of zeros, which is useful for testing or as a placeholder in simulations.

    Attributes
    ----------
    Name : str
        The name of the sensor (default is "Dummysensor").
    SensorData : np.ndarray
        A 1D array to store the sensor's force-torque data (6 values: 3 forces and 3 torques).
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the dummy sensor with default parameters.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `sensor.__init__`.

        Returns
        -------
        None
            This constructor initializes the dummy sensor object in place.
        """
        sensor.__init__(self, **kwargs)
        self.Name: str = "Dummysensor"

    def GetRawFT(self) -> WrenchType:
        """
        Implements the `GetRawFT` method to return a force-torque data vector of zeros.

        This method simulates a dummy sensor by providing a 6D zero vector for force and torque values.

        Returns
        -------
        np.ndarray
            A 6D zero vector representing the force and torque data from the sensor.
        """
        self.SensorData: np.ndarray = np.zeros(6)
        return self.SensorData


def issensor(obj: object) -> bool:
    """
    Checks if the given object is an instance of the `sensor` class or its subclasses.

    This function uses the `isinstance()` method to check if the object `obj` is an instance of the `sensor` class,
    which includes instances of any class that inherits from `sensor`.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        `True` if the object is an instance of the `sensor` class or its subclasses, `False` otherwise.
    """
    return isinstance(obj, sensor)


if __name__ == "__main__":
    # Example usage of the classes
    # Create an ATI force-torque sensor
    ft = ati_ft_sensor(host="192.168.1.30")
    ft.Update()
    ft.ZeroingFT(1)
    ft.Update()
    print("Force-Torque Data:", ft.FT)
