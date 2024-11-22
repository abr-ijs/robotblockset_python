from abc import abstractmethod
import numpy as np
from time import perf_counter, sleep
import copy

from  robotblockset.tools import _load, rbs_object, vector


class sensor(rbs_object):
    def __init__(self, **kwargs):
        rbs_object.__init__(self)
        self.Name = "Sensor"
        self._verbose = 1  # verbose level
        self._robot = None  # robot to which sensor is attached
        self._last_update = -100

    def simtime(self):
        return perf_counter()

    @abstractmethod
    def GetState(self):
        pass

    def SetTsamp(self, tsamp):
        self.tsamp = tsamp

    def Update(self):
        self.GetState()

    def AttachTo(self, robot):
        self._robot = robot

    def Detach(self):
        self._robot = None

    def GetAttachedRobot(self):
        if self._robot is None:
            return [None, "None"]
        else:
            return [self._robot, self._robot.Name]


class force_torque_sensor(sensor):
    def __init__(self, **kwargs):
        sensor.__init__(self, **kwargs)
        self.SensorData = np.zeros(6)
        self.Load = _load()
        self._offset = np.zeros(6)

    def __del__(self):
        if self._robot is not None:
            self._robot.FTSensor = None

    @property
    def FT(self):
        return copy.deepcopy(self.SensorData)

    @property
    def F(self):
        return copy.deepcopy(self.SensorData[:3])

    @property
    def Trq(self):
        return copy.deepcopy(self.SensorData[3:])

    @abstractmethod
    def GetRawFT(self):
        pass

    def GetState(self):
        self.GetFT()

    def GetFT(self, avg_time=0):
        if avg_time > self.tsamp:
            _n = avg_time // self.tsamp - 1
        else:
            _n = 1
        if (self.simtime() - self._last_update) > (self.tsamp * 0.0001):
            _FT = np.zeros(6)
            for i in range(_n):
                _FT += self.GetRawFT()
                if _n > 1:
                    sleep(self.tsamp)
            self.SensorData = _FT / _n - self._offset
            self._last_update = self.simtime()
            return self.SensorData

    def ZeroingFT(self, time=0):
        self._offset = self.GetFT(time)

    def SetLoad(self, load=None, mass=None, COM=None, inertia=None, offset=None):
        if isinstance(load, _load):
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

    def GetLoad(self):
        return self.Load

    def SetOffset(self, offset):
        _off = vector(offset, dim=6)
        self._offset = _off

    def UpdateOffset(self, offset):
        _off = vector(offset, dim=6)
        self._offset -= _off


class dummysensor(sensor):
    def __init__(self, **kwargs):
        sensor.__init__(self, **kwargs)
        self.Name = "Dummysensor"

    def GetRawFT(self):
        self.SensorData = np.zeros(6)


def issensor(obj):
    return isinstance(obj, sensor)
