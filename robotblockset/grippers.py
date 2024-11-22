from abc import abstractmethod
from time import perf_counter

from  robotblockset.tools import rbs_object


class gripper(rbs_object):
    def __init__(self, **kwargs):
        rbs_object.__init__(self)
        self.Name = "Gripper"
        self._verbose = 1  # verbose level
        self._state = -1  # gripper state
        self._robot = None  # robot to which gripper is attached

    def __del__(self):
        if self._robot is not None:
            self._robot.Gripper = None

    def simtime(self):
        return perf_counter()

    @abstractmethod
    def Move(self, width, **kwargs):
        pass

    def Open(self, **kwargs):
        _succx = self.Move(self._width_max, **kwargs)
        self._state = 0
        return _succx

    def Close(self, **kwargs):
        _succx = self.Move(0, **kwargs)
        self._state = 1
        return _succx

    def Grasp(self, **kwargs):
        if "width" in kwargs:
            _width = max(min(kwargs["width"], self._width_max), 0)
            del kwargs["width"]
        else:
            _width = 0
        _succx = self.Move(_width, **kwargs)
        self._state = 1
        return _succx

    def Homing(self, **kwargs):
        kwargs.setdefault("check", True)
        return self.Open(**kwargs)

    def isOpened(self):
        return self._state == 0

    def isClosed(self):
        return self._state == 1

    def GetState(self):
        if self._state == 0:
            return "Opened"
        elif self._state == 1:
            return "Closed"
        else:
            return "Undefined"

    def AttachTo(self, robot):
        self._robot = robot

    def Detach(self):
        self._robot = None

    def GetAttachedRobot(self):
        if self._robot is None:
            return [None, "None"]
        else:
            return [self._robot, self._robot.Name]


class dummygripper(gripper):
    def __init__(self, **kwargs):
        gripper.__init__(self, **kwargs)
        self.Name = "DummyGripper"
        self._state = -1

    def Open(self, **kwargs):
        self._state = 0
        return 1

    def Close(self, **kwargs):
        self._state = 1
        return 1

    def Grasp(self, **kwargs):
        self._state = 0
        return 1

    def Move(self, width, **kwargs):
        self._state = -1
        return 1

    def Homing(self, **kwargs):
        self._state = 0
        return 1


def isgripper(obj):
    return isinstance(obj, gripper)
