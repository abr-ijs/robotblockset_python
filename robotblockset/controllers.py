from abc import abstractmethod
import numpy as np
from robotblockset.tools import rbs_object

class rbs_controller_type(rbs_object):
    @abstractmethod
    def GoTo_q(self):
        pass

    @abstractmethod
    def GoTo_X(self):
        pass

    def GetJointStiffness(self):
        self.Message("Compliance not suppported", 3)
        return np.ones(self.nj) * 100000

    def SetJointStiffness(self, stiffness, **kwargs):
        self.Message("Compliance not suppported", 3)

    def GetJointDamping(self):
        self.Message("Compliance not suppported", 3)
        return np.ones(self.nj)

    def SetJointDamping(self, damping, **kwargs):
        self.Message("Compliance not suppported", 3)

    def SetJointSoft(self, softness, **kwargs):
        self.Message("Compliance not suppported", 3)

    def SetJointStiff(self):
        self.SetJointSoft(1.0)

    def SetJointCompliant(self):
        self.SetJointSoft(0.0)

    def GetCartesianStiffness(self):
        self.Message("Compliance not suppported", 3)
        return np.ones(6) * 100000

    def SetCartesianStiffness(self, stiffness, **kwargs):
        self.Message("Compliance not suppported", 3)

    def GetCartesianDamping(self):
        self.Message("Compliance not suppported", 3)
        return np.ones(6)

    def SetCartesianDamping(self, damping, **kwargs):
        self.Message("Compliance not suppported", 3)

    def SetCartesianSoft(self, softness, **kwargs):
        self.Message("Compliance not suppported", 3)

    def SetCartesianStiff(self):
        self.SetCartesianSoft(1.0)

    def SetCartesianCompliant(self):
        self.SetCartesianSoft(0.0)

class joint_controller_type(rbs_controller_type):
    def GoTo_X(self):
        raise TypeError("cartesian commands should not be sent to joint controller")
    
class cartesian_controller_type(rbs_controller_type):
    def GoTo_q(self):
        raise TypeError("joint commands should not be sent to cartesian controller")

class compliant_controller_type(rbs_controller_type):
    #@abstractmethod
    pass
