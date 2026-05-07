"""Classes for robots using simulation in Genesis.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Any, Optional, Union

try:
    import genesis as gs
    from genesis.repr_base import RBC
except Exception as e:
    raise e from RuntimeError("Genesis not installed. \nYou can install genesis through pip:\n   pip install genesis-world")
try:
    import torch
except Exception as e:
    raise e from RuntimeError("Torch not installed. \nYou have to install it")

from robotblockset.tools import isvector, vector
from robotblockset.transformations import map_pose
from robotblockset.robot_spec import panda_spec, fr3_spec, lwr_spec, iiwa_spec, ur10_spec, ur10e_spec, ur5_spec, ur5e_spec, crx20_spec, hc20_spec
from robotblockset.robots import robot, MotionResultCodes
from robotblockset.rbs_typing import ArrayLike, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, HomogeneousMatrixType


# Base class for robots using simulation in Python Genesis
class robot_genesis(robot):
    def __init__(self, scene: gs.Scene = None, robot_entity: RBC = None, robot_name: str = "", **kwargs: Any) -> None:
        """Create a robot interface backend by a Genesis scene."""
        robot.__init__(self, **kwargs)
        if scene is None:
            raise ValueError("Genesis scene is not defined")
        self.scene = scene
        if robot_entity is None:
            raise ValueError("Genesis robot is not defined")
        self.robot_entity = robot_entity
        if robot_name == "":
            raise ValueError("Robot name is not defined")
        self.Name = robot_name + "_Genesis"

        self.tsamp = self.scene.dt

        self._connected = True
        self.Message("Robot connected to Genesis", 1)

        self.BaseName = robot_name
        self._control_strategy = "JointPosition"

        kwargs.setdefault("JointNames", None)
        if isinstance(kwargs["JointNames"], str):
            if kwargs["JointNames"].lower() == "auto":
                self.JointNames = [joint.name for joint in self.robot_entity.joints][: self.nj]
            elif kwargs["JointNames"].lower() == "gen":
                self.JointNames = []
                for i in range(self.nj):
                    self.JointNames.append(self.BaseName + "_joint" + str(i + 1))
            else:
                raise ValueError(f"Argument 'JointNames':{kwargs['JointNames']} is invalid. Only `auto` or `gen` are accepted.")
        elif kwargs["JointNames"] is None:
            if hasattr(self, "joint_names"):
                self.JointNames = [self.BaseName + "_" + jnt for jnt in self.joint_names]
            else:
                self.JointNames = []
                for i in range(self.nj):
                    self.JointNames.append(self.BaseName + "_joint" + str(i + 1))
        else:
            self.JointNames = kwargs["JointNames"]

        self._JointPosHandles = [self.robot_entity.get_joint(joint).dofs_idx_local[0] for joint in self.JointNames]
        self._JointVelHandles = self._JointPosHandles
        self._ActuatorHandles = self._JointPosHandles

        self.Init()

    def Init(self) -> None:
        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    def simtime(self) -> float:
        """
        Returns the current simulation timefrom Genesis simulator.

        Returns
        -------
        float
            The current simulation time in seconds since an arbitrary point (see ResetTime).
        """
        return self.scene.t * self.scene.dt

    def _sleep(self, time: float) -> None:
        """
        Pause execution for the given duration.

        This method uses the Genesis simulation time to simulate pause
        for the specified duration.
        """
        _nsamp = round(time / self.scene.dt)
        for i in range(_nsamp):
            self.scene.step()

    def GetState(self) -> None:
        """Update robot state from Genesis data buffers."""
        self._robottime = self.scene.t

        self._actual.q = gs.utils.tensor_to_array(self.robot_entity.get_dofs_position())[self._JointPosHandles]
        self._actual.qdot = gs.utils.tensor_to_array(self.robot_entity.get_dofs_velocity())[self._JointVelHandles]
        x, J = self.Kinmodel()
        self._actual.x = x
        self._actual.v = J @ self._actual.qdot
        self._actual.FT = np.zeros(6)
        self._actual.trq = gs.utils.tensor_to_array(self.robot_entity.get_dofs_force())[self._JointVelHandles]
        self._tt = self.simtime()
        self._last_update = self.simtime()

    def Restart(self, qpos: Optional[ArrayLike] = None, u: Optional[ArrayLike] = None, reset: bool = True, keyframe: Optional[int] = None) -> None:
        """Restart the simulation.

        Reset the simulation and optionally set joint positions/inputs."""
        pos = gs.utils.tensor_to_array(self.robot_entity.get_dofs_position())
        if qpos is not None:
            if isvector(qpos, dim=pos.shape[0]):
                self.robot_entity.set_dofs_position(torch.from_numpy(qpos).to(gs.device))
            elif isvector(qpos, dim=self.nj):
                self.robot_entity.set_dofs_position(torch.from_numpy(qpos).to(gs.device), self._JointPosHandles)
        if u is not None:
            if isvector(u, dim=self.nj):
                self.robot_entity.set_dofs_position(torch.from_numpy(u).to(gs.device), self._JointPosHandles)
                self.SendRobot_u(u)
        self.robot_entity.set_dofs_velocity(np.zeros(pos.shape[0]))
        self.scene.step()
        self.ResetCurrentTarget()
        self.ResetTime()

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """Update joint positions and wait

        This method sets the commanded joint positions (`q`), velocities (`qdot`), and torques (`trq`),
        then sends them to the robot and waits for the specified time (`wait`).

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        qdot : JointVelocityType, optional
            Desired joint velocities (nj,).
        trq : JointTorqueType, optional
            Desired joint torques (nj,).
        wait : float, optional
            Time to wait (in seconds) after commanding the robot to move.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        if qdot is None:
            qdot = np.zeros(self.nj)
        else:
            qdot = vector(qdot, dim=self.nj)
        if trq is None:
            trq = np.zeros(self.nj)
        else:
            trq = vector(trq, dim=self.nj)
        if wait is None or wait < self.tsamp:
            wait = self.tsamp

        self.SendRobot_u(q)
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        if np.floor(self._command.mode) == 1:
            x, J = self.Kinmodel(q)
            self._command.x = x
            self._command.v = J @ qdot
        _nsamp = round(wait / self.tsamp)
        self.GetState()
        self.Update()
        for _ in range(_nsamp):
            self._sleep(self.tsamp)
            self.GetState()
            self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value

    def SendRobot_u(self, u: JointConfigurationType) -> None:
        """Send joint commands to Genesis actuators."""
        self._command.u = u
        if self.isReady:
            self.robot_entity.control_dofs_position(u, self._ActuatorHandles[: self.nj])

    def SetRobotPose(self, x: Union[Pose3DType, HomogeneousMatrixType]) -> None:
        """
        Set the robot base pose.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            The pose of the base (7,) or (4, 4).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the base pose shape is not recognized.
        """
        self.SetBasePose(x)
        p = map_pose(T=self.TBase, out="p")
        Q = map_pose(T=self.TBase, out="Q")
        self.robot_entity.set_pos(p)
        self.robot_entity.set_quat(Q)
        self.ResetCurrentTarget()


# Robot classes for specific robot models in Genesis
class panda(robot_genesis, panda_spec):
    def __init__(self, scene: gs.Scene = None, robot_entity: RBC = None, robot_name: str = "panda", **kwargs: Any) -> None:
        """Create a Panda robot in Genesis."""
        panda_spec.__init__(self)
        robot_genesis.__init__(self, scene=scene, robot_entity=robot_entity, robot_name=robot_name, **kwargs)


class fr3(robot_genesis, fr3_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "fr3", **kwargs: Any) -> None:
        """Create an FR3 robot in Genesis."""
        fr3_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class lwr(robot_genesis, lwr_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "LWR", **kwargs: Any) -> None:
        """Create an LWR robot in Genesis."""
        lwr_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class iiwa(robot_genesis, iiwa_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "iiwa", **kwargs: Any) -> None:
        """Create a KUKA iiwa robot in Genesis."""
        iiwa_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class ur10(robot_genesis, ur10_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "ur10", **kwargs: Any) -> None:
        """Create a UR10 robot in Genesis."""
        ur10_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class ur10e(robot_genesis, ur10e_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "ur10e", **kwargs: Any) -> None:
        """Create a UR10e robot in Genesis."""
        ur10e_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class ur5(robot_genesis, ur5_spec):
    def __init__(self, scene: gs.Scene = None, robot_entity: RBC = None, robot_name: str = "ur5", **kwargs: Any) -> None:
        """Create a UR5 robot in Genesis."""
        ur5_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class ur5e(robot_genesis, ur5e_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "ur5e", **kwargs: Any) -> None:
        """Create a UR5e robot in Genesis."""
        ur5_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class crx20(robot_genesis, crx20_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "CRX20", **kwargs: Any) -> None:
        """Create a CRX20 robot in Genesis."""
        crx20_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)


class hc20(robot_genesis, hc20_spec):
    def __init__(self, scene: gs.Scene, robot_entity: RBC, robot_name: str = "hc20", **kwargs: Any) -> None:
        """Create an HC20 robot in Genesis."""
        hc20_spec.__init__(self)
        robot_genesis.__init__(self, scene, robot_entity, robot_name, **kwargs)
