"""Robot interfaces for synchronous Python MuJoCo scenes.

This module provides RobotBlockSet robot backends built on
`scene_pymujoco_sim`, including the generic synchronous PyMuJoCo robot
interface, concrete robot wrappers, and an internal joint-control variant.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Any, List, Optional, Sequence, Union

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

from robotblockset.mujoco.scene_pymujoco_sim import mujoco_scene
from robotblockset.mujoco.tools_pymujoco import get_joints_under_body, get_actuators_for_joints
from robotblockset.tools import isvector, vector
from robotblockset.transformations import map_pose, r2q, checkx, world2frame
from robotblockset.robot_spec import panda_spec, fr3_spec, lwr_spec, iiwa_spec, ur10_spec, ur10e_spec, ur5_spec, ur5e_spec, crx20_spec, hc20_spec, z1_spec, b2_spec
from robotblockset.robots import robot, MotionResultCodes, CommandModeCodes
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, QuaternionType, RotationMatrixType, Vector3DType


# Base class for robots using simulation in Python MuJoCo
class robot_pymujoco(robot):
    """
    Synchronous PyMuJoCo-backed robot interface operating on a :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`.

    Attributes
    ----------
    scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`
        MuJoCo scene object that owns the model and data.
    BaseName : str
        Base model name used to derive joint, actuator, and sensor names.
    JointNames : list[str]
        Ordered list of robot joint names.
    ActuatorNames : list[str]
        Ordered list of actuator names used for joint commands.
    MocapNames : list[str]
        Names of MuJoCo mocap bodies associated with the scene.

    """

    def __init__(self, robot_name: str, scene: mujoco_scene = None, tsamp: float = 0.0, **kwargs: Any) -> None:
        """Create a robot interface backed by a MuJoCo scene.

        Parameters
        ----------
        robot_name : str
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        tsamp : float, optional
            Requested control sampling period in seconds.
        **kwargs : Any
            Additional keyword arguments.

        Notes
        -----
        When constrructing objects of this class, the following keyword arguments are supported for explicit configuration of model element names. If not provided, default names based on the platform name are used.:
            `JointNames` : list[str] or str, optional
                Explicit joint names, ``"auto"`` to use model-based names, or
                ``"gen"`` to generate names from `robot_name`.
            `ActuatorNames` : list[str] or str, optional
                Explicit actuator names, or ``"auto"`` to use model-based names.
            `FlangeName` : str, optional
                Name of the flange or end-effector body/site.
            `TCPName` : str, optional
                Name of the tool center point body/site.
            `SensorJointPosNames` : list[str], optional
                Sensor names used to read joint positions.
            `SensorJointVelNames` : list[str], optional
                Sensor names used to read joint velocities.
            `SensorPosName` : str, optional
                Sensor name used to read Cartesian position.
            `SensorOriName` : str, optional
                Sensor name used to read Cartesian orientation.
            `SensorLinVelName` : str, optional
                Sensor name used to read Cartesian linear velocity.
            `SensorRotVelName` : str, optional
                Sensor name used to read Cartesian angular velocity.
            `SensorForceName` : str, optional
                Sensor name used to read force measurements.
            `SensorTorqueName` : str, optional
                Sensor name used to read torque measurements.

        Raises
        ------
        Exception
            If joint or actuator names do not resolve uniquely in the MJCF model.

        Returns
        -------
        None
            This constructor initializes the synchronous PyMuJoCo robot interface in place.
        """
        robot.__init__(self, **kwargs)
        if scene is None:
            raise ValueError("MuJoCo scene is not defined")
        self.scene = scene
        self._connected = True
        self.Name = robot_name + "_PyMuJoCo"
        self.Message("Robot connected to MuJoCo", 1)

        self.BaseName = robot_name
        self._control_strategy = "JointPosition"

        if tsamp is None or tsamp < self.scene.model.opt.timestep:
            self.tsamp = self.scene.model.opt.timestep

        kwargs.setdefault("JointNames", None)
        if isinstance(kwargs["JointNames"], str):
            if kwargs["JointNames"].lower() == "auto":
                _joint_names, _joint_ids = get_joints_under_body(self.scene.model, self.BaseName)
                self.JointNames = _joint_names[: self.nj]
                _joint_ids = _joint_ids[: self.nj]
            elif kwargs["JointNames"].lower() == "ori":
                if hasattr(self, "joint_names"):
                    self.JointNames = self.joint_names
                else:
                    raise ValueError("Robot specification does not define joint names.")
            else:
                raise ValueError(f"Argument 'JointNames':{kwargs['JointNames']} is invalid. Only `auto` or `ori` are accepted.")
        elif kwargs["JointNames"] is None:
            if hasattr(self, "joint_names") and self.scene.model.joint(0).name == self.BaseName + "_" + self.joint_names[0]:
                self.JointNames = [self.BaseName + "_" + jnt for jnt in self.joint_names]
            else:
                self.JointNames = []
                for i in range(self.nj):
                    self.JointNames.append(self.BaseName + "_joint" + str(i + 1))
        else:
            self.JointNames = kwargs["JointNames"]

        kwargs.setdefault("ActuatorNames", None)
        if isinstance(kwargs["ActuatorNames"], str):
            if kwargs["ActuatorNames"].lower() == "auto":
                _joint_names, _joint_ids = get_joints_under_body(self.scene.model, self.BaseName)
                _joint_ids = _joint_ids[: self.nj]
                self.ActuatorNames, _ = get_actuators_for_joints(self.scene.model, joints=_joint_ids)
            elif kwargs["ActuatorNames"].lower() == "ori":
                if hasattr(self, "actuator_names"):
                    self.ActuatorNames = self.actuator_names
                else:
                    raise ValueError("Robot specification does not define actuator names.")
            else:
                raise ValueError(f"Argument 'ActuatorNames':{kwargs['ActuatorNames']} is invalid. Only `auto` or `ori`¸ are accepted.")
        elif kwargs["ActuatorNames"] is None:
            if hasattr(self, "actuator_names"):
                self.ActuatorNames = [self.BaseName + "_" + act for act in self.actuator_names]
            else:
                self.ActuatorNames = []
                for i in range(self.nj):
                    self.ActuatorNames.append(self.BaseName + "_actuator" + str(i + 1))
        else:
            self.ActuatorNames = kwargs["ActuatorNames"]

        kwargs.setdefault("FlangeName", None)
        if kwargs["FlangeName"] is not None:
            self.FlangeName = kwargs["FlangeName"]
        else:
            self.FlangeName = self.BaseName + "_flange"

        kwargs.setdefault("TCPName", None)
        if kwargs["TCPName"]:
            self.TCPName = kwargs["TCPName"]
        else:
            self.TCPName = self.BaseName + "_TCP"

        kwargs.setdefault("SensorJointPosNames", None)
        if kwargs["SensorJointPosNames"] is not None:
            self.SensorJointPosNames = kwargs["SensorJointPosNames"]
        else:
            self.SensorJointPosNames = []
            for i in range(self.nj):
                self.SensorJointPosNames.append(self.BaseName + "_pos_joint" + str(i + 1))

        kwargs.setdefault("SensorJointVelNames", None)
        if kwargs["SensorJointVelNames"] is not None:
            self.SensorJointVelNames = kwargs["SensorJointVelNames"]
        else:
            self.SensorJointVelNames = []
            for i in range(self.nj):
                self.SensorJointVelNames.append(self.BaseName + "_vel_joint" + str(i + 1))

        kwargs.setdefault("SensorPosName", None)
        if kwargs["SensorPosName"] is not None:
            self.SensorPosName = kwargs["SensorPosName"]
        else:
            self.SensorPosName = self.BaseName + "_pos"

        kwargs.setdefault("SensorOriName", None)
        if kwargs["SensorOriName"] is not None:
            self.SensorOriName = kwargs["SensorOriName"]
        else:
            self.SensorOriName = self.BaseName + "_ori"

        kwargs.setdefault("SensorLinVelName", None)
        if kwargs["SensorLinVelName"] is not None:
            self.SensorLinVelName = kwargs["SensorLinVelName"]
        else:
            self.SensorLinVelName = self.BaseName + "_v"

        kwargs.setdefault("SensorRotVelName", None)
        if kwargs["SensorRotVelName"] is not None:
            self.SensorRotVelName = kwargs["SensorRotVelName"]
        else:
            self.SensorRotVelName = self.BaseName + "_w"

        kwargs.setdefault("SensorForceName", None)
        if kwargs["SensorForceName"] is not None:
            self.SensorForceName = kwargs["SensorForceName"]
        else:
            self.SensorForceName = self.BaseName + "_force"

        kwargs.setdefault("SensorTorqueName", None)
        if kwargs["SensorTorqueName"] is not None:
            self.SensorTorqueName = kwargs["SensorTorqueName"]
        else:
            self.SensorTorqueName = self.BaseName + "_torque"

        self.MocapHandles = [None] * self.scene.model.nmocap

        if self.scene.model.nu > 0:
            self._ctrl = self.scene.data.ctrl.copy()

        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo handles and cached sensor indices.

        Notes
        -----
        The method resolves joint, actuator, site, sensor, and mocap handles
        and then initializes the RobotBlockSet state.
        """
        self._BaseHandle = self.scene.model.body(self.BaseName).id
        self.UpdateRobotBaseFromModel()

        self._SiteNamesList = [self.scene.model.site(i).name for i in range(self.scene.model.nsite)]
        if self.FlangeName in self._SiteNamesList and self.TCPName in self._SiteNamesList:
            i1 = self.scene.model.site(self.FlangeName).id
            i2 = self.scene.model.site(self.TCPName).id
            site_pos = np.array(self.scene.data.site_xpos.copy())
            site_mat = np.array(self.scene.data.site_xmat.copy()).reshape((-1, 3, 3))
            pEE = site_pos[i1]
            REE = site_mat[i1]
            pHand = site_pos[i2]
            RHand = site_mat[i2]
            self.TCP = map_pose(R=REE.T @ RHand, p=REE.T @ (pHand - pEE), out="T")

        self._JointNamesList = [self.scene.model.joint(i).name for i in range(self.scene.model.njnt)]
        self._ActuatorNamesList = [self.scene.model.actuator(i).name for i in range(self.scene.model.nu)]
        self._JointPosHandles = np.array(
            [self.scene.model.jnt_qposadr[self.scene.model.joint(self.JointNames[i]).id] if self.JointNames[i] in self._JointNamesList else -1 for i in range(self.nj)],
            dtype=np.int32,
        )
        self._JointVelHandles = np.array(
            [self.scene.model.jnt_dofadr[self.scene.model.joint(self.JointNames[i]).id] if self.JointNames[i] in self._JointNamesList else -1 for i in range(self.nj)],
            dtype=np.int32,
        )
        self._ActuatorHandles = np.array([self.scene.model.actuator(self.ActuatorNames[i]).id if self.ActuatorNames[i] in self._ActuatorNamesList else -1 for i in range(self.nj)], dtype=np.int32)
        if np.any(self._JointPosHandles == -1) or np.unique(self._JointPosHandles).size != self._JointPosHandles.size:
            raise Exception("Check naming of joints in MJCF model")
        if np.any(self._ActuatorHandles == -1) or np.unique(self._ActuatorHandles).size != self._ActuatorHandles.size:
            raise Exception("Check naming of actuators in MJCF model")
        self._SensorNamesList = [self.scene.model.sensor(i).name for i in range(self.scene.model.nsensor)]

        self._SensorJointPosHandles = -np.ones(self.nj, dtype=np.int32)
        for i in range(self.nj):
            if self.SensorJointPosNames[i] in self._SensorNamesList:
                idx = self.scene.model.sensor(self.SensorJointPosNames[i]).id
                adr = self.scene.model.sensor_adr[idx]
                dim = self.scene.model.sensor_dim[idx]
                if dim != 1:
                    raise Exception("Wrong joint sensor in model")
                self._SensorJointPosHandles[i] = adr
            else:
                self._SensorJointPosHandles = -np.ones(self.nj, dtype=np.int32)
                break

        self._SensorJointVelHandles = -np.ones(self.nj, dtype=np.int32)
        for i in range(self.nj):
            if self.SensorJointVelNames[i] in self._SensorNamesList:
                idx = self.scene.model.sensor(self.SensorJointVelNames[i]).id
                adr = self.scene.model.sensor_adr[idx]
                dim = self.scene.model.sensor_dim[idx]
                if dim != 1:
                    raise Exception("Wrong joint sensor in model")
                self._SensorJointVelHandles[i] = adr
            else:
                self._SensorJointVelHandles = -np.ones(self.nj, dtype=np.int32)
                break

        if self.SensorPosName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorPosName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorPosHandles = list(range(adr, adr + dim))
        else:
            self._SensorPosHandles = None

        if self.SensorOriName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorOriName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorOriHandles = list(range(adr, adr + dim))
        else:
            self._SensorOriHandles = None

        if self.SensorLinVelName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorLinVelName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorLinVelHandles = list(range(adr, adr + dim))
        else:
            self._SensorLinVelHandles = None

        if self.SensorRotVelName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorRotVelName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorRotVelHandles = list(range(adr, adr + dim))
        else:
            self._SensorRotVelHandles = None

        if self.SensorForceName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorForceName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorForceHandles = list(range(adr, adr + dim))
        else:
            self._SensorForceHandles = None

        if self.SensorTorqueName in self._SensorNamesList:
            idx = self.scene.model.sensor(self.SensorTorqueName).id
            adr = self.scene.model.sensor_adr[idx]
            dim = self.scene.model.sensor_dim[idx]
            self._SensorTorqueHandles = list(range(adr, adr + dim))
        else:
            self._SensorTorqueHandles = None

        self.MocapHandles = np.where(self.scene.model.body_mocapid >= 0)[0]
        self.MocapNames = [self.scene.model.body(i).name for i in range(self.scene.model.nbody) if self.scene.model.body_mocapid[i] >= 0]

        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    @property
    def t(self) -> float:
        """
        Get the elapsed time since the robot was initiated.

        Returns
        -------
        float
            Time difference in seconds.
        """
        return self.simtime() - self._tt0

    @property
    def c(self) -> JointTorqueType:
        """
        Coriolis, centrifugal, gravitational joint torques

        Returns
        -------
        JointTorqueType
            Joint force/torque bias (gravity, ...).
        """
        return self.scene.data.qfrc_bias[self._JointVelHandles]

    @property
    def H(self) -> np.ndarray:
        """
        Inertia matrix H(q) in joint space.

        Returns
        -------
        np.ndarray:
            Symmetric inertia matrix in joint space.
        """
        sys_H_inv = np.zeros((self.scene.model.nv, self.scene.model.nv))

        mujoco.mj_solveM(self.scene.model, self.scene.data, sys_H_inv, np.eye(self.scene.model.nv))
        H_inv = sys_H_inv[np.ix_(self._JointVelHandles, self._JointVelHandles)]

        if abs(np.linalg.det(H_inv)) >= 1e-2:
            _H = np.linalg.inv(H_inv)
        else:
            _H = np.linalg.pinv(H_inv, rcond=1e-2)
        return _H

    @property
    def M(self) -> np.ndarray:
        """
        Inertia matrix H(q) in task space.

        Returns
        -------
        np.ndarray:
            Symmetric inertia matrix in task space.
        """
        M_inv = self.J @ np.linalg.inv(self.H) @ self.J.T

        if abs(np.linalg.det(M_inv)) >= 1e-2:
            _M = np.linalg.inv(M_inv)
        else:
            _M = np.linalg.pinv(M_inv, rcond=1e-2)
        return _M

    def simtime(self) -> float:
        """
        Returns the current simulation timefrom MuJoCO simulator.

        Returns
        -------
        float
            The current simulation time in seconds since an arbitrary point (see ResetTime).
        """
        return self.scene.data.time  # perf_counter()

    def _sleep(self, time: float) -> None:
        """
        Pause execution for the given duration.

        Parameters
        ----------
        time : float
            Duration to wait in seconds of MuJoCo simulation time.

        This method uses the MuJoCO simulation time to simulate pause
        for the specified duration.
        """
        _nsamp = round(time / self.scene.model.opt.timestep)
        for i in range(_nsamp):
            self.scene.mj_step()

    def GetState(self) -> None:
        """
        Update robot state from MuJoCo data buffers.

        Notes
        -----
        Joint, Cartesian, force-torque, and object-following state are updated
        from the current scene buffers.
        """
        sensors = self.scene.data.sensordata.copy()
        self._robottime = self.scene.data.time

        if all(self._SensorJointPosHandles >= 0):
            self._actual.q = np.take(sensors, self._SensorJointPosHandles)
        else:
            self._actual.q = np.take(self.scene.data.qpos.copy(), self._JointPosHandles)
        if all(self._SensorJointVelHandles >= 0):
            self._actual.qdot = np.take(sensors, self._SensorJointVelHandles)
        else:
            self._actual.qdot = np.take(self.scene.data.qvel.copy(), self._JointVelHandles)

        if (self._SensorPosHandles is not None) and (self._SensorOriHandles is not None):
            _x = checkx(np.take(sensors, self._SensorPosHandles + self._SensorOriHandles))
            self._actual.x = self.WorldToBase(_x)
        else:
            x, J = self.Kinmodel()
            self._actual.x = x
        if (self._SensorLinVelHandles is not None) and (self._SensorRotVelHandles is not None):
            _v = np.take(sensors, self._SensorLinVelHandles + self._SensorRotVelHandles)
            self._actual.v = self.WorldToBase(_v, typ="Twist")
        else:
            self._actual.v = self.Jacobi() @ self._actual.qdot
        if (self._SensorForceHandles is not None) and (self._SensorTorqueHandles is not None):
            self._actual.FT = np.take(sensors, self._SensorForceHandles + self._SensorTorqueHandles)
        else:
            self._actual.FT = np.zeros(6)
            self._actual.trq = np.zeros(self.nj)

        self._actual.trq = np.zeros(self.nj)
        self._actual.trqExt = np.zeros(self.nj)

        if self.EEFixed:
            self.TObject = map_pose(x=self.BaseToWorld(self._actual.x), out="T")

        self._tt = self.simtime()
        self._last_update = self.simtime()

    def isActive(self) -> bool:
        """
        Check whether the simulator is running.

        Returns
        -------
        bool
            ``True`` if the scene is not paused.
        """
        return not self.scene.pause

    def Restart(self, qpos: Optional[ArrayLike] = None, u: Optional[ArrayLike] = None, reset: bool = True, keyframe: Optional[int] = None) -> None:
        """
        Restart the simulation.

        Parameters
        ----------
        qpos : ArrayLike, optional
            Full generalized-position vector to apply after reset.
        u : ArrayLike, optional
            Joint command vector to apply after reset.
        reset : bool, optional
            If ``True``, reset the simulator before applying state updates.
        keyframe : int, optional
            Keyframe index used for simulator reset.
        """
        if reset:
            self.scene.mj_reset(keyframe)
        if qpos is not None:
            if isvector(qpos, dim=self.scene.model.nq):
                self.scene.data.qpos = qpos
        if u is not None:
            if isvector(u, dim=self.nj):
                self.scene.data.qpos[self._JointPosHandles] = u
                self.SendRobot_u(u)
        self.scene.data.qvel = np.zeros(self.scene.model.nv)
        mujoco.mj_forward(self.scene.model, self.scene.data)
        self.ResetCurrentTarget()
        self.ResetTime()

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Update joint positions and wait.

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

        Notes
        -----
        The method sends the joint command to MuJoCo, updates the RobotBlockSet
        command state, and advances the synchronous simulator for the requested
        wait time.
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
        if np.floor(self._command.mode) == CommandModeCodes.JOINT.value:
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

    def SetStrategy(self, strategy: str) -> None:
        """
        Set the control strategy.

        Parameters
        ----------
        strategy : str
            Requested control strategy.

        Notes
        -----
        The synchronous PyMuJoCo wrapper currently keeps a single
        joint-position strategy and accepts this method for compatibility.
        """
        pass

    def SendRobot_u(self, u: JointConfigurationType) -> None:
        """
        Send joint commands to MuJoCo actuators.

        Parameters
        ----------
        u : JointConfigurationType
            Joint command vector.
        """
        self._command.u = u
        if self.isReady:
            self._ctrl = self.scene.data.ctrl.copy()
            for i, x in zip(self._ActuatorHandles, u):
                self._ctrl[i] = x
            self.scene.data.ctrl = self._ctrl

    def SendCtrl(self, u: ArrayLike) -> None:
        """
        Send a full control vector to the MuJoCo actuators.

        Parameters
        ----------
        u : ArrayLike
            Full actuator control vector.
        """
        self._command.u = np.take(u, self._ActuatorHandles)
        if self.isReady:
            self._ctrl = u
            self.scene.data.ctrl = self._ctrl

    def SendAuxCtrl(self, idx: Sequence[int], val: ArrayLike) -> None:
        """
        Update selected actuator controls by index.

        Parameters
        ----------
        idx : Sequence[int]
            Actuator indices to update.
        val : ArrayLike
            Control values to assign.
        """
        if self.scene.model.nu > 0:
            self._ctrl = self.scene.data.ctrl.copy()
            for i, x in zip(idx, val):
                self._ctrl[i] = x
            self.scene.data.ctrl = self._ctrl

    def GetAuxJointPos(self, ide: Sequence[int]) -> JointConfigurationType:
        """
        Return joint positions for auxiliary joints by index.

        Parameters
        ----------
        ide : Sequence[int]
            Joint-position indices to read.

        Returns
        -------
        JointConfigurationType
            Joint positions for the selected indices.
        """
        return np.take(self.data.qpos, ide)

    def GetSensor(self, ide: Union[str, int] = None) -> np.ndarray:
        """
        Read sensor data by name or id, or return the full sensor array.

        Parameters
        ----------
        ide : str | int, optional
            Optional sensor name or id.

        Returns
        -------
        np.ndarray
            Selected sensor data or full sensor data.
        """
        if ide is not None:
            return self.scene.data.sensor(ide).data
        else:
            return self.scene.data.sensordata

    def GetContacts(self, ide: Union[str, int] = None) -> Optional[np.ndarray]:
        """
        Return contact forces for a geom or for all contacts.

        Parameters
        ----------
        ide : str | int, optional
            Optional geom name or id.

        Returns
        -------
        np.ndarray | None
            Contact forces in world coordinates, or ``None`` if no contacts are
            present.
        """
        contacts = self.scene.data.contact
        ncon = len(contacts)
        if ncon > 0:
            if ide is not None:
                if isinstance(ide, str):
                    idx = self.scene.geom(ide).id
                else:
                    idx = ide
                ii = list(set(np.where(contacts.geom1 == idx)[0]) | set(np.where(contacts.geom2 == idx)[0]))
            else:
                ii = list(range(ncon))
            Fx = np.empty((len(ii), 3))
            forcetorque = np.zeros(6)
            for i, ix in enumerate(ii):
                mujoco.mj_contactForce(self.scene.model, self.scene.data, ix, forcetorque)
                R = contacts.frame[i].reshape(3, 3)
                F = forcetorque[:3]
                Fx[i, :] = np.squeeze(R @ F)
            return Fx
        else:
            return None

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
        if self.BaseName in self.MocapNames:
            self.SetMocapPose(self.BaseName, x)
            self.ResetCurrentTarget()

    def SetMocapPose(self, ide: Union[str, int], x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]) -> None:
        """Set pose of a mocap body

        Parameters
        ----------
        ide : Union[str, int]
            mocap body name or id
        x : Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]
            mocap pose

        Raises
        ------
        ValueError
            Wrong pose shape

        Note
        ----
        When mocap body names are used, mocap bodies have to be first bodies in the model!
        """
        if self.scene.model.nmocap > 0:
            mocap_pos = self.scene.data.mocap_pos.copy()
            mocap_quat = self.scene.data.mocap_quat.copy()
            if isinstance(ide, str):
                idx = self.MocapNames.index(ide)
            else:
                idx = ide
            x = self.spatial(x)
            if x.shape == (4, 4):
                xx = map_pose(T=x)
                mocap_pos[idx, :] = xx[:3]
                mocap_quat[idx, :] = xx[3:]
            elif x.shape == (3, 3):
                xx = r2q(x)
                mocap_quat[idx, :] = xx
            elif isvector(x, dim=7):
                mocap_pos[idx, :] = x[:3]
                mocap_quat[idx, :] = x[3:]
            elif isvector(x, dim=3):
                mocap_pos[idx, :] = x
            elif isvector(x, dim=4):
                mocap_quat[idx, :] = x
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
            self.scene.data.mocap_pos = mocap_pos
            self.scene.data.mocap_quat = mocap_quat

    def GetMocapPose(self, ide: Union[str, int], out: str = "x") -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]:
        """
        Return mocap body pose in the requested output format.

        Parameters
        ----------
        ide : str | int
            Mocap body name or id.
        out : str, optional
            Output pose format.

        Returns
        -------
        Pose3DType | HomogeneousMatrixType | Vector3DType | RotationMatrixType | None
            Mocap body pose in the requested format, or ``None`` if the mocap
            body could not be resolved.
        """
        if self.scene.model.nmocap > 0:
            if isinstance(ide, str):
                idx = self.MocapNames.index(ide)
                if idx < 0:
                    self.Message(f"No mocap body with name '{ide}' exits", 2)
                    return None
            else:
                idx = ide
            return map_pose(p=self.scene.data.mocap_pos[idx], Q=self.scene.data.mocap_quat[idx], out=out)

    def GetObjectData(self, ide: Union[str, int]) -> Any:
        """
        Return raw MuJoCo body data for a body name or id.

        Parameters
        ----------
        ide : str | int
            Body name or id.

        Returns
        -------
        Any
            MuJoCo body data object.
        """
        return self.scene.data.body(ide).copy()

    def GetObjectPose(self, typ: str, ide: Union[str, int], out: str = "x") -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]:
        """
        Return the pose of a body, site, or geom in the requested format.

        Parameters
        ----------
        typ : str
            Object type; one of ``"body"``, ``"site"``, or ``"geom"``.
        ide : str | int
            Object name or id.
        out : str, optional
            Output pose format.

        Returns
        -------
        Pose3DType | HomogeneousMatrixType | Vector3DType | RotationMatrixType | None
            Object pose in the requested format, or ``None`` if ``typ`` is not
            supported.
        """
        if typ in set(["body", "site", "geom"]):
            if isinstance(ide, str):
                pos = eval(f"self.scene.data.{typ}('{ide}').xpos.copy()")
                R = np.array(eval(f"self.scene.data.{typ}('{ide}').xmat.copy()")).reshape(3, 3)
            else:
                pos = eval(f"self.scene.data.{typ}({ide}).xpos.copy()")
                R = np.array(eval(f"self.scene.data.{typ}({ide}).xmat.copy()")).reshape(3, 3)
            return map_pose(p=pos, R=R, out=out)

    def SetObjectPose(self, ide: Union[str, int], x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]) -> None:
        """
        Set a MuJoCo body pose from a spatial representation.

        Parameters
        ----------
        ide : str | int
            Body name or id.
        x : Pose3DType | HomogeneousMatrixType | RotationMatrixType | Vector3DType | QuaternionType | ArrayLike
            Body pose, position, orientation, or homogeneous transform.

        Raises
        ------
        ValueError
            If the pose shape is not supported.
        """
        x = self.spatial(x)
        if x.shape == (4, 4):
            xx = map_pose(T=x)
            pos = xx[:3]
            quat = xx[3:]
        elif x.shape == (3, 3):
            xx = r2q(x)
            quat = xx
        elif isvector(x, dim=7):
            pos = x[:3]
            quat = x[3:]
        elif isvector(x, dim=3):
            pos = x
        elif isvector(x, dim=4):
            quat = x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")
        self.scene.data.body(ide).xpos = pos
        self.scene.data.body(ide).xquat = quat

    def SetEquality(self, ide: Union[str, int], val: Union[int, bool]) -> None:
        """
        Set an equality constraint activation flag.

        Parameters
        ----------
        ide : str | int
            Equality constraint name or id.
        val : int | bool
            Activation flag value.
        """
        self.scene.data.eq_active[self.scene.model.eq(ide).id] = val

    def SetBasePlatform(self, platform: Any, x: Optional[Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType]] = None) -> None:
        """
        Attach or detach a base platform.

        Parameters
        ----------
        platform : Any
            Platform object to attach, or ``None`` to detach the current
            platform.
        x : Pose3DType | HomogeneousMatrixType | RotationMatrixType | Vector3DType, optional
            Optional platform pose.
        """
        if platform is None:
            self.Platform.Detach()
            self.Platform = None
        else:
            self.Platform = platform
            self.Platform.AttachTo(self)
            if x is not None:
                self.Platform.SetRobotBase(x)
            else:
                self.UpdateRobotBaseFromModel()

    def UpdateRobotBaseFromModel(self) -> HomogeneousMatrixType:
        """
        Update the cached robot base pose from the MuJoCo model.

        Returns
        -------
        HomogeneousMatrixType
            Current robot base pose.
        """
        if self._BaseHandle >= 0:
            body_pos = self.scene.data.body(self._BaseHandle).xpos.copy()
            body_quat = self.scene.data.body(self._BaseHandle).xquat.copy()
            _T = map_pose(Q=body_quat, p=body_pos, out="T")
            self.TBase = _T
            if self.Platform is not None:
                self.Platform.TRobotBase = world2frame(_T, self.Platform.T)
                self.Platform.GetState()
        return self.TBase


# Robot classes for specific robot models in MuJoCo
class panda(robot_pymujoco, panda_spec):
    """Synchronous PyMuJoCo robot wrapper for the Franka Panda manipulator."""

    def __init__(self, robot_name: str = "panda", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a Panda robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        panda_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class fr3(robot_pymujoco, fr3_spec):
    """Synchronous PyMuJoCo robot wrapper for the Franka Research 3 manipulator."""

    def __init__(self, robot_name: str = "fr3", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create an FR3 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        fr3_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class lwr(robot_pymujoco, lwr_spec):
    """Synchronous PyMuJoCo robot wrapper for the KUKA LWR manipulator."""

    def __init__(self, robot_name: str = "LWR", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create an LWR robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        lwr_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class iiwa(robot_pymujoco, iiwa_spec):
    """Synchronous PyMuJoCo robot wrapper for the KUKA iiwa manipulator."""

    def __init__(self, robot_name: str = "iiwa14", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a KUKA iiwa robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        iiwa_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class ur10(robot_pymujoco, ur10_spec):
    """Synchronous PyMuJoCo robot wrapper for the Universal Robots UR10 manipulator."""

    def __init__(self, robot_name: str = "ur10", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a UR10 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        ur10_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class ur10e(robot_pymujoco, ur10e_spec):
    """Synchronous PyMuJoCo robot wrapper for the Universal Robots UR10e manipulator."""

    def __init__(self, robot_name: str = "ur10e", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a UR10e robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        ur10e_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class ur5(robot_pymujoco, ur5_spec):
    """Synchronous PyMuJoCo robot wrapper for the Universal Robots UR5 manipulator."""

    def __init__(self, robot_name: str = "ur5", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a UR5 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        ur5_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class ur5e(robot_pymujoco, ur5e_spec):
    """Synchronous PyMuJoCo robot wrapper for the Universal Robots UR5e manipulator."""

    def __init__(self, robot_name: str = "ur5e", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a UR5e robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        ur5e_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class crx20(robot_pymujoco, crx20_spec):
    """Synchronous PyMuJoCo robot wrapper for the FANUC CRX-20 manipulator."""

    def __init__(self, robot_name: str = "CRX20", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create a CRX20 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        crx20_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class hc20(robot_pymujoco, hc20_spec):
    """Synchronous PyMuJoCo robot wrapper for the Yaskawa HC20 manipulator."""

    def __init__(self, robot_name: str = "hc20", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """Create an HC20 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        hc20_spec.__init__(self)
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)


class z1(robot_pymujoco, z1_spec):
    """Synchronous PyMuJoCo robot wrapper for the Unitree Z1 arm."""

    def __init__(self, robot_name: str = "z1", **kwargs: Any) -> None:
        """Create a Z1 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        z1_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_pymujoco.__init__(self, robot_name, **kwargs)


class b2(robot_pymujoco, b2_spec):
    """Synchronous PyMuJoCo robot wrapper for the Unitree B2 platform-arm system."""

    def __init__(self, robot_name: str = "b2", **kwargs: Any) -> None:
        """Create a B2 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        **kwargs : Any
            Additional keyword arguments passed to `robot_pymujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        b2_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_pymujoco.__init__(self, robot_name, **kwargs)


# Internal control
class robot_pymujoco_joint_control(robot_pymujoco):
    """
    Synchronous PyMuJoCo robot interface with internal joint-space impedance control.

    Attributes
    ----------
    control_target_q : np.ndarray
        Desired joint positions for the internal controller.
    control_target_qdot : np.ndarray
        Desired joint velocities for the internal controller.
    control_target_trq : np.ndarray
        Desired feed-forward torques for the internal controller.
    control_Kp : np.ndarray
        Joint proportional gains.
    control_Kd : np.ndarray
        Joint derivative gains.
    control_Kg : float
        Gravity-compensation scaling factor.
    control_use_H : bool
        Indicates whether the joint-space inertia matrix is used in control.
    control_e : np.ndarray
        Current joint-position error.
    """

    def __init__(self, robot_name: str = "Panda", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """
        Create a MuJoCo robot with internal joint impedance control.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to :class:`robot_pymujoco`.
        """
        robot_pymujoco.__init__(self, robot_name, scene=scene, **kwargs)
        self.__dict__.update(kwargs)
        self.control_target_q = self.q_home.copy()
        self.control_target_qdot = np.zeros(self.nj)
        self.control_target_trq = np.zeros(self.nj)
        self.control_Kp = np.ones(self.nj) * 400
        self.control_Kd = np.ones(self.nj) * 40
        self.control_Kg = 1
        self.control_use_H = True  # use inertia matrix in control
        self.control_e = np.zeros(self.nj)

        # set internal joint controller
        mujoco.set_mjcb_control(self.joint_impedance_control)

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Update internal joint-control targets and wait.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions.
        qdot : JointVelocityType, optional
            Desired joint velocities.
        trq : JointTorqueType, optional
            Desired feed-forward joint torques.
        wait : float, optional
            Synchronization time after the target update.

        Returns
        -------
        int
            Motion result code.

        Notes
        -----
        The method updates the internal joint-impedance targets instead of
        writing actuator commands directly.
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

        # set new target for internal controller
        self.control_target_q = q.copy()
        self.control_target_qdot = qdot.copy()
        self.control_target_trq = trq.copy()

        self._command.q = q.copy()
        self._command.qdot = qdot.copy()
        self._command.trq = trq.copy()
        if np.floor(self._command.mode) == CommandModeCodes.JOINT.value:
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

    def joint_impedance_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        Perform joint impedance control and write actuator torques.

        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model object.
        data : mujoco.MjData
            MuJoCo data object holding the current simulator state.

        Notes
        -----
        The controller is a PD controller with an additional feed-forward term
        based on MuJoCo joint-bias torques.
        """
        # Get the current joint positions and velocities
        _q = np.take(self.scene.data.qpos.copy(), self._JointPosHandles)
        _qd = np.take(self.scene.data.qvel.copy(), self._JointVelHandles)

        # Compute the position and velocity errors
        self.control_e = self.control_target_q - _q
        self.control_ed = self.control_target_qdot - _qd

        # Compute the joint torques using the PD control law, plus feed-forward based on joint force bias
        if self.control_use_H:
            _tq = self.H @ (self.control_Kp * self.control_e + self.control_Kd * self.control_ed) + self.control_Kg * data.qfrc_bias[self._JointVelHandles]
        else:
            _tq = (self.control_Kp * self.control_e + self.control_Kd * self.control_ed) + self.control_Kg * data.qfrc_bias[self._JointVelHandles]

        # Apply the computed torques to the actuators
        _ctrl = data.ctrl.copy()  # Copy current control values
        for i, x in zip(self._ActuatorHandles, _tq):
            _ctrl[i] = x  # Set the control for each actuator to the computed torque
        data.ctrl = _ctrl  # Update the control command in the data object


class panda_joint_control(robot_pymujoco_joint_control, panda_spec):
    """
    Synchronous PyMuJoCo Panda robot with internal joint-space impedance control.

    Attributes
    ----------
    All attributes from :class:`robot_pymujoco_joint_control` and
    :class:`panda_spec`.
    """

    def __init__(self, robot_name: str = "Panda", scene: Optional[mujoco_scene] = None, ActuatorNames: List[str] = ["Panda_mot_joint1", "Panda_mot_joint2", "Panda_mot_joint3", "Panda_mot_joint4", "Panda_mot_joint5", "Panda_mot_joint6", "Panda_mot_joint7"], **kwargs: Any) -> None:
        """
        Create a Panda robot with internal joint impedance control.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco_sim.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        ActuatorNames : list[str], optional
            Explicit actuator names for torque control.
        **kwargs : Any
            Additional keyword arguments passed to
            :class:`robot_pymujoco_joint_control`.
        """
        panda_spec.__init__(self)
        robot_pymujoco_joint_control.__init__(self, robot_name, scene=scene, ActuatorNames=ActuatorNames, **kwargs)
        self.__dict__.update(kwargs)
