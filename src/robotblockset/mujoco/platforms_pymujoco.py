"""PyMuJoCo Platforms Module.

This module provides platform implementations for the Python MuJoCo simulator,
mirroring the platform interface in `robotblockset.platforms`.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from time import sleep
from typing import Any, Optional, Sequence, Union

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

from robotblockset.mujoco.scene_pymujoco import mujoco_scene
from robotblockset.mujoco.tools_pymujoco import get_joints_under_body, get_actuators_for_joints
from robotblockset.tools import isvector, vector
from robotblockset.transformations import map_pose, r2q, checkx, x2t, v2s, q2r
from robotblockset.platform_spec import tiagobase_spec, mir100_spec
from robotblockset.platforms import platform, MotionResultCodes
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointVelocityType, Pose3DType, QuaternionType, RotationMatrixType, Vector3DType

ObjectIdType = Union[str, int]
PoseInputType = Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]
PoseOutputType = Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]


class platform_pymujoco(platform):
    """PyMuJoCo-backed mobile platform interface operating on a :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`."""

    def __init__(self, platform_name: str, scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """
        Initialize a PyMuJoCo platform interface.

        Parameters
        ----------
        platform_name : str
            Base name of the platform model in MuJoCo.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            MuJoCo scene instance to control. Must be provided.
        **kwargs : Any
            Additional keyword arguments.

        Notes
        -----
        When constrructing objects of this class, the following keyword arguments are supported for explicit configuration of model element names. If not provided, default names based on the platform name are used.:
            `JointNames` : list[str], optional
                Explicit list of platform joint names.
            `ActuatorNames` : list[str], optional
                Explicit list of actuator names used for velocity commands.
            `RobotBaseName` : str, optional
                Name of the robot base body used as the platform reference frame.
            `SensorJointPosNames` : list[str], optional
                Sensor names used to read joint positions.
            `SensorJointVelNames` : list[str], optional
                Sensor names used to read joint velocities.
            `SensorPosName` : str, optional
                Sensor name used to read the platform position.
            `SensorOriName` : str, optional
                Sensor name used to read the platform orientation.
            `SensorLinVelName` : str, optional
                Sensor name used to read the platform linear velocity.
            `SensorRotVelName` : str, optional
                Sensor name used to read the platform angular velocity.
            `SensorForceName` : str, optional
                Sensor name used to read force measurements.
            `SensorTorqueName` : str, optional
                Sensor name used to read torque measurements.

        Raises
        ------
        Exception
            If joint or actuator names do not resolve uniquely in the MJCF model.
        """
        platform.__init__(self, **kwargs)
        if scene is None:
            raise ValueError("MuJoCo scene is not defined")
        self.scene = scene
        self._connected = True
        self.Name = platform_name + "_PyMuJoCo"
        self.Message("Platform connected to MuJoCo", 1)

        self.BaseName = platform_name
        self._control_strategy = "CartesianVelocity"

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

        kwargs.setdefault("RobotBaseName", None)
        if kwargs["RobotBaseName"]:
            self.RobotBaseName = kwargs["RobotBaseName"]
        else:
            self.RobotBaseName = None

        kwargs.setdefault("SensorJointPosNames", None)
        if kwargs["SensorJointPosNames"]:
            self.SensorJointPosNames = kwargs["SensorJointPosNames"]
        else:
            self.SensorJointPosNames = []
            for i in range(self.nj):
                self.SensorJointPosNames.append(self.BaseName + "_pos_joint" + str(i + 1))

        kwargs.setdefault("SensorJointVelNames", None)
        if kwargs["SensorJointVelNames"]:
            self.SensorJointVelNames = kwargs["SensorJointVelNames"]
        else:
            self.SensorJointVelNames = []
            for i in range(self.nj):
                self.SensorJointVelNames.append(self.BaseName + "_vel_joint" + str(i + 1))

        kwargs.setdefault("SensorPosName", None)
        if kwargs["SensorPosName"]:
            self.SensorPosName = kwargs["SensorPosName"]
        else:
            self.SensorPosName = self.BaseName + "_pos"

        kwargs.setdefault("SensorOriName", None)
        if kwargs["SensorOriName"]:
            self.SensorOriName = kwargs["SensorOriName"]
        else:
            self.SensorOriName = self.BaseName + "_ori"

        kwargs.setdefault("SensorLinVelName", None)
        if kwargs["SensorLinVelName"]:
            self.SensorLinVelName = kwargs["SensorLinVelName"]
        else:
            self.SensorLinVelName = self.BaseName + "_v"

        kwargs.setdefault("SensorRotVelName", None)
        if kwargs["SensorRotVelName"]:
            self.SensorRotVelName = kwargs["SensorRotVelName"]
        else:
            self.SensorRotVelName = self.BaseName + "_w"

        kwargs.setdefault("SensorForceName", None)
        if kwargs["SensorForceName"]:
            self.SensorForceName = kwargs["SensorForceName"]
        else:
            self.SensorForceName = self.BaseName + "_force"
        kwargs.setdefault("SensorTorqueName", None)
        if kwargs["SensorTorqueName"]:
            self.SensorTorqueName = kwargs["SensorTorqueName"]
        else:
            self.SensorTorqueName = self.BaseName + "_torque"

        self.MocapHandles = [None] * self.scene.model.nmocap

        if self.scene.model.nu > 0:
            self._ctrl = self.scene.data.ctrl.copy()

        self.tsamp = 0.01  # sampling rate
        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo handles and cached sensor indices.

        Returns
        -------
        None
        """
        self._BodyNamesList = [self.scene.model.body(i).name for i in range(self.scene.model.nbody)]
        self._SiteNamesList = [self.scene.model.site(i).name for i in range(self.scene.model.nsite)]
        self._SensorNamesList = [self.scene.model.sensor(i).name for i in range(self.scene.model.nsensor)]

        self._BaseHandle = self.scene.model.body(self.BaseName).id
        if self.RobotBaseName is not None and self.RobotBaseName in self._BodyNamesList:
            _p1 = self.scene.data.body(self.BaseName).xpos
            _R1 = self.scene.data.body(self.BaseName).xmat.reshape(3, 3)
            _p2 = self.scene.data.body(self.RobotBaseName).xpos
            _R2 = self.scene.data.body(self.RobotBaseName).xmat.reshape(3, 3)
            self.TRobotBase = map_pose(R=_R1.T @ _R2, p=_R1.T @ (_p2 - _p1), out="T")

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

    def simtime(self) -> float:
        """
        Return the current simulation time from the MuJoCo simulator.

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
            Requested pause duration in seconds.

        Returns
        -------
        None

        Notes
        -----
        This method uses MuJoCo simulation time rather than wall-clock time.
        """
        t0 = self.simtime()
        while self.simtime() - t0 < time:
            sleep(0.001)

    def GetState(self) -> None:
        """
        Update platform state from MuJoCo data buffers.

        Returns
        -------
        None
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
            self._actual.x = checkx(np.take(sensors, self._SensorPosHandles + self._SensorOriHandles))
        if (self._SensorLinVelHandles is not None) and (self._SensorRotVelHandles is not None):
            self._actual.v = np.take(sensors, self._SensorLinVelHandles + self._SensorRotVelHandles)
        if (self._SensorForceHandles is not None) and (self._SensorTorqueHandles is not None):
            self._actual.FT = np.take(sensors, self._SensorForceHandles + self._SensorTorqueHandles)
        else:
            self._actual.FT = np.zeros(6)
            self._actual.trq = np.zeros(self.nj)

        self._actual.trq = np.zeros(self.nj)
        self._actual.trqExt = np.zeros(self.nj)

        if self.Robot is not None:
            _tmp = x2t(self._actual.x) @ self.TRobotBase
            self.Robot.SetBasePose(_tmp)
            RR = np.eye(6)
            RR[:3, 3:] = v2s(q2r(self._actual.x[3:]) @ self.TRobotBase[:3, 3]).T
            self.Robot.SetBaseVel(RR @ self.actual.v)

        self._tt = self.simtime()
        self._last_update = self.simtime()

    def isReady(self) -> bool:
        """
        Check if the platform is connected.

        Returns
        -------
        bool
            True if the simulator connection is active.
        """
        return self._connected

    def isActive(self) -> bool:
        """
        Check if the simulator is connected.

        Returns
        -------
        bool
            True if the simulation is connected.
        """
        s1 = self.scene.time
        sleep(0.1)
        s2 = self.scene.time
        return s2 > s1

    def Restart(self, qpos: Optional[ArrayLike] = None, u: Optional[JointVelocityType] = None, reset: bool = True, keyframe: Optional[int] = None) -> None:
        """
        Reset the simulation and optionally set joint positions/inputs.

        Parameters
        ----------
        qpos : ArrayLike, optional
            Joint positions for the platform.
        u : JointVelocityType, optional
            Joint velocity command applied after the reset.
        reset : bool, optional
            If True, reset the simulation state.
        keyframe : int, optional
            Optional keyframe index for reset.
        """
        if reset:
            self.scene.mj_pause()
            self.scene.mj_reset(keyframe)
            mujoco.mj_forward(self.scene.model, self.scene.data)
            self.scene.mj_run()
        if qpos is not None:
            if isvector(qpos, dim=self.scene.model.nq):
                self.scene.data.qpos = qpos
        if u is not None:
            if isvector(u, dim=self.nj):
                self.scene.data.qpos[self._JointPosHandles] = u
                self.SendRobot_u(u)
        self.scene.data.qvel = np.zeros(self.scene.model.nv)
        self.ResetCurrentTarget()
        self.ResetTime()

    def Stop(self) -> None:
        """
        Stop platform motion and clear velocity commands.

        Returns
        -------
        None
        """
        self.GetState()
        self.scene.data.qvel = np.zeros(self.scene.model.nv)
        self._command.ux = np.zeros(2)
        self.SendRobot_u(np.zeros(self.nj))
        platform.Stop(self)

    def Set_vel(self, v: ArrayLike, wait: Optional[float] = None) -> int:
        """
        Update platform velocities (forward, turn) and wait.

        Parameters
        ----------
        v : ArrayLike
            Desired planar velocity ``[vx, wz]``.
        wait : float, optional
            Time to wait (in seconds) to synchronize the command to move. If omitted, ``self.tsamp`` is used.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        if wait is None:
            wait = self.tsamp
        self._synchro_control(wait)
        v = vector(v, dim=2)
        _fac = max(max(v / self.v_max), max(v / self.v_min), 1)
        v = v / _fac
        u = self.pJs @ v
        _fac = max(max(np.abs(u) / self.qdot_max), 1)
        u = u / _fac
        v = v / _fac
        self._command.ux = v
        self.SendRobot_u(u)
        self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value

    def SendRobot_u(self, u: JointVelocityType) -> None:
        """
        Send joint commands to platform actuators.

        Parameters
        ----------
        u : JointVelocityType
            Actuator command vector.
        """
        self._command.u = u
        if self.isReady:
            self._ctrl = self.scene.data.ctrl.copy()
            for i, x in zip(self._ActuatorHandles, u):
                self._ctrl[i] = x
            self.scene.data.ctrl = self._ctrl

    def SendCtrl(self, u: ArrayLike) -> None:
        """
        Send a full control vector to the MuJoCo platform.

        Parameters
        ----------
        u : ArrayLike
            Full actuator control vector for the MuJoCo scene.
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
            Control values written to the selected actuators.

        Returns
        -------
        None
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
            Joint indices in ``self.data.qpos``.

        Returns
        -------
        JointConfigurationType
            Joint positions for the requested indices.
        """
        return np.take(self.data.qpos, ide)

    def GetSensor(self, *ide: ObjectIdType) -> np.ndarray:
        """
        Read sensor data by name/id or return the full sensor array.

        Parameters
        ----------
        *ide : str or int
            Optional sensor identifier. When omitted, all sensor samples are returned.

        Returns
        -------
        np.ndarray
            Sensor data slice or full sensor data if no id is provided.
        """
        if len(ide) > 0:
            return self.scene.data.sensor(ide).data
        else:
            return self.scene.data.sensordata

    def GetContacts(self, *ide: ObjectIdType) -> Optional[np.ndarray]:
        """
        Return contact forces for a geom or for all contacts.

        Parameters
        ----------
        *ide : str or int
            Optional geom identifier. When omitted, all contacts are reported.

        Returns
        -------
        np.ndarray or None
            Contact force array of shape (N,3), or None if no contacts.
        """
        contacts = self.scene.data.contact
        ncon = len(contacts)
        if ncon > 0:
            if len(ide) > 0:
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

    def SetMocapPose(self, ide: ObjectIdType, x: PoseInputType) -> None:
        """
        Set the pose of a mocap body.

        Parameters
        ----------
        ide : ObjectIdType
            Mocap body name or index.
        x : PoseInputType
            Pose represented as position, quaternion, pose vector, rotation matrix,
            or homogeneous transform.

        Raises
        ------
        ValueError
            If ``x`` has an unsupported shape.

        Notes
        -----
        When mocap body names are used, mocap bodies must be the first bodies in the model.
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

    def GetMocapPose(self, ide: ObjectIdType, out: str = "x") -> Optional[PoseOutputType]:
        """
        Return mocap body pose in the requested output format.

        Parameters
        ----------
        ide : ObjectIdType
            Mocap body name or id.
        out : str, optional
            Output pose format understood by :func:`robotblockset.transformations.map_pose`.

        Returns
        -------
        PoseOutputType or None
            Mocap pose in the requested representation, or ``None`` if unavailable.
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

    def GetObjectData(self, ide: ObjectIdType) -> Any:
        """
        Return raw MuJoCo body data for a body name/id.

        Parameters
        ----------
        ide : ObjectIdType
            Body name or id.

        Returns
        -------
        Any
            Raw MuJoCo body data object.
        """
        return self.scene.data.body(ide).copy()

    def GetObjectPose(self, typ: str, ide: ObjectIdType, out: str = "x") -> Optional[PoseOutputType]:
        """
        Return the pose of a body/site/geom in the requested format.

        Parameters
        ----------
        typ : str
            Object type ("body", "site", or "geom").
        ide : ObjectIdType
            Object name or id.
        out : str, optional
            Output pose format understood by :func:`robotblockset.transformations.map_pose`.

        Returns
        -------
        PoseOutputType or None
            Pose of the requested object, or ``None`` when the object type is unsupported.
        """
        if typ in set(["body", "site", "geom"]):
            pos = eval(f"self.scene.data.{typ}({ide}).xpos.copy()")
            quat = eval(f"self.scene.data.{typ}({ide}).xquat.copy()")
            return map_pose(p=pos, Q=quat, out=out)

    def SetObjectPose(self, ide: ObjectIdType, x: PoseInputType) -> None:
        """
        Set a body pose from a spatial representation.

        Parameters
        ----------
        ide : ObjectIdType
            Body name or id.
        x : PoseInputType
            Pose represented as position, quaternion, pose vector, rotation matrix,
            or homogeneous transform.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``x`` has an unsupported shape.
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

    def SetEquality(self, ide: ObjectIdType, val: Union[int, bool]) -> None:
        """
        Set an equality constraint activation flag.

        Parameters
        ----------
        ide : ObjectIdType
            Equality constraint name or id.
        val : Union[int, bool]
            Activation flag.

        Returns
        -------
        None
        """
        self.scene.data.eq_active[self.scene.model.eq(ide).id] = val


class tiagobase(platform_pymujoco, tiagobase_spec):
    """PyMuJoCo platform wrapper for the PAL Robotics Tiago Base."""

    def __init__(self, platform_name: str = "tiagobase", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """
        Create a TiagoBase platform backed by PyMuJoCo.

        Parameters
        ----------
        platform_name : str, optional
            Base name of the platform in the MuJoCo model.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to :class:`platform_pymujoco`,
            including optional joint, actuator, base-body, and sensor names.
        """
        tiagobase_spec.__init__(self)
        platform_pymujoco.__init__(self, platform_name, scene=scene, **kwargs)


class mir100_pymujoco(platform_pymujoco, mir100_spec):
    """PyMuJoCo platform wrapper for the MiR100 mobile base."""

    def __init__(self, platform_name: str = "mir", scene: Optional[mujoco_scene] = None, **kwargs: Any) -> None:
        """
        Create a MiR100 platform backed by PyMuJoCo.

        Parameters
        ----------
        platform_name : str, optional
            Base name of the platform in the MuJoCo model.
        scene : :class:`~robotblockset.mujoco.scene_pymujoco.mujoco_scene`, optional
            Scene instance that owns the MuJoCo model and data.
        **kwargs : Any
            Additional keyword arguments passed to :class:`platform_pymujoco`,
            including optional joint, actuator, base-body, and sensor names.
        """
        mir100_spec.__init__(self)
        platform_pymujoco.__init__(self, platform_name, scene=scene, **kwargs)
