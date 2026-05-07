"""MuJoCo Platforms Module.

This module provides platform implementations for the MuJoCo simulator,
mirroring the platform interface in `robotblockset.platforms`.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from time import sleep
from copy import deepcopy
from typing import Any, Optional, Sequence, Union

from robotblockset.tools import isvector, vector, find_rows
from robotblockset.transformations import map_pose, r2q, checkx, x2t, v2s, q2r
from robotblockset.mujoco.mujoco_api import mjInterface
from robotblockset.platform_spec import tiagobase_spec, mir100_spec
from robotblockset.platforms import platform, MotionResultCodes
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointVelocityType, Pose3DType, QuaternionType, RotationMatrixType, Vector3DType

ObjectIdType = Union[str, int]
PoseInputType = Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]
PoseOutputType = Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]


class platform_mujoco(platform):
    """MuJoCo-backed mobile platform interface using the socket-based server API."""

    def __init__(self, platform_name: str, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, **kwargs: Any) -> None:
        """
        Initialize a MuJoCo platform interface.

        Parameters
        ----------
        platform_name : str
            Base name of the platform model in MuJoCo.
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If None, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        port : int, optional
            Port of the MuJoCo simulator.
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
            self.scene = mjInterface(host=host, port=port)
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
        self.Name = platform_name + "_MuJoCo"
        self.Message("Platform connected to MuJoCo", 1)

        self._info = self.scene.mj_info()

        self.BaseName = platform_name
        self._control_strategy = "CartesianVelocity"

        kwargs.setdefault("JointNames", None)
        if isinstance(kwargs["JointNames"], str):
            if kwargs["JointNames"].lower() == "ori":
                if hasattr(self, "joint_names"):
                    self.JointNames = self.joint_names
                else:
                    raise ValueError("Robot specification does not define joint names.")
            else:
                raise ValueError(f"Argument 'JointNames':{kwargs['JointNames']} is invalid. Only 'ori' is accepted.")
        elif kwargs["JointNames"] is None:
            if hasattr(self, "joint_names") and self.scene.mj_name2id("joint", self.BaseName + "_" + self.joint_names[0]) > -1:
                self.JointNames = [self.BaseName + "_" + jnt for jnt in self.joint_names]
            else:
                self.JointNames = []
                for i in range(self.nj):
                    self.JointNames.append(self.BaseName + "_joint" + str(i + 1))
        else:
            self.JointNames = kwargs["JointNames"]

        kwargs.setdefault("ActuatorNames", None)
        if isinstance(kwargs["ActuatorNames"], str):
            if kwargs["ActuatorNames"].lower() == "ori":
                if hasattr(self, "actuator_names"):
                    self.ActuatorNames = self.actuator_names
                else:
                    raise ValueError("Robot specification does not define actuator names.")
            else:
                raise ValueError(f"Argument 'ActuatorNames':{kwargs['ActuatorNames']} is invalid. Only 'ori' is accepted.")
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

        self.MocapHandles = [None] * self._info.nmocap

        if self._info.nu > 0:
            self._ctrl = self.scene.mj_get_control()

        self.tsamp = 0.01  # sampling rate
        self.Init()

    def Close(self) -> None:
        """
        Close the MuJoCo connection.

        Returns
        -------
        None
        """
        if self.scene.mj_close() == 0:
            self.Message("MuJoCo scene disconnected")

    def Init(self) -> None:
        """
        Initialize MuJoCo handles and cached sensor indices.

        Returns
        -------
        None
        """
        self._JointPosHandles = [None] * self.nj
        self._JointVelHandles = [None] * self.nj
        self._ActuatorHandles = [None] * self.nj
        self._SensorJointPosHandles = [None] * self.nj
        self._SensorJointVelHandles = [None] * self.nj
        for i in range(self.nj):
            joint_id = self.scene.mj_name2id("joint", self.JointNames[i])
            if joint_id >= 0:
                self._JointPosHandles[i] = self._info.jnt_qposadr[joint_id]
                self._JointVelHandles[i] = self._info.jnt_dofadr[joint_id]
            else:
                self._JointPosHandles[i] = -1
                self._JointVelHandles[i] = -1
            self._ActuatorHandles[i] = self.scene.mj_name2id("actuator", self.ActuatorNames[i])
            idx = self.scene.mj_name2id("sensor", self.SensorJointPosNames[i])
            if idx >= 0:
                adr = self._info.sensor_adr[idx]
                dim = self._info.sensor_dim[idx]
                if dim != 1:
                    raise Exception("Wrong joint sensor in model")
                self._SensorJointPosHandles[i] = adr
            idx = self.scene.mj_name2id("sensor", self.SensorJointVelNames[i])
            if idx >= 0:
                adr = self._info.sensor_adr[idx]
                dim = self._info.sensor_dim[idx]
                if dim != 1:
                    raise Exception("Wrong joint sensor in model")
                self._SensorJointVelHandles[i] = adr

        if any(handle == -1 for handle in self._JointPosHandles) or len(set(self._JointPosHandles)) != len(self._JointPosHandles):
            raise Exception("Check naming of joints in MJCF model")

        if any(handle == -1 for handle in self._ActuatorHandles) or len(set(self._ActuatorHandles)) != len(self._ActuatorHandles):
            raise Exception("Check naming of actuators in MJCF model")

        self._BaseHandle = self.scene.mj_name2id("body", self.BaseName)
        if self._BaseHandle >= 0 and self.RobotBaseName is not None:
            i1 = self._BaseHandle
            i2 = self.scene.mj_name2id("body", self.RobotBaseName)
            if i1 >= 0 and i2 >= 0:
                _b = self.scene.mj_get_body()
                _pos = np.array(_b.pos[: _b.nbody])
                _mat = np.array(_b.mat[: _b.nbody]).reshape((-1, 3, 3))
                _p1 = _pos[i1]
                _R1 = _mat[i1]
                _p2 = _pos[i2]
                _R2 = _mat[i2]
                self.TRobotBase = map_pose(R=_R1.T @ _R2, p=_R1.T @ (_p2 - _p1), out="T")

        idx = self.scene.mj_name2id("sensor", self.SensorPosName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorPosHandles = list(range(adr, adr + dim))
        else:
            self._SensorPosHandles = None

        idx = self.scene.mj_name2id("sensor", self.SensorOriName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorOriHandles = list(range(adr, adr + dim))
        else:
            self._SensorOriHandles = None

        idx = self.scene.mj_name2id("sensor", self.SensorLinVelName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorLinVelHandles = list(range(adr, adr + dim))
        else:
            self._SensorLinVelHandles = None

        idx = self.scene.mj_name2id("sensor", self.SensorRotVelName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorRotVelHandles = list(range(adr, adr + dim))
        else:
            self._SensorRotVelHandles = None

        idx = self.scene.mj_name2id("sensor", self.SensorForceName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorForceHandles = list(range(adr, adr + dim))
        else:
            self._SensorForceHandles = None

        idx = self.scene.mj_name2id("sensor", self.SensorTorqueName)
        if idx >= 0:
            adr = self._info.sensor_adr[idx]
            dim = self._info.sensor_dim[idx]
            self._SensorTorqueHandles = list(range(adr, adr + dim))
        else:
            self._SensorTorqueHandles = None

        self.scene.mj_pause()
        mocap = self.scene.mj_get_mocap()
        mocap_x = deepcopy(mocap)
        mocap_x.pos = np.random.randint(0, 10, (self._info.nmocap, 3))
        self.scene.mj_set_mocap(mocap_x)
        sleep(0.02)
        bodies = self.scene.mj_get_body()
        self.scene.mj_set_mocap(mocap)
        self.MocapNames = []
        for i in range(self._info.nmocap):
            idx = find_rows(bodies.pos, mocap_x.pos[i, :])
            self.MocapHandles[i] = idx[0]
            self.MocapNames.append(self.scene.mj_id2name("body", idx[0]))
        self.scene.mj_run()

        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    def GetState(self) -> None:
        """
        Update platform state from MuJoCo data buffers.

        Returns
        -------
        None
        """
        self._state = self.scene.mj_get_state()
        self._robottime = self._state.time
        self._sensor = self.scene.mj_get_sensor()
        if all(self._SensorJointPosHandles):
            self._actual.q = np.take(self._sensor.sensordata, self._SensorJointPosHandles)
        else:
            self._actual.q = np.take(self._state.qpos, self._JointPosHandles)
        if all(self._SensorJointVelHandles):
            self._actual.qdot = np.take(self._sensor.sensordata, self._SensorJointVelHandles)
        else:
            self._actual.qdot = np.take(self._state.qvel, self._JointVelHandles)

        if (self._SensorPosHandles is not None) and (self._SensorOriHandles is not None):
            self._actual.x = checkx(np.take(self._sensor.sensordata, self._SensorPosHandles + self._SensorOriHandles))
        if (self._SensorLinVelHandles is not None) and (self._SensorRotVelHandles is not None):
            self._actual.v = np.take(self._sensor.sensordata, self._SensorLinVelHandles + self._SensorRotVelHandles)
        if (self._SensorForceHandles is not None) and (self._SensorTorqueHandles is not None):
            self._actual.FT = np.take(self._sensor.sensordata, self._SensorForceHandles + self._SensorTorqueHandles)
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
            if self.Robot.EEFixed:
                self.TObject = map_pose(x=self.Robot.BaseToWorld(self.Robot._actual.x), out="T")

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
        self._connected = self.scene.mj_connected()
        return self._connected

    def isActive(self) -> bool:
        """
        Check if the simulator is connected.

        Returns
        -------
        bool
            True if the simulation is connected.
        """
        s1 = self.scene.mj_get_state()
        sleep(0.1)
        s2 = self.scene.mj_get_state()
        return s2.time > s1.time

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
        if self.isReady:
            if reset:
                self.scene.mj_pause()
                if keyframe is None:
                    self.scene.mj_reset()
                else:
                    self.scene.mj_reset(keyframe)
                self.scene.mj_run()
            if qpos is not None:
                if isvector(qpos, dim=self.info.nq):
                    self._state.qpos = qpos
                    self._command.q = np.take(qpos, self._JointPosHandles)
            if u is not None:
                if isvector(u, dim=self.nj):
                    self.SendRobot_u(u)
                    self._state.qpos[self._JointPosHandles] = u
                    self.scene.mj_set_state(self._state)
            self._state.qvel = np.zeros(self._state.nv)
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
        self._state.qvel = np.zeros(self._info.nv)
        self.scene.mj_set_state(self._state)
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
            self._ctrl = self.scene.mj_get_control()
            for i, x in zip(self._ActuatorHandles, u):
                self._ctrl.ctrl[i] = x
            self.scene.mj_set_control(self._ctrl)

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
            self._ctrl.ctrl = u
            self.scene.mj_set_control(self._ctrl)

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
        if self.isReady() and self._info.nu > 0:
            self._ctrl = self.scene.mj_get_control()
            for i, x in zip(idx, val):
                self._ctrl.ctrl[i] = x
            self.scene.mj_set_control(self._ctrl)

    def GetAuxJointPos(self, idx: Sequence[int]) -> JointConfigurationType:
        """
        Return joint positions for auxiliary joints by index.

        Parameters
        ----------
        idx : Sequence[int]
            Joint indices in the simulator state vector.

        Returns
        -------
        JointConfigurationType
            Joint positions for the requested indices.
        """
        self._state = self.scene.mj_get_state()
        return np.take(self._state, idx)

    def GetSensor(self, *ide: ObjectIdType) -> Optional[np.ndarray]:
        """
        Read sensor data by name/id or return the full sensor array.

        Parameters
        ----------
        *ide : str or int
            Optional sensor identifier. When omitted, all sensor samples are returned.

        Returns
        -------
        np.ndarray or None
            Sensor data slice or full sensor data if no id is provided.
        """
        self._sensor = self.scene.mj_get_sensor()
        if len(ide) > 0:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("sensor", ide)
            else:
                idx = ide
            if idx >= 0:
                adr = self._info.sensor_adr[idx]
                dim = self._info.sensor_dim[idx]
                return self._sensor.sensordata[adr : adr + dim]
            else:
                return None
        else:
            return self._sensor.sensordata

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
        contacts = self.scene.mj_get_contact()
        if contacts.ncon > 0:
            if len(ide) > 0:
                if isinstance(ide, str):
                    idx = self.scene.mj_name2id("geom", ide)
                else:
                    idx = ide
                ii = list(set(np.where(contacts.geom1 == idx)[0]) | set(np.where(contacts.geom2 == idx)[0]))
            else:
                ii = list(range(contacts.ncon))
            Fx = np.empty((len(ii), 3))
            for i, ix in enumerate(ii):
                R = contacts.frame[i, :].reshape(3, 3)
                F = contacts.force[i, :].reshape(3, 1)
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
        if self.isReady and self._info.nmocap > 0:
            mocap = self.scene.mj_get_mocap()
            if isinstance(ide, str):
                idx = self.MocapNames.index(ide)
            else:
                idx = ide
            x = self.spatial(x)
            if x.shape == (4, 4):
                xx = map_pose(T=x)
                mocap.pos[idx, :] = xx[:3]
                mocap.quat[idx, :] = xx[3:]
            elif x.shape == (3, 3):
                xx = r2q(x)
                mocap.quat[idx, :] = xx
            elif isvector(x, dim=7):
                mocap.pos[idx, :] = x[:3]
                mocap.quat[idx, :] = x[3:]
            elif isvector(x, dim=3):
                mocap.pos[idx, :] = x
            elif isvector(x, dim=4):
                mocap.quat[idx, :] = x
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
            self.scene.mj_set_mocap(mocap)

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
        if self.isReady and (self._info.nmocap > 0):
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("body", ide)
            else:
                idx = ide
            val = self.scene.body()
            return map_pose(
                p=np.array(val.pos[idx]),
                R=np.array(val.mat[idx]).reshape(3, 3),
                out=out,
            )

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
        if self.isReady and (typ in set(["body", "site", "geom"])):
            if isinstance(ide, str):
                idx = self.scene.mj_name2id(typ, ide)
            else:
                idx = ide
            val = eval("self.scene.mj_get_" + typ + "()")
            return map_pose(
                p=np.array(val.pos[idx]),
                R=np.array(val.mat[idx]).reshape(3, 3),
                out=out,
            )

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
        if self.isReady:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("body", ide)
            else:
                idx = ide
            body = self.scene.mj_get_onebody(idx)
            x = self.spatial(x)
            if x.shape == (4, 4):
                xx = map_pose(T=x)
                body.pos = xx[:3]
                body.quat = xx[3:]
            elif x.shape == (3, 3):
                xx = r2q(x)
                body.quat = xx
            elif isvector(x, dim=7):
                body.pos = x[:3]
                body.quat = x[3:]
            elif isvector(x, dim=3):
                body.pos = x
            elif isvector(x, dim=4):
                body.quat = x
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
            self.scene.mj_set_onebody(body)

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
        if self.isReady:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("equality", ide)
                if idx < 0:
                    self.Message(f"No equality with name '{ide}' exits", 2)
                    return
            else:
                idx = ide
            self.scene.mj_equality(idx, val)

    def SimulatorMessage(self, msg: str) -> None:
        """
        Send a message to the simulator UI.

        Parameters
        ----------
        msg : str
            Message to display.

        Returns
        -------
        None
        """
        if self._connected:
            self.scene.mj_message(msg)

    def sim(self, dt: float) -> None:
        """
        Advance the simulator for a fixed duration.

        Parameters
        ----------
        dt : float
            Duration, in seconds, to advance the simulator.

        Returns
        -------
        None
        """
        if self.isReady():
            self._state = self.scene.mj_get_state()
            t0 = self._state.time
            t1 = t0
            while t1 < t0 + dt:
                sensor = self.scene.mj_update(self._ctrl)
                t1 = sensor.time
            self.Update()


class tiagobase(platform_mujoco, tiagobase_spec):
    """MuJoCo platform wrapper for the PAL Robotics Tiago Base."""

    def __init__(self, platform_name: str = "tiagobase", **kwargs: Any) -> None:
        """
        Create a TiagoBase platform backed by MuJoCo.

        Parameters
        ----------
        platform_name : str, optional
            Base name of the platform in the MuJoCo model.
        **kwargs : Any
            Additional keyword arguments passed to :class:`platform_mujoco`,
            including optional joint, actuator, base-body, and sensor names.
        """
        tiagobase_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        platform_mujoco.__init__(self, platform_name, **kwargs)


class mir100(platform_mujoco, mir100_spec):
    """MuJoCo platform wrapper for the MiR100 mobile base."""

    def __init__(self, platform_name: str = "mir", **kwargs: Any) -> None:
        """
        Create a MiR100 platform backed by MuJoCo.

        Parameters
        ----------
        platform_name : str, optional
            Base name of the platform in the MuJoCo model.
        **kwargs : Any
            Additional keyword arguments passed to :class:`platform_mujoco`,
            including optional joint, actuator, base-body, and sensor names.
        """
        mir100_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        platform_mujoco.__init__(self, platform_name, **kwargs)
