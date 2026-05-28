"""Robot interfaces for the socket-based MuJoCo backend.

This module provides RobotBlockSet robot backends that communicate with an
external MuJoCo simulator through `mjInterface`, together with concrete robot
wrappers for supported manipulators and mobile platforms.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Any, Optional, Sequence, Union
from time import sleep
from copy import deepcopy

from robotblockset.tools import isvector, vector, find_rows
from robotblockset.transformations import map_pose, r2q, checkx, world2frame
from robotblockset.mujoco.mujoco_api import mjInterface
from robotblockset.robot_spec import panda_spec, fr3_spec, lwr_spec, iiwa_spec, ur10_spec, ur10e_spec, ur5_spec, ur5e_spec, crx20_spec, hc20_spec, hc30_spec, z1_spec, b2_spec
from robotblockset.robots import robot, MotionResultCodes, CommandModeCodes
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, QuaternionType, RotationMatrixType, Vector3DType

mujoco_scene = mjInterface


class robot_mujoco(robot):
    """
    MuJoCo-backed robot interface using the socket-based server API.

    Attributes
    ----------
    scene : mjInterface
        Socket-based MuJoCo interface used to exchange state and commands.
    BaseName : str
        Base model name used to derive joint, actuator, and sensor names.
    JointNames : list[str]
        Ordered list of robot joint names.
    ActuatorNames : list[str]
        Ordered list of actuator names used for joint commands.
    MocapNames : list[str]
        Names of MuJoCo mocap bodies associated with the scene.

    """

    def __init__(self, robot_name: str, scene: Optional[mjInterface] = None, host: str = "localhost", port: int = 50000, **kwargs: Any) -> None:
        """Create a MuJoCo-backed robot interface.

        Parameters
        ----------
        robot_name : str
            Base name of the robot model in MuJoCo.
        scene : mjInterface, optional
            Existing MuJoCo interface instance. If `None`, a new connection is created.
        host : str, optional
            Hostname of the MuJoCo simulator.
        port : int, optional
            Port of the MuJoCo simulator.
        **kwargs : Any
            Additional keyword arguments.

        Notes
        -----
        When constrructing objects of this class, the following keyword arguments are supported for explicit configuration of model element names. If not provided, default names based on the platform name are used.:
            `JointNames` : list[str] or str, optional
                Explicit joint names, or ``"gen"`` to generate names from `robot_name`.
            `ActuatorNames` : list[str], optional
                Explicit actuator names.
            `FlangeName` : str, optional
                Name of the flange or end-effector body.
            `TCPName` : str, optional
                Name of the tool center point body or site.
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
        """
        robot.__init__(self, **kwargs)
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
        self.Name = robot_name + "_MuJoCo"
        self.Message("Robot connected to MuJoCo", 1)

        self._info = self.scene.mj_info()

        self.BaseName = robot_name
        self._control_strategy = "JointPosition"

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

        self.MocapHandles = [None] * self._info.nmocap

        if self._info.nu > 0:
            self._ctrl = self.scene.mj_get_control()

        self.tsamp = 0.01  # sampling rate
        self.Init()

    def Init(self) -> None:
        """
        Initialize MuJoCo handles and cached sensor indices.

        Notes
        -----
        The method resolves joint, actuator, site, sensor, and mocap handles
        and then initializes the RobotBlockSet state.
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
        self.UpdateRobotBaseFromModel()
        i1 = self.scene.mj_name2id("site", self.FlangeName)
        i2 = self.scene.mj_name2id("site", self.TCPName)
        if i1 >= 0 and i2 >= 0:
            si = self.scene.mj_get_site()
            site_pos = np.array(si.pos[: si.nsite])
            site_mat = np.array(si.mat[: si.nsite]).reshape((-1, 3, 3))
            pEE = site_pos[i1]
            REE = site_mat[i1]
            pHand = site_pos[i2]
            RHand = site_mat[i2]
            self.TCP = map_pose(R=REE.T @ RHand, p=REE.T @ (pHand - pEE), out="T")

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

    # def simtime(self):
    #     self._state = self.scene.mj_get_state()
    #     return self._state.time

    def GetState(self) -> None:
        """
        Update robot state from MuJoCo data buffers.

        Notes
        -----
        Joint, Cartesian, force-torque, and object-following state are updated
        from the current simulator buffers.
        """
        self._state = self.scene.mj_get_state()
        self._sensor = self.scene.mj_get_sensor()
        self._robottime = self._state.time
        if all(self._SensorJointPosHandles):
            self._actual.q = np.take(self._sensor.sensordata, self._SensorJointPosHandles)
        else:
            self._actual.q = np.take(self._state.qpos, self._JointPosHandles)
        if all(self._SensorJointVelHandles):
            self._actual.qdot = np.take(self._sensor.sensordata, self._SensorJointVelHandles)
        else:
            self._actual.qdot = np.take(self._state.qvel, self._JointVelHandles)

        if (self._SensorPosHandles is not None) and (self._SensorOriHandles is not None):
            _x = checkx(np.take(self._sensor.sensordata, self._SensorPosHandles + self._SensorOriHandles))
            self._actual.x = self.WorldToBase(_x)
        else:
            x, J = self.Kinmodel()
            self._actual.x = x
        if (self._SensorLinVelHandles is not None) and (self._SensorRotVelHandles is not None):
            _v = np.take(self._sensor.sensordata, self._SensorLinVelHandles + self._SensorRotVelHandles)
            self._actual.v = self.WorldToBase(_v, typ="Twist")
        else:
            self._actual.v = self.Jacobi() @ self._actual.qdot
        if (self._SensorForceHandles is not None) and (self._SensorTorqueHandles is not None):
            self._actual.FT = np.take(self._sensor.sensordata, self._SensorForceHandles + self._SensorTorqueHandles)
        else:
            self._actual.FT = np.zeros(6)
            self._actual.trq = np.zeros(self.nj)

        self._actual.trq = np.zeros(self.nj)
        self._actual.trqExt = np.zeros(self.nj)

        if self.EEFixed:
            self.TObject = map_pose(x=self.BaseToWorld(self._actual.x), out="T")

        self._tt = self._state.time
        self._last_update = self.simtime()

    def isReady(self) -> bool:
        """
        Return whether the simulator connection is active.

        Returns
        -------
        bool
            ``True`` if the MuJoCo connection is active.
        """
        self._connected = self.scene.mj_connected()
        return self._connected

    def isActive(self) -> bool:
        """
        Check whether the simulator is advancing in time.

        Returns
        -------
        bool
            ``True`` if the simulator time is progressing.
        """
        s1 = self.scene.mj_get_state()
        sleep(0.1)
        s2 = self.scene.mj_get_state()
        return s2.time > s1.time

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

        Notes
        -----
        The method optionally resets the simulator, applies state and control
        vectors, clears joint velocities, and resets RobotBlockSet timing and
        targets.
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
                if isvector(qpos, dim=self._info.nq):
                    self._state.qpos = qpos
                    self.scene.mj_set_state(self._state)
            if u is not None:
                if isvector(u, dim=self.nj):
                    self.SendRobot_u(u)
                    self._state.qpos[self._JointPosHandles] = u
                    self.scene.mj_set_state(self._state)
            self._state.qvel = np.zeros(self._state.nv)
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
            Time to wait (in seconds) to synchronize the command to move. If omitted, ``self.tsamp`` is used.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).

        Notes
        -----
        The method sends the joint command to MuJoCo, updates the RobotBlockSet
        command state, and synchronizes with the requested wait time.
        """
        if qdot is None:
            qdot = np.zeros(self.nj)
        else:
            qdot = vector(qdot, dim=self.nj)
        if trq is None:
            trq = np.zeros(self.nj)
        else:
            trq = vector(trq, dim=self.nj)
        if wait is None:
            wait = self.tsamp

        self._synchro_control(wait)
        self.SendRobot_u(q)
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        if np.floor(self._command.mode) == CommandModeCodes.JOINT.value:
            x, J = self.Kinmodel(q)
            self._command.x = x
            self._command.v = J @ qdot
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
        The socket-based MuJoCo wrapper currently keeps a single joint-position
        strategy and accepts this method for compatibility.
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
            self._ctrl = self.scene.mj_get_control()
            for i, x in zip(self._ActuatorHandles, u):
                self._ctrl.ctrl[i] = x
            self.scene.mj_set_control(self._ctrl)

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
            Control values to assign.
        """
        if self.isReady() and self._info.nu > 0:
            self._ctrl = self.scene.mj_get_control()
            for i, x in zip(idx, val):
                self._ctrl.ctrl[i] = x
            self.scene.mj_set_control(self._ctrl)

    def GetAuxJointPos(self, idx: Sequence[int]) -> Optional[np.ndarray]:
        """
        Return joint positions for auxiliary joints by index.

        Parameters
        ----------
        idx : Sequence[int]
            Joint-position indices to read.

        Returns
        -------
        np.ndarray | None
            Joint positions for the selected indices, if available.
        """
        if self.isReay():
            self._state = self.scene.mj_get_state()
            return np.take(self._state, idx)

    def GetSensor(self, ide: Union[str, int] = None) -> Optional[np.ndarray]:
        """
        Read sensor data by name or id, or return the full sensor array.

        Parameters
        ----------
        ide : str | int, optional
            Optional sensor name or id.

        Returns
        -------
        np.ndarray | None
            Selected sensor data, full sensor data, or ``None`` if the sensor
            could not be resolved.
        """
        self._sensor = self.scene.mj_get_sensor()
        if ide is not None:
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
        contacts = self.scene.mj_get_contact()
        if contacts.ncon > 0:
            if ide is not None:
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

    def GetMocapPose(self, ide: Union[str, int], out: str = "x") -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]:
        """Return mocap body pose in the requested output format."""
        if self.isReady and (self._info.nmocap > 0):
            if isinstance(ide, str):
                idx = self.MocapNames.index(ide)
                if idx < 0:
                    self.Message(f"No mocap body with name '{ide}' exits", 2)
                    return None
            else:
                idx = ide
            val = self.scene.body()
            return map_pose(
                p=np.array(val.pos[idx]),
                R=np.array(val.mat[idx]).reshape(3, 3),
                out=out,
            )

    def GetObjectData(self, ide: Union[str, int]) -> Optional[Any]:
        """Return raw MuJoCo body data for a body name/id."""
        if self.isReady:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("body", ide)
                if idx < 0:
                    self.Message(f"No body with name '{ide}' exits", 2)
                    return None
            else:
                idx = ide
        return self.scene.mj_get_onebody(idx)

    def GetObjectPose(self, typ: str, ide: Union[str, int], out: str = "x") -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]:
        """Return the pose of a body/site/geom in the requested format."""
        if self.isReady and (typ in set(["body", "site", "geom"])):
            if isinstance(ide, str):
                idx = self.scene.mj_name2id(typ, ide)
                if idx < 0:
                    self.Message(f"No {typ} with name '{ide}' exits", 2)
                    return None
            else:
                idx = ide
            val = eval("self.scene.mj_get_" + typ + "()")
            return map_pose(
                p=np.array(val.pos[idx]),
                R=np.array(val.mat[idx]).reshape(3, 3),
                out=out,
            )

    def SetObjectPose(self, ide: Union[str, int], x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]) -> None:
        """Set a MuJoCo body pose from a spatial representation."""
        if self.isReady:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("body", ide)
                if idx < 0:
                    self.Message(f"No body with name '{ide}' exits", 2)
                    return
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

    def SetEquality(self, ide: Union[str, int], val: Union[int, bool]) -> None:
        """Set an equality constraint activation flag."""
        if self.isReady:
            if isinstance(ide, str):
                idx = self.scene.mj_name2id("equality", ide)
                if idx < 0:
                    self.Message(f"No equality with name '{ide}' exits", 2)
                    return
            else:
                idx = ide
            self.scene.mj_equality(idx, val)

    def SetBasePlatform(self, platform: Any, x: Optional[Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType]] = None) -> None:
        """Attach or detach a base platform.

        Attach or detach a base platform and optionally set its pose."""
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
        """Update the cached robot base pose from the MuJoCo model."""
        if self._BaseHandle >= 0:
            bb = self.scene.mj_get_body()
            body_pos = np.array(bb.pos[: bb.nbody])
            body_mat = np.array(bb.mat[: bb.nbody]).reshape((-1, 3, 3))
            _T = map_pose(R=body_mat[self._BaseHandle], p=body_pos[self._BaseHandle], out="T")
            self.TBase = _T
            if self.Platform is not None:
                self.Platform.TRobotBase = world2frame(_T, self.Platform.T)
                self.Platform.GetState()
        return self.TBase

    def SimulatorMessage(self, msg: str) -> None:
        """Send a message to the simulator UI."""
        if self._connected:
            self.scene.mj_message(msg)

    def sim(self, dt: float) -> None:
        """Advance the simulator for a fixed duration."""
        if self.isReady():
            self._state = self.scene.mj_get_state()
            t0 = self._state.time
            t1 = t0
            while t1 < t0 + dt:
                sensor = self.scene.mj_update(self._ctrl)
                t1 = sensor.time
            self.GetState()
            self.Update()


class panda(robot_mujoco, panda_spec):
    """MuJoCo robot wrapper for the Franka Panda manipulator."""

    def __init__(self, robot_name: str = "panda", **kwargs: Any) -> None:
        """Create a Panda robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        **kwargs : Any
            Additional keyword arguments passed to `robot_mujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        panda_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class fr3(robot_mujoco, fr3_spec):
    """MuJoCo robot wrapper for the Franka Research 3 manipulator."""

    def __init__(self, robot_name: str = "fr3", **kwargs: Any) -> None:
        """Create an FR3 robot in MuJoCo.

        Parameters
        ----------
        robot_name : str, optional
            Base name of the robot model in MuJoCo.
        **kwargs : Any
            Additional keyword arguments passed to `robot_mujoco`, including
            optional joint, actuator, flange, TCP, and sensor names.
        """
        fr3_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class lwr(robot_mujoco, lwr_spec):
    """MuJoCo robot wrapper for the KUKA LWR manipulator."""

    def __init__(self, robot_name: str = "LWR", **kwargs: Any) -> None:
        """Create an LWR robot in MuJoCo."""
        lwr_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class iiwa(robot_mujoco, iiwa_spec):
    """MuJoCo robot wrapper for the KUKA iiwa manipulator."""

    def __init__(self, robot_name: str = "iiwa14", **kwargs: Any) -> None:
        """Create a KUKA iiwa robot in MuJoCo."""
        iiwa_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class ur10(robot_mujoco, ur10_spec):
    """MuJoCo robot wrapper for the Universal Robots UR10 manipulator."""

    def __init__(self, robot_name: str = "ur10", **kwargs: Any) -> None:
        """Create a UR10 robot in MuJoCo."""
        ur10_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class ur10e(robot_mujoco, ur10e_spec):
    """MuJoCo robot wrapper for the Universal Robots UR10e manipulator."""

    def __init__(self, robot_name: str = "ur10e", **kwargs: Any) -> None:
        """Create a UR10e robot in MuJoCo."""
        ur10e_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class ur5(robot_mujoco, ur5_spec):
    """MuJoCo robot wrapper for the Universal Robots UR5 manipulator."""

    def __init__(self, robot_name: str = "ur5", **kwargs: Any) -> None:
        """Create a UR5 robot in MuJoCo."""
        ur5_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class ur5e(robot_mujoco, ur5e_spec):
    """MuJoCo robot wrapper for the Universal Robots UR5e manipulator."""

    def __init__(self, robot_name: str = "ur5e", **kwargs: Any) -> None:
        """Create a UR5e robot in MuJoCo."""
        ur5e_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class crx20(robot_mujoco, crx20_spec):
    """MuJoCo robot wrapper for the FANUC CRX-20 collaborative manipulator."""

    def __init__(self, robot_name: str = "CRX20", **kwargs: Any) -> None:
        """Create a CRX20 robot in MuJoCo."""
        crx20_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class hc20(robot_mujoco, hc20_spec):
    """MuJoCo robot wrapper for the Yaskawa HC20 manipulator."""

    def __init__(self, robot_name: str = "hc20", **kwargs: Any) -> None:
        """Create an HC20 robot in MuJoCo."""
        hc20_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class hc30(robot_mujoco, hc30_spec):
    """MuJoCo robot wrapper for the Yaskawa HC30 manipulator."""

    def __init__(self, robot_name: str = "hc30", **kwargs: Any) -> None:
        """Create an HC30 robot in MuJoCo."""
        hc30_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class z1(robot_mujoco, z1_spec):
    """MuJoCo robot wrapper for the Unitree Z1 arm."""

    def __init__(self, robot_name: str = "z1", **kwargs: Any) -> None:
        """Create an Z1 robot in MuJoCo."""
        z1_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, **kwargs)


class b2(robot_mujoco, b2_spec):
    """MuJoCo robot wrapper for the Unitree B2 platform-arm system."""

    def __init__(self, robot_name: str = "b2", **kwargs: Any) -> None:
        """Create an B2 robot in MuJoCo."""
        b2_spec.__init__(self)
        kwargs.setdefault("host", "localhost")
        robot_mujoco.__init__(self, robot_name, JointNames="ori", ActuatorNames="ori", **kwargs)
