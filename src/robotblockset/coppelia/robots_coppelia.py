"""Robot interfaces for CoppeliaSim via the ZMQ Remote API.

This module provides RobotBlockSet robot backends for CoppeliaSim. It includes
the generic `robot_coppelia` interface for state acquisition, joint-space
commanding, object pose access, and simulator lifecycle handling, together with
concrete robot wrappers such as `lwr`, `ur10`, `panda`, and `iiwa` that define
robot-specific joint naming and base-frame conventions.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from typing import Union, Optional, Any
from time import sleep
import ctypes

from robotblockset.tools import isvector, vector
from robotblockset.transformations import map_pose, r2q, world2frame, rp2t, rot_z
from robotblockset.robot_spec import panda_spec, lwr_spec, iiwa_spec, ur10_spec  # , ur5_spec, ur5e_spec, ur10e_spec
from robotblockset.robots import robot, MotionResultCodes
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, RotationMatrixType, Vector3DType

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except Exception as e:
    raise e from RuntimeError("Python interface for CoppeliaSim not installed. \nYou can install it  through pip:\n   pip install coppeliasim-zmqremoteapi-client")

c_float_p = ctypes.POINTER(ctypes.c_float)


class robot_coppelia(robot):
    """
    Interface class for controlling a robot in CoppeliaSim via RemoteAPI.

    This class extends the base `robot` class and provides simulation-specific
    setup and naming conventions for joints, end effectors, and sensors. It uses
    the CoppeliaSim RemoteAPI to establish communication with the simulation.

    Attributes
    ----------
    Name : str
        Name of the RobotBlockSet robot instance.
    BaseName : str
        Base name of the robot in the CoppeliaSim scene hierarchy.
    JointNames : list[str] or None
        Explicit list of joint object names, if configured.
    JointNamesAlias : str or None
        Alias used when joints are accessed by indexed object names.
    FlangeName : str
        Name of the end-effector flange object in the scene.
    TCPName : str
        Name of the tool center point object in the scene.
    SensorForceName : str
        Name of the force-torque sensor object in the scene.
    client : RemoteAPIClient
        Remote API client connected to CoppeliaSim.
    sim : Any
        Simulation API object returned by the remote client.
    """

    def __init__(self, robot_name: str, host: str = "localhost", port: int = 23000, **kwargs: Any) -> None:
        """Create a CoppeliaSim-backed robot interface.

        Parameters
        ----------
        robot_name : str
            Base name of the robot used in the CoppeliaSim scene.
        host : str, optional
            Host running the CoppeliaSim ZMQ Remote API server.
        port : int, optional
            Port used by the CoppeliaSim ZMQ Remote API server.
        **kwargs : Any
            Additional keyword arguments used to configure joint, flange, TCP,
            and sensor object names.

        Returns
        -------
        None
            This constructor initializes the CoppeliaSim robot interface in place.
        """
        robot.__init__(self, **kwargs)
        self.Name = robot_name + "_Coppelia"
        self.BaseName = robot_name

        kwargs.setdefault("JointNames", None)
        kwargs.setdefault("JointNamesTemplate", None)
        kwargs.setdefault("JointNamesAlias", None)

        if kwargs["JointNames"] is not None:
            self.JointNames = kwargs["JointNames"]
        else:
            self.JointNames = []
            for i in range(self.nj):
                if kwargs["JointNamesTemplate"] is not None:
                    self.JointNames.append(kwargs["JointNamesTemplate"] % (i + 1))
                elif kwargs["JointNamesAlias"] is not None:
                    self.JointNames = None
                    self.JointNamesAlias = kwargs["JointNamesAlias"]
                else:
                    self.JointNames.append(f"{self.BaseName}_joint{i + 1}")
        kwargs.setdefault("FlangeName", None)
        if kwargs["FlangeName"] is not None:
            self.FlangeName = kwargs["FlangeName"]
        else:
            self.FlangeName = "EE"
        kwargs.setdefault("TCPName", None)
        if kwargs["TCPName"] is not None:
            self.TCPName = kwargs["TCPName"]
        else:
            self.TCPName = "TCP"
        kwargs.setdefault("SensorForceName", None)
        if kwargs["SensorForceName"] is not None:
            self.SensorForceName = kwargs["SensorForceName"]
        else:
            self.SensorForceName = "FT_sensor"

        self._control_strategy = "JointPosition"

        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require("sim")

        if not self.isReady():
            self.sim.startSimulation()
            sleep(0.5)

        self.tsamp = 0.05
        _ts = round(self.sim.getSimulationTimeStep(), 3)
        if self.tsamp < _ts:
            self.WarningMessage(f"tsamp={self.tsamp} is smaller than simulation time step ({_ts}) and will be set to t={_ts}")
            self.tsamp = _ts

        self.Message("Robot connected to Coppelia", 1)
        self.Init()

    def __del__(self) -> None:
        """Release the CoppeliaSim robot interface.

        Returns
        -------
        None
            This destructor closes the simulation interface.
        """
        self.Close()

    def Close(self) -> None:
        """
        Close or clean up the simulation interface.

        Currently does not stop the simulation explicitly.

        Returns
        -------
        None
            This method is reserved for simulation cleanup.
        """
        # self.sim.stopSimulation()
        pass

    def Init(self) -> None:
        """
        Initialize the robot from the CoppeliaSim scene.

        This method performs the following:
        - Retrieves the robot base handle.
        - Computes the robot's base transform matrix.
        - Retrieves joint handles based on configured joint names or aliases.
        - Retrieves and computes the TCP (Tool Center Point) transform relative to the end effector.
        - Retrieves the force/torque sensor handle if defined.
        - Initializes simulation-related internal state, joint positions, and handles.

        Raises
        ------
        ValueError
            If the robot base is not found in the CoppeliaSim scene.

        Returns
        -------
        None
            This method initializes simulator handles and cached robot state.
        """
        h = self.sim.getObject(f"/{self.BaseName}", {"noError": True})
        if h == -1:
            raise ValueError(f"Robot {self.BaseName} not found in the scene. ")
        else:
            self._BaseHandle = h
            self.TBase = map_pose(x=self.sim.getObjectPose(self._BaseHandle | self.sim.handleflag_wxyzquat), out="T") @ np.linalg.inv(self.BaseOffset)

        self._JointHandles = [None] * self.nj
        self._actual.q = [None] * self.nj
        self._actual.qdot = [None] * self.nj

        for i in range(self.nj):
            if self.JointNames is None and self.JointNamesAlias is not None:
                self._JointHandles[i] = self.sim.getObject(f"/{self.BaseName}/{self.JointNamesAlias}", {"index": i})
            else:
                self._JointHandles[i] = self.sim.getObject(f"/{self.BaseName}/{self.JointNames[i]}")

        if self.FlangeName is not None and self.TCPName is not None:
            self._TCPHandle = self.sim.getObject(f"/{self.BaseName}/{self.TCPName}", {"noError": True})
            self._EEHandle = self.sim.getObject(f"/{self.BaseName}/{self.FlangeName}", {"noError": True})
            if self._TCPHandle > -1 and self._EEHandle > -1:
                _TTCP = map_pose(x=self.sim.getObjectPose(self._TCPHandle | self.sim.handleflag_wxyzquat), out="T")
                _TEE = map_pose(x=self.sim.getObjectPose(self._EEHandle | self.sim.handleflag_wxyzquat), out="T")
                self.TCP = np.linalg.inv(_TEE) @ _TTCP

            else:
                self._TCPHandle = None
                self._EEHandle = None
                self.TCP = np.eye(4)
        else:
            self._TCPHandle = None
            self._EEHandle = None
            self.TCP = np.eye(4)

        if self.SensorForceName is not None:
            self._SensorForceHandle = self.sim.getObject(f"/{self.BaseName}/{self.SensorForceName}", {"noError": True})
            if self._SensorForceHandle == -1:
                self._SensorForceHandle = None

        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    def GetState(self) -> None:
        """
        Update the robot's internal state from the CoppeliaSim simulation.

        This method queries joint positions, velocities, and torques, along with the
        TCP pose and velocity (if available), and force/torque sensor data (if available).
        The update occurs only if the elapsed simulation time exceeds the sampling period.

        Updates the following internal state attributes:
        - self._actual.q      : Joint positions
        - self._actual.qdot   : Joint velocities
        - self._actual.trq    : Joint torques
        - self._actual.x      : TCP pose (in base frame)
        - self._actual.v      : TCP velocity (in base frame)
        - self._actual.FT     : Force/torque sensor readings
        - self._actual.trqExt : External joint torques (zeroed by default)

        Notes
        -----
        - Uses self.tsamp as the minimum time interval between updates.
        - Falls back to forward kinematics if TCP is not defined.
        - Sets force/torque to zeros if sensor is unavailable or inactive.

        Returns
        -------
        None
            This method refreshes the internal robot state caches.
        """
        if (self.simtime() - self._last_update) > self.tsamp:
            for i in range(self.nj):
                self._actual.q[i] = self.joint_k * self.sim.getJointPosition(self._JointHandles[i])
                self._actual.qdot[i] = self.joint_k * self.sim.getJointVelocity(self._JointHandles[i])
                self._actual.trq[i] = self.joint_k * self.sim.getJointForce(self._JointHandles[i])

            if self._TCPHandle is not None:
                _x = self.sim.getObjectPose(self._TCPHandle | self.sim.handleflag_wxyzquat, -1)
                self._actual.x = self.WorldToBase(_x)
                _pd, _w = self.sim.getObjectVelocity(self._TCPHandle, -1)
                _v = np.concatenate([_pd, _w])
                self._actual.v = self.WorldToBase(_v, typ="Twist")
            else:
                x, J = self.Kinmodel(self._actual.q)
                self._actual.x = x
                self._actual.v = J @ self._actual.qdot

            if self._SensorForceHandle is not None:
                _res, _F, _T = self.sim.readForceSensor(self._SensorForceHandle)
                if _res > 0:
                    _FT = np.concatenate((_F, _T))
                    if self.FTSensorFrame is None:
                        _frame = self.TCP
                    else:
                        _frame = self.FTSensorFrame
                    _FT2TCP = np.linalg.pinv(_frame) @ self.TCP
                    _FT = world2frame(_FT, _FT2TCP, typ="Wrench")
                    self._actual.FT = _FT
                else:
                    self._actual.FT = np.zeros(6)

            self._actual.trqExt = np.zeros(self.nj)

            self._tt = self.simtime()
            self._last_update = self.simtime()

    def isReady(self) -> bool:
        """
        Check whether the simulation is running.

        Returns
        -------
        bool
            True if the simulation is not in a stopped state, False otherwise.
        """
        return self.sim.getSimulationState() != self.sim.simulation_stopped

    def isActive(self) -> bool:
        """
        Check if the robot target is active.

        Returns
        -------
        bool
            Indicating if the robot target is active.
        """
        return True

    def Restart(self, qpos: Optional[JointConfigurationType] = None) -> None:
        """
        Restart the simulation.

        Stops and restarts the CoppeliaSim simulation. This method waits briefly
        after each step to allow the simulator to cleanly stop and start.

        Parameters
        ----------
        qpos : JointConfigurationType, optional
            Placeholder for future support of setting initial joint positions after restart.
            Currently unused.

        Returns
        -------
        None
            This method restarts the CoppeliaSim simulation.
        """
        self.sim.stopSimulation()
        t0 = self.simtime()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            if (self.simtime() - t0) > 10.0:
                self.WarningMessage("Simulation did not stop within 10 seconds", 2)
                return
            sleep(0.1)
        self.sim.startSimulation()
        sleep(1.0)

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
        if wait is None:
            wait = self.tsamp

        self._synchro_control(wait)
        for i in range(self.nj):
            self.sim.setJointTargetPosition(self._JointHandles[i], self.joint_k * q[i])
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        if np.floor(self._command.mode) == 1:
            x, J = self.Kinmodel(q)
            self._command.x = x
            self._command.v = J @ qdot
        self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value

    def SetStrategy(self, strategy: str) -> None:
        """
        Set the robot's control strategy.

        Parameters
        ----------
        strategy : str
            Name of the control strategy to be used (e.g., 'JointPosition').

        Returns
        -------
        None
            This placeholder currently does not change controller behavior.

        Notes
        -----
        This method is currently a placeholder and does not implement any behavior.
        """
        pass

    def GetObjectPose(self, ide: Union[str, int], out: str = "x") -> Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]:
        """
        Retrieve the pose of an object from the simulation.

        Parameters
        ----------
        ide : Union[str, int]
            Object name or handle.
        out : str, optional
            Output format.

        Returns
        -------
        Optional[Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]]
            Pose in the requested format if found, else None.
        """
        if isinstance(ide, str):
            handle = self.sim.getObject(ide, {"noError": True})
            if handle == -1:
                self.Message(f"No object with name '{ide}' exists", 2)
                return None
        else:
            handle = ide

        return map_pose(x=self.sim.getObjectPose(handle | self.sim.handleflag_wxyzquat, -1), out=out)

    def SetObjectPose(self, ide: Union[str, int], x: ArrayLike) -> None:
        """
        Set the pose of an object in the simulation.

        Parameters
        ----------
        ide : Union[str, int]
            Object name or body ID.
        x : ArrayLike
            Pose to be set. Supported formats:
                - 4x4 transformation matrix
                - 3x3 rotation matrix
                - 7D vector [pos(3), quat(4)]
                - 3D position vector
                - 4D quaternion

        Returns
        -------
        None
            This method updates the pose of the selected simulation object.

        Raises
        ------
        ValueError
            If the shape of `x` is not one of the supported formats.
        """
        if self.isReady():
            if isinstance(ide, str):
                handle = self.sim.getObject(ide, {"noError": True})
                if handle == -1:
                    self.Message(f"No object with name '{ide}' exists", 2)
                    return None
            else:
                handle = ide

            x = self.spatial(x)
            xx = self.GetObjectPose(handle)
            if x.shape == (4, 4):
                xx = map_pose(T=x)
            elif x.shape == (3, 3):
                xx[3:] = r2q(x)
            elif isvector(x, dim=7):
                xx = x
            elif isvector(x, dim=3):
                xx[:3] = x
            elif isvector(x, dim=4):
                xx[3:] = x
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

            self.sim.setObjectPose(handle | self.sim.handleflag_wxyzquat, xx.tolist())

    def UpdateRobotBaseFromModel(self) -> HomogeneousMatrixType:
        """Update the cached robot base transform from the CoppeliaSim model.

        Returns
        -------
        HomogeneousMatrixType
            Current robot base transform in homogeneous matrix form.
        """
        if self._BaseHandle is not None:
            self.TBase = map_pose(x=self.sim.getObjectPose(self._BaseHandle | self.sim.handleflag_wxyzquat), out="T") @ np.linalg.inv(self.BaseOffset)
        return self.TBase


class lwr(robot_coppelia, lwr_spec):
    """CoppeliaSim wrapper for the KUKA LWR manipulator."""

    def __init__(self, robot_name: str = "LBR4p", **kwargs: Any) -> None:
        """Create a CoppeliaSim KUKA LWR robot wrapper.

        Parameters
        ----------
        robot_name : str, optional
            Robot object name in the CoppeliaSim scene.
        **kwargs : Any
            Additional keyword arguments passed to `robot_coppelia.__init__`.

        Returns
        -------
        None
            This constructor initializes the LWR robot wrapper in place.
        """
        lwr_spec.__init__(self)
        self.joint_k = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
        self.BaseOffset = rp2t(rot_z(np.pi), [0.0, 0.0, 0.059])  # Robot base offset between Coppelia and correct model
        self.joint_k = 1.0  # Joint position factors between Coppelia and correct model
        robot_coppelia.__init__(self, robot_name, **kwargs)

    def __del__(self) -> None:
        robot_coppelia.__del__(self)
        self.Message("Robot deleted", 2)


class ur10(robot_coppelia, ur10_spec):
    """CoppeliaSim wrapper for the Universal Robots UR10 manipulator."""

    def __init__(self, robot_name: str = "UR10", **kwargs: Any) -> None:
        """Create a CoppeliaSim UR10 robot wrapper.

        Parameters
        ----------
        robot_name : str, optional
            Robot object name in the CoppeliaSim scene.
        **kwargs : Any
            Additional keyword arguments passed to `robot_coppelia.__init__`.

        Returns
        -------
        None
            This constructor initializes the UR10 robot wrapper in place.
        """
        ur10_spec.__init__(self)
        self.BaseOffset = rp2t(rot_z(np.pi), [0.0, 0.0, 0.019])  # Robot base offset between Coppelia and correct model
        self.joint_k = 1.0  # Joint position factors between Coppelia and correct model
        robot_coppelia.__init__(self, robot_name, **kwargs)

    def __del__(self) -> None:
        robot_coppelia.__del__(self)
        self.Message("Robot deleted", 2)


class panda(robot_coppelia, panda_spec):
    """CoppeliaSim wrapper for the Franka Emika Panda manipulator."""

    def __init__(self, robot_name: str = "Franka", JointNamesAlias: str = "joint", **kwargs: Any) -> None:
        """Create a CoppeliaSim Franka Panda robot wrapper.

        Parameters
        ----------
        robot_name : str, optional
            Robot object name in the CoppeliaSim scene.
        JointNamesAlias : str, optional
            Indexed alias used to access joint objects in the scene.
        **kwargs : Any
            Additional keyword arguments passed to `robot_coppelia.__init__`.

        Returns
        -------
        None
            This constructor initializes the Panda robot wrapper in place.
        """
        panda_spec.__init__(self)
        self.BaseOffset = rp2t(np.eye(3), [-0.04873, 0, 0.07])  # Robot base offset between Coppelia and correct model
        self.joint_k = 1.0  # Joint position factors between Coppelia and correct model
        robot_coppelia.__init__(self, robot_name, JointNamesAlias=JointNamesAlias, **kwargs)

    def __del__(self) -> None:
        robot_coppelia.__del__(self)
        self.Message("Robot deleted", 2)


class iiwa(robot_coppelia, iiwa_spec):
    """CoppeliaSim wrapper for the KUKA iiwa manipulator."""

    def __init__(self, robot_name: str = "iiwa", JointNamesAlias: str = "joint", **kwargs: Any) -> None:
        """Create a CoppeliaSim KUKA iiwa robot wrapper.

        Parameters
        ----------
        robot_name : str, optional
            Robot object name in the CoppeliaSim scene.
        JointNamesAlias : str, optional
            Indexed alias used to access joint objects in the scene.
        **kwargs : Any
            Additional keyword arguments passed to `robot_coppelia.__init__`.

        Returns
        -------
        None
            This constructor initializes the iiwa robot wrapper in place.
        """
        iiwa_spec.__init__(self)
        self.BaseOffset = rp2t(np.eye(3), [0.00726, 0.00005, 0.07866])  # Robot base offset between Coppelia and correct model
        self.joint_k = 1.0  # Joint position factors between Coppelia and correct model
        robot_coppelia.__init__(self, robot_name, JointNamesAlias=JointNamesAlias, **kwargs)

    def __del__(self) -> None:
        robot_coppelia.__del__(self)
        self.Message("Robot deleted", 2)


if __name__ == "__main__":
    # import sys

    # sys.path.append("robot_models")
    # sys.path.append("sim")
    # sys.path.append("..")

    print("Start")
    r = panda(SensorForceName="connection", host="178.172.42.81")

    r.JMove(r.q_home + 0.2)
    print("q: ", r.q)
    print("x: ", r.x)
