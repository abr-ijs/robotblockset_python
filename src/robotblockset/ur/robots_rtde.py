"""UR10 robot interface using the UR RTDE protocol.

This module provides a high-level `ur10` robot class built on RobotBlockSet,
wrapping Universal Robots' RTDE control, receive, IO, and dashboard interfaces.
It also defines status codes and helpers for interpreting robot and safety state.

Copyright (c) 2025- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from time import sleep
from enum import IntEnum
from typing import Optional, Iterable, Union, Any, Tuple, List
from threading import Thread

from robotblockset.transformations import map_pose, pa2x, x2pa, x2x, t2x, x2t
from robotblockset.robots import robot, MotionResultCodes, CommandModeCodes
from robotblockset.robot_spec import ur10_spec, ur10e_spec
from robotblockset.tools import rbs_type, isscalar, vector, isvector, check_option, load_params
from robotblockset.robot_models import kinmodel_ur10, kinmodel_ur10e
from robotblockset.rbs_typing import (
    ArrayLike,
    HomogeneousMatrixType,
    JointConfigurationType,
    JointPathType,
    JointTorqueType,
    JointVelocityType,
    Pose3DType,
    QuaternionType,
    RotationMatrixType,
    TCPType,
    Vector3DType,
    Velocity3DType,
    WrenchType,
    JacobianType,
)

try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    from rtde_io import RTDEIOInterface as RTDEIO
    from dashboard_client import DashboardClient
except Exception as e:
    raise e from RuntimeError("Python interface for RTDE not installed. \nYou can install ur_rtde through pip:\n   pip install ur_rtde")


class URStatusCode(IntEnum):
    """Status and error codes used by the UR RTDE robot interface."""

    OK = 0
    # Connection layer
    NOT_CONNECTED = 10
    # Safety layer
    EMERGENCY_STOP = 20
    PROTECTIVE_STOP = 21
    SAFEGUARD_STOP = 22
    SAFETY_FAULT = 23
    SAFETY_VIOLATION = 24
    SAFETY_STOPPED = 25
    # Robot mode / state layer
    RECOVERY_MODE = 30
    POWER_OFF = 31
    POWER_ON_NO_MOTION = 32
    BACKDRIVE = 33
    DISCONNECTED = 34
    NO_CONTROLLER = 35
    BOOTING = 36
    UPDATING_FIRMWARE = 37
    # Advisory
    REDUCED_MODE = 90


UR_CODE_DESCRIPTIONS = {
    URStatusCode.OK: "OK",
    URStatusCode.NOT_CONNECTED: "One or more RTDE/Dashboard sockets not connected",
    URStatusCode.EMERGENCY_STOP: "Emergency stop active (system/robot E-stop)",
    URStatusCode.PROTECTIVE_STOP: "Protective stop active",
    URStatusCode.SAFEGUARD_STOP: "Safeguard stop active (safety device triggered)",
    URStatusCode.SAFETY_FAULT: "Safety fault",
    URStatusCode.SAFETY_VIOLATION: "Safety violation (limits exceeded)",
    URStatusCode.SAFETY_STOPPED: "Stopped due to safety (generic)",
    URStatusCode.RECOVERY_MODE: "Recovery mode",
    URStatusCode.POWER_OFF: "Controller powered off",
    URStatusCode.POWER_ON_NO_MOTION: "Powered/idle; drives not enabled",
    URStatusCode.BACKDRIVE: "Backdrive / hand-guiding active",
    URStatusCode.DISCONNECTED: "Robot disconnected",
    URStatusCode.NO_CONTROLLER: "No controller detected",
    URStatusCode.BOOTING: "Controller booting",
    URStatusCode.UPDATING_FIRMWARE: "Updating firmware",
    URStatusCode.REDUCED_MODE: "Reduced mode active",
}

_SAFETY_BIT_NAMES = {
    0: "normal_mode",
    1: "reduced_mode",
    2: "protective_stopped",
    3: "recovery_mode",
    4: "safeguard_stopped",
    5: "system_emergency_stopped",
    6: "robot_emergency_stopped",
    7: "emergency_stopped",
    8: "violation",
    9: "fault",
    10: "stopped_due_to_safety",
}


def _active_safety_flags(bits: int, *, include_bit0: bool = False) -> list[str]:
    flags = []
    for i, name in _SAFETY_BIT_NAMES.items():
        if not include_bit0 and i == 0:
            continue
        if (bits >> i) & 1:
            flags.append(name)
    return flags


def _decode_robot_mode(mode: int) -> str:
    names = {
        -1: "NO_CONTROLLER",
        0: "DISCONNECTED",
        1: "CONFIRM_SAFETY",
        2: "BOOTING",
        3: "POWER_OFF",
        4: "POWER_ON",
        5: "IDLE",
        6: "BACKDRIVE",
        7: "RUNNING",
        8: "UPDATING_FIRMWARE",
    }
    return names.get(mode, f"UNKNOWN({mode})")


def _sbit(bits: int, idx: int) -> bool:
    return ((bits >> idx) & 1) == 1


class robot_ur_rtde(robot):
    """
    High-level interface for UR robots using the RTDE protocol.

    This class integrates the Universal Robots RTDE (Real-Time Data Exchange)
    interfaces for control, monitoring, and IO with the RobotBlockSet framework.
    It enables commanding joint or Cartesian motion, monitoring robot status,
    performing force control, and interacting with the dashboard client.

    Parameters
    ----------
    robot_name : str
        Name of the robot instance, used for logging and identification.
    host : str, optional
        IP address of the UR robot controller. Default is ``"192.168.56.101"``.

    Attributes
    ----------
    Name : str
        Name of the robot instance.
    tsamp : float
        Sampling time used for RTDE servoing commands (typically ``1/125 s``).
    velocity : float
        Default motion speed scaling factor (0–1).
    acceleration : float
        Default acceleration scaling factor (0–1).
    lookahead_time : float
        Lookahead time for RTDE servoing.
    gain : float
        Gain applied to RTDE servoing.
    rtde_c : RTDEControl
        RTDE control interface instance.
    rtde_r : RTDEReceive
        RTDE receive (state feedback) interface instance.
    rtde_i : RTDEIO
        RTDE IO interface instance.
    dash_c : DashboardClient
        Dashboard interface for higher-level controller actions.

    Notes
    -----
    - Control commands are sent using the RTDE protocol at real-time rates.
    - Cartesian pose commands are automatically mapped to the robot base frame.
    - Force mode allows compliant movement based on external wrench feedback.
    - The robot must be powered on with drives enabled before motion commands.
    - Safety states and error codes can be interpreted using `Check()`.
    """

    def __init__(self, robot_name: str, host: str = "192.168.56.101") -> None:
        """
        Create a UR robot interface using the RTDE protocol.

        Parameters
        ----------
        robot_name : str
            Name of the robot instance, used for logging and identification.
        host : str, optional
            IP address of the UR robot controller. Default is ``"192.168.56.101"``.

        Returns
        -------
        None
            This constructor initializes the RTDE robot interface in place.
        """
        robot.__init__(self)

        self.Name = robot_name
        self.tsamp = 1.0 / 125.0
        self.velocity = 0.5
        self.acceleration = 0.5
        self.lookahead_time = 0.1
        self.gain = 300
        self._robottime = 0

        self._connected = False

        self._control_strategy = "JointPosition"

        self.rtde_c = RTDEControl(host)
        self.rtde_r = RTDEReceive(host)
        self.rtde_i = RTDEIO(host)
        self.dash_c = DashboardClient(host)
        self.dash_c.connect()
        self.t_start = self.rtde_c.initPeriod()
        self._internal_kinematics = False
        self.Message("Robot connected", 1)
        self.Init()

        self._connected = self.dash_c.isConnected()
        self.Message("Initialized", 2)

    def __del__(self) -> None:
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self.rtde_i.disconnect()
        self.dash_c.disconnect()

    # States
    def GetState(self) -> None:
        """Read and cache the current robot state.

        Returns
        -------
        None
            This method refreshes the internal joint, Cartesian, and wrench state.
        """

        self._tt = self.simtime()  # _state.time
        self._robottime = self.rtde_r.getTimestamp()
        _pos = self.rtde_r.getActualQ()
        _vel = self.rtde_r.getActualQd()
        self._actual.q = rbs_type(_pos)
        self._actual.qdot = rbs_type(_vel)

        if self._default.Kinematics == "Robot":
            _x = pa2x(self.rtde_r.getActualTCPPose())
            _xd = self.rtde_r.getActualTCPSpeed()
        else:
            _x, J = self.Kinmodel(self._actual.q, internal_kinematics=False)
            _xd = J @ self._actual.qdot
        self._actual.x = rbs_type(_x)
        self._actual.v = rbs_type(_xd)

        _FT = self.rtde_r.getActualTCPForce()
        self._actual.FT = rbs_type(_FT)

        # _rpos = self.rtde_r.getTargetQ()
        # _rvel = self.rtde_r.getTargetQd()
        # _rx = pa2x(self.rtde_r.getTargetTCPPose())
        # _rxd = self.rtde_r.getTargetTCPSpeed()
        # self._command.q = rbs_type(_rpos)
        # self._command.qdot = rbs_type(_rvel)
        # self._command.x = rbs_type(_rx)
        # self._command.v = rbs_type(_rxd)

        self._last_update = self.simtime()  # Do not change !

    # Strategies
    def AvailableStrategies(self) -> List[str]:
        """
        Get the available control strategies for the robot.

        Returns
        -------
        List[str]
            Names of the supported control strategies.
        """
        return ["JointPosition", "CartesianPosition", "JointPositionForced"]

    def SetStrategy(self, strategy: str) -> None:
        """
        Set the control strategy for the robot.

        Parameters
        ----------
        strategy : str
            Control strategy to activate.

        Returns
        -------
        None
            This method updates the active control strategy.
        """
        if strategy.lower() == "jointposition":
            self._control_strategy = "JointPosition"
        elif strategy.lower() == "jointpositionforced":
            self._control_strategy = "JointPositionForced"
        elif strategy.lower() == "cartesianposition":
            self._control_strategy = "CartesianPosition"
        else:
            raise ValueError(f"Strategy {strategy} is not supported")
        self.Message(f"Selected control strategy: {self._control_strategy}", 2)

    def SetTeachMode(self) -> None:
        """
        Enable teach mode on the robot.

        Returns
        -------
        None
            This method enables hand-guiding mode through RTDE.
        """
        self.rtde_c.teachMode()
        self.Message("Robot is entering Teach mode.", 2)

    def EndTeachMode(self) -> None:
        """
        Disable teach mode on the robot.

        Returns
        -------
        None
            This method restores the standard robot control mode.
        """
        self.ResetCurrentTarget()
        self.rtde_c.endTeachMode()
        self.Message("Robot is ending Teach mode.", 2)

    # Status
    def isConnected(self) -> bool:
        """
        Checks if the robot is connected.

        Returns
        -------
        bool
            True if the robot is connected, False otherwise.
        """
        _status_cont = self.rtde_c.isConnected()
        _status_recv = self.rtde_r.isConnected()
        _status_dash = self.dash_c.isConnected()
        print(f"RTDE Control connected  : {_status_cont}")
        print(f"RTDE Receive connected  : {_status_recv}")
        print(f"RTDE Dashboard connected: {_status_dash}")
        return all((_status_cont, _status_recv, _status_dash))

    def Connect(self) -> bool:
        """
        Attempt to reconnect to the robot dashboard interface.

        Returns
        -------
        bool
            ``True`` if the reconnection was successful,
            ``False`` if it failed.

        Notes
        -----
        - Only reconnects the **dashboard client**, not RTDE interfaces.
        - Use this when communication was interrupted.
        - For a full reconnection of all interfaces, use a dedicated restart method.
        """
        return self.dash_c.reconnect()

    def Disconnect(self) -> None:
        """
        Disconnect all active RTDE and dashboard connections.

        This method sequentially terminates communication with:
        - RTDE control interface
        - RTDE receive interface
        - RTDE I/O interface
        - Dashboard client

        Returns
        -------
        None
            This method closes all active RTDE and dashboard connections.

        Notes
        -----
        - After calling this, no robot commands can be executed until a proper reconnection is made.
        - Recommended to call before shutting down the application.
        """
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self.rtde_i.disconnect()
        self.dash_c.disconnect()

    def isReady(self) -> bool:
        """
        Check if the robot is ready for operations.

        This method checks the `_connected` attribute to determine if the robot is connected
        and operational.

        Returns
        -------
        bool
            `True` if the robot is connected and ready for operations, otherwise `False`.
        """
        return self._connected

    def inMotion(self) -> bool:
        """
        Check if the robot is in motion.

        Returns
        -------
        bool
            `True` if the robot is in motion.

        Notes
        -----
        This function will always return `True` in modes other than the standard position mode, e.g. in force and teach mode.
        """
        return not self.rtde_c.isSteady()

    def isAsynchronousMotionRunning(self) -> bool:
        """
        Check if the robot is currently executing an asynchronous operation.

        Returns
        -------
        bool
            `True` if an asynchronous operation is running, `False` otherwise.
        Notes
        -----
        - Asynchronous operations include non-blocking movements or actions initiated through the RTDE interface.
        - This method relies on the RTDE control interface's ability to report the status of asynchronous operations.
        """
        return self.rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning()

    def PowerOn(self) -> None:
        """
        Power on the robot arm.

        Sends a command to the robot controller to enable power to the actuators.
        The robot remains stationary until brakes are released or motion is commanded.

        Returns
        -------
        None

        Notes
        -----
        - This does *not* automatically release the brakes.
        - To enable motion, call :meth:`BreakRelease()` afterward.
        """
        self.dash_c.powerOn()

    def PowerOff(self) -> None:
        """
        Power off the robot arm.

        Disables actuator power and stops any ongoing motion immediately.

        Returns
        -------
        None

        Notes
        -----
        - Should only be used when the robot is in a safe state.
        - The brakes will engage automatically to hold position.
        """
        self.dash_c.powerOff()

    def BreakRelease(self) -> None:
        """
        Release the robot arm brakes.

        Enables movement of the robot after powering on. This allows motion
        commands to be executed. Required after calling :meth:`PowerOn()`.

        Returns
        -------
        None

        Notes
        -----
        - The robot must be powered on first via :meth:`PowerOn()`.
        - Use with caution—ensure the robot is in a safe environment.
        """
        self.dash_c.breakRelease()

    def CheckStatus(self, silent: bool = False) -> tuple:
        """
        Check the current status of the robot.

        Parameters
        ----------
        silent : bool, optional
            If `True`, suppress status messages.

        Returns
        -------
        tuple[URStatusCode, list[str]]
            Status code and a list of short human-readable status descriptions.

        Note
        ----
        When not silent it prints detailed multi-line status & recovery hints.
        """
        lines = []
        code = URStatusCode.OK

        # 1) Connections
        con_errors = []
        if not self.rtde_c.isConnected():
            con_errors.append("RTDE Control: not connected")
        if not self.rtde_r.isConnected():
            con_errors.append("RTDE Receive: not connected")
        if not self.dash_c.isConnected():
            con_errors.append("Dashboard: not connected")

        if con_errors:
            lines.extend(con_errors)
            code = URStatusCode.NOT_CONNECTED

        # 2) Robot mode + safety bits
        try:
            mode = self.rtde_r.getRobotMode()
        except AttributeError:
            mode = self.rtde_r.getRobot()

        try:
            sbits = self.rtde_r.getSafetyStatusBits()
        except Exception:
            sbits = 0

        mode_name = _decode_robot_mode(mode)
        lines.append(f"RobotMode: {mode_name}")
        # lines.append(f"SafetyBits: 0b{sbits:11b}")
        active = _active_safety_flags(sbits, include_bit0=False)
        lines.append("Active safety flags: " + (", ".join(active) if active else "none"))

        # Decode bits
        is_reduced = _sbit(sbits, 1)
        is_protective = _sbit(sbits, 2)
        is_recovery = _sbit(sbits, 3)
        is_safeguard = _sbit(sbits, 4)
        is_sys_estop = _sbit(sbits, 5)
        is_robot_estop = _sbit(sbits, 6)
        is_estop = _sbit(sbits, 7)
        is_violation = _sbit(sbits, 8)
        is_fault = _sbit(sbits, 9)
        is_stopped_saf = _sbit(sbits, 10)

        # 3) Prioritized diagnosis
        if is_sys_estop or is_robot_estop or is_estop:
            code = max(code, URStatusCode.EMERGENCY_STOP)
            lines.append("Recovery: Release all E-stops; reset on pendant/Dashboard.")
        elif is_protective:
            code = max(code, URStatusCode.PROTECTIVE_STOP)
            lines.append("Recovery: Resolve cause; then 'Unlock protective stop'.")
        elif is_safeguard:
            code = max(code, URStatusCode.SAFEGUARD_STOP)
            lines.append("Recovery: Restore safety device; reset safeguard stop.")
        elif is_fault:
            code = max(code, URStatusCode.SAFETY_FAULT)
            lines.append("Recovery: Clear safety fault; check safety I/O; power cycle if needed.")
        elif is_violation:
            code = max(code, URStatusCode.SAFETY_VIOLATION)
            lines.append("Recovery: Move back within safety limits; reset violation.")
        elif is_stopped_saf:
            code = max(code, URStatusCode.SAFETY_STOPPED)
            lines.append("Recovery: Clear safety stop; reset on pendant/Dashboard.")

        if is_recovery:
            code = max(code, URStatusCode.RECOVERY_MODE)
            lines.append("Hint: Robot in recovery mode; perform recovery on pendant.")

        # Robot mode mapping
        if mode == -1:
            code = max(code, URStatusCode.NO_CONTROLLER)
            lines.append("Recovery: Check controller power/cabling.")
        elif mode == 0:
            code = max(code, URStatusCode.DISCONNECTED)
            lines.append("Recovery: Check network and controller power.")
        elif mode == 2:
            code = max(code, URStatusCode.BOOTING)
            lines.append("Info: Booting; wait until idle.")
        elif mode == 3:
            code = max(code, URStatusCode.POWER_OFF)
            lines.append("Recovery: Power on the robot.")
        elif mode in (4, 5):  # POWER_ON / IDLE
            if code == URStatusCode.OK:
                code = URStatusCode.POWER_ON_NO_MOTION
                lines.append("Hint: Powered but idle; enable drives to start motion.")
        elif mode == 6:
            code = max(code, URStatusCode.BACKDRIVE)
            lines.append("Hint: Exit backdrive/hand-guiding to enable motion.")
        elif mode == 8:
            code = max(code, URStatusCode.UPDATING_FIRMWARE)
            lines.append("Info: Firmware update in progress.")

        if is_reduced and code == URStatusCode.OK:
            code = URStatusCode.REDUCED_MODE
            lines.append("Advisory: Reduced mode active.")

        # Normalize to OK if actually RUNNING with no safety stops
        if (mode == 7) and not any([is_protective, is_safeguard, is_sys_estop, is_robot_estop, is_estop, is_fault, is_violation, is_stopped_saf]):
            if code in (URStatusCode.POWER_ON_NO_MOTION, URStatusCode.REDUCED_MODE, URStatusCode.OK):
                code = URStatusCode.OK
                lines.append("OK: Robot RUNNING without safety stops.")

        # If earlier we flagged NOT_CONNECTED but all are connected, clear it
        if code == URStatusCode.NOT_CONNECTED and not con_errors:
            code = URStatusCode.OK

        msg = "\n".join(lines)
        code_int = int(code)
        if code_int == URStatusCode.OK:
            code_desc = []
        else:
            code_desc = [UR_CODE_DESCRIPTIONS.get(code, f"Unknown code {code_int}")]
        if not silent:
            self.Message(f"Robot status:\n{msg}", 1)
        return code, code_desc

    def Check(self, silent: bool = False) -> list:
        """
        Return human-readable robot status messages.

        Parameters
        ----------
        silent : bool, optional
            If `True`, suppress status messages.

        Returns
        -------
        list[str]
            Short human-readable status descriptions.

        Note
        ----
        When not silent it prints detailed multi-line status & recovery hints.
        """
        return self.CheckStatus(silent=silent)[1]

    def ErrorRecovery(self) -> bool:
        """Attempt to recover the robot from common recoverable errors.

        Returns
        -------
        bool
            `True` if recovery succeeded or no recovery was required, else `False`.
        """
        code, code_desc = self.CheckStatus(silent=True)
        if code == 0:
            self.Message("No errors", 1)
            return True
        elif code == URStatusCode.PROTECTIVE_STOP:
            self.Message("Unlocking Protective Stop", 1)
            self.dash_c.unlockProtectiveStop()
            self.rtde_c.reuploadScript()
            return True
        else:
            self.Message(f"Error: {code_desc}", 0)
            return False

    # Movements
    def InternalStopMotion(self) -> None:
        """
        Internal method to stop the robot motion immediately.

        This method sends a stop command to the robot controller to halt all
        ongoing movements. It is intended for use within motion monitoring loops
        when an abort condition is detected.

        Returns
        -------
        None

        Notes
        -----
        - Use with caution; ensure the robot is in a safe state before stopping.
        - This does not perform any additional safety checks or state updates.
        """
        if self._control_strategy.startswith("Cartesian"):
            self.rtde_c.stopL(self._default.TaskDeceleration)
        else:
            self.rtde_c.stopJ(self._default.JointDeceleration)
        self.Stop()

    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """Update joint positions and wait

        Parameters
        ----------
        q : JointConfigurationType
            desired joint positions (nj, )
        qdot : JointVelocityType, optional
            desired joint velocities (nj, )
        trq : JointTorqueType, optional
            additional joint torques (nj, )
        wait : float, optional
            Maximal wait time since last update. Defaults to 0.

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
        # self.rtde_c.waitPeriod(self.t_start)
        _res = self.rtde_c.servoJ(q, self.velocity, self.acceleration, self.tsamp, self.lookahead_time, self.gain)
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q, internal_kinematics=False)
        self._command.x = x
        self._command.v = J @ qdot
        self.Update()
        # self.t_start = self.rtde_c.initPeriod()
        if _res:
            return MotionResultCodes.MOTION_SUCCESS.value
        else:
            return MotionResultCodes.RTDE_ERROR.value

    def JMove(
        self,
        q: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        acc: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        wait: Optional[float] = None,
        traj: Optional[str] = None,
        added_trq: Optional[JointTorqueType] = None,
        min_joint_dist: Optional[float] = None,
        asynchronous: Optional[bool] = None,
        use_internal: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Thread, int]:
        """
        Moves the robot to a specified joint position with specified velocity and trajectory.

        Parameters
        ----------
        q : JointConfigurationType
            Desired joint positions (nj,).
        t : float, optional
            Time for the movement, by default None.
        vel : float, optional
            Desired velocity for the movement, by default None.
        acc = : float, optional
            Acceleration for the movement, by default None (uses default acceleration).
        vel_fac : JointConfigurationType, optional
            Velocity scaling factor for each joint, by default None.
        wait : float, optional
            Time to wait after movement is completed, by default None.
        traj : str, optional
            Trajectory type, by default None.
        added_trq : JointTorqueType, optional
            Additional joint torques, by default None.
        min_joint_dist : float, optional
            Minimum distance to target joint positions for movement execution, by default None.
        asynchronous : bool, optional
            If True, executes the movement asynchronously, by default False.
        use_internal : bool, optional
            If True, uses internal methods for movement execution, by default False.
        **kwargs : dict
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        Union[Thread, int]
            The thread executing the asynchronous movement, or the status code if `asynchronous` is False.

        Notes
        -----
        - If `asynchronous` is set to True, the movement will be executed in a separate thread.
        - The control strategy should be set to "Joint" for this function to work.
        """
        if wait is None:
            wait = self._default.Wait
        if use_internal is None:
            use_internal = self._default.UseInternal

        if use_internal:
            if asynchronous is None:
                asynchronous = True
            _a = "ASYNC " if asynchronous else " "

            if not self.Start():
                return MotionResultCodes.NOT_READY.value

            self.rtde_c.servoStop()
            if self._control_strategy.startswith("Joint"):
                if vel is None and t is not None:
                    vel = np.max(np.abs(q - self._actual.q)) / t
                if vel is None:
                    vel = self._default.JointVelocity
                if acc is None:
                    acc = self._default.JointAcceleration
                self.Message(f"JMove using moveJ {_a} started: {q} with velocity {np.max(vel):.1f}rd/s and acceleration {acc}rd/s2", 2)
                self._command.mode = CommandModeCodes.INTERNAL_JOINT.value
                _tmpstat = self.rtde_c.moveJ(q, vel, acc, asynchronous)
            elif self._control_strategy.startswith("Cartesian"):
                if vel is None and t is not None:
                    _x = self.Kinmodel(q)[0]
                    vel = np.max(np.abs(_x[:3] - self._actual.x[:3])) / t
                if vel is None:
                    vel = self._default.TaskVelocity
                if acc is None:
                    acc = self._default.TaskAcceleration
                self.Message(f"JMove using moveL_FK {_a} started: {q} with velocity {np.max(vel):.1f}m/s and acceleration {acc}m/s2", 2)
                self._command.mode = CommandModeCodes.INTERNAL_JOINT_CARTESIAN.value
                _tmpstat = self.rtde_c.moveL_FK(q, vel, acc, asynchronous)

            if _tmpstat:
                self._command.q = q
                self._command.qdot = np.zeros(self.nj)

                x, J = self.Kinmodel(q)
                self._command.x = x
                self._command.v = np.zeros(6)
                self.Update()
                if asynchronous:
                    while self.rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning():
                        if self._do_motion_check:
                            if self._abort:
                                self.WarningMessage("Motion aborted by user")
                                self.InternalStopMotion()
                                return MotionResultCodes.MOTION_ABORTED.value
                            elif self._do_motion_check and self._motion_check_callback is not None:
                                self._last_status = self._motion_check_callback(self)
                                if self._last_status > 0:
                                    self.WarningMessage("Motion aborted")
                                self.InternalStopMotion()
                                return self._last_status
                        sleep(self.tsamp)
                        self.Update()

                _t_traj = self.simtime()
                while (self.simtime() - _t_traj) < wait:
                    self._sleep(self.tsamp)
                    self.Update()
                self.Message("JMove finished", 2)
                self.Stop()
                return MotionResultCodes.MOTION_SUCCESS.value
            else:
                self.WarningMessage("Internal controller problem - JMove not executed or finished")
                self.Stop()
                return MotionResultCodes.RTDE_ERROR.value

        elif not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JMove not executed")
            return MotionResultCodes.WRONG_STRATEGY.value

        if asynchronous is None:
            asynchronous = False
        if asynchronous:
            self.Message("ASYNC JMove", 2)
            _th = Thread(target=self._JMove, args=(q,), kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "wait": wait, "traj": traj, "added_trq": added_trq, "min_joint_dist": min_joint_dist, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JMove(q, t=t, vel=vel, vel_fac=vel_fac, wait=wait, traj=traj, added_trq=added_trq, min_joint_dist=min_joint_dist, **kwargs)

    def GoTo_T(self, x: Union[Pose3DType, HomogeneousMatrixType], v: Optional[Velocity3DType] = None, FT: Optional[WrenchType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """
        Move the robot to the target pose and velocity in Cartesian space.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., Pose, Transformation matrix).
        v : Velocity3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : WrenchType, optional
            Target force/torque in Cartesian space. Default is a zero wrench vector (6,).
        wait : float, optional
            The time to wait after the movement, by default the sample time (`self.tsamp`).
        **kwargs : dict
            Additional keyword arguments passed to other methods, including `task_space`.

        Returns
        -------
        int
            The status of the move (0 for success, non-zero for error).

        Raises
        ------
        ValueError
            If the provided task space is not supported.

        Notes
        -----
        The method first converts the input `x`, `v`, and `FT` based on the specified task space.
        The robot will be moved using Cartesian control.
        """
        x = x2x(x)
        if v is None:
            v = np.zeros(6)
        else:
            v = vector(v, dim=6)
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)
        if wait is None:
            wait = self.tsamp
        task_space = kwargs.get("task_space", "World")
        if check_option(task_space, "World"):
            x = self.WorldToBase(x)
            v = self.WorldToBase(v, typ="Twist")
            FT = self.WorldToBase(FT, typ="Wrench")
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            x = self.ObjectToWorld(x)
            v = self.ObjectToWorld(v, typ="Twist")
            FT = self.ObjectToWorld(FT, typ="Wrench")
            x = self.WorldToBase(x)
            v = self.WorldToBase(v, typ="Twist")
            FT = self.WorldToBase(FT, typ="Wrench")
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        if self._control_strategy == "JointPositionForced":
            self._command.rx = x
            self._command.rv = v
            self._last_status = self.GoTo_TC(x, v=v, FT=FT, **kwargs)
        else:
            self._last_status = self.GoTo_X(x, v, FT, wait=wait, **kwargs)

        return self._last_status

    def GoTo_X(self, x: Union[Pose3DType, HomogeneousMatrixType], v: Optional[Velocity3DType] = None, FT: Optional[WrenchType] = None, wait: Optional[float] = None, **kwargs: Any) -> int:
        """Update task pose and wait

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., Pose, Transformation matrix).
        v : Velocity3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : WrenchType, optional
            Target force/torque in Cartesian space. Default is a zero wrench vector (6,).
        wait : float, optional
            The time to wait after the movement, by default the sample time (`self.tsamp`).

        The robot will be moved using Cartesian control.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        x = x2x(x)
        _x = map_pose(x=x, out="pa")
        if v is None:
            v = np.zeros(6)
        else:
            v = vector(v, dim=6)
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)
        if wait is None:
            wait = self.tsamp
        self._synchro_control(wait)
        # self.rtde_c.waitPeriod(self.t_start)
        _res = self.rtde_c.servoL(_x, self.velocity, self.acceleration, self.tsamp, self.lookahead_time, self.gain)
        self._command.x = x
        self._command.v = v
        self._command.q = self.IKin(x, self._actual.q)[0]
        self._command.qdot = np.zeros(self.nj)

        self.Update()
        # self.t_start = self.rtde_c.initPeriod()
        if _res:
            return MotionResultCodes.MOTION_SUCCESS.value
        else:
            return MotionResultCodes.RTDE_ERROR.value

    def CMove(
        self,
        x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType],
        t: Optional[float] = None,
        vel: Optional[float] = None,
        acc: Optional[float] = None,
        vel_fac: Optional[float] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        min_pos_dist: Optional[float] = None,
        min_ori_dist: Optional[float] = None,
        asynchronous: Optional[bool] = None,
        use_internal: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Thread, int]:
        """
        Executes a Cartesian move with specified target position, velocity, and trajectory.

        The robot moves its end-effector to a target position with optional velocity, trajectory type,
        and additional force/torque settings.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType]
            The target Cartesian position (7,) or (4,4) or (3,3), where 7 represents position and orientation,
            and (4,4) is a homogeneous transformation matrix.
        t : float, optional
            The duration for the movement, by default None.
        vel : float, optional
            The velocity at which the end-effector moves, by default None.
        acc = : float, optional
            Acceleration for the movement, by default None (uses default acceleration).
        vel_fac : float, optional
            A factor to scale the velocity, by default None.
        traj : str, optional
            The trajectory type, by default None.
        short : bool, optional
            Whether to shorten the path, by default None.
        wait : float, optional
            The wait time after the movement, by default None.
        task_space : str, optional
            The task space reference frame, by default None.
        added_FT : WrenchType, optional
            Additional force/torque to be applied, by default None.
        state : str, optional
            The state of the robot (e.g., "Commanded" or "Actual"), by default "Commanded".
        min_pos_dist : float, optional
            The minimum position distance for stopping, by default None.
        min_ori_dist : float, optional
            The minimum orientation distance for stopping, by default None.
        asynchronous : bool, optional
            Whether the motion should be performed asynchronously, by default False.
        use_internal : bool, optional
            If True, uses internal methods for movement execution, by default False.
        **kwargs : dict
            Additional keyword arguments passed to internal methods.

        Returns
        -------
        Union[Thread, int]
            The thread executing the asynchronous movement, or the status code if synchronous.

        Raises
        ------
        ValueError
            If an unsupported task space or parameter shape is provided.
        """
        if wait is None:
            wait = self._default.Wait

        if use_internal is None:
            use_internal = self._default.UseInternal

        if use_internal:
            _x = x2x(x)
            if check_option(task_space, "Tool"):
                task_space = "Robot"
                T0 = self.GetPose(out="T", task_space="Robot", kinematics=kwargs["kinematics"], state="Commanded")
                _x = t2x(T0 @ x2t(_x))

            if check_option(task_space, "World"):
                _x = self.WorldToBase(_x)
            elif check_option(task_space, "Robot"):
                pass
            elif check_option(task_space, "Object"):
                _x = self.ObjectToWorld(_x)
                _x = self.WorldToBase(_x)
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
            _x = map_pose(x=_x, out="pa")

            if asynchronous is None:
                asynchronous = True
            _a = "ASYNC " if asynchronous else " "

            if not self.Start():
                return MotionResultCodes.NOT_READY.value

            self.rtde_c.servoStop()
            q = self.IKin(_x, q0=self._actual.q)[0]
            if self._control_strategy.startswith("Joint"):
                if vel is None and t is not None:
                    vel = np.max(np.abs(q - self._actual.q)) / t
                if vel is None:
                    vel = self._default.JointVelocity
                if acc is None:
                    acc = self._default.JointAcceleration
                self.Message(f"CMove using moveJ_IK {_a} started: {_x} with velocity {np.max(vel):.1f}rd/s and acceleration {acc}rd/s2", 2)
                self._command.mode = CommandModeCodes.INTERNAL_CARTESIAN_JOINT.value
                _tmpstat = self.rtde_c.moveJ_IK(_x, vel, acc, asynchronous)
            elif self._control_strategy.startswith("Cartesian"):
                if vel is None and t is not None:
                    vel = np.max(np.abs(_x[:3] - self._actual.x[:3])) / t
                if vel is None:
                    vel = self._default.TaskVelocity
                if acc is None:
                    acc = self._default.TaskAcceleration
                self.Message(f"CMove using moveL {_a} started: {_x} with velocity {np.max(vel):.1f}m/s and acceleration {acc}m/s2", 2)
                self._command.mode = CommandModeCodes.INTERNAL_CARTESIAN.value
                _tmpstat = self.rtde_c.moveL(_x, vel, acc, asynchronous)

            if _tmpstat:
                self._command.q = q
                self._command.qdot = np.zeros(self.nj)

                self._command.x = pa2x(_x)
                self._command.v = np.zeros(6)
                self.Update()
                if asynchronous:
                    while self.rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning():
                        if self._do_motion_check:
                            if self._abort:
                                self.WarningMessage("Motion aborted by user")
                                self.InternalStopMotion()
                                return MotionResultCodes.MOTION_ABORTED.value
                            elif self._do_motion_check and self._motion_check_callback is not None:
                                self._last_status = self._motion_check_callback(self)
                                if self._last_status > 0:
                                    self.WarningMessage("Motion aborted")
                                self.InternalStopMotion()
                                return self._last_status
                        sleep(self.tsamp)
                        self.Update()

                _t_traj = self.simtime()
                while (self.simtime() - _t_traj) < wait:
                    self._sleep(self.tsamp)
                    self.Update()

                self.Message("CMove finished", 2)
                self.Stop()
                return MotionResultCodes.MOTION_SUCCESS.value
            else:
                self.WarningMessage("Internal controller problem - CMove not executed or finished")
                self.Stop()
                return MotionResultCodes.RTDE_ERROR.value

        if asynchronous is None:
            asynchronous = False

        if asynchronous:
            self.Message("ASYNC CMove", 2)
            _th = Thread(
                target=self._CMove,
                args=(x,),
                kwargs={
                    "t": t,
                    "vel": vel,
                    "vel_fac": vel_fac,
                    "traj": traj,
                    "short": short,
                    "wait": wait,
                    "task_space": task_space,
                    "added_FT": added_FT,
                    "state": state,
                    "min_pos_dist": min_pos_dist,
                    "min_ori_dist": min_ori_dist,
                    **kwargs,
                },
                daemon=True,
            )
            _th.start()
            return _th
        else:
            return self._CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, min_pos_dist=min_pos_dist, min_ori_dist=min_ori_dist, **kwargs)

    def ForceMode(
        self,
        task_frame: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike],
        selection: Iterable[int],
        FT: WrenchType,
        type: int = 2,
        limits: ArrayLike = (1, 1, 1, 1, 1, 1),
        task_space: Optional[str] = None,
    ) -> None:
        """
        Activate Cartesian force control mode.

        Parameters
        ----------
        task_frame : Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]
            or as a homogeneous transformation matrix.
        selection : Iterable[int]
            ``1`` enables control along the corresponding axis.
        FT : WrenchType
            Desired force and torque values in the task frame.
        type : int, optional
            An integer [1;3] specifying how the robot interprets the force frame.
            1: The force frame is transformed in a way such that its y-axis is
                aligned with a vector pointing from the robot tcp towards the
                origin of the force frame.
            2: The force frame is not transformed.
            3: The force frame is transformed in a way such that its x-axis
               is the projection of the robot tcp velocity vector onto the
               x-y plane of the force frame.
        limits : ArrayLike, optional
            along/about the axis. Default is ``[1, 1, 1, 1, 1, 1]``.
        task_space : str, optional
            Specifies the space of `task_frame`. Supported values: ``"World"``,
            ``"Robot"``, ``"Object"``, ``"Tool"``. If ``None``, the robot's
            default task space is used.

        Raises
        ------
        ValueError
            If input has invalid dimension, type, or unsupported options.

        Notes
        -----
        Force control is executed in compliance mode, allowing interaction with
        the environment. Use ``ForceModeStop()`` to exit force mode safely.

        """
        if not isvector(selection, dim=6):
            raise ValueError("Selection must be 6d vector")
        if not all((a == 1 or a == 0) for a in selection):
            raise ValueError("Selection vector elements must be 0 or 1")

        if not isvector(FT, dim=6):
            raise ValueError("Force/torque must be 6d vector")

        if task_space is None:
            task_space = self._default.TaskSpace
        _x = x2x(task_frame)
        if check_option(task_space, "World"):
            _x = self.BaseToWorld(_x)
        elif check_option(task_space, "Object"):
            _x = self.BaseToWorld(_x)
            _x = self.WorldToObject(_x)
        elif check_option(task_space, "Tool"):
            _x = t2x(self.T * x2t(_x))
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        _frame = map_pose(x=_x, out="pa")

        if not (isscalar(type) and type in [1, 2, 3]):
            raise ValueError("Type parameter mut be integer [1;3]")

        if not isvector(limits, dim=6):
            raise ValueError("Limits must be 6d vector")

        self.Message(f"Force mode activated in direction {selection}", 2)
        self.rtde_c.forceMode(_frame, selection, FT, type, limits)

    def ForceModeDamping(self, damping: float = 0.005) -> None:
        """
        Set damping for force mode.

        Parameters
        ----------
        damping : float, optional
            responsiveness but may reduce stability. Default is ``0.005``.

        Raises
        ------
        ValueError
            If damping is outside allowed range [0, 1].
        """
        if damping < 0 or damping > 1:
            raise ValueError("Damping must be between 0 and 1")
        self.rtde_c.forceModeSetDamping(damping)

    def ForceModeScaling(self, scaling: float = 1.0) -> None:
        """
        Set gain scaling for force mode.

        Parameters
        ----------
        scaling : float, optional
            aggressive force application. Default is ``1.0``.

        Raises
        ------
        ValueError
            If scaling is outside allowed range [0, 2].
        """
        if scaling < 0 or scaling > 2:
            raise ValueError("Damping must be between 0 and 2")
        self.rtde_c.forceModeSetGainScaling(scaling)

    def ForceModeStop(self) -> None:
        """
        Exit Cartesian force control mode.

        Stops force mode execution, clears popups, reuploads control script, and
        resets current motion target.

        Notes
        -----
        - Recommended to call after force interaction is complete.
        - Robot may return to non-compliant mode, ensure stability post-contact.
        """
        self.Message("Force mode deactivated", 2)
        self.rtde_c.forceModeStop()
        sleep(0.1)
        self.dash_c.closePopup()
        sleep(0.1)
        self.rtde_c.reuploadScript()
        self.ResetCurrentTarget()

    def Contacts(self, direction: Iterable[float] = (0, 0, 0, 0, 0, 0)) -> Optional[JointPathType]:
        """
        Detect contact between the robot tool and an external object.

        Parameters
        ----------
        direction : Iterable[float], optional
            of contact detection in the robot base coordinate system.
            - If the first three elements are all zeros (default), contact is detected
            from all directions.
            - Elements 4–6 are reserved but must still be provided for compatibility.

            Defaults to ``(0, 0, 0, 0, 0, 0)``.

        Returns
        -------
        JointPathType or None
            An array of historical joint positions up to the point **just before**
            contact was detected.
            - Returns a position history if contact occurred (i.e., result > 0).
            - Returns ``None`` if **no contact** is detected.

        Notes
        -----
        - Based on RTDE `toolContact()` detection. A positive return value indicates
        the number of cycles back to where the contact started.
        - The returned joint positions represent the robot pose immediately before impact.
        - Only works in real robot mode with enabled force sensing.
        - May require tuned safety settings to trigger detection.
        """
        _res = self.rtde_c.toolContact(direction)
        if _res > 0:
            return self.rtde_c.getActualJointPositionsHistory(_res)
        else:
            return None

    def MoveUntilContact(self, speed: Iterable[float], direction: Iterable[float] = (0, 0, 0, 0, 0, 0), acceleration: float = 0.5) -> bool:
        """
        Move the robot until physical contact is detected.

        The robot moves using Cartesian speed control until a collision is detected
        along a specified direction. After contact, it automatically retracts to
        the initial point of impact.

        Parameters
        ----------
        speed : Iterable[float]
            Desired tool (TCP) spatial velocity `[vx, vy, vz, wx, wy, wz]` in meters per
            second and radians per second. Must be a 6D vector.

        direction : Iterable[float], optional
            Contact detection direction. The first three elements define a 3D vector in
            the robot base frame for detecting contact forces.
            - If all three are **zero** (default), contact is detected from **all directions**.
            - You may also pass `get_target_tcp_speed()` to detect contact in the current
            movement direction.

        acceleration : float, optional
            TCP Cartesian acceleration in m/s². Default is ``0.5``.

        Returns
        -------
        bool
            ``True`` if contact was detected and motion stopped successfully.
            ``False`` only if an internal RTDE failure occurs.

        Raises
        ------
        ValueError
            If `speed` is not a valid 6D vector.

        Notes
        -----
        - The robot automatically retracts slightly upon detection.
        - This method blocks until contact is detected.
        - For asynchronous monitoring, use `startContactDetection()` instead.
        - Contact detection sensitivity depends on robot configuration and safety settings.
        """
        if vector(speed, dim=6):
            return self.rtde_c.moveUntilContact(speed, direction, acceleration)
        else:
            raise ValueError("Speed must be 6D vector")

    def FreeDrive(self, free_axes: Iterable[int] = (1, 1, 1, 1, 1, 1), feature: Iterable[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> bool:
        """
        Enable robot freedrive mode (hand-guiding).

        In freedrive mode, the robot can be physically guided by hand, similar to
        pressing the robot’s *Freedrive* button on the teach pendant. This can be
        useful for manual positioning, demonstration-based programming, or
        fine-tuning poses.

        Parameters
        ----------
        free_axes : Iterable[int], optional
            A 6-element list specifying which axes are free to move:
            - ``1`` → Axis enabled for hand-guiding
            - ``0`` → Axis locked
            The default ``(1, 1, 1, 1, 1, 1)`` enables full 6-DoF movement.

        feature : Iterable[float], optional
            frame for freedrive constraints. Normally `[0, 0, 0, 0, 0, 0]`.
            Only relevant when using constrained freedrive behavior.

        Returns
        -------
        bool
            ``True`` if freedrive mode was successfully activated,
            ``False`` if activation failed.

        Notes
        -----
        - While in freedrive mode, the robot **cannot execute motion commands**
        (e.g., ``moveJ`` or ``moveL``).
        - To exit freedrive mode, call :meth:`EndFreeDrive()`.
        - Axis constraints allow partial hand-guiding (e.g., only in X/Y plane).
        """
        return self.rtde_c.freedriveMode(free_axes, feature)

    def EndFreeDrive(self) -> bool:
        """
        Exit robot freedrive (hand-guiding) mode.

        Restores normal position control mode after freedrive mode was active.
        Once called, the robot will resume standard motion behavior and will
        no longer be compliant to manual manipulation.

        Returns
        -------
        bool
            ``True`` if the robot successfully exited freedrive mode,
            ``False`` if the transition failed.

        Notes
        -----
        - This method should be called after :meth:`FreeDrive()` has been used.
        - Motion commands (e.g., ``moveJ`` or ``moveL``) are only permitted once
        freedrive mode has been disabled.
        - Ensure the robot is in a stable pose before exiting freedrive.
        """
        return self.rtde_c.endFreedriveMode()

    def SetTCP(self, *tcp: TCPType, **kwargs: Any) -> None:
        """
        Set the TCP for the robot, i.e. the transformation from the output
        flange coordinate system to the TCP as a pose.

        Parameters
        ----------
        tcp : TCPType
            The transformation matrix or pose of the gripper TCP. Default is the identity matrix.

        Returns
        -------
        None
        """
        if len(tcp) == 0:
            tcp = np.eye(4)
        robot.SetTCP(self, tcp, frame="Gripper")
        _tcp = map_pose(x2x(self.TCP), out="pa")
        self.rtde_c.setTcp(_tcp)

    def GetTCP(self, source: str = "Gripper", out: str = "T") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]:
        """
        Get the Tool Center Point (TCP) of the robot.

        Parameters
        ----------
        out : str, optional
            The output format of the gripper TCP. Default is "T" (transformation matrix).

        Returns
        -------
        Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]
            The gripper TCP in the specified output format.
        """
        if check_option(source, "Flange"):
            _tcp = map_pose(pa=self.rtde_c.getTCPOffset(), out="T")
            robot.SetTCP(self, _tcp, frame="Flange")
        return map_pose(T=self.TCP, out=out)

    def ZeroFTSensor(self) -> None:
        """
        Zeroes the TCP force/torque measurement from the builtin force/torque
        sensor by subtracting the current measurement from the subsequent.
        """
        self.rtde_c.zeroFtSensor()

    def SetLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[Vector3DType] = None) -> None:
        """
        Set the load properties of the robot.

        Parameters
        ----------
        load : load_params, optional
            The load object to be assigned, by default None.
        mass : float, optional
            The mass of the load, by default None.
        COM : Vector3DType, optional
            The center of mass of the load, by default None.
        """
        if isinstance(load, load_params):
            self.Load = load
        else:
            if mass is not None:
                self.Load.mass = mass
            if COM is not None:
                self.Load.COM = COM
        self.rtde_c.setPayload(self.Load.mass, self.Load.COM)

    def IKin(self, x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike], q0: JointConfigurationType, task_space: Optional[str] = None, **kwargs: Any) -> Tuple[JointConfigurationType, int]:
        """
        Compute inverse kinematics to obtain joint positions for a desired pose.

        This method computes the inverse kinematic solution using the RTDE interface.
        The pose is first transformed according to the specified task space, before
        passing it to the IK solver. The solution closest to the provided joint
        reference `q0` is selected.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType, Vector3DType, QuaternionType, ArrayLike]
            Desired TCP pose. Can be a 4×4 transformation matrix, position/orientation
            vector (size 6 or 7), or other format supported by `x2x`.
        q0 : JointConfigurationType
            Joint position reference (starting estimate). The solution closest to this
            is preferred.
        task_space : str, optional
            Coordinate system interpretation for `x`. Supported:
            ``"World"``, ``"Robot"``, ``"Object"``.
            If ``None``, uses the robot's default task space.
        **kwargs : dict, optional
            Additional IK solver arguments (such as `max_position_error`,
            `max_orientation_error`), if supported by the RTDE implementation.

        Returns
        -------
        JointConfigurationType
            Computed joint values (typically of size 6).
        int
            Status code:
            - ``0`` → IK solution valid
            - ``1`` → IK solution invalid or unstable (popup may have been closed)

        Raises
        ------
        ValueError
            If `task_space` is unsupported.

        Notes
        -----
        - If no TCP is provided to the RTDE interface, the active TCP is used.
        - If IK fails (i.e., returned vector length ≠ 6), the method waits briefly,
        closes any potential robot popup, and returns status ``1``.
        - Returned joint values are not guaranteed to be collision-free or optimal.
        """
        if task_space is None:
            task_space = self._default.TaskSpace

        rx = x2x(x)
        q0 = self.jointvar(q0)

        if check_option(task_space, "World"):
            rx = self.WorldToBase(rx)
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            rx = self.ObjectToWorld(rx)
            rx = self.WorldToBase(rx)
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        _q = rbs_type(self.rtde_c.getInverseKinematics(x2pa(rx), q0))
        if _q.shape[0] == 6:
            return _q, 0
        else:
            sleep(0.1)
            self.ClosePopup()
            return _q, 1

    def ClosePopup(self) -> None:
        """
        Close an active informational popup on the robot controller.

        This method dismisses a non-critical popup dialog on the robot’s teach pendant,
        such as messages requiring user confirmation. It does not handle safety-related
        popups.

        Returns
        -------
        None

        Notes
        -----
        - Use this when non-safety robot messages block program execution.
        - For safety-related dialogs, use :meth:`CloseSafetyPopup()` instead.
        """
        self.dash_c.closePopup()

    def CloseSafetyPopup(self) -> None:
        """
        Close a safety-related popup on the robot controller.

        This method clears critical safety popups—such as protective stops or safety
        alerts—that may require operator acknowledgment before motion resumes.

        Returns
        -------
        None

        Notes
        -----
        - Only call this if you have verified that the robot is in a safe state.
        - May require additional recovery actions depending on the safety condition.
        """
        self.dash_c.closeSafetyPopup()


class ur10(robot_ur_rtde, ur10_spec):
    """RTDE-backed wrapper for the Universal Robots UR10 manipulator."""

    def __init__(self, robot_name: str = "ur10", host: str = "192.168.56.101", **kwargs: Any) -> None:
        """Create a UR10 robot."""
        ur10_spec.__init__(self)
        robot_ur_rtde.__init__(self, robot_name, host=host, **kwargs)

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x", internal_kinematics: Optional[bool] = None) -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute the forward kinematics of the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles as input to the kinematic model.
        tcp : TCPType, optional
            Tool Center Point (TCP) pose or transformation, by default None.
        out : str, optional
            Output format for the result (pose, position, etc.), by default "x".
        internal_kinematics : bool, optional
            If `None` default is used

        Returns
        -------
        tuple
            Pose (or position/rotation) and JacobianType.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        if internal_kinematics is None:
            _ik = self._internal_kinematics
        else:
            _ik = internal_kinematics
        if _ik:
            if tcp.shape == (4, 4):
                _tcp = map_pose(T=tcp, out="pa")
            elif tcp.shape[0] == 3:
                _tcp = map_pose(p=tcp, out="pa")
            elif tcp.shape[0] == 7:
                _tcp = map_pose(x=tcp, out="pa")
            elif tcp.shape[0] == 6:
                _tcp = map_pose(pRPY=tcp, out="pa")
            else:
                raise ValueError("kinmodel: tcp is not SE3")
            _J = kinmodel_ur10(q, tcp=tcp, out=out)[-1]
            if out == "pR":
                _p, _R = map_pose(pa=self.rtde_c.getForwardKinematics(q, _tcp), out=out)
                return _p, _R, _J
            else:
                _x = map_pose(pa=self.rtde_c.getForwardKinematics(q, _tcp), out=out)
                return _x, _J
        else:
            return kinmodel_ur10(q, tcp=tcp, out=out)


class ur10e(robot_ur_rtde, ur10e_spec):
    """RTDE-backed wrapper for the Universal Robots UR10e manipulator."""

    def __init__(self, robot_name: str = "ur10e", host: str = "192.168.56.101", **kwargs: Any) -> None:
        """Create a UR10e robot."""
        ur10e_spec.__init__(self)
        robot_ur_rtde.__init__(self, robot_name, host=host, **kwargs)

    def Kinmodel(self, q: Optional[JointConfigurationType] = None, tcp: Optional[TCPType] = None, out: str = "x", internal_kinematics: Optional[bool] = None) -> Union[Tuple[Pose3DType, JacobianType], Tuple[HomogeneousMatrixType, JacobianType], Tuple[Vector3DType, RotationMatrixType, JacobianType]]:
        """
        Compute the forward kinematics of the robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint angles as input to the kinematic model.
        tcp : TCPType, optional
            Tool Center Point (TCP) pose or transformation, by default None.
        out : str, optional
            Output format for the result (pose, position, etc.), by default "x".
        internal_kinematics : bool, optional
            If `None` default is used

        Returns
        -------
        tuple
            Pose (or position/rotation) and JacobianType.
        """
        if q is None:
            q = np.copy(self._actual.q)
        if tcp is None:
            tcp = self.TCP
        if internal_kinematics is None:
            _ik = self._internal_kinematics
        else:
            _ik = internal_kinematics
        if _ik:
            if tcp.shape == (4, 4):
                _tcp = map_pose(T=tcp, out="pa")
            elif tcp.shape[0] == 3:
                _tcp = map_pose(p=tcp, out="pa")
            elif tcp.shape[0] == 7:
                _tcp = map_pose(x=tcp, out="pa")
            elif tcp.shape[0] == 6:
                _tcp = map_pose(pRPY=tcp, out="pa")
            else:
                raise ValueError("kinmodel: tcp is not SE3")
            _J = kinmodel_ur10e(q, tcp=tcp, out=out)[-1]
            if out == "pR":
                _p, _R = map_pose(pa=self.rtde_c.getForwardKinematics(q, _tcp), out=out)
                return _p, _R, _J
            else:
                _x = map_pose(pa=self.rtde_c.getForwardKinematics(q, _tcp), out=out)
                return _x, _J
        else:
            return kinmodel_ur10e(q, tcp=tcp, out=out)


if __name__ == "__main__":
    from robotblockset.transformations import rot_x

    # Run robot
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    r = ur10(host="192.168.56.101")
    r._verbose = 3
    print("TCP:", r.GetTCP())
    r.SetTCP([0, 0, 0.11752])
    print("TCP:", r.GetTCP())
    print("Robot:", r.Name)
    print("q: ", r.q)
    print("x: ", r.x)

    r.JMove(r.q_home, 5)
    r.CMoveFor([0, 0, 0.1], 4)
    r.ForceMode(np.eye(4), [0, 0, 1, 0, 0, 0], [0, 0, -10, 0, 0, 0])
    r.Wait(2)
    r.ForceMode(np.eye(4), [0, 0, 1, 0, 0, 0], [0, 0, 10, 0, 0, 0])
    r.Wait(2)
    r.ForceModeStop()

    r.CMoveFor([0, 0, -0.1], 4)
    x = r.Kinmodel(out="x")[0]
    print("Robot pose:\n ", x)
    J = r.Jacobi()
    print("Robot Jacobian:\n ", J)

    print("Strategy:", r.GetStrategy())
    print(
        "GetPose(task_space='Robot',kinematics='Robot','State','Commanded'): ",
        r.GetPose(task_space="Robot", kinematics="Robot", state="Commanded"),
    )
    print(
        "GetPose(task_space='Robot',kinematics='Calculated','State','Commanded'): ",
        r.GetPose(task_space="Robot", kinematics="Calculated", state="Commanded"),
    )
    print("GetPose(task_space='Robot',kinematics='Robot'): ", r.GetPose(task_space="Robot", kinematics="Robot"))
    print("GetPose(task_space='Robot',kinematics='Calculated'): ", r.GetPose(task_space="Robot", kinematics="Calculated"))
    print(
        "GetPose(task_space='World',kinematics='Robot','State','Commanded'): ",
        r.GetPose(task_space="World", kinematics="Robot", state="Commanded"),
    )
    print(
        "GetPose(task_space='World',kinematics='Calculated','State','Commanded'): ",
        r.GetPose(task_space="World", kinematics="Calculated", state="Commanded"),
    )
    print("GetPose(task_space='World',kinematics='Robot'): ", r.GetPose(task_space="World", kinematics="Robot"))
    print("GetPose(task_space='World',kinematics='Calculated'): ", r.GetPose(task_space="World", kinematics="Calculated"))

    print("IKin:", r.IKin(map_pose(p=[0.5, 0.2, 0.5], Q=rot_x(np.pi)), r.q_home))

    print("Pose: ", r.GetPose())

    r.GetVel()

    print("FT: ", r.GetFT())
