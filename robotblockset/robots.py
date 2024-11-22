from abc import abstractmethod
import numpy as np
from time import perf_counter, sleep
from threading import Timer, Semaphore, Thread
import platform
import copy

from robotblockset.tools import _struct, _load, rbs_object, rbs_type, check_option, vector, isscalar, isvector, matrix, ismatrix, grad, normalize
from robotblockset.trajectories import jtraj, ctraj, carctraj, interpPath, interpCartesianPath, gradientPath, gradientCartesianPath, uniqueCartesianPath
from robotblockset.RBF import decodeRBF, decodeCartesianRBF
from robotblockset.transformations import map_pose, q2r, r2q, x2x, x2t, t2x, v2s, xerr, qerr, terr, world2frame

flag = True


def _dummy():
    global flag
    flag = False


class _actual(_struct):
    def __init__(self):
        self.q = None
        self.qdot = None
        self.trq = None
        self.x = None
        self.v = None
        self.trq = None
        self.FT = None
        self.trqExt = None


class _command(_struct):
    def __init__(self):
        self.q = None
        self.qdot = None
        self.trq = None
        self.x = None
        self.v = None
        self.FT = None
        self.u = None
        self.data = None
        self.mode = None


class _default(_struct):
    def __init__(self):
        self.State = "Actual"
        self.TaskSpace = "World"
        self.TaskPoseForm = "Pose"
        self.TaskOriForm = "Quaternion"
        self.TaskVelForm = "Twist"
        self.TaskFTForm = "Wrench"
        self.TaskErrForm = "Task"
        self.Kinematics = "Robot"
        self.TCPFrame = "Gripper"
        self.Source = "Robot"
        self.Strategy = "JointPosition"
        self.NullSpaceTask = "JointLimits"
        self.RotDirShort = True
        self.Traj = "Poly"
        self.TaskDOF = np.ones(6)
        self.VelocityScaling = 1  # Scale factor for velocity
        self.MinJointDist = 0.01
        self.MinPosDist = 0.0001
        self.MinOriDist = 0.001
        self.PosErr = 0.0001
        self.OriErr = 0.001
        self.AddedTrq = None  # Added joint torques (depends on nj, which is not yet defined)
        self.AddedFT = np.zeros(6)  # Added end-effector FT
        self.Kp = 10  # Kinematic controller: position P gain
        self.Kff = 10  # Kinematic controller: velocity FF gain
        self.Kns = 1  # Kinematic controller: null-space gain
        self.Kns0 = 0.1  # Kinematic controller: null-space gain for joint limits
        self.Wait = 0.02
        self.UpdateTime = 1.0
        self.TrajSampTimeFac = 5


class robot(rbs_object):
    def __init__(self, **kwargs):
        rbs_object.__init__(self)
        self.Name = "Robot"
        self.tsamp = 0.01  # sampling rate
        self.TCP = np.eye(4)  # robot TCP transformation matrix
        self.TBase = np.eye(4)  # robot base transformation matrix
        self.TObject = np.eye(4)  # object transformation matrix
        self.TCPGripper = np.eye(4)  # gripper TCP transformation matrix
        self.Load = _load()  # Load
        self.Gripper = None  # gripper object attached to robot
        self.FTSensor = None  # F/T sensor attached to robot
        self.FTSensorFrame = np.eye(4)  # F/T sensor transformation matrix from EE
        self.Platform = None  # B ase platform object to which robot is attached
        self.user = None  # user data or object
        self.Tag = None

        self._t0 = 0  # initial time
        self._tt = 0  # actual robot time
        self._tt0 = 0  # initial robot time
        self._robottime = 0  # time from simulator
        self._connected = False
        self._last_update = -100
        self._last_control_time = -100
        self._command = _command()  # Commended values
        self._actual = _actual()  # Measured values
        self._default = _default()  # Default options
        self._do_update = True  # Enables state update and optional callback
        self._do_capture = False  # Enables calling calback function
        self._capture_callback = None  # Callback function in Update
        self._do_motion_check = False  # Enables checks during motion
        self._motion_check_callback = None  # Callback executed during motion
        self._motion_error = None  # status of current motion
        self._control_strategy = "JointPosition"  # Control strategy
        self._semaphore = Semaphore(1)  # used for asynchronous motion
        self._threads_active = platform.system() == "Linux"  # used in sinhro for sleep()
        self._abort = False  # abort current motion

    def jointvar(self, x):
        x = np.asarray(x)
        if x.shape[-1] == self.nj:
            return x
        else:
            raise TypeError("Parameter has not proper shape")

    def spatial(self, x):
        x = rbs_type(x)
        if x.shape == (7,) or x.shape == (4, 4) or x.shape == (3,) or x.shape == (4,) or x.shape == (3, 3) or x.shape == (6,):
            return x
        elif x.shape == (3, 4):
            x = np.vstack(x, np.array([0, 0, 0, 1]))
            return x
        else:
            raise TypeError("Parameter has not proper shape")

    def simtime(self):
        return perf_counter()

    def _sinhro_control(self, wait):
        dt = self.simtime() - self._last_control_time
        while dt < wait:
            if self._threads_active:
                sleep((wait - dt) * 0.5)
            dt = self.simtime() - self._last_control_time
        # print(f"Time:{self.simtime():8.3f} Robot:{self._robottime:8.3f}  dt:{dt*1000.:6.2f}ms")
        self._last_control_time = self.simtime()

    def UseThreads(self, active):
        self._threads_active = active

    def SetTsamp(self, tsamp):
        self.tsamp = tsamp
        self._default.Wait = tsamp

    def ResetTime(self):
        self.GetState()
        self._t0 = self.simtime()
        self._tt0 = self._tt
        self.Update()

    def isReady(self):
        return self._connected

    def isActive(self):
        return True

    @property
    def Time(self):
        return self.simtime() - self._t0

    @property
    def t(self):
        return self._tt - self._tt0

    @property
    def command(self):
        return copy.deepcopy(self._command)

    @property
    def actual(self):
        return copy.deepcopy(self._actual)

    @property
    def q(self):
        return copy.deepcopy(self._actual.q)

    @property
    def qdot(self):
        return copy.deepcopy(self._actual.qdot)

    @property
    def trq(self):
        return copy.deepcopy(self._actual.trq)

    @property
    def trqExt(self):
        return copy.deepcopy(self._actual.trqExt)

    @property
    def x(self):
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="x"))

    @property
    def p(self):
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="p"))

    @property
    def Q(self):
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="Q"))

    @property
    def R(self):
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="R"))

    @property
    def T(self):
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="T"))

    @property
    def v(self):
        return copy.deepcopy(self.GetVel(state="Actual", task_space="World", out="Twist"))

    @property
    def pdot(self):
        return copy.deepcopy(self.GetVel(state="Actual", task_space="World", out="Linear"))

    @property
    def w(self):
        return copy.deepcopy(self.GetVel(state="Actual", task_space="World", out="Angular"))

    @property
    def FT(self):
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Wrench"))

    @property
    def F(self):
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Force"))

    @property
    def Trq(self):
        return copy.deepcopy(self.GetFT(state="Actual", task_space="World", out="Torque"))

    @property
    def q_ref(self):
        return copy.deepcopy(self._command.q)

    @property
    def qdot_ref(self):
        return copy.deepcopy(self._command.qdot)

    @property
    def x_ref(self):
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="x"))

    @property
    def p_ref(self):
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="p"))

    @property
    def Q_ref(self):
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="q"))

    @property
    def R_ref(self):
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="R"))

    @property
    def T_ref(self):
        return copy.deepcopy(self.GetPose(state="Command", task_space="World", out="T"))

    @property
    def v_ref(self):
        return copy.deepcopy(self.GetVel(state="Command", task_space="World", out="Twist"))

    @property
    def pdot_ref(self):
        return copy.deepcopy(self.GetVel(state="Command", task_space="World", out="Linear"))

    @property
    def w_ref(self):
        return copy.deepcopy(self.GetVel(state="Command", task_space="World", out="Angular"))

    @property
    def FT_ref(self):
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Wrench"))

    @property
    def F_ref(self):
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Force"))

    @property
    def Trq_ref(self):
        return copy.deepcopy(self.GetFT(state="Command", task_space="World", out="Torque"))

    @property
    def q_err(self):
        return self.q_ref - self.q

    @property
    def qdot_err(self):
        return self.qdot_ref - self.qdot

    @property
    def x_err(self):
        return xerr(self.x_ref, self.x)

    @property
    def p_err(self):
        return self.p_ref - self.p

    @property
    def Q_err(self):
        return qerr(self.Q_ref, self.Q)

    @property
    def R_err(self):
        return self.R_ref @ self.R.T

    @property
    def T_err(self):
        return terr(self.T_ref, self.T)

    @property
    def v_err(self):
        return self.v_ref - self.v

    @property
    def pdot_err(self):
        return self.pdot_ref - self.pdot

    @property
    def w_err(self):
        return self.w_ref - self.w

    # Initialization and update
    def InitObject(self):
        self._command.q = np.zeros(self.nj)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.u = np.zeros(self.nj)
        self._command.x = np.zeros(7)
        self._command.v = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.data = None
        self._command.mode = 0
        self._actual.q = np.zeros(self.nj)
        self._actual.qdot = np.zeros(self.nj)
        self._actual.trq = np.zeros(self.nj)
        self._actual.x = np.zeros(7)
        self._actual.v = np.zeros(6)
        self._actual.FT = np.zeros(6)
        self._actual.trqExt = np.zeros(self.nj)
        self._default.AddedTrq = np.zeros(self.nj)

    def Init(self):
        self.InitObject()
        self.GetState()
        self.ResetCurrentTarget()
        self.ResetTime()
        self.Message("Initialized", 2)

    def GetState(self):
        self._tt = self.simtime()
        self._last_update = self.simtime()

    def Update(self):
        if self._do_update:
            self.GetState()
            if self._do_capture and self._capture_callback is not None:
                self._capture_callback(self)

    def EnableUpdate(self):
        self._do_update = True

    def DisableUpdate(self):
        self._do_update = False

    def GetUpdateStatus(self):
        return self._do_update

    def ResetCurrentTarget(self, do_move=False, **kwargs):
        self.GetState()
        self._command.q = copy.deepcopy(self._actual.q)
        self._command.qdot = np.zeros(self.nj)
        self._command.trq = np.zeros(self.nj)
        self._command.x = copy.deepcopy(self._actual.x)
        self._command.v = np.zeros(6)
        self._command.FT = np.zeros(6)
        self._command.trq = np.zeros(self.nj)
        sleep(0.1)
        self.Update()
        if do_move:
            self.Message("Moving to actual configuration", 2)
            self.Start()
            self._semaphore.acquire()
            if self._control_strategy.startswith("Joint"):
                tmperr = self.GoTo_q(self._command.q, np.zeros(self.nj), np.zeros(self.nj), 1)
            else:
                tmperr = self.GoTo_T(self._command.x, wait=1, **kwargs)
            self.Stop()
            self._semaphore.release()
            return tmperr
        else:
            return 0

    def ResetTaskTarget(self):
        _x, _J = self.Kinmodel(self._command.q)
        self._command.x = _x
        self._command.v = _J @ self._command.qdot

    # Get joint variables
    def GetJointPos(self, state=None):
        if state is None:
            state = self._default.State
        if check_option(state, "Actual"):
            self.GetState()
            return copy.deepcopy(self._actual.q)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.q)
        else:
            raise ValueError(f"State '{state}' not supported")

    def GetJointVel(self, state=None):
        if state is None:
            state = self._default.State
        if check_option(state, "Actual"):
            self.GetState()
            return copy.deepcopy(self._actual.qdot)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.qdot)
        else:
            raise ValueError(f"State '{state}' not supported")

    def GetJointTrq(self, state=None):
        if state is None:
            state = self._default.State
        if check_option(state, "Actual"):
            self.GetState()
            return copy.deepcopy(self._actual.trq)
        elif check_option(state, "Commanded"):
            return copy.deepcopy(self._command.trq)
        else:
            raise ValueError(f"State '{state}' not supported")

    # Get task space variables
    def GetPose(self, out=None, task_space=None, kinematics=None, state=None):
        """Get robot end-effector pose

        Parameters
        ----------
        out : str, optional
            Output form, by default "x" ("Pose")
        task_space : str, optional
            Task space frame, by default "World"
        kinematics : str, optional
            Used kinematics: read from robot or calculated from joint variables, by default "Robot"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            End-effector pose
        """
        if out is None:
            out = self._default.TaskPoseForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        self.GetState()
        if check_option(kinematics, "Calculated"):
            _x, _ = self.Kinmodel(self.GetJointPos(state=state))
        elif check_option(kinematics, "Robot"):
            if check_option(state, "Actual"):
                _x = copy.deepcopy(self._actual.x)
            elif check_option(state, "Commanded"):
                _x = copy.deepcopy(self._command.x)
            else:
                raise ValueError(f"State '{state}' not supported in GetPose")
        else:
            raise ValueError(f"Kinematics calculation '{kinematics}' not supported in GetPose")
        if check_option(task_space, "World"):
            _x = self.BaseToWorld(_x)
        elif check_option(task_space, "Object"):
            _x = self.BaseToWorld(_x)
            _x = self.WorldToObject(_x)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(x=_x, out=out)

    def GetPos(self, out="p", task_space=None, kinematics=None, state=None):
        """Get robot end-effector position

        Parameters
        ----------
        out : str, optional
            Output form, by default "p" ("Position")
        task_space : str, optional
            Task space frame, by default "World"
        kinematics : str, optional
            Used kinematics: read from robot or calculated from joint variables, by default "Robot"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            End-effector position (3,)
        """
        if out in ["Position", "p"]:
            return self.GetPose(out=out, task_space=task_space, kinematics=kinematics, state=state)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetPos")

    def GetOri(self, out="Q", task_space=None, kinematics=None, state=None):
        """Get robot end-effector orientation

        Parameters
        ----------
        out : str, optional
            Output form, by default "Q" ("Quaternion")
        task_space : str, optional
            Task space frame, by default "World"
        kinematics : str, optional
            Used kinematics: read from robot or calculated from joint variables, by default "Robot"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            End-effector orientation (4,) or (3,3)
        """
        if out in ["Quaternion", "Q", "RotationMatrix", "R"]:
            return self.GetPose(out=out, task_space=task_space, kinematics=kinematics, state=state)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetOri")

    def GetVel(self, out="Twist", task_space=None, kinematics=None, state=None):
        """Get robot end-effector velocity

        Parameters
        ----------
        out : str, optional
            Output form, by default "Twist"
        task_space : str, optional
            Task space frame, by default "World"
        kinematics : str, optional
            Used kinematics: read from robot or calculated from joint variables, by default "Robot"
        state : str, optional
            Variable state, by default "Actual"

        Returns
        -------
        array of floats
            End-effector velocity (6,) or (3,)
        """
        if out is None:
            out = self._default.TaskVelForm
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        self.GetState()
        if check_option(kinematics, "Calculated"):
            if check_option(state, "Actual") or check_option(state, "Commanded"):
                _qq = self.GetJointPos(state=state)
                _qqdot = self.GetJointVel(state=state)
            else:
                raise ValueError(f"State '{state}' not supported")
            _J = self.Jacobi(_qq)
            if check_option(task_space, "World"):
                _vv = self.BaseToWorld(_J @ _qqdot)
            elif check_option(task_space, "Object"):
                _vv = self.BaseToWorld(_J @ _qqdot)
                _vv = self.WorldToObject(_vv)
            elif check_option(task_space, "Robot"):
                _vv = _J @ _qqdot
            else:
                raise ValueError(f"Task space '{state}' not supported")
        elif check_option(kinematics, "Robot"):
            if check_option(state, "Actual"):
                _vv = copy.deepcopy(self._actual.v)
            elif check_option(state, "Commanded"):
                _vv = copy.deepcopy(self._command.v)
            else:
                raise ValueError(f"State '{state}' not supported")
            if check_option(task_space, "World"):
                pass
            elif check_option(task_space, "Object"):
                _vv = self.WorldToObject(_vv)
            elif check_option(task_space, "Robot"):
                _vv = self.WorldToBase(_vv)
            else:
                raise ValueError(f"Task space '{state}' not supported")
        else:
            raise ValueError(f"Kinematics calculation '{kinematics}' not supported")
        if check_option(out, "Twist"):
            return _vv
        elif check_option(out, "Linear"):
            return _vv[:3]
        elif check_option(out, "Angular"):
            return _vv[3:]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    def GetFT(self, out=None, source=None, task_space=None, kinematics=None, state=None, avg_time=0):
        if out is None:
            out = self._default.TaskFTForm
        if source is None:
            source = self._default.Source
        if state is None:
            state = self._default.State
        if task_space is None:
            task_space = self._default.TaskSpace
        if kinematics is None:
            kinematics = self._default.Kinematics
        self.GetState()
        if check_option(state, "Actual"):
            _R = q2r(self._actual.x[3:])
            if check_option(source, "External"):
                if self.FTSensor:
                    _FT = self.FTSensor.GetFT(avg_time=avg_time)
                    _FT2TCP = np.linalg.pinv(self.FTSensorFrame) @ self.TCP
                    Rsensor = self.R @ _FT2TCP[:3, :3].T
                    _FT -= -(-9.81 * self.FTSensor.Load.mass * np.hstack((Rsensor[2, :], v2s(self.FTSensor.Load.COM) @ Rsensor[2, :])))
                    _FT = world2frame(_FT, _FT2TCP, typ="Wrench")
                    # in EE (tool) CS
                else:
                    raise ValueError(f"No FT sensor assigned to robot")
            elif check_option(source, "Robot"):
                if check_option(kinematics, "Robot"):
                    _FT = self._actual.FT  # in EE (tool) CS
                elif check_option(kinematics, "Calculated"):
                    _J = self.Jacobi()
                    _FT = np.linalg.pinv(_J.T) @ self._actual.trqExt  # in robot CS
                    _FT = np.hstack((_R.T @ _FT[:3], _R.T @ _FT[3:]))  # rotate from robot CS to EE (tool) CS
                else:
                    raise ValueError(f"Kinematics calculation '{kinematics}' not supported")
            else:
                raise ValueError(f"Source '{source}' not supported")
            if check_option(task_space, "World"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))  # rotate from EE (tool) CS to robot CS
                _FT = self.BaseToWorld(_FT, typ="Wrench")
            elif check_option(task_space, "Object"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))  # rotate from EE (tool) CS to robot CS
                _FT = self.BaseToWorld(_FT, typ="Wrench")
                _FT = self.WorldToObject(_FT, typ="Wrench")
            elif check_option(task_space, "Robot"):
                _FT = np.hstack((_R @ _FT[:3], _R @ _FT[3:]))  # rotate from EE (tool) CS to robot CS
            elif check_option(task_space, "Tool"):
                pass
            else:
                raise ValueError(f"Task space '{state}' not supported")
        elif check_option(state, "Commanded"):
            _FT = copy.deepcopy(self._command.FT)  # in robot CS
            if check_option(task_space, "World"):
                _FT = self.BaseToWorld(_FT, typ="Wrench")
            elif check_option(task_space, "Object"):
                _FT = self.BaseToWorld(_FT, typ="Wrench")
                _FT = self.WorldToObject(_FT, typ="Wrench")
            elif check_option(task_space, "Robot"):
                pass
            elif check_option(task_space, "Tool"):
                _FT = np.hstack((_R.T @ _FT[:3], _R.T @ _FT[3:]))  # rotate from EE (tool) CS to robot CS
            else:
                raise ValueError(f"Task space '{state}' not supported")
        else:
            raise ValueError(f"State '{state}' not supported")

        if check_option(out, "Wrench"):
            return _FT
        elif check_option(out, "Force"):
            return _FT[:3]
        elif check_option(out, "Torque"):
            return _FT[3:]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    # Joint space motion
    def GoToActual(self, **kwargs):
        self.ResetCurrentTarget(do_move=True, **kwargs)

    def GoTo_q(self, q, qdot, trq, wait):
        self._command.q = q
        self._command.qdot = qdot
        self._command.trq = trq
        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = J @ qdot

        self._actual.q = q
        self._actual.qdot = qdot
        self._actual.trq = trq
        self._actual.x = x
        self._actual.v = J @ qdot
        sleep(wait)

    def GoTo_qtraj(self, q, qdot, qddot, time):
        _time = vector(time)
        _n = _time.shape[0]
        _q = matrix(q, shape=(_n, self.nj))
        _qdot = matrix(qdot, shape=(_n, self.nj))
        _qddot = matrix(qddot, shape=(_n, self.nj))
        self.WarningMessage("Joint trajectory controller not implemented!")

    def _loop_joint_traj(self, qi, qdoti, trq, time, wait=0):
        if self._control_strategy in ["JointPositionTrajectory"]:
            self.GoTo_qtraj(qi, qdoti, np.zeros(self.nj), time)
            self.Update()

            _t_traj = self.simtime()
            while (self._motion_error is None) or ((self.simtime() - _t_traj) < time[-1]):
                self._last_control_time = self.simtime()
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self.StopMotion()
                    return 99
                elif self._do_motion_check and self._motion_check_callback is not None:
                    tmperr = self._motion_check_callback(self)
                    if tmperr > 0:
                        self.WarningMessage("Motion aborted")
                        self.StopMotion()
                    return tmperr
                sleep(self.tsamp)
                self.Update()

            t_traj = self.simtime()
            while (self._motion_error is None) or ((self.simtime() - _t_traj) < wait):
                sleep(self.tsamp)
                self.Update()

            if (self._motion_error is not None) and (self._motion_error != 0):
                self.WarningMessage(f"Motion aborted due to motion controller error ({self._motion_error})")
                return self._motion_error
            else:
                print("End status:", self._motion_error, (self.simtime() - _t_traj), time[-1])
                return 0
        else:
            for qt, qdt in zip(qi, qdoti):
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self._semaphore.release()
                    self.StopMotion()
                    return 99
                elif self._do_motion_check and self._motion_check_callback is not None:
                    tmperr = self._motion_check_callback(self)
                    if tmperr > 0:
                        self.WarningMessage("Motion aborted")
                        self._semaphore.release()
                        self.StopMotion()
                        return tmperr
                self.GoTo_q(qt, qdt, trq, self.tsamp)
            self.GoTo_q(qi[-1, :], np.zeros(self.nj), trq, wait)
            return 0

    def JMove(self, q, t=None, vel=None, vel_fac=None, wait=None, traj=None, added_trq=None, min_joint_dist=None, asynchronous=False):
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JMove not executed")
            return 1
        if asynchronous:
            self.Message("ASYNC JMove", 2)
            _th = Thread(target=self._JMove, args=(q), kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "wait": wait, "traj": traj, "added_trq": added_trq, "min_joint_dist": min_joint_dist}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JMove(q, t=t, vel=vel, vel_fac=vel_fac, wait=wait, traj=traj, added_trq=added_trq, min_joint_dist=min_joint_dist)

    def _JMove(self, q, t=None, vel=None, vel_fac=None, wait=None, traj=None, added_trq=None, min_joint_dist=None):
        """_summary_

        Parameters
        ----------
        q : _type_
            _description_
        t : _type_
            _description_
        traj : _type_, optional
            _description_, by default None
        wait : _type_, optional
            _description_, by default None
        added_trq : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        self._semaphore.acquire()
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(trq, dim=self.nj)
        if min_joint_dist is None:
            min_joint_dist = self._default.MinJointDist

        q = self.jointvar(q)
        if self.CheckJointLimits(q):
            raise ValueError(f"Joint positions out of range")

        dist = np.abs(q - self._command.q)
        if all(np.abs(dist) < min_joint_dist):
            self.Message("JMove not executed - close to target", 2)
            self._semaphore.release()
            return 0

        if t is not None:
            if not isscalar(t) or t <= 0:
                raise ValueError(f"Time must be non-negative scalar")
            elif t <= 10 * self.tsamp:
                t = None
        if t is None:
            _time = np.arange(0.0, 1 + self.tsamp, self.tsamp)
            if vel is None:
                if vel_fac is None:
                    vel_fac = self._default.VelocityScaling
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=self.nj)
                _vel = self.qdot_max * vel_fac
            else:
                if isscalar(vel):
                    _vel = np.ones(self.nj) * vel
                else:
                    _vel = vector(vel, dim=self.nj)
            _vel = np.clip(_vel, 0, self.qdot_max)
            self.Message(f"JMove started: {q} with velocity {100 * np.max(_vel / self.qdot_max):.1f}%", 2)
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.qdot_max
            self.Message(f"JMove started to: {q} in {_time[-1]:.1f}s", 2)

        self.Start()
        self._command.mode = 1
        tmperr = 0

        q0 = self.GetJointPos(state="Commanded")
        qi, qdoti, _ = jtraj(q0, q, _time, traj=traj)
        _fac = np.max(np.max(np.abs(qdoti), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _time = np.arange(0.0, (_time[-1] * _fac) + self.tsamp, self.tsamp)
            qi, qdoti, _ = jtraj(q0, q, _time, traj=traj)
        tmperr = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait)

        self.Stop()
        self.Message("JMove finished", 2)
        self._semaphore.release()
        return tmperr

    def JMoveFor(self, dq, t=None, vel=None, vel_fac=None, state="Commanded", traj=None, wait=None, added_trq=None, asynchronous=False):
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JMoveFor not executed")
            return 1
        dq = self.jointvar(dq)
        q0 = self.GetJointPos(state=state)
        q = q0 + dq
        self.Message("JMoveFor -> JMove", 2)
        tmperr = self.JMove(q, t=t, vel=vel, vel_fac=vel_fac, traj=traj, wait=wait, added_trq=added_trq, asynchronous=asynchronous)
        return tmperr

    def JLine(self, q, t=None, vel=None, vel_fac=None, state="Commanded", wait=None, added_trq=None, asynchronous=False):
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JLine not executed")
            return 1
        self.Message("JLine -> JMove", 2)
        tmperr = self.JMove(q, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", wait=wait, added_trq=added_trq, asynchronous=asynchronous)
        return tmperr

    def JPath(self, path, t, wait=None, traj=None, added_trq=None, asynchronous=False):
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JPath not executed")
            return 1
        if asynchronous:
            _th = Thread(target=self._JPath, args=(path, t), kwargs={"wait": wait, "traj": traj, "added_trq": added_trq}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JPath(path, t, wait=wait, traj=traj, added_trq=added_trq)

    def _JPath(self, path, t, wait=None, traj=None, added_trq=None):
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(trq, dim=self.nj)

        tmperr = 0
        # if not isscalar(t) or t <= 0:
        #    raise ValueError(f"Time must be non-negative scalar")
        if isscalar(t) or t[0] == 0:
            _dist = np.abs(path[0, :] - self.q_ref)
            _qerr = np.max(_dist / self.qdot_max) * 2
            if _qerr > 0.01:
                self.Message(f"Move to path -> JPath ({_dist})", 2)
                tmperr = self._JMove(path[0, :], max(_qerr, 0.2), wait=0, added_trq=added_trq)
                if tmperr > 0:
                    self.WarningMessage("Robot did not moved to path start")
                    return tmperr

        _n = np.shape(path)[0]
        if not isscalar(t) and len(t) == _n:
            if self._control_strategy in ["JointPositionTrajectory"]:
                _time = t
                qi = path
                qdoti = gradientPath(qi, _time)
            else:
                if t[0] > 0:
                    path = np.vstack((self.q_ref, path))
                    t = np.concatenate(([0], t))
                _time = np.arange(0.0, max(t) + self.tsamp, self.tsamp)
                qi = interpPath(t, path, _time)
                qdoti = gradientPath(qi, _time)
        else:
            if not isscalar(t):
                t = max(t)
            _s = np.linspace(0, t, _n)
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            qi = interpPath(_s, path, _time)
            qdoti = gradientPath(qi, _time)
            _fac = np.max(np.max(np.abs(qdoti), axis=0) / self.qdot_max)
            if _fac > 1:
                _s = np.linspace(0, t * _fac + self.tsamp, _n)
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                qi = interpPath(_s, path, _time)
                qdoti = gradientPath(qi, _time)

        self.Message(f"JPath started: {path.shape[0]} points in {t}s ", 2)
        self.Start()
        self._command.mode = 1.1
        self._semaphore.acquire()

        tmperr = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait)

        self.Stop()
        self.Message("JPath finished", 2)
        self._semaphore.release()
        return tmperr

    def JRBFPath(self, pathRBF, t, direction="Forward", wait=None, traj=None, added_trq=None, asynchronous=False):
        if not self._control_strategy.startswith("Joint"):
            self.WarningMessage("Not in joint control mode - JRBFPath not executed")
            return 1
        if asynchronous:
            _th = Thread(target=self._JRBFPath, args=(pathRBF, t), kwargs={"direction": direction, "wait": wait, "traj": traj, "added_trq": added_trq}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JRBFPath(pathRBF, t, direction=direction, wait=wait, traj=traj, added_trq=added_trq)

    def _JRBFPath(self, pathRBF, t, direction="Forward", wait=None, traj=None, added_trq=None):
        if traj is None:
            traj = self._default.Traj
        if wait is None:
            wait = self._default.Wait
        if added_trq is None:
            trq = self._default.AddedTrq
        else:
            trq = vector(trq, dim=self.nj)

        tmperr = 0
        if not isscalar(t) or t <= 0:
            raise ValueError(f"Time must be non-negative scalar")

        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        _n = len(_time)
        _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
        qi = decodeRBF(_s, pathRBF)
        qdoti = gradientPath(qi, _time)
        _fac = np.max(np.max(np.abs(qdoti), axis=0) / self.qdot_max)
        if _fac > 1:
            _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
            _n = len(_time)
            _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
            qi = decodeRBF(_s, pathRBF)
            qdoti = gradientPath(qi, _time)

        self.Message("JRBFPath started", 2)
        self.Start()
        self._command.mode = 1.2
        self._semaphore.acquire()

        if direction == "Backward":
            tmperr = self._loop_joint_traj(qi[::-1, :], qdoti[::-1, :], trq, _time, wait=wait)
        else:
            tmperr = self._loop_joint_traj(qi, qdoti, trq, _time, wait=wait)

        self.Stop()
        self.Message("JRBFPath finished", 2)
        self._semaphore.release()
        return tmperr

    # Task space motion
    def GoTo_T(self, x, v=None, FT=None, wait=None, **kwargs):
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
        if self._control_strategy.startswith("Cartesian"):
            tmperr = self.GoTo_X(x, v, FT, wait, **kwargs)
        else:
            tmperr = self.GoTo_TC(x, v=v, FT=FT, wait=wait, **kwargs)
        return tmperr

    def GoTo_JT(
        self,
        x,
        t,
        wait=None,
        traj_samp_fac=None,
        max_iterations=1000,
        pos_err=None,
        ori_err=None,
        task_space=None,
        task_DOF=None,
        null_space_task=None,
        task_cont_space="Robot",
        q_opt=None,
        v_ns=None,
        qdot_ns=None,
        x_opt=None,
        Kp=None,
        Kns=None,
        state="Commanded",
        **kwargs,
    ):
        if wait is None:
            wait = self._default.Wait
        if traj_samp_fac is None:
            traj_samp_fac = self._default.TrajSampTimeFac
        else:
            traj_samp_fac = int(traj_samp_fac)
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kns is None:
            Kns = self._default.Kns

        self.Message("Cartersian motion -> joint motion", 2)

        N = len(t)
        q_init = self.GetJointPos(state=state)
        if N == 1:
            rx = x2x(x)
            q_path, tmperr = self.IKin(
                rx,
                q_init,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_space=task_space,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                task_cont_space=task_cont_space,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
                save_path=True,
            )
            _time = np.arange(0.0, t + self.tsamp, self.tsamp * traj_samp_fac)
            _time[-1] = t
            _s = np.linspace(0, t, q_path.shape[0])
            _s[-1] = t
            qi = interpPath(_s, q_path, _time)
        else:
            if x.ndim == 3:
                rx = uniqueCartesianPath(t2x(x))
            else:
                rx = uniqueCartesianPath(x)
            q_path, tmperr = self.IKinPath(
                rx,
                q_init,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_space=task_space,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                task_cont_space=task_cont_space,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
            )
            _time = np.arange(0.0, t[-1] + self.tsamp, self.tsamp * traj_samp_fac)
            _time[-1] = t[-1]
            _s = np.linspace(0, t[-1], q_path.shape[0])
            _s[-1] = t[-1]
            qi = interpPath(_s, q_path, _time)

        if tmperr == 0:
            qi[-1, :] = q_path[-1, :]
            qdoti = gradientPath(qi, _time)
            qdoti[-1, :] = qdoti[-1, :] * 0
            self.GoTo_qtraj(qi, qdoti, np.zeros(self.nj), _time)
        else:
            self.WarningMessage("Cartesian movement not feasible!")

        return tmperr

    def GoTo_TC(
        self,
        x,
        v=None,
        FT=None,
        wait=None,
        pos_err=None,
        ori_err=None,
        task_space=None,
        task_DOF=None,
        null_space_task=None,
        task_cont_space="Robot",
        q_opt=None,
        v_ns=None,
        qdot_ns=None,
        x_opt=None,
        Kp=None,
        Kns=None,
        **kwargs,
    ):
        """Kinematic controller

        Parameters
        ----------
        x : _type_
            _description_
        v : _type_, optional
            _description_, by default None
        FT : _type_, optional
            _description_, by default None
        wait : _type_, optional
            _description_, by default None
        pos_err : _type_, optional
            _description_, by default None
        ori_err : _type_, optional
            _description_, by default None
        task_space : _type_, optional
            _description_, by default None
        task_DOF : _type_, optional
            _description_, by default None
        null_space_task : _type_, optional
            _description_, by default None
        task_cont_space : str, optional
            _description_, by default "Robot"
        q_opt : _type_, optional
            _description_, by default None
        v_ns : _type_, optional
            _description_, by default None
        qdot_ns : _type_, optional
            _description_, by default None
        x_opt : _type_, optional
            _description_, by default None
        Kp : _type_, optional
            _description_, by default None
        Kns : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        if v is None:
            v = np.zeros(6)
        else:
            v = vector(v, dim=6)
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)
        if wait is None:
            wait = self._default.Wait
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kns is None:
            Kns = self._default.Kns

        tx = self.simtime()
        rx = x2x(x)
        Sind = np.where(task_DOF > 0)[0]
        uNS = np.zeros(self.nj)

        if check_option(task_space, "World"):
            self._command.x = rx
            self._command.v = v
            self._command.FT = FT
            rx = self.WorldToBase(rx)
            v = self.WorldToBase(v)
            FT = self.WorldToBase(FT)
        elif check_option(task_space, "Robot"):
            self._command.x = self.BaseToWorld(rx)
            self._command.v = self.BaseToWorld(v)
            self._command.FT = self.BaseToWorld(FT)
        elif check_option(task_space, "Object"):
            rx = self.ObjectToWorld(rx)
            v = self.ObjectToWorld(v)
            FT = self.ObjectToWorld(FT)
            self._command.x = rx
            self._command.v = v
            self._command.FT = FT
            rx = self.WorldToBase(rx)
            v = self.WorldToBase(v)
            FT = self.WorldToBase(FT)
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        imode = self._command.mode
        if check_option(null_space_task, "None"):
            self._command.mode = 2.1
        elif check_option(null_space_task, "Manipulability"):
            self._command.mode = 2.2
        elif check_option(null_space_task, "JointLimits"):
            self._command.mode = 2.3
            q_opt = (self.q_max + self.q_min) / 2
        elif check_option(null_space_task, "ConfOptimization"):
            self._command.mode = 2.4
            q_opt = vector(q_opt, dim=self.nj)
        elif check_option(null_space_task, "PoseOptimization"):
            self._command.mode = 2.5
            km = self.Kinmodel(self.q_home)
            x_opt = x2x(x_opt)
            if check_option(task_space, "World"):
                x_opt = self.WorldToBase(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.ObjectToWorld(x_opt)
                x_opt = self.WorldToBase(x_opt)
        elif check_option(null_space_task, "TaskVelocity"):
            self._command.mode = 2.6
            rv = vector(v_ns, dim=6)
            if check_option(task_space, "World"):
                rv = self.WorldToBase(rv)
            elif check_option(task_space, "Object"):
                rv = self.ObjectToWorld(rv)
                rv = self.WorldToBase(rv)
        elif check_option(null_space_task, "JointVelocity"):
            self._command.mode = 2.7
            rqdn = vector(qdot_ns, dim=self.nj)
        else:
            raise ValueError(f"Null-space task '{null_space_task}' not supported")

        rp = copy.deepcopy(rx[:3])
        rR = copy.deepcopy(q2r(rx[3:]))

        while True:
            qq = self._command.q
            np.set_printoptions(formatter={"float": "{: 0.4f}".format})
            p, R, J = self.Kinmodel(qq, out="pR")
            ep = rp - p
            eR = qerr(r2q(rR @ R.T))
            ee = np.hstack((ep, eR))

            if check_option(task_cont_space, "World"):
                RC = np.kron(np.eye(2), self.TBase[:3, :3]).T
            elif check_option(task_cont_space, "Robot"):
                RC = np.eye(6)
            elif check_option(task_cont_space, "Tool"):
                RC = np.kron(np.eye(2), R).T
            elif check_option(task_cont_space, "Object"):
                RC = np.kron(np.eye(2), self.TObject[:3, :3]).T
            else:
                raise ValueError(f"Task space '{task_cont_space}' not supported")

            ee = RC @ ee
            J = RC @ J
            v = RC @ v
            ux = v + Kp * ee
            trq = J.T @ FT
            ux = ux[Sind]
            JJ = J[Sind, :]
            Jp = np.linalg.pinv(JJ)
            NS = np.eye(self.nj) - Jp @ JJ

            if check_option(null_space_task, "None"):
                qdn = np.zeros(self.nj)
            elif check_option(null_space_task, "Manipulability"):
                fun = lambda q: self.Manipulability(q, task_space=task_space, task_DOF=task_DOF)
                qdotn = grad(fun, qq)
                qdn = Kns * qdotn
            elif check_option(null_space_task, "JointLimits"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "ConfOptimization"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "PoseOptimization"):
                een = xerr(x_opt, map_pose(p=p, R=R))
                qdn = Kns * np.linalg.pinv(J) @ een
            elif check_option(null_space_task, "TrackPath"):
                qdn = Kns * np.linalg.pinv(J) @ ee
            elif check_option(null_space_task, "TaskVelocity"):
                qdn = np.linalg.pinv(J) @ rv
            elif check_option(null_space_task, "JointVelocity"):
                qdn = rqdn

            uNS = NS @ qdn
            u = Jp @ ux + uNS
            rqd = Jp @ v[Sind] + uNS
            np.clip(u, -self.qdot_max, self.qdot_max)
            rq = qq + u * self.tsamp
            if self.CheckJointLimits(rq):
                self._command.mode = imode
                self._command.qdot = np.zeros(self.nj)
                self._command.v = np.zeros(6)
                self.WarningMessage(f"Joint limits reached: {self.q}")
                return 0

            self.GoTo_q(rq, rqd, trq, self.tsamp)
            # self.Update()

            if self.simtime() - tx > wait or (np.linalg.norm(ep) < pos_err and np.linalg.norm(eR) < ori_err):
                self._command.mode = imode
                return 0

    def _loop_cartesian_traj(self, xi, vi, FT, time, wait=0, **kwargs):
        tmperr = 0
        if self._control_strategy in ["JointPositionTrajectory"]:
            tmperr = self.GoTo_JT(xi, time, wait=wait, **kwargs)
            if tmperr == 0:
                self.Update()
                _t_traj = self.simtime()
                while (self.simtime() - _t_traj) < (time[-1] + wait):
                    self._last_control_time = self.simtime()
                    if self._abort:
                        self.WarningMessage("Motion aborted by user")
                        self._semaphore.release()
                        self.StopMotion()
                        return 99
                    elif self._do_motion_check and self._motion_check_callback is not None:
                        tmperr = self._motion_check_callback(self)
                        if tmperr > 0:
                            self.WarningMessage("Motion aborted")
                            self._semaphore.release()
                            self.StopMotion()
                            self._command.mode = -2
                        return tmperr
                    elif (self._motion_error is not None) and (self._motion_error != 0):
                        self.WarningMessage("Motion aborted due to motion controller error")
                        self._semaphore.release()
                        return self._motion_error
                    sleep(self.tsamp)
                    self.Update()
        else:
            for xt, vt in zip(xi, vi):
                if self._abort:
                    self.WarningMessage("Motion aborted by user")
                    self._semaphore.release()
                    self.StopMotion()
                    return 99
                elif self._do_motion_check and self._motion_check_callback is not None:
                    tmperr = self._motion_check_callback(self)
                    if tmperr > 0:
                        self._command.qdot = np.zeros(self.nj)
                        self._command.v = np.zeros(6)
                        self.WarningMessage("Motion check stopped motion")
                        self._semaphore.release()
                        self.StopMotion()
                        return tmperr
                tmperr = self.GoTo_T(xt, vt, FT, wait=0, **kwargs)
                if tmperr > 0:
                    self.WarningMessage("Motion aborted")
                    self._semaphore.release()
                    self.StopMotion()
                    return tmperr
            tmperr = self.GoTo_T(xi[-1, :], np.zeros(6), FT, wait=wait, **kwargs)
        return tmperr

    def CMove(self, x, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", min_pos_dist=None, min_ori_dist=None, asynchronous=False, **kwargs):
        if asynchronous:
            self.Message("ASYNC CMove", 2)
            _th = Thread(
                target=self._CMove,
                args=(x, t),
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

    def _CMove(self, x, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", min_pos_dist=None, min_ori_dist=None, **kwargs):
        self._semaphore.acquire()
        if traj is None:
            traj = self._default.Traj
        if short is None:
            short = self._default.RotDirShort
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)
        if min_pos_dist is None:
            min_pos_dist = self._default.MinPosDist
        if min_ori_dist is None:
            min_ori_dist = self._default.MinOriDist

        kwargs.setdefault("kinematics", self._default.Kinematics)

        x = self.spatial(x)
        if wait is None:
            wait = self.tsamp

        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            if x.shape == (4, 4):
                rT = T0 @ x
            elif isvector(x, dim=7):
                rT = T0 @ x2t(x)
            elif x.shape == (3, 3):
                rT = T0 @ map_pose(R=x, out="T")
            elif isvector(x, dim=3):
                rT = T0 @ map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                rT = T0 @ map_pose(Q=x, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
        else:
            if x.shape == (4, 4):
                rT = x
            elif isvector(x, dim=7):
                rT = x2t(x)
            elif x.shape == (3, 3):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(R=x, p=p0, out="T")
            elif isvector(x, dim=4):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(Q=x, p=p0, out="T")
            elif isvector(x, dim=3):
                R0 = self.GetOri(state=state, out="R", task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(p=x, R=R0, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        kwargs["task_space"] = task_space
        rx = t2x(rT)

        dist = xerr(rx, self._command.x)
        if np.linalg.norm(dist[:3]) < min_pos_dist and np.linalg.norm(dist[3:]) < min_ori_dist:
            self.Message("CMove not executed - close to target", 2)
            self._semaphore.release()
            return 0

        if t is not None:
            if not isscalar(t) or t <= 0:
                raise ValueError(f"Time must be non-negative scalar")
            elif t <= 10 * self.tsamp:
                t = None
        if t is None:
            _time = np.arange(0.0, 1 + self.tsamp, self.tsamp)
            if vel is None:
                if vel_fac is None:
                    vel_fac = self._default.VelocityScaling
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=6)
                elif isvector(vel_fac, dim=2):
                    vel_fac = np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3)))
                _vel = self.v_max * vel_fac
                self.Message(f"CMove started: {rx} with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
            else:
                if isscalar(vel):
                    # _vel = np.ones(6) * vel
                    _vel = np.concatenate((normalize(dist[:3]) * vel, self.v_max[3:]))
                    self.Message(f"CMove started: {rx} with velocity {vel:.1f}m/s", 2)
                elif isvector(vel, dim=2):
                    # _vel = np.concatenate((vel[0] * np.ones(3), vel[1] * np.ones(3)))
                    _norm = np.linalg.norm(dist[:3])
                    if _norm < 1e-3:
                        _dp = np.ones(3)
                    else:
                        _dp = dist[:3] / _norm
                    _norm = np.linalg.norm(dist[3:])
                    if _norm < 1e-3:
                        _dr = np.ones(3)
                    else:
                        _dr = dist[3:] / _norm
                    _vel = np.concatenate((_dp * vel[0], _dr * vel[1]))
                    self.Message(f"CMove started: {rx} with velocity {vel[0]:.1f}m/s and {vel[1]:.1f}rd/s ", 2)
                else:
                    _vel = vector(vel, dim=6)
                    self.Message(f"CMove started: {rx} with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
            _vel = np.clip(_vel, 0, self.v_max)
            _vel[np.where(_vel < 1e-3)[0]] = np.inf
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CMove started: {rx} in {_time[-1]:.1f}s", 2)

        self.Start()
        self._command.mode = 2
        tmperr = 0

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        xi, vi, _ = ctraj(x0, rx, _time, traj=traj, short=short)
        _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _time = np.arange(0.0, _time[-1] * _fac + self.tsamp, self.tsamp)
            xi, vi, _ = ctraj(x0, rx, _time, traj=traj, short=short)

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CMove finished", 2)
        self._semaphore.release()
        return tmperr

    def CMoveFor(self, dx, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        if task_space is None:
            task_space = self._default.TaskSpace
        kwargs.setdefault("kinematics", self._default.Kinematics)
        dx = self.spatial(dx)
        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            if isvector(dx, dim=3):
                rT = T0 @ map_pose(p=dx, out="T")
            elif dx.shape == (3, 3):
                rT = T0 @ map_pose(R=dx, out="T")
            elif isvector(dx, dim=4):
                rT = T0 @ map_pose(Q=dx, out="T")
            else:
                raise ValueError(f"Parameter shape {dx.shape} not supported")
        else:
            rT = self.GetPose(out="T", task_space=task_space, kinematics=kwargs["kinematics"], state=state)
            if isvector(dx, dim=3):
                rT[:3, 3] += dx
            elif dx.shape == (3, 3):
                rT[:3, :3] = dx @ rT[:3, :3]
            elif isvector(dx, dim=4):
                rT[:3, :3] = q2r(dx) @ rT[:3, :3]
            else:
                raise ValueError(f"Parameter shape {dx.shape} not supported")
        rx = t2x(rT)
        self.Message("CMoveFor -> CMove", 2)
        tmperr = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def CApproach(self, x, dx, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        if task_space is None:
            task_space = self._default.TaskSpace
        kwargs.setdefault("kinematics", self._default.Kinematics)
        _x = self.spatial(x)
        dx = vector(dx, dim=3)
        if _x.shape == (4, 4):
            rx = map_pose(T=_x)
        elif isvector(x, dim=7):
            rx = _x
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")
        rx[:3] += dx
        self.Message("CApproach -> CMove", 2)
        tmperr = self.CMove(rx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def CLine(self, x, t=None, vel=None, vel_fac=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("CLine -> CMove", 2)
        tmperr = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def CLineFor(self, dx, t=None, vel=None, vel_fac=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("CLineFor -> CMoveFor", 2)
        tmperr = self.CMoveFor(dx, t=t, vel=vel, vel_fac=vel_fac, traj="Trap", short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def CArc(self, x, pC, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        if asynchronous:
            self.Message("ASYNC CArc", 2)
            _th = Thread(target=self._CArc, args=(x, pC), kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "traj": traj, "short": short, "wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._CArc(x, pC, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

    def _CArc(self, x, pC, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, task_space=None, added_FT=None, state="Commanded", **kwargs):
        self._semaphore.acquire()
        if traj is None:
            traj = self._default.Traj
        if short is None:
            short = self._default.RotDirShort
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)
        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs["task_space"] = task_space

        x = self.spatial(x)
        pC = vector(pC, dim=3)
        if wait is None:
            wait = self.tsamp
        if FT is None:
            FT = np.zeros(6)
        else:
            FT = vector(FT, dim=6)

        if check_option(task_space, "Tool"):
            task_space = "World"
            T0 = self.GetPose(out="T", task_space="World", kinematics=kwargs["kinematics"], state=state)
            rpC = T0[:, 3, :3] @ pC
            if x.shape == (4, 4):
                rT = T0 @ x
            elif isvector(x, dim=7):
                rT = T0 @ x2t(x)
            elif x.shape == (3, 3):
                rT = T0 @ map_pose(R=x, out="T")
            elif isvector(x, dim=3):
                rT = T0 @ map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                rT = T0 @ map_pose(Q=x, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")
        else:
            rpC = pC
            if x.shape == (4, 4):
                rT = x
            elif isvector(x, dim=7):
                rT = x2t(x)
            elif x.shape == (3, 3):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(R=x, p=p0, out="T")
            elif isvector(x, dim=4):
                p0 = self.GetPos(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(Q=x, p=p0, out="T")
            elif isvector(x, dim=3):
                R0 = self.GetOri(state=state, out="R", task_space=task_space, kinematics=kwargs["kinematics"])
                rT = map_pose(p=x, R=R0, out="T")
            else:
                raise ValueError(f"Parameter shape {x.shape} not supported")

        rx = t2x(rT)

        if t is not None:
            if not isscalar(t) or t <= 0:
                raise ValueError(f"Time must be non-negative scalar")
            elif t <= 10 * self.tsamp:
                t = None
        if t is None:
            _time = np.arange(0.0, 1 + self.tsamp, self.tsamp)
            if vel is None:
                if vel_fac is None:
                    vel_fac = self._default.VelocityScaling
                elif not isscalar(vel_fac):
                    vel_fac = vector(vel_fac, dim=6)
                elif isvector(vel_fac, dim=2):
                    vel_fac = np.concatenate((vel_fac[0] * np.ones(3), vel_fac[1] * np.ones(3)))
                _vel = self.v_max * vel_fac
            else:
                if isscalar(vel):
                    _vel = np.ones(6) * vel
                elif isvector(vel, dim=2):
                    _vel = np.concatenate((vel[0] * np.ones(3), vel[1] * np.ones(3)))
                else:
                    _vel = vector(vel, dim=6)
            _vel = np.clip(_vel, 0, self.v_max)
            self.Message(f"CArc started: {rx}/{rpC} with velocity {100 * np.max(_vel / self.v_max):.1f}%", 2)
        else:
            _time = np.arange(0.0, t + self.tsamp, self.tsamp)
            _vel = self.v_max
            self.Message(f"CArc started: {rx}/{rpC} in {_time[-1]:.1f}s", 2)

        self.Start()
        self._command.mode = 2
        tmperr = 0

        x0 = self.GetPose(state=state, task_space=task_space, kinematics=kwargs["kinematics"])
        xi, vi, _ = carctraj(x0, rx, rpC, _time, traj=traj, short=short)
        _fac = np.max(np.max(np.abs(vi), axis=0) / _vel)
        if (_fac > 1) or (t is None):
            _time = np.arange(0.0, _time[-1] * _fac + self.tsamp, self.tsamp)
            xi, vi, _ = carctraj(x0, rx, rpC, _time, traj=traj, short=short)

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CArc finished", 2)
        self._semaphore.release()
        return tmperr

    def CPath(self, path, t, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        if asynchronous:
            self.Message("ASYNC CPath", 2)
            _th = Thread(target=self._CPath, args=(path, t), kwargs={"wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._CPath(path, t, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

        _th.start()
        return _th

    def _CPath(self, path, t, direction="Forward", wait=None, task_space=None, added_FT=None, state="Commanded", **kwargs):
        if path.ndim == 3:
            path = uniqueCartesianPath(t2x(path))
        else:
            path = uniqueCartesianPath(path)
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)

        kwargs.setdefault("kinematics", self._default.Kinematics)

        N = path.shape[0]
        rx_init = self.GetPose(task_space=task_space, state="Commanded", out="x")
        if not isscalar(t) and len(t) == path.shape[0]:
            if t[0] > 0:
                path = np.vstack((rx_init, path))
                t = np.concatenate(([0], t))
            _s = t
            N += 1
            t = max(t)
        else:
            if not isscalar(t):
                t = max(t)
            _s = np.linspace(0, t, N)
        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        xi = interpCartesianPath(_s, path, _time)
        vi = gradientCartesianPath(xi, _time)
        _fac = np.max(np.max(np.abs(vi), axis=0) / self.v_max)
        if _fac > 1:
            _s = np.linspace(0.0, t * _fac, N)
            _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
            xi = interpCartesianPath(_s, path, _time)
            vi = gradientCartesianPath(xi, _time)
        N = _time.size

        tmperr = 0
        _dist = xerr(path[0, :], rx_init)
        xe = np.amax(np.abs(_dist) / self.v_max) * 2
        if xe > 0.02:
            self.Message(f"Move to path -> CMove ({_dist})", 2)
            tmprr = self._CMove(path[0], max(xe, 0.2), traj="Poly", shape=True, wait=0, task_space=task_space, added_FT=FT, **kwargs)
            if tmperr > 0:
                self.WarningMessage("Robot did not moved to path start")
                return tmperr

        self.Message(f"CPath started {path.shape[0]} points in {t}s", 2)
        self.Start()
        self._command.mode = 2
        self._semaphore.acquire()

        self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Message("CPath finished", 2)
        self.Stop()
        self._semaphore.release()
        return tmperr

    def CRBFPath(self, pathRBF, t, direction="Forward", wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        if asynchronous:
            self.Message("ASYNC CRBFPath", 2)
            _th = Thread(target=self._CRBFPath, args=(pathRBF, t), kwargs={"direction": direction, "wait": wait, "task_space": task_space, "added_FT": added_FT, "state": state, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._CRBFPath(pathRBF, t, direction=direction, wait=wait, task_space=task_space, added_FT=added_FT, state=state, **kwargs)

        _th.start()
        return _th

    def _CRBFPath(self, pathRBF, t, direction="Forward", wait=None, task_space=None, added_FT=None, state="Commanded", **kwargs):
        if wait is None:
            wait = self._default.Wait
        if task_space is None:
            task_space = self._default.TaskSpace
        if added_FT is None:
            FT = self._default.AddedFT
        else:
            FT = vector(added_FT, dim=6)
        kwargs.setdefault("kinematics", self._default.Kinematics)
        kwargs.setdefault("task_space", task_space)

        tmperr = 0
        if not isscalar(t) or t <= 0:
            raise ValueError(f"Time must be non-negative scalar")

        _time = np.arange(0.0, t + self.tsamp, self.tsamp)
        _n = len(_time)
        _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
        if pathRBF["w"].shape[1] == 3:
            pi = decodeRBF(_s, pathRBF)
            pdi = gradientPath(pi, _time)
            _fac = np.max(np.max(np.abs(pdi), axis=0) / self.v_max[:3])
            if _fac > 1:
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                _n = len(_time)
                _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
                xi = decodeRBF(_s, pathRBF)
                xdi = np.hstack((gradientPath(xi, _time), np.zeros((_n, 3))))
            vi = np.hstack((xdi, np.zeros((_n, 3))))
        elif pathRBF["w"].shape[1] == 7:
            xi = decodeCartesianRBF(_s, pathRBF)
            vi = gradientCartesianPath(xi, _time)
            _fac = np.max(np.max(np.abs(vi), axis=0) / self.v_max)
            if _fac > 1:
                _time = np.arange(0.0, t * _fac + self.tsamp, self.tsamp)
                _n = len(_time)
                _s = np.linspace(pathRBF["c"][0], pathRBF["c"][-1], _n)
                xi = decodeCartesianRBF(_s, pathRBF)
                vi = gradientCartesianPath(xi, _time)
        else:
            raise ValueError(f"Wrong RBF path size {pathRBF['w'].shape[1]}. Must be 3 or 7.")

        if direction == "Backward":
            _initial_x = xi[-1, :]
        else:
            _initial_x = xi[0, :]

        tmperr = 0
        xe = np.amax(np.abs(self.TaskDistance(_initial_x)) / self.v_max) * 2
        if xe > 0.02:
            self.Message("Move to path -> CMove", 2)
            tmprr = self._CMove(_initial_x, max(xe, 0.2), traj="Poly", short=True, wait=0, added_FT=FT, **kwargs)
            if tmperr > 0:
                self.WarningMessage("Robot did not moved to path start")
                return tmperr

        self.Message("CRBFPath started", 2)
        self.Start()
        self._command.mode = 2
        self._semaphore.acquire()

        if direction == "Backward":
            tmperr = self._loop_cartesian_traj(xi[::-1, :], vi[::-1, :], FT, _time, wait=wait, **kwargs)
        else:
            tmperr = self._loop_cartesian_traj(xi, vi, FT, _time, wait=wait, **kwargs)

        self.Stop()
        self.Message("CRBFPath finished", 2)
        self._semaphore.release()
        return tmperr

    # Tool space motion
    def TMove(self, x, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("TMove -> CMove", 2)
        tmperr = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Tool", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def TLine(self, x, t=None, vel=None, vel_fac=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("TLine -> CLine", 2)
        tmperr = self.CLine(x, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Tool", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    # Object space motion
    def OMove(self, x, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OMove -> CMove", 2)
        tmperr = self.CMove(x, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OMoveFor(self, x, dx, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OMoveFor -> CMoveFor", 2)
        tmperr = self.CMoveFor(x, dx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OApproach(self, x, dx, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OApproach -> CApproach", 2)
        tmperr = self.CApproach(x, dx, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OLine(self, x, t=None, vel=None, vel_fac=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OLine -> CLine", 2)
        tmperr = self.CLine(x, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OLineFor(self, x, dx, t=None, vel=None, vel_fac=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OLineFor -> CLineFor", 2)
        tmperr = self.CLineFor(x, dx, t=t, vel=vel, vel_fac=vel_fac, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OArc(self, x, pC, t=None, vel=None, vel_fac=None, traj=None, short=None, wait=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OArc -> CArc", 2)
        tmperr = self.CArc(self, x, pC, t=t, vel=vel, vel_fac=vel_fac, traj=traj, short=short, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def OPath(self, path, t, wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("OPath -> CPath", 2)
        tmperr = self.CPath(self, path, t, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    def ORBFPath(self, pathRBF, t, direction="Forward", wait=None, task_space=None, added_FT=None, state="Commanded", asynchronous=False, **kwargs):
        self.Message("ORBFPath -> CRBFPath", 2)
        tmperr = self.CBFPath(self, pathRBF, t, direction=direction, wait=wait, task_space="Object", added_FT=added_FT, state=state, asynchronous=asynchronous, **kwargs)
        return tmperr

    # Control strategy
    def AvailableStrategies(self):
        return [self._control_strategy]

    def SetStrategy(self, strategy):
        pass

    def GetStrategy(self):
        return self._control_strategy

    def isStrategy(self, strategy):
        return check_option(self._control_strategy, strategy)

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

    # Transformations
    def BaseToWorld(self, x, typ=None):
        """Map from robot base frame to world frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and Jacobian matrix (6,nj)

        Parameters
        ----------
        x : array of floats
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - Jacobian (6, nj)
        typ: str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported
        """
        R0 = self.TBase[:3, :3]
        p0 = self.TBase[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            return self.TBase @ x
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToBase(self, x, typ=None):
        """Map from world frame to robot base frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and Jacobian matrix (6,nj)

        Parameters
        ----------
        x : array of floats
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - Jacobian (6, nj)
        typ: str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported
        """
        R0 = self.TBase[:3, :3].T
        p0 = -R0 @ self.TBase[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def ObjectToWorld(self, x, typ=None):
        """Map from object frame to world frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and Jacobian matrix (6,nj)

        Parameters
        ----------
        x : array of floats
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - Jacobian (6, nj)
        typ: str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported
        """
        R0 = self.TObject[:3, :3]
        p0 = self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            return self.TBase @ x
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    def WorldToObject(self, x, typ=None):
        """Map from world frame to object frame

        Supported arguments: pose (7,), Homogenous matrix (4, 4), rotation matrix (3, 3),
        position (3,), twist (6,) and Jacobian matrix (6,nj)

        Parameters
        ----------
        x : array of floats
            argument to map:
            - pose (7,) or (4, 4)
            - position (3, )
            - orientation (4,) or (3, 3)
            - velocity or force (6, )
            - Jacobian (6, nj)
        typ: str, optional
            Transformation type (None or ``Wrench``)

        Returns
        -------
        array of floats
            mapped argument

        Raises
        ------
        ValueError
            Parameter shape not supported
        """
        R0 = self.TObject[:3, :3].T
        p0 = -R0 @ self.TObject[:3, 3]
        x = np.asarray(x)
        if x.shape == (4, 4):
            p, R = map_pose(T=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif isvector(x, dim=7):
            p, R = map_pose(x=x, out="pR")
            return map_pose(p=R0 @ p + p0, R=R0 @ R, out="x")
        elif x.shape == (3, 3):
            return R0 @ x
        elif isvector(x, dim=4):
            return r2q(R0 @ q2r(x))
        elif isvector(x, dim=3):
            return R0 @ x + p0
        elif isvector(x, dim=6):
            RR = np.block([[R0, np.zeros((3, 3))], [np.zeros((3, 3)), R0]])
            if typ == "Wrench":  # wrench (F)
                RR[3:6, :3] = v2s(p0) @ R0
            return RR @ x
        elif x.shape == (6, self.nj):
            return np.vstack((R0 @ x[:3, :], R0 @ x[3:, :]))
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

    # Kinematic utilities
    @abstractmethod
    def Kinmodel(self, *args, tcp=None, out="x"):
        pass

    def DKin(self, *q, out=None, task_space=None):
        if len(q) > 0:
            _q = self.jointvar(q[0])
        else:
            _q = self._actual.q
        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace
        _x = self.Kinmodel(_q)[0]
        if check_option(task_space, "World"):
            _x = self.BaseToWorld(_x)
        elif check_option(task_space, "Object"):
            _x = self.BaseToWorld(_x)
            _x = self.WorldToObject(_x)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(x=_x, out=out)

    def DKinPath(self, path, out=None):
        """Direct kinematics for a path

        Parameters
        ----------
        path : array of floats
            Path in joint space - poses (n, nj)

        Returns
        -------
        array of floats
            Task positions at target pose (n, 7) or (4, 4, n)
        """

        if out is None:
            out = self._default.TaskPoseForm
        _path = rbs_type(path)
        _n = np.shape(_path)[0]
        _xpath = np.nan * np.zeros((_n, 7))
        for i in range(_n):
            _x = self.DKin(_path[i, :])
            _xpath[i, :] = _x

        return map_pose(x=_xpath, out=out)

    def IKin(self, x, q0, max_iterations=1000, pos_err=None, ori_err=None, task_space=None, task_DOF=None, null_space_task=None, task_cont_space="Robot", q_opt=None, v_ns=None, qdot_ns=None, x_opt=None, Kp=None, Kns=None, save_path=False):
        """Inverse kinematics

        Parameters
        ----------
        x : array of floats
            Target Cartesian pose
        q0 :array of floats
            Initial joint positions (nj,)

        Returns
        -------
        array of floats
            Joint positions at target pose (nj,)
        """
        if pos_err is None:
            pos_err = self._default.PosErr
        if ori_err is None:
            ori_err = self._default.OriErr
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        if null_space_task is None:
            null_space_task = self._default.NullSpaceTask
        if q_opt is None:
            q_opt = self.q_home
        if x_opt is None:
            x_opt = self.Kinmodel(q_opt)[0]
            if check_option(task_space, "World"):
                x_opt = self.BaseToWorld(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.BaseToWorld(x_opt)
                x_opt = self.WorldToObject(x_opt)
            elif check_option(task_space, "Robot"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported")
        if v_ns is None:
            v_ns = np.zeros(6)
        if qdot_ns is None:
            qdot_ns = np.zeros(self.nj)

        if Kp is None:
            Kp = self._default.Kp
        if Kns is None:
            Kns = self._default.Kns

        _max_err = np.ones(6)
        _max_err[:3] = pos_err
        _max_err[3:] = ori_err

        rx = x2x(x)
        q0 = self.jointvar(q0)

        Sind = np.where(task_DOF > 0)[0]
        uNS = np.zeros(self.nj)

        if check_option(task_space, "World"):
            rx = self.WorldToBase(rx)
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            rx = self.ObjectToWorld(rx)
            rx = self.WorldToBase(rx)
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        imode = self._command.mode
        if check_option(null_space_task, "None"):
            pass
        elif check_option(null_space_task, "Manipulability"):
            pass
        elif check_option(null_space_task, "JointLimits"):
            q_opt = (self.q_max + self.q_min) / 2
        elif check_option(null_space_task, "ConfOptimization"):
            q_opt = vector(q_opt, dim=self.nj)
        elif check_option(null_space_task, "PoseOptimization"):
            km = self.Kinmodel(self.q_home)
            x_opt = x2x(x_opt)
            if check_option(task_space, "World"):
                x_opt = self.WorldToBase(x_opt)
            elif check_option(task_space, "Object"):
                x_opt = self.ObjectToWorld(x_opt)
                x_opt = self.WorldToBase(x_opt)
        elif check_option(null_space_task, "TaskVelocity"):
            rv = vector(v_ns, dim=6)
            if check_option(task_space, "World"):
                rv = self.WorldToBase(rv)
            elif check_option(task_space, "Object"):
                rv = self.ObjectToWorld(rv)
                rv = self.WorldToBase(rv)
        elif check_option(null_space_task, "JointVelocity"):
            rqdn = vector(qdot_ns, dim=self.nj)
        else:
            raise ValueError(f"Null-space task '{null_space_task}' not supported")

        rp = copy.deepcopy(rx[:3])
        rR = copy.deepcopy(q2r(rx[3:]))
        _iterations = 0
        qq = q0
        if save_path:
            q_path = q0.reshape((1, self.nj))

        while True:
            p, R, J = self.Kinmodel(qq, out="pR")
            ep = rp - p
            eR = qerr(r2q(rR @ R.T))
            ee = np.hstack((ep, eR))
            if np.all(np.abs(ee) < _max_err):
                if save_path:
                    return q_path, 0
                else:
                    return qq, 0

            if check_option(task_cont_space, "World"):
                RC = np.kron(np.eye(2), self.TBase[:3, :3]).T
            elif check_option(task_cont_space, "Robot"):
                RC = np.eye(6)
            elif check_option(task_cont_space, "Tool"):
                RC = np.kron(np.eye(2), R).T
            elif check_option(task_cont_space, "Object"):
                RC = np.kron(np.eye(2), self.TObject[:3, :3]).T
            else:
                raise ValueError(f"Task space '{task_cont_space}' not supported")

            ee = RC @ ee
            J = RC @ J
            ux = Kp * ee
            ux = ux[Sind]
            JJ = J[Sind, :]
            Jp = np.linalg.pinv(JJ)
            NS = np.eye(self.nj) - Jp @ JJ

            if check_option(null_space_task, "None"):
                qdn = np.zeros(self.nj)
            elif check_option(null_space_task, "Manipulability"):
                fun = lambda q: self.Manipulability(q, task_space=task_space, task_DOF=task_DOF)
                qdotn = grad(fun, qq)
                qdn = Kns * qdotn
            elif check_option(null_space_task, "JointLimits"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "ConfOptimization"):
                qdn = Kns * (q_opt - qq)
            elif check_option(null_space_task, "PoseOptimization"):
                een = xerr(x_opt, map_pose(p=p, R=R))
                qdn = Kns * np.linalg.pinv(J) @ een
            elif check_option(null_space_task, "TaskVelocity"):
                qdn = np.linalg.pinv(J) @ rv
            elif check_option(null_space_task, "JointVelocity"):
                qdn = rqdn

            uNS = NS @ qdn
            u = Jp @ ux + uNS
            qq = qq + u * self.tsamp

            if save_path:
                q_path = np.vstack((q_path, qq))

            if self.CheckJointLimits(qq):
                self.WarningMessage(f"Joint limits reached: {qq}")
                qq = np.nan * qq
                if save_path:
                    return q_path, 1
                else:
                    return qq, 1

            _iterations += 1
            if _iterations > max_iterations:
                self.WarningMessage(f"No solution found in {_iterations} iterations")
                qq = np.nan * qq
                if save_path:
                    return q_path, 2
                else:
                    return qq, 2

    def IKinPath(self, path, q0, max_iterations=100, pos_err=None, ori_err=None, task_space=None, task_DOF=None, null_space_task=None, task_cont_space="Robot", q_opt=None, v_ns=None, qdot_ns=None, x_opt=None, Kp=None, Kns=None):
        """Inverse kinematics for a path

        Parameters
        ----------
        path : array of floats
            Path in Cartesian space - poses (n,7) or (n,4,4)
        q0 :array of floats
            Initial joint positions (nj,)

        Returns
        -------
        array of floats
            Joint positions at target pose (n, nj)
        """
        if path.ndim == 3:
            _path = uniqueCartesianPath(t2x(path))
        elif ismatrix(path, shape=7):
            _path = uniqueCartesianPath(path)
        else:
            raise ValueError(f"Path shape {path.shape} not supported")

        _n = np.shape(_path)[0]
        _qpath = np.nan * np.zeros((_n, self.nj))
        _q = self.jointvar(q0)
        tmperr = 0
        for i in range(_n):
            _x = _path[i, :]
            try:
                _q, tmperr = self.IKin(
                    _x,
                    _q,
                    max_iterations=max_iterations,
                    pos_err=pos_err,
                    ori_err=ori_err,
                    task_space=task_space,
                    task_DOF=task_DOF,
                    null_space_task=null_space_task,
                    task_cont_space=task_cont_space,
                    q_opt=q_opt,
                    v_ns=v_ns,
                    qdot_ns=qdot_ns,
                    x_opt=x_opt,
                    Kp=Kp,
                    Kns=Kns,
                )
                _qpath[i, :] = _q
                if tmperr != 0:
                    return _qpath, tmperr
            except:
                self.Message(f"No solution found for path point sample {i}", 2)
                tmperr = 3
                break
        return _qpath, tmperr

    def Jacobi(self, *q, tcp=None):
        if len(q) > 0:
            qq = self.jointvar(q[0])
        else:
            qq = self._actual.q
        if tcp is None:
            tcp = self.TCP
        km = self.Kinmodel(qq, tcp=tcp)
        return km[-1]

    def Manipulability(self, q, task_space=None, task_DOF=None):
        if task_space is None:
            task_space = self._default.TaskSpace
        if task_DOF is None:
            task_DOF = self._default.TaskDOF
        else:
            task_DOF = vector(task_DOF, dim=6)
        J = self.Jacobi(q)
        if check_option(task_space, "World"):
            J = self.WorldToBase(J)
        elif check_option(task_space, "Robot"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported")
        Sind = np.where(task_DOF > 0)[0]
        JJ = J[Sind, :]
        return np.sqrt(np.linalg.det(JJ @ JJ.T))

    def JointDistance(self, q, state="Actual"):
        """Distance between current position and q

        Parameters
        ----------
        q : array of floats
            joint position (nj,)
        state : str
            joint positions state (`Actual`, `Command`)

        Returns
        -------
        array of floats
            distance to current q (nj,)
        """
        q = self.jointvar(q)
        return q - self.GetJointPos(state=state)

    def TaskDistance(self, x, out="x", task_space="World", state="Actual", kinematics="Calculated"):
        """Distance between current pose and x

        Parameters
        ----------
        x : array of floats
            current pose
        out : str
            output form (`x`, `p`, `Q`)

        Returns
        -------
        array of floats
            distance to current pose
        """
        x = self.spatial(x)

        if x.shape == (4, 4):
            rx = t2x(x)
        elif isvector(x, dim=7):
            rx = x
        elif x.shape == (3, 3):
            rx = map_pose(R=x)
            out = "Q"
        elif isvector(x, dim=3):
            rx = map_pose(p=x)
            out = "p"
        elif isvector(x, dim=4):
            rx = map_pose(Q=x)
            out = "Q"
        else:
            raise ValueError(f"Parameter shape {x.shape} not supported")

        dx = xerr(rx, self.GetPose(task_space=task_space, state=state, kinematics=kinematics))
        if out == "x":
            return dx
        elif out == "Q":
            return dx[3:]
        elif out == "p":
            return dx[:3]
        else:
            raise ValueError(f"Output form '{out}' not supported")

    def CheckJointLimits(self, q):
        """Check if q in joint range

        Parameters
        ----------
        q : array of floats
            joint position (nj,)

        Returns
        -------
        bool
            True if one joint out of limits
        """
        return np.any(self.q_max - q < 0) or np.any(q - self.q_min < 0)

    def DistToJointLimits(self, *q):
        """Distance to joint limits

        Parameters
        ----------
        q : array of floats, optional
            joint position (nj,)

        Returns
        -------
        array of floats
            minimal distance to joint limits (nj,)
        array of floats
            distance to lower joint limits (nj,)
        array of floats
            distance to upper joint limits (nj,)
        """
        if len(q) == 0:
            q = self._actual.q
        else:
            q = self.jointvar(q[0])
        dqUp = self.q_max - q
        dqLow = q - self.q_min
        dq = np.fmin(dqLow, dqUp)
        return dq, dqLow, dqUp

    # Gripper
    def SetGripper(self, grip):
        if self.Gripper is not None:
            self.Gripper.Detach()
        self.Gripper = grip
        if grip is not None:
            grip.AttachTo(self)

    def GetGripper(self):
        if self.Gripper is None:
            return [None, "None"]
        else:
            return [self.Gripper, self.Gripper.Name]

    # F/T sensor
    def SetFTSensor(self, sensor):
        if self.FTSensor is not None:
            self.FTSensor.Detach()
        self.FTSensor = sensor
        if sensor is not None:
            self.FTSensor.AttachTo(self)

    def GetFTSensor(self):
        if self.FTSensor is None:
            return [None, "None"]
        else:
            return [self.FTSensor, self.FTSensor.Name]

    def SetFTSensorFrame(self, x):
        x = self.spatial(x)
        if x.shape == (4, 4):
            _T = x
        elif x.shape == (3, 3):
            _T = map_pose(R=x, out="T")
        elif isvector(x, dim=7):
            _T = map_pose(x=x, out="T")
        elif isvector(x, dim=3):
            _T = map_pose(p=x, out="T")
        elif isvector(x, dim=4):
            _T = map_pose(Q=x, out="T")
        else:
            raise ValueError(f"FT sensor frame shape {x.shape} not supported")
        self.FTSensorFrame = _T

    def GetFTSensorFrame(self, out=None):
        if out is None:
            out = self._default.TaskPoseForm
        return map_pose(T=self.FTSensorFrame, out=out)

    def GetFTSensorPose(self, out=None, task_space=None):
        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace
        _T = self.T @ np.linalg.inv(self.TCP) @ self.FTSensorFrame
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Robot"):
            _T = self.WorldToBase(_T)
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(T=_T, out=out)

    def SetFTSensorLoad(self, load=None, mass=None, COM=None, inertia=None, offset=None):
        if self.FTSensor is not None:
            self.FTSensor.SetLoad(load=load, mass=mass, COM=COM, inertia=inertia, offset=offset)

    def GetFTSensorLoad(self):
        if self.FTSensor is None:
            return None
        else:
            return self.FTSensor.GetLoad()

    # Load
    def SetLoad(self, load=None, mass=None, COM=None, inertia=None):
        if isinstance(load, _load):
            self.Load = load
        else:
            if mass is not None:
                self.Load.mass = mass
            if COM is not None:
                self.Load.COM = COM
            if inertia is not None:
                self.Load.inertia = inertia

    def GetLoad(self):
        return self.Load

    # TCP
    def SetTCP(self, *tcp, frame="Gripper"):
        if len(tcp) == 0:
            tcp = np.eye(4)
        self._SetTCP(tcp, frame=frame)

    def _SetTCP(self, *tcp, frame="Gripper"):
        if len(tcp) > 0:
            x = self.spatial(tcp[0])
            if x.shape == (4, 4):
                _tcp = x
            elif x.shape == (3, 3):
                _tcp = map_pose(R=x, out="T")
            elif isvector(x, dim=7):
                _tcp = map_pose(x=x, out="T")
            elif isvector(x, dim=3):
                _tcp = map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                _tcp = map_pose(Q=x, out="T")
            else:
                raise ValueError(f"TCP shape {x.shape} not supported")
        else:
            _tcp = np.eye(4)
        if check_option(frame, "Robot"):
            newTCP = _tcp
        elif check_option(frame, "Gripper"):
            newTCP = self.TCPGripper @ _tcp
        else:
            raise ValueError(f"Frame '{frame}' not supported")
        self.TCP = newTCP
        rx, rJ = self.Kinmodel(self._command.q)
        self._command.x = self.BaseToWorld(rx)
        self._command.v = self.BaseToWorld(rJ @ self._command.qdot)

    def GetTCP(self, out="T"):
        return map_pose(T=self.TCP, out=out)

    def SetTCPGripper(self, *tcp):
        if len(tcp) > 0:
            x = self.spatial(tcp[0])
            if x.shape == (4, 4):
                _tcp = x
            elif x.shape == (3, 3):
                _tcp = map_pose(R=x, out="T")
            elif isvector(x, dim=7):
                _tcp = map_pose(x=x, out="T")
            elif isvector(x, dim=3):
                _tcp = map_pose(p=x, out="T")
            elif isvector(x, dim=4):
                _tcp = map_pose(Q=x, out="T")
            else:
                raise ValueError(f"TCP shape {x.shape} not supported")
        else:
            _tcp = np.eye(4)
        self.TCPGripper = _tcp

    def GetTCPGripper(self, out="T"):
        return map_pose(T=self.TCPGripper, out=out)

    # Object
    def SetObject(self, *x):
        if len(x) == 0:
            _x = self._actual.x
        else:
            _x = self.spatial(x[0])
        if _x.shape == (4, 4):
            _T = _x
        elif isvector(_x, dim=7):
            _T = x2t(_x)
        else:
            raise ValueError(f"Object pose shape {_x.shape} not supported")
        self.TObject = _T

    def GetObject(self, out="T", task_space=None):
        if task_space is None:
            task_space = self._default.TaskSpace
        _T = self.TObject
        if check_option(task_space, "World"):
            pass
        elif check_option(task_space, "Object"):
            _T = self.WorldToObject(_T)
        elif check_option(task_space, "Robot"):
            _T = self.WorldToBase(_T)
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetObject")
        return map_pose(T=_T, out=out)

    # Base and platform
    def SetBase(self, x):
        _x = self.spatial(x)
        if _x.shape == (4, 4):
            _T = _x
        elif isvector(_x, dim=7):
            _T = x2t(_x)
        else:
            raise ValueError(f"Base pose shape {_x.shape} not supported")
        self.TBase = _T
        _x, _J = self.Kinmodel(self._command.q)
        self._command.x = _x
        self._command.v = _J @ self._command.qdot

    def GetBase(self, out="T"):
        _T = self.TBase
        return map_pose(T=_T, out=out)

    def SetBasePlatform(self, platform):
        self.Platform = platform
        self.Platform.AttachTo(self)

    def GetBasePlatform(self):
        if self.Platform is None:
            return [None, "None"]
        else:
            return [self.Platform, self.Platform.Name]

    def UpdateRobotBase(self):
        if self.Platform is not None:
            self.TBase = self.Platform.GetRobotBasePose(out="T")
        return self.TBase

    # Movements
    def Start(self):
        self._command.mode = 0.5
        self._last_control_time = self.simtime()
        self._abort = False
        self._motion_error = None
        self.Update()

    def Stop(self):
        self._command.mode = 0
        self._command.qdot = np.zeros(self.nj)
        self._command.v = np.zeros(6)
        self._abort = False
        self._motion_error = None
        self.Update()

    def Abort(self, *args):
        if len(args) == 0:
            abort = True
        else:
            abort = args[0]
        self.Message("Abort: ", abort)
        self._abort = abort
        self._command.mode = -2
        self.Update()

    def StopMotion(self):
        if self._control_strategy in ["JointPositionTrajectory"]:
            self.GoTo_qtraj(self.q, np.zeros(self.nj), np.zeros(self.nj), self.tsamp)
        self.Stop()

    def WaitUntilStopped(self, eps=0.001):
        self.GetState()
        while np.linalg.norm(self._actual.qdot) > eps:
            self.GetState()

    def Wait(self, t, dt=None):
        self._semaphore.acquire()
        self.Message(f"Wait for {t:.3f}s", 3)
        if dt is None:
            dt = self.tsamp
        tx = self.simtime()
        imode = self._command.mode
        self._command.mode = -1
        while self.simtime() - tx < t:
            self.GetState()
            self.Update()
            sleep(dt)
        self._command.mode = imode
        self._semaphore.release()

    def Restart(self):
        self.Stop()
        self.Start()

    def SetMotionCheckCallback(self, fun):
        self._motion_check_callback = fun

    def EnableMotionCheck(self, check=True):
        self._do_motion_check = check

    def DisableMotionCheck(self):
        self._do_motion_check = False

    # Utilities
    def SetCaptureCallback(self, fun):
        self._capture_callback = fun

    def StartCapture(self):
        if not self._do_update:
            self.WarningMessage("Update is not enabled")
        self._do_capture = True
        self.Message("Capture started", 2)
        self.Update()

    def StopCapture(self):
        self.Message("Capture stopped", 2)
        self._do_capture = False

    def SetUserData(self, data):
        self._command.data = data
        self.Message(f"User data: {data}", 3)
        self.Update()

    def GetUserdata(self):
        return self._command.data


def isrobot(obj):
    return isinstance(obj, robot)


def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))


def dkin(q, kinmodel, tcp=np.eye(4), out="x"):
    """Direct kinematics

    Parameters
    ----------
    q: array of floats
        Joint positions (nj,)
    kinmodel: function
        Direct kinematics function
    tcp: array of floatd
        Tool center point pose (7,) or (4,4)

    Returns
    -------
    array of floats
        task positions (7,)
    """
    _q = rbs_type(q)
    return kinmodel(_q, tcp=tcp, out=out)[0]


def dkinpath(path, kinmodel, tcp=np.eye(4), out="x"):
    """Direct kinematics for a path

    Parameters
    ----------
    path : array of floats
        Path in joint space - poses (n, nj)
    kinmodel: function
        Direct kinematics function
    TCP: array of floatd
        Tool center point pose (7,) or (4,4)

    Returns
    -------
    array of floats
        Task positions at target pose (n, 7) or (4, 4, n)
    """
    _path = rbs_type(path)
    _n = np.shape(_path)[0]
    _xpath = np.nan * np.zeros((_n, 7))
    for i in range(_n):
        _x = dkin(_path[i, :], kinmodel, tcp=tcp, out="x")
        _xpath[i, :] = _x
    return _xpath


def ikin(
    x,
    q0,
    kinmodel,
    tcp=np.eye(4),
    tsamp=0.01,
    max_iterations=1000,
    pos_err=0.0001,
    ori_err=0.001,
    task_DOF=[1, 1, 1, 1, 1, 1],
    null_space_task="None",
    q_min=None,
    q_max=None,
    q_opt=None,
    v_ns=None,
    qdot_ns=None,
    x_opt=None,
    Kp=10,
    Kns=1,
    save_path=False,
):
    """Inverse kinematics

    Parameters
    ----------
    x : array of floats
        Target Cartesian pose
    q0: array of floats
        Initial joint positions (nj,)
    kinmodel: function
        Direct kinematics function
    tcp: array of floatd
        Tool center point pose (7,) or (4,4)

    Returns
    -------
    array of floats
        Joint positions at target pose (nj,)
    """
    rx = x2x(x)
    q0 = rbs_type(q0)
    nj = q0.shape[0]

    if v_ns is None:
        v_ns = np.zeros(6)
    if qdot_ns is None:
        qdot_ns = np.zeros(nj)
    if q_min is None or q_max is None:
        raise ValueError("Joint limits q_min and q_max have to be defined")
    task_DOF = vector(task_DOF, dim=6)

    if check_option(null_space_task, "None"):
        pass
    elif check_option(null_space_task, "Manipulability"):
        pass
    elif check_option(null_space_task, "JointLimits"):
        q_opt = (q_max + q_min) / 2
    elif check_option(null_space_task, "ConfOptimization"):
        if q_opt is None:
            raise ValueError("Optimal joint configuration q_opt has to be defined")
        q_opt = vector(q_opt, dim=nj)
    elif check_option(null_space_task, "PoseOptimization"):
        if x_opt is None:
            raise ValueError("Optimal task pose x_opt has to be defined")
        x_opt = x2x(x_opt)
    elif check_option(null_space_task, "TaskVelocity"):
        rv = vector(v_ns, dim=6)
    elif check_option(null_space_task, "JointVelocity"):
        rqdn = vector(qdot_ns, dim=nj)
    else:
        raise ValueError(f"Null-space task '{null_space_task}' not supported")

    _max_err = np.ones(6)
    _max_err[:3] = pos_err
    _max_err[3:] = ori_err

    rp = copy.deepcopy(rx[:3])
    rR = copy.deepcopy(q2r(rx[3:]))
    _iterations = 0
    qq = q0
    if save_path:
        q_path = q0.reshape((1, nj))

    Sind = np.where(task_DOF > 0)[0]
    uNS = np.zeros(nj)

    while True:
        p, R, J = kinmodel(qq, tcp=tcp, out="pR")
        ep = rp - p
        eR = qerr(r2q(rR @ R.T))
        ee = np.hstack((ep, eR))
        if np.all(np.abs(ee) < _max_err):
            if save_path:
                return q_path, 0
            else:
                return qq, 0

        ux = Kp * ee
        ux = ux[Sind]
        JJ = J[Sind, :]
        Jp = np.linalg.pinv(JJ)
        NS = np.eye(nj) - Jp @ JJ

        if check_option(null_space_task, "None"):
            qdn = np.zeros(nj)
        elif check_option(null_space_task, "Manipulability"):
            fun = lambda q: manipulability(kinmodel(q, tcp=tcp)[1])
            qdotn = grad(fun, qq)
            qdn = Kns * qdotn
        elif check_option(null_space_task, "JointLimits"):
            qdn = Kns * (q_opt - qq)
        elif check_option(null_space_task, "ConfOptimization"):
            qdn = Kns * (q_opt - qq)
        elif check_option(null_space_task, "PoseOptimization"):
            een = xerr(x_opt, map_pose(p=p, R=R))
            qdn = Kns * np.linalg.pinv(J) @ een
        elif check_option(null_space_task, "TaskVelocity"):
            qdn = np.linalg.pinv(J) @ rv
        elif check_option(null_space_task, "JointVelocity"):
            qdn = rqdn

        uNS = NS @ qdn
        u = Jp @ ux + uNS
        qq = qq + u * tsamp

        if save_path:
            q_path = np.vstack((q_path, qq))

        if np.any(q_max - qq < 0) or np.any(qq - q_min < 0):
            print(f"Joint limits reached: {qq}")
            qq = np.nan * qq
            if save_path:
                return q_path, 1
            else:
                return qq, 1

        _iterations += 1
        if _iterations > max_iterations:
            print(f"No solution found in {_iterations} iterations")
            qq = np.nan * qq
            if save_path:
                return q_path, 2
            else:
                return qq, 2


def ikinpath(
    path,
    q0,
    kinmodel,
    tcp=np.eye(4),
    tsamp=0.01,
    max_iterations=1000,
    pos_err=0.001,
    ori_err=0.001,
    task_DOF=[1, 1, 1, 1, 1, 1],
    null_space_task="None",
    q_min=None,
    q_max=None,
    q_opt=None,
    v_ns=None,
    qdot_ns=None,
    x_opt=None,
    Kp=10,
    Kns=1,
):
    """Inverse kinematics for a path

    Parameters
    ----------
    path : array of floats
        Path in Cartesian space - poses (n,7) or (n,4,4)
    q0 :array of floats
        Initial joint positions (nj,)
    kinmodel: function
        Direct kinematics function
    TCP: array of floatd
        Tool center point pose (7,) or (4,4)

    Returns
    -------
    array of floats
        Joint positions at target pose (n, nj)
    """
    if path.ndim == 3:
        _path = uniqueCartesianPath(t2x(path))
    elif ismatrix(path, shape=7):
        _path = uniqueCartesianPath(path)
    else:
        raise ValueError(f"Path shape {path.shape} not supported")

    _q = rbs_type(q0)
    nj = _q.shape[0]
    _n = np.shape(_path)[0]
    _qpath = np.nan * np.zeros((_n, nj))
    tmperr = 0
    for i in range(_n):
        _x = _path[i, :]
        try:
            _q, tmperr = ikin(
                _x,
                _q,
                kinmodel,
                tcp=tcp,
                q_min=q_min,
                q_max=q_max,
                max_iterations=max_iterations,
                pos_err=pos_err,
                ori_err=ori_err,
                task_DOF=task_DOF,
                null_space_task=null_space_task,
                q_opt=q_opt,
                v_ns=v_ns,
                qdot_ns=qdot_ns,
                x_opt=x_opt,
                Kp=Kp,
                Kns=Kns,
            )
            if tmperr != 0:
                return _qpath, tmperr
            _qpath[i, :] = _q
        except:
            print(f"No solution found for path point sample {i}")
            tmperr = 3
            break
    return _qpath, tmperr


if __name__ == "__main__":
    from robotblockset.transformations import rot_x

    from robotblockset.robot_spec import panda_spec

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    # Robot without a scene!
    class panda_test(panda_spec):
        def __init__(self):
            panda_spec.__init__(self)
            robot.__init__(self)
            self.Init()

        def __del__(self):
            self.Message("Robot deleted", 2)

    r = panda_test()
    r.JMove(r.q_home, 0.1)
    print("q: ", r.q)
    print("x: ", r.x)

    x = r.Kinmodel(out="x")[0]
    print("Robot pose:\n ", x)
    J = r.Jacobi()
    print("Robot Jacobian:\n ", J)

    print("Pose: ", r.GetPose())

    print("Velocity: ", r.GetVel())

    print("FT: ", r.GetFT())
