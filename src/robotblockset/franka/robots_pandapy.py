"""Franka robot interface via panda_py.

High-level interface for controlling Franka Emika Panda / FR3 robots via panda_py.

This module provides:

- A thin wrapper class :class:`Desk` around :mod:`panda_py.Desk` to manage
  Franka Desk operations such as unlocking/locking the robot, activating or
  deactivating the FCI, and taking or releasing control.

- A high-level robot class :class:`panda` that integrates:
    * Franka-specific kinematic and dynamic specifications from
      :mod:`robotblockset.robot_spec` (:class:`panda_spec`, :class:`fr3_spec`),
    * Generic robot functionality from :class:`robotblockset.robots.robot`,
    * Low-level control and state access via :mod:`panda_py` and :mod:`panda_py.libfranka`,
    * Controller implementations from :mod:`panda_py.controllers`.

The :class:`panda` class offers:

- Access to model quantities (Jacobian, gravity, Coriolis, inertia matrix).
- Analytical forward and inverse kinematics for the Panda/FR3 kinematic chain.
- Joint-space and Cartesian-space motion commands (JMove, CMove, GoTo_q, GoTo_T, GoTo_X).
- Configuration and switching of control strategies (e.g. joint or Cartesian impedance).
- Convenience methods for setting and querying Cartesian and joint compliance
  (stiffness, damping, “softness” factors).
- Handling of collision behavior, contact and collision monitoring.
- Management of the tool center point (TCP), stiffness frame, and external load
  properties consistent with Franka’s internal model.

The module is intended to be used as a higher-level abstraction on top of
:mod:`panda_py`, providing a more robotics-oriented, task-space–aware API that
integrates with the RobotBlockSet ecosystem.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from time import sleep
from typing import Any, List, Optional, Tuple, Union
import logging
from threading import Thread
import panda_py  # pyright: ignore[reportMissingImports]
from panda_py import libfranka, controllers  # pyright: ignore[reportMissingImports]

from robotblockset.transformations import map_pose, frame2world, spatial2t, spatial2x, x2x, x2t, t2x, q_wxyz2xyzw, xerr
from robotblockset.robots import robot, MotionResultCodes
from robotblockset.robot_spec import panda_spec, fr3_spec
from robotblockset.rbs_typing import ArrayLike, HomogeneousMatrixType, JointConfigurationType, JointTorqueType, JointVelocityType, Pose3DType, RotationMatrixType, TCPType, Vector3DType, Velocity3DType, WrenchType, JacobianType
from robotblockset.tools import _struct, load_params, rbs_type, isscalar, vector, isvector, ismatrix, check_option, matrix


class Desk:
    """Wrapper around :class:`panda_py.Desk` for Franka Desk management.

    This helper connects to the robot control unit's web-based Desk interface
    and exposes common remote operations such as locking or unlocking the
    brakes, activating or deactivating the FCI, and taking or releasing Desk
    control.

    Recent Desk software versions allow only one user to control the Desk at a
    time. Control is mediated by a token managed internally by
    :class:`panda_py.Desk`.

    Parameters
    ----------
    hostname : str, optional
        IP address or hostname of the control unit running Franka Desk.
    username : str, optional
        Username used to log into Franka Desk.
    password : str, optional
        Password for the given Desk user.
    """

    def __init__(self, hostname: str = "192.168.3.100", username: str = "user", password: str = "password") -> None:
        """Create a Desk client and log into the Franka Desk interface.

        Parameters
        ----------
        hostname : str, optional
            IP address or hostname of the control unit running Franka Desk.
        username : str, optional
            Username used to log into Franka Desk.
        password : str, optional
            Password for the given Desk user.

        Returns
        -------
        None
            This constructor initializes the Desk client in place.
        """
        self.hostname = hostname
        self.Username = username
        self.password = password
        #  activate information log level
        logging.basicConfig(level=logging.INFO)

        # define Desk connection
        self.desk = panda_py.Desk(hostname=self.hostname, username=self.Username, password=self.password)

    def Unlock(self) -> None:
        """Unlock the robot brakes.

        This mirrors the ``panda-unlock`` CLI behavior only partially. The CLI
        command unlocks the brakes and then activates the FCI, while this method
        only unlocks the brakes.
        """
        self.desk.unlock()

    def Lock(self) -> None:
        """Lock the robot brakes.

        This mirrors the ``panda-lock`` CLI behavior only partially. The CLI
        command locks the brakes and then deactivates the FCI, while this method
        only locks the brakes.
        """
        self.desk.lock()

    def ActivateFCI(self) -> None:
        """Activate the Franka Research Interface (FCI).

        The robot brakes must be unlocked first. On older Desk versions this
        operation has no effect.
        """
        self.desk.activate_fci()

    def DeactivateFCI(self) -> None:
        """Deactivate the Franka Research Interface (FCI).

        On older Desk versions this operation has no effect.
        """
        self.desk.deactivate_fci()

    def TakeControl(self) -> None:
        """Take control of the Desk using the configured user credentials.

        If another user already controls the Desk, forceful takeover requires
        confirmation of physical access on the robot's Pilot interface. This
        wrapper uses the default non-forced takeover behavior.
        """
        self.desk.take_control()

    def ReleaseControl(self) -> None:
        """Release control of the Desk.

        Releasing control allows another user to take Desk control without
        requiring physical access to the robot.
        """
        self.desk.release_control()

    def HasControl(self) -> bool:
        """Return whether this client currently controls the Desk.

        Returns
        -------
        bool
            ``True`` if this Desk client currently has control, otherwise
            ``False``.
        """
        return self.desk.has_control()


class FrankaCollisionBehaviour(_struct):
    """Container for Franka collision and contact threshold settings."""

    def __init__(self) -> None:
        """Initialize collision threshold containers."""
        self.lower_torque_thresholds = None
        self.upper_torque_thresholds = None
        self.lower_force_thresholds = None
        self.upper_force_thresholds = None


class FrankaDefaults(_struct):
    """Container for default Franka stiffness, damping, and collision settings."""

    def __init__(self) -> None:
        """Initialize default Franka compliance settings.

        Initialize default Franka robot compliance and collision settings."""
        self.JointStiffness = np.array([600, 600, 600, 600, 250, 150, 50])
        self.JointDamping = np.array([50, 50, 50, 20, 20, 20, 10])
        self.JointFilter = 1.0
        self.CartesianImpedance = np.diag([800, 800, 800, 40, 40, 40]) * 2
        self.CartesianDampingRatio = 1.0
        self.CartesianNullspaceStiffness = 0.5
        self.CartesianFilter = 1.0
        self.MinSoftnessForMotion = 0.005  # Expected minimal compliance which can allow motion
        self.CollisionBehavior = FrankaCollisionBehaviour()
        self.CollisionBehavior.lower_torque_thresholds = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.upper_torque_thresholds = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0])  # [Nm])
        self.CollisionBehavior.lower_force_thresholds = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])
        self.CollisionBehavior.upper_force_thresholds = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0])  # [N, N, N, Nm, Nm, Nm])


class panda(panda_spec, fr3_spec, robot):
    """High-level interface for Franka Emika Panda / FR3 robots via panda_py.

    Parameters
    ----------
    hostname : str, optional
        IP address or hostname of the Panda / FR3 robot
    name : str, optional
        Name identifier for the robot instance (default is 'panda')
    model : str, optional
        Robot model, either 'panda' or 'fr3' (default is 'panda')
    control_strategy : str, optional
        Initial control strategy to use. Default is ``"JointImpedance"``.
    **kwargs : Any
        Additional keyword arguments passed to `robot.__init__`.

    Attributes
    ----------
    hostname : str
        Robot controller hostname or IP address.
    name : str
        Instance name used for logging and identification.
    model : Any
        Active Franka model interface returned by `panda_py`.
    panda : panda_py.Panda
        Low-level Panda robot client.
    robot : Any
        Low-level robot interface returned by `panda_py`.
    """

    def __init__(self, hostname: str = "192.168.3.100", name: str = "panda", model: str = "panda", control_strategy: str = "JointImpedance", **kwargs: Any) -> None:
        """Create a Panda/FR3 robot interface backed by `panda_py`.

        Parameters
        ----------
        hostname : str, optional
            Robot controller hostname or IP address.
        name : str, optional
            Name used to identify the robot instance.
        model : str, optional
            Franka robot model, either ``"panda"`` or ``"fr3"``.
        control_strategy : str, optional
            Initial control strategy to activate after initialization.
        **kwargs : Any
            Additional keyword arguments passed to `robot.__init__`.

        Returns
        -------
        None
            This constructor initializes the Panda/FR3 robot interface in place.
        """
        # Panda hostname/IP and Desk login information of your robot
        self.hostname = hostname
        self.name = name
        self.model = model
        self.control_strategy = control_strategy

        if model == "panda":
            panda_spec.__init__(self)
        elif model == "fr3":
            fr3_spec.__init__(self)
        else:
            raise ValueError(f"Model '{model}' not supported")
        robot.__init__(self, **kwargs)

        # Robot instance from panda_py
        self.panda = panda_py.Panda(hostname)
        self.robot = self.panda.get_robot()
        self.model = self.panda.get_model()

        self.tsamp = 0.01  # 100ms default sample time
        self._active = True
        self._connected = False

        # Initialize franka parameters
        self._franka_default = FrankaDefaults()

        # Initialize panda specific parameters for controllers
        self._strategy_controller_mapping = {}
        strategy_controller_specs = (
            ("JointImpedance", "JointPosition"),
            ("CartesianImpedance", "CartesianImpedance"),
            ("CartesianImpedanceExtended", "CartesianImpedanceExtended"),
        )
        for strategy_name, controller_name in strategy_controller_specs:
            controller_cls = getattr(controllers, controller_name, None)
            if controller_cls is not None:
                self._strategy_controller_mapping[strategy_name] = controller_cls()

        self.joint_stiffness = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        self.joint_damping = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])
        self.joint_filter = 1.0
        self.cartesian_impedance = np.diag([800, 800, 800, 40, 40, 40])
        self.cartesian_damping_ratio = 1.0
        self.cartesian_nullspace_stiffness = 0.5
        self.cartesian_filter = 1.0

        self.Init()
        self.Update_tcp_from_franka_state()

        self._control_strategy = None
        self.SetStrategy(control_strategy)

        self._connected = True
        self.Message("Initialized", 1)

    @property
    def J(self, frame: Union[str, int] = "EE", state: Optional[Any] = None) -> JacobianType:
        """
        Jacobian matrix in joint space.

        Parameters
        ----------
        frame : Union[str, int], optional
            Frame for which the Jacobian is evaluated. Supported string values are
            ``"EE"``, ``"Flange"``, and ``"Stiffness"``.
        state : Any, optional
            Robot state. If `None`, the current state is used.

        Returns
        -------
        JacobianType
            Jacobian matrix in joint space.
        """
        if frame == "EE":
            frame = libfranka.Frame.kEndEffector
        elif frame == "Flange":
            frame = libfranka.Frame.kFlange
        elif frame == "Stiffness":
            frame = libfranka.Frame.kStiffness
        elif isinstance(frame, (int, np.integer)) and 0 <= frame <= 9:
            frame = libfranka.Frame(frame)
        if state is None:
            self.UpdateState()
            state = self._state
        return np.reshape(self.model.zero_jacobian(frame, state), (6, 7), order="F")

    @property
    def c(self, state: Optional[Any] = None) -> JointTorqueType:
        """
        Coriolis and centrifugal joint torques.

        Parameters
        ----------
        state : Any, optional
            The robot state. If None, the current state is used, by default None.

        Returns
        -------
        JointTorqueType
            Coriolis and centrifugal joint torques.
        """
        if state is None:
            self.UpdateState()
            state = self._state
        return self.model.coriolis(state)

    @property
    def g(self, state: Optional[Any] = None) -> JointTorqueType:
        """
        Gravitational joint torques

        Parameters
        ----------
        state : Any, optional
            The robot state. If None, the current state is used, by default None.

        Returns
        -------
        JointTorqueType
            Gravity joint torques.
        """
        if state is None:
            self.UpdateState()
            state = self._state
        return self.model.gravity(state)

    @property
    def H(self, state: Optional[Any] = None) -> np.ndarray:
        """
        Inertia matrix H(q) in joint space.

        Parameters
        ----------
        state : Any, optional
            The robot state. If None, the current state is used, by default None.

        Returns
        -------
        np.ndarray
            Symmetric inertia matrix in joint space.
        """
        if state is None:
            self.UpdateState()
            state = self._state
        return np.reshape(self.model.mass(state), (7, 7), order="F")

    def FK(self, q: Optional[JointConfigurationType] = None, out: str = "x") -> Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]:
        """
        Analytical forward kinematics of Panda robot.

        Parameters
        ----------
        q : JointConfigurationType, optional
            Joint positions (nj, )
        out : str, optional
            Output format for the result (pose, position, etc.), by default "x".

        Returns
        -------
        Union[Pose3DType, HomogeneousMatrixType, Vector3DType, RotationMatrixType]
            End-effector pose in the output format requested by `out`.

        Notes
        -----
        Assumes default TCP.
        """
        if q is None:
            qq = self._actual.q
        else:
            qq = self.jointvar(q)
        T = panda_py.fk(qq)
        return map_pose(T=T, out=out)

    def IK(self, T: Union[Pose3DType, HomogeneousMatrixType], q7: float = np.pi / 4, q_initial: Optional[JointConfigurationType] = None) -> JointConfigurationType:
        """
        Analytical inverse kinematics of Panda robot.

        Parameters
        ----------
        T : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose (7,) or (4, 4).
        q7 : float, optional
            Desired angle for joint 7, by default np.pi / 4.
        q_initial : JointConfigurationType, optional
            Initial guess for the joint positions (nj, ), by default None (uses home position).

        Returns
        -------
        JointConfigurationType
            Joint positions (nj, ) that achieve the desired end-effector pose.

        Notes
        -----
        Assumes default TCP.
        """
        if q_initial is None:
            q_initial = self.q_home
        _T = spatial2t(T)
        _q = panda_py.ik(_T, q_initial, q7)
        return _q

    # Robot state
    def UpdateState(self) -> None:
        """Updates robot state internal variable"""
        try:
            self._state = self.robot.read_once()
        except:
            self._state = self.panda.get_state()

    def GetState(self) -> None:
        """Update the cached robot state.

        Returns
        -------
        None
            This method refreshes the internal state, command, and wrench caches.
        """

        self.UpdateState()
        self._robottime = self._state.time.to_sec()

        self._actual.q = rbs_type(self._state.q)
        self._actual.qdot = rbs_type(self._state.dq)
        self._actual.trq = rbs_type(self._state.tau_J)

        if self._control_strategy in ["CartesianImpedance"]:
            self._command.q = rbs_type(self._state.q_d)
            self._command.qdot = rbs_type(self._state.dq_d)
            # T_D = np.reshape(self._state.O_T_EE_d, (4, 4), order="F")   # TODO Preveri, kdaj se updata O_T_EE
            # self._command.x = t2x(T_D)

        T = np.reshape(self._state.O_T_EE, (4, 4), order="F")
        self._actual.x = t2x(T)
        self._actual.v = self.Jacobi(self._state.q) @ self._state.dq

        # Get safety status
        self.joint_contacts = self._state.joint_contact
        self.joint_collisions = self._state.joint_collision
        self.cartesian_contacts = self._state.cartesian_contact
        self.cartesian_collisions = self._state.cartesian_collision

        self._actual.FT = self._state.K_F_ext_hat_K

        self.StiffnessFrame = np.reshape(self._state.EE_T_K, (4, 4), order="F")
        self._actual.FT = frame2world(self._state.K_F_ext_hat_K, self.StiffnessFrame, 1)  # external EE wrench in tool CS (considering EETK)
        self._actual.F = self._actual.FT[:2]
        self._actual.T = self._actual.FT[2:]
        self._actual.trqExt = self._state.tau_ext_hat_filtered

        if self.EEFixed:
            self.TObject = map_pose(x=self.BaseToWorld(self._actual.x), out="T")

        self._tt = self._state.time.to_sec()
        self._last_update = self.simtime()  # Do not change !

    # Movements
    def GoTo_q(self, q: JointConfigurationType, qdot: Optional[JointVelocityType] = None, trq: Optional[JointTorqueType] = None, wait: Optional[float] = None, stiffness: Optional[JointConfigurationType] = None, damping: Optional[JointConfigurationType] = None, **kwargs: Any) -> int:
        """
        Update joint positions and wait

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
        stiffness : JointConfigurationType, optional
            Desired joint stiffness (nj, )
        damping : JointConfigurationType, optional
            Desired joint damping (nj, ).

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        if qdot is None:
            qdot = np.zeros(self.nj)
        else:
            qdot = vector(qdot, dim=self.nj)
        if wait is None:
            wait = self.tsamp

        self._synchro_control(wait)
        if stiffness is not None:
            stiffness = vector(stiffness, dim=self.nj)
            self.ctrl.set_stiffness(stiffness)
        if damping is not None:
            damping = vector(damping, dim=self.nj)
            self.ctrl.set_damping(damping)
        self.ctrl.set_control(q, qdot)

        self._command.q = q
        self._command.qdot = qdot

        x, J = self.Kinmodel(q)
        self._command.x = x
        self._command.v = J @ qdot
        self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value

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
            WrenchType, optional, NOT USED!
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
        if wait is None:
            wait = self.tsamp
        task_space = kwargs.get("task_space", "World")
        if check_option(task_space, "World"):
            x = self.WorldToBase(x)
            v = self.WorldToBase(v, typ="Twist")
        elif check_option(task_space, "Robot"):
            pass
        elif check_option(task_space, "Object"):
            x = self.ObjectToWorld(x)
            v = self.ObjectToWorld(v, typ="Twist")
            x = self.WorldToBase(x)
            v = self.WorldToBase(v, typ="Twist")
        else:
            raise ValueError(f"Task space '{task_space}' not supported")

        if self._control_strategy in ["JointImpedance", "JointPosition"]:
            self._command.rx = x
            self._command.rv = v
            self._last_status = self.GoTo_TC(x, v=v, **kwargs)
        elif self._control_strategy in ["CartesianImpedance", "CartesianImpedanceExtended"]:
            self._last_status = self.GoTo_X(x, wait=wait, **kwargs)
        else:
            raise ValueError(f"Control strategy '{self._control_strategy}' not supported")

        return self._last_status

    def GoTo_X(self, x: Union[Pose3DType, HomogeneousMatrixType], v: Optional[Velocity3DType] = None, FT: Optional[WrenchType] = None, wait: Optional[float] = None, impedance: Optional[ArrayLike] = None, damping: Optional[ArrayLike] = None, R: Optional[ArrayLike] = None, **kwargs: Any) -> int:
        """Update task pose and wait

        The robot will be moved using Cartesian control.

        Parameters
        ----------
        x : Union[Pose3DType, HomogeneousMatrixType]
            Target end-effector pose in Cartesian space. Can be in different forms (e.g., Pose, Transformation matrix).
        v : Velocity3DType, optional
            Target end-effector velocity in Cartesian space. Default is a zero velocity vector (6,).
        FT : WrenchType, optional
            WrenchType, optional, NOT USED!
            Target force/torque in Cartesian space. Default is a zero wrench vector (6,).
        wait : float, optional
            The time to wait after the movement, by default the sample time (`self.tsamp`).
        impedance : ArrayLike, optional
            The Cartesian impedance to set during the movement.
        damping : ArrayLike, optional
            The Cartesian damping to set during the movement.
        R : ArrayLike, optional
            The rotation matrix for impedance and damping directions, if applicable.
        **kwargs : dict
            Additional keyword arguments for special use.

        Returns
        -------
        int
            Status of the move (0 for success, non-zero for error).
        """
        x = x2x(x)
        p = x[:3]
        Q = q_wxyz2xyzw(x[3:])
        if wait is None:
            wait = self.tsamp
        self._synchro_control(wait)
        if self._control_strategy in ["CartesianImpedance"]:
            if impedance is not None:
                impedance = matrix(impedance, shape=(6, 6))
                self.ctrl.set_impedance(impedance)
            if damping is not None:
                if not isscalar(damping):
                    raise ValueError("Damping must be a scalar value for CartesianImpedance strategy")
                self.ctrl.set_damping_ratio(damping, R)
            self.ctrl.set_control(p, Q)
        elif self._control_strategy in ["CartesianImpedanceExtended"]:
            if R is not None:
                R = matrix(R, shape=(6, 6))
            else:
                R = np.eye(6)
            if impedance is not None:
                impedance = vector(impedance, dim=6)
                self.ctrl.set_impedance(impedance, R)
            if damping is not None:
                damping = vector(damping, dim=6)
                self.ctrl.set_damping_ratio(damping, R)
            self.ctrl.set_control(p, Q, impedance, FT, R)
        else:
            raise ValueError(f"Control strategy '{self._control_strategy}' not supported")
        self._command.x = x
        self._command.v = v
        self._command.q = self._actual.q.copy()
        self._command.qdot = self._actual.qdot.copy()
        self.Update()
        return MotionResultCodes.MOTION_SUCCESS.value

    def JMove(
        self,
        q: JointConfigurationType,
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[JointConfigurationType] = None,
        wait: Optional[float] = None,
        traj: Optional[str] = None,
        added_trq: Optional[JointTorqueType] = None,
        min_joint_dist: Optional[float] = None,
        asynchronous: bool = False,
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
        if use_internal is None:
            use_internal = self._default.UseInternal

        if use_internal:
            if t is not None:
                vel = np.abs(q - self._actual.q) / t
                vel_fac = np.max(vel / self.qdot_max)
            elif vel is not None:
                vel_fac = np.max(vel / self.qdot_max)
            if vel_fac is None:
                vel_fac = self._default.VelocityScaling
            else:
                vel_fac = np.clip(vel_fac, 0.01, 1)

            self.Message(f"Internal JMove started: {q} with velocity factor {vel_fac:.2f}", 1)
            _tmperr = self.panda.move_to_joint_position(q, speed_factor=vel_fac if vel_fac is not None else 0.2)
            self._command.q = q
            self._command.qdot = np.zeros(7)
            x, J = self.Kinmodel(q)
            self._command.x = x
            self._command.v = np.zeros(6)
            self.Update()
            self.Message("Internal JMove finshed", 1)
            self.SetStrategy(self.control_strategy)
            return MotionResultCodes.MOTION_FAILURE.value if _tmperr else MotionResultCodes.MOTION_SUCCESS.value
        elif asynchronous:
            if not self._control_strategy.startswith("Joint"):
                self.WarningMessage("Not in joint control mode - JMove not executed")
                return MotionResultCodes.WRONG_STRATEGY.value

            self.Message("ASYNC JMove", 2)
            _th = Thread(target=self._JMove, args=(q,), kwargs={"t": t, "vel": vel, "vel_fac": vel_fac, "wait": wait, "traj": traj, "added_trq": added_trq, "min_joint_dist": min_joint_dist, **kwargs}, daemon=True)
            _th.start()
            return _th
        else:
            return self._JMove(q, t=t, vel=vel, vel_fac=vel_fac, wait=wait, traj=traj, added_trq=added_trq, min_joint_dist=min_joint_dist, **kwargs)

    def CMove(
        self,
        x: Union[Pose3DType, HomogeneousMatrixType, RotationMatrixType],
        t: Optional[float] = None,
        vel: Optional[float] = None,
        vel_fac: Optional[float] = None,
        traj: Optional[str] = None,
        short: Optional[bool] = None,
        wait: Optional[float] = None,
        task_space: Optional[str] = None,
        added_FT: Optional[WrenchType] = None,
        state: str = "Commanded",
        min_pos_dist: Optional[float] = None,
        min_ori_dist: Optional[float] = None,
        asynchronous: bool = False,
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
        if use_internal is None:
            use_internal = self._default.UseInternal

        if use_internal:
            _x = spatial2x(x)
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

            if t is not None:
                vel = np.abs(xerr(_x, self._actual.x)) / t
                vel_fac = np.max(vel / self.v_max)
            elif vel is not None:
                vel_fac = np.max(vel / self.v_max)
            if vel_fac is None:
                vel_fac = self._default.VelocityScaling
            else:
                vel_fac = np.clip(vel_fac, 0.01, 1)
            _p = _x[:3]
            _Q = q_wxyz2xyzw(x[3:])

            self.Message(f"Internal CMove started: {_x} with velocity factor {vel_fac:.2f}", 1)
            _tmperr = self.panda.move_to_pose(_p, _Q, speed_factor=vel_fac if vel_fac is not None else 0.2)
            self._command.x = x
            self._command.v = np.zeros(6)
            self._command.q = self._actual.q.copy()
            self._command.qdot = self._actual.qdot.copy()
            self.Update()
            self.Message("Internal CMove finished", 1)
            self.SetStrategy(self.control_strategy)
            return MotionResultCodes.MOTION_FAILURE.value if _tmperr else MotionResultCodes.MOTION_SUCCESS.value
        elif asynchronous:
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

    # Behaviour and collisions
    def CheckContacts(self) -> bool:
        """
        Checks if a contact is ocurring on any of joint axes or the robot's Cartesian axes.

        Returns
        -------
        bool
            True if a contact is detected, False otherwise.
        """
        self.UpdateState()
        for v in self._state.joint_contact:
            if v > 0:
                return True
        for v in self._state.cartesian_contact:
            if v > 0:
                return True
        # If we get to here, no collisions are detected
        return False

    def GetCollisions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the current joint and Cartesian collisions.

        Returns
        -------
        tuple
            A tuple containing two lists: joint collisions and Cartesian collisions.
        """
        _qcol = self.joint_collisions
        _xcol = self.cartesian_collisions
        return _qcol, _xcol

    def GetContacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the current joint and Cartesian contacts.

        Returns
        -------
        tuple
            A tuple containing two lists: joint contacts and Cartesian contacts.
        """
        _qcon = self.joint_contacts
        _xcon = self.cartesian_contacts
        return _qcon, _xcon

    def SetCollisionBehavior(self, tq: Optional[Union[float, JointTorqueType]] = None, F: Optional[Union[float, WrenchType]] = None, tq_low: Optional[Union[float, JointTorqueType]] = None, F_low: Optional[Union[float, WrenchType]] = None) -> None:
        """
        Sets the collision behavior.

        Parameters
        ----------
        tq : Union[float, JointTorqueType], optional
            collision joint torque treshold [Nm], if None defualts are used. By default None
        F : Union[float, WrenchType], optional
            collision task force treshold [N], if None defualts are used. By default None
        tq_low : Union[float, JointTorqueType], optional
            lower collision joint torque treshold [Nm], by default None
        F_low : Union[float, WrenchType], optional
            lower collision task force treshold [N], by default None

        Returns
        -------
        None

        Notes
        -----
        Forces or torques between lower and upper threshold are shown as contacts
        in the RobotState. Forces or torques above the upper threshold are
        registered as collision and cause the robot to stop moving.

        """
        if tq is None:
            tq = self._franka_default.collision_behavior.upper_torque_thresholds
        if F is None:
            F = self._franka_default.collision_behavior.upper_force_thresholds
        if isscalar(tq):
            tq = np.ones(7) * tq
        else:
            tq = vector(tq, dim=7)
        if isscalar(F):
            F = np.ones(6) * F
        else:
            F = vector(F, dim=6)
        if tq_low is None:
            tq_low = tq
        elif isscalar(tq_low):
            tq_low = np.ones(7) * tq_low
        else:
            tq_low = vector(tq_low, dim=7)
        if F_low is None:
            F_low = F
        elif isscalar(F_low):
            F_low = np.ones(6) * F_low
        else:
            F_low = vector(F_low, dim=6)
        self.robot.set_collision_behavior(tq_low, tq, F_low, F)

    # Strategies
    def AvailableStrategies(self) -> List[str]:
        """
        Returns a list of available control strategies for the robot.

        Returns
        -------
        List[str]
            A list of available control strategies.
        """
        return list(self._strategy_controller_mapping.keys())

    def SetStrategy(self, new_strategy: str) -> None:
        """
        Switch the active control strategy.

        Parameters
        ----------
        new_strategy : str
            Name of the control strategy to activate.

        Returns
        -------
        None
            This method activates the requested controller in place.

        Raises
        ------
        ValueError
            If the specified strategy is not supported.
        """
        if new_strategy in self._strategy_controller_mapping:
            self.ctrl = self._strategy_controller_mapping[new_strategy]
            self._control_strategy = new_strategy
            # self.panda.stop_controller()
            self.panda.start_controller(self.ctrl)
        else:
            raise ValueError(f"Strategy '{new_strategy}' not supported")

    def TeachingMode(self, mode: bool, damping: Optional[ArrayLike] = None) -> None:
        """
        Enable or disable gravity-compensated teaching mode.

        In teaching mode the robot becomes compliant / backdrivable so that it can
        be guided by hand. Optionally, joint-space damping factors can be provided
        to adjust how easily each joint can be moved.

        Parameters
        ----------
        mode : bool
            If True, enable teaching mode (robot is backdrivable and the current
            controller is paused). If False, disable teaching mode and restart
            the previously active controller.
        damping : ArrayLike, optional
            Joint-space damping factors in the range [0, 1]. If a scalar is given,
            the same damping value is applied to all 7 joints. If an array is given,
            it must be a 7D vector specifying the damping factor per joint.
            If None, the default damping behavior of :mod:`panda_py` is used.

        Raises
        ------
        ValueError
            If `damping` is provided but is neither a scalar nor a 7D vector.

        Notes
        -----
        - Damping values are clipped to the interval [0, 1].
        - When ``mode`` is set to ``False``, this method calls
          :meth:`ResetCurrentTarget` and restarts the current controller via
          ``self.panda.start_controller(self.ctrl)``.
        """
        if damping is None:
            self.panda.teaching_mode(mode)
        else:
            if isscalar(damping):
                damping = np.zeros(7) * damping
            elif isvector(damping, dim=7):
                pass
            else:
                raise ValueError("Damping must be scalar of 7D vector, all positive.")
            damping = np.clip(damping, 0, 1)
            self.panda.teaching_mode(mode, damping=damping)
        if not mode:
            self.ResetCurrentTarget()
            self.panda.start_controller(self.ctrl)

    def SetTeachMode(self) -> None:
        """
        Enable teach mode.

        Returns
        -------
        None
            This method enables compliant hand-guiding mode.
        """
        self.TeachingMode(True)
        self.Message("Robot is entering Teach mode.", 2)

    def EndTeachMode(self) -> None:
        """
        Disable teach mode.

        Returns
        -------
        None
            This method restores the standard robot control mode.
        """
        self.ResetCurrentTarget()
        self.TeachingMode(False)
        self.Message("Robot is ending Teach mode.", 2)

    # Cartesian compliance
    def GetCartesianCompliance(self) -> Tuple[np.ndarray, float, float]:
        """
        Returns the current Cartesian compliance settings.

        Returns
        -------
        Tuple[np.ndarray, float, float]
            A tuple containing the Cartesian stiffness, damping ratio, and nullspace damping.
        """
        return self.cartesian_impedance, self.cartesian_damping_ratio, self.cartesian_nullspace_stiffness

    def SetCartesianCompliance(self, impedance: Optional[ArrayLike] = None, damping_ratio: Optional[float] = None, nullspace_stiffness: Optional[float] = None, hold_pose: bool = True) -> None:
        """
        Sets the Cartesian compliance settings.

        Parameters
        ----------
        impedance : ArrayLike, optional
            The Cartesian impedance (stiffness) to set. Can be a 6D vector or a 6x6 matrix or 2D vector.
        damping_ratio : float, optional
            The damping ratio to set.
        nullspace_stiffness : float, optional
            The nullspace stiffness to set.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        cartesian_controller = self._strategy_controller_mapping.get("CartesianImpedance")
        if impedance is not None:
            if isvector(impedance, 6):
                impedance = np.diag(impedance)
            elif isvector(impedance, 2):
                Kp = np.ones(3) * impedance[0]
                Kr = np.ones(3) * impedance[1]
                impedance = np.diag(np.hstack((Kp, Kr)))
            elif ismatrix(impedance, (6, 6)):
                pass
            else:
                raise ValueError("Impedance must be a 6D vector or a 6x6 matrix")
            self.cartesian_impedance = impedance
            if cartesian_controller is not None:
                cartesian_controller.set_impedance(impedance)
        if damping_ratio is not None:
            self.cartesian_damping_ratio = damping_ratio
            if cartesian_controller is not None:
                cartesian_controller.set_damping_ratio(damping_ratio)
        if nullspace_stiffness is not None:
            self.cartesian_nullspace_stiffness = nullspace_stiffness
            if cartesian_controller is not None:
                cartesian_controller.set_nullspace_stiffness(nullspace_stiffness)

    def GetCartesianStiffness(self) -> np.ndarray:
        """
        Returns the current Cartesian stiffness settings.

        Returns
        -------
        np.ndarray
            The current Cartesian stiffness settings.
        """
        return self.cartesian_stiffness

    def SetCartesianStiffness(self, impedance: Optional[ArrayLike] = None, hold_pose: bool = True) -> None:
        """
        Sets the Cartesian impedance settings.

        Parameters
        ----------
        impedance : ArrayLike, optional
            The Cartesian impedance (stiffness) to set. Can be a 6D vector or a 6x6 matrix or 2D vector.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        self.SetCartesianCompliance(impedance=impedance, hold_pose=hold_pose)

    def GetCartesianDamping(self) -> float:
        """
        Returns the current Cartesian damping settings.

        Returns
        -------
        float
            The current Cartesian damping settings.
        """
        return self.cartesian_damping_ratio

    def SetCartesianDamping(self, damping_ratio: Optional[float] = None, hold_pose: bool = True) -> None:
        """
        Sets the Cartesian damping settings.

        Parameters
        ----------
        damping_ratio : float, optional
            The damping ratio to set.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        return self.SetCartesianCompliance(damping_ratio=damping_ratio, hold_pose=hold_pose)

    def SetCartesianSoft(self, stiffness: ArrayLike, hold_pose: bool = True) -> None:
        """
        Sets the Cartesian stiffness to a fraction of the default stiffness.
        Parameters
        ----------
        stiffness : ArrayLike
            The fraction of the default stiffness to set. Can be a scalar or a vector of shape (2,), shape (3,) or (6,).
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None

        Notes
        -----
        - If `stiffness` is a scalar, it is applied uniformly to all Cartesian axes.
        - If `stiffness` is a 2D vector, the first element is applied to the position axes and the second to the rotation axes.
        - If `stiffness` is a 3D vector, it is applied to both position and rotation axes.
        - The stiffness values are clipped between MinSoftnessForMotion and 1.0 before being applied.
        """
        if isscalar(stiffness):
            fac_p = np.ones(3) * stiffness
            fac_r = fac_p
        elif isvector(stiffness, dim=2):
            fac_p = np.zeros(3) * stiffness
            fac_r = np.zeros(3) * stiffness
        elif isvector(stiffness, dim=3):
            fac_p = stiffness
            fac_r = stiffness
        else:
            fac = vector(stiffness, dim=6)
            fac_p = fac[:3]
            fac_r = fac[3:]

        fac_p = np.clip(fac_p, self._franka_default.MinSoftnessForMotion, 1.0)
        fac_r = np.clip(fac_r, self._franka_default.MinSoftnessForMotion, 1.0)
        fac = np.max(np.hstack((fac_p, fac_r)))
        self.SetCartesianStiffness(impedance=self._franka_default.CartesianImpedance * fac, hold_pose=hold_pose)

    # Joint compliance
    def GetJointCompliance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the current joint compliance settings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the current joint stiffness and damping settings.
        """
        return self.joint_stiffness, self.joint_damping

    def SetJointCompliance(self, stiffness: Optional[ArrayLike] = None, damping: Optional[ArrayLike] = None, hold_pose: bool = True) -> None:
        """
        Sets the joint compliance settings.

        Parameters
        ----------
        stiffness : ArrayLike, optional
            The joint stiffness to set. Can be a 7D vector or a scalar.
        damping : ArrayLike, optional
            The joint damping to set. Can be a 7D vector or a scalar.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        joint_controller = self._strategy_controller_mapping.get("JointImpedance")
        if stiffness is not None:
            if isvector(stiffness, 7):
                pass
            elif isscalar(stiffness):
                stiffness = np.ones(7) * stiffness
            else:
                raise ValueError("Stiffness must be a 7D vector or scalar")
            self.joint_stiffness = stiffness
            if joint_controller is not None:
                joint_controller.set_stiffness(stiffness)
        if damping is not None:
            self.joint_damping = damping
            if joint_controller is not None:
                joint_controller.set_damping(damping)

    def GetJointStiffness(self) -> np.ndarray:
        """
        Returns the current joint stiffness settings.

        Returns
        -------
        np.ndarray
            The current joint stiffness settings.
        """
        return self.joint_stiffness

    def SetJointStiffness(self, stiffness: Optional[ArrayLike] = None, hold_pose: bool = True) -> None:
        """
        Sets the joint stiffness settings.

        Parameters
        ----------
        stiffness : ArrayLike, optional
            The joint stiffness to set. Can be a 7D vector or a scalar.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        self.SetJointCompliance(stiffness=stiffness, hold_pose=hold_pose)

    def GetJointDamping(self) -> np.ndarray:
        """
        Returns the current joint damping settings.

        Returns
        -------
        np.ndarray
            The current joint damping settings.
        """
        return self.joint_damping

    def SetJointDamping(self, damping: Optional[ArrayLike] = None, hold_pose: bool = True) -> None:
        """
        Sets the joint damping settings.

        Parameters
        ----------
        damping : ArrayLike, optional
            The joint damping to set. Can be a 7D vector or a scalar.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None
        """
        self.SetJointCompliance(damping=damping, hold_pose=hold_pose)

    def SetJointSoft(self, stiffness_factor: ArrayLike, hold_pose: bool = True) -> None:
        """
        Sets the joint stiffness to a fraction of the default stiffness.

        Parameters
        ----------
        stiffness_factor : ArrayLike
            The fraction of the default stiffness to set. Can be a scalar or a 7D vector.
        hold_pose : bool, optional
            If True, holds the current pose while setting compliance.

        Returns
        -------
        None

        Notes
        -----
        - If `stiffness` is a scalar, it is applied uniformly to all joint axes.
        - The stiffness values are clipped between MinSoftnessForMotion and 1.0 before being applied.
        """
        if isscalar(stiffness_factor):
            fac = np.ones(7) * stiffness_factor
        else:
            fac = vector(stiffness_factor, dim=7)
        fac = np.clip(fac, self._franka_default.MinSoftnessForMotion, 1.0)
        self.SetJointCompliance(stiffness=self._franka_default.JointStiffness * fac, damping=self._franka_default.JointDamping * fac, hold_pose=hold_pose)

    # TCP and load
    def Update_tcp_from_franka_state(self) -> None:
        """
        Reads the currently set TCP in franka desk, uses it as default gripper TCP,
        and as the TCPin the robot object.

        Returns
        -------
        None
        """
        _TCP = self._state.F_T_EE
        self.TCPGripper = np.reshape(_TCP, newshape=(4, 4), order="f")
        robot.SetTCP(self, tcp=self.TCPGripper, frame="Flange")

    def SetTCP(self, tcp: Optional[TCPType] = None, frame: str = "Gripper") -> None:
        """
        Set the Tool Center Point (TCP) of the robot.

        Parameters
        ----------
        tcp : TCPType, optional
            The transformation matrix or pose of the TCP. Default is the identity matrix.
        frame : str, optional
            The frame to which the TCP is referenced. Can be "Gripper" or "Flange". Default is "Gripper".

        Returns
        -------
        None
        """
        if tcp is not None:
            _tcp = spatial2t(tcp)
        else:
            _tcp = np.eye(4)
        robot.SetTCP(self, tcp=_tcp, frame=frame)
        if check_option(frame, "Gripper"):
            NE_TCP = tcp
        elif check_option(frame, "Flange"):
            NE_TCP = self.TCPGripper / tcp
        else:
            raise ValueError(f"Frame {frame} not supported")
        self.SetEEFrame(NE_TCP)

    def SetEEFrame(self, NE_T_EE: HomogeneousMatrixType) -> None:
        """
        Sets the transformation $^{NE}T_{EE}$ from nominal end effector to end effector frame.

        Parameters
        ----------
        NE_T_EE : HomogeneousMatrixType
            The transformation matrix from nominal end effector frame to end effector frame.

        Notes
        -----
        The transformation matrix is sent as a vectorized 4x4 matrix in column-major format.
        """
        if any(NE_T_EE[:3, 3] > 0.5):
            self.WarningMessage("Setting EE frame with translation larger than 0.5m not possible")
        else:
            _frame = spatial2t(NE_T_EE)
            self.panda.stop_controller()
            self.robot.set_ee(_frame.flatten(order="F"))
            self.panda.start_controller(self.ctrl)

    def SetStiffnessFrame(self, EE_T_K: HomogeneousMatrixType) -> None:
        """
        Sets the transformation $^{EE}T_K$ from end effector frame to stiffness frame.
        The transformation matrix is represented as a vectorized 4x4 matrix in column-major format.

        Parameters
        ----------
        EE_T_K : HomogeneousMatrixType
            The transformation matrix from end effector frame to stiffness frame.

        Returns
        -------
        None

        Notes
        -----
        The transformation matrix is sent as a vectorized 4x4 matrix in column-major format.
        """
        _frame = spatial2t(EE_T_K)
        self.StiffnessFrame = _frame
        self.robot.set_k(_frame.flatten(order="F"))

    def GetStiffnessFrame(self) -> HomogeneousMatrixType:
        """
        Gets the transformation $^{EE}T_K$ from end effector frame to stiffness frame.

        Returns
        -------
        HomogeneousMatrixType
            The transformation matrix from end effector frame to stiffness frame.
        """
        return self.StiffnessFrame

    def SetLoad(self, load: Optional[load_params] = None, mass: Optional[float] = None, COM: Optional[Vector3DType] = None, inertia: Optional[np.ndarray] = None) -> None:
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
        inertia : np.ndarray, optional
            The inertia of the load, by default None.

        Returns
        -------
        None

        Notes
        -----
        The center of mass is specified in the end effector frame.
        The inertia matrix should be provided in the end effector frame.
        """
        if isinstance(load, load_params):
            self.Load = load
        else:
            if mass is not None:
                if mass < 0:
                    raise ValueError("Load mass cannot be negative")
                self.Load.mass = mass
            if COM is not None:
                if not isvector(COM, 3):
                    raise ValueError("Load COM must be a vector of shape (3,)")
                self.Load.COM = COM
            if inertia is not None:
                if not ismatrix(inertia, (3, 3)):
                    raise ValueError("Load inertia must be a 3x3 matrix")
                self.Load.inertia = inertia

        self.robot.set_load(self.Load.mass, self.Load.COM, self.Load.inertia.flatten(order="F"))

    # Status
    def isConnected(self) -> bool:
        """
        Checks if the robot is connected.

        Returns
        -------
        bool
            True if the robot is connected, False otherwise.
        """
        return self._connected

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
        self.UpdateState()
        _mode = self._state.robot_mode.value
        return _mode in [1, 2]

    def isActive(self) -> bool:
        """
        Check if the connection to robot is established.

        Returns
        -------
        bool
            Indicating if the connection to robot is established.
        """
        return self.isReady()

    def Check(self, silent: bool = False) -> List[str]:
        """
        Checks the status of the robot.

        Parameters
        ----------
        silent : bool, optional
            Present for API compatibility. It is currently not used.

        Returns
        -------
        List[str]
            List of active robot status conditions.
        """
        self.UpdateState()
        _mode = self._state.robot_mode
        _err = []
        if _mode.value not in [1, 2]:
            _err.append(_mode.name)
        return _err

    def ErrorRecovery(self) -> int:
        """
        Recover the Panda robot from an error state.

        Returns
        -------
        int
            Motion result code returned by the recovery routine.
        """
        self.ResetCurrentTarget()
        self.panda.recover()
        sleep(1)
        self.SetStrategy(self.control_strategy)
        return MotionResultCodes.MOTION_SUCCESS.value

    def StartCapture(self, use_internal: Optional[bool] = None, max_samples: Optional[int] = None) -> None:
        """
        Start the capture process, ensuring that the update is enabled.
        """
        if use_internal is None:
            use_internal = self._default.UseInternal

        if use_internal:
            if max_samples is None:
                self.WarningMessage("Capture buffer size 'max_samples' not defined. Capture is not started!")
                return
            self.panda.enable_logging(max_samples)
            self.Message("Internal capture started", 2)
        else:
            if not self._do_update:
                self.WarningMessage("Update is not enabled")
            self._do_capture = True
            self.Message("Capture started", 2)
            self.Update()

    def StopCapture(self) -> None:
        """
        Stop the capture process.

        Returns
        -------
        None
        """
        self.Message("Capture stopped", 2)
        self._do_capture = False
        self.panda.disable_logging()

    def GetLastCapturedData(self) -> dict:
        return self.panda.get_log()
