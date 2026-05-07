"""OMPL MuJoCo Motion Planning Utilities.

This module provides a collection of utilities that integrate the
`MuJoCo` physics engine with the `OMPL` (Open Motion Planning Library)
to perform collision-aware motion planning in joint space.

It offers helper functions for:

- Creating OMPL state spaces and converting between OMPL states and
  NumPy vectors.
- Building MuJoCo-based state-validity checkers that detect collisions
  using MuJoCo's contact data.
- Configuring a wide selection of OMPL planners, including
  sampling-based, optimal, and meta-planners (e.g., BIT*, RRT#, PRM*,
  CForest, APS).
- Performing complete motion-planning queries returning both raw and
  interpolated paths.
- Planning motions for MuJoCo-modeled robots (or multi-robot systems)
  with optional clearance inflation for more conservative planning.

The core functionality revolves around mapping OMPL states to MuJoCo
`qpos`, invoking MuJoCo forward dynamics to detect collisions, and using
standard OMPL pipelines to generate feasible paths.

Dependencies
------------
- **MuJoCo** (https://mujoco.org/)
- **OMPL** (https://ompl.kavrakilab.org/)

Typical Workflow
----------------
1. Extract robot joint limits from a MuJoCo model.
2. Build an OMPL RealVector state space.
3. Construct a MuJoCo-based validity checker using
   :func:`make_mujoco_validity_fn`.
4. Set and configure an OMPL planner.
5. Solve the motion-planning problem with :func:`plan_motion` or
   :func:`plan_robot_motion`.

Notes
-----
- MuJoCo `MjData` is not thread-safe. For safety, a private copy of
  `MjData` is used internally for validity checking.
- Some OMPL planners may not be available in all Python builds; unknown
  planner names result in informative errors.
- Collision checking can consider only robot contacts or all contacts,
  depending on user settings.

This module is intended for research, prototyping, and teaching
applications involving sampling-based planning for articulated robots in
simulation.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np
from copy import copy
from typing import Callable, Optional, Tuple, Iterable, Sequence, Set, TYPE_CHECKING

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

try:
    from ompl import base as ob, geometric as og
except Exception as e:
    raise e from RuntimeError("Open Motion Planning Library (OMPL) not installed. \nYou can install OMPL through pip:\n   pip install ompl")


from robotblockset.mujoco.tools_pymujoco import get_robot_joints_data, get_body_descendants, get_geoms_of_body
from robotblockset.tools import check_option
from robotblockset.rbs_typing import JointConfigurationType

if TYPE_CHECKING:
    from robotblockset.robots import robot


def make_state_space(bounds: np.ndarray) -> ob.RealVectorStateSpace:
    """
    Create a RealVector state space with the given bounds.

    Parameters
    ----------
    bounds : np.ndarray
        Per-dimension lower/upper limits; shape ``(n, 2)``.

    Returns
    -------
    ompl.base.RealVectorStateSpace
        State space with bounds applied.

    Raises
    ------
    ValueError
        If ``bounds`` is not ``(n, 2)`` or any ``low > high``.
    """
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must have shape (n, 2)")
    if arr.shape[0] == 0:
        raise ValueError("bounds must define at least one dimension")
    if np.any(arr[:, 0] > arr[:, 1]):
        i = int(np.argmax(arr[:, 0] > arr[:, 1]))
        raise ValueError(f"low > high for dimension {i}: {arr[i]}")

    dim = int(arr.shape[0])
    space = ob.RealVectorStateSpace(dim)
    rb = ob.RealVectorBounds(dim)
    for i, (lo, hi) in enumerate(arr):
        rb.setLow(i, float(lo))
        rb.setHigh(i, float(hi))
    space.setBounds(rb)
    return space


def vec_to_state(space: ob.StateSpace, v: Sequence[float]) -> ob.State:
    """
    Convert a vector of coordinates to an OMPL state.

    Parameters
    ----------
    space : ob.StateSpace
        Target state space (e.g., ``RealVectorStateSpace``).
    v : Sequence[float]
        Coordinates; length must equal ``space.getDimension()``.

    Returns
    -------
    ompl.base.State
        Newly created state with values from ``v``.

    Raises
    ------
    ValueError
        If ``len(v)`` does not match the space dimension.
    """
    dim = space.getDimension()
    if len(v) != dim:
        raise ValueError(f"Expected vector of length {dim}, got {len(v)}")
    s = ob.State(space)
    for i, val in enumerate(v):
        s[i] = float(val)
    return s


def state_to_vec(s: ob.State, n: int) -> np.ndarray:
    """
    Convert an OMPL state to a NumPy vector.

    Parameters
    ----------
    s : ob.State
        State to convert.
    n : int
        Dimension of the state (e.g., ``space.getDimension()``).

    Returns
    -------
    (n,) ndarray of float
        State coordinates as a NumPy array.
    """
    return np.fromiter((s[i] for i in range(n)), dtype=float, count=n)


def make_mujoco_validity_fn(model: mujoco.MjModel, data: mujoco.MjData, robot_geoms_ids: Iterable[int], qaddr: Sequence[int], *, only_robot_contacts: bool = True) -> Callable[[ob.State], bool]:
    """
    Build an OMPL state-validity function that uses MuJoCo contacts.

    The returned callable maps an OMPL state (joint vector) to ``True``
    iff the MuJoCo scene has **no contacts**. By default, only contacts
    where **either geom belongs to the robot** are considered (set
    ``only_robot_contacts=False`` to consider all contacts).

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    data : mujoco.MjData
        A reference data object providing the template ``qpos`` (start pose).
        This object is **not** mutated.
    robot_geoms_ids : Iterable[int]
        IDs of geoms that belong to the robot (used to filter contacts).
    qaddr : Sequence[int]
        Indices into ``qpos`` for the planned joints (order must match the
        OMPL RealVector state).
    only_robot_contacts : bool, optional
        If ``True`` (default), only contacts where ``geom1`` or ``geom2`` is
        in ``robot_geom_ids`` are considered. If ``False``, *any* contact
        invalidates the state.

    Returns
    -------
    Callable[[ompl.base.State], bool]
        A function that returns ``True`` for valid (collision-free) states.

    Notes
    -----
    - A private ``MjData`` is created for thread-safety; MuJoCo data is not
      thread-safe to share across calls.
    - ``contact.dist`` is positive when separated, 0 at touching, and negative
      when penetrating.
    """
    # Normalize inputs
    robot_geoms: Set[int] = {int(g) for g in robot_geoms_ids}
    qpos_template = np.array(data.qpos, copy=True)

    # Private work buffer for kinematics/contacts
    _data = copy(data)

    def is_valid(state: ob.State) -> bool:
        # Fill qpos from OMPL state
        q = qpos_template.copy()
        for i, adr in enumerate(qaddr):
            q[adr] = float(state[i])
        _data.qpos[:] = q
        _data.qvel[:] = 0.0

        # Compute kinematics + contacts
        mujoco.mj_forward(model, _data)

        # Reject if any relevant contact violates clearance
        for k in range(_data.ncon):
            c = _data.contact[k]
            if only_robot_contacts and (c.geom1 not in robot_geoms and c.geom2 not in robot_geoms):
                continue
            return False
        return True

    return is_valid


def set_optimizing_planner(ss: og.SimpleSetup, name: str, *, objective: Optional[ob.OptimizationObjective] = None) -> ob.Planner:
    """
    Attach an **optimizing** OMPL planner to a ``SimpleSetup`` by name.

    This convenience maps common planner names (e.g., ``"BIT*"`` or ``"RRT#"``)
    to their Python classes in ``ompl.geometric`` and sets a path-length
    optimization objective by default.

    Parameters
    ----------
    ss : og.SimpleSetup
        Problem container with a configured state space and validity checker.
    name : str
        Planner name (case-insensitive). Supported keys include:
        ``"PRM*"`` (PRMstar), ``"LazyPRM*"`` (LazyPRMstar),
        ``"RRT*"`` (RRTstar), ``"RRT#"`` (RRTsharp), ``"RRTX"`` (RRTXstatic),
        ``"Informed RRT*"`` (InformedRRTstar), ``"BIT*"`` (BITstar),
        ``"ABIT*"`` (ABITstar), ``"AIT*"`` (AITstar), ``"EIT*"`` (EITstar),
        ``"LBTRRT"`` (LBTRRT), ``"SST"`` (SST), ``"T-RRT"`` (TRRT),
        ``"SPARS"`` (SPARS), ``"SPARS2"`` (SPARStwo), ``"FMT*"`` (FMT),
        ``"ST-RRT*"`` (STRRTstar),
        meta-planners: ``"CForest"`` (CForest), ``"APS"`` / ``"AnytimePathShortening"``.
    objective : ob.OptimizationObjective, optional
        Custom objective (defaults to ``PathLengthOptimizationObjective``).
        Meta-planners and *-star variants expect objectives that obey the
        triangle inequality (e.g., path length).

    Returns
    -------
    ompl.base.Planner
        The instantiated planner attached to ``ss`` via ``ss.setPlanner(...)``.

    Raises
    ------
    ValueError
        If the planner name is unknown or not available in the current build.

    Notes
    -----
    - Meta-planners require adding sub-planners:

      - ``CForest`` runs several optimal planners in parallel; add with
        ``planner.addPlanner(...)``.
      - ``AnytimePathShortening`` (``APS``) wraps one or more planners and repeatedly
        hybridizes/shortcuts; add with ``planner.addPlanner(...)``.

    - Some classes such as ``ABITstar``, ``AITstar``, ``EITstar``, and
      ``RRTXstatic``
      may not be present in older OMPL Python wheels.
    """
    si = ss.getSpaceInformation()

    # Default objective: path length
    if objective is None:
        objective = ob.PathLengthOptimizationObjective(si)
    ss.setOptimizationObjective(objective)

    key = name.strip().lower().replace(" ", "")
    key = key.replace("star", "*").replace("rrtsharp", "rrt#").replace("strrt*", "st-rrt*")

    table = {
        "prm*": og.PRMstar,  # PRM*
        "lazyprm*": og.LazyPRMstar,  # LazyPRM*
        "rrt*": og.RRTstar,  # RRT*
        "rrt#": og.RRTsharp,  # RRT#
        "rrtx": og.RRTXstatic,  # RRTX (static variant)
        "informedrrt*": og.InformedRRTstar,  # Informed RRT*
        "bit*": og.BITstar,  # BIT*
        "abit*": og.ABITstar,  # ABIT*
        "ait*": og.AITstar,  # AIT*
        "eit*": og.EITstar,  # EIT*
        "lbtrrt": og.LBTRRT,  # LBTRRT
        "sst": og.SST,  # Sparse Stable RRT
        "t-rrt": og.TRRT,  # T-RRT
        "spars": og.SPARS,  # SPARS
        "spars2": og.SPARStwo,  # SPARS2
        "fmt*": og.FMT,  # FMT*
        "st-rrt*": og.STRRTstar,  # ST-RRT*
    }

    # Meta-planners
    if key in ("cforest", "aps", "anytimepathshortening"):
        if key == "cforest":
            planner = og.CForest(si)
            # Add your sub-planners (examples):
            planner.addPlanner(og.RRTstar(si))
            planner.addPlanner(og.BITstar(si))
        else:
            planner = og.AnytimePathShortening(si)
            planner.addPlanner(og.BITstar(si))
            planner.addPlanner(og.InformedRRTstar(si))
        ss.setPlanner(planner)
        return planner

    # Regular planners
    if key not in table:
        raise ValueError(f"Unknown/unsupported planner '{name}'")
    PlannerCls = table[key]
    try:
        planner = PlannerCls(si)
    except Exception as e:
        raise ValueError(f"Planner '{name}' not available in this OMPL build") from e

    ss.setPlanner(planner)
    return planner


def plan_motion(
    q_start: JointConfigurationType,
    q_goal: JointConfigurationType,
    bounds: np.ndarray,
    validity_fn: Callable[[ob.State], bool],
    *,
    algorithm: Optional[str] = "RRTConnect",
    objective: Optional[ob.OptimizationObjective] = None,
    max_planning_time: float = 5.0,
    edge_resolution: float = 0.01,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Plan a collision-free joint-space path with OMPL and return both the raw and
    interpolated waypoint arrays.

    Parameters
    ----------
    q_start : JointConfigurationType
        Start joint vector (length ``n``). Must lie within ``bounds``.
    q_goal : JointConfigurationType
        Goal joint vector (length ``n``). Must lie within ``bounds``.
    bounds : np.ndarray
        Joint limits for a RealVector state space; each row is ``[low, high]``.
    validity_fn : Callable[[ob.State], bool]
        State-validity checker that returns ``True`` iff the state is collision-free.
        (Typically calls MuJoCo to check contacts.)
    algorithm : str, optional
        Planner to use. If ``None``, no planner is set (use OMPL defaults).
        Supported optimization planners include:
        ``"PRM*"`` (PRMstar), ``"LazyPRM*"`` (LazyPRMstar),
        ``"RRT*"`` (RRTstar), ``"RRT#"`` (RRTsharp), ``"RRTX"`` (RRTXstatic),
        ``"Informed RRT*"`` (InformedRRTstar), ``"BIT*"`` (BITstar),
        ``"ABIT*"`` (ABITstar), ``"AIT*"`` (AITstar), ``"EIT*"`` (EITstar),
        ``"LBTRRT"`` (LBTRRT), ``"SST"`` (SST), ``"T-RRT"`` (TRRT),
        ``"SPARS"`` (SPARS), ``"SPARS2"`` (SPARStwo), ``"FMT*"`` (FMT),
        and ``"ST-RRT*"`` (STRRTstar).
        For the optimization planners, a path-length objective in joint space is used.
    objective : ob.OptimizationObjective, optional
        Custom objective (defaults to ``PathLengthOptimizationObjective``).
        Meta-planners and star-variant planners expect objectives that obey the
        triangle inequality (e.g., path length).
    max_planning_time : float, optional
        Initial solve time budget in seconds. The routine may call ``solve`` again
        to continue improving the solution.
    edge_resolution : float, optional
        Fraction of state space extent used for discrete validity checking along
        edges (smaller is safer but slower), e.g. ``0.01`` = 1% of extent.

    Returns
    -------
    path_int : (m, n) ndarray or None
        Interpolated path waypoints after simplification (``m`` points, including
        start and goal). ``None`` if planning failed.
    waypoints : (k, n) ndarray or None
        Raw solution path waypoints before interpolation (``k`` points). ``None``
        if planning failed.

    Raises
    ------
    ValueError
        If an unsupported ``algorithm`` is requested.

    Notes
    -----
    - This builds a RealVector state space from ``bounds``.
    - Post-processing uses ``SimpleSetup.simplifySolution()`` for portability across
      OMPL Python builds, then upsamples to 100 points via ``path.interpolate(100)``.
    """
    # Set up the OMPL planning problem
    space = make_state_space(bounds)
    ss = og.SimpleSetup(space)

    # Set validity function
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validity_fn))

    # Set planner
    if algorithm is None:
        pass
    elif check_option(algorithm, "RRTConnect"):
        planner = og.RRTConnect(ss.getSpaceInformation())
        ss.setPlanner(planner)
    else:
        set_optimizing_planner(ss, algorithm, objective=objective)

    ss.getSpaceInformation().setStateValidityCheckingResolution(edge_resolution)

    # Define start/goal states
    start_state = vec_to_state(space, q_start)
    goal_state = vec_to_state(space, q_goal)
    ss.setStartAndGoalStates(start_state, goal_state, 1e-3)

    # Solve (allow a bit more time to improve)
    if ss.solve(max_planning_time):
        pass
        # ss.solve(2.0)  # continues refinement
    else:
        return None, None

    # Raw solution waypoints
    path = ss.getSolutionPath()
    nq = space.getDimension()
    waypoints = [state_to_vec(path.getState(i), nq) for i in range(path.getStateCount())]

    # Path post-processing (portable across OMPL Python builds)
    ss.simplifySolution()
    path = ss.getSolutionPath()

    # Interpolated path with 100 points
    path.interpolate(100)
    path_int = [state_to_vec(path.getState(i), nq) for i in range(path.getStateCount())]

    return np.asarray(path_int, dtype=float).reshape(-1, nq), np.asarray(waypoints, dtype=float).reshape(-1, nq)


def plan_robot_motion(
    robot: "robot",
    q_start: JointConfigurationType,
    q_goal: JointConfigurationType,
    algorithm: Optional[str] = "RRTConnect",
    max_planning_time: float = 5.0,
    clearance: float = 0.0,
    only_robot_contacts: bool = True,
    edge_resolution: float = 0.01,
    validity_fn: Optional[Callable[[ob.State], bool]] = None,
    objective: Optional[ob.OptimizationObjective] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Plan a collision-free joint-space motion for a single robot or a composite
    multi-robot object using MuJoCo for collision checks and OMPL for planning.

    The function temporarily **inflates the robot geoms' collision margins** by
    ``clearance`` during planning (for a safety buffer) and restores the original
    margins afterwards.

    Parameters
    ----------
    robot : 'robot'
        robot
        Either a single robot object exposing ``scene``, ``JointNames``,
        ``BaseName``; or a composite with ``robots`` (each having those fields).
    q_start : JointConfigurationType
        Start joint configuration (order must match ``joint_names``).
    q_goal : JointConfigurationType
        Goal joint configuration.
    algorithm : str, optional
        {'RRTConnect', 'RRTstar', 'InformedRRTstar', 'BITstar', None}, optional
        Planner to use (forwarded to ``plan_motion``).  Supportedplanners include:
        ``"PRM*"`` (PRMstar), ``"LazyPRM*"`` (LazyPRMstar),
        ``"RRT*"`` (RRTstar), ``"RRT#"`` (RRTsharp), ``"RRTX"`` (RRTXstatic),
        ``"Informed RRT*"`` (InformedRRTstar), ``"BIT*"`` (BITstar),
        ``"ABIT*"`` (ABITstar), ``"AIT*"`` (AITstar), ``"EIT*"`` (EITstar),
        ``"LBTRRT"`` (LBTRRT), ``"SST"`` (SST), ``"T-RRT"`` (TRRT),
        ``"SPARS"`` (SPARS), ``"SPARS2"`` (SPARStwo), ``"FMT*"`` (FMT),
        ``"ST-RRT*"`` (STRRTstar),
        For the optimization planners, a path-length objective in joint space is used.

    max_planning_time : float, optional
        Time budget (seconds) given to the planner.
    clearance : float, optional
        Extra safety distance [m]; implemented here by inflating robot geoms'
        MuJoCo ``margin`` during planning (default 0.0).
    only_robot_contacts : bool, optional
        If ``True`` (default), only robot contacts are considered. If ``False``,
        *any* contact invalidates the state.
    edge_resolution : float, optional
        along edges (smaller = denser checks).
    validity_fn : Callable[[ob.State], bool], optional
        Custom validity function (overrides the default MuJoCo-based checker).
    objective : ob.OptimizationObjective, optional
        Custom objective (defaults to ``PathLengthOptimizationObjective``).

    Returns
    -------
    path_int : (m, n) ndarray or None
        Interpolated path from ``plan_motion`` (or ``None`` if planning failed).
    waypoints : (k, n) ndarray or None
        Raw solution waypoints from ``plan_motion`` (or ``None`` if failed).

    Raises
    ------
    ValueError
        If joint dimensions do not match between inputs and model.
    """
    # Resolve scene/model/data + aggregate joint names and robot bases
    if hasattr(robot, "robots"):
        robots = robot.robots
        _scene = robots[0].scene
        _model = _scene.model
        _data = _scene.data
        joint_names = [name for rbt in robots for name in rbt.JointNames]
        base_names: Iterable[str] = [rbt.BaseName for rbt in robots]
    else:
        _scene = robot.scene
        _model = _scene.model
        _data = _scene.data
        joint_names = list(robot.JointNames)
        base_names = [robot.BaseName]

    # Build joint addresses and bounds (user-provided helper)
    qaddr, bounds = get_robot_joints_data(_model, joint_names)
    n = bounds.shape[0]
    if len(q_start) != n or len(q_goal) != n:
        raise ValueError(f"Expected q_start/q_goal of length {n}; got {len(q_start)} and {len(q_goal)}")

    # Collect robot bodies and geoms (subtrees of the base bodies)
    robot_bodies_ids = get_body_descendants(_model, base_names, include_self=True, dedup=True)
    robot_geoms_ids = get_geoms_of_body(_model, robot_bodies_ids)

    # Validity function:
    if validity_fn is None:
        validity_fn = make_mujoco_validity_fn(_model, _data, robot_geoms_ids, qaddr, only_robot_contacts=only_robot_contacts)

    # --- Temporarily inflate geom margins by `clearance` and plan
    # Use array API if available; otherwise fall back to per-geom setter.
    prev_margins = None
    use_array_api = hasattr(_model, "geom_margin")

    try:
        if clearance > 0.0:
            if use_array_api:
                idx = np.asarray(robot_geoms_ids, dtype=int)
                prev_margins = _model.geom_margin[idx].copy()
                _model.geom_margin[idx] = clearance
            else:
                prev_margins = {g: _scene.model.geom(g).margin for g in robot_geoms_ids}
                for g in robot_geoms_ids:
                    _scene.model.geom(g).margin = clearance

        # Call your planner (returns (path_int, waypoints))
        path = plan_motion(
            q_start,
            q_goal,
            bounds,
            validity_fn=validity_fn,
            algorithm=algorithm,
            objective=objective,
            max_planning_time=max_planning_time,
            edge_resolution=edge_resolution,
        )
    finally:
        # Restore original margins
        if prev_margins is not None:
            if use_array_api and isinstance(prev_margins, np.ndarray):
                _model.geom_margin[idx] = prev_margins
            else:
                for g, val in prev_margins.items():
                    _scene.model.geom(g).margin = float(val)

    return path
