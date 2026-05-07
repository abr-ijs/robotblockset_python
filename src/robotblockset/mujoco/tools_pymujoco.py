"""Utility helpers for working with Python MuJoCo models and data.

This module provides convenience functions for querying MuJoCo model
structure, navigating bodies and geoms, accessing keyframes and actuator
ranges, and performing environment-distance or collision checks.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Iterable, Sequence, Tuple, List, Union, SupportsInt, Optional, Dict, Any
import numpy as np
from copy import copy
import xml.etree.ElementTree as ET

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

# Constants
BODY = mujoco.mjtObj.mjOBJ_BODY
GEOM = mujoco.mjtObj.mjOBJ_GEOM
JOINT = mujoco.mjtObj.mjOBJ_JOINT
KEY = mujoco.mjtObj.mjOBJ_KEY
ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR


def save_state_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, k: int, time: Optional[float] = None, include_act: bool = True, include_mocap: bool = True) -> None:
    """
    Save the current simulator state into an existing MuJoCo keyframe slot.

    Copies ``data`` fields into the model’s keyframe arrays at index ``k``:
    simulation time, generalized position/velocity (``qpos``, ``qvel``), and
    optionally actuator activations (``act``) and mocap poses
    (``mocap_pos``, ``mocap_quat``).

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model containing the keyframe storage.
    data : mujoco.MjData
        Current simulator data whose state will be stored.
    k : int
        Keyframe index in ``[0, model.nkey)`` to overwrite.
    time : float, optional
        Timestamp to store in the keyframe. If ``None`` (default), uses
        ``data.time``.
    include_act : bool, optional
        If ``True`` (default) and the model has actuators (``model.na > 0``),
        store ``data.act`` in ``model.key_act``.
    include_mocap : bool, optional
        If ``True`` (default) and the model has mocap bodies (``model.nmocap > 0``),
        store mocap positions and orientations in ``model.key_mpos`` and
        ``model.key_mquat``.

    Raises
    ------
    IndexError
        If ``k`` is outside ``[0, model.nkey)``.

    Notes
    -----
    - The number of keyframes (``model.nkey``) is fixed by the MJCF; you cannot
      add new keyframes at runtime—only overwrite existing ones.
    - Controls (``ctrl``) are **not** part of keyframes.

    Returns
    -------
    None
        This method overwrites the selected keyframe in place.
    """
    if not (0 <= k < model.nkey):
        raise IndexError(f"keyframe index {k} out of range [0, {model.nkey})")

    nq, nv, na, nm = model.nq, model.nv, model.na, model.nmocap

    # time
    model.key_time[k] = data.time if time is None else float(time)

    # generalized position/velocity
    model.key_qpos[k * nq : (k + 1) * nq] = data.qpos
    model.key_qvel[k * nv : (k + 1) * nv] = data.qvel

    # activations (if any actuators with dynamics)
    if include_act and na > 0:
        model.key_act[k * na : (k + 1) * na] = data.act

    # mocap bodies (if any)
    if include_mocap and nm > 0:
        model.key_mpos[k * (3 * nm) : (k + 1) * (3 * nm)] = data.mocap_pos.ravel()
        model.key_mquat[k * (4 * nm) : (k + 1) * (4 * nm)] = data.mocap_quat.ravel()


def get_keyframe_qpos(model: mujoco.MjModel, name: str) -> np.ndarray:
    """
    Retrieve the ``qpos`` vector stored in a named keyframe.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model containing keyframes.
    name : str
        Keyframe name as defined in the MJCF (``<key name="...">``).

    Returns
    -------
    numpy.ndarray
        A copy of the generalized positions ``qpos`` stored in the keyframe.

    Raises
    ------
    RuntimeError
        If a keyframe with the given ``name`` is not found.
    """
    key_id = mujoco.mj_name2id(model, KEY, name)
    if key_id < 0:
        raise RuntimeError(f"Keyframe '{name}' not found")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    return data.qpos.copy()


def get_robot_joints_data(model: mujoco.MjModel, joint_names: Sequence[str]) -> Tuple[List[int], np.ndarray]:
    """
    Collect qpos addresses and joint bounds for a list of named 1-DoF joints.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.
    joint_names : Sequence[str]
        Names of joints to include (order defines the state-vector order).

    Returns
    -------
    qaddr : list of int
        Indices into `data.qpos` for each joint (length = n_joints).
    bounds : (n_joints, 2) ndarray of float
        Lower/upper bounds per joint; unlimited hinge/slide get sane defaults.

    Raises
    ------
    ValueError
        If any joint name is not found in the model.
    """
    qaddr: List[int] = []
    bounds: List[Tuple[float, float]] = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found.")
        lo, hi = model.jnt_range[jid]
        if lo == 0.0 and hi == 0.0:
            # Unlimited in XML/defaults -> fall back by joint type
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = -np.pi, np.pi
            else:  # slide (1-DoF)
                lo, hi = -1.0, 1.0
        qaddr.append(int(model.jnt_qposadr[jid]))
        bounds.append((float(lo), float(hi)))
    return qaddr, np.asarray(bounds, dtype=float)


def get_body_id(model: mujoco.MjModel, name_or_id: Union[int, str]) -> int:
    """
    Resolve a body name or id to an integer id.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str]
        Body identifier as an integer id or body name.

    Returns
    -------
    int
        Body id.

    Raises
    ------
    ValueError
        If a name is provided and not found.
    """
    if isinstance(name_or_id, int):
        return int(name_or_id)
    bid = mujoco.mj_name2id(model, BODY, name_or_id)
    if bid < 0:
        raise ValueError(f"Body '{name_or_id}' not found")
    return bid


def get_body_children(model: mujoco.MjModel, name_or_id: Union[int, str]) -> List[int]:
    """
    Direct child bodies (ids) of a given body.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str]
        Parent body identifier as an integer id or body name.

    Returns
    -------
    list of int
        Child body ids (may be empty).
    """
    bid = get_body_id(model, name_or_id)
    return np.where(model.body_parentid == bid)[0].astype(int).tolist()


def get_body_descendants(model: mujoco.MjModel, name_or_id: Union[int, str, Iterable[Union[int, str]]], include_self: bool = False, dedup: bool = True) -> List[int]:
    """
    All descendant body ids for one or more roots.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str, Iterable[Union[int, str]]]
        Root body identifier or iterable of root body identifiers.
    include_self : bool, optional
        If `True`, include each root body in the result.
    dedup : bool, optional
        If `True`, avoid duplicates if subtrees overlap.

    Returns
    -------
    list of int
        Descendant body ids (order: DFS).
    """
    # Normalize roots
    if isinstance(name_or_id, Iterable) and not isinstance(name_or_id, (str, bytes)):
        roots = [get_body_id(model, b) for b in name_or_id]
    else:
        roots = [get_body_id(model, name_or_id)]

    # Build adjacency
    children = [[] for _ in range(model.nbody)]
    for b in range(model.nbody):
        p = int(model.body_parentid[b])
        if p >= 0:
            children[p].append(b)

    out: List[int] = []
    seen: set[int] = set()
    for root in roots:
        stack = [root] if include_self else children[root].copy()
        while stack:
            cur = stack.pop()
            if dedup and cur in seen:
                continue
            out.append(cur)
            if dedup:
                seen.add(cur)
            stack.extend(children[cur])
    return out


def get_body_names(model: mujoco.MjModel, ids: Iterable[int]) -> List[str]:
    """
    Map body IDs to names, falling back to ``'bodyN'`` if unnamed.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    ids : Iterable[int]
        Body IDs to resolve.

    Returns
    -------
    list of str
        Resolved body names in the same order as ``ids``. For unnamed bodies,
        the placeholder ``'body{N}'`` is returned where ``N`` is the body ID.
    """
    names: List[str] = []
    for b in ids:
        n = mujoco.mj_id2name(model, BODY, int(b))
        names.append(n if n is not None else f"body{int(b)}")
    return names


def get_geoms_of_body(model: mujoco.MjModel, name_or_id: Union[int, str, Iterable[Union[int, str]]], dedup: bool = True) -> List[int]:
    """
    Geom ids directly attached to the given body/bodies (no descendants).

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str, Iterable[Union[int, str]]]
        One body identifier or an iterable of body identifiers.
    dedup : bool, optional
        If `True`, remove duplicates when multiple bodies are passed.

    Returns
    -------
    list of int
        Geom ids.
    """

    def _geoms_for_one(b) -> List[int]:
        bid = get_body_id(model, b)
        start = int(model.body_geomadr[bid])
        count = int(model.body_geomnum[bid])
        return list(range(start, start + count))

    if isinstance(name_or_id, Iterable) and not isinstance(name_or_id, (str, bytes)):
        out: List[int] = []
        for b in name_or_id:
            out.extend(_geoms_for_one(b))
        if dedup:
            seen: set[int] = set()
            out = [g for g in out if (g not in seen and not seen.add(g))]
        return out
    else:
        return _geoms_for_one(name_or_id)


def get_geom_names(model: mujoco.MjModel, geom_ids: Iterable[SupportsInt]) -> List[str]:
    """
    Map geom IDs to names, falling back to ``'geomN'`` if unnamed.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    geom_ids : Iterable[SupportsInt]
        Geom IDs to resolve.

    Returns
    -------
    list of str
        Resolved geom names in the same order as ``geom_ids``. For unnamed geoms,
        the placeholder ``'geom{N}'`` is returned where ``N`` is the geom ID.
    """
    out: List[str] = []
    for gid in geom_ids:
        n = mujoco.mj_id2name(model, GEOM, int(gid))
        out.append(n if n is not None else f"geom{int(gid)}")
    return out


def get_geom_id(model: mujoco.MjModel, name_or_id: Union[int, str]) -> int:
    """
    Resolve a geom name or id to an integer geom ID.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str]
        Geom identifier: an existing geom ID (int) or geom name (str).

    Returns
    -------
    int
        Geom ID.

    Raises
    ------
    ValueError
        If a geom with the given name does not exist.
    """
    if isinstance(name_or_id, int):
        return int(name_or_id)
    gid = mujoco.mj_name2id(model, GEOM, name_or_id)
    if gid < 0:
        raise ValueError(f"Geom '{name_or_id}' not found")
    return gid


def get_body_of_geom(model: mujoco.MjModel, name_or_id: Union[int, str], return_name: bool = False) -> Union[int, str]:
    """
    Get the body that owns a given geom.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    name_or_id : Union[int, str]
        Geom id or name.
    return_name : bool, optional
        If True, return the body name instead of id.

    Returns
    -------
    int or str
        Body id (default) or body name if `return_name=True`.
    """
    gid = get_geom_id(model, name_or_id)
    bid = int(model.geom_bodyid[gid])
    if return_name:
        nm = mujoco.mj_id2name(model, BODY, bid)
        return nm if nm is not None else f"body{bid}"
    return bid


def get_bodies_of_geoms(model: mujoco.MjModel, geoms: Iterable[Union[int, str]], return_names: bool = False) -> List[Union[int, str]]:
    """
    Vectorized version: get bodies for multiple geoms.

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    geoms : Iterable[Union[int, str]]
        Geom ids/names.
    return_names : bool, optional
        If True, return body names.

    Returns
    -------
    list of int or str
        Body ids (default) or names.
    """
    out: List[Union[int, str]] = []
    for g in geoms:
        out.append(get_body_of_geom(model, g, return_name=return_names))
    return out


def get_joints_under_body(model: mujoco.MjModel, parent_body: Union[int, str]) -> List[str]:
    """
    Return a list of all joint names in the subtree rooted at the given body.

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    parent_body_name_or_id : int or str
        Name or id of the root body (e.g. "Panda" or 1).

    Returns
    -------
    list of str
        Names of all joints in that subtree (excluding unnamed joints).
    """
    root_bid = get_body_id(model, parent_body)

    joint_names: List[str] = []
    joint_ids: List[int] = []

    def collect(bid: int):
        # joints attached to this body
        jadr = model.body_jntadr[bid]
        jnum = model.body_jntnum[bid]

        for jid in range(jadr, jadr + jnum):
            name = mujoco.mj_id2name(model, JOINT, jid)
            joint_ids.append(jid)
            if name:
                joint_names.append(name)
            else:
                # even unnamed joints get an id; we still record the id
                joint_names.append("")

        # recurse into children
        for child_bid in get_body_children(model, bid):
            collect(child_bid)

    collect(root_bid)
    return joint_names, joint_ids


def get_actuators_for_joints(model: mujoco.MjModel, robot_name: Optional[str] = None, joints: Optional[Union[List[str], List[int]]] = None) -> Tuple[List[str], List[int]]:
    """
    Return actuator names and IDs for the robot defined either by:
      - robot_name (body subtree), or
      - joints (list of joint names or list of joint IDs).

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    robot_name : str, optional
        Name of the root body of the robot. If provided, joints_under_body is used.
    joints : Union[List[str], List[int]], optional
        List of joint names or joint IDs.

    Returns
    -------
    actuator_names : list[str]
    actuator_ids   : list[int]

    Raises
    ------
    ValueError if neither robot_name nor joints is provided.
    """

    if joints is None and robot_name is None:
        raise ValueError("Either robot_name or joints must be provided.")

    if joints is not None:
        if len(joints) == 0:
            raise ValueError("joints list must not be empty.")

        if isinstance(joints[0], str):
            # joint names → joint IDs
            joint_ids = []
            for jn in joints:
                jid = mujoco.mj_name2id(model, JOINT, jn)
                if jid < 0:
                    raise ValueError(f"Joint '{jn}' not found in model.")
                joint_ids.append(jid)
        else:
            joint_ids = list(map(int, joints))
    else:
        # Use robot_name: extract joints from subtree
        _, joint_ids = get_joints_under_body(model, robot_name)

    # Convert to set for fast lookup
    joint_ids_set = set(joint_ids)

    actuator_names: List[str] = []
    actuator_ids: List[int] = []

    for aid in range(model.nu):
        # actuator_trnid[aid][0] → joint id used by actuator
        jid = model.actuator_trnid[aid][0]

        if jid in joint_ids_set:
            actuator_ids.append(aid)
            name = mujoco.mj_id2name(model, ACTUATOR, aid)
            actuator_names.append(name if name else "")

    return actuator_names, actuator_ids


def min_body_env_distance_x(model: mujoco.MjModel, data: mujoco.MjData, body_name_or_id: Union[int, str], distmax: Optional[float] = None, env_geom_ids: Optional[Sequence[int]] = None) -> Tuple[float, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Minimal signed distance between all geoms of one body and a set of
    'environment' geoms (defaults to all geoms not in this body).

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    data : mujoco.MjData
        mujoco.MjData
    body_name_or_id : Union[int, str]
        Body whose geoms we treat as 'robot body'.
    distmax : float, optional
        Max distance searched; defaults to 2 * model.stat.extent.
    env_geom_ids : Sequence[int], optional
        Explicit list of environment geom ids. If None, all geoms whose
        body != this body are used.

    Returns
    -------
    mindist : float
        Smallest signed distance (margin-subtracted).
    pair : (int, int) or None
        (geom_id_body, geom_id_env) that attains the minimum, or None.
    fromto : np.ndarray or None
        Segment [x1,y1,z1,x2,y2,z2] in world coordinates, or None.
    """
    # Make sure transforms, broadphase, etc. are up to date
    mujoco.mj_forward(model, data)

    if distmax is None:
        distmax = float(model.stat.extent) * 2.0

    # Body id + all geoms directly attached (use your helper)
    body_id = get_body_id(model, body_name_or_id)
    body_geom_ids = get_geoms_of_body(model, body_id)

    # Default environment: all geoms not attached to this body
    if env_geom_ids is None:
        env_geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] != body_id]

    mindist = distmax
    best_pair: Optional[Tuple[int, int]] = None
    best_fromto: Optional[np.ndarray] = None

    fromto = np.zeros(6, dtype=float)

    print(f"Computing min distance between body {body_name_or_id} geoms {body_geom_ids} and env geoms {env_geom_ids}")
    for g_body in body_geom_ids:
        for g_env in env_geom_ids:
            d = mujoco.mj_geomDistance(model, data, g_body, g_env, distmax, fromto)
            if d < mindist:
                mindist = d
                best_pair = (g_body, g_env)
                best_fromto = fromto.copy()
                print(f"New min dist: {mindist} between geoms {best_pair} ffromto {best_fromto}")

    if best_pair is None:
        return distmax, None, None

    return mindist, best_pair, best_fromto


def min_robot_env_distance_x(
    model: mujoco.MjModel, data: mujoco.MjData, robot_roots: Union[int, str, Iterable[Union[int, str]]], distmax: Optional[float] = None, exclude_bodies: Optional[Union[int, str, Iterable[Union[int, str]]]] = None, env_bodies: Optional[Union[int, str, Iterable[Union[int, str]]]] = None
) -> tuple[float, Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Compute the minimal signed distance between robot and environment geoms.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    data : mujoco.MjData
        MuJoCo runtime data associated with `model`.
    robot_roots : Union[int, str, Iterable[Union[int, str]]]
        Root body identifier, body name, or iterable of roots defining the
        robot subtree. All descendant bodies are included in the robot set.
    distmax : float, optional
        Maximum distance to search. If `None`, defaults to
        ``2 * model.stat.extent``.
    exclude_bodies : Union[int, str, Iterable[Union[int, str]]], optional
        Body identifier, body name, or iterable of bodies excluded from the
        default environment set.
    env_bodies : Union[int, str, Iterable[Union[int, str]]], optional
        Explicit body identifier, body name, or iterable of bodies that define
        the environment. When provided, this overrides the default environment
        construction based on `exclude_bodies`.

    Returns
    -------
    mindist : float
        Smallest signed distance between any robot geom and any environment
        geom.
    argmin : dict or None
        Metadata describing the minimizing pair, or `None` if no environment
        geoms are available. The dictionary contains the robot body id and
        name, together with the minimizing robot and environment geom ids.
    fromto : np.ndarray or None
        World-coordinate segment ``[x1, y1, z1, x2, y2, z2]`` returned by the
        MuJoCo distance query, or `None` when no pair is found.
    """

    mujoco.mj_forward(model, data)

    if distmax is None:
        distmax = float(model.stat.extent) * 2.0

    # -- Helper: convert name/id/iterable -> body-id set ------------------------
    def to_body_ids(x) -> set[int]:
        if x is None:
            return set()
        if isinstance(x, (int, str)):
            return {get_body_id(model, x)}
        ids = set()
        for elem in x:
            ids.add(get_body_id(model, elem))
        return ids

    excluded_body_ids = to_body_ids(exclude_bodies)
    explicit_env_body_ids = to_body_ids(env_bodies)

    # -- 1) Robot bodies (descendants of roots) ---------------------------------
    if isinstance(robot_roots, Iterable) and not isinstance(robot_roots, (str, bytes)):
        robot_body_list: list[int] = []
        for root in robot_roots:
            robot_body_list.extend(get_body_descendants(model, root, include_self=True, dedup=False))
        robot_bodies = set(robot_body_list)
    else:
        robot_bodies = set(get_body_descendants(model, robot_roots, include_self=True, dedup=True))

    # -- 2) Environment bodies ---------------------------------------------------
    if env_bodies is not None:
        # User explicitly defines environment
        env_bodies_set = explicit_env_body_ids
    else:
        # Default environment = all bodies except robot and exclude_bodies
        env_bodies_set = {b for b in range(model.nbody) if b not in robot_bodies and b not in excluded_body_ids}

    # -- 3) Environment geoms ---------------------------------------------------
    env_geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in env_bodies_set]

    if not env_geom_ids:
        return distmax, None, None

    # -- 4) Loop over robot bodies, find minimal distance ------------------------
    global_min = distmax
    global_argmin: Optional[Dict[str, Any]] = None
    global_fromto: Optional[np.ndarray] = None

    for b in robot_bodies:
        d, pair, fromto = min_body_env_distance(
            model,
            data,
            body_name_or_id=b,
            distmax=global_min,
            env_geom_ids=env_geom_ids,
        )

        if pair is None:
            continue

        if d < global_min:
            global_min = d
            geom_robot, geom_env = pair
            global_argmin = {
                "body_id": int(b),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                "geom_robot": int(geom_robot),
                "geom_env": int(geom_env),
            }
            global_fromto = fromto.copy() if fromto is not None else None

    return global_min, global_argmin, global_fromto


def min_body_env_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name_or_id: Union[int, str],
    distmax: Optional[float] = None,
    env_bodies: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    contype: Optional[int] = None,
) -> Tuple[float, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Minimal signed distance between all geoms of one body and geoms belonging
    to a set of environment bodies.

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    data : mujoco.MjData
        mujoco.MjData
    body_name_or_id : Union[int, str]
        The body whose geoms form the 'robot side' of the distance query.
    distmax : float, optional
        Max distance to search; defaults to 2 * model.stat.extent.
    env_bodies : Union[int, str, Iterable[Union[int, str]]], optional
        Bodies that define the environment. If None, all bodies except this
        body are treated as environment.
    contype : int, optional
        If None, all geom pairs are considered (pure geometric distance).
        If an int bitmask, only geom pairs that *could* collide according to
        MuJoCo's contact filtering (contype/conaffinity) and that involve this
        bit are considered.

    Returns
    -------
    mindist : float
        Smallest signed distance (margin-subtracted).
    pair : (int, int) or None
        (geom_id_body, geom_id_env) that attains the minimum, or None.
    fromto : np.ndarray or None
        Segment [x1,y1,z1,x2,y2,z2] in world coordinates, or None.
    """
    mujoco.mj_forward(model, data)

    if distmax is None:
        distmax = float(model.stat.extent) * 2.0

    # --- Robot body and its geoms ---
    body_id = get_body_id(model, body_name_or_id)
    body_geom_ids = get_geoms_of_body(model, body_id)

    # --- Environment bodies -> IDs ---
    def to_body_ids(x) -> set[int]:
        if x is None:
            return set()
        if isinstance(x, (int, str)):
            return {get_body_id(model, x)}
        ids: set[int] = set()
        for b in x:
            ids.add(get_body_id(model, b))
        return ids

    if env_bodies is None:
        env_body_ids = {b for b in range(model.nbody) if b != body_id}
    else:
        env_body_ids = to_body_ids(env_bodies)
        # Make sure we don't include the body itself as environment
        if body_id in env_body_ids:
            env_body_ids.remove(body_id)

    # --- Environment geoms ---
    env_geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in env_body_ids]

    if not env_geom_ids or not body_geom_ids:
        return distmax, None, None

    mindist = distmax
    best_pair: Optional[Tuple[int, int]] = None
    best_fromto: Optional[np.ndarray] = None

    fromto = np.zeros(6, dtype=float)

    for g_body in body_geom_ids:
        for g_env in env_geom_ids:
            # Optional contact-filtering via contype/conaffinity
            if contype is not None:
                # Require that at least one geom uses this contype bit
                if (model.geom_contype[g_body] & contype) == 0 and (model.geom_contype[g_env] & contype) == 0:
                    continue

                # Require that the pair *could* collide according to MuJoCo
                if (model.geom_contype[g_body] & model.geom_conaffinity[g_env]) == 0 and (model.geom_contype[g_env] & model.geom_conaffinity[g_body]) == 0:
                    continue

            d = mujoco.mj_geomDistance(model, data, g_body, g_env, distmax, fromto)
            if d == 0.0 and not np.all(fromto[:3] - fromto[3:] == 0.0):
                # mj_geomDistance returns 0.0 and zero segment when distance > distmax
                continue
            if d < mindist:
                mindist = d
                best_pair = (g_body, g_env)
                best_fromto = fromto.copy()
                # print(f"New min dist: {mindist} between geoms {best_pair} fromto {best_fromto} dist: {best_fromto[:3] - best_fromto[3:]}")

    if best_pair is None:
        return distmax, None, None

    return mindist, best_pair, best_fromto


def min_robot_env_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_roots: Union[int, str, Iterable[Union[int, str]]],
    distmax: Optional[float] = None,
    exclude_bodies: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    env_bodies: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    contype: Optional[int] = None,
) -> Tuple[float, Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Compute the minimal signed distance between robot and environment geoms.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    data : mujoco.MjData
        MuJoCo runtime data associated with `model`.
    robot_roots : Union[int, str, Iterable[Union[int, str]]]
        Root body identifier, body name, or iterable of roots defining the
        robot subtree. All descendant bodies are included in the robot set.
    distmax : float, optional
        Maximum distance to search. If `None`, defaults to
        ``2 * model.stat.extent``.
    exclude_bodies : Union[int, str, Iterable[Union[int, str]]], optional
        Body identifier, body name, or iterable of bodies excluded from the
        default environment set.
    env_bodies : Union[int, str, Iterable[Union[int, str]]], optional
        Explicit body identifier, body name, or iterable of bodies that define
        the environment. When provided, this overrides the default environment
        construction based on `exclude_bodies`.
    contype : int, optional
        Optional contact-type bitmask used to restrict the tested geom pairs to
        those that could collide according to MuJoCo contact filtering.

    Returns
    -------
    mindist : float
        Smallest signed distance between any robot geom and any environment
        geom.
    argmin : dict or None
        Metadata describing the minimizing pair, or `None` if no valid pair is
        found. The dictionary contains the robot body id and name, together
        with the minimizing robot and environment geom ids.
    fromto : np.ndarray or None
        Closest-point segment ``[x1, y1, z1, x2, y2, z2]`` in world coordinates,
        or `None` if no valid pair is found.

    Notes
    -----
    Environment bodies are selected as follows:

    - If `env_bodies` is provided, only those bodies are treated as the
      environment.
    - Otherwise, the environment consists of all bodies except the robot
      subtree and any bodies listed in `exclude_bodies`.

    If `contype` is provided, only geom pairs compatible with MuJoCo
    `contype`/`conaffinity` filtering are considered.
    """
    mujoco.mj_forward(model, data)

    if distmax is None:
        distmax = float(model.stat.extent) * 2.0

    # -- Helpers ---------------------------------------------------------------
    def to_body_ids(x) -> set[int]:
        if x is None:
            return set()
        if isinstance(x, (int, str)):
            return {get_body_id(model, x)}
        ids: set[int] = set()
        for b in x:
            ids.add(get_body_id(model, b))
        return ids

    excluded_body_ids = to_body_ids(exclude_bodies)
    explicit_env_body_ids = to_body_ids(env_bodies)

    # -- 1) Robot bodies (descendants of robot_roots) --------------------------
    if isinstance(robot_roots, Iterable) and not isinstance(robot_roots, (str, bytes)):
        robot_body_list: list[int] = []
        for root in robot_roots:
            robot_body_list.extend(get_body_descendants(model, root, include_self=True, dedup=False))
        robot_bodies = set(robot_body_list)
    else:
        robot_bodies = set(get_body_descendants(model, robot_roots, include_self=True, dedup=True))

    # -- 2) Environment bodies -------------------------------------------------
    if env_bodies is not None:
        env_bodies_set = explicit_env_body_ids
    else:
        env_bodies_set = {b for b in range(model.nbody) if b not in robot_bodies and b not in excluded_body_ids}

    if not env_bodies_set or not robot_bodies:
        return distmax, None, None

    # -- 3) Loop over robot bodies, find global min ----------------------------
    global_min = distmax
    global_argmin: Optional[Dict[str, Any]] = None
    global_fromto: Optional[np.ndarray] = None

    for b in robot_bodies:
        d, pair, fromto = min_body_env_distance(
            model,
            data,
            body_name_or_id=b,
            distmax=global_min,  # shrink search radius progressively
            env_bodies=env_bodies_set,
            contype=contype,
        )

        if pair is None:
            continue

        if d < global_min:
            global_min = d
            geom_robot, geom_env = pair
            global_argmin = {
                "body_id": int(b),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                "geom_robot": int(geom_robot),
                "geom_env": int(geom_env),
            }
            global_fromto = fromto.copy() if fromto is not None else None

    return global_min, global_argmin, global_fromto


def check_collisions(q: Sequence[float], model: mujoco.MjModel, data: mujoco.MjData, robot_geoms_ids: Iterable[int], qaddr: Sequence[int], clearance: float = 0.0) -> bool:
    """
    Check whether a configuration produces any contact closer than `clearance`
    involving the specified robot geoms.

    Parameters
    ----------
    q : Sequence[float]
        Joint vector; order must match `qaddr`.
    model : mujoco.MjModel
        mujoco.MjModel
    data : mujoco.MjData
        Provides a qpos template; not mutated.
    robot_geoms_ids : Iterable[int]
        Geom ids that belong to the robot.
    qaddr : Sequence[int]
        Indices in qpos for planned joints.
    clearance : float, optional
        Any contact with `dist < clearance` counts as a collision.

    Returns
    -------
    bool
        True if **in collision** (or violating clearance), False otherwise.
    """
    # Prepare a private data and template qpos
    work = copy(data)
    qpos = np.array(data.qpos, copy=True)
    qpos[list(qaddr)] = np.asarray(q, dtype=float)
    work.qpos[:] = qpos
    work.qvel[:] = 0.0

    mujoco.mj_forward(model, work)

    robot_geoms = {int(g) for g in robot_geoms_ids}
    for i in range(work.ncon):
        con = work.contact[i]
        if con.geom1 not in robot_geoms and con.geom2 not in robot_geoms:
            continue
        return True
    return False


def check_path_for_collisions(path: np.ndarray, robot: Any, clearance: float = 0.0) -> Union[np.ndarray, bool]:
    """
    Check a single configuration or an array of configurations for collisions.

    Parameters
    ----------
    path : np.ndarray
        One q vector or a stack of q vectors.
    robot : Any
        Robot handle (single or composite with `.robots`).
    clearance : float, optional
        Distance threshold.

    Returns
    -------
    bool or (m,) ndarray of bool
        Collision flags.
    """
    # Resolve model/data and joint mapping
    if hasattr(robot, "robots"):
        _model = robot.robots[0].scene.model
        _data = robot.robots[0].scene.data
        joint_names = [name for r in robot.robots for name in r.JointNames]
        qaddr, _ = get_robot_joints_data(_model, joint_names)
        bases = [r.BaseName for r in robot.robots]
    else:
        _model = robot.scene.model
        _data = robot.scene.data
        qaddr, _ = get_robot_joints_data(_model, robot.JointNames)
        bases = [robot.BaseName]

    robot_bodies_ids = get_body_descendants(_model, bases, include_self=True)
    robot_geoms_ids = get_geoms_of_body(_model, robot_bodies_ids)

    if path.ndim == 1:
        return check_collisions(path, _model, _data, robot_geoms_ids, qaddr, clearance=clearance)
    else:
        return np.array(
            [check_collisions(path[i], _model, _data, robot_geoms_ids, qaddr, clearance=clearance) for i in range(path.shape[0])],
            dtype=bool,
        )


def make_free_camera(model: mujoco.MjModel, azimuth: float = 140.0, elevation: float = -20.0, distance: Optional[float] = None, lookat: Optional[Sequence[float]] = None, fovy: Optional[float] = None) -> mujoco.MjvCamera:
    """
    Create a configured FREE camera for use with `mujoco.Renderer.update_scene`.

    Parameters
    ----------
    model : mujoco.MjModel
        mujoco.MjModel
    azimuth : float, optional
        Yaw in degrees.
    elevation : float, optional
        Pitch in degrees.
    distance : float, optional
        Camera distance; defaults to `1.5 * model.stat.extent`.
    lookat : Sequence[float], optional
        World-space point the camera looks at; defaults to `model.stat.center`.
    fovy : float, optional
        Vertical field-of-view (degrees). If given, sets `model.vis.global.fovy`.

    Returns
    -------
    mujoco.MjvCamera
        Configured camera object.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if lookat is None:
        lookat = model.stat.center
    if distance is None:
        distance = 1.5 * model.stat.extent
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    cam.lookat[:] = lookat
    if fovy is not None:
        model.vis.global_.fovy = float(fovy)  # vertical FOV in degrees
    return cam


def get_actuator_range(model: mujoco.MjModel, actuator_name: Union[int, str]) -> np.ndarray:
    """
    Retrieves the control range of a specific actuator in the MuJoCo model.

    The control range defines the minimum and maximum values that can be applied
    to the actuator during the simulation. This function supports retrieving
    the control range using either the actuator's index or its name.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model object from which to retrieve the actuator's control range.
    actuator_name : Union[int, str]
        The identifier of the actuator for which the control range is to be retrieved.
        This can be either the actuator's index (int) or name (str).

    Returns
    -------
    np.ndarray
        An array containing the control range [min, max] for the specified actuator.
    """
    return model.actuator(actuator_name).ctrlrange


def print_body_tree_model(model: mujoco.MjModel, parent: Union[int, str], level: int = 0) -> None:
    """
    Print the body tree starting from the given parent body

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model.
    parent : Union[int, str]
        Body id or body name from which to start printing the body tree.
    level : int, optional
        Current level in the body tree (used for indentation).
    """

    # Resolve parent body id
    if isinstance(parent, str):
        bid = mujoco.mj_name2id(model, BODY, parent)
        if bid < 0:
            raise ValueError(f"Body '{parent}' not found in model.")
    else:
        bid = int(parent)

    def _print_body(bid: int, level: int) -> None:
        body_name = mujoco.mj_id2name(model, BODY, bid) or f"body[{bid}]"

        # header on root call
        if level == 0:
            print(f'Body Tree for "{body_name}"')

        # collect joints attached to this body
        jadr = model.body_jntadr[bid]
        jnum = model.body_jntnum[bid]

        joint_strs = []
        for jid in range(jadr, jadr + jnum):
            jname = mujoco.mj_id2name(model, JOINT, jid) or f"joint[{jid}]"
            jtype_enum = mujoco.mjtJoint(model.jnt_type[jid])
            jtype_name = jtype_enum.name.split("_")[-1]  # e.g. mjJNT_HINGE → HINGE
            joint_strs.append(f"{jname}-{jtype_name}")

        # line for this body
        line = "-" + body_name
        if joint_strs:
            line += " (Joints: " + ",".join(joint_strs) + ")"

        indent = "-" * level
        print(indent + line)

        # recurse into children
        child_ids = np.where(model.body_parentid == bid)[0]
        child_ids = [x for x in child_ids if x != 0]
        for cbid in child_ids:
            _print_body(int(cbid), level + 1)

    _print_body(bid, level)


def print_body_tree_from_xml(xml_path: str, root_body_name: str = "Robot") -> None:
    """
    Print the body tree starting from the given root body name in an MJCF XML file.

    Parameters
    ----------
    xml_path : str
        Path to the MJCF XML file.
    root_body_name : str, optional
        Name of the root body from which to start printing.

    Returns
    -------
    None
    """
    tree = ET.parse(xml_path)
    mjcf_root = tree.getroot()

    # Find the root body element with the given name
    root_body = None
    for body in mjcf_root.iter("body"):
        if body.get("name") == root_body_name:
            root_body = body
            break

    if root_body is None:
        print(f'Body with name "{root_body_name}" not found in MJCF file.')
        return

    def _print_subtree(parent_body: ET.Element, level: int) -> None:
        # Iterate over direct child bodies
        for body in parent_body.findall("body"):
            body_name = body.get("name", "unnamed_body")

            # Collect joints on this body
            joint_strs = []
            for j in body.findall("joint"):
                jname = j.get("name", "")
                jtype = j.get("type", "hinge")  # MuJoCo default type is hinge
                if jname:
                    joint_strs.append(f"{jname}-{jtype}")
                else:
                    joint_strs.append(jtype)

            line = "-" + body_name
            if joint_strs:
                line += " (Joints: " + ",".join(joint_strs) + ")"

            print("-" * level + line)

            # Recurse into this child's children
            _print_subtree(body, level + 1)

    # Header, like your original function
    print(f'Body Tree for "{root_body_name}"')
    _print_subtree(root_body, level=0)
