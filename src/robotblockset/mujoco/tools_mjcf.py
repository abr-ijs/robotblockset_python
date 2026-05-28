"""Helpers for inspecting and editing MuJoCo MJCF specifications.

This module contains utility functions for traversing MJCF body trees,
querying actuator-to-joint relationships, attaching gripper specifications to
robot models, and performing attribute replacements on serialized MJCF.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from robotblockset.rbs_typing import ArrayLike
from robotblockset.tools import replace_attr_values_in_xml

if TYPE_CHECKING:
    from robotblockset.mujoco.scene_pymujoco import mujoco_scene

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")


def actuators_for_joint(joint: Any) -> List[Any]:
    """
    Find actuators associated with an MJCF joint.

    Parameters
    ----------
    joint : Any
        Joint element from an MJCF Python specification. The object is expected
        to provide ``root`` and ``name`` attributes, as MuJoCo MJCF spec joint
        objects do.

    Returns
    -------
    list[Any]
        Actuator elements that reference ``joint``. MuJoCo actuator tags such as
        ``motor``, ``position``, ``velocity``, and ``general`` are checked.
    """
    root = joint.root  # the spec's root element
    actuators = []

    # Known actuator element tags under <actuator>
    actuator_tags = (
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "general",
        "cylinder",
        "muscle",
        "spatial",
        "servo",
    )

    for tag in actuator_tags:
        for act in root.find_all(tag):
            # For parsed specs, act.joint is usually a reference to the joint element.
            # In older/edge cases it may be a string name; handle both.
            ref = getattr(act, "joint", None)
            if ref is joint:
                actuators.append(act)
            elif isinstance(ref, str) and ref == joint.name:
                actuators.append(act)

    return actuators


def print_body_tree_simple(parent: mujoco.MjsBody, level: int = 0) -> None:
    """
    Print a compact MJCF body tree.

    Parameters
    ----------
    parent : mujoco.MjsBody
        MJCF body at which tree traversal starts.
    level : int, optional
        Current indentation level. This is used internally during recursion.

    Returns
    -------
    None
        The tree is written to standard output.

    Notes
    -----
    This helper prints body names and joint type suffixes only. Use
    :func:`print_body_tree` when actuator names should also be included.
    """
    if level == 0:
        print(f'Body Tree for "{parent.name}"')
    body = parent.first_body()
    while body:
        tmp = "-" + body.name
        tmp1 = "".join([j.name + "-" + j.type.name.split("_")[-1] + "," for j in body.joints])
        tmp += " (Joints: " + tmp1[:-1] + ")" if tmp1 else ""

        print("".join(["-" for i in range(level)]) + tmp)
        print_body_tree_simple(body, level + 1)
        body = parent.next_body(body)


def print_body_tree(parent: mujoco.MjsBody, spec: mujoco.MjSpec, level: int = 0) -> None:
    """
    Print an MJCF body tree with joint and actuator information.

    Parameters
    ----------
    parent : mujoco.MjsBody
        MJCF body at which tree traversal starts.
    spec : mujoco.MjSpec
        MJCF specification that owns ``parent``. Its actuators are searched to
        identify the actuator targeting each joint.
    level : int, optional
        Current indentation level. This is used internally during recursion.

    Returns
    -------
    None
        The tree is written to standard output.
    """
    if level == 0:
        print(f'Body Tree for "{parent.name}"')

    body = parent.first_body()
    while body:
        tmp = "-" + body.name

        joint_parts = []
        for j in body.joints:
            # Joint type as suffix (e.g. mjJNT_HINGE -> HINGE)
            jtype = j.type.name.split("_")[-1] if hasattr(j.type, "name") else str(j.type)

            # Find first actuator that targets this joint
            act_name: Optional[str] = next((a.name for a in spec.actuators if a.target == j.name), None)

            if act_name:
                joint_parts.append(f"{j.name}-{jtype}[Actuator: {act_name}]")
            else:
                joint_parts.append(f"{j.name}-{jtype}")

        if joint_parts:
            tmp += " (Joints: " + ",".join(joint_parts) + ")"

        print("-" * level + tmp)

        # Recurse into children
        print_body_tree(body, spec, level + 1)
        body = parent.next_body(body)


def attach_gripper_to_robot(
    robot_spec: mujoco.MjSpec,
    gripper_spec: mujoco.MjSpec,
    robot_site_name: str = "gripper_mount",
    prefix: str = "gripper_",
) -> Optional[Any]:
    """
    Attach a gripper MJCF specification to a robot mount site.

    Parameters
    ----------
    robot_spec : mujoco.MjSpec
        Robot MJCF specification to modify.
    gripper_spec : mujoco.MjSpec
        Gripper MJCF specification to attach.
    robot_site_name : str, optional
        Name of the site in ``robot_spec`` used as the attachment point.
    prefix : str, optional
        Prefix applied by MuJoCo to names imported from ``gripper_spec``.

    Returns
    -------
    Any or None
        Attachment frame returned by :meth:`mujoco.MjSpec.attach`, or ``None``
        if ``robot_site_name`` is not found.

    Notes
    -----
    ``robot_spec`` is modified in place by MuJoCo's attach operation.
    """
    # 1) Find the mount site on the robot
    site = robot_spec.site(robot_site_name)
    if site is None:
        print(f'Site "{robot_site_name}" not found in robot_spec.')
        return None

    # 2) Attach the entire gripper spec at that site
    #    - This attaches gripper_spec.worldbody to the robot at `site`
    #    - prefix is applied to all names from the gripper to avoid clashes
    frame = robot_spec.attach(gripper_spec, site=site, prefix=prefix)

    return frame


def replace_in_mjcf_file(spec: mujoco.MjSpec, old: str, new: str, substring: bool = False) -> mujoco.MjSpec:
    """
    Replace XML attribute values in an MJCF specification.

    Parameters
    ----------
    spec : mujoco.MjSpec
        Source MJCF specification to serialize before replacement.
    old : str
        Attribute value to search for.
    new : str
        Replacement attribute value.
    substring : bool, optional
        If ``True``, replace matching substrings inside attribute values. If
        ``False``, replace only exact attribute-value matches.

    Returns
    -------
    mujoco.MjSpec
        New MJCF specification parsed from the updated XML text.
    """
    xml_text = spec.to_xml()
    new_xml, _nrep = replace_attr_values_in_xml(xml_text, old, new, substring=substring)
    return mujoco.MjSpec.from_string(new_xml)


def add_polyline_to_scene(
    scene: "mujoco_scene",
    name: str,
    points: ArrayLike,
    radius: float = 0.004,
    rgba: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    group: int = 0,
    contype: int = 0,
    conaffinity: int = 0,
) -> "mujoco_scene":
    """
    Add a polyline to a MuJoCo scene as capsule geoms.

    The scene must have been loaded from XML so that ``scene.spec`` is
    available. The function copies the current specification, appends a named
    body containing one unnamed capsule geom per segment, recompiles the model,
    and replaces ``scene.model`` and ``scene.data``.

    Parameters
    ----------
    scene : mujoco_scene
        Scene object whose ``spec``, ``model``, and ``data`` attributes will be
        updated.
    name : str
        Name of the body that contains the generated capsule geoms.
    points : ArrayLike
        Polyline vertices with shape ``(N, 3)``. At least two points are
        required.
    radius : float, optional
        Capsule radius.
    rgba : tuple[float, float, float, float], optional
        RGBA color applied to each capsule.
    group : int, optional
        MuJoCo geom visualization group.
    contype : int, optional
        MuJoCo collision type bitmask.
    conaffinity : int, optional
        MuJoCo collision affinity bitmask.

    Returns
    -------
    mujoco_scene
        The same scene object with updated specification, model, and data.

    Raises
    ------
    ValueError
        If ``scene.spec`` is unavailable, if ``name`` is empty, or if
        ``points`` is not an ``(N, 3)`` array with ``N >= 2``.
    """
    if scene.spec is None:
        raise ValueError("Scene must be loaded from XML so scene.spec exists.")
    if not name:
        raise ValueError("name must be a non-empty body name.")

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError("points must be an (N,3) array with N >= 2")

    spec = scene.spec.copy()
    worldbody = spec.worldbody
    body = worldbody.add_body(name=name, group=group)

    for i in range(len(pts) - 1):
        p0 = pts[i].tolist()
        p1 = pts[i + 1].tolist()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=p0 + p1,
            size=[radius],
            rgba=list(rgba),
            contype=contype,
            conaffinity=conaffinity,
            group=group,
        )

    # Compile the modified spec back into a MuJoCo model
    new_model = spec.compile()

    # Replace the scene model/data
    scene.spec = spec
    scene.model = new_model
    scene.data = mujoco.MjData(new_model)
    mujoco.mj_forward(new_model, scene.data)

    return scene
