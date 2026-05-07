"""Helpers for inspecting and editing MuJoCo MJCF specifications.

This module contains utility functions for traversing MJCF body trees,
querying actuator-to-joint relationships, attaching gripper specifications to
robot models, and performing attribute replacements on serialized MJCF.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from typing import Any, List, Optional
from robotblockset.tools import replace_attr_values_in_xml

try:
    import mujoco
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")


def actuators_for_joint(joint: Any) -> List[Any]:
    """
    Find all actuators in the same MJCF spec that are associated with a given joint.

    Parameters
    ----------
    joint : Any
        A joint element from an MJCF Python spec (e.g. created via mjcf.RootElement()
        or loaded with mjcf.from_xml_path / MjSpec.from_file).

    Returns
    -------
    list of mjcf.Element
        List of actuator elements (e.g. <motor>, <position>, <velocity>, <general>, ...)
        that reference this joint.
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
    Print the body tree starting from the given parent body.

    Note that parent must be mjcf Python spec objects (MjcfElement / MjBody)

    Parameters
    ----------
    parent : mujoco.MjsBody
        The parent body from which to start printing the body tree.
    level : int, optional
        The current level in the body tree (used for indentation).

    Returns
    -------
    None
    """
    if level == 0:
        print(f'Body Tree for "{parent.name}"')
    body = parent.first_body()
    while body:
        tmp = "-" + body.name
        tmp1 = "".join([j.name + "-" + j.type.name.split("_")[-1] + "," for j in body.joints])
        tmp += " (Joints: " + tmp1[:-1] + ")" if tmp1 else ""

        print("".join(["-" for i in range(level)]) + tmp)
        print_body_tree(body, level + 1)
        body = parent.next_body(body)


def print_body_tree(parent: mujoco.MjsBody, spec: mujoco.MjSpec, level: int = 0) -> None:
    """
    Print the body tree starting from the given parent body, including joint
    types and associated actuator names (if any).

    Note that `parent` must be an mjcf Python spec body object (MjBody from MjSpec).

    Parameters
    ----------
    parent : mujoco.MjsBody
        The parent body from which to start printing the body tree.
    spec : mujoco.MjSpec
        The full MJCF spec (used to search actuators).
    level : int, optional
        The current level in the body tree (used for indentation).

    Returns
    -------
    None
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


def attach_gripper_to_robot(robot_spec: mujoco.MjSpec, gripper_spec: mujoco.MjSpec, robot_site_name: str = "gripper_mount", prefix: str = "gripper-") -> Optional[Any]:
    """
    Attach a gripper spec to a robot spec at a given mount site.

    Parameters
    ----------
    robot_spec : mujoco.MjSpec
        Robot MJCF specification.
    gripper_spec : mujoco.MjSpec
        Gripper MJCF specification.
    robot_site_name : str, optional
        Name of the mounting site on the robot.
    prefix : str, optional
        Prefix applied to gripper names to avoid clashes.

    Returns
    -------
    object or None
        Attachment frame created at the mount site, or None if the site is missing.
    """
    # 1) Find the mount site on the robot
    site = robot_spec.site(robot_site_name)
    if site is None:
        print(f'Site "{robot_site_name}" not found in robot_spec.')
        return

    # 2) Attach the entire gripper spec at that site
    #    - This attaches gripper_spec.worldbody to the robot at `site`
    #    - prefix is applied to all names from the gripper to avoid clashes
    frame = robot_spec.attach(gripper_spec, site=site, prefix=prefix)

    return frame


def replace_in_mjcf_file(spec: mujoco._specs.MjSpec, old: str, new: str, substring: bool = False) -> mujoco.MjSpec:
    """
    Replace attribute values in an MJCF specification and return a new spec.

    Parameters
    ----------
    spec : mujoco.MjSpec
        Source MJCF specification to serialize and modify.
    old : str
        Attribute value to replace.
    new : str
        Replacement attribute value.
    substring : bool, optional
        If `True`, replace matching substrings inside attribute values; otherwise
        require an exact match.

    Returns
    -------
    mujoco.MjSpec
        New MuJoCo specification parsed from the updated XML text.
    """
    xml_text = spec.to_xml()
    new_xml, _nrep = replace_attr_values_in_xml(xml_text, old, new, substring=substring)
    return mujoco.MjSpec.from_string(new_xml)
