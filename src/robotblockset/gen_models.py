"""Kinematic model generation utilities.

This module provides functions and tools to generate the kinematic models for robots using the Denavit-Hartenberg.
(DH) convention, kinematic models and gravity models using URDF robot definitions.

It includes functions to compute the DH transformation matrix, substitute sine and cosine functions
in symbolic expressions, and generate Python scripts for the robot's kinematics and Jacobian matrix and gravity model.

The models are generated symbolically using the `sympy` library, and the resulting models can be used for
forward kinematics, Jacobian computation, gravity models, and analysis of robot motion.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from yourdfpy import URDF
from scipy.spatial.transform import Rotation
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

from robotblockset.rbs_typing import HomogeneousMatrixType, RotationMatrixType, Vector3DType


def show_urdf(urdf_path: Union[str, Path]) -> None:
    """
    Lists URDF joint structure

    Parameters
    ----------
    urdf_path : Union[str, Path]
        Path to the URDF file.

    Returns
    -------
    None
        This function has no return value. It prints the URDF joint table.
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # URDF root is usually <robot>
    joints = root.findall("joint")

    print("Joint         Type                                     Name                                   Parent                                    Child")
    print("------- ---------- ---------------------------------------- ---------------------------------------- ----------------------------------------")

    for i, joint in enumerate(joints, start=1):
        jtype = joint.get("type", "")
        jname = joint.get("name", "")

        parent = joint.find("parent")
        child = joint.find("child")

        parent_link = parent.get("link", "") if parent is not None else ""
        child_link = child.get("link", "") if child is not None else ""

        # Match MATLAB formatting widths closely
        print(f"{i:5d}  {jtype:>10s} {jname:>40s} {parent_link:>40s} {child_link:>40s}")


def show_urdf_tree(urdf_path: Union[str, Path], start_link: Optional[str] = None, max_depth: Optional[int] = None) -> None:
    """
    Print a simple textual tree of a URDF model.

    This helper parses the URDF XML and prints parent/child links connected by joints.
    It does not render geometry; it only shows the kinematic structure.

    Parameters
    ----------
    urdf_path : Union[str, Path]
        Path to the URDF file.
    start_link : str, optional
        Root link name to start from. If None, the base link is inferred.
    max_depth : int, optional
        Maximum recursion depth to print. If None, prints the full tree.

    Returns
    -------
    None
        This function has no return value. It prints the URDF kinematic tree.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = [link.get("name") for link in root.findall("link")]
    joints = []
    children = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        jname = joint.get("name")
        jtype = joint.get("type")
        joints.append((parent, child, jname, jtype))
        children.add(child)

    if start_link is None:
        bases = [lnk for lnk in links if lnk not in children]
        start_link = bases[0] if bases else (links[0] if links else None)

    if start_link is None:
        print("No links found in URDF.")
        return

    adjacency: Dict[str, List[Tuple[str, str, str]]] = {}
    for parent, child, jname, jtype in joints:
        adjacency.setdefault(parent, []).append((child, jname, jtype))

    def walk(link: str, prefix: str = "", depth: int = 0) -> None:
        print(f"{prefix}{link}")
        if max_depth is not None and depth >= max_depth:
            return
        for child, jname, jtype in adjacency.get(link, []):
            print(f"{prefix}  |- {jname} [{jtype}] -> {child}")
            walk(child, prefix + "    ", depth + 1)

    walk(start_link)


def _parse_mjcf_vec(value: Optional[str], default: Tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    return np.fromstring(value, sep=" ", dtype=float)


def _mjcf_rotation(elem: ET.Element, angle_unit: str) -> RotationMatrixType:
    if elem.get("quat") is not None:
        quat = _parse_mjcf_vec(elem.get("quat"), (1.0, 0.0, 0.0, 0.0))
        if quat.shape != (4,):
            raise ValueError(f"Invalid MJCF quaternion '{elem.get('quat')}'")
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    if elem.get("euler") is not None:
        euler = _parse_mjcf_vec(elem.get("euler"), (0.0, 0.0, 0.0))
        return Rotation.from_euler("xyz", euler, degrees=angle_unit != "radian").as_matrix()
    if elem.get("axisangle") is not None:
        axisangle = _parse_mjcf_vec(elem.get("axisangle"), (1.0, 0.0, 0.0, 0.0))
        if axisangle.shape != (4,):
            raise ValueError(f"Invalid MJCF axisangle '{elem.get('axisangle')}'")
        axis = axisangle[:3]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.eye(3)
        angle = axisangle[3]
        if angle_unit != "radian":
            angle = np.deg2rad(angle)
        return Rotation.from_rotvec(axis / norm * angle).as_matrix()
    return np.eye(3)


def _mjcf_transform(elem: ET.Element, angle_unit: str) -> HomogeneousMatrixType:
    T = np.eye(4)
    T[:3, :3] = _mjcf_rotation(elem, angle_unit)
    pos = _parse_mjcf_vec(elem.get("pos"), (0.0, 0.0, 0.0))
    if pos.shape != (3,):
        raise ValueError(f"Invalid MJCF position '{elem.get('pos')}'")
    T[:3, 3] = pos
    return T


def _find_mjcf_worldbody(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF file does not contain <worldbody>")
    return worldbody


def _find_mjcf_body(worldbody: ET.Element, body_name: str) -> ET.Element:
    for body in worldbody.iter("body"):
        if body.get("name") == body_name:
            return body
    raise ValueError(f"Body '{body_name}' not found in MJCF model")


def show_mjcf(xml_path: Union[str, Path]) -> None:
    """
    List the joint structure of an MJCF model.

    Parameters
    ----------
    xml_path : Union[str, Path]
        Path to the MJCF XML file.

    Returns
    -------
    None
        This function has no return value. It prints the MJCF joint table.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    worldbody = _find_mjcf_worldbody(root)

    rows = []

    def walk(body: ET.Element, parent_name: str) -> None:
        body_name = body.get("name", "")
        joints = body.findall("joint")
        if joints:
            for i, joint in enumerate(joints, start=1):
                joint_name = joint.get("name", f"{body_name}_joint{i}")
                joint_type = joint.get("type", "hinge")
                rows.append((joint_name, joint_type, parent_name, body_name))
        else:
            rows.append((f"{parent_name}_to_{body_name}", "fixed", parent_name, body_name))
        for child in body.findall("body"):
            walk(child, body_name)

    for body in worldbody.findall("body"):
        walk(body, "world")

    print("Joint         Type                                     Name                                   Parent                                    Child")
    print("------- ---------- ---------------------------------------- ---------------------------------------- ----------------------------------------")
    for i, (jname, jtype, parent_name, child_name) in enumerate(rows, start=1):
        print(f"{i:5d}  {jtype:>10s} {jname:>40s} {parent_name:>40s} {child_name:>40s}")


def show_mjcf_tree(xml_path: Union[str, Path], start_body: Optional[str] = None, max_depth: Optional[int] = None) -> None:
    """
    Print a simple textual tree of an MJCF model.

    Parameters
    ----------
    xml_path : Union[str, Path]
        Path to the MJCF XML file.
    start_body : str, optional
        Root body to start from. If omitted, the first body under `worldbody` is used.
    max_depth : int, optional
        Maximum recursion depth to print. If omitted, prints the full tree.

    Returns
    -------
    None
        This function has no return value. It prints the MJCF body tree.
    """
    root = ET.parse(xml_path).getroot()
    worldbody = _find_mjcf_worldbody(root)

    if start_body is None:
        bodies = worldbody.findall("body")
        if not bodies:
            print("No bodies found in MJCF.")
            return
        body = bodies[0]
    else:
        body = _find_mjcf_body(worldbody, start_body)

    def walk(current: ET.Element, prefix: str = "", depth: int = 0) -> None:
        name = current.get("name", "<unnamed>")
        print(f"{prefix}{name}")
        if max_depth is not None and depth >= max_depth:
            return
        joints = current.findall("joint")
        if joints:
            for joint in joints:
                print(f"{prefix}  |- {joint.get('name', '<unnamed>')} [{joint.get('type', 'hinge')}]")
        for child in current.findall("body"):
            walk(child, prefix + "    ", depth + 1)

    if start_body is None:
        print("world")
        walk(body, "  ", 0)
    else:
        walk(body)


def _load_mjcf_adapter(xml_path: Union[str, Path]) -> SimpleNamespace:
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    worldbody = _find_mjcf_worldbody(root)
    compiler = root.find("compiler")
    angle_unit = "degree"
    if compiler is not None:
        angle_unit = compiler.get("angle", "degree")

    robot = SimpleNamespace(joint_names=[], joint_map={}, link_map={"world": SimpleNamespace(name="world", inertial=None)})

    def add_link(body_name: str, body: ET.Element) -> None:
        inertial_elem = body.find("inertial")
        inertial = None
        if inertial_elem is not None:
            inertial = SimpleNamespace(
                origin=_mjcf_transform(inertial_elem, angle_unit),
                mass=float(inertial_elem.get("mass", "0")),
            )
        robot.link_map[body_name] = SimpleNamespace(name=body_name, inertial=inertial)

    def add_joint(name: str, parent_name: str, child_name: str, origin: HomogeneousMatrixType, joint_type: str = "fixed", axis: Optional[Vector3DType] = None) -> None:
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        robot.joint_names.append(name)
        robot.joint_map[name] = SimpleNamespace(
            name=name,
            parent=parent_name,
            child=child_name,
            origin=origin,
            type=joint_type,
            axis=np.array(axis, dtype=float),
        )

    def walk(body: ET.Element, parent_name: str) -> None:
        body_name = body.get("name")
        if body_name is None:
            raise ValueError("Encountered MJCF body without a name")
        add_link(body_name, body)

        body_tf = _mjcf_transform(body, angle_unit)
        joints = body.findall("joint")
        if not joints:
            add_joint(f"{parent_name}_to_{body_name}", parent_name, body_name, body_tf, "fixed", np.array([0.0, 0.0, 1.0]))
        elif len(joints) == 1:
            joint = joints[0]
            joint_tf = np.eye(4)
            joint_tf[:3, 3] = _parse_mjcf_vec(joint.get("pos"), (0.0, 0.0, 0.0))
            add_joint(
                joint.get("name", f"{parent_name}_to_{body_name}"),
                parent_name,
                body_name,
                body_tf @ joint_tf,
                {"hinge": "revolute", "slide": "prismatic", "fixed": "fixed"}.get(joint.get("type", "hinge"), "fixed"),
                _parse_mjcf_vec(joint.get("axis"), (0.0, 0.0, 1.0)),
            )
        else:
            current_parent = parent_name
            current_origin = body_tf
            for i, joint in enumerate(joints):
                if i == len(joints) - 1:
                    child_name = body_name
                else:
                    child_name = f"{body_name}__joint{i + 1}"
                    robot.link_map[child_name] = SimpleNamespace(name=child_name, inertial=None)
                joint_tf = np.eye(4)
                joint_tf[:3, 3] = _parse_mjcf_vec(joint.get("pos"), (0.0, 0.0, 0.0))
                add_joint(
                    joint.get("name", f"{current_parent}_to_{child_name}"),
                    current_parent,
                    child_name,
                    current_origin @ joint_tf,
                    {"hinge": "revolute", "slide": "prismatic", "fixed": "fixed"}.get(joint.get("type", "hinge"), "fixed"),
                    _parse_mjcf_vec(joint.get("axis"), (0.0, 0.0, 1.0)),
                )
                current_parent = child_name
                current_origin = np.eye(4)

        for child in body.findall("body"):
            walk(child, body_name)

    for body in worldbody.findall("body"):
        walk(body, "world")

    return robot


def _infer_mjcf_terminal_link(robot_name: str, robot: SimpleNamespace) -> str:
    candidates = [
        f"{robot_name}_flange",
        f"{robot_name}_TCP",
        "flange",
        "TCP",
    ]
    for candidate in candidates:
        if candidate in robot.link_map:
            return candidate
    children = {joint.child for joint in robot.joint_map.values()}
    parents = {joint.parent for joint in robot.joint_map.values()}
    terminal = [link for link in children if link not in parents]
    if not terminal:
        raise ValueError("Could not infer final MJCF body")
    return terminal[-1]


def dh_transform(a: float, alpha: float, d: float, theta: float) -> sp.Matrix:
    """
    Compute the DH transformation matrix.

    The function generates the transformation matrix using Denavit-Hartenberg (DH) parameters.

    Parameters
    ----------
    a : float
        The link length (a) from the previous joint to the current joint.
    alpha : float
        The link twist (alpha), which is the angle between consecutive z-axes.
    d : float
        The link offset (d), which is the distance along the previous z-axis to the common normal.
    theta : float
        The joint angle (theta), which is the angle about the previous z-axis.

    Returns
    -------
    sp.Matrix
        The DH transformation matrix (4x4).
    """
    return sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
            [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def subsincos(s: str, n: int) -> str:
    """
    Substitute sin and cos functions with shortcuts in a string.

    This function replaces sine and cosine terms in a symbolic string with pre-defined shortcuts
    for efficiency in symbolic expression manipulation.

    Parameters
    ----------
    s : str
        The input string containing sin and cos expressions.
    n : int
        The number of substitutions to perform based on joint angles and DH parameters.

    Returns
    -------
    str
        The modified string with sin and cos functions replaced by shortcuts.
    """
    s = f"{s}"
    for i in range(0, n):
        s = s.replace(f"cos(q{i})", f"c{i}")
        s = s.replace(f"sin(q{i})", f"s{i}")
        s = s.replace(f"cos(alpha{i})", f"ca{i}")
        s = s.replace(f"sin(alpha{i})", f"sa{i}")
    return s


def gen_kinmodel_dh_all(number_of_joints: int, filename: Optional[str] = None) -> None:
    """
    Generate a symbolic kinematic model for a robot with revolute joints only using
    Denavit-Hartenberg parameters and write it to a Python file.

    This function constructs the forward kinematics and Jacobian of a serial robot manipulator
    using its Denavit-Hartenberg (DH) parameters. It then generates a Python function
    implementing these symbolic expressions for later numerical use.

    Parameters
    ----------
    number_of_joints : int
        Number of joints.

    filename : str, optional
        Path to the output Python file. If not specified, defaults to 'robot_models.py'.

    Returns
    -------
    None
        The function writes the symbolic kinematic model to a Python script and does not return a value.
    """

    if filename is None:
        filename = "robot_models.py"

    nj = number_of_joints
    q = sp.symbols(f"q:{nj}")
    a = sp.symbols(f"a:{nj}")
    alpha = sp.symbols(f"alpha:{nj}")
    d = sp.symbols(f"d:{nj}")

    T = sp.eye(4)
    TX = sp.MutableDenseNDimArray.zeros(4, 4, nj + 1)

    for i in range(nj):
        TX[:, :, i] = T
        T_i = dh_transform(a[i], alpha[i], d[i], q[i])
        T = T * T_i
    TX[:, :, nj] = T

    p = T[:3, 3]  # Position
    R = T[:3, :3]  # Rotation matrix

    Jp = sp.Matrix([[sp.diff(p[j], q[i]) for i in range(nj)] for j in range(3)])
    Jr = sp.zeros(3, nj)
    for i in range(nj):
        Jr[:, i] = TX[:3, 2, i]

    # J = sp.Matrix.vstack(Jp, Jr)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2024 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
        f.write("from robotblockset.transformations import map_pose \n")
        f.write("\n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def kinmodel_dh_{nj}dof(q: np.ndarray, DH: dict, tcp: np.ndarray = None, out: str = 'x')-> list:\n")
    f.write('    """\n')
    f.write(f"    Compute forward kinematics and Jacobian for the {nj} DOF robot with revolute joints.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    DH : dict\n")
    f.write("        DH parameters: 'a', 'alpha', 'd', 'theta'.\n")
    f.write("    tcp : np.ndarray\n")
    f.write("        Tool centre point (optional).\n")
    f.write("    out : string\n")
    f.write("        Output form (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    p : np.ndarray\n")
    f.write("        Position of the end effector.\n")
    f.write("    R : np.ndarray\n")
    f.write("        Rotation matrix of the end effector (3, 3).\n")
    f.write("    J : np.ndarray\n")
    f.write("        Jacobian matrix (6, nj).\n")
    f.write('    """\n\n')

    for i in range(nj):
        f.write(f"    c{i} = np.cos(q[{i}] + DH['theta'][{i}])\n")
        f.write(f"    s{i} = np.sin(q[{i}] + DH['theta'][{i}])\n")
    f.write("\n")

    for i in range(nj):
        f.write(f"    a{i} = DH['a'][{i}]\n")
    f.write("\n")

    for i in range(nj):
        f.write(f"    ca{i} = np.cos(DH['alpha'][{i}])\n")
        f.write(f"    sa{i} = np.sin(DH['alpha'][{i}])\n")
    f.write("\n")

    for i in range(nj):
        f.write(f"    d{i} = DH['d'][{i}]\n")
    f.write("\n")

    f.write("    p = np.zeros(3)\n")
    for i in range(3):
        f.write(f"    p[{i}] = {subsincos((p[i]), nj)}\n")

    f.write("    R = np.zeros((3,3))\n")
    for i in range(3):
        for j in range(3):
            f.write(f"    R[{i}, {j}] = {subsincos((R[i, j]), nj)}\n")

    f.write(f"    Jp = np.zeros((3, {nj}))\n")
    for i in range(3):
        for j in range(nj):
            f.write(f"    Jp[{i}, {j}] = {subsincos((Jp[i, j]), nj)}\n")

    f.write(f"    Jr = np.zeros((3,{nj}))\n")
    for i in range(3):
        for j in range(nj):
            f.write(f"    Jr[{i}, {j}] = {subsincos((Jr[i, j]), nj)}\n")
    f.write("\n")

    f.write("    if tcp is not None:\n")
    f.write("        tcp = np.array(tcp)\n")
    f.write("        if tcp.shape == (4, 4):\n")
    f.write("            p_tcp = tcp[:3, 3]\n")
    f.write("            R_tcp = tcp[:3, :3]\n")
    f.write("        elif tcp.shape[0] == 3:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = np.eye(3)\n")
    f.write("        elif tcp.shape[0] == 7:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(Q=tcp[3:7], out='R')\n")
    f.write("        elif tcp.shape[0] == 6:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(RPY=tcp[3:6], out='R')\n")
    f.write("        else:\n")
    f.write("            raise ValueError('kinmodel: tcp is not SE3')\n")
    f.write("        v = R @ p_tcp\n")
    f.write("        s = np.array([\n")
    f.write("            [0, -v[2], v[1]],\n")
    f.write("            [v[2], 0, -v[0]],\n")
    f.write("            [-v[1], v[0], 0]])\n")
    f.write("        p = p + R @ p_tcp\n")
    f.write("        Jp = Jp + s.T @ Jr\n")
    f.write("        R = R @ R_tcp\n")
    f.write("\n")

    f.write("    J = np.vstack((Jp, Jr))\n")
    f.write("\n")
    f.write("    if out=='pR':\n")
    f.write("        return p, R, J\n")
    f.write("    else:\n")
    f.write("        return map_pose(R=R, p=p, out=out), J\n")
    f.write("\n")
    f.close()

    print(f"Kinematic model for {nj} DOF robot with revolute joints using DH parameters has been generated in {filename}.")


def gen_kinmodel_dh(robot: Dict[str, Any], filename: Optional[str] = None) -> None:
    """
    Generate a symbolic kinematic model using Denavit-Hartenberg parameters and write it to a Python file.

    This function constructs the forward kinematics and Jacobian of a serial robot manipulator
    using its Denavit-Hartenberg (DH) parameters. It then generates a Python function
    implementing these symbolic expressions for later numerical use.

    Parameters
    ----------
    robot : Dict[str, Any]
        A dictionary defining the robot's configuration with the following keys:
            - 'nj' : int
                Number of joints.
            - 'a' : list of float
                DH 'a' parameters (link lengths).
            - 'alpha' : list of float
                DH 'alpha' parameters (link twists).
            - 'd' : list of float
                DH 'd' parameters (link offsets).
            - 'theta' : list of float, optional
                DH 'theta' parameters (joint angle offsets). If not provided, defaults to zeros.
            - 'name' : str, optional
                Name of the robot (used in function name generation).
            - 'description' : str, optional
                Description of the robot (used in docstring of the generated function).

    filename : str, optional
        Path to the output Python file. If not specified, defaults to 'robot_models.py'.

    Returns
    -------
    None
        The function writes the symbolic kinematic model to a Python script and does not return a value.
    """

    if filename is None:
        filename = "robot_models.py"

    nj = robot["nj"]
    q = sp.symbols(f"q:{nj}")
    a = [0 if val == 0 else sp.symbols(f"a{i}") for i, val in enumerate(robot["a"])]
    alpha = robot["alpha"]
    d = [0 if val == 0 else sp.symbols(f"d{i}") for i, val in enumerate(robot["d"])]
    theta = robot.get("theta", [0] * nj)

    T = sp.eye(4)
    TX = sp.MutableDenseNDimArray.zeros(4, 4, nj + 1)

    for i in range(nj):
        TX[:, :, i] = T
        T_i = dh_transform(a[i], alpha[i], d[i], q[i])
        T = T * T_i
    TX[:, :, nj] = T

    p = T[:3, 3]  # Position
    R = T[:3, :3]  # Rotation matrix

    Jp = sp.Matrix([[sp.diff(p[j], q[i]) for i in range(nj)] for j in range(3)])
    Jr = sp.zeros(3, nj)
    for i in range(nj):
        Jr[:, i] = TX[:3, 2, i]

    # J = sp.Matrix.vstack(Jp, Jr)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2024 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
        f.write("from robotblockset.transformations import map_pose \n")
        f.write("\n")
        f.write("pi = np.pi\n")
        f.write("\n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def kinmodel_{robot.get('name', 'robot')}(q: np.ndarray, tcp: np.ndarray = None, out: str = 'x')-> list:\n")
    f.write('    """\n')
    f.write(f"    Compute forward kinematics and Jacobian for the {robot.get('description', 'robot')}.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    tcp : np.ndarray\n")
    f.write("        Tool centre point (optional).\n")
    f.write("    out : string\n")
    f.write("        Output form (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    p : np.ndarray\n")
    f.write("        Position of the end effector.\n")
    f.write("    R : np.ndarray\n")
    f.write("        Rotation matrix of the end effector (3, 3).\n")
    f.write("    J : np.ndarray\n")
    f.write("        Jacobian matrix (6, nj).\n")
    f.write('    """\n\n')

    for i in range(nj):
        if theta[i] == 0:
            f.write(f"    c{i} = np.cos(q[{i}])\n")
            f.write(f"    s{i} = np.sin(q[{i}])\n")
        else:
            f.write(f"    c{i} = np.cos(q[{i}] + {theta[i]})\n")
            f.write(f"    s{i} = np.sin(q[{i}] + {theta[i]})\n")
    f.write("\n")

    for i in range(nj):
        if a[i] != 0:
            f.write(f"    a{i} = {robot['a'][i]}\n")
    f.write("\n")

    for i in range(nj):
        if d[i] != 0:
            f.write(f"    d{i} = {robot['d'][i]}\n")
    f.write("\n")

    f.write("    p = np.zeros(3)\n")
    for i in range(3):
        f.write(f"    p[{i}] = {subsincos((p[i]), nj)}\n")

    f.write("    R = np.zeros((3,3))\n")
    for i in range(3):
        for j in range(3):
            f.write(f"    R[{i}, {j}] = {subsincos((R[i, j]), nj)}\n")

    f.write(f"    Jp = np.zeros((3, {nj}))\n")
    for i in range(3):
        for j in range(nj):
            f.write(f"    Jp[{i}, {j}] = {subsincos((Jp[i, j]), nj)}\n")

    f.write(f"    Jr = np.zeros((3,{nj}))\n")
    for i in range(3):
        for j in range(nj):
            f.write(f"    Jr[{i}, {j}] = {subsincos((Jr[i, j]), nj)}\n")
    f.write("\n")

    f.write("    if tcp is not None:\n")
    f.write("        tcp = np.array(tcp)\n")
    f.write("        if tcp.shape == (4, 4):\n")
    f.write("            p_tcp = tcp[:3, 3]\n")
    f.write("            R_tcp = tcp[:3, :3]\n")
    f.write("        elif tcp.shape[0] == 3:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = np.eye(3)\n")
    f.write("        elif tcp.shape[0] == 7:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(Q=tcp[3:7], out='R')\n")
    f.write("        elif tcp.shape[0] == 6:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(RPY=tcp[3:6], out='R')\n")
    f.write("        else:\n")
    f.write("            raise ValueError('kinmodel: tcp is not SE3')\n")
    f.write("        v = R @ p_tcp\n")
    f.write("        s = np.array([\n")
    f.write("            [0, -v[2], v[1]],\n")
    f.write("            [v[2], 0, -v[0]],\n")
    f.write("            [-v[1], v[0], 0]])\n")
    f.write("        p = p + R @ p_tcp\n")
    f.write("        Jp = Jp + s.T @ Jr\n")
    f.write("        R = R @ R_tcp\n")
    f.write("\n")

    f.write("    J = np.vstack((Jp, Jr))\n")
    f.write("\n")
    f.write("    if out=='pR':\n")
    f.write("        return p, R, J\n")
    f.write("    else:\n")
    f.write("        return map_pose(R=R, p=p, out=out), J\n")
    f.write("\n")
    f.close()

    print(f"Kinematic model for '{robot.get('name', 'robot')}' using DH parameters has been generated in {filename}.")


def symbolic_origin_matrix(origin_np: HomogeneousMatrixType, pos_prefix: str = "p", pos_index: int = 0, angle_prefix: str = "a", angle_index: int = 0) -> Tuple[sp.Matrix, List[Tuple[sp.Symbol, float]], int, List[Tuple[sp.Symbol, float]], int]:
    """
    Convert a numeric 4x4 transformation matrix into a symbolic transformation matrix.

    This function analyzes the given transformation matrix to detect significant translation
    and rotation values. Nonzero translations and non-multiple-of-pi/2 rotations are
    replaced with symbolic variables. Useful for creating parameterized symbolic robot
    models from numeric transformation matrices.

    Parameters
    ----------
    origin_np : HomogeneousMatrixType
        A 4x4 homogeneous transformation matrix (rotation and translation).
    pos_prefix : str, optional
        Prefix for generated symbolic position variables. Default is "p".
    pos_index : int, optional
        Starting index for symbolic position variables. Default is 0.
    angle_prefix : str, optional
        Prefix for generated symbolic angle variables. Default is "a".
    angle_index : int, optional
        Starting index for symbolic angle variables. Default is 0.

    Returns
    -------
    T_sym : sympy.Matrix
        The 4x4 symbolic transformation matrix.
    pos_vars : list of tuple
        List of (symbol, value) tuples for the position components that were symbolized.
    pos_index : int
        Final position variable index after processing.
    angle_vars : list of tuple
        List of (symbol, value) tuples for the angle components that were symbolized.
    angle_index : int
        Final angle variable index after processing.
    """
    R_numeric = origin_np[:3, :3]
    t_numeric = origin_np[:3, 3]

    # Convert rotation matrix to rpy
    rpy = Rotation.from_matrix(R_numeric).as_euler("xyz", degrees=False)

    angle_vars = []
    pos_vars = []

    # Substitute angles
    rpy_sym = []
    for angle in rpy:
        angle_multiple = angle / sp.pi * 2
        if abs(angle) < 1e-6:
            rpy_sym.append(0)
        elif abs(angle_multiple - round(angle_multiple)) < 1e-4:
            rpy_sym.append(sp.Rational(round(angle_multiple), 2) * sp.pi)
        else:
            angle_index += 1
            sym_a = sp.Symbol(f"{angle_prefix}{angle_index}", real=True)
            rpy_sym.append(sym_a)
            angle_vars.append((sym_a, angle))

    # Substitute positions
    pos_sym = []
    for val in t_numeric:
        if abs(val) < 1e-6:
            pos_sym.append(0)
        else:
            pos_index += 1
            sym_p = sp.Symbol(f"{pos_prefix}{pos_index}", real=True)
            pos_sym.append(sym_p)
            pos_vars.append((sym_p, val))

    # Build symbolic transformation matrix
    Rx = sp.rot_axis1(-rpy_sym[0])
    Ry = sp.rot_axis2(-rpy_sym[1])
    Rz = sp.rot_axis3(-rpy_sym[2])
    R_sym = Rz * Ry * Rx

    T_sym = sp.eye(4)
    T_sym[:3, :3] = R_sym
    T_sym[:3, 3] = sp.Matrix(pos_sym)

    return T_sym, pos_vars, pos_index, angle_vars, angle_index


def joint_transform(joint_type: str, q: Union[sp.Symbol, float]) -> sp.Matrix:
    """
    Return the homogeneous transformation matrix for a single joint based on its type and parameter.

    This function generates a 4x4 symbolic transformation matrix for a joint transformation
    using either a rotation or translation along or about a principal axis, depending on
    the joint type. The transformation is symbolic when `q` is a `sympy.Symbol`.

    Parameters
    ----------
    joint_type : str
        Type of the joint. Supported types are:
            - "Rx", "-Rx" : Rotation about the X-axis
            - "Ry", "-Ry" : Rotation about the Y-axis
            - "Rz", "-Rz", "R", "-R" : Rotation about the Z-axis
            - "Px", "-Px" : Translation along the X-axis
            - "Py", "-Py" : Translation along the Y-axis
            - "Pz", "-Pz", "P", "-P" : Translation along the Z-axis
            - "F" : Fixed joint (identity transform)
    q : Union[sp.Symbol, float]
        The joint variable (angle or displacement). Typically a symbolic variable.

    Returns
    -------
    T : sp.Matrix
        A 4x4 symbolic homogeneous transformation matrix representing the joint motion.

    Raises
    ------
    ValueError
        If the joint type is not recognized.
    """
    T = sp.eye(4)
    if joint_type == "Rx":
        T[:3, :3] = sp.rot_axis1(-q)
    elif joint_type == "-Rx":
        T[:3, :3] = sp.rot_axis1(q)
    elif joint_type == "Ry":
        T[:3, :3] = sp.rot_axis2(-q)
    elif joint_type == "-Ry":
        T[:3, :3] = sp.rot_axis2(q)
    elif joint_type in ["Rz", "R"]:
        T[:3, :3] = sp.rot_axis3(-q)
    elif joint_type in ["-Rz", "-R"]:
        T[:3, :3] = sp.rot_axis3(q)
    elif joint_type == "Px":
        T[0, 3] = q
    elif joint_type == "-Px":
        T[0, 3] = -q
    elif joint_type == "Py":
        T[1, 3] = q
    elif joint_type == "-Py":
        T[1, 3] = -q
    elif joint_type in ["Pz", "P"]:
        T[2, 3] = q
    elif joint_type in ["-Pz", "-P"]:
        T[2, 3] = -q
    elif joint_type == "F":
        T = sp.eye(4)
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")
    return T


def get_joint_chain(robot: URDF, start_link: str, end_link: str) -> List[str]:
    """
    Reconstruct the sequence of joint names forming the kinematic chain from `start_link` to `end_link`.

    This function traverses the robot's kinematic tree using `yourdfpy`'s internal maps
    to determine the joint path between two links.

    Parameters
    ----------
    robot : URDF
        The robot model parsed with `yourdfpy`, containing joint and link information.
    start_link : str
        The name of the starting link in the kinematic chain.
    end_link : str
        The name of the target/end link in the kinematic chain.

    Returns
    -------
    List[str]
        A list of joint names representing the ordered kinematic path from `start_link` to `end_link`.

    Raises
    ------
    ValueError
        If the path cannot be found (e.g., links are not connected or reachable).
    """
    # Step 1: Build child -> parent link map and link -> joint map
    child_to_joint = {robot.joint_map[joint].child: robot.joint_map[joint] for joint in robot.joint_names}
    child_to_parent = {robot.joint_map[joint].child: robot.joint_map[joint].parent for joint in robot.joint_names}

    chain = []
    current_link = end_link

    while current_link != start_link:
        if current_link not in child_to_joint:
            raise ValueError(f"No joint leads to link '{current_link}'")
        joint = child_to_joint[current_link]
        chain.insert(0, joint.name)
        current_link = child_to_parent[current_link]
        if current_link is None:
            raise ValueError(f"Link '{start_link}' is not reachable from '{end_link}'")

    return chain


def infer_joint_type(joint_type: str, axis: Vector3DType) -> str:
    """
    Infer symbolic joint type string (e.g., "Rx", "Pz", "F") from URDF joint type and axis.

    Parameters
    ----------
    joint_type : str
        The type of joint as defined in the URDF. Valid values include:
        - "revolute"
        - "continuous"
        - "prismatic"
        - "fixed"
    axis : Vector3DType
        A 3-element vector indicating the axis of motion or rotation.

    Returns
    -------
    str
        A symbolic joint type used in the kinematic model:
        - "Rx", "Ry", "Rz", "-Rx", etc. for revolute/continuous joints
        - "Px", "Py", "Pz", "-Px", etc. for prismatic joints
        - "F" for fixed joints

    Raises
    ------
    ValueError
        If the joint type is unknown or the axis is unsupported.
    """
    if joint_type in ["revolute", "continuous"]:
        if np.allclose(axis, [1, 0, 0]):
            return "Rx"
        elif np.allclose(axis, [-1, 0, 0]):
            return "-Rx"
        elif np.allclose(axis, [0, 1, 0]):
            return "Ry"
        elif np.allclose(axis, [0, -1, 0]):
            return "-Ry"
        elif np.allclose(axis, [0, 0, 1]):
            return "Rz"
        elif np.allclose(axis, [0, 0, -1]):
            return "-Rz"
        else:
            raise ValueError(f"Unsupported revolute axis: {axis}")
    elif joint_type == "prismatic":
        if np.allclose(axis, [1, 0, 0]):
            return "Px"
        elif np.allclose(axis, [-1, 0, 0]):
            return "-Px"
        elif np.allclose(axis, [0, 1, 0]):
            return "Py"
        elif np.allclose(axis, [0, -1, 0]):
            return "-Py"
        elif np.allclose(axis, [0, 0, 1]):
            return "Pz"
        elif np.allclose(axis, [0, 0, -1]):
            return "-Pz"
        else:
            raise ValueError(f"Unsupported prismatic axis: {axis}")
    elif joint_type == "fixed":
        return "F"
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")


def replace_trig_expressions(expr_str: str, nq: int, na: int) -> str:
    """
    Replace standard trigonometric expressions in a string with abbreviated variable names.

    This function is typically used to simplify symbolic expressions for code generation
    by replacing `cos(qi)` with `ci`, `sin(qi)` with `si`, and similarly for angle parameters `ai`.

    Parameters
    ----------
    expr_str : str
        The input expression as a string (e.g., symbolic expression converted to string).
    nq : int
        Number of joint variables `q`. Replaces `cos(qi)` and `sin(qi)` for i = 1 to `nq`.
    na : int
        Number of angle variables `a`. Replaces `cos(ai)` and `sin(ai)` for i = 1 to `na`.

    Returns
    -------
    str
        Modified string with trigonometric expressions replaced by shorthand notation.
    """
    for i in range(1, nq + 1):
        expr_str = expr_str.replace(f"cos(q{i})", f"c{i}")
        expr_str = expr_str.replace(f"sin(q{i})", f"s{i}")
    for i in range(1, na + 1):
        expr_str = expr_str.replace(f"cos(a{i})", f"ca{i}")
        expr_str = expr_str.replace(f"sin(a{i})", f"sa{i}")
    return expr_str


def compute_forward_kinematics(robot: URDF, start_link: str, end_link: str) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, List[Tuple[sp.Symbol, float]], List[Tuple[sp.Symbol, float]]]:
    """
    Compute symbolic forward kinematics and Jacobians from `start_link` to `end_link`.

    This function traverses the joint chain between two links in a URDF model (via `yourdfpy`),
    constructs symbolic transformation matrices, and computes the end-effector position,
    rotation, and Jacobians with respect to joint variables.

    Parameters
    ----------
    robot : URDF
        A parsed URDF robot model using `yourdfpy`.
    start_link : str
        The base link of the kinematic chain.
    end_link : str
        The end-effector or target link.

    Returns
    -------
    p : sympy.Matrix (3x1)
        The symbolic position vector of the end-effector.
    R : sympy.Matrix (3x3)
        The symbolic rotation matrix of the end-effector.
    Jp : sympy.Matrix (3 x n)
        The positional Jacobian of the end-effector with respect to joint variables.
    Jr : sympy.Matrix (3 x n)
        The rotational Jacobian (joint axes) for revolute joints; zeros for prismatic.
    Q : sympy.Matrix (n x 1)
        Column vector of symbolic joint variables.
    a_vars : list of tuple
        List of (symbol, value) tuples for symbolic rotation angles introduced by `symbolic_origin_matrix`.
    p_vars : list of tuple
        List of (symbol, value) tuples for symbolic translations introduced by `symbolic_origin_matrix`.

    Raises
    ------
    ValueError
        If the joint chain cannot be found between the specified links.
    """
    joint_chain = get_joint_chain(robot, start_link, end_link)
    T = sp.eye(4)
    joint_syms = []
    joint_types = []
    joint_axes = []
    Q = []
    p_vars = []
    p_idx = 0
    a_vars = []
    a_idx = 0
    nj = 0
    for joint_name in joint_chain:
        joint = robot.joint_map[joint_name]
        joint_type = infer_joint_type(joint.type, joint.axis)
        joint_types.append(joint_type)
        T_origin, p_local, p_idx, a_local, a_idx = symbolic_origin_matrix(joint.origin, pos_index=p_idx, angle_index=a_idx)
        a_vars.extend(a_local)
        p_vars.extend(p_local)
        if joint_type != "F":
            q = sp.Symbol(f"q{nj + 1}", real=True)
            joint_syms.append(q)
            Q.append(q)
            T_joint = joint_transform(joint_type, q)
            if "R" in joint_type:
                joint_axes.append(T[:3, :3] @ T_origin[:3, :3] @ sp.Matrix(np.int64(joint.axis)))
            else:
                joint_axes.append(sp.Matrix([0, 0, 0]))
            nj += 1
        else:
            T_joint = sp.eye(4)
        T = T * T_origin * T_joint
    p = T[:3, 3]
    R = T[:3, :3]
    Q = sp.Matrix(Q)
    Jp = p.jacobian(Q)
    Jr = sp.Matrix.hstack(*[a for a in joint_axes if a.shape == (3, 1)])
    return p, R, Jp, Jr, Q, a_vars, p_vars


def gen_kinmodel_urdf(robot_name: str, urdf_path: str, description: Optional[str] = None, initial_link: Optional[str] = None, final_link: Optional[str] = None, prefix: str = "", filename: Optional[str] = None) -> None:
    """
    Generate a Python script implementing the symbolic kinematic model of a robot using a URDF file.

    This function parses the URDF model of a robot, symbolically computes forward kinematics and Jacobians,
    and writes the resulting symbolic expressions to a Python function in a script file.
    It supports optional TCP transforms and generates code ready for numerical evaluation.

    Parameters
    ----------
    robot_name : str
        The name of the robot (used in generated function name).
    urdf_path : str
        Path to the URDF file defining the robot model.
    description : str, optional
        Description of the robot to include in the generated docstring.
        Defaults to the value of `robot_name`.
    initial_link : str, optional
        The base link from which to start the kinematic chain.
        Defaults to "world".
    final_link : str, optional
        The target link to which forward kinematics are computed.
        Defaults to "<robot_name>_flange".
    prefix : str, optional
        Optional prefix for the generated function name (e.g., for namespacing).
    filename : str, optional
        File where the kinematic model will be written.
        Defaults to "robot_models.py".

    Returns
    -------
    None
        This function writes the output to a Python file and prints a success message.
    """
    if filename is None:
        filename = "robot_models.py"

    if description is None:
        description = robot_name
    if initial_link is None:
        initial_link = "world"
    if final_link is None:
        final_link = robot_name + "_flange"
    if prefix:  # Add prefix to the robot name
        robot_name = prefix + robot_name
    robot = URDF.load(urdf_path)
    p, R, Jp, Jr, Q, a_vars, p_vars = compute_forward_kinematics(robot, initial_link, final_link)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2025 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
        f.write("from robotblockset.transformations import map_pose \n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def kinmodel_{robot_name}(q: np.ndarray, tcp: np.ndarray = None, out: str = 'x')-> list:\n")
    f.write('    """\n')
    f.write(f"    Compute forward kinematics and Jacobian for the {description}.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    tcp : np.ndarray\n")
    f.write("        Tool centre point (optional).\n")
    f.write("    out : string\n")
    f.write("        Output form (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    p : np.ndarray\n")
    f.write("        Position of the end effector.\n")
    f.write("    R : np.ndarray\n")
    f.write("        Rotation matrix of the end effector (3, 3).\n")
    f.write("    J : np.ndarray\n")
    f.write("        Jacobian matrix (6, nj).\n")
    f.write('    """\n\n')

    nq = len(Q)
    na = len(a_vars)

    for i in range(1, nq + 1):
        f.write(f"    c{i} = np.cos(q[{i - 1}])\n")
        f.write(f"    s{i} = np.sin(q[{i - 1}])\n")
    f.write("\n")
    for i, (a_sym, a_val) in enumerate(a_vars):
        a_index = i + 1
        multiple = a_val / np.pi * 2
        if abs(multiple - round(multiple)) < 1e-4:
            k = int(round(multiple))
            f.write(f"    ca{a_index} = np.cos({k} * np.pi / 2)\n")
            f.write(f"    sa{a_index} = np.sin({k} * np.pi / 2)\n")
        else:
            f.write(f"    ca{a_index} = np.cos({float(a_val):.6f})\n")
            f.write(f"    sa{a_index} = np.sin({float(a_val):.6f})\n")
    for i, (p_sym, p_val) in enumerate(p_vars):
        f.write(f"    {p_sym} = {float(p_val):.6f}\n")
    f.write("\n")
    f.write("    p = np.array([\n")
    for pi in p:
        expr = replace_trig_expressions(str(pi), nq, na)
        f.write(f"        {expr},\n")
    f.write("    ])\n\n")
    f.write("    R = np.array([\n")
    for row in R.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")
    f.write("    Jp = np.array([\n")
    for row in Jp.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")
    f.write("    Jr = np.array([\n")
    for row in Jr.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")

    f.write("    if tcp is not None:\n")
    f.write("        tcp = np.array(tcp)\n")
    f.write("        if tcp.shape == (4, 4):\n")
    f.write("            p_tcp = tcp[:3, 3]\n")
    f.write("            R_tcp = tcp[:3, :3]\n")
    f.write("        elif tcp.shape[0] == 3:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = np.eye(3)\n")
    f.write("        elif tcp.shape[0] == 7:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(Q=tcp[3:7], out='R')\n")
    f.write("        elif tcp.shape[0] == 6:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(RPY=tcp[3:6], out='R')\n")
    f.write("        else:\n")
    f.write("            raise ValueError('kinmodel: tcp is not SE3')\n")
    f.write("        v = R @ p_tcp\n")
    f.write("        s = np.array([\n")
    f.write("            [0, -v[2], v[1]],\n")
    f.write("            [v[2], 0, -v[0]],\n")
    f.write("            [-v[1], v[0], 0]])\n")
    f.write("        p = p + R @ p_tcp\n")
    f.write("        Jp = Jp + s.T @ Jr\n")
    f.write("        R = R @ R_tcp\n")
    f.write("\n")

    f.write("    J = np.vstack((Jp, Jr))\n")
    f.write("\n")
    f.write("    if out=='pR':\n")
    f.write("        return p, R, J\n")
    f.write("    else:\n")
    f.write("        return map_pose(R=R, p=p, out=out), J\n")
    f.write("\n")
    f.close()

    print(f"Kinematic model for '{robot_name}' has been generated in {filename}.")


def compute_gravity_model(
    robot: URDF, start_link: str, end_link: str, include_link_masses: bool = True
) -> Tuple[sp.Matrix, List[sp.Symbol], List[Tuple[sp.Symbol, float]], List[Tuple[sp.Symbol, float]], List[Tuple[sp.Symbol, float]], List[Tuple[sp.Symbol, float]]]:  # gravity torque vector  # joint variables Q  # a_vars  # p_vars  # Lc_vars (COM positions)  # mass_vars
    """
    Compute the symbolic gravity load model of a robot from its URDF.

    This function calculates the symbolic gravitational torque vector acting on each joint
    due to the link masses and an optional external load at the end-effector.

    Parameters
    ----------
    robot : URDF
        The robot model parsed with `yourdfpy`.
    start_link : str
        The base link from which to begin the kinematic chain.
    end_link : str
        The end-effector or terminal link of the chain.
    include_link_masses : bool, optional
        If True, include the robot link masses in the gravity model.
        If False, only the external load at the end-effector is included.

    Returns
    -------
    grav : sympy.Matrix (n x 1)
        Symbolic gravity torque vector.
    Q : list of sympy.Symbol
        List of symbolic joint variables.
    a_vars : list of tuple
        List of (symbol, value) tuples for rotation angle parameters from link origins.
    p_vars : list of tuple
        List of (symbol, value) tuples for translation parameters from link origins.
    Lc_vars : list of tuple
        List of (symbol, value) tuples for center-of-mass offset positions.
    mass_vars : list of tuple
        List of (symbol, value) tuples for link masses used in the gravity model.

    Notes
    -----
    - Uses symbolic variables for link masses and COM locations if they are nonzero.
    - Adds external load torque at the end-effector using symbolic mass `mLoad` and symbolic COM `mCOM`.
    - Only links with inertial information are included in the computation.
    """

    joint_chain = get_joint_chain(robot, start_link, end_link)
    T = sp.eye(4)
    Q = []
    p_vars = []
    p_idx = 0
    a_vars = []
    a_idx = 0
    Lc_vars = []
    mass_vars = []
    nj = 0
    TL = []
    joint_inds = []

    for joint_name in joint_chain:
        joint = robot.joint_map[joint_name]
        joint_type = infer_joint_type(joint.type, joint.axis)
        T_origin, p_local, p_idx, a_local, a_idx = symbolic_origin_matrix(joint.origin, pos_index=p_idx, angle_index=a_idx)
        a_vars.extend(a_local)
        p_vars.extend(p_local)

        if joint_type != "F":
            q = sp.Symbol(f"q{nj + 1}", real=True)
            Q.append(q)
            T_joint = joint_transform(joint_type, q)
            nj += 1
            joint_inds.append(nj)
        else:
            T_joint = sp.eye(4)
            joint_inds.append(0)

        T = T * T_origin * T_joint
        TL.append(T)

    # COM transforms and gravity contribution
    grav = sp.zeros(nj, 1)
    ggrav = sp.Symbol("ggrav", real=True)
    GT = sp.Matrix([0, 0, ggrav])

    if include_link_masses:
        for i, joint_name in enumerate(joint_chain):
            link = robot.link_map[robot.joint_map[joint_name].child]
            if link.inertial is not None:
                com_np = link.inertial.origin[:3, 3]
                mass = link.inertial.mass
                com_sym = []
                for k in range(3):
                    if abs(com_np[k]) < 1e-8:
                        com_sym.append(0)
                    else:
                        sym_lc = sp.Symbol(f"Lc{len(Lc_vars) + 1}", real=True)
                        com_sym.append(sym_lc)
                        Lc_vars.append((sym_lc, com_np[k]))
                T_com = sp.eye(4)
                T_com[:3, 3] = sp.Matrix(com_sym)
                Tm = TL[i] * T_com
                Jp = Tm[:3, 3].jacobian(sp.Matrix(Q))
                if abs(mass) > 1e-8:
                    sym_mass = sp.Symbol(f"m{len(mass_vars) + 1}", real=True)
                    mass_vars.append((sym_mass, mass))
                    grav -= (sym_mass * GT.T * Jp).T

    # External load
    mL = sp.Symbol("mLoad", real=True)
    mCOM = sp.Matrix([sp.Symbol(f"mCOM{i + 1}", real=True) for i in range(3)])
    TmCOM = sp.eye(4)
    TmCOM[:3, 3] += mCOM
    Tm = TL[-1] * TmCOM
    Jp = Tm[:3, 3].jacobian(sp.Matrix(Q))
    grav -= (mL * GT.T * Jp).T
    return grav, Q, a_vars, p_vars, Lc_vars, mass_vars


def gen_gravmodel_urdf(robot_name: str, urdf_path: str, description: Optional[str] = None, initial_link: Optional[str] = None, final_link: Optional[str] = None, prefix: str = "", filename: Optional[str] = None) -> None:
    """
    Generate a Python function that computes symbolic gravity torques for a robot defined in a URDF.

    This function parses the URDF of a robot, computes symbolic gravitational torques for all joints,
    and generates a Python function that can evaluate these torques numerically given joint positions,
    load mass, and load center-of-mass.

    Parameters
    ----------
    robot_name : str
        Base name of the robot (used in function naming).
    urdf_path : str
        Path to the URDF file describing the robot.
    description : str, optional
        Descriptive text for the robot used in docstring. Defaults to `robot_name`.
    initial_link : str, optional
        Base link for the gravity model computation. Defaults to "world".
    final_link : str, optional
        End-effector link. Defaults to '<robot_name>_flange'.
    prefix : str, optional
        Optional prefix for the generated function name.
    filename : str, optional
        Output filename to write the generated Python model. Defaults to 'robot_models.py'.

    Returns
    -------
    None
        This function writes the gravity model function into a file and prints a confirmation message.
    """
    if filename is None:
        filename = "robot_models.py"

    if description is None:
        description = robot_name
    if initial_link is None:
        initial_link = "world"
    if final_link is None:
        final_link = robot_name + "_flange"
    if prefix:  # Add prefix to the robot name
        robot_name = prefix + robot_name
    robot = URDF.load(urdf_path)
    grav, Q, a_vars, p_vars, Lc_vars, mass_vars = compute_gravity_model(robot, initial_link, final_link)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2025 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def gravmodel_{robot_name}(q: np.ndarray, load: float = None, COM: np.ndarray = None)-> np.array:\n")
    f.write('    """\n')
    f.write(f"    Compute gravity torques for the {description}.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    load : float\n")
    f.write("        Load mass. (optional)\n")
    f.write("    COM : np.array\n")
    f.write("        load center-of-mass (3,) (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    g : np.ndarray\n")
    f.write("        joint gravity torques.\n")
    f.write('    """\n\n')

    f.write("    if load is None:\n")
    f.write("        load = 0.0\n")
    f.write("    if COM is None:\n")
    f.write("        COM = [0.0, 0.0, 0.0]\n")

    nq = len(Q)
    na = len(a_vars)

    f.write("    mLoad = load\n")
    f.write("    mCOM1, mCOM2, mCOM3 = COM\n")
    f.write("    ggrav = -9.81\n")
    f.write("\n")

    for i in range(1, nq + 1):
        f.write(f"    c{i} = np.cos(q[{i - 1}])\n")
        f.write(f"    s{i} = np.sin(q[{i - 1}])\n")
    f.write("\n")

    for i, (a_sym, a_val) in enumerate(a_vars):
        a_index = i + 1
        multiple = a_val / np.pi * 2
        if abs(multiple - round(multiple)) < 1e-4:
            k = int(round(multiple))
            f.write(f"    ca{a_index} = np.cos({k} * np.pi / 2)\n")
            f.write(f"    sa{a_index} = np.sin({k} * np.pi / 2)\n")
        else:
            f.write(f"    ca{a_index} = np.cos({float(a_val):.6f})\n")
            f.write(f"    sa{a_index} = np.sin({float(a_val):.6f})\n")
    f.write("\n")

    for i, (p_sym, p_val) in enumerate(p_vars):
        f.write(f"    {p_sym} = {float(p_val):.6f}\n")
    f.write("\n")

    for i, (Lc_sym, Lc_val) in enumerate(Lc_vars):
        f.write(f"    {Lc_sym} = {float(Lc_val):.6f}\n")
    f.write("\n")

    for i, (mass_sym, mass_val) in enumerate(mass_vars):
        f.write(f"    {mass_sym} = {float(mass_val):.6f}\n")
    f.write("\n")

    f.write("    g = np.zeros((%d,))\n" % nq)
    for i, gi in enumerate(grav):
        expr = replace_trig_expressions(str(gi), nq, na)
        f.write(f"    g[{i}] = {expr}\n")
    f.write("\n")

    f.write("    return g\n")

    print(f"Gravity model for '{robot_name}' has been generated in {filename}.")


def gen_kinmodel_mjcf(robot_name: str, xml_path: str, description: Optional[str] = None, initial_link: Optional[str] = None, final_link: Optional[str] = None, prefix: str = "", filename: Optional[str] = None) -> None:
    """
    Generate a Python script implementing the symbolic kinematic model of a robot using an MJCF file.

    Parameters
    ----------
    robot_name : str
        The name of the robot (used in generated function name).
    xml_path : str
        Path to the MJCF XML file defining the robot model.
    description : str, optional
        Description of the robot to include in the generated docstring.
    initial_link : str, optional
        The base link from which to start the kinematic chain. Defaults to "world".
    final_link : str, optional
        The target link to which forward kinematics are computed.
    prefix : str, optional
        Optional prefix for the generated function name.
    filename : str, optional
        File where the kinematic model will be written. Defaults to "robot_models.py".

    Returns
    -------
    None
        This function writes the kinematic model into a Python file and prints a confirmation message.
    """
    if filename is None:
        filename = "robot_models.py"

    if description is None:
        description = robot_name
    if initial_link is None:
        initial_link = "world"

    robot = _load_mjcf_adapter(xml_path)
    if final_link is None:
        final_link = _infer_mjcf_terminal_link(robot_name, robot)
    if prefix:
        robot_name = prefix + robot_name

    p, R, Jp, Jr, Q, a_vars, p_vars = compute_forward_kinematics(robot, initial_link, final_link)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2025 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
        f.write("from robotblockset.transformations import map_pose \n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def kinmodel_{robot_name}(q: np.ndarray, tcp: np.ndarray = None, out: str = 'x')-> list:\n")
    f.write('    """\n')
    f.write(f"    Compute forward kinematics and Jacobian for the {description}.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    tcp : np.ndarray\n")
    f.write("        Tool centre point (optional).\n")
    f.write("    out : string\n")
    f.write("        Output form (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    p : np.ndarray\n")
    f.write("        Position of the end effector.\n")
    f.write("    R : np.ndarray\n")
    f.write("        Rotation matrix of the end effector (3, 3).\n")
    f.write("    J : np.ndarray\n")
    f.write("        Jacobian matrix (6, nj).\n")
    f.write('    """\n\n')

    nq = len(Q)
    na = len(a_vars)

    for i in range(1, nq + 1):
        f.write(f"    c{i} = np.cos(q[{i - 1}])\n")
        f.write(f"    s{i} = np.sin(q[{i - 1}])\n")
    f.write("\n")
    for i, (_, a_val) in enumerate(a_vars):
        a_index = i + 1
        multiple = a_val / np.pi * 2
        if abs(multiple - round(multiple)) < 1e-4:
            k = int(round(multiple))
            f.write(f"    ca{a_index} = np.cos({k} * np.pi / 2)\n")
            f.write(f"    sa{a_index} = np.sin({k} * np.pi / 2)\n")
        else:
            f.write(f"    ca{a_index} = np.cos({float(a_val):.6f})\n")
            f.write(f"    sa{a_index} = np.sin({float(a_val):.6f})\n")
    for p_sym, p_val in p_vars:
        f.write(f"    {p_sym} = {float(p_val):.6f}\n")
    f.write("\n")
    f.write("    p = np.array([\n")
    for pi in p:
        f.write(f"        {replace_trig_expressions(str(pi), nq, na)},\n")
    f.write("    ])\n\n")
    f.write("    R = np.array([\n")
    for row in R.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")
    f.write("    Jp = np.array([\n")
    for row in Jp.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")
    f.write("    Jr = np.array([\n")
    for row in Jr.tolist():
        row_exprs = [replace_trig_expressions(str(e), nq, na) for e in row]
        f.write("        [" + ", ".join(row_exprs) + "],\n")
    f.write("    ])\n\n")

    f.write("    if tcp is not None:\n")
    f.write("        tcp = np.array(tcp)\n")
    f.write("        if tcp.shape == (4, 4):\n")
    f.write("            p_tcp = tcp[:3, 3]\n")
    f.write("            R_tcp = tcp[:3, :3]\n")
    f.write("        elif tcp.shape[0] == 3:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = np.eye(3)\n")
    f.write("        elif tcp.shape[0] == 7:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(Q=tcp[3:7], out='R')\n")
    f.write("        elif tcp.shape[0] == 6:\n")
    f.write("            p_tcp = tcp[:3]\n")
    f.write("            R_tcp = map_pose(RPY=tcp[3:6], out='R')\n")
    f.write("        else:\n")
    f.write("            raise ValueError('kinmodel: tcp is not SE3')\n")
    f.write("        v = R @ p_tcp\n")
    f.write("        s = np.array([\n")
    f.write("            [0, -v[2], v[1]],\n")
    f.write("            [v[2], 0, -v[0]],\n")
    f.write("            [-v[1], v[0], 0]])\n")
    f.write("        p = p + R @ p_tcp\n")
    f.write("        Jp = Jp + s.T @ Jr\n")
    f.write("        R = R @ R_tcp\n")
    f.write("\n")

    f.write("    J = np.vstack((Jp, Jr))\n")
    f.write("\n")
    f.write("    if out=='pR':\n")
    f.write("        return p, R, J\n")
    f.write("    else:\n")
    f.write("        return map_pose(R=R, p=p, out=out), J\n")
    f.write("\n")
    f.close()

    print(f"Kinematic model for '{robot_name}' from MJCF has been generated in {filename}.")


def gen_gravmodel_mjcf(robot_name: str, xml_path: str, description: Optional[str] = None, initial_link: Optional[str] = None, final_link: Optional[str] = None, prefix: str = "", filename: Optional[str] = None, load_only: bool = False) -> None:
    """
    Generate a Python function that computes symbolic gravity torques for a robot defined in an MJCF file.

    Parameters
    ----------
    robot_name : str
        Base name of the robot (used in function naming).
    xml_path : str
        Path to the MJCF XML file describing the robot.
    description : str, optional
        Descriptive text for the robot used in docstring.
    initial_link : str, optional
        Base link for the gravity model computation. Defaults to "world".
    final_link : str, optional
        End-effector link.
    prefix : str, optional
        Optional prefix for the generated function name.
    filename : str, optional
        Output filename to write the generated Python model. Defaults to "robot_models.py".
    load_only : bool, optional
        and ignores the masses of the robot links.

    Returns
    -------
    None
        This function writes the gravity model into a Python file and prints a confirmation message.
    """
    if filename is None:
        filename = "robot_models.py"

    if description is None:
        description = robot_name
    if initial_link is None:
        initial_link = "world"

    robot = _load_mjcf_adapter(xml_path)
    if final_link is None:
        final_link = _infer_mjcf_terminal_link(robot_name, robot)
    if prefix:
        robot_name = prefix + robot_name

    grav, Q, a_vars, p_vars, Lc_vars, mass_vars = compute_gravity_model(robot, initial_link, final_link, include_link_masses=not load_only)

    if not os.path.exists(filename):
        f = open(filename, "w")
        f.write('"""Robot models\n')
        f.write("\n")
        f.write("Copyright (c) 2025 by IJS Leon Zlajpah \n")
        f.write("\n")
        f.write('"""\n')
        f.write("import numpy as np \n")
    else:
        f = open(filename, "a")

    f.write("\n")
    f.write(f"def gravmodel_{robot_name}(q: np.ndarray, load: float = None, COM: np.ndarray = None)-> np.array:\n")
    f.write('    """\n')
    f.write(f"    Compute gravity torques for the {description}.\n\n")
    f.write("    \n")
    f.write("    Parameters:\n")
    f.write("    ----------\n")
    f.write("    q : np.ndarray\n")
    f.write("        Joint angles/positions.\n")
    f.write("    load : float\n")
    f.write("        Load mass. (optional)\n")
    f.write("    COM : np.array\n")
    f.write("        Load center-of-mass (3,) (optional).\n\n")
    f.write("    \n")
    f.write("    Returns:\n")
    f.write("    -------\n")
    f.write("    g : np.ndarray\n")
    f.write("        Joint gravity torques.\n")
    f.write('    """\n\n')

    f.write("    if load is None:\n")
    f.write("        load = 0.0\n")
    f.write("    if COM is None:\n")
    f.write("        COM = [0.0, 0.0, 0.0]\n")

    nq = len(Q)
    na = len(a_vars)

    f.write("    mLoad = load\n")
    f.write("    mCOM1, mCOM2, mCOM3 = COM\n")
    f.write("    ggrav = -9.81\n")
    f.write("\n")

    for i in range(1, nq + 1):
        f.write(f"    c{i} = np.cos(q[{i - 1}])\n")
        f.write(f"    s{i} = np.sin(q[{i - 1}])\n")
    f.write("\n")

    for i, (_, a_val) in enumerate(a_vars):
        a_index = i + 1
        multiple = a_val / np.pi * 2
        if abs(multiple - round(multiple)) < 1e-4:
            k = int(round(multiple))
            f.write(f"    ca{a_index} = np.cos({k} * np.pi / 2)\n")
            f.write(f"    sa{a_index} = np.sin({k} * np.pi / 2)\n")
        else:
            f.write(f"    ca{a_index} = np.cos({float(a_val):.6f})\n")
            f.write(f"    sa{a_index} = np.sin({float(a_val):.6f})\n")
    f.write("\n")

    for p_sym, p_val in p_vars:
        f.write(f"    {p_sym} = {float(p_val):.6f}\n")
    f.write("\n")

    for Lc_sym, Lc_val in Lc_vars:
        f.write(f"    {Lc_sym} = {float(Lc_val):.6f}\n")
    f.write("\n")

    for mass_sym, mass_val in mass_vars:
        f.write(f"    {mass_sym} = {float(mass_val):.6f}\n")
    f.write("\n")

    f.write("    g = np.zeros((%d,))\n" % nq)
    for i, gi in enumerate(grav):
        f.write(f"    g[{i}] = {replace_trig_expressions(str(gi), nq, na)}\n")
    f.write("\n")

    f.write("    return g\n")
    f.close()

    print(f"Gravity model for '{robot_name}' from MJCF has been generated in {filename}.")


if __name__ == "__main__":
    from robotblockset.robot_dh_parameters import panda, iiwa, lwr, ur10, ur5, ur10e, ur5e, pa10, jaco2

    models_filename = "test_robot_models.py"
    gen_kinmodel_dh(panda, filename=models_filename)
    gen_kinmodel_dh(iiwa, filename=models_filename)
    gen_kinmodel_dh(lwr, filename=models_filename)
    gen_kinmodel_dh(ur10, filename=models_filename)
    gen_kinmodel_dh(ur5, filename=models_filename)
    gen_kinmodel_dh(ur10e, filename=models_filename)
    gen_kinmodel_dh(ur5e, filename=models_filename)
    gen_kinmodel_dh(pa10, filename=models_filename)
    gen_kinmodel_dh(jaco2, filename=models_filename)

    gen_kinmodel_dh_all(7, filename=models_filename)

    # gen_kinmodel_urdf("urdf_robot", "robotblockset/urdf_models/panda/panda.urdf", description="My Sample Robot", initial_link="world", final_link="panda_link8", prefix="", filename=models_filename)

    # gen_gravmodel_urdf("urdf_robot", "robotblockset/urdf_models/panda/panda.urdf", description="My Sample Robot", initial_link="world", final_link="panda_link8", prefix="", filename=models_filename)
