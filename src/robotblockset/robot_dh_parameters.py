"""Robot DH Parameter Definitions.

This module defines Denavit-Hartenberg (DH) parameter dictionaries for a set of common
industrial robot arms, including:

- Franka Emika Panda
- KUKA LBR iiwa and LWR
- Universal Robots UR5, UR10, UR5e, UR10e
- Mitsubishi PA10
- Kinova Jaco2

Each dictionary includes:
- 'name' : Robot identifier
- 'description' : Human-readable name
- 'nj' : Number of joints
- 'a', 'alpha', 'd', 'theta' : Lists of DH parameters, compatible with symbolic modeling tools

These definitions are useful for generating symbolic kinematic models, Jacobians, and dynamic models.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import sympy as sp


# Define the DH parameters for the Franka Emika Panda robot arm
panda = {
    "name": "panda",
    "description": "FRANKA-Emika Panda",
    "nj": 7,
    "a": [0, 0, 0.0825, -0.0825, 0, 0.088, 0],
    "alpha": [-sp.pi / 2, sp.pi / 2, sp.pi / 2, -sp.pi / 2, sp.pi / 2, sp.pi / 2, 0],
    "d": [0.333, 0, 0.316, 0, 0.384, 0, 0.107],
    "theta": [0] * 7,
}

# DH parameters for the KUKA LBR iiwa robot arm
iiwa = {
    "name": "iiwa",
    "description": "KUKA LBR iiwa",
    "nj": 7,
    "a": [0, 0, 0, 0, 0, 0, 0],
    "alpha": [-sp.pi / 2, sp.pi / 2, sp.pi / 2, -sp.pi / 2, -sp.pi / 2, sp.pi / 2, 0],
    "d": [0.36, 0, 0.42, 0, 0.4, 0, 0.126],
    "theta": [0] * 7,
}

# DH parameters for the KUKA LWR robot arm
lwr = {
    "name": "lwr",
    "description": "KUKA LWR",
    "nj": 7,
    "a": [0] * 7,
    "alpha": [sp.pi / 2, -sp.pi / 2, -sp.pi / 2, sp.pi / 2, sp.pi / 2, -sp.pi / 2, 0],
    "d": [0.31, 0, 0.4, 0, 0.39, 0, 0.078],
    "theta": [0] * 7,
}

# DH parameters for the UR5 robot arm
ur5 = {
    "name": "ur5",
    "description": "UR5",
    "nj": 6,
    "a": [0, -0.425, -0.39225, 0, 0, 0],
    "alpha": [sp.pi / 2, 0, 0, sp.pi / 2, -sp.pi / 2, 0],
    "d": [0.089159, 0, 0, 0.10915, 0.09456, 0.0823],
    "theta": [0] * 6,
}

# DH parameters for the UR10 robot arm
ur10 = {
    "name": "ur10",
    "description": "UR10",
    "nj": 6,
    "a": [0, -0.612, -0.5723, 0, 0, 0],
    "alpha": [sp.pi / 2, 0, 0, sp.pi / 2, -sp.pi / 2, 0],
    "d": [0.1273, 0, 0, 0.163941, 0.1157, 0.0922],
    "theta": [0] * 6,
}

# DH parameters for the UR5e robot arm
ur5e = {
    "name": "ur5e",
    "description": "UR5e",
    "nj": 6,
    "a": [0, -0.425, -0.3922, 0, 0, 0],
    "alpha": [sp.pi / 2, 0, 0, sp.pi / 2, -sp.pi / 2, 0],
    "d": [0.1625, 0, 0, 0.1333, 0.0997, 0.0996],
    "theta": [0] * 6,
}

# DH parameters for the UR10e robot arm
ur10e = {
    "name": "ur10e",
    "description": "UR10e",
    "nj": 6,
    "a": [0, -0.6127, -0.57155, 0, 0, 0],
    "alpha": [sp.pi / 2, 0, 0, sp.pi / 2, -sp.pi / 2, 0],
    "d": [0.1807, 0, 0, 0.17415, 0.11985, 0.11655],
    "theta": [0] * 6,
}

# DH parameters for the Mitsubishi PA10 robot arm
pa10 = {
    "name": "pa10",
    "description": "Mitsubishi PA10",
    "nj": 7,
    "a": [0] * 7,
    "alpha": [-sp.pi / 2, sp.pi / 2, -sp.pi / 2, sp.pi / 2, -sp.pi / 2, sp.pi / 2, 0],
    "d": [0.315, 0, 0.45, 0, 0.5, 0, 0.08],
    "theta": [0] * 7,
}

# DH parameters for the Kinova Jaco2 robot arm
D1, D2, D3 = 0.2755, 0.4100, 0.2073
D4, D5, D6 = 0.0741, 0.0741, 0.1600
e2 = 0.0098
sa = sp.sin(sp.pi / 6)
s2a = sp.sin(sp.pi / 3)
d4b = D3 + (sa / s2a) * D4
d5b = (sa / s2a) * D4 + (sa / s2a) * D5
d6b = (sa / s2a) * D5 + D6
jaco2 = {
    "name": "jaco2",
    "description": "Kinova Jaco2",
    "nj": 6,
    "a": [0, D2, 0, 0, 0, 0],
    "alpha": [sp.pi / 2, sp.pi, sp.pi / 2, sp.pi / 3, sp.pi / 3, sp.pi],
    "d": [D1, 0, -e2, -d4b, -d5b, -d6b],
    "theta": [0] * 6,
}
