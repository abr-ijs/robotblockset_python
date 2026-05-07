# RobotBlockset for Python (RBS)

[TOC]



## RBS


### Introduction

RobotBlockset (RBS) is a Python robotics framework for modelling, motion generation, trajectory execution, simulation, and deployment on real robotic systems. It provides a unified programming interface that supports the full workflow from robot and task definition to planning, testing, and execution, reducing the effort required to move algorithms from development environments to practical applications.

The framework combines platform-independent robotics functionality with backend interfaces to simulators, middleware, and hardware-specific systems. This structure makes RBS suitable for rapid prototyping, research, and industrial use, especially in applications that require consistent behavior across simulation and real-world execution.

### Software Architecture

The software architecture of RBS closely follows the layered design of the original RBS framework. At its core, the architecture is divided into three conceptual layers, which together ensure modularity, extensibility, and platform independence.

The top layer, referred to as the core submodules, contains the essential robotics functionality. This includes robot modelling, kinematics, transformation handling, motion generation, and trajectory computation. These components are implemented in a platform-independent manner and define the fundamental capabilities of the framework.

Below the core layer, middleware interfaces provide communication mechanisms that connect RBS to external systems. These interfaces enable interaction with robotics frameworks such as ROS and ROS2, as well as custom communication protocols and real-time systems. By isolating communication details within this layer, the framework ensures that higher-level application logic remains independent of the underlying infrastructure.

The bottom layer consists of target platforms, also referred to as backends, which connect the framework either to simulation environments or to physical robotic systems. This layered design enables the same high-level code to operate across different execution environments with minimal modification, which is a key advantage of the RBS approach.

### Object-Oriented Design

RBS is built entirely on an object-oriented design, which reflects both modern software engineering practices and the structure of robotic systems. The framework models robots, tools, sensors, trajectories, and environments as classes, each encapsulating both data and behaviour.

At the centre of this design is a generic robot class that defines common functionalities such as motion execution, kinematic computations, and trajectory handling. Specific robot implementations extend this base class by incorporating hardware-specific or simulation-specific features. This inheritance-based structure allows new robotic platforms to be integrated into the framework with relatively little effort, while maintaining consistency across different robot types.

The object-oriented approach also facilitates code reuse and modularity. Complex robotic applications can be constructed by combining and extending existing classes, which significantly accelerates development and experimentation.

### Backends and System Integration

One of the distinguishing features of RBS is its ability to interface seamlessly with both simulation environments and real robotic systems. This is achieved through a flexible backend architecture that abstracts the underlying execution platform.

For simulation purposes, RBS provides interfaces to several widely used physics engines. These include MuJoCo, which is particularly well suited for modelling contact dynamics and force interactions, as well as CoppeliaSim and Genesis, which offer comprehensive environments for multi-robot simulation, sensor integration, and visualization. Each of these simulators provides different advantages, and the framework allows the user to select the most appropriate one depending on the application.

In addition to simulation, RBS supports execution on real robotic platforms. This is accomplished through interfaces to manufacturer-specific APIs, as well as through middleware systems such as ROS and ROS2. These interfaces enable communication via topics, services, and action protocols, allowing the framework to operate within distributed robotic systems.

A notable advantage of this design is that the same high-level commands used in simulation can be applied to real robots with minimal changes. This greatly simplifies the transition from development to deployment and reduces the risk of inconsistencies between simulated and real-world behaviour.

### Spatial Variables and Transformations

Accurate representation of spatial relationships is fundamental in robotics, and RBS provides comprehensive support for handling spatial variables and transformations. The framework includes standard representations such as vectors, rotation matrices, homogeneous transformation matrices, Euler angles, and quaternions.

Homogeneous transformation matrices play a central role, as they provide a unified way to represent both position and orientation. These transformations are used extensively for describing robot poses, defining coordinate frames, and computing relationships between different parts of a robotic system.

The framework includes a wide range of utilities for manipulating these transformations, including composition, inversion, and conversion between different representations. In addition, interpolation methods are provided to enable smooth transitions between poses. For orientation interpolation, spherical linear interpolation is used to ensure continuous and physically meaningful rotational motion.

### Path Generation

Path generation in RBS focuses on the geometric aspect of motion, independent of time. The framework supports the generation of paths in both joint space and Cartesian space, allowing flexibility depending on the task requirements.

Geometric paths can be constructed using a variety of methods, including linear interpolation, circular motion, spline-based approaches, and radial basis function interpolation. When operating in Cartesian space, both position and orientation must be considered. Position is typically interpolated linearly, while orientation is interpolated using methods such as spherical linear interpolation to ensure smooth rotational behaviour.

This separation between geometric path generation and temporal trajectory generation allows greater flexibility, as the same path can later be executed with different timing constraints.

### Trajectory Generation

Trajectory generation extends the concept of paths by introducing time as an explicit parameter. In RBS, trajectories are generated by assigning time-dependent profiles to previously defined paths, while respecting constraints on velocity, acceleration, and higher-order derivatives.

The framework supports several trajectory generation methods, including polynomial interpolation and spline-based approaches. These methods allow the generation of smooth motion profiles suitable for both simulation and execution on real robotic systems.

An important feature of RBS is the ability to monitor trajectory execution in real time. This is achieved through callback functions that can be defined by the user. Such callbacks enable continuous observation of system variables, including forces at the end-effector, and can be used to implement adaptive behaviours. This functionality is particularly important for tasks involving physical interaction, such as grasping, surface following, or human-robot collaboration.

### Collision-Free Path Planning

For applications operating in complex environments, RBS provides support for collision-free path planning. This functionality is achieved through integration with motion planning libraries such as OMPL, combined with collision checking performed in simulation environments.

The planning process typically involves generating candidate paths in the robot's configuration space and verifying their validity with respect to environmental constraints. Collision detection mechanisms ensure that the robot avoids both self-collisions and collisions with external objects. The resulting paths can then be refined and converted into executable trajectories.

This integrated approach allows the framework to support the complete workflow from path planning to execution, ensuring consistency between planning and physical behaviour.

### Time-Optimal Trajectories

While path planning determines a feasible geometric route for the robot, trajectory generation defines how this path is executed over time. In many applications, especially in industrial robotics, it is not sufficient for a trajectory to be merely feasible; it must also be efficient. Minimizing execution time while respecting physical and dynamic constraints is therefore a central objective, and RBS addresses this through time-optimal trajectory generation.

Given a predefined path and a set of dynamic constraints, the framework computes a trajectory that minimizes execution time without violating limits on velocity, acceleration, or actuator capabilities. This process is essential for optimizing cycle times in repetitive tasks and improving overall system efficiency.

The Python implementation provides additional advantages in this context due to its support for multi-threaded execution. This makes it possible to coordinate multiple robots or tasks concurrently while ensuring proper synchronization and avoiding conflicts, such as simultaneous commands being sent to the same robot.

### Multi-Robot Systems

The RBS provides comprehensive capabilities for multi-robot systems enabling users to control multiple robots in various coordination modes. This includes independent operation with asynchronous motion, synchronized multi-robot systems, and specialized bimanual coordination for dual-arm tasks.

RBS allows each robot in a multi-robot setup to be controlled independently, maintaining its own controller, state, and task definitions. Users can issue commands to individual robots in any desired temporal order. A key feature is support for asynchronous motion: robot movements can be executed in separate threads, allowing one robot to start moving while the program immediately continues to command others. This enables parallel execution of motions, with explicit synchronization (e.g., via thread joins) when needed. For example, one robot can perform a Cartesian move while another executes a joint-space motion simultaneously, improving efficiency in scenarios requiring concurrent actions without blocking the main program flow.

For coordinated operations, RBS offers the multi-robot class, which wraps multiple robot instances into a single combined object. This system exposes a concatenated joint vector and per-robot task-space states, while preserving individual robot attributes like bases, tool-center points (TCPs), grippers, and sensors. Commands are synchronized, ensuring all robots move as part of one unified trajectory. Users can execute joint-space or task-space motions across all robots with a single high-level command, such as moving to home configurations or reaching specific poses. The system supports trajectory recording and visualization, allowing analysis of synchronized behaviors. State management is handled carefully, with options to refresh combined states after independent movements and align commanded targets.

RBS specializes in bimanual robots through special classes, designed for dual-arm manipulation where robots grasp, move, and manipulate objects collaboratively. Unlike independent or multi-robot setups, bimanual systems define tasks using coordinated variables: an absolute task describing the global motion of the pair (e.g., the pose of the first robot) and a relative task describing the positioning of one end-effector relative to the other. This enables precise control over cooperative actions, such as maintaining a fixed distance or orientation between end-effectors while moving the pair. The system supports functional redundancy, allowing selective task constraints (e.g., prioritizing relative pose over absolute pose), and path planning for complex trajectories. Motion execution includes recording and plotting of task evolutions, facilitating analysis of coordinated behaviors.

These capabilities make RBS suitable for applications ranging from simple parallel operations to advanced collaborative robotics, with asynchronous motion enhancing flexibility in independent control and synchronized systems ensuring precise coordination. The framework integrates seamlessly with simulation environments like MuJoCo, supporting visualization and real-time adjustments.


## Installation

RBS is a normal Python package. You can install it either from a release wheel or directly from this repository.

Recommended Python version: **Python 3.10+**.

### Base installation

From a downloaded release wheel from [repo.ijs.si](https://repo.ijs.si/leon/robotblockset_python/-/releases):

```bash
pip install <downloaded-wheel>.whl
```

From this repository:

```bash
pip install .
```

The base package installs the dependencies used by the core tutorials and utilities:

- `numpy>=1.24`
- `quaternionic>=1.0.12`
- `matplotlib>=3.7.5`
- `scipy`
- `sympy`
- `pyyaml`
-   `mujoco`
- `mediapy`
- `ipython`

This is sufficient for the pure Python parts of RBS and most kinematics and transformation utilities, and for most of the tutorials.

### Optional installs

Optional dependency groups are defined in `pyproject.toml` and can be installed with pip extras like, e.q.:

```bash
pip install "robotblockset[models]"
```

When installing from this repository, use:

```bash
pip install ".[models]"
```

Available extras:

- `models`: kinematic model generation support (`yourdfpy`)
- `docs`: Sphinx API documentation build dependencies
- `franka`: Franka Panda / FR3 support through `panda-python`
- `ur`: Universal Robots RTDE support
- `robotiq`: Robotiq gripper support
- `path`: OMPL support for collision-free path planning
- `cameras`: camera and calibration dependencies
- `genesis`: Genesis simulation backend
- `coppelia`: CoppeliaSim remote API backend

Multiple extras can be installed at once:

```bash
pip install ".[models,path,cameras]"
```

To install all optional Python dependencies se:

```bash
pip install ".[models,docs,franka,ur,robotiq,path,cameras,genesis,coppelia]"
```

### Optional packages by backend

Install only the packages needed for the workflows you use.

#### MuJoCo

Official website and documentation:

- https://mujoco.org/

Required when MuJoCo is used as backend as in most of the general tutorials  and parts of the camera calibration tutorials:

```bash
pip install mujoco mediapy
```

For collision-free planning with OMPL:

```bash
pip install ompl
```

RBS uses the official `mujoco` Python package and also supports `simmujoco`, an extended build of MuJoCo `simulate` with a socket interface for external control. Build and usage instructions are in `robotblockset/mujoco/simmujoco/README.md`.

##### MJCF models

RBS provides a set of ready-to-use MJCF robot and scene models. The MJCF model assets are distributed separately as `robotblockset-mjcf` to keep the main `robotblockset` distribution small. Install with `pip install robotblockset[mjcf]`; this pulls `robotblockset-mjcf` and places files under `robotblockset/mujoco/mjcf_models`, with related meshes and textures under `robotblockset/mujoco/mjcf_models/assets`.


Included models cover several robots and scenes such as Panda, FR3, iiwa14, UR10e, HC20, MiR100, TiagoBase, Unitree B2, grippers, camera models, calibration scenes, and example workcells.

#### Genesis

Official website and project pages:

- https://genesis-embodied-ai.github.io/
- https://github.com/Genesis-Embodied-AI/Genesis

Required by `tutorial_genesis`:

```bash
pip install genesis-world torch
```

RBS imports `genesis` from the `genesis-world` package, and the backend also requires PyTorch.

#### Franka Robotics via `panda_py`

Official project pages:

- https://github.com/JeanElsner/panda-py
- https://franka.de/

Required by `tutorial_franka_pandapy.ipynb` and the modules in `robotblockset.franka`:

```bash
pip install panda-python
```

This backend is intended for direct connection to Franka Panda / FR3 robots. In practice you also need:

- a robot with FCI enabled
- network access to the controller
- a `panda_py` / `libfranka` version compatible with your robot software

#### Universal Robots via RTDE

Official project pages:

- https://pypi.org/project/ur-rtde/
- https://www.universal-robots.com/

Required by `tutorial_ur_rtde.ipynb` and the modules in `robotblockset.ur`:

```bash
pip install ur_rtde
```

#### CoppeliaSim

Official website and documentation:

- https://www.coppeliarobotics.com/
- https://manual.coppeliarobotics.com/

Required by the modules in `robotblockset.coppelia`:

```bash
pip install coppeliasim-zmqremoteapi-client
```

You also need a local CoppeliaSim installation with the `zmqRemoteApi` server enabled.

#### Cameras and calibration

The camera modules are split by hardware. The image-processing utilities and calibration notebooks rely mainly on OpenCV, Pydantic, and often MuJoCo.

Common camera/calibration packages:

```bash
pip install opencv-contrib-python pydantic open3d
```

Additional packages by camera type:

- Intel RealSense: `pip install pyrealsense2`
  Official docs: https://dev.intelrealsense.com/docs/docs-get-started
- Basler: `pip install pypylon`
  Official docs: https://docs.baslerweb.com/pythonProgGuide.html
- ZED: install the ZED SDK first, then its Python bindings
  Official docs: https://www.stereolabs.com/docs/

The camera calibration tutorials also import:

```bash
pip install mujoco mediapy
```

#### ROS / ROS2

Official documentation:

- ROS1: https://wiki.ros.org/
- ROS2: https://docs.ros.org/en/jazzy/index.html

If you use the ROS or ROS2 backends, install the middleware through your ROS distribution rather than plain `pip`.

ROS1:

```bash
sudo apt install python3-rospy
```

ROS2:

```bash
sudo apt update
sudo apt install ros-<ros-distro>-rclpy
```

For ROS2 camera support you typically also need:

```bash
sudo apt install ros-<ros-distro>-cv-bridge
```

> ⚠️Important: if you use `cv_bridge`, prefer a NumPy 1.x environment for now, for example:

```bash
pip install "numpy<2"
```

The reason is that `cv_bridge` builds distributed through ROS packages are often compiled against NumPy 1.x and may fail to import with NumPy 2.x, typically with errors such as `_ARRAY_API not found`.

For Franka ROS2 support, RBS expects packages such as `franka_ros2` and `franka_msgs` to be available in the ROS workspace.

If you work with custom message packages or a preconfigured environment, using the institute Docker/workspace setup may be easier:

- https://repo.ijs.si/hcr/rbs-docker

### Optional devices and utilities

#### SpaceMouse

To use a 3Dconnexion SpaceMouse with RBS:

```bash
pip install pyspacemouse easyhid
```

On Windows, you may also need `hidapi.dll` available on `PATH`.

On Linux, if the device is detected but cannot be opened, create an appropriate `udev` rule, reload the rules, and ensure your user belongs to the `input` group.

#### Other useful packages

Depending on your workflow, these can also be useful:

```bash
pip install pynput
pip install pyformulas
pip install aiohttp aiofiles
```

## Documentation

RBS provides several tutorial notebooks in `robotblockset/tutorials`, which can help you to get started and explore specific backends and workflows:

- `tutorial_spatial_operations`
- `tutorial_motion_generation`
- `tutorial_robots`
- `tutorial_platforms`
- `tutorial_mobile_robots`
- `tutorial_multi_robots`
- `tutorial_kinematic_models`
- `tutorial_optimal_trajectory`
- `tutorial_generation_collision-free_trajectories`
- `tutorial_image_video_pymujoco`
- `tutorial_mujoco`
- `tutorial_generate_MJCF_scene`
- `tutorial_genesis`
- `tutorial_graphics`
- `tutorial_franka_pandapy`
- `tutorial_franka_ros2`
- `tutorial_rbf`
- `tutorial_ur_rtde`
- `tutorial_calibrate_camera_charuco`
- `tutorial_calibrate_camera_checker`
- `tutorial_image_transform`

RBS also provides example notebooks and scripts in `robotblockset/examples`, which can be additional help when adapting the toolbox to your own robots, scenes, and applications.

## API documentation

The repository includes a Sphinx configuration under `docs` for generating API documentation directly from module, class, method, and function docstrings.

Install the package together with the documentation dependency:

```bash
pip install -e ".[docs]"
```

Then build the HTML documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser after the build completes.

The Sphinx setup mocks optional backend dependencies such as ROS, MuJoCo, camera SDKs, and vendor-specific drivers so the API reference can be built without installing every robotics stack.

## Troubleshooting

If you get on Windows following error:

```
tkinter.TclError: Can't find a usable init.tcl in the following directories: C:/Python313/lib/tcl8.6 C:/lib/tcl8.6 C:/lib/tcl8.6 C:/library C:/library C:/tcl8.6.14/library C:/tcl8.6.14/library 

This probably means that Tcl wasn't installed properly.  
```

the solution is to  set the environment variable manually:

1. Open **Control Panel** → **System** → **Advanced system settings**.
2. Go to **Environment Variables**.
3. Under **System Variables**, click **New**.
4. Set:
   - **Variable name:** `TCL_LIBRARY`
   - **Variable value:** `C:\Python313\tcl\tcl8.6` (adjust if your folder is different)

## Citation

Please cite the following article(s) in your publications if it helps your research :

```latex
@InProceedings{10.1007/978-3-031-59257-7_44,
author="{\v{Z}}lajpah, Leon and Petri{\v{c}}, Tadej",
editor="Pisla, Doina and Carbone, Giuseppe and Condurache, Daniel and Vaida, Calin",
title="RobotBlockSet (RBS)---A Comprehensive Robotics Framework",
booktitle="Advances in Service and Industrial Robotics",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="439--450",
isbn="978-3-031-59257-7"
}
@InProceedings{10.1007/978-3-032-02106-9_25,
author="Simoni{\v{c}}, Mihael and Kuster, Boris and Mavsar, Matija and Nimac, Peter and {\v{Z}}lajpah, Leon",
editor="Jovanovi{\'{c}}, Kosta and Rodi{\'{c}}, Aleksandar and Rakovi{\'{c}}, Mirko",
title="robotblockset{\_}python: Python Version of the RobotBlockSet",
booktitle="Advances in Service and Industrial Robotics",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="221--229",
isbn="978-3-032-02106-9"
}
```

​                      

------



Copyright: Leon Žlajpah, Jožef Stefan Insitute

