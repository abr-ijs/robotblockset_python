# RobotBlockset for Python (RBS)


## Synopsis

RobotBlockset (RBS) toolbox provides tools for designing, simulating, and testing  robot manipulators in Python. The toolbox provides the means to describe the position and orientation of objects in 3D space and spatial velocities using homogenous matrices or  quaternions. It defines it's own quaternion class with all necessary quaternion operations  and provides functions for relevant transformations between different representations. For robot manipulators the toolbox provides algorithms for spatial trajectory generation, forward and inverse kinematics, considering also robot intrinsic and user defined functional redundancy, and control algorithms for motion control. 

For robot models, grippers and sensors custom classes are defined, which let you execute your robot applications by connecting the robot models to different external simulation environments (not included in this distribution). The toolbox provides models of some common robot manipulators from KUKA LWR, Franka Emika and Universal Robotics. The main advantage of the toolbox is that it provides an unified higher level syntax to operate different robots. It allows the user to operate the robot in the simulation environment or to connect directly to  a robotics platform and operate a real robot.

### Dependencies

Installed with RBS:

```python
pip instal numpy
pip instal quaternionic
pip instal scipy
pip intsall matplotlib
```

If ROS is used, then it is necessary to install them together with Python bindings. For ROS:

```
sudo apt install python3-rospy
```


Can be usefull:

```
pip install pynput
pip install pyformulas
pip install aiohttp aiofiles
```


