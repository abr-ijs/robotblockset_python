"""Common typing aliases used across robotblockset.

This module is based on the airo-typing module (https://github.com/airo-ugent/airo-mono).

Copyright (c) 2025- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

#######################
# spatial algebra types
#######################
ArrayLike = Union[np.ndarray, List[float], List[int], Tuple[float, ...], Tuple[int, ...]]
"""array-like inputs converted via rbs_type"""

Vector2DType = np.ndarray
"""a (2,) np array that represents a 2D position/translation/direction"""
Vector2DArrayType = np.ndarray
"""a (N,2) np array that represents N 2D positions/translations/directions"""
Vectors2DType = Union[Vector2DType, Vector2DArrayType]
"""a convenience type that represents a (2,) 2D vector or (N,2) array of 3D vectors."""

Vector3DType = np.ndarray
""" a (3,) np array that represents a 3D position/translation/direction"""
Vector3DArrayType = np.ndarray
""" a (N,3) np array that represents N 3D positions/translations/directions"""
Vectors3DType = Union[Vector3DType, Vector3DArrayType]
""" a convenience type that represents a (3,) 3D vector or (N,3) array of 3D vectors."""

Pose3DType = np.ndarray
""" a (7, ) np array that represents 3D positions/quaterion"""
Pose3DArrayType = np.ndarray
""" a (N,7) np array that represents N 3D positions/quaterion"""
Poses3DType = Union[Pose3DType, Pose3DArrayType]
"""a convenience type that represents a (7,) 3D pose or (N,7) array of 3D poses."""

Velocity3DType = np.ndarray
""" a (6, ) np array that represents 3D velocities"""
Velocity3DArrayType = np.ndarray
""" a (N,6) np array that represents N 3D velocities"""
Velocities3DType = Union[Velocity3DType, Velocity3DArrayType]
"""a convenience type that represents a (6,) 3D pose or (6,7) array of 3D velocities."""

Acceleration3DType = np.ndarray
""" a (6, ) np array that represents 3D accelerations"""
Acceleration3DArrayType = np.ndarray
""" a (N,6) np array that represents N 3D accelerations """
Accelerations3DType = Union[Acceleration3DType, Acceleration3DArrayType]
"""a convenience type that represents a (6,) 3D pose or (6,7) array of 3D accelerations."""

QuaternionObjectArrayType = np.ndarray
"""Quaternionic QArray interface with an ndarray representation."""
QuaternionType = np.ndarray
"""scalar-first quaternion <w,x,y,z> that represents a rotation around the <x,y,z> axis with angle <theta>"""
QuaternionArrayType = np.ndarray
"""a (N,4) npy array that represets quaternion that represents a rotation around the <x,y,z> axis with angle <theta>"""
QuaternionsType = Union[QuaternionType, QuaternionArrayType]
"""a convenience type that represents a (4,) quaternion or (N,4) array of quaternions."""

RotationMatrixType = np.ndarray
"""3x3 rotation matrix"""
RotationMatrixArrayType = np.ndarray
"""a (N, 3, 3) array of 3x3 rotation matrices"""
RotationMatricesType = Union[RotationMatrixType, RotationMatrixArrayType]
"""a convenience type that represents a (3,3) rotation matrix or (N,3,3) array of rotation matrices."""

EulerAnglesType = np.ndarray
"""XYZ angles of rotation around the axes of the original frame (extrinsic). First rotate around X, then around Y, finally around Z."""
AxisAngleType = Tuple[Vector3DType, float]
""" a tuple of a unit vector representing the axis of rotation and a float representing the angle of rotation in radians. The rotation is right-handed around the axis."""

RotationVectorType = np.ndarray
""" Rotation vector <x*theta,y*theta,z*theta> that represents a rotation around the <x,y,z> axis with angle <theta>. The rotation is right-handed around the axis."""

HomogeneousMatrixType = np.ndarray
"""4x4 homogeneous transform matrix <<R,T>|<0,0,0,1>> that represents the pose of a frame A in another frame B. Shorthand notation is T^A_B.
The upper left 3x3 block R is the rotation matrix that represents the orientation of frame A with respect to frame B, and the upper right 3x1 block T is the translation vector
that represents the position of frame A with respect to frame B, expressed in frame B."""
HomogeneousMatrixArrayType = np.ndarray
"""array (N, 4,4) of homogeneous transform matrices"""
HomogeneousMatricesType = Union[HomogeneousMatrixType, HomogeneousMatrixArrayType]
"""a convenience type that represents a (4,4) homogenous matrix or (N,4,4) array of homogenous matrices."""

TCPType = Union[Pose3DType, HomogeneousMatrixType, Vector3DType]
"""a TCP pose expressed as (7,), (4,4), or (3,) array"""

CartesianPathType = Union[Pose3DArrayType, HomogeneousMatrixArrayType]
""" a (T, N) array of joint states (can be position/velocity/acceleration) that describe a path in joint space"""

WrenchType = np.ndarray
""" a (6,) numpy array that represents a wrench applied on a frame and expressed in a (possibly different) frame as [Fx,Fy,Fz,Tx,Ty,Tz].
Shorthand notation is W^F_E, where F is the frame the wrench is applied on, and E is the frame the wrench is expressed in.
"""

TwistType = np.ndarray
""" a (6,) numpy array that represents the spatial velocity or an incremental motion of one frame as measured in another frame (and possibly expressed in a third frame).
Shorthand notation is ^C T^B_A, where C is the frame the velocity is measured in, B is the frame the velocity is expressed in.
"""

#####################
# Manipulator types #
#####################

JointConfigurationType = np.ndarray
"""an (N,) numpy array that represents the joint angles for a robot"""

JointVelocityType = np.ndarray
"""an (N,) numpy array that represents the joint velocities for a robot"""

JointAccelerationType = np.ndarray
"""an (N,) numpy array that represents the joint accelerations for a robot"""

JointTorqueType = np.ndarray
"""an (N,) numpy array that represents the joint torques for a robot"""

JointPathType = np.ndarray
""" a (T, N) array of joint states (can be position/velocity/acceleration) that describe a path in joint space"""

TimesType = np.ndarray
""" a (T,) array of monotonically increasing times (float), corresponding to a path"""

JointPathConstraintType = Tuple[Callable[[JointConfigurationType], float], float]
"""a tuple of a constraint function and a tolerance value: when the constraint function's absolute output is smaller than the tolerance, the constraint is satisfied."""

JacobianType = np.ndarray
""" a (6, N) array representing the Jacobian matrix"""

ForwardKinematicsFunctionType = Callable[[JointConfigurationType], HomogeneousMatrixType]
""" a function that computes the forward kinematics of a given joint configuration"""

InverseKinematicsFunctionType = Callable[[HomogeneousMatrixType], List[JointConfigurationType]]
""" a function that computes one or more inverse kinematics solutions of a given TCP pose"""

JointConfigurationCheckerType = Callable[[JointConfigurationType], bool]
""" a function that checks a certain condition on a joint configuration, e.g. collision checking"""

######################
# camera related types
######################

OpenCVIntImageType = np.ndarray
"""an image in the OpenCV format: BGR, uint8, (H,W,C)"""

NumpyFloatImageType = np.ndarray
""" a float image in the numpy format: RGB, float (0-1), (H,W,C)"""
NumpyIntImageType = np.ndarray
""" an int image in the numpy format: RGB, uint8 (0-255), (H,W,C)"""
TorchFloatImageType = np.ndarray
""" an image in the torch format: RGB, float(0-1), (C,H,W)"""

NumpyDepthMapType = np.ndarray
""" a depth map (z-buffer),float, (H,W)"""
NumpyConfidenceMapType = np.ndarray
""" a confidence map (higher is more confidence), float(0-1), (H,W)"""

CameraResolutionType = Tuple[int, int]
""" a tuple of image (width, height) in pixels"""

CameraIntrinsicsMatrixType = np.ndarray
"""3x3 camera intrinsics matrix

K = [[fx,s,cx],[0,fy,cy],[0,0,1]]
see e.g. https://ksimek.github.io/2013/08/13/intrinsic/ for more details """

CameraExtrinsicMatrixType = HomogeneousMatrixType
"""4x4 camera extrinsic matrix,
this is the homogeneous matrix that describes the camera pose in the world frame"""

##########################
# 3D and point cloud types
##########################

BoundingBox3DType = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
""" a tuple of two tuples that represent the min and max corners of a 3D bounding box"""

PointCloudPositionsType = Vector3DArrayType
""" a (N,3) float32 numpy array that represents a point cloud"""

PointCloudColorsType = np.ndarray
""" a (N,3) uint8 numpy array that represents the RGB colors of a point cloud"""

PointCloudAttributesType = Dict[str, np.ndarray]
""" a dictionary of numpy arrays that represent additional attributes of a point cloud, e.g. normals, confidence, etc. """


@dataclass
class PointCloud:
    points: PointCloudPositionsType
    colors: Optional[PointCloudColorsType] = None
    attributes: Optional[PointCloudAttributesType] = None
