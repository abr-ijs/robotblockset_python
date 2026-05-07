"""Spatial pose and point utilities.

This module defines spatial utility models and transformation helpers for points and poses.
It provides typed pose representations, conversion between homogeneous matrices and structured pose models,
and point-transformation utilities that work directly with NumPy arrays. The module streamlines common spatial
operations used by camera and calibration workflows while preserving a user-friendly API.

Key functionalities include:
- Structured 3D pose models for position and Euler-angle orientation.
- Conversion between homogeneous transforms and typed pose objects.
- Homogeneous-coordinate helpers for batched point transformations.
- Utilities for transforming single points and point sets with 4x4 matrices.
- Input-shape normalization for robust NumPy-based spatial computations.
- Lightweight abstractions optimized for internal camera workflow usage.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import numpy as np
from robotblockset.rbs_typing import HomogeneousMatrixType, Vector3DArrayType, Vectors3DType
from robotblockset.transformations import map_pose
from pydantic import BaseModel


class Position(BaseModel):
    """Position in 3D space, all units are in meters."""

    x: float
    y: float
    z: float


class EulerAngles(BaseModel):
    roll: float
    pitch: float
    yaw: float


class Pose(BaseModel):
    """Pose of an object in 3D space, all units are in meters and radians.

    The euler angles are  extrinsic (rotations about the axes xyz of the original coordinate system, which is assumed
    to remain motionless).
    """

    position_in_meters: Position
    rotation_euler_xyz_in_radians: EulerAngles

    @classmethod
    def from_homogeneous_matrix(cls, matrix: HomogeneousMatrixType) -> Pose:
        """Build a pose from a homogeneous matrix.

        Creates a Pose object from a 4x4 homogeneous transformation matrix."""
        se3_pose = map_pose(T=matrix, out="pRPY")
        position = se3_pose[:3]
        euler_angles = se3_pose[-1:-4:-1]  # yaw, pitch, roll]

        position_model = Position(x=position[0], y=position[1], z=position[2])
        euler_angles_model = EulerAngles(roll=euler_angles[0], pitch=euler_angles[1], yaw=euler_angles[2])

        pose = cls(position_in_meters=position_model, rotation_euler_xyz_in_radians=euler_angles_model)
        return pose

    def as_homogeneous_matrix(self) -> HomogeneousMatrixType:
        """Returns the pose as a 4x4 homogeneous transformation matrix."""
        position = self.position_in_meters
        euler_angles = self.rotation_euler_xyz_in_radians

        position_array = np.array([position.x, position.y, position.z])
        RPY_array = np.array([euler_angles.yaw, euler_angles.pitch, euler_angles.roll])

        pose_matrix = map_pose(RPY=RPY_array, p=position_array, out="T")

        return pose_matrix


class _HomogeneousPoints:
    """Helper class to facilitate multiplicating 4x4 matrices with one or more 3D points.
    This class internally handles the addition / removal of a dimension to the points.
    """

    # TODO: extend to generic dimensions (1D,2D,3D).
    def __init__(self, points: Vectors3DType):
        if not self.is_valid_points_type(points):
            raise ValueError(f"Invalid argument for {_HomogeneousPoints.__name__}.__init__ ")

        points = _HomogeneousPoints.ensure_array_2d(points)
        self._homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)

    @staticmethod
    def is_valid_points_type(points: Vectors3DType) -> bool:
        if len(points.shape) == 1:
            if len(points) == 3:
                return True
        elif len(points.shape) == 2:
            if points.shape[1] == 3:
                return True
        return False

    @staticmethod
    def ensure_array_2d(points: Vectors3DType) -> Vector3DArrayType:
        """Ensure points are a 2D array.

        If points is a single shape (3,) point, then it is reshaped to (1,3)."""
        if len(points.shape) == 1:
            if len(points) != 3:
                raise ValueError("points has only one dimension, but it's length is not 3")
            points = points.reshape((1, 3))
        return points

    @property
    def homogeneous_points(self) -> np.ndarray:
        """Nx4 matrix representing the homogeneous points"""
        return self._homogeneous_points

    @property
    def points(self) -> Vectors3DType:
        """Nx3 matrix representing the points"""
        # normalize points (for safety, should never be necessary with affine transforms)
        # but we've had bugs of this type with projection operations, so better safe than sorry?
        scalars = self._homogeneous_points[:, 3][:, np.newaxis]
        points = self.homogeneous_points[:, :3] / scalars
        # TODO: if the original poitns was (1,3) matrix, then the resulting points would be a (3,) vector.
        #  Is this desirable? and if not, how to avoid it?
        if points.shape[0] == 1:
            # single point -> create vector from 1x3 matrix
            return points[0]
        else:
            return points

    def apply_transform(self, homogeneous_transform_matrix: HomogeneousMatrixType) -> None:
        self._homogeneous_points = (homogeneous_transform_matrix @ self.homogeneous_points.transpose()).transpose()


def transform_points(homogeneous_transform_matrix: HomogeneousMatrixType, points: Vectors3DType) -> Vectors3DType:
    """Applies a transform to a (set of) point(s).

    Parameters
    ----------
        homogeneous_transform_matrix : HomogeneousMatrixType
            _description_
        points : PointsType
            _description_
    Returns
    -------
        PointsType: (3,) vector or (N,3) matrix.
    """
    homogeneous_points = _HomogeneousPoints(points)
    homogeneous_points.apply_transform(homogeneous_transform_matrix)
    return homogeneous_points.points
