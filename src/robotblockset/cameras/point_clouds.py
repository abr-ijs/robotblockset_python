"""Point-cloud processing and visualization utilities."""

import numpy as np
import open3d as o3d
import open3d.core as o3c

from typing import Any, Tuple
from robotblockset.rbs_typing import BoundingBox3DType, HomogeneousMatrixType, PointCloud, Vector3DType, Vector3DArrayType, Vectors3DType


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


def filter_point_cloud(point_cloud: PointCloud, mask: Any) -> PointCloud:
    """
    Filter a point cloud by a mask.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to filter.
    mask : Any
        Mask used to index the point cloud arrays.

    Returns
    -------
    PointCloud
        Filtered point cloud.
    """
    points = point_cloud.points[mask]
    colors = None if point_cloud.colors is None else point_cloud.colors[mask]

    attributes = None
    if point_cloud.attributes is not None:
        attributes = {}
        for key, value in point_cloud.attributes.items():
            attributes[key] = value[mask]

    point_cloud_filtered = PointCloud(points, colors, attributes)
    return point_cloud_filtered


def generate_point_cloud_crop_mask(point_cloud: PointCloud, bounding_box: BoundingBox3DType) -> np.ndarray:
    """
    Build a point-cloud crop mask.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to crop.
    bounding_box : BoundingBox3DType
        Bounding box that surrounds the points to keep.

    Returns
    -------
    np.ndarray
        Mask that can be used to filter the point cloud.
    """
    points = point_cloud.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
    crop_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return crop_mask


def crop_point_cloud(point_cloud: PointCloud, bounding_box: BoundingBox3DType) -> PointCloud:
    """
    Crop a point cloud to a bounding box.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to crop.
    bounding_box : BoundingBox3DType
        Bounding box that surrounds the points to keep.

    Returns
    -------
    PointCloud
        Cropped point cloud.
    """
    crop_mask = generate_point_cloud_crop_mask(point_cloud, bounding_box)
    return filter_point_cloud(point_cloud, crop_mask.nonzero())


def transform_point_cloud(point_cloud: PointCloud, frame_transformation: HomogeneousMatrixType) -> PointCloud:
    """
    Transform a point cloud to another frame.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to transform.
    frame_transformation : HomogeneousMatrixType
        Transformation matrix from the current frame to the target frame.

    Returns
    -------
    PointCloud
        Transformed point cloud.
    """
    new_points = transform_points(frame_transformation, point_cloud.points)
    return PointCloud(new_points, point_cloud.colors, point_cloud.attributes)


def point_cloud_to_open3d(point_cloud: PointCloud) -> Any:  # TODO: change Any back to o3d.t.geometry.PointCloud
    """
    Convert a PointCloud to an Open3D tensor point cloud.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to convert.

    Returns
    -------
    Any
        Open3D tensor point cloud.
    """
    positions = o3c.Tensor.from_numpy(point_cloud.points)

    map_to_tensors = {
        "positions": positions,
    }

    if point_cloud.colors is not None:
        colors = o3c.Tensor.from_numpy(point_cloud.colors / 255.0)
        map_to_tensors["colors"] = colors

    if point_cloud.attributes is not None:
        for attribute_name, array in point_cloud.attributes.items():
            map_to_tensors[attribute_name] = o3c.Tensor.from_numpy(array)

    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    return pcd


def open3d_to_point_cloud(pcd: Any) -> PointCloud:  # TODO: change Any back to o3d.t.geometry.PointCloud
    """
    Convert an Open3D point cloud to PointCloud.

    Parameters
    ----------
    pcd : Any
        Open3D tensor point cloud.

    Returns
    -------
    PointCloud
        Converted point cloud dataclass.
    """
    points = pcd.point.positions.numpy()
    if "colors" in pcd.point:
        colors = pcd.point.colors.numpy()
        if colors.dtype == np.float32:
            colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = None

    attributes = {}
    for attribute_name, array in pcd.point.items():
        if attribute_name in ["positions", "colors"]:
            continue
        attributes[attribute_name] = array.numpy()

    return PointCloud(points, colors)


def open3d_point(position: Vector3DType, color: Tuple[float, float, float], radius: float = 0.01) -> Any:  # Change Any back to o3d.geometry.TriangleMesh
    """Creates a small sphere mesh for visualization in open3d.

    Parameters
    ----------
        position : Vector3DType
            3D position of the point
        color : Tuple[float, float, float]
            RGB color of the point as 0-1 floats
        radius : float, optional
            radius of the sphere

    Returns
    -------
        sphere: an open3d mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere
