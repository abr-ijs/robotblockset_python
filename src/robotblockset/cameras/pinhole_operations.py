"""Pinhole camera projection and back-projection utilities."""

from typing import Union, List
import numpy as np

from robotblockset.cameras.spatial_operations import _HomogeneousPoints
from robotblockset.rbs_typing import CameraIntrinsicsMatrixType, CameraExtrinsicMatrixType, Vector2DArrayType, Vector3DArrayType, Vector3DType, HomogeneousMatrixType, NumpyDepthMapType


def project_points_to_image_plane(
    positions_in_camera_frame: Union[Vector3DArrayType, Vector3DType],
    camera_intrinsics: CameraIntrinsicsMatrixType,
) -> Vector2DArrayType:
    """
    Project 3D camera-frame points to image pixels.

    Parameters
    ----------
    positions_in_camera_frame : Union[Vector3DArrayType, Vector3DType]
        3D point positions in the camera frame.
    camera_intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix with shape ``(3, 3)``.

    Returns
    -------
    Vector2DArrayType
        Pixel coordinates on the image plane.
    """

    homogeneous_positions_in_camera_frame = _HomogeneousPoints(positions_in_camera_frame).homogeneous_points.T
    homogeneous_positions_on_image_plane = camera_intrinsics @ homogeneous_positions_in_camera_frame[:3, ...]
    positions_on_image_plane = homogeneous_positions_on_image_plane[:2, ...] / homogeneous_positions_on_image_plane[2, ...]
    return positions_on_image_plane.T


def multiview_triangulation_midpoint(
    extrinsics_matrices: List[CameraExtrinsicMatrixType],
    intrinsics_matrices: List[CameraIntrinsicsMatrixType],
    image_coordinates: Vector2DArrayType,
) -> Vector3DType:
    """
    Triangulate a point from multiple views using the midpoint method.

    Parameters
    ----------
    extrinsics_matrices : List[CameraExtrinsicMatrixType]
        Extrinsics matrices for each viewpoint.
    intrinsics_matrices : List[CameraIntrinsicsMatrixType]
        Intrinsics matrices for each viewpoint.
    image_coordinates : Vector2DArrayType
        Image coordinates of the 3D point for each viewpoint.

    Returns
    -------
    Vector3DType
        Estimated 3D position in the world frame.
    """

    # determine the rays for each camera in the world frame
    rays = []
    for extrinsics_matrix, intrinsics_matrix, image_point in zip(extrinsics_matrices, intrinsics_matrices, image_coordinates):
        ray = extrinsics_matrix[:3, :3] @ np.linalg.inv(intrinsics_matrix) @ np.array([image_point[0], image_point[1], 1])
        ray = ray / np.linalg.norm(ray)
        rays.append(ray)

    lhs = 0
    rhs = 0
    for i, ray in enumerate(rays):
        rhs += (np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]) @ extrinsics_matrices[i][:3, 3]
        lhs += np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]

    lhs_inv = np.linalg.inv(lhs)
    midpoint = lhs_inv @ rhs
    return midpoint


def calculate_triangulation_errors(
    extrinsics_matrices: List[CameraExtrinsicMatrixType],
    intrinsics_matrices: List[CameraIntrinsicsMatrixType],
    image_coordinates: Vector2DArrayType,
    point: Vector3DType,
) -> List[float]:
    """
    Compute per-view triangulation errors.

    Parameters
    ----------
    extrinsics_matrices : List[CameraExtrinsicMatrixType]
        Extrinsics matrices for each viewpoint.
    intrinsics_matrices : List[CameraIntrinsicsMatrixType]
        Intrinsics matrices for each viewpoint.
    image_coordinates : Vector2DArrayType
        Image coordinates of the 3D point for each viewpoint.
    point : Vector3DType
        Estimated 3D point in the world frame.

    Returns
    -------
    List[float]
        Euclidean distances between the point and each viewing ray.
    """
    errors = []
    for extrinsics_matrix, intrinsics_matrix, image_point in zip(extrinsics_matrices, intrinsics_matrices, image_coordinates):
        ray = extrinsics_matrix[:3, :3] @ np.linalg.inv(intrinsics_matrix) @ np.array([image_point[0], image_point[1], 1])
        ray = ray / np.linalg.norm(ray)
        error = np.linalg.norm((np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]) @ ((extrinsics_matrix[:3, 3]) - point))
        errors.append(error.item())
    return errors


"""methods for getting from 2D image coordinates to 3D world coordinates using depth information (a.k.a. unprojection)"""


def unproject_using_depthmap(
    image_coordinates: Vector2DArrayType,
    depth_map: NumpyDepthMapType,
    camera_intrinsics: CameraIntrinsicsMatrixType,
    depth_heuristic_mask_size: int = 11,
    depth_heuristic_percentile: float = 0.05,
) -> np.ndarray:
    """
    Unprojects image coordinates to 3D positions using a depth map.

    Parameters
    ----------
        image_coordinates : Vector2DArrayType
            numpy array of shape (N, 2) containing the 2D pixel coordinates of the points on the image plane
        depth_map : NumpyDepthMapType
            numpy array of shape (height, width) containing the depth values for each pixel, i.e. the z-value of the 3D point in the camera frame corresponding to the pixel
        camera_intrinsics : CameraIntrinsicsMatrixType
            camera intrinsics matrix as a numpy array of shape (3, 3)

    Returns
    -------
        numpy array of shape (N, 3) containing the 3D positions of the points in the camera frame

    """

    # TODO: should we make this extraction method more generic? though I prefer to keep it simple and not add too many options
    depth_values = extract_depth_from_depthmap_heuristic(image_coordinates, depth_map, depth_heuristic_mask_size, depth_heuristic_percentile)
    return unproject_onto_depth_values(image_coordinates, depth_values, camera_intrinsics)


def unproject_onto_depth_values(
    image_coordinates: Vector2DArrayType,
    depth_values: np.ndarray,
    camera_intrinsics: CameraIntrinsicsMatrixType,
) -> np.ndarray:
    """
    Unprojects image coordinates to 3D positions using depth values for each coordinate.

    Parameters
    ----------
        image_coordinates : Vector2DArrayType
            numpy array of shape (N, 2) containing the 2D pixel coordinates of the points on the image plane
        depth_values : np.ndarray
            numpy array of shape (N,) containing the depth values for each pixel, i.e. the z-value of the 3D point in the camera frame corresponding to the pixel
        camera_intrinsics : CameraIntrinsicsMatrixType
            camera intrinsics matrix as a numpy array of shape (3, 3)

    Returns
    -------
        numpy array of shape (N, 3) containing the 3D positions of the points in the camera frame
    """
    if image_coordinates.shape[0] != depth_values.shape[0]:
        raise IndexError(f"coordinates and depth values must have the same length (but they are: {image_coordinates.shape[0]} and {depth_values.shape[0]})")

    homogeneous_coords = np.ones((image_coordinates.shape[0], 3))
    homogeneous_coords[:, :2] = image_coordinates
    homogeneous_coords = np.transpose(homogeneous_coords)
    rays_in_camera_frame = np.linalg.inv(camera_intrinsics) @ homogeneous_coords  # shape is cast by numpy to column vector!

    z_values_in_camera_frame = depth_values

    t = z_values_in_camera_frame / rays_in_camera_frame[2, :]

    positions_in_camera_frame = t * rays_in_camera_frame

    homogeneous_positions_in_camera_frame = _HomogeneousPoints(positions_in_camera_frame.T).homogeneous_points.T
    return homogeneous_positions_in_camera_frame[:3, ...].T


def unproject_onto_world_z_plane(
    image_coordinates: Vector2DArrayType,
    camera_intrinsics: CameraIntrinsicsMatrixType,
    camera_in_frame_pose: HomogeneousMatrixType,
    height: float,
) -> Vector3DArrayType:
    """
    Unproject image points onto a world Z plane.

    Parameters
    ----------
    image_coordinates : Vector2DArrayType
        Pixel coordinates on the image plane.
    camera_intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix with shape ``(3, 3)``.
    camera_in_frame_pose : HomogeneousMatrixType
        Camera pose in the target frame.
    height : float
        Height of the plane in the target frame.

    Returns
    -------
    Vector3DArrayType
        3D points on the plane ``Z = height`` in the target frame.
    """
    # convert to homogeneous coordinates and transpose to column vectors
    homogeneous_coords = np.ones((image_coordinates.shape[0], 3))
    homogeneous_coords[:, :2] = image_coordinates
    homogeneous_coords = np.transpose(homogeneous_coords)

    camera_frame_ray_vector = np.linalg.inv(camera_intrinsics) @ homogeneous_coords

    translation = camera_in_frame_pose[0:3, 3]
    rotation_matrix = camera_in_frame_pose[0:3, 0:3]

    world_frame_ray_vectors = rotation_matrix @ camera_frame_ray_vector
    world_frame_ray_vectors = np.transpose(world_frame_ray_vectors)
    t = (height - translation[2]) / world_frame_ray_vectors[:, 2]
    points = t[:, np.newaxis] * world_frame_ray_vectors + translation
    return points


def extract_depth_from_depthmap_heuristic(
    image_coordinates: Vector2DArrayType,
    depth_map: NumpyDepthMapType,
    mask_size: int = 11,
    depth_percentile: float = 0.05,
) -> np.ndarray:
    """
    A simple heuristic to get more robust depth values of the depth map. Especially with keypoints we are often interested in points
    on the edge of an object, or even worse on a corner. Not only are these regions noisy by themselves but the keypoints could also be
    be a little off.

    This function takes the percentile of a region around the specified point and assumes we are interested in the nearest object present.
    This is not always true (think about the backside of a box looking under a 45 degree angle) but it serves as a good proxy. The more confident
    you are of your keypoints and the better the heatmaps are, the lower you could set the mask size and percentile. If you are very, very confident
    you could directly take the point cloud as well instead of manually querying the heatmap, but I find that they are more noisy.

    Also note that this function assumes there are no negative infinity values (no objects closer than 30cm!)

    Returns
    -------
        (np.ndarray) a 1D array of the depth values for the specified coordinates
    """

    if mask_size % 2 == 0:
        raise ValueError("only odd sized markers allowed")
    if depth_percentile >= 0.25:
        # TODO: The question in this error message implies that we should not raise an error, but instead log a warning.
        raise ValueError("For straight corners, about 75 percent of the region will be background. Are your sure you want the percentile to be lower?")
    # check all coordinates are within the size of the depth map to avoid unwanted wrapping of the array indices
    if np.max(image_coordinates[:, 1]) >= depth_map.shape[0]:
        raise IndexError("V coordinates out of bounds")
    if np.max(image_coordinates[:, 0]) >= depth_map.shape[1]:
        raise IndexError("U coordinates out of bounds")
    if np.min(image_coordinates) < 0:
        raise IndexError("coordinates out of bounds")

    # convert coordinates to integers
    image_coordinates = image_coordinates.astype(np.int32)

    # extract depth values by taking the percentile of the depth values in a region around the point
    mask_size_squared = mask_size**2
    depth_regions = np.empty((image_coordinates.shape[0], mask_size_squared))
    for i in range(image_coordinates.shape[0]):
        # Calculate the desired mask boundaries (using int32 to avoid overflow)
        v_start_desired = int(image_coordinates[i, 1]) - mask_size // 2
        v_end_desired = int(image_coordinates[i, 1]) + mask_size // 2 + 1
        u_start_desired = int(image_coordinates[i, 0]) - mask_size // 2
        u_end_desired = int(image_coordinates[i, 0]) + mask_size // 2 + 1

        # Clip the mask boundaries to the image boundaries
        v_start = max(0, v_start_desired)
        v_end = min(depth_map.shape[0], v_end_desired)
        u_start = max(0, u_start_desired)
        u_end = min(depth_map.shape[1], u_end_desired)

        # Extract the valid depth region
        depth_region = depth_map[v_start:v_end, u_start:u_end]

        # Flatten and pad with NaN if the masked area is smaller than needed
        flattened_region = depth_region.flatten()
        if flattened_region.size < mask_size_squared:
            padded_region = np.full(mask_size_squared, np.nan)
            padded_region[: flattened_region.size] = flattened_region
            depth_regions[i, :] = padded_region
        else:
            depth_regions[i, :] = flattened_region

    depth_values = np.nanquantile(depth_regions, depth_percentile, axis=1)

    return depth_values
