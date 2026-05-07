"""Camera calibration utilities.

This module defines utilities and data models for camera and hand-eye calibration workflows.
It provides ArUco/ChArUco detection result representations, camera intrinsics data structures, calibration
computation helpers, pose conversion and visualization tools, and serialization helpers for calibration artifacts.
The module supports building robust calibration pipelines from image data, robot poses, and OpenCV-based methods.

Key functionalities include:
- ArUco and ChArUco marker/corner detection result handling.
- Camera intrinsics representation and conversion from matrix form.
- Calibration computations for camera and hand-eye estimation.
- Board pose estimation and visualization on images.
- Pose and calibration artifact serialization to JSON.
- OpenCV calibration method mapping and interoperability helpers.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import glob
import json
import os
import re
import numpy as np
import cv2
from pydantic import BaseModel
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union
from robotblockset.rbs_typing import ArrayLike, CameraIntrinsicsMatrixType, CameraResolutionType, OpenCVIntImageType, HomogeneousMatrixType, Pose3DType

from robotblockset.cameras.interfaces import RGBCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.transformations import map_pose, spatial2t

# from loguru import logger
from robotblockset.tools import get_logger

logger = get_logger(__name__)

ArucoDictType = cv2.aruco.Dictionary
CharucoBoardType = cv2.aruco.CharucoBoard

CHECKER_BOARD = 0
CHARUCO_BOARD = 1

DEFAULT_ARUCO_DICT: ArucoDictType = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DEFAULT_CHARUCO_BOARD: CharucoBoardType = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.031, DEFAULT_ARUCO_DICT)

cv2_CALIBRATION_METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Haraud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class ArucoMarkerDetectionResult:
    """
    Result of ArUco marker detection in a single image.
    """

    corners: np.ndarray  # (N,1,4,2)
    ids: np.ndarray  # (N,1)
    image: OpenCVIntImageType


@dataclass
class CharucoCornerDetectionResult(ArucoMarkerDetectionResult):
    """
    Result of ChArUco corner detection derived from ArUco detections.
    """

    # corners: np.ndarray(M,1,2)
    # ids: np.ndarray(M,1)
    # image: OpenCVIntImageType
    pass


class Resolution(BaseModel):
    """
    Image resolution in pixels.
    """

    width: int
    height: int

    def as_tuple(self) -> CameraResolutionType:
        """
        Return resolution as ``(width, height)`` tuple.

        Returns
        -------
        CameraResolutionType
            Resolution tuple.
        """
        return self.width, self.height


class FocalLengths(BaseModel):
    """
    Camera focal lengths in pixels.
    """

    fx: float
    fy: float


class PrincipalPoint(BaseModel):
    """
    Camera principal point in pixels.
    """

    cx: float
    cy: float


RadialDistortionCoefficients = List[float]
TangentialDistortionCoefficients = List[float]


class CameraIntrinsics(BaseModel):
    """A format for storing the camera intrinsics at a specific image resolution."""

    image_resolution: Resolution
    focal_lengths_in_pixels: FocalLengths
    principal_point_in_pixels: PrincipalPoint

    # Distortion coefficients are stored so you can add as many as you want.
    radial_distortion_coefficients: Optional[RadialDistortionCoefficients] = None
    tangential_distortion_coefficients: Optional[TangentialDistortionCoefficients] = None

    @classmethod
    def from_matrix_and_resolution(cls, intrinsics_matrix: CameraIntrinsicsMatrixType, resolution: Tuple[int, int]) -> CameraIntrinsics:
        """
        Create a camera intrinsics model from matrix form.

        Parameters
        ----------
        intrinsics_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix with shape ``(3, 3)``.
        resolution : Tuple[int, int]
            Image resolution as ``(width, height)``.

        Returns
        -------
        CameraIntrinsics
            Parsed intrinsics model.
        """
        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]

        width, height = resolution

        camera_intrinsics = cls(
            image_resolution=Resolution(width=width, height=height),
            focal_lengths_in_pixels=FocalLengths(fx=fx, fy=fy),
            principal_point_in_pixels=PrincipalPoint(cx=cx, cy=cy),
        )
        return camera_intrinsics

    def as_matrix(self) -> CameraIntrinsicsMatrixType:
        """
        Return camera intrinsics as a matrix.

        Returns
        -------
        CameraIntrinsicsMatrixType
            Camera intrinsics matrix with shape ``(3, 3)``.
        """
        fx = self.focal_lengths_in_pixels.fx
        fy = self.focal_lengths_in_pixels.fy
        cx = self.principal_point_in_pixels.cx
        cy = self.principal_point_in_pixels.cy

        intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics_matrix


class BoardDetectionResults(NamedTuple):
    """
    Aggregated marker and corner detections for a calibration board.
    """

    charuco_corners: Optional[np.ndarray]
    charuco_ids: Optional[np.ndarray]
    aruco_corners: Optional[np.ndarray]
    aruco_ids: Optional[np.ndarray]


class BoardPose(NamedTuple):
    """
    Board pose represented by OpenCV rotation and translation vectors.
    """

    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]

    def as_homogeneous_matrix(self) -> Optional[HomogeneousMatrixType]:
        """
        Convert pose vectors to a homogeneous transform.

        Returns
        -------
        HomogeneousMatrixType or None
            Homogeneous transform matrix when pose vectors are present,
            otherwise ``None``.
        """
        if self.rvec is None or self.tvec is None:
            return None
        return map_pose(A=self.rvec, p=self.tvec, out="T")

    def as_pose(self) -> Optional[Pose3DType]:
        """
        Convert pose vectors to pose-plus-quaternion representation.

        Returns
        -------
        Pose3DType or None
            Pose vector when pose vectors are present, otherwise ``None``.
        """
        if self.rvec is None or self.tvec is None:
            return None
        return map_pose(A=self.rvec, p=self.tvec)


class PointReferences(NamedTuple):
    """
    Matched object/image point references used in calibration.
    """

    object_points: np.ndarray
    image_points: np.ndarray


# Loading and saving
def load_images(images_dir: str) -> List[OpenCVIntImageType]:
    """
    Load images from a directory.

    Parameters
    ----------
    images_dir : str
        Directory containing image files.

    Returns
    -------
    list[OpenCVIntImageType]
        Loaded images sorted by filename.
    """
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
    image_files.sort()
    images = [cv2.imread(image_path) for image_path in image_files]
    return images


def save_images(images_dir: str, images: List[OpenCVIntImageType], prefix: str = "image") -> None:
    """
    Save images as JPEG files to a directory.

    Parameters
    ----------
    images_dir : str
        Output directory.
    images : List[OpenCVIntImageType]
        Images to save.
    prefix : str, optional
        str, default="image"
        File prefix used for generated filenames.

    Returns
    -------
    None
    """
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_{i:02d}.jpg"), image)


# Detection
def detect_aruco_markers(image: OpenCVIntImageType, dictionary: ArucoDictType) -> Optional[ArucoMarkerDetectionResult]:
    """
    Detect ArUco markers in an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Input image.
    dictionary : ArucoDictType
        ArUco dictionary used for detection.

    Returns
    -------
    ArucoMarkerDetectionResult or None
        Marker detection result, or ``None`` if no markers are found.
    """
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    if marker_corners is None or marker_ids is None:
        return None
    marker_corners_array = np.stack(marker_corners)
    marker_corners_array = refine_corner_detection(image, marker_corners_array)
    result = ArucoMarkerDetectionResult(marker_corners_array, marker_ids, image)
    return result


def detect_charuco_corners(image: OpenCVIntImageType, markers_detection_result: ArucoMarkerDetectionResult, charuco_board: CharucoBoardType) -> Optional[CharucoCornerDetectionResult]:
    """
    Detect ChArUco corners from previously detected ArUco markers.

    Parameters
    ----------
    image : OpenCVIntImageType
        Input image.
    markers_detection_result : ArucoMarkerDetectionResult
        ArUco marker detection output.
    charuco_board : CharucoBoardType
        Target ChArUco board definition.

    Returns
    -------
    CharucoCornerDetectionResult or None
        ChArUco corner detection result, or ``None`` if interpolation fails.
    """
    nb_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=markers_detection_result.corners,  # type: ignore # typed as Seq but accepts np.ndarray
        markerIds=markers_detection_result.ids,
        image=image,
        board=charuco_board,
    )
    if charuco_corners is None or charuco_ids is None:
        return None
    charuco_corners = refine_corner_detection(image, charuco_corners)
    result = CharucoCornerDetectionResult(charuco_corners, charuco_ids, image)
    return result


def refine_corner_detection(image: OpenCVIntImageType, corners: np.ndarray) -> np.ndarray:
    """
    Refine detected corners with sub-pixel accuracy.

    Parameters
    ----------
    image : OpenCVIntImageType
        Input image.
    corners : np.ndarray
        Corner array to refine.

    Returns
    -------
    np.ndarray
        Refined corners with the original input shape.
    """

    # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.1)
    corners_shape = corners.shape
    corners = np.reshape(corners, (-1, 2))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # use a small window size, to avoid influence of a neighboring marker/ checkerboard tile
    # even then this sometimes gave worse results than without the refinement, so keep an eye on this
    corners = cv2.cornerSubPix(gray_image, corners, (3, 3), (-1, -1), term)
    corners = np.reshape(corners, corners_shape)
    return corners


def get_poses_of_aruco_markers(markers_detection_result: ArucoMarkerDetectionResult, marker_size: float, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None) -> Optional[List[HomogeneousMatrixType]]:
    """
    Estimate poses of detected ArUco markers.

    Parameters
    ----------
    markers_detection_result : ArucoMarkerDetectionResult
        Detected marker corners and ids.
    marker_size : float
        Marker side length in meters.
    camera_matrix : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients.

    Returns
    -------
    list[HomogeneousMatrixType] or None
        Marker poses as homogeneous transforms, or ``None`` on failure.
    """
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners=markers_detection_result.corners,  # type: ignore # typed as Seq but accepts np.ndarray
        markerLength=marker_size,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    if rvecs is None and tvecs is None:
        return None
    elif rvecs.shape != tvecs.shape:
        raise ValueError("rvecs and tvecs should have the same shape. Do you have multiple markers with the same ID?")

    # combine the rvecs and tvecs into a single pose matrix
    marker_poses_in_camera_frame = [map_pose(A=rvec[0], p=tvec, out="T") for rvec, tvec in zip(rvecs, tvecs)]
    return marker_poses_in_camera_frame


def get_pose_of_charuco_board(charuco_corners_detection_result: CharucoCornerDetectionResult, charuco_board: CharucoBoardType, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None) -> Optional[HomogeneousMatrixType]:
    """
    Estimate the pose of a detected ChArUco board.

    Parameters
    ----------
    charuco_corners_detection_result : CharucoCornerDetectionResult
        Detected ChArUco corners and ids.
    charuco_board : CharucoBoardType
        Target ChArUco board definition.
    camera_matrix : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients.

    Returns
    -------
    HomogeneousMatrixType or None
        Board pose in camera frame, or ``None`` if estimation fails.

    Notes
    -----
    The board frame origin is in the top-left board corner.
    """
    charuco_corners = charuco_corners_detection_result.corners
    charuco_ids = charuco_corners_detection_result.ids

    # Use matchImagePoints to get the object and image points
    obj_points, img_points = charuco_board.matchImagePoints(charuco_corners, charuco_ids)  # type: ignore  # mypy does not accept these types, but they are correct
    if obj_points is None or img_points is None:
        return None

    # Use solvePnP for pose estimation
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)  # type: ignore  # mypy does not accept these types, but they are correct
    if not success or rvec is None or tvec is None:
        return None
    # combine the rvec and tvec into a single pose matrix
    charuco_pose_in_camera_frame = map_pose(A=rvec.flatten(), p=tvec.flatten(), out="T")
    return charuco_pose_in_camera_frame


def detect_charuco_board(image: OpenCVIntImageType, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None, aruco_dict: ArucoDictType = DEFAULT_ARUCO_DICT, charuco_board: CharucoBoardType = DEFAULT_CHARUCO_BOARD) -> Optional[HomogeneousMatrixType]:
    """
    Detect the pose of a ChArUco board from an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Input image.
    camera_matrix : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients.
    aruco_dict : ArucoDictType, optional
        ArucoDictType, default=DEFAULT_ARUCO_DICT
        ArUco dictionary used for marker detection.
    charuco_board : CharucoBoardType, optional
        CharucoBoardType, default=DEFAULT_CHARUCO_BOARD
        ChArUco board definition.

    Returns
    -------
    HomogeneousMatrixType or None
        Board pose in camera frame, or ``None`` if not detected.
    """

    aruco_result = detect_aruco_markers(image, aruco_dict)
    if not aruco_result:
        return None

    charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)
    if not charuco_result:
        return None

    charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, camera_matrix, dist_coeffs)
    return charuco_pose


def display_detection_results(image_of_board: OpenCVIntImageType, original_board: OpenCVIntImageType, detection_results: BoardDetectionResults, point_references: PointReferences) -> None:
    """
    Plot detected board corners against the reference board image.

    Parameters
    ----------
    image_of_board : OpenCVIntImageType
        Captured board image.
    original_board : OpenCVIntImageType
        Reference board image.
    detection_results : BoardDetectionResults
        Marker and corner detections.
    point_references : PointReferences
        Object and image point correspondences.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    img_rgb = cv2.cvtColor(image_of_board, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Image")

    axes[1].imshow(img_rgb)
    axes[1].axis("off")
    axes[1].scatter(
        np.array(detection_results.aruco_corners).squeeze().reshape(-1, 2)[:, 0],
        np.array(detection_results.aruco_corners).squeeze().reshape(-1, 2)[:, 1],
        s=5,
        c="green",
        marker="x",
    )
    axes[1].set_title("Aruco corners")

    axes[2].imshow(img_rgb)
    axes[2].axis("off")
    axes[2].scatter(detection_results.charuco_corners.squeeze()[:, 0], detection_results.charuco_corners.squeeze()[:, 1], s=20, edgecolors="red", marker="o", facecolors="none")
    axes[2].set_title("Checker corners")

    axes[3].imshow(cv2.cvtColor(original_board, cv2.COLOR_BGR2RGB))
    axes[3].scatter(point_references.object_points.squeeze()[:, 0], point_references.object_points.squeeze()[:, 1])
    axes[3].set_title("Result")

    fig.tight_layout()
    fig.savefig("test.png", dpi=900)
    plt.show()


# Visualisation
def draw_frame_on_image(image: OpenCVIntImageType, frame_pose_in_camera: HomogeneousMatrixType, camera_matrix: CameraIntrinsicsMatrixType, length: float = 0.2) -> OpenCVIntImageType:
    """
    Draw the 2D projection of a 3D frame on an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Image to annotate.
    frame_pose_in_camera : HomogeneousMatrixType
        Frame pose expressed in camera coordinates.
    camera_matrix : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    length : float, optional
        float, default=0.2
        Axis length for drawing.

    Returns
    -------
    OpenCVIntImageType
        Annotated image.
    """
    rvec = map_pose(T=frame_pose_in_camera, out="A")
    tvec = map_pose(T=frame_pose_in_camera, out="p")
    image = cv2.drawFrameAxes(image, camera_matrix, None, rvec, tvec, length)  # type: ignore  # mypy does not accept these types, but they are correct
    return image


def visualize_aruco_detections(image: OpenCVIntImageType, aruco_result: ArucoMarkerDetectionResult) -> OpenCVIntImageType:
    """
    Draw detected ArUco markers and ids on an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Image to annotate.
    aruco_result : ArucoMarkerDetectionResult
        ArUco detection result.

    Returns
    -------
    OpenCVIntImageType
        Annotated image.
    """
    image = cv2.aruco.drawDetectedMarkers(image, [x for x in aruco_result.corners], aruco_result.ids)
    return image


def visualize_charuco_detection(image: OpenCVIntImageType, result: CharucoCornerDetectionResult) -> OpenCVIntImageType:
    """
    Draw detected ChArUco corners and ids on an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Image to annotate.
    result : CharucoCornerDetectionResult
        ChArUco detection result.

    Returns
    -------
    OpenCVIntImageType
        Annotated image.
    """
    image = cv2.aruco.drawDetectedCornersCharuco(image, np.array(result.corners), np.array(result.ids), (255, 255, 0))
    return image


def detect_and_visualize_charuco_pose(image: OpenCVIntImageType, intrinsics: CameraIntrinsicsMatrixType, aruco_dict: ArucoDictType = DEFAULT_ARUCO_DICT, charuco_board: CharucoBoardType = DEFAULT_CHARUCO_BOARD, draw_aruco_detection: bool = True, draw_charuco_detection: bool = True) -> Optional[HomogeneousMatrixType]:
    """
    Detect and visualize a ChArUco board pose in an image.

    Parameters
    ----------
    image : OpenCVIntImageType
        Input image to annotate.
    intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    aruco_dict : ArucoDictType, optional
        ArucoDictType, default=DEFAULT_ARUCO_DICT
        ArUco dictionary used for marker detection.
    charuco_board : CharucoBoardType, optional
        CharucoBoardType, default=DEFAULT_CHARUCO_BOARD
        ChArUco board definition.
    draw_aruco_detection : bool, optional
        bool, default=True
        Draw ArUco detections on the image.
    draw_charuco_detection : bool, optional
        bool, default=True
        Draw ChArUco detections on the image.

    Returns
    -------
    HomogeneousMatrixType or None
        Detected board pose, or ``None`` when detection fails.
    """
    aruco_result = detect_aruco_markers(image, aruco_dict)
    if not aruco_result:
        return None

    if draw_aruco_detection:
        image = visualize_aruco_detections(image, aruco_result)

    charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)
    if not charuco_result:
        return None

    if draw_charuco_detection:
        image = visualize_charuco_detection(image, charuco_result)

    if charuco_result.corners.shape[0] >= 6:
        charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, intrinsics, None)
    else:
        charuco_pose = None
    if charuco_pose is None:
        return None

    image = draw_frame_on_image(image, charuco_pose, intrinsics)

    return charuco_pose


def visualize_board_live(
    board: CharucoBoardType,
    dict: ArucoDictType,
    camera: RGBCamera,
    draw_aruco_detection: bool = True,
    draw_charuco_detection: bool = True,
) -> None:
    """
    Show a live window with ArUco/ChArUco detections from a camera stream.

    Parameters
    ----------
    board : CharucoBoardType
        ChArUco board model.
    dict : ArucoDictType
        ArUco marker dictionary used for detection.
    camera : RGBCamera
        Camera providing RGB images and intrinsics.
    draw_aruco_detection : bool, optional
        bool, default=True
        Draw detected ArUco markers.
    draw_charuco_detection : bool, optional
        bool, default=True
        Draw detected ChArUco corners.

    Returns
    -------
    None
    """

    window_name = "Charuco detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("press Q to quit")
    while True:
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        intrinsics = camera.intrinsics_matrix()
        detect_and_visualize_charuco_pose(image, intrinsics, dict, board, draw_aruco_detection=draw_aruco_detection, draw_charuco_detection=draw_aruco_detection)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


# Calibration
def compute_hand_eye_calibration_error(tcp_poses_in_base: List[HomogeneousMatrixType], board_poses_in_camera: List[HomogeneousMatrixType], camera_pose: HomogeneousMatrixType) -> float:
    """
    Compute average residual error for the hand-eye calibration equation.

    Parameters
    ----------
    tcp_poses_in_base : List[HomogeneousMatrixType]
        TCP poses in base frame.
    board_poses_in_camera : List[HomogeneousMatrixType]
        Board poses in camera frame.
    camera_pose : HomogeneousMatrixType
        (eye-in-hand).

    Returns
    -------
    float
        Mean residual of the AX=XB consistency equation.
    """
    error = 0.0
    for i in range(len(tcp_poses_in_base) - 1):
        tcp_pose_in_base = tcp_poses_in_base[i]
        board_pose_in_camera = board_poses_in_camera[i]

        tcp_pose_in_base_2 = tcp_poses_in_base[i + 1]
        board_pose_in_camera_2 = board_poses_in_camera[i + 1]

        # cf https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        # for the AX=XB equation
        left_side = tcp_pose_in_base @ camera_pose @ board_pose_in_camera
        right_side = tcp_pose_in_base_2 @ camera_pose @ board_pose_in_camera_2
        error += float(np.linalg.norm(left_side - right_side))
    return error / (len(tcp_poses_in_base) - 1)


def eye_in_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    board_poses_in_camera: List[HomogeneousMatrixType],
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """
    Estimate camera pose for eye-in-hand calibration.

    Parameters
    ----------
    tcp_poses_in_base : List[HomogeneousMatrixType]
        TCP poses in base frame.
    board_poses_in_camera : List[HomogeneousMatrixType]
        Board poses in camera frame.
    method : int, optional
        int, default=cv2.CALIB_HAND_EYE_ANDREFF
        OpenCV hand-eye calibration method constant.

    Returns
    -------
    tuple[HomogeneousMatrixType or None, float or None]
        Estimated camera pose in TCP frame and residual error.
    """
    tcp_orientations_as_R_in_base = [tcp_pose[:3, :3] for tcp_pose in tcp_poses_in_base]
    tcp_positions_in_base = [tcp_pose[:3, 3] for tcp_pose in tcp_poses_in_base]
    marker_orientations_as_R_in_camera = [board_pose[:3, :3] for board_pose in board_poses_in_camera]
    marker_positions_in_camera = [board_pose[:3, 3] for board_pose in board_poses_in_camera]
    try:
        camera_rotation_matrix, camera_translation = cv2.calibrateHandEye(
            tcp_orientations_as_R_in_base,
            tcp_positions_in_base,
            marker_orientations_as_R_in_camera,
            marker_positions_in_camera,
            None,
            None,
            method,
        )
    except cv2.error:
        return None, None

    if camera_rotation_matrix is None or camera_translation is None:
        return None, None

    # We've noticed that the OpenCV output can contains NaNs, which crashes here.
    try:
        camera_pose_in_tcp_frame = map_pose(R=camera_rotation_matrix, p=camera_translation, out="T")
    except ValueError:
        return None, None

    # camera_pose_in_tcp_frame = map_pose(R=camera_rotation_matrix, p=camera_translation, out="T")

    calibration_error = compute_hand_eye_calibration_error(tcp_poses_in_base, board_poses_in_camera, camera_pose_in_tcp_frame)
    return camera_pose_in_tcp_frame, calibration_error


def eye_to_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    board_poses_in_camera: List[HomogeneousMatrixType],
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """
    Estimate camera pose for eye-to-hand calibration.

    Parameters
    ----------
    tcp_poses_in_base : List[HomogeneousMatrixType]
        TCP poses in base frame.
    board_poses_in_camera : List[HomogeneousMatrixType]
        Board poses in camera frame.
    method : int, optional
        int, default=cv2.CALIB_HAND_EYE_ANDREFF
        OpenCV hand-eye calibration method constant.

    Returns
    -------
    tuple[HomogeneousMatrixType or None, float or None]
        Estimated camera pose in base frame and residual error.
    """
    #  Invert the tcp_poses to make the AX=XB problem for eye_to_hand mode equivalent to the eye_in_hand mode.
    # cf https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    # cf https://forum.opencv.org/t/eye-to-hand-calibration/5690/2
    base_pose_in_tcp_frame = [np.linalg.inv(tcp_pose) for tcp_pose in tcp_poses_in_base]

    camera_pose_in_base, calibration_error = eye_in_hand_pose_estimation(base_pose_in_tcp_frame, board_poses_in_camera, method)
    return camera_pose_in_base, calibration_error


def compute_calibration(
    board_poses_in_camera: List[HomogeneousMatrixType],
    tcp_poses_in_base: List[HomogeneousMatrixType],
    mode: str = "eye_in_hand",
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """
    Compute hand-eye calibration for a selected mode and method.

    Parameters
    ----------
    board_poses_in_camera : List[HomogeneousMatrixType]
        Board poses in camera frame.
    tcp_poses_in_base : List[HomogeneousMatrixType]
        TCP poses in base frame.
    mode : str, optional
        str, default="eye_in_hand"
        Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.
    method : int, optional
        int, default=cv2.CALIB_HAND_EYE_ANDREFF
        OpenCV hand-eye calibration method constant.

    Returns
    -------
    tuple[HomogeneousMatrixType or None, float or None]
        Estimated camera pose and residual error.
    """
    if mode == "eye_in_hand":
        # pose of camera in tcp frame
        camera_pose, calibration_error = eye_in_hand_pose_estimation(tcp_poses_in_base, board_poses_in_camera, method)
    elif mode == "eye_to_hand":
        # pose of camera in base frame
        camera_pose, calibration_error = eye_to_hand_pose_estimation(tcp_poses_in_base, board_poses_in_camera, method)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return camera_pose, calibration_error


def save_board_detections(
    results_dir: str,
    board_poses_in_camera: List[Optional[HomogeneousMatrixType]],
    images: List[OpenCVIntImageType],
    intrinsics: CameraIntrinsicsMatrixType,
) -> None:
    """
    Save board-detection preview images with projected board frame.

    Parameters
    ----------
    results_dir : str
        Output directory.
    board_poses_in_camera : List[Optional[HomogeneousMatrixType]]
        Board poses in camera frame; entries may be ``None``.
    images : List[OpenCVIntImageType]
        Source images.
    intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.

    Returns
    -------
    None
    """

    board_detections_dir = os.path.join(results_dir, "board_detections")
    os.makedirs(board_detections_dir, exist_ok=True)

    for i, (board_pose, image) in enumerate(zip(board_poses_in_camera, images)):
        image_annotated = image.copy()
        if board_pose is None:
            continue
        draw_frame_on_image(image_annotated, board_pose, intrinsics)
        detection_filepath = os.path.join(board_detections_dir, f"board_detection_{i:04d}.jpg")
        cv2.imwrite(detection_filepath, image_annotated)


def draw_base_pose_on_image(
    image: OpenCVIntImageType,
    intrinsics: CameraIntrinsicsMatrixType,
    camera_pose: Optional[HomogeneousMatrixType],
    mode: str = "eye_in_hand",
    tcp_pose: Optional[HomogeneousMatrixType] = None,
) -> None:
    """
    Draw robot base frame on an image using calibration output.

    Parameters
    ----------
    image : OpenCVIntImageType
        Image to annotate.
    intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    camera_pose : HomogeneousMatrixType, optional
        Camera pose in base frame (eye-to-hand) or TCP frame (eye-in-hand).
    mode : str, optional
        str, default="eye_in_hand"
        Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.
    tcp_pose : HomogeneousMatrixType, optional
        TCP pose in base frame corresponding to the image.

    Returns
    -------
    None
    """
    if camera_pose is None:
        return

    if mode == "eye_to_hand":
        X_B_C = camera_pose  # Camera in base frame
        X_C_B = np.linalg.inv(X_B_C)
    if mode == "eye_in_hand":
        if tcp_pose is None:
            return  # tcp pose is required to visualize base in eye_in_hand mode

        X_TCP_C = camera_pose  # Camera in TCP frame
        X_B_TCP = tcp_pose
        X_C_TCP = np.linalg.inv(X_TCP_C)
        X_TCP_B = np.linalg.inv(X_B_TCP)
        X_C_B = X_C_TCP @ X_TCP_B

    base_pose_in_camera = X_C_B
    draw_frame_on_image(image, base_pose_in_camera, intrinsics)


def extrinsic_calibration_all_methods(
    results_dir: str,
    images: List[OpenCVIntImageType],
    tcp_poses_in_base: List[HomogeneousMatrixType],
    intrinsics: CameraIntrinsicsMatrixType,
    mode: str = "eye_in_hand",
    aruco_dict: ArucoDictType = DEFAULT_ARUCO_DICT,
    charuco_board: CharucoBoardType = DEFAULT_CHARUCO_BOARD,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Run extrinsic calibration with all supported OpenCV hand-eye methods.

    Parameters
    ----------
    results_dir : str
        Output directory for result files.
    images : List[OpenCVIntImageType]
        Calibration-board images.
    tcp_poses_in_base : List[HomogeneousMatrixType]
        TCP poses in base frame.
    intrinsics : CameraIntrinsicsMatrixType
        Camera intrinsics matrix.
    mode : str, optional
        str, default="eye_in_hand"
        Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.
    aruco_dict : ArucoDictType, optional
        ArucoDictType, default=DEFAULT_ARUCO_DICT
        ArUco dictionary used for board detection.
    charuco_board : CharucoBoardType, optional
        CharucoBoardType, default=DEFAULT_CHARUCO_BOARD
        ChArUco board definition.

    Returns
    -------
    tuple[dict[str, Any], dict[str, float]]
        Estimated camera poses and residual errors keyed by method name.
    """
    calibration_errors_filepath = os.path.join(results_dir, "residual_errors.json")
    calibration_errors = {}
    calibration_result_poses = {}

    board_poses_in_camera = [detect_charuco_board(image, intrinsics, aruco_dict=aruco_dict, charuco_board=charuco_board) for image in images]

    save_board_detections(results_dir, board_poses_in_camera, images, intrinsics)

    # Removes poses where no board was detected
    tcp_poses_in_base = [tcp_poses_in_base[i] for i, board_pose in enumerate(board_poses_in_camera) if board_pose is not None]
    board_poses_in_camera: List[HomogeneousMatrixType] = [board_pose for board_pose in board_poses_in_camera if board_pose is not None]  # type: ignore
    logger.info(f"Board poses were detected in {len(board_poses_in_camera)} of the calibration samples.")

    for name, method in cv2_CALIBRATION_METHODS.items():
        camera_pose, calibration_error = compute_calibration(board_poses_in_camera, tcp_poses_in_base, mode, method)  # type: ignore
        if calibration_error is None:
            calibration_error = np.inf

        logger.info(f"Residual error {name}: {calibration_error:.4f}")

        calibration_errors[name] = calibration_error
        calibration_result_poses[name] = camera_pose

        with open(calibration_errors_filepath, "w") as f:
            json.dump(calibration_errors, f, indent=4)

        if camera_pose is None:
            continue

        # Save the camera pose
        pose_path = os.path.join(results_dir, f"camera_pose_{name}.json")
        save_pose_to_json(pose_path, camera_pose)

        # Save an image with the pose drawn on it (use last image taken)
        image = images[-1].copy()
        draw_base_pose_on_image(image, intrinsics, camera_pose, mode, tcp_poses_in_base[-1])

        # Write residual error on image
        error_str = f"{name}: {calibration_error:.4f}"
        cv2.putText(image, error_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(results_dir, f"base_pose_in_camera_{name}.jpg"), image)

    return calibration_result_poses, calibration_errors


def load_calibration_data(
    calibration_dir: str,
) -> Tuple[List[OpenCVIntImageType], List[HomogeneousMatrixType], CameraIntrinsicsMatrixType, CameraResolutionType]:
    """
    Load calibration images, TCP poses, and camera intrinsics from disk.

    Parameters
    ----------
    calibration_dir : str
        Calibration root directory containing a ``data`` subdirectory.

    Returns
    -------
    tuple[list[OpenCVIntImageType], list[HomogeneousMatrixType], CameraIntrinsicsMatrixType, CameraResolutionType]
        Images, TCP poses, intrinsics matrix, and image resolution.
    """
    data_dir = os.path.join(calibration_dir, "data")

    # Loading the intrinsics and resolution
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_path, "r") as f:
        camera_intrinsics = CameraIntrinsics.model_validate_json(f.read())

    resolution = camera_intrinsics.image_resolution.as_tuple()
    intrinsics = camera_intrinsics.as_matrix()

    image_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.png")))
    pose_paths = sorted(glob.glob(os.path.join(data_dir, "tcp_pose_*.json")))

    images = [cv2.imread(image_path) for image_path in image_paths]
    tcp_poses = []
    for filepath in pose_paths:
        pose = load_pose_from_json(filepath)
        tcp_poses.append(pose)

    return images, tcp_poses, intrinsics, resolution


def load_pose_from_json(path: Union[str, Path]) -> HomogeneousMatrixType:
    """
    Load a pose from a JSON file as a homogeneous transform.

    Parameters
    ----------
    path : Union[str, Path]
        Path to a JSON file with ``position_in_meters`` and
        ``rotation_euler_xyz_in_radians`` fields.

    Returns
    -------
    HomogeneousMatrixType
        Homogeneous transformation matrix with shape ``(4, 4)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required keys are missing in the JSON file.
    ValueError
        If the values cannot be converted to floats.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    pos = data["position_in_meters"]
    rot = data["rotation_euler_xyz_in_radians"]

    x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
    roll = float(rot["roll"])
    pitch = float(rot["pitch"])
    yaw = float(rot["yaw"])
    return map_pose(p=[x, y, z], RPY=[yaw, pitch, roll], out="T")


def save_pose_to_json(path: Union[str, Path], x: ArrayLike) -> None:
    """
    Write a pose to a JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the output JSON file.
    x : ArrayLike
        Spatial pose representation accepted by :func:`spatial2t`, such as a
        homogeneous transform, rotation matrix, pose vector, position vector,
        quaternion, or position plus axis-angle vector.

    Returns
    -------
    None
    """
    T = spatial2t(x)
    p = map_pose(T=T, out="p")
    rpy = map_pose(T=T, out="RPY")

    data = {
        "position_in_meters": {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
        },
        "rotation_euler_xyz_in_radians": {
            "roll": float(rpy[2]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[0]),
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def T_to_pretty_string(T: HomogeneousMatrixType, precision: int = 6) -> str:
    """
    Format a homogeneous transform as a multiline aligned matrix string.

    Parameters
    ----------
    T : HomogeneousMatrixType
        Homogeneous transform matrix.
    precision : int, default=6
        Number of decimal places.

    Returns
    -------
    str
        Pretty-printed matrix text.
    """
    fmt = f"{{:{precision + 3}.{precision}f}}"
    lines = []
    for r in range(4):
        lines.append("[" + "  ".join(fmt.format(float(v)) for v in T[r, :]) + "]")
    return "\n".join(lines)


def find_method_pairs(folder: Path) -> List[Tuple[str, Path, Path]]:
    """
    Find matching calibration image/pose files by method name.

    Parameters
    ----------
    folder : Path
        Path
        Directory containing result files.

    Returns
    -------
    list[tuple[str, Path, Path]]
        Tuples of ``(method, image_path, json_path)`` for matching:
        ``base_pose_in_camera_<method>.jpg`` and
        ``camera_pose_<method>.json``.
    """
    img_re = re.compile(r"^base_pose_in_camera_(.+)\.jpg$", re.IGNORECASE)
    json_re = re.compile(r"^camera_pose_(.+)\.json$", re.IGNORECASE)

    imgs: Dict[str, Path] = {}
    jsons: Dict[str, Path] = {}

    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = img_re.match(p.name)
        if m:
            imgs[m.group(1)] = p
            continue
        m = json_re.match(p.name)
        if m:
            jsons[m.group(1)] = p

    methods = sorted(set(imgs.keys()) & set(jsons.keys()))
    pairs = [(method, imgs[method], jsons[method]) for method in methods]
    return pairs


def draw_matrix_overlay(bgr: OpenCVIntImageType, text: str) -> OpenCVIntImageType:
    """
    Overlay multiline text (matrix) on the image.

    Parameters
    ----------
    bgr : OpenCVIntImageType
        Input BGR image.
    text : str
        Multiline text to overlay.

    Returns
    -------
    OpenCVIntImageType
        Annotated image.
    """
    out = bgr.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1
    thickness = 1

    # background box
    lines = text.splitlines()
    line_h = 40
    x0, y0 = 20, 80
    box_w = 10 + max([cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines] + [300])
    box_h = 10 + line_h * len(lines)
    cv2.rectangle(out, (x0 - 8, y0 - 18), (x0 - 8 + box_w, y0 - 18 + box_h), (0, 0, 0), -1)

    y = y0 + 15
    for line in lines:
        cv2.putText(out, line, (x0, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    return out


def load_calibration_results(results_dir: str, overlay: bool = False, cols: int = 3, save_grid: Optional[str] = None) -> List[Tuple[str, OpenCVIntImageType, HomogeneousMatrixType]]:
    """
    Load saved calibration result images and poses from a results directory.

    Parameters
    ----------
    results_dir : str
        Directory containing per-method image and JSON pose outputs.
    overlay : bool, optional
        bool, default=False
        Overlay transform text on loaded images.
    cols : int, optional
        int, default=3
        Number of columns for optional result grid visualization.
    save_grid : str, optional
        Output path for saving the visualization grid.

    Returns
    -------
    list[tuple[str, OpenCVIntImageType, HomogeneousMatrixType]]
        Tuples of method name, image, and loaded pose matrix.

    Raises
    ------
    FileNotFoundError
        If ``results_dir`` does not exist.
    RuntimeError
        If no matching image/JSON result pairs are found.
    """
    folder = Path(results_dir)
    if not folder.exists():
        raise FileNotFoundError(folder)

    pairs = find_method_pairs(folder)
    if not pairs:
        raise RuntimeError("No matching (image,json) pairs found.")

    # Load all results
    results = []
    for method, img_path, json_path in pairs:
        T = load_pose_from_json(json_path)

        # Print to console
        print(f"\n=== Method: {method} ===")
        # print(f"Image: {img_path.name}")
        # print(f"JSON : {json_path.name}")
        print("T_camera_in_base:\n", T)

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"WARNING: could not read image: {img_path}")
            continue

        if overlay:
            overlay_text = T_to_pretty_string(T, precision=3)
            bgr = draw_matrix_overlay(bgr, overlay_text)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results.append((method, rgb, T))

    # Display as grid
    n = len(results)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (method, rgb, _) in enumerate(results):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(rgb)
        ax.set_title(method)
        ax.axis("off")

    plt.tight_layout()

    if save_grid:
        plt.savefig(save_grid, dpi=200)
        print(f"\nSaved grid to: {save_grid}")

    plt.show()
    return results
