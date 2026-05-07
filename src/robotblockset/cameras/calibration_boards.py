"""Calibration board utilities.

This module defines calibration board utilities and classes used for camera and hand-eye calibration workflows.
It provides board abstractions for ChArUco and checkerboard targets, board detection and pose estimation helpers,
point correspondence generation, intrinsics result models, and persistence/export helpers for calibration artifacts.
The module also includes visualization and validation utilities for robust calibration data processing.

Key functionalities include:
- ChArUco and checkerboard board modeling and rendering.
- Board detection and corner extraction from images.
- Object/image point reference generation for calibration.
- Board pose estimation and visualization utilities.
- Camera intrinsics result representation and JSON serialization.
- Helpers for calibration data storage and compatibility with OpenCV calibration pipelines.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import json
import re
import os
import numpy as np
import cv2
from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union
from robotblockset.rbs_typing import CameraIntrinsicsMatrixType, CameraResolutionType, OpenCVIntImageType, HomogeneousMatrixType, Pose3DType

from robotblockset.cameras.camera_calibration import cv2_CALIBRATION_METHODS, compute_calibration, draw_base_pose_on_image, save_board_detections, save_pose_to_json

from robotblockset.transformations import map_pose, spatial2t

# from loguru import logger
from robotblockset.tools import get_logger

logger = get_logger(__name__)

ArucoDictType = cv2.aruco.Dictionary
CharucoBoardType = cv2.aruco.CharucoBoard

DEFAULT_ARUCO_DICT: ArucoDictType = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DEFAULT_CHARUCO_BOARD: CharucoBoardType = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.031, DEFAULT_ARUCO_DICT)


def _criterion_flag(name: str, default: int) -> int:
    value = getattr(cv2, name, default)
    return value if isinstance(value, int) else default


_TERM_CRITERIA_EPS = _criterion_flag("TERM_CRITERIA_EPS", 2)
_TERM_CRITERIA_MAX_ITER = _criterion_flag("TERM_CRITERIA_MAX_ITER", 1)
_TERM_CRITERIA_COUNT = _criterion_flag("TERM_CRITERIA_COUNT", 1)

CHECKER_BOARD = 0
CHARUCO_BOARD = 1


# Helpers
def _slugify(s: str) -> str:
    """
    Convert free-form text into a filesystem-friendly slug.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Sanitized slug.
    """
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s


def _split_opencv_distortion(dist: np.ndarray) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Map OpenCV distortion vector to (radial, tangential).

    OpenCV (common) order:
      [k1, k2, p1, p2, k3, k4, k5, k6, ...]
    We store:
      radial: [k1, k2, k3, k4, k5, k6] (as many as present)
      tangential: [p1, p2] (if present)
    """
    d = np.asarray(dist, dtype=float).reshape(-1)
    if d.size == 0:
        return None, None

    tangential = None
    if d.size >= 4:
        tangential = [float(d[2]), float(d[3])]

    radial: List[float] = []
    if d.size >= 1:
        radial.append(float(d[0]))  # k1
    if d.size >= 2:
        radial.append(float(d[1]))  # k2
    if d.size >= 5:
        radial.append(float(d[4]))  # k3
    if d.size >= 6:
        radial.append(float(d[5]))  # k4
    if d.size >= 7:
        radial.append(float(d[6]))  # k5
    if d.size >= 8:
        radial.append(float(d[7]))  # k6

    return (radial if radial else None), tangential


def _ensure_obj_img_shapes(obj_pts: np.ndarray, img_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize shapes to OpenCV-friendly formats:
      object: (N,1,3) float32
      image : (N,1,2) float32
    """
    obj_pts = np.asarray(obj_pts)
    img_pts = np.asarray(img_pts)

    if obj_pts.ndim == 2 and obj_pts.shape[1] == 3:
        obj_pts = obj_pts.reshape(-1, 1, 3)
    if img_pts.ndim == 2 and img_pts.shape[1] == 2:
        img_pts = img_pts.reshape(-1, 1, 2)

    return obj_pts, img_pts


def _checkerboard_object_points(pattern_size: Tuple[int, int], square_length_m: float) -> np.ndarray:
    """
    pattern_size = (cols, rows) inner corners.
    Returns (N,3) in meters.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_length_m)
    return objp


def _as_K(K_init: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Validate and normalize an initial intrinsics matrix.

    Parameters
    ----------
    K_init : np.ndarray, optional
        Candidate camera intrinsics matrix.

    Returns
    -------
    np.ndarray or None
        Normalized ``float64`` matrix with shape ``(3, 3)``, or ``None``.

    Raises
    ------
    ValueError
        If the provided matrix is not ``3x3``.
    """
    if K_init is None:
        return None
    K = np.asarray(K_init, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K_init must be 3x3, got {K.shape}")
    return K


def _as_dist(dist_init: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Validate and normalize OpenCV distortion coefficients.

    Parameters
    ----------
    dist_init : np.ndarray, optional
        Candidate distortion vector.

    Returns
    -------
    np.ndarray or None
        Distortion coefficients with shape ``(N, 1)``, or ``None``.

    Raises
    ------
    ValueError
        If fewer than four coefficients are provided.
    """
    if dist_init is None:
        return None
    d = np.asarray(dist_init, dtype=np.float64).reshape(-1, 1)
    if d.shape[0] < 4:
        raise ValueError(f"dist_init must have at least 4 coeffs, got {d.shape[0]}")
    return d


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


# Helper definitions
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
        """Convert extrinsics to a homogeneous matrix.

        Convert (rvec, tvec) to a 4x4 homogeneous transformation matrix."""
        if self.rvec is None or self.tvec is None:
            return None
        return map_pose(A=self.rvec, p=self.tvec, out="T")

    def as_pose(self) -> Optional[Pose3DType]:
        """Convert (rvec, tvec) to a array (7,) - pose + quaternion."""
        if self.rvec is None or self.tvec is None:
            return None
        return map_pose(A=self.rvec, p=self.tvec)


class PointReferences(NamedTuple):
    """
    Matched object/image point references used in calibration.
    """

    object_points: np.ndarray
    image_points: np.ndarray


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
        """Build intrinsics from matrix and resolution.

        Creates a CameraIntrinsics object from a 3x3 matrix and an image resolution (width, height)."""
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
        """Return the intrinsics matrix.

        Returns the camera intrinsics as a 3x3 matrix, often called K."""
        fx = self.focal_lengths_in_pixels.fx
        fy = self.focal_lengths_in_pixels.fy
        cx = self.principal_point_in_pixels.cx
        cy = self.principal_point_in_pixels.cy

        intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics_matrix


@dataclass(frozen=True)
class CameraIntrinsicCalibrationResults:
    """
    Container for OpenCV intrinsic calibration outputs.
    """

    repError: float
    camMatrix: np.ndarray  # 3x3
    distcoeff: np.ndarray  # OpenCV dist vector (N,) or (N,1)
    rvecs: np.ndarray
    tvecs: np.ndarray
    image_size: Tuple[int, int]

    def to_camera_intrinsics(self, include_distortion: bool = True) -> CameraIntrinsics:
        """
        Convert OpenCV calibration output (K, dist) to your CameraIntrinsics model.

        Parameters
        ----------
        include_distortion : bool, optional
            bool, default=True
            If ``True``, include radial and tangential distortion coefficients
            extracted from ``distcoeff``.

        Returns
        -------
        CameraIntrinsics
            Camera intrinsics model built from ``camMatrix`` and ``image_size``.
        """
        K = np.asarray(self.camMatrix, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(f"camMatrix must be 3x3, got {K.shape}")

        intr = CameraIntrinsics.from_matrix_and_resolution(K, self.image_size)

        if include_distortion:
            radial, tangential = _split_opencv_distortion(self.distcoeff)
            intr.radial_distortion_coefficients = radial
            intr.tangential_distortion_coefficients = tangential

        return intr

    def write_intrinsics_json(
        self,
        camera: str,
        stream: str,
        out_dir: Union[str, Path] = ".",
        include_distortion: bool = True,
    ) -> Path:
        """
        Write camera intrinsics to ``<camera>_<stream>_intrinsic_calibration.json``.

        Parameters
        ----------
        camera : str
            Camera name used in the output filename.
        stream : str
            Stream identifier used in the output filename.
        out_dir : Union[str, Path], optional
            str or Path, default="."
            Output directory.
        include_distortion : bool, optional
            bool, default=True
            If ``True``, include distortion coefficients in the saved payload.

        Returns
        -------
        Path
            Full path to the saved JSON file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{_slugify(camera)}_{_slugify(stream)}_intrinsic_calibration.json"
        path = out_dir / filename

        intr = self.to_camera_intrinsics(include_distortion=include_distortion)

        # Pydantic v2: model_dump_json; v1: json()
        try:
            json_str = intr.model_dump_json(indent=2)  # pydantic v2
        except AttributeError:
            json_str = intr.json(indent=2)  # pydantic v1

        path.write_text(json_str, encoding="utf-8")
        return path


# Calibration boards
@dataclass
class CharucoBoard:
    """
    ChArUco board wrapper for OpenCV 4.11.

    - Uses meters internally by default.
    - Builds cv2.aruco objects in __post_init__ so the instance is always ready to use.
    - margin_size_m and scale_px_per_m are per-instance fields (not class vars).
    - Includes unit helpers and a nice repr.
    """

    type = CHARUCO_BOARD
    # ---- Board geometry ----
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.040  # meters
    marker_length_m: float = 0.031  # meters

    # ---- Rendering / display helpers ----
    margin_size_m: float = 0.0  # meters (used only for image() rendering)
    scale_px_per_m: float = 1000.0  # pixels per meter for image() rendering (1000 => 1px/mm)

    # ---- Dictionary / detection ----
    dictionary_id: int = cv2.aruco.DICT_4X4_50
    min_corners: int = 10

    # ---- Derived OpenCV objects (created in __post_init__) ----
    dictionary: Any = field(init=False, repr=False)
    board: Any = field(init=False, repr=False)
    detector: Any = field(init=False, repr=False)
    aruco_params: Any = field(init=False, repr=False)
    charuco_params: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate geometry parameters and initialize OpenCV ChArUco objects.

        Returns
        -------
        None
        """
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("squares_x and squares_y must be >= 2")

        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")

        if self.square_length_m <= 0 or self.marker_length_m <= 0:
            raise ValueError("square_length_m and marker_length_m must be > 0")

        if self.scale_px_per_m <= 0:
            raise ValueError("scale_px_per_m must be > 0")

        if self.margin_size_m < 0:
            raise ValueError("margin_size_m must be >= 0")

        # --- build OpenCV objects ---
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)

        self.board = cv2.aruco.CharucoBoard(
            size=(self.squares_x, self.squares_y),
            squareLength=self.square_length_m,
            markerLength=self.marker_length_m,
            dictionary=self.dictionary,
        )

        self.aruco_params = cv2.aruco.DetectorParameters()
        self.charuco_params = cv2.aruco.CharucoParameters()
        self.detector = cv2.aruco.CharucoDetector(self.board, self.charuco_params, self.aruco_params)

    def __repr__(self) -> str:
        """
        Return compact string representation of board geometry.

        Returns
        -------
        str
            Board description string.
        """
        return "CharucoBoard(" f"{self.squares_x}x{self.squares_y} squares, " f"square={self.square_length_m:.6g} m, " f"marker={self.marker_length_m:.6g} m, " f"dict={self.dictionary_id}, " f"min_corners={self.min_corners}" ")"

    @property
    def size(self) -> Tuple[int, int]:
        """
        Return board size as ``(squares_x, squares_y)``.

        Returns
        -------
        tuple[int, int]
            Number of squares along X and Y.
        """
        return self.squares_x, self.squares_y

    def detect(self, gray_image: OpenCVIntImageType) -> Optional[BoardDetectionResults]:
        """
        Detect ChArUco board in a grayscale image.

        Parameters
        ----------
        gray_image : OpenCVIntImageType
            Input grayscale image with shape ``(H, W)``. If a color image is
            provided, it is converted to grayscale.

        Returns
        -------
        BoardDetectionResults or None
            Detection result containing ChArUco and ArUco corners/ids, or
            ``None`` when no valid board detection is found.
        """
        if gray_image.ndim != 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        _tmp = BoardDetectionResults(*self.detector.detectBoard(gray_image))
        if _tmp.aruco_corners is None or len(_tmp.aruco_corners) == 0:
            _tmp = None
        elif _tmp.charuco_corners is None or len(_tmp.charuco_corners) == 0:
            _tmp = None
        return _tmp

    def get_poses_of_aruco_markers(self, detection_result: BoardDetectionResults, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None) -> Optional[List[HomogeneousMatrixType]]:
        """
        Estimate poses of detected ArUco markers.

        Parameters
        ----------
        detection_result : BoardDetectionResults
            Detection result containing ArUco corners and ids.
        camera_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        dist_coeffs : np.ndarray, optional
            Distortion coefficients passed to pose estimation.

        Returns
        -------
        list[HomogeneousMatrixType] or None
            Marker poses as homogeneous transforms, or ``None`` when pose
            estimation fails.
        """
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=detection_result.aruco_corners,  # type: ignore # typed as Seq but accepts np.ndarray
            markerLength=self.marker_length_m,
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

    def detectPose(self, gray_image: OpenCVIntImageType, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None, board_detection: Optional[BoardDetectionResults] = None) -> Optional[BoardPose]:
        """
        Detect checkerboard pose in a grayscale image.

        Parameters
        ----------
        gray_image : OpenCVIntImageType
            Input grayscale image with shape ``(H, W)``. If a color image is
            provided, it is converted to grayscale.
        camera_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        dist_coeffs : np.ndarray, optional
            Distortion coefficients passed to ``cv2.solvePnP``.
        board_detection : BoardDetectionResults, optional
            Precomputed board detections to reuse instead of running detection.

        Returns
        -------
        BoardPose or None
            Estimated board pose, or ``None`` if pose estimation fails.
        """
        if gray_image.ndim != 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        if board_detection is None:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_image, self.dictionary)
            if marker_corners is None or marker_ids is None:
                return BoardPose(rvec=None, tvec=None)
            marker_corners_array = np.stack(marker_corners)
            term = (_TERM_CRITERIA_EPS + _TERM_CRITERIA_COUNT, 100, 0.1)
            corners_shape = marker_corners_array.shape
            corners = np.reshape(marker_corners_array, (-1, 2))
            corners = cv2.cornerSubPix(gray_image, corners, (3, 3), (-1, -1), term)
            marker_corners_array = np.reshape(corners, corners_shape)

            nb_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners_array, marker_ids, gray_image, self.board)
            if charuco_corners is None or charuco_ids is None:
                return None

        else:
            charuco_corners = board_detection.charuco_corners
            charuco_ids = board_detection.charuco_ids

        if charuco_corners is None:
            return None

        obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)  # type: ignore  # mypy does not accept these types, but they are correct
        if obj_points is None or img_points is None:
            return None

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)  # type: ignore  # mypy does not accept these types, but they are correct
        if not success or rvec is None or tvec is None:
            return None
        else:
            return BoardPose(rvec=rvec, tvec=tvec)

    def drawFrame(self, image_bgr: OpenCVIntImageType, pose: Union[BoardPose, Pose3DType, HomogeneousMatrixType], camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None, length: float = 0.1) -> np.ndarray:
        """
        Draw board coordinate frame on a BGR image.

        Parameters
        ----------
        image_bgr : OpenCVIntImageType
            Input image.
        pose : Union[BoardPose, Pose3DType, HomogeneousMatrixType]
            Pose to render.
        camera_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        dist_coeffs : np.ndarray, optional
            Distortion coefficients.
        length : float, optional
            float, default=0.1
            Axis length in meters.

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        out = image_bgr.copy()
        if pose is not None:
            if isinstance(pose, BoardPose):
                rvec = pose.rvec
                tvec = pose.tvec
            else:
                T = spatial2t(pose)
                rvec = map_pose(T=T, out="A")
                tvec = map_pose(T=T, out="p")
            cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, length)
        return out

    def drawDetection(self, image_bgr: OpenCVIntImageType, board_detection: BoardDetectionResults, draw_aruco: bool = True, draw_charuco: bool = True) -> np.ndarray:
        """
        Draw detected ArUco and/or ChArUco corners on an image.

        Parameters
        ----------
        image_bgr : OpenCVIntImageType
            Input BGR image.
        board_detection : BoardDetectionResults
            Detection result containing corners and ids.
        draw_aruco : bool, optional
            bool, default=True
            Draw ArUco marker outlines when available.
        draw_charuco : bool, optional
            bool, default=True
            Draw ChArUco corners when available.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        out = image_bgr.copy()
        if board_detection is not None:
            if draw_aruco and board_detection.aruco_corners is not None and len(board_detection.aruco_corners) > 0:
                image_bgr = cv2.aruco.drawDetectedMarkers(out, [x for x in board_detection.aruco_corners], board_detection.aruco_ids)

            if draw_charuco and board_detection.charuco_corners is not None and len(board_detection.charuco_corners) > 0:
                image_bgr = cv2.aruco.drawDetectedCornersCharuco(out, np.array(board_detection.charuco_corners), np.array(board_detection.charuco_ids), (255, 255, 0))
        return out

    def image(self, margin_size_m: Optional[float] = None, scale_px_per_m: Optional[float] = None) -> np.ndarray:
        """
        Render the ChArUco board as an image.

        Parameters
        ----------
        margin_size_m : float, optional
            Margin around the board in meters. If ``None``, instance default is
            used.
        scale_px_per_m : float, optional
            Rendering scale in pixels per meter. If ``None``, instance default
            is used.

        Returns
        -------
        np.ndarray
            Rendered board image.
        """
        if margin_size_m is None:
            margin_size_m = self.margin_size_m
        if scale_px_per_m is None:
            scale_px_per_m = self.scale_px_per_m

        if margin_size_m < 0:
            raise ValueError("margin_size_m must be >= 0")
        if scale_px_per_m <= 0:
            raise ValueError("scale_px_per_m must be > 0")

        # Board physical size in meters: (squares_x * square_length, squares_y * square_length)
        w_px = int(round(self.squares_x * self.square_length_m * scale_px_per_m))
        h_px = int(round(self.squares_y * self.square_length_m * scale_px_per_m))
        margin_px = int(round(margin_size_m * scale_px_per_m))

        return self.board.generateImage([w_px, h_px], marginSize=margin_px)

    # Calibration
    def intrinsic_calibration(
        self,
        images: List[OpenCVIntImageType],
        calibration_flags: int = 0,
        silent: bool = True,
        K_init: Optional[np.ndarray] = None,
        dist_init: Optional[np.ndarray] = None,
        refine: bool = True,
        freeze_principal_point: bool = False,
        freeze_focal_length: bool = False,
        fix_aspect_ratio: bool = False,
        zero_tangent_dist: bool = False,
        fix_k: Optional[List[int]] = None,  # e.g. [3,4,5,6] to fix K3..K6
    ) -> CameraIntrinsicCalibrationResults:
        """
        Intrinsic calibration using OpenCV 4.11 ChArUcoDetector.

        Parameters
        ----------
        images : List[OpenCVIntImageType]
            Calibration images containing the board.
        calibration_flags : int, optional
            int, default=0
            Additional OpenCV calibration flags.
        silent : bool, optional
            bool, default=True
            If ``False``, show detection visualization while collecting points.
        K_init : np.ndarray, optional
            Initial ``3x3`` intrinsics matrix.
        dist_init : np.ndarray, optional
            Initial distortion coefficients.
        refine : bool, optional
            bool, default=True
            If ``True``, refine provided intrinsics/distortion as initial guess.
            If ``False``, keep provided intrinsics fixed.
        freeze_principal_point : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_PRINCIPAL_POINT``.
        freeze_focal_length : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_FOCAL_LENGTH``.
        fix_aspect_ratio : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_ASPECT_RATIO`` (requires ``K_init``).
        zero_tangent_dist : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_ZERO_TANGENT_DIST``.
        fix_k : List[int], optional
            Distortion coefficient indices to fix (values in ``1..6``).

        Returns
        -------
        CameraIntrinsicCalibrationResults
            OpenCV calibration output (RMS, K, distortion, rvecs, tvecs, image
            size).

        Notes
        -----
        With ``refine=True``, ``K_init``/``dist_init`` are used as initial
        guesses via ``cv2.CALIB_USE_INTRINSIC_GUESS``. With ``refine=False``,
        intrinsics are fixed using ``cv2.CALIB_FIX_INTRINSIC``.
        """
        total_object_points: List[np.ndarray] = []
        total_image_points: List[np.ndarray] = []
        image_size: Optional[tuple] = None  # (w,h)
        if not silent:
            ori_image = self.image()

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                h, w = gray.shape[:2]
                image_size = (w, h)

            det = BoardDetectionResults(*self.detect(gray))
            if det is None:
                continue

            if det.charuco_ids is None or det.charuco_corners is None:
                continue
            if len(det.charuco_ids) < self.min_corners:
                continue

            obj_pts, img_pts = self.board.matchImagePoints(det.charuco_corners, det.charuco_ids)

            if obj_pts is None or img_pts is None:
                continue
            if len(obj_pts) < self.min_corners:
                continue

            obj_pts, img_pts = _ensure_obj_img_shapes(obj_pts, img_pts)
            total_object_points.append(obj_pts)
            total_image_points.append(img_pts)

            if not silent:
                point_references = PointReferences(obj_pts * self.scale_px_per_m, img_pts * self.scale_px_per_m)
                display_detection_results(gray, ori_image, det, point_references)

        if image_size is None:
            raise RuntimeError("No readable images found.")
        if len(total_object_points) < 5:
            raise RuntimeError(f"Not enough valid frames for calibration (got {len(total_object_points)}).")

        # --- Build flags ---
        flags = int(calibration_flags)

        if freeze_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if freeze_focal_length:
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        if fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST

        if fix_k:
            for k in fix_k:
                if k == 1:
                    flags |= cv2.CALIB_FIX_K1
                elif k == 2:
                    flags |= cv2.CALIB_FIX_K2
                elif k == 3:
                    flags |= cv2.CALIB_FIX_K3
                elif k == 4:
                    flags |= cv2.CALIB_FIX_K4
                elif k == 5:
                    flags |= cv2.CALIB_FIX_K5
                elif k == 6:
                    flags |= cv2.CALIB_FIX_K6
                else:
                    raise ValueError("fix_k entries must be in {1,2,3,4,5,6}")

        # --- Optional initial intrinsics ---
        K0 = _as_K(K_init)
        d0 = _as_dist(dist_init)

        if (K0 is not None) or (d0 is not None):
            if refine:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS

                # If only one provided, create a reasonable default for the other
                if K0 is None:
                    w, h = image_size
                    fx = fy = float(max(w, h))
                    cx, cy = float(w) / 2.0, float(h) / 2.0
                    K0 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                if d0 is None:
                    d0 = np.zeros((5, 1), dtype=np.float64)
            else:
                # Keep intrinsics fixed
                flags |= cv2.CALIB_FIX_INTRINSIC
                if K0 is None:
                    raise ValueError("refine=False requires K_init (intrinsics must be provided).")
                if d0 is None:
                    d0 = np.zeros((5, 1), dtype=np.float64)

        # NOTE: CALIB_FIX_ASPECT_RATIO requires a valid initial K (fx/fy ratio)
        if fix_aspect_ratio and K0 is None:
            raise ValueError("fix_aspect_ratio=True requires K_init (or provide K_init via K_init=...).")

        # --- Calibrate / refine ---
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(total_object_points, total_image_points, image_size, K0, d0, flags=flags)

        return CameraIntrinsicCalibrationResults(repError=rms, camMatrix=K, distcoeff=dist, rvecs=rvecs, tvecs=tvecs, image_size=image_size)

    def extrinsic_calibration_all_methods(self, results_dir: str, images: List[OpenCVIntImageType], tcp_poses_in_base: List[HomogeneousMatrixType], intrinsics: CameraIntrinsicsMatrixType, mode: str = "eye_in_hand") -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Run all OpenCV extrinsic calibration methods.

        Computes the calibration solution for all methods available in OpenCV and saves the results to a directory.

        Parameters
        ----------
        results_dir : str
            Directory to save calibration outputs. It must already exist.
        images : List[OpenCVIntImageType]
            Captured calibration-board images.
        tcp_poses_in_base : List[HomogeneousMatrixType]
            TCP poses in the robot base frame.
        intrinsics : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        mode : str, optional
            Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.

        Returns
        -------
        tuple[dict[str, Any], dict[str, float]]
            Two dictionaries: estimated camera poses per method and residual
            calibration errors per method.
        """
        calibration_errors_filepath = os.path.join(results_dir, "residual_errors.json")
        calibration_errors = {}
        calibration_result_poses = {}

        board_poses_in_camera = [self.detectPose(image, intrinsics).as_homogeneous_matrix() for image in images]

        # Removes poses where no board was detected
        tcp_poses_in_base = [tcp_poses_in_base[i] for i, board_pose in enumerate(board_poses_in_camera) if board_pose is not None]
        board_poses_in_camera: List[HomogeneousMatrixType] = [board_pose for board_pose in board_poses_in_camera if board_pose is not None]  # type: ignore
        logger.info(f"Board poses were detected in {len(board_poses_in_camera)} of the calibration samples.")
        save_board_detections(results_dir, board_poses_in_camera, images, intrinsics)

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

    # ---------- Unit helper constructors ----------
    @classmethod
    def from_mm(cls, *, squares_x: int = 7, squares_y: int = 5, square_length_mm: float = 40.0, marker_length_mm: float = 31.0, margin_size_mm: float = 0.0, dictionary_id: int = cv2.aruco.DICT_4X4_50, min_corners: int = 10, scale_px_per_mm: float = 1.0) -> "CharucoBoard":
        """
        Construct a board using millimeters.

        Parameters
        ----------
        squares_x : int, optional
            int, default=7
            Number of board squares along X.
        squares_y : int, optional
            int, default=5
            Number of board squares along Y.
        square_length_mm : float, optional
            float, default=40.0
            Square size in millimeters.
        marker_length_mm : float, optional
            float, default=31.0
            Marker size in millimeters.
        margin_size_mm : float, optional
            float, default=0.0
            Rendering margin in millimeters.
        dictionary_id : int, optional
            int, default=cv2.aruco.DICT_4X4_50
            OpenCV ArUco dictionary id.
        min_corners : int, optional
            int, default=10
            Minimum detected corners required for use.
        scale_px_per_mm : float, optional
            float, default=1.0
            Rendering scale; ``1.0`` means 1 pixel per millimeter.

        Returns
        -------
        CharucoBoard
            Configured board instance.
        """
        return cls(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_m=square_length_mm / 1000.0,
            marker_length_m=marker_length_mm / 1000.0,
            margin_size_m=margin_size_mm / 1000.0,
            scale_px_per_m=scale_px_per_mm * 1000.0,
            dictionary_id=dictionary_id,
            min_corners=min_corners,
        )

    def as_mm(self) -> dict:
        """
        Return board geometry expressed in millimeters.

        Returns
        -------
        dict
            Dictionary containing board geometry and configuration values.
        """
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_mm": self.square_length_m * 1000.0,
            "marker_length_mm": self.marker_length_m * 1000.0,
            "margin_size_mm": self.margin_size_m * 1000.0,
            "dictionary_id": self.dictionary_id,
            "min_corners": self.min_corners,
        }


@dataclass
class CheckerBoard:
    """
    Checkerboard (classic chessboard) wrapper.

    - Uses meters internally by default.
    - Provides detection via findChessboardCornersSB (robust) with optional subpixel refinement.
    - Provides object points for solvePnP/calibration.
    - Includes unit helpers and a nice repr.

    Notes
    -----
    ``pattern_size`` uses INNER corners:
    ``cols`` is the number of inner corners along X (width), and ``rows`` is
    the number of inner corners along Y (height).
    """

    type = CHECKER_BOARD
    # ---- Board geometry (INNER corners) ----
    cols: int = 7
    rows: int = 6
    square_length_m: float = 0.0104  # meters

    # ---- Detection tuning ----
    use_sb: bool = True  # use findChessboardCornersSB if True, else classic findChessboardCorners
    refine_subpix: bool = True  # run cornerSubPix after detection (often helps)
    fast_check: bool = True  # only for classic detector; SB ignores this flag
    subpix_win: Tuple[int, int] = (11, 11)
    subpix_criteria: Tuple[int, int, float] = (_TERM_CRITERIA_EPS + _TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    min_corners = 10

    # ---- Rendering / display helpers ----
    margin_size_m: float = 0.0
    scale_px_per_m: float = 1000.0  # 1000 => 1px/mm

    # ---- Cached object points ----
    _obj_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate checkerboard parameters and precompute object points.

        Returns
        -------
        None
        """
        if self.cols < 2 or self.rows < 2:
            raise ValueError("cols and rows must be >= 2 (inner corners)")

        if self.square_length_m <= 0:
            raise ValueError("square_length_m must be > 0")

        if self.scale_px_per_m <= 0:
            raise ValueError("scale_px_per_m must be > 0")

        if self.margin_size_m < 0:
            raise ValueError("margin_size_m must be >= 0")

        # Precompute object points in board frame (Z=0 plane)
        # Order matches OpenCV's expected corner ordering for chessboard.
        self.min_corners = self.rows * self.cols
        objp = np.zeros((self.rows * self.cols, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        objp *= float(self.square_length_m)
        self._obj_points = objp

    def __repr__(self) -> str:
        """
        Return compact string representation of checkerboard geometry.

        Returns
        -------
        str
            Checkerboard description string.
        """
        return "CheckerBoard(" f"{self.cols}x{self.rows} inner corners, " f"square={self.square_length_m:.6g} m, " f"use_sb={self.use_sb}, " f"refine_subpix={self.refine_subpix}" ")"

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """
        Return checkerboard inner-corner pattern size.

        Returns
        -------
        tuple[int, int]
            Pattern size as ``(cols, rows)``.
        """
        return (self.cols, self.rows)

    @property
    def object_points(self) -> np.ndarray:
        """
        Return checkerboard object points in board coordinates.

        Returns
        -------
        np.ndarray
            Object points with shape ``(N, 3)`` in meters.
        """
        return self._obj_points.copy()

    def detect(self, gray_image: OpenCVIntImageType) -> BoardDetectionResults:
        """
        Detect checkerboard corners in a grayscale image.

        Parameters
        ----------
        gray_image : OpenCVIntImageType
            Input grayscale image with shape ``(H, W)``. If a color image is
            provided, it is converted to grayscale.

        Returns
        -------
        BoardDetectionResults
            Detection result with checkerboard corners in
            ``charuco_corners`` (shape ``(N, 1, 2)``) and synthetic ids in
            ``charuco_ids`` (shape ``(N, 1)``), or ``None`` fields if not found.
        """
        gray = gray_image
        if gray_image.ndim != 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        if self.use_sb:
            found, corners = cv2.findChessboardCornersSB(gray, self.pattern_size)
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            if self.fast_check:
                flags |= cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not found or corners is None:
            return BoardDetectionResults(charuco_corners=None, charuco_ids=None, aruco_corners=None, aruco_ids=None)

        # Ensure float32 shape (N,1,2)
        corners = corners.astype(np.float32)

        if self.refine_subpix:
            cv2.cornerSubPix(
                gray,
                corners,
                winSize=self.subpix_win,
                zeroZone=(-1, -1),
                criteria=self.subpix_criteria,
            )

        ids = [i for i in range(len(corners))]
        ids = np.array(ids, dtype=np.int32).reshape(-1, 1)
        return BoardDetectionResults(charuco_corners=corners, charuco_ids=ids, aruco_corners=None, aruco_ids=None)

    def detectPose(self, gray_image: OpenCVIntImageType, camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None, board_detection: Optional[BoardDetectionResults] = None) -> Optional[BoardPose]:
        """
        Detect checkerboard pose in a grayscale image.

        Parameters
        ----------
        gray_image : OpenCVIntImageType
            Input grayscale image with shape ``(H, W)``. If a color image is
            provided, it is converted to grayscale.
        camera_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        dist_coeffs : np.ndarray, optional
            Distortion coefficients passed to ``cv2.solvePnP``.
        board_detection : BoardDetectionResults, optional
            Precomputed board detections to reuse instead of running detection.

        Returns
        -------
        BoardPose or None
            Estimated board pose, or ``None`` if pose estimation fails.
        """
        if gray_image.ndim != 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        if board_detection is None:
            if self.use_sb:
                found, corners = cv2.findChessboardCornersSB(gray_image, self.pattern_size)
            else:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                if self.fast_check:
                    flags |= cv2.CALIB_CB_FAST_CHECK
                found, corners = cv2.findChessboardCorners(gray_image, self.pattern_size, flags)

            if not found or corners is None:
                return BoardPose(rvec=None, tvec=None)

            # Ensure float32 shape (N,1,2)
            corners = corners.astype(np.float32)

            if self.refine_subpix:
                cv2.cornerSubPix(
                    gray_image,
                    corners,
                    winSize=self.subpix_win,
                    zeroZone=(-1, -1),
                    criteria=self.subpix_criteria,
                )
        else:
            corners = board_detection.charuco_corners
            if corners is None:
                return BoardPose(rvec=None, tvec=None)

        success, rvec, tvec = cv2.solvePnP(self.object_points, corners, camera_matrix, dist_coeffs)
        if not success:
            return BoardPose(rvec=None, tvec=None)
        else:
            return BoardPose(rvec=rvec, tvec=tvec)

    def drawFrame(self, image_bgr: OpenCVIntImageType, pose: Union[BoardPose, Pose3DType, HomogeneousMatrixType], camera_matrix: CameraIntrinsicsMatrixType, dist_coeffs: Optional[np.ndarray] = None, length: float = 0.1) -> np.ndarray:
        """
        Draw checkerboard coordinate frame on a BGR image.

        Parameters
        ----------
        image_bgr : OpenCVIntImageType
            Input image.
        pose : Union[BoardPose, Pose3DType, HomogeneousMatrixType]
            Pose to render.
        camera_matrix : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        dist_coeffs : np.ndarray, optional
            Distortion coefficients.
        length : float, optional
            float, default=0.1
            Axis length in meters.

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        out = image_bgr.copy()
        if pose is not None:
            if isinstance(pose, BoardPose):
                rvec = pose.rvec
                tvec = pose.tvec
            else:
                T = spatial2t(pose)
                rvec = map_pose(T=T, out="A")
                tvec = map_pose(T=T, out="p")
            cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, length)
        return out

    def drawDetection(self, image_bgr: OpenCVIntImageType, board_detection: BoardDetectionResults, found: bool = True) -> np.ndarray:
        """
        Draw detected checkerboard corners on an image.

        Parameters
        ----------
        image_bgr : OpenCVIntImageType
            Input BGR image.
        board_detection : BoardDetectionResults
            Detection result containing checkerboard corners.
        found : bool, optional
            bool, default=True
            Status flag forwarded to OpenCV drawing utility.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        out = image_bgr.copy()
        cv2.drawChessboardCorners(out, self.pattern_size, board_detection.charuco_corners, found)
        return out

    def image(self, margin_size_m: Optional[float] = None, scale_px_per_m: Optional[float] = None) -> np.ndarray:
        """
        Render a synthetic checkerboard image for printing/debugging.
        This is not required for detection, but mirrors CharucoBoard.image() usability.

        Parameters
        ----------
        margin_size_m : float, optional
            Margin around the rendered board in meters.
        scale_px_per_m : float, optional
            Rendering scale in pixels per meter.

        Returns
        -------
        np.ndarray
            Rendered checkerboard image.

        Notes
        -----
        The rendered board has ``(cols + 1) x (rows + 1)`` squares because
        ``cols`` and ``rows`` represent inner corners.
        """
        if margin_size_m is None:
            margin_size_m = self.margin_size_m
        if scale_px_per_m is None:
            scale_px_per_m = self.scale_px_per_m

        if margin_size_m < 0:
            raise ValueError("margin_size_m must be >= 0")
        if scale_px_per_m <= 0:
            raise ValueError("scale_px_per_m must be > 0")

        squares_x = self.cols + 1
        squares_y = self.rows + 1

        w_px = int(round(squares_x * self.square_length_m * scale_px_per_m))
        h_px = int(round(squares_y * self.square_length_m * scale_px_per_m))
        margin_px = int(round(margin_size_m * scale_px_per_m))

        W = w_px + 2 * margin_px
        H = h_px + 2 * margin_px

        img = np.ones((H, W), dtype=np.uint8) * 255  # white background

        # draw squares
        for y in range(squares_y):
            for x in range(squares_x):
                if (x + y) % 2 == 0:
                    x0 = margin_px + int(round(x * self.square_length_m * scale_px_per_m))
                    y0 = margin_px + int(round(y * self.square_length_m * scale_px_per_m))
                    x1 = margin_px + int(round((x + 1) * self.square_length_m * scale_px_per_m))
                    y1 = margin_px + int(round((y + 1) * self.square_length_m * scale_px_per_m))
                    cv2.rectangle(img, (x0, y0), (x1, y1), color=0, thickness=-1)

        return img

    # Calibration
    def intrinsic_calibration(
        self,
        images: List[OpenCVIntImageType],
        calibration_flags: int = 0,
        silent: bool = True,
        K_init: Optional[np.ndarray] = None,
        dist_init: Optional[np.ndarray] = None,
        refine: bool = True,
        freeze_principal_point: bool = False,
        freeze_focal_length: bool = False,
        fix_aspect_ratio: bool = False,
        zero_tangent_dist: bool = False,
        fix_k: Optional[List[int]] = None,  # e.g. [3,4,5,6] to fix K3..K6
    ) -> CameraIntrinsicCalibrationResults:
        """
        Intrinsic calibration of a checkerboard dataset.

        Parameters
        ----------
        images : List[OpenCVIntImageType]
            Calibration images containing the board.
        calibration_flags : int, optional
            int, default=0
            Additional OpenCV calibration flags.
        silent : bool, optional
            bool, default=True
            If ``False``, show detection visualization while collecting points.
        K_init : np.ndarray, optional
            Initial ``3x3`` intrinsics matrix.
        dist_init : np.ndarray, optional
            Initial distortion coefficients.
        refine : bool, optional
            bool, default=True
            If ``True``, refine provided intrinsics/distortion as initial guess.
            If ``False``, keep provided intrinsics fixed.
        freeze_principal_point : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_PRINCIPAL_POINT``.
        freeze_focal_length : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_FOCAL_LENGTH``.
        fix_aspect_ratio : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_FIX_ASPECT_RATIO`` (requires ``K_init``).
        zero_tangent_dist : bool, optional
            bool, default=False
            Apply ``cv2.CALIB_ZERO_TANGENT_DIST``.
        fix_k : List[int], optional
            Distortion coefficient indices to fix (values in ``1..6``).

        Returns
        -------
        CameraIntrinsicCalibrationResults
            OpenCV calibration output (RMS, K, distortion, rvecs, tvecs, image
            size).

        Notes
        -----
        With ``refine=True``, ``K_init``/``dist_init`` are used as initial
        guesses via ``cv2.CALIB_USE_INTRINSIC_GUESS``. With ``refine=False``,
        intrinsics are fixed using ``cv2.CALIB_FIX_INTRINSIC``.
        """
        total_object_points: List[np.ndarray] = []
        total_image_points: List[np.ndarray] = []
        image_size: Optional[tuple] = None  # (w,h)

        # Precompute object points (same for every frame)
        obj_pts = _checkerboard_object_points(self.pattern_size, self.square_length_m)

        if not silent:
            ori_image = self.image()

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                h, w = gray.shape[:2]
                image_size = (w, h)

            det = self.detect(gray)

            if det.charuco_ids is None or det.charuco_corners is None:
                continue
            if len(det.charuco_ids) < self.min_corners:
                continue

            img_pts = det.charuco_corners
            obj_pts, img_pts = _ensure_obj_img_shapes(obj_pts, img_pts)
            total_object_points.append(obj_pts)
            total_image_points.append(img_pts)

            if not silent:
                point_references = PointReferences(obj_pts * self.scale_px_per_m, img_pts * self.scale_px_per_m)
                display_detection_results(gray, ori_image, det, point_references)

        if image_size is None:
            raise RuntimeError("No readable images found.")
        if len(total_object_points) < 5:
            raise RuntimeError(f"Not enough valid frames for calibration (got {len(total_object_points)}).")

        # --- Build flags ---
        flags = int(calibration_flags)

        if freeze_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if freeze_focal_length:
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        if fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST

        if fix_k:
            for k in fix_k:
                if k == 1:
                    flags |= cv2.CALIB_FIX_K1
                elif k == 2:
                    flags |= cv2.CALIB_FIX_K2
                elif k == 3:
                    flags |= cv2.CALIB_FIX_K3
                elif k == 4:
                    flags |= cv2.CALIB_FIX_K4
                elif k == 5:
                    flags |= cv2.CALIB_FIX_K5
                elif k == 6:
                    flags |= cv2.CALIB_FIX_K6
                else:
                    raise ValueError("fix_k entries must be in {1,2,3,4,5,6}")

        # --- Optional initial intrinsics ---
        K0 = _as_K(K_init)
        d0 = _as_dist(dist_init)

        if (K0 is not None) or (d0 is not None):
            if refine:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS

                # If only one provided, create a reasonable default for the other
                if K0 is None:
                    w, h = image_size
                    fx = fy = float(max(w, h))
                    cx, cy = float(w) / 2.0, float(h) / 2.0
                    K0 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                if d0 is None:
                    d0 = np.zeros((5, 1), dtype=np.float64)
            else:
                # Keep intrinsics fixed
                flags |= cv2.CALIB_FIX_INTRINSIC
                if K0 is None:
                    raise ValueError("refine=False requires K_init (intrinsics must be provided).")
                if d0 is None:
                    d0 = np.zeros((5, 1), dtype=np.float64)

        # NOTE: CALIB_FIX_ASPECT_RATIO requires a valid initial K (fx/fy ratio)
        if fix_aspect_ratio and K0 is None:
            raise ValueError("fix_aspect_ratio=True requires K_init (or provide K_init via K_init=...).")

        # --- Calibrate / refine ---
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(total_object_points, total_image_points, image_size, K0, d0, flags=flags)

        return CameraIntrinsicCalibrationResults(repError=rms, camMatrix=K, distcoeff=dist, rvecs=rvecs, tvecs=tvecs, image_size=image_size)

    def extrinsic_calibration_all_methods(
        self,
        results_dir: str,
        images: List[OpenCVIntImageType],
        tcp_poses_in_base: List[HomogeneousMatrixType],
        intrinsics: CameraIntrinsicsMatrixType,
        mode: str = "eye_in_hand",
    ) -> Tuple[dict, dict]:
        """Run all OpenCV extrinsic calibration methods.

        Computes the calibration solution for all methods available in OpenCV and saves the results to a directory.

        Parameters
        ----------
        results_dir : str
            Directory to save calibration outputs. It must already exist.
        images : List[OpenCVIntImageType]
            Captured calibration-board images.
        tcp_poses_in_base : List[HomogeneousMatrixType]
            TCP poses in the robot base frame.
        intrinsics : CameraIntrinsicsMatrixType
            Camera intrinsics matrix.
        mode : str, optional
            Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.

        Returns
        -------
        tuple[dict, dict]
            Two dictionaries: estimated camera poses per method and residual
            calibration errors per method.
        """
        calibration_errors_filepath = os.path.join(results_dir, "residual_errors.json")
        calibration_errors = {}
        calibration_result_poses = {}

        board_poses_in_camera = [self.detectPose(image, intrinsics).as_homogeneous_matrix() for image in images]

        # Removes poses where no board was detected
        tcp_poses_in_base = [tcp_poses_in_base[i] for i, board_pose in enumerate(board_poses_in_camera) if board_pose is not None]
        board_poses_in_camera: List[HomogeneousMatrixType] = [board_pose for board_pose in board_poses_in_camera if board_pose is not None]  # type: ignore
        logger.info(f"Board poses were detected in {len(board_poses_in_camera)} of the calibration samples.")
        save_board_detections(results_dir, board_poses_in_camera, images, intrinsics)

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

    @classmethod
    def from_mm(cls, *, cols: int = 6, rows: int = 4, square_length_mm: float = 40.0, margin_size_mm: float = 0.0, scale_px_per_mm: float = 1.0, use_sb: bool = True, refine_subpix: bool = True, fast_check: bool = True) -> "CheckerBoard":
        """
        Construct a checkerboard using millimeter units.

        Parameters
        ----------
        cols : int, optional
            int, default=6
            Number of inner corners along X.
        rows : int, optional
            int, default=4
            Number of inner corners along Y.
        square_length_mm : float, optional
            float, default=40.0
            Square size in millimeters.
        margin_size_mm : float, optional
            float, default=0.0
            Rendering margin in millimeters.
        scale_px_per_mm : float, optional
            float, default=1.0
            Rendering scale; ``1.0`` means 1 pixel per millimeter.
        use_sb : bool, optional
            bool, default=True
            Use ``cv2.findChessboardCornersSB`` when ``True``.
        refine_subpix : bool, optional
            bool, default=True
            Refine detected corners to subpixel precision.
        fast_check : bool, optional
            bool, default=True
            Enable fast-check flag for classic corner detector.

        Returns
        -------
        CheckerBoard
            Configured checkerboard instance.
        """
        return cls(
            cols=cols,
            rows=rows,
            square_length_m=square_length_mm / 1000.0,
            margin_size_m=margin_size_mm / 1000.0,
            scale_px_per_m=scale_px_per_mm * 1000.0,
            use_sb=use_sb,
            refine_subpix=refine_subpix,
            fast_check=fast_check,
        )

    def as_mm(self) -> dict:
        """
        Return checkerboard geometry expressed in millimeters.

        Returns
        -------
        dict
            Dictionary containing checkerboard geometry and detection settings.
        """
        return {
            "cols_inner": self.cols,
            "rows_inner": self.rows,
            "square_length_mm": self.square_length_m * 1000.0,
            "margin_size_mm": self.margin_size_m * 1000.0,
            "use_sb": self.use_sb,
            "refine_subpix": self.refine_subpix,
            "fast_check": self.fast_check,
        }
