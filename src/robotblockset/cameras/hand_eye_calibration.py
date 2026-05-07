"""Hand-eye calibration workflow.

This module defines the interactive hand-eye calibration workflow for camera-robot setups.
It supports both eye-in-hand and eye-to-hand configurations, including sample collection,
live board detection visualization, repeated calibration over accumulated samples, and result selection
based on calibration error metrics. The module integrates robot poses, captured images, and camera intrinsics
to drive end-to-end calibration sessions.

Key functionalities include:
- Interactive calibration loop for eye-in-hand and eye-to-hand modes.
- Synchronized acquisition of robot TCP poses and camera images.
- Live ChArUco detection and base-pose visualization during data collection.
- Incremental calibration execution as new samples are captured.
- Multi-method calibration evaluation and best-result tracking by error.
- Structured persistence of collected samples and calibration outputs.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import json
import os
from typing import List, Optional

import cv2
from robotblockset.cameras.collect_calibration_data import create_data_dir, save_calibration_sample
from robotblockset.cameras.camera_calibration import extrinsic_calibration_all_methods, draw_base_pose_on_image
from robotblockset.cameras.camera_calibration import ArucoDictType, CharucoBoardType, detect_and_visualize_charuco_pose
from robotblockset.cameras.interfaces import RGBCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.cameras.camera_calibration import CameraIntrinsics
from robotblockset.rbs_typing import HomogeneousMatrixType, OpenCVIntImageType
from robotblockset.robots import robot

# from loguru import logger
from robotblockset.tools import get_logger

logger = get_logger(__name__)


def do_camera_robot_calibration(
    mode: str,
    aruco_dict: ArucoDictType,
    charuco_board: CharucoBoardType,
    camera: RGBCamera,
    robot: robot,
    calibration_dir: Optional[str],
) -> None:
    """Run hand-eye calibration.

    Do hand-eye calibration, both eye-in-hand and eye-to-hand are supported.

    Parameters
    ----------
    mode : str
        Calibration mode, either ``"eye_in_hand"`` or ``"eye_to_hand"``.
    aruco_dict : ArucoDictType
        ArUco dictionary used for the ChArUco board.
    charuco_board : CharucoBoardType
        ChArUco board used for calibration.
    camera : RGBCamera
        Camera used to collect the calibration data.
    robot : robot
        Robot used to collect the calibration data.
    calibration_dir : str, optional
        Directory in which calibration samples and results are saved. If
        ``None``, a new directory is created automatically.
    """

    data_dir = create_data_dir(calibration_dir)
    calibration_dir = os.path.dirname(data_dir)  # TODO clean this up

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution

    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    window_name = f"{mode} calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    robot.teachMode()

    MIN_POSES = 3
    tcp_poses_in_base: List[HomogeneousMatrixType] = []
    images: List[OpenCVIntImageType] = []
    camera_pose_best = None

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_visualize_charuco_pose(image, intrinsics, aruco_dict, charuco_board)
        tcp_pose = robot.GetPose()  # TODO: v katerem task_space
        draw_base_pose_on_image(image, intrinsics, camera_pose_best, mode, tcp_pose)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.EndTeachMode()
            break

        if key == ord("s"):
            # TODO reject samples where no board was detected?
            sample_index = len(tcp_poses_in_base)
            tcp_pose, image_bgr = save_calibration_sample(sample_index, robot, camera, data_dir)
            logger.info(f"Saved {sample_index + 1} sample(s).")

            tcp_poses_in_base.append(tcp_pose)
            images.append(image_bgr)

            n_samples = len(tcp_poses_in_base)
            if n_samples < MIN_POSES:
                continue

            # The the calibration with the new set of samples
            results_dir = os.path.join(calibration_dir, f"results_n={n_samples}")
            os.makedirs(results_dir)
            logger.info(f"Running calibration with {n_samples} (image, tcp_pose) pairs")
            logger.info(f"Saving calibration results to {results_dir}")
            poses_dict, errors_dict = extrinsic_calibration_all_methods(results_dir, images, tcp_poses_in_base, intrinsics, mode, aruco_dict, charuco_board)

            min_error_key = min(errors_dict, key=lambda x: errors_dict.get(x) or float("inf"))
            camera_pose_best = poses_dict[min_error_key]
