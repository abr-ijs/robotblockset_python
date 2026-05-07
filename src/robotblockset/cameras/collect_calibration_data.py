"""Calibration data collection utilities.

This module defines utilities for collecting camera-robot calibration datasets.
It provides helpers for preparing calibration directories, acquiring synchronized image and robot TCP pose samples,
saving calibration artifacts, and running interactive/manual collection loops with live board detection feedback.
The module is designed to support reproducible hand-eye and camera calibration workflows.

Key functionalities include:
- Creation and validation of calibration data directories.
- Synchronized capture of camera images and robot TCP poses.
- Persistent storage of image samples and pose metadata.
- Camera intrinsics export for calibration session reproducibility.
- Interactive and manual data collection loops.
- Live ChArUco detection visualization during sample acquisition.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import datetime
import json
import os
from typing import Optional, Tuple

import cv2
from robotblockset.cameras.interfaces import RGBCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.cameras.camera_calibration import CameraIntrinsics, save_pose_to_json, detect_and_visualize_charuco_pose

from robotblockset.transformations import map_pose
from robotblockset.robots import robot
from robotblockset.rbs_typing import HomogeneousMatrixType, OpenCVIntImageType
from robotblockset.tools import get_logger

logger = get_logger(__name__)


def create_data_dir(calibration_dir: Optional[str] = None) -> str:
    """
    Create an empty calibration data directory.

    Ensures that ``calibration_dir`` exists and has a ``data`` subfolder where
    new calibration samples can be stored.

    Parameters
    ----------
    calibration_dir : str, optional
        Calibration directory to use. If ``None``, a new directory is created
        in the current working directory.

    Returns
    -------
    str
        Path to the ``data`` subfolder of the calibration directory.
    """
    if calibration_dir is None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        calibration_dir = os.path.join(os.getcwd(), f"calibration_{datetime_str}")

    os.makedirs(calibration_dir, exist_ok=True)

    data_dir = os.path.join(calibration_dir, "data")

    # If data_dir already exists, check whether it is empty
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) != 0:
        logger.warning(f"The data subfolder of {calibration_dir} already exists and is not empty.")

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def save_calibration_sample(sample_index: int, robot: robot, camera: RGBCamera, data_dir: str) -> Tuple[HomogeneousMatrixType, OpenCVIntImageType]:
    """
    Save one calibration sample.

    A calibration sample consists of an image and the robot TCP pose.

    Parameters
    ----------
    sample_index : int
        Sample index used in the saved file names.
    robot : robot
        Robot used to collect the data.
    camera : RGBCamera
        Camera used to collect the data.
    data_dir : str
        Directory where the sample is saved.

    Returns
    -------
    Tuple[HomogeneousMatrixType, OpenCVIntImageType]
        Saved TCP pose and captured image.
    """
    # Robot has to be completely still at moment of the image capture

    ROBOT_STOP_WAIT_TIME = 0.5
    robot.Wait(ROBOT_STOP_WAIT_TIME)

    image_rgb = camera.get_rgb_image_as_int()
    image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

    tcp_pose = robot.GetPose()  # TODO: v katerem task_space

    suffix = f"{sample_index:04d}"
    image_filename = f"image_{suffix}.png"
    tcp_pose_filename = f"tcp_pose_{suffix}.json"
    image_filepath = os.path.join(data_dir, image_filename)
    tcp_pose_filepath = os.path.join(data_dir, tcp_pose_filename)

    cv2.imwrite(image_filepath, image_bgr)

    pose = map_pose(x=tcp_pose, out="T")
    save_pose_to_json(tcp_pose_filepath, pose)

    return pose, image_bgr


def prepare_collect_calibration_data(camera: RGBCamera, calibration_dir: Optional[str]) -> None:
    """
    Prepare hand-eye calibration data collection.

    Parameters
    ----------
    camera : RGBCamera
        Camera to use for collecting the data.
    calibration_dir : str, optional
        Directory where calibration data will be stored. If ``None``, a new
        directory is created.
    """
    # from loguru import logger

    data_dir = create_data_dir(calibration_dir)

    logger.info(f"Saving calibration data to {data_dir}")
    resolution = camera.resolution
    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    return data_dir


def manually_collect_calibration_data(robot: robot, camera: RGBCamera, calibration_dir: Optional[str]) -> None:
    """Collect calibration data samples for hand-eye calibration.

    Parameters
    ----------
        robot : robot
            the robot to use for collecting the data.
        camera : RGBCamera
            the camera to use for collecting the data.
        calibration_dir : str, optional
            directory to save the calibration data to, if None a directory will be created
    """
    # from loguru import logger

    data_dir = create_data_dir(calibration_dir)

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution
    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    window_name = "Calibration data collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # For now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
    robot.SetTeachMode()  # type: ignore
    sample_index = 0

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_visualize_charuco_pose(image, intrinsics)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.EndTeachMode()  # type: ignore
            break

        if key == ord("s"):
            save_calibration_sample(sample_index, robot, camera, data_dir)  # type: ignore
            sample_index += 1
            logger.info(f"Saved {sample_index} sample(s).")
