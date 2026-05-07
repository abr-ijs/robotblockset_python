"""RealSense RGB-D camera backend.

This module defines a RealSense RGB-D camera wrapper built on ``pyrealsense2``.
It provides synchronized color and depth acquisition, optional point-cloud and confidence-map generation,
camera intrinsics access, and post-processing utilities such as depth alignment and hole filling.
The implementation exposes RealSense devices through the shared robotblockset RGB-D camera interface
for consistent use in perception and calibration pipelines.

Key functionalities include:
- RealSense device integration via the ``RGBDCamera`` interface.
- Configurable color/depth streaming resolution and frame rate.
- Synchronized RGB, depth, and optional confidence-map acquisition.
- Optional point-cloud generation from aligned depth and color data.
- Camera intrinsics retrieval from active device stream profiles.
- Depth post-processing support, including alignment and hole filling.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np
from robotblockset.cameras.interfaces import RGBDCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.rbs_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)

import pyrealsense2 as rs  # type: ignore

# from loguru import logger
from robotblockset.tools import get_logger

logger = get_logger(__name__)


class Realsense(RGBDCamera):
    """Wrapper around the pyrealsense2 library to use the RealSense cameras (tested for the D415 and D435).

    Design decisions we made for this class:
    * Depth and color fps are the same
    * Depth resolution is automatically set
    * Depth frames are always aligned to color frames
    * Hole filling is enabled by default
    * Getting the confidence map is optional (disabled by default). The L515 supports a built-in confidence map, but the D415 and D435 do not. We do not support the L515 at this time.
    """

    # Built-in resolutions (16:9 aspect ratio) for convenience
    # for all resolutions see: realsense_scan_profiles.py
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    RESOLUTION_540 = (960, 540)
    RESOLUTION_480 = (848, 480)

    def __init__(
        self,
        resolution: CameraResolutionType = RESOLUTION_1080,
        fps: int = 30,
        enable_depth: bool = True,
        enable_pointcloud: bool = True,
        enable_confidence_map: bool = False,
        enable_hole_filling: bool = True,
        serial_number: Optional[str] = None,
    ) -> None:
        """
        Initialize a RealSense RGB-D camera wrapper.

        Parameters
        ----------
        resolution : CameraResolutionType, optional
            CameraResolutionType, default=RESOLUTION_1080
            Color stream resolution ``(width, height)`` in pixels.
        fps : int, optional
            int, default=30
            Target frame rate for color and depth streams.
        enable_depth : bool, optional
            bool, default=True
            Enable depth acquisition.
        enable_pointcloud : bool, optional
            bool, default=True
            Enable point-cloud computation from depth and color.
        enable_confidence_map : bool, optional
            bool, default=False
            Enable confidence map computation based on stereo infrared frames.
        enable_hole_filling : bool, optional
            bool, default=True
            Enable RealSense hole-filling post-processing on depth frames.
        serial_number : str, optional
            Device serial number. If omitted, the default device is used.

        Raises
        ------
        ValueError
            If point cloud or confidence map is enabled while depth is disabled.
        RuntimeError
            If pipeline startup fails due to invalid configuration or device
            issues.
        """
        self._resolution = resolution
        self._fps = fps
        self._depth_enabled = enable_depth
        self._pointcloud_enabled = enable_pointcloud
        if self._pointcloud_enabled and not self._depth_enabled:
            raise ValueError("enable_point_cloud can only be True if enable_depth is also True")
        self._confidence_enabled = enable_confidence_map
        if self._confidence_enabled and not self._depth_enabled:
            raise ValueError("enable_confidence_map can only be True if enable_depth is also True")
        self.hole_filling_enabled = enable_hole_filling
        self.serial_number = serial_number

        config = rs.config()

        if serial_number is not None:
            # Note: Invalid serial_number leads to RuntimeError for pipeline.start(config)
            config.enable_device(serial_number)

        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)

        # Use max resolution that can handle the fps for depth (will be change by align_transform)
        depth_resolution = Realsense.RESOLUTION_720 if fps <= 30 else Realsense.RESOLUTION_480
        if self._depth_enabled:
            config.enable_stream(
                rs.stream.depth,
                depth_resolution[0],
                depth_resolution[1],
                rs.format.z16,
                fps,
            )

        if self._confidence_enabled:
            config.enable_stream(
                rs.stream.infrared,
                1,
                depth_resolution[0],
                depth_resolution[1],
                rs.format.y8,
                fps,
            )
            config.enable_stream(
                rs.stream.infrared,
                2,
                depth_resolution[0],
                depth_resolution[1],
                rs.format.y8,
                fps,
            )

        # Avoid having to reconnect the USB cable, see https://github.com/IntelRealSense/librealsense/issues/6628#issuecomment-646558144
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        self.pipeline = rs.pipeline()

        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        device = profile.get_device()
        self.Name = device.get_info(rs.camera_info.name).replace(" ", "_")

        # Get intrinsics matrix
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self._intrinsics_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        if self._depth_enabled:
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_factor = depth_sensor.get_depth_scale()
            self._setup_depth_transforms()
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.color_scheme, 2)  # 2 = White to Black

    def __enter__(self) -> RGBDCamera:
        """
        Enter context manager.

        Returns
        -------
        RGBDCamera
            The current camera instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit context manager and stop the RealSense pipeline.

        Parameters
        ----------
        exc_type : Any
            Exception type raised inside the context, if any.
        exc_value : Any
            Exception instance raised inside the context, if any.
        traceback : Any
            Exception traceback, if any.

        Returns
        -------
        None
        """
        self.pipeline.stop()

    def _setup_depth_transforms(self) -> None:
        """
        Initialize depth-alignment and depth-filter transforms.

        Returns
        -------
        None
        """
        # Configure depth filters and transfrom, adapted from:
        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        # https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
        self.align_transform = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter()

    @property
    def fps(self) -> int:
        """
        Return the configured camera frame rate.

        Returns
        -------
        int
            Frame rate in frames per second.
        """
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        """
        Return the configured image resolution.

        Returns
        -------
        CameraResolutionType
            Resolution ``(width, height)`` in pixels.
        """
        return self._resolution

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        """
        Return the color camera intrinsics matrix.

        Returns
        -------
        CameraIntrinsicsMatrixType
            Camera matrix with shape ``(3, 3)``.
        """
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        """
        Grab and cache the latest composite frame and derived depth data.

        Returns
        -------
        None
        """
        self._composite_frame = self.pipeline.wait_for_frames()

        if not self._depth_enabled:
            return

        try:
            aligned_frames = self.align_transform.process(self._composite_frame)
        except RuntimeError as e:
            # Sometimes, the realsense SDK throws an error with aligning RGB and depth.
            # This can happen if the CPU is busy: https://github.com/IntelRealSense/librealsense/issues/6628#issuecomment-647379900
            # A solution is to try again. Here, we only try again once; if the error occurs again, we raise it
            # and let the user deal with it.
            logger.error(f"Error while grabbing images:\n{e}.\nWill retry in 1 second.")
            time.sleep(1)
            aligned_frames = self.align_transform.process(self._composite_frame)

        self._depth_frame = aligned_frames.get_depth_frame()

        if self.hole_filling_enabled:
            self._depth_frame = self.hole_filling.process(self._depth_frame)

        # Compute point cloud.
        if self._pointcloud_enabled:
            self._point_cloud = self._compute_point_cloud(aligned_frames)

        if not self._confidence_enabled:
            return

        self._infrared_frame_1 = aligned_frames.get_infrared_frame(1)
        self._infrared_frame_2 = aligned_frames.get_infrared_frame(2)

    def _compute_point_cloud(self, aligned_frames: Any) -> PointCloud:
        """
        Compute a colored point cloud from aligned depth and color frames.

        Parameters
        ----------
        aligned_frames : Any
            Composite frame with depth aligned to color.

        Returns
        -------
        PointCloud
            Point cloud with 3D positions and RGB colors.
        """
        pcd = rs.pointcloud()
        color_frame = aligned_frames.get_color_frame()
        pcd.map_to(color_frame)
        points = pcd.calculate(self._depth_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        tex_coords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
        color_image: NumpyIntImageType = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape
        # tex_coords shape is (N, 2)
        u = tex_coords[:, 0]
        v = tex_coords[:, 1]
        # convert to pixel coordinates
        x = np.clip((u * w).astype(np.int32), 0, w - 1)
        y = np.clip((v * h).astype(np.int32), 0, h - 1)
        # index color_image
        colors = color_image[y, x]  # shape (N, 3)
        return PointCloud(vertices, colors)

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        """
        Return the latest RGB frame as normalized float image.

        Returns
        -------
        NumpyFloatImageType
            RGB image in floating-point representation.
        """
        image = self._retrieve_rgb_image_as_int()
        return ImageConverter.from_numpy_int_format(image).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """
        Return the latest RGB frame as integer image.

        Returns
        -------
        NumpyIntImageType
            RGB image in integer representation.

        Raises
        ------
        RuntimeError
            If no frame has been grabbed yet.
        """
        if not isinstance(self._composite_frame, rs.composite_frame):
            raise RuntimeError("_grab_images must be called before retrieving images")
        color_frame = self._composite_frame.get_color_frame()
        image: NumpyIntImageType = np.asanyarray(color_frame.get_data())
        return image

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        """
        Return the latest metric depth map.

        Returns
        -------
        NumpyDepthMapType
            Depth map in meters.

        Raises
        ------
        RuntimeError
            If depth is disabled.
        """
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        frame = self._depth_frame
        image = np.asanyarray(frame.get_data()).astype(np.float32)
        return image * self.depth_factor

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        """
        Return the latest colorized depth image.

        Returns
        -------
        NumpyIntImageType
            Colorized depth image.

        Raises
        ------
        RuntimeError
            If depth is disabled.
        """
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        frame = self._depth_frame
        frame_colorized = self.colorizer.colorize(frame)
        image = np.asanyarray(frame_colorized.get_data())  # this is uint8 with 3 channels
        return image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        """
        Return the latest computed colored point cloud.

        Returns
        -------
        PointCloud
            Colored point cloud.

        Raises
        ------
        RuntimeError
            If point-cloud computation is disabled.
        """
        if not self._pointcloud_enabled:
            raise RuntimeError("Cannot retrieve point cloud if point cloud is disabled")
        return self._point_cloud

    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        """
        Return a confidence map derived from stereo disparity filtering.

        Returns
        -------
        NumpyConfidenceMapType
            Confidence map normalized to ``[0, 1]``.

        Raises
        ------
        RuntimeError
            If confidence retrieval is disabled or no frame is available.
        """
        # Compute confidence map based on the disparity between the two IR images.
        if not self._confidence_enabled:
            raise RuntimeError("Cannot retrieve confidence data if confidence is disabled")
        if not isinstance(self._composite_frame, rs.composite_frame):
            raise RuntimeError("_grab_images must be called before retrieving images")
        ir1_frame = self._infrared_frame_1
        ir2_frame = self._infrared_frame_2

        # Convert images to numpy
        ir1 = np.asanyarray(ir1_frame.get_data())
        ir2 = np.asanyarray(ir2_frame.get_data())

        # default values for SGBM according to OpenCV docs
        wls_lambda = 8000.0
        wls_sigma = 1.5
        if not hasattr(self, "_stereo_sgbm"):
            max_disp = 160  # must be divisible by 16
            window_size = 3
            p1 = 216  # 24 * window_size ** 2
            p2 = 864  # 96 * window_size ** 2
            pre_filter_cap = 63

            self._stereo_sgbm = cv2.StereoSGBM.create(
                minDisparity=0,
                numDisparities=max_disp,
                blockSize=window_size,
                P1=p1,
                P2=p2,
                preFilterCap=pre_filter_cap,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
            )

        left_matcher = self._stereo_sgbm
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        left_disp = left_matcher.compute(ir1, ir2).astype(np.float32) / 16.0
        right_disp = right_matcher.compute(ir2, ir1).astype(np.float32) / 16.0
        wls_filter.setLambda(wls_lambda)
        wls_filter.setSigmaColor(wls_sigma)
        wls_filter.filter(left_disp, ir1, disparity_map_right=right_disp)
        confidence_map = wls_filter.getConfidenceMap()

        return confidence_map / 255.0


if __name__ == "__main__":
    import robotblockset.cameras.manual_test_hw as test

    camera = Realsense(fps=30, resolution=Realsense.RESOLUTION_1080, enable_hole_filling=True)

    # Perform tests
    test.manual_test_camera(camera)
    test.manual_test_rgb_camera(camera)
    test.manual_test_depth_camera(camera)
    test.profile_rgb_throughput(camera)
    test.profile_rgbd_throughput(camera)

    # Live viewer
    cv2.namedWindow("RealSense RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Map", cv2.WINDOW_NORMAL)

    while True:
        color_image = camera.get_rgb_image_as_int()
        color_image = ImageConverter.from_numpy_int_format(color_image).image_in_opencv_format
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()

        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("RealSense Depth Image", depth_image)
        cv2.imshow("RealSense Depth Map", depth_map)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
