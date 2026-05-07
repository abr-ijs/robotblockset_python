"""MuJoCo camera backend.

This module defines a MuJoCo-backed RGB camera wrapper that implements the shared camera interface.
It provides image capture from MuJoCo scene cameras, configurable resolution and frame-rate settings,
and optional intrinsics handling for simulation-based perception workflows. The implementation enables
simulated camera streams to be used interchangeably with hardware camera backends.

Key functionalities include:
- MuJoCo scene camera integration through the ``RGBCamera`` interface.
- Configurable camera selection, resolution, and frame-rate parameters.
- RGB frame capture and retrieval in float and uint8 formats.
- Optional camera intrinsics matrix support for simulated calibration compatibility.
- Context-manager hooks for integration in resource-managed workflows.
- Standardized behavior aligned with other robotblockset camera backends.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from robotblockset.cameras.interfaces import RGBCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.rbs_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType


class MujocoCam(RGBCamera):
    """
    MuJoCo camera wrapper implementing the RGB camera interface.
    """

    # Some standard resolutions that are likely to be supported by webcams.
    # 16:9
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    # 4:3
    RESOLUTION_960 = (1280, 960)
    RESOLUTION_768 = (1024, 768)
    RESOLUTION_480 = (640, 480)

    def __init__(
        self,
        scene: Any,
        mujoco_camera_id: int = 0,
        scene_option: Optional[Any] = None,
        intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None,
        resolution: CameraResolutionType = RESOLUTION_480,
        fps: int = 30,
    ) -> None:
        """
        Create a MuJoCo camera wrapper.

        Parameters
        ----------
        scene : Any
            MuJoCo scene handle.
        mujoco_camera_id : int, optional
            int, default=0
            Index of the camera in the MuJoCo model.
        scene_option : Any, optional
            Optional MuJoCo scene options for rendering.
        intrinsics_matrix : CameraIntrinsicsMatrixType, optional
            Camera intrinsics matrix with shape ``(3, 3)``.
        resolution : CameraResolutionType, optional
            CameraResolutionType, default=RESOLUTION_480
            Image resolution ``(width, height)`` in pixels.
        fps : int, optional
            int, default=30
            Target frame rate.
        """
        self.scene = scene
        self.cam_id = mujoco_camera_id
        self.name = scene.model.camera(mujoco_camera_id).name
        self.scene_option = scene_option
        self._resolution = resolution
        if intrinsics_matrix is None:
            self._intrinsics_matrix = np.eye(3)
        else:
            self._intrinsics_matrix = intrinsics_matrix
        self._fps = fps

    @property
    def fps(self) -> int:
        """The frames per second of the camera."""
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        """The resolution of the camera, in pixels."""
        return self._resolution

    def __enter__(self) -> RGBCamera:
        """
        Enter context manager.

        Returns
        -------
        RGBCamera
            The current camera instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit context manager.

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
        pass

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        """
        Obtain the camera intrinsics matrix.

        Returns
        -------
        CameraIntrinsicsMatrixType
            Camera intrinsics matrix with shape ``(3, 3)``.

        Raises
        ------
        RuntimeError
            If intrinsics were not provided.
        """
        if self._intrinsics_matrix is None:
            raise RuntimeError("OpenCVVideoCapture does not have a preset intrinsics matrix. Pass it to the constructor if you know it.")
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        """Capture the latest RGB image from the MuJoCo scene."""
        image = self.scene.mj_capture_camera(self.cam_id, width=self.resolution[0], height=self.resolution[1], scene_option=self.scene_option)
        self._frame = image

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        """Return the current RGB image as float image."""
        return ImageConverter.from_numpy_int_format(self._frame).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """Return the current RGB image as uint8 image."""
        return self._frame
