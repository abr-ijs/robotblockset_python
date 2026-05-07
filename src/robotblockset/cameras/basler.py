"""Basler camera backend.

This module defines a Basler camera backend using the pypylon SDK for the shared RGB camera interface.
It provides camera discovery utilities and a concrete ``RGBCamera`` implementation with configurable
exposure/gain behavior, optional camera intrinsics handling, and low-latency image acquisition.
The module enables Basler hardware to integrate seamlessly into robotblockset camera and calibration workflows.

Key functionalities include:
- Basler device discovery via transport-layer enumeration.
- Concrete ``RGBCamera`` implementation for Basler RGB streams.
- Camera selection by serial number and connection lifecycle management.
- Configurable manual/auto exposure and gain controls.
- RGB image retrieval in uint8 and float formats with low-latency grabbing strategy.
- Optional intrinsics matrix support for calibration-aware pipelines.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

import numpy as np
from robotblockset.rbs_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType

# Basler SDK (pypylon)
try:
    from pypylon import pylon  # pyright: ignore[reportMissingImports]
except Exception as e:  # pragma: no cover
    pylon = None
    _PYPYLON_IMPORT_ERROR = e


from robotblockset.cameras.interfaces import RGBCamera


def list_basler_cameras() -> None:
    """
    Print discovered Basler cameras.

    This helper enumerates transport-layer devices visible to pypylon and
    prints each camera IP address and serial number.

    Returns
    -------
    None
    """
    factory = pylon.TlFactory.GetInstance()
    for d in factory.EnumerateDevices():
        print(f"Camera: IP={d.GetIpAddress()}  serial_number={d.GetSerialNumber()}")


class BaslerRGBCamera(RGBCamera):
    """
    Basler RGB camera implementation compatible with the RGBCamera interface.

    Notes
    -----
    Requires ``pypylon`` (Basler pylon SDK Python bindings). Returns RGB
    images ``(H, W, 3)`` as ``uint8`` or ``float32`` in ``[0, 1]`` and uses
    ``GrabStrategy_LatestImageOnly`` to keep latency low.
    """

    Name: str = "BaslerRGBCamera"

    def __init__(
        self,
        serial_number: Optional[str] = None,
        *,
        intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None,
        exposure_time_us: Optional[float] = None,
        gain: Optional[float] = None,
        auto_exposure: Optional[bool] = None,
        auto_gain: Optional[bool] = None,
        frame_timeout_ms: int = 2000,
    ) -> None:
        """
        Initialize a Basler RGB camera wrapper.

        Parameters
        ----------
        serial_number : str, optional
            Target camera serial number. If omitted, the first discovered camera
            is used.
        intrinsics_matrix : CameraIntrinsicsMatrixType, optional
            Camera intrinsics matrix with shape ``(3, 3)``.
        exposure_time_us : float, optional
            Manual exposure time in microseconds.
        gain : float, optional
            Manual gain value.
        auto_exposure : bool, optional
            Enable/disable continuous exposure auto mode.
        auto_gain : bool, optional
            Enable/disable continuous gain auto mode.
        frame_timeout_ms : int, optional
            int, default=2000
            Timeout for frame acquisition in milliseconds.

        Raises
        ------
        ValueError
            If ``intrinsics_matrix`` does not have shape ``(3, 3)``.
        """
        super().__init__()
        self._serial_number = serial_number
        self._intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None
        if intrinsics_matrix is not None:
            intrinsics_array = np.asarray(intrinsics_matrix, dtype=np.float64)
            if intrinsics_array.shape != (3, 3):
                raise ValueError(f"intrinsics_matrix must have shape (3, 3), got {intrinsics_array.shape}")
            self._intrinsics_matrix = intrinsics_array
        self._exposure_time_us = exposure_time_us
        self._gain = gain
        self._auto_exposure = auto_exposure
        self._auto_gain = auto_gain
        self._frame_timeout_ms = int(frame_timeout_ms)

        self._camera: Optional["pylon.InstantCamera"] = None  # pyright: ignore[reportInvalidTypeForm]
        self._converter: Optional["pylon.ImageFormatConverter"] = None  # pyright: ignore[reportInvalidTypeForm]

        self._latest_rgb_uint8: Optional[NumpyIntImageType] = None  # (H, W, 3) RGB
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def connect(self) -> None:
        """
        Connect to the camera and start grabbing.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If pypylon is not available.
        RuntimeError
            If the configured camera serial number cannot be found.
        """
        if pylon is None:  # pragma: no cover
            raise ImportError("pypylon is not available. Install pypylon and the Basler pylon SDK.") from _PYPYLON_IMPORT_ERROR

        if self._camera is not None:
            return  # already connected

        factory = pylon.TlFactory.GetInstance()

        if self._serial_number:
            devices = factory.EnumerateDevices()
            match = None
            for d in devices:
                if d.GetSerialNumber() == self._serial_number:
                    match = d
                    break
            if match is None:
                raise RuntimeError(f"Basler camera with serial '{self._serial_number}' not found.")
            device = match
        else:
            device = factory.CreateFirstDevice()

        cam = pylon.InstantCamera(factory.CreateDevice(device))
        cam.Open()

        # Optional configuration
        self._apply_configuration(cam)

        # Converter to RGB8 packed
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Cache dimensions after opening/config
        try:
            self._width = int(cam.Width.Value)
            self._height = int(cam.Height.Value)
        except Exception:
            self._width = None
            self._height = None

        # Start grabbing
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self._camera = cam
        self._converter = converter

    def disconnect(self) -> None:
        """
        Stop grabbing and close the camera.

        Returns
        -------
        None
        """
        if self._camera is None:
            return
        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        finally:
            try:
                if self._camera.IsOpen():
                    self._camera.Close()
            finally:
                self._camera = None
                self._converter = None
                self._latest_rgb_uint8 = None

    def __enter__(self) -> "BaslerRGBCamera":
        """
        Enter context manager and connect the camera.

        Returns
        -------
        BaslerRGBCamera
            Connected camera instance.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """
        Exit context manager and disconnect the camera.

        Parameters
        ----------
        exc_type : Type[BaseException], optional
            Exception type raised inside the context.
        exc : BaseException, optional
            Exception instance raised inside the context.
        tb : TracebackType, optional
            Traceback associated with the exception.

        Returns
        -------
        None
        """
        self.disconnect()

    @property
    def fps(self) -> float:
        """
        Return the configured acquisition frame rate.

        Returns
        -------
        float
            Frame rate in Hz when available; ``nan`` if the camera does not
            expose the relevant node.

        Raises
        ------
        RuntimeError
            If the camera is not connected.
        """
        cam = self._camera
        if cam is None or not cam.IsOpen():
            raise RuntimeError("Camera not connected; fps is unavailable.")

        # Many Basler cameras expose AcquisitionFrameRate and enable flags
        for name in ("AcquisitionFrameRateAbs", "AcquisitionFrameRate"):
            try:
                node = getattr(cam, name)
                return float(node.Value)
            except Exception:
                continue

        # If not available, return NaN instead of guessing
        return float("nan")

    @property
    def resolution(self) -> CameraResolutionType:
        """
        Return camera resolution in pixels.

        Returns
        -------
        CameraResolutionType
            ``(width, height)`` in pixels.

        Raises
        ------
        RuntimeError
            If resolution is unavailable because the camera is not connected and
            no cached frame size exists.
        """
        if self._camera is not None and self._camera.IsOpen():
            try:
                return int(self._camera.Width.Value), int(self._camera.Height.Value)
            except Exception:
                pass
        # Fallback to cached values if available
        if self._width is not None and self._height is not None:
            return self._width, self._height
        raise RuntimeError("Camera not connected; resolution is unknown.")

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        """
        Return the camera intrinsics matrix.

        Returns
        -------
        CameraIntrinsicsMatrixType
            Intrinsics matrix with shape ``(3, 3)`` in pixel units.

        Raises
        ------
        RuntimeError
            If intrinsics were not provided at construction.
        """
        if self._intrinsics_matrix is None:
            raise RuntimeError("No intrinsics matrix provided. Pass intrinsics_matrix=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]) " "from your calibration results.")
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        """
        Fetch the latest frame into the internal RGB buffer.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the camera is not connected, not grabbing, grabbing fails, or the
            grabbed image has an unexpected shape.
        """
        cam = self._camera
        conv = self._converter
        if cam is None or conv is None:
            raise RuntimeError("Camera not connected. Call connect() first.")

        if not cam.IsGrabbing():
            raise RuntimeError("Camera is connected but not grabbing. Call connect() again.")

        grab = cam.RetrieveResult(self._frame_timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                raise RuntimeError(f"Grab failed: {grab.ErrorCode} {grab.ErrorDescription}")

            img = conv.Convert(grab)
            arr = img.GetArray()  # usually HxWx3 uint8
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise RuntimeError(f"Unexpected image shape from Basler: {arr.shape}")

            # Ensure contiguous and owned buffer (safe after grab is released)
            self._latest_rgb_uint8 = np.ascontiguousarray(arr, dtype=np.uint8)

            # Cache dimensions
            self._height, self._width = self._latest_rgb_uint8.shape[:2]
        finally:
            grab.Release()

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """
        Return the latest RGB frame as ``uint8``.

        Returns
        -------
        NumpyIntImageType
            RGB image with shape ``(H, W, 3)`` and ``uint8`` dtype.

        Raises
        ------
        RuntimeError
            If no frame is currently available.
        """
        if self._latest_rgb_uint8 is None:
            raise RuntimeError("No frame available yet. Call _grab_images() first.")
        return self._latest_rgb_uint8

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        """
        Return the latest RGB frame as normalized ``float32``.

        Returns
        -------
        NumpyFloatImageType
            RGB image with shape ``(H, W, 3)`` and values in ``[0, 1]``.
        """
        img_u8 = self._retrieve_rgb_image_as_int()
        return img_u8.astype(np.float32) / 255.0

    def _apply_configuration(self, cam: "pylon.InstantCamera") -> None:  # pyright: ignore[reportInvalidTypeForm]
        """
        Apply exposure and gain settings to a camera instance.

        Parameters
        ----------
        cam : 'pylon.InstantCamera'
            Open camera object to configure.

        Returns
        -------
        None
        """
        # Auto settings (names differ slightly across Basler models)
        if self._auto_exposure is not None:
            for node_name in ("ExposureAuto",):
                try:
                    node = getattr(cam, node_name)
                    node.Value = "Continuous" if self._auto_exposure else "Off"
                    break
                except Exception:
                    continue

        if self._auto_gain is not None:
            for node_name in ("GainAuto",):
                try:
                    node = getattr(cam, node_name)
                    node.Value = "Continuous" if self._auto_gain else "Off"
                    break
                except Exception:
                    continue

        # Manual exposure
        if self._exposure_time_us is not None:
            for node_name in ("ExposureTimeAbs", "ExposureTime"):
                try:
                    getattr(cam, node_name).Value = float(self._exposure_time_us)
                    break
                except Exception:
                    continue

        # Manual gain
        if self._gain is not None:
            for node_name in ("GainRaw", "Gain"):
                try:
                    getattr(cam, node_name).Value = float(self._gain)
                    break
                except Exception:
                    continue
