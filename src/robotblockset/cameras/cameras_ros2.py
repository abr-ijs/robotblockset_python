"""ROS 2 camera backends.

This module defines ROS2-based camera backends that implement the shared robotblockset camera interfaces.
It provides ROS2 node wrappers for RGB and depth camera streams, including subscription management, image conversion,
intrinsics extraction from camera info messages, optional point-cloud handling, and threaded executor control.
The module enables ROS2 camera topics to be consumed through a unified API consistent with other camera backends.

Key functionalities include:
- ROS2 ``Node``-based camera wrappers compatible with ``RGBCamera`` and depth interfaces.
- Subscription handling for RGB, depth, camera info, and optional point-cloud topics.
- Conversion of ROS image messages to standardized NumPy/OpenCV image formats.
- Intrinsics matrix and resolution extraction from ``CameraInfo`` messages.
- Threaded ROS2 spinning, synchronization, and graceful node shutdown management.
- Unified camera data access for downstream calibration and perception workflows.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any, Optional
from time import time
from threading import Event, Lock, Thread

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
except Exception as e:
    raise e from RuntimeError("ROS2 rclpy not installed.\nYou can install rclpy with commands:\n   sudo apt update\nsudo apt install ros-<ros-distro>-rclpy")

try:
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
except Exception as e:
    raise e from RuntimeError("Problems with importing ROS2 sensor_msgs messages. Check if all are installed.")

try:
    from cv_bridge import CvBridge
except Exception as e:
    raise e from RuntimeError("ROS2 cv_bridge not installed. Install it via your ROS2 distribution.")

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None

from robotblockset.cameras.interfaces import DepthCamera, RGBCamera
from robotblockset.cameras.image_converter import ImageConverter
from robotblockset.cameras.point_clouds import open3d_to_point_cloud
from robotblockset.rbs_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)


class camera_ros2(Node, RGBCamera):
    """
    Generic ROS2 RGB camera.

    Parameters
    ----------
    name : str
        Node name.
    namespace : str, optional
        ROS2 namespace.
    rgb_topic : str
        RGB image topic.
    camera_info_topic : str | None
        Camera info topic (for intrinsics). If None, intrinsics are unavailable.
    wait_for_first_frame : bool
        Whether to wait for the first RGB frame before returning from _grab_images.
    timeout_sec : float
        Timeout for waiting on the first RGB frame.
    auto_spin : bool
        Start spinning thread automatically.
    rgb_encoding : str | None
        Override encoding for RGB conversion.
    """

    Name = "camera_ros2"

    def __init__(
        self,
        name: str = "camera",
        namespace: str = "",
        rgb_topic: str = "image_raw",
        camera_info_topic: Optional[str] = "camera_info",
        wait_for_first_frame: bool = True,
        timeout_sec: float = 2.0,
        auto_spin: bool = True,
        rgb_encoding: Optional[str] = None,
    ) -> None:
        Node.__init__(self, name)

        if namespace is None or namespace == "":
            self._namespace = ""
        else:
            self._namespace = "/" + str(namespace).strip("/")

        self.Name = name

        self._bridge = CvBridge()
        self._lock = Lock()
        self._rgb_event = Event()

        self._rgb_image: Optional[NumpyIntImageType] = None
        self._frame_rgb: Optional[NumpyIntImageType] = None

        self._intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None
        self._resolution: Optional[CameraResolutionType] = None

        self._fps: Optional[float] = None
        self._last_rgb_time: Optional[float] = None

        self._wait_for_first_frame = wait_for_first_frame
        self._frame_timeout = timeout_sec
        self._rgb_encoding_override = rgb_encoding

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spinning = False
        self._spin_thread: Optional[Thread] = None

        rgb_topic_full = f"{self._namespace}/{rgb_topic.strip('/')}"
        self._rgb_sub = self.create_subscription(
            msg_type=Image,
            topic=rgb_topic_full,
            callback=self._rgb_callback,
            qos_profile=qos_profile_sensor_data,
        )

        if camera_info_topic is not None:
            info_topic_full = f"{self._namespace}/{camera_info_topic.strip('/')}"
            self._info_sub = self.create_subscription(
                msg_type=CameraInfo,
                topic=info_topic_full,
                callback=self._camera_info_callback,
                qos_profile=qos_profile_sensor_data,
            )

        if auto_spin:
            self._start_spinning(wait_for_frame=wait_for_first_frame, timeout_sec=timeout_sec)

    def __enter__(self) -> RGBCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._shutdown()

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Spinning
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while rclpy.ok() and getattr(self, "_spinning", False):
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            try:
                self.get_logger().error(f"Spin loop stopped: {e}")
            except Exception:
                pass

    def _start_spinning(self, wait_for_frame: bool = True, timeout_sec: float = 10.0) -> None:
        if self._spinning:
            return
        self._spinning = True
        self._spin_thread = Thread(target=self._run, args=(), kwargs={}, daemon=True)
        self._spin_thread.start()

        if wait_for_frame:
            if not self._rgb_event.wait(timeout=timeout_sec):
                self.get_logger().warning("No RGB frame received within timeout; continuing without confirmation.")

    def _shutdown(self, join_timeout: float = 2.0) -> None:
        try:
            self._spinning = False
            if self._spin_thread is not None and self._spin_thread.is_alive():
                self._spin_thread.join(timeout=join_timeout)
        except Exception:
            pass

        try:
            if hasattr(self, "_executor") and self._executor is not None:
                try:
                    self._executor.remove_node(self)
                except Exception:
                    pass
                self._executor.shutdown()
        except Exception:
            pass

        try:
            self.destroy_node()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if msg.k is not None and len(msg.k) == 9:
            k = np.array(msg.k, dtype=float).reshape((3, 3))
            self._intrinsics_matrix = k
        if msg.width > 0 and msg.height > 0:
            self._resolution = (int(msg.width), int(msg.height))

    def _rgb_callback(self, msg: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        encoding = self._rgb_encoding_override or msg.encoding
        rgb = self._convert_to_rgb(image, encoding)

        now = time()
        with self._lock:
            self._rgb_image = rgb
            if self._resolution is None:
                self._resolution = (rgb.shape[1], rgb.shape[0])
            if self._last_rgb_time is not None:
                dt = now - self._last_rgb_time
                if dt > 0:
                    fps = 1.0 / dt
                    self._fps = fps if self._fps is None else 0.9 * self._fps + 0.1 * fps
            self._last_rgb_time = now

        self._rgb_event.set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _convert_to_rgb(self, image: np.ndarray, encoding: Optional[str]) -> NumpyIntImageType:
        enc = (encoding or "").lower()
        if image.ndim == 2:
            if enc in ("mono16", "16uc1"):
                image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image

        if image.shape[2] == 4:
            if enc in ("rgba8",):
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        if enc in ("bgr8",):
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    # ------------------------------------------------------------------
    # Camera interface
    # ------------------------------------------------------------------
    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        if self._intrinsics_matrix is None:
            raise RuntimeError("No camera intrinsics received. Is the camera_info topic available?")
        return self._intrinsics_matrix

    @property
    def fps(self) -> float:
        return float(self._fps) if self._fps is not None else 0.0

    @property
    def resolution(self) -> CameraResolutionType:
        if self._resolution is None:
            raise RuntimeError("No resolution available yet. Wait for the first frame or provide camera_info.")
        return self._resolution

    def _grab_images(self) -> None:
        if self._wait_for_first_frame and self._rgb_image is None:
            if not self._rgb_event.wait(timeout=self._frame_timeout):
                raise RuntimeError("No RGB image received within timeout.")
        with self._lock:
            if self._rgb_image is None:
                raise RuntimeError("No RGB image received yet.")
            self._frame_rgb = self._rgb_image

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        if self._frame_rgb is None:
            raise RuntimeError("_grab_images must be called before retrieving images")
        return ImageConverter.from_numpy_int_format(self._frame_rgb).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        if self._frame_rgb is None:
            raise RuntimeError("_grab_images must be called before retrieving images")
        return self._frame_rgb


class Realsense(camera_ros2, DepthCamera):
    """
    ROS2 RGBD RealSense camera wrapper based on ROS2 topics.

    Parameters
    ----------
    depth_topic : str
        Depth image topic.
    pointcloud_topic : str | None
        Point cloud topic. If None, point clouds are computed from RGBD using Open3D.
    depth_scale : float | None
        Scale from depth units to meters for 16UC1 depth images. Defaults to 0.001.
    enable_depth : bool
        Enable depth subscription and depth retrieval.
    enable_pointcloud : bool
        Enable point cloud subscription if a topic is provided and sensor_msgs_py is available.
    """

    Name = "realsense_ros2"

    def __init__(
        self,
        name: str = "realsense",
        namespace: str = "",
        rgb_topic: str = "color/image_raw",
        camera_info_topic: Optional[str] = "color/camera_info",
        depth_topic: Optional[str] = "aligned_depth_to_color/image_raw",
        pointcloud_topic: Optional[str] = "depth/color/points",
        wait_for_first_frame: bool = True,
        timeout_sec: float = 2.0,
        auto_spin: bool = True,
        depth_scale: Optional[float] = None,
        enable_depth: bool = False,
        enable_pointcloud: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            rgb_topic=rgb_topic,
            camera_info_topic=camera_info_topic,
            wait_for_first_frame=wait_for_first_frame,
            timeout_sec=timeout_sec,
            auto_spin=False,
        )

        self._depth_event = Event()
        self._depth_map: Optional[NumpyDepthMapType] = None
        self._frame_depth: Optional[NumpyDepthMapType] = None
        self._depth_scale = depth_scale
        self._depth_enabled = enable_depth

        if enable_pointcloud and not self._depth_enabled:
            raise ValueError("enable_pointcloud can only be True if enable_depth is also True")

        self._pointcloud_enabled = enable_pointcloud and pointcloud_topic is not None and pc2 is not None
        self._point_cloud: Optional[PointCloud] = None
        self._frame_point_cloud: Optional[PointCloud] = None

        if self._depth_enabled and depth_topic is not None:
            depth_topic_full = f"{self._namespace}/{depth_topic.strip('/')}"
            self._depth_sub = self.create_subscription(
                msg_type=Image,
                topic=depth_topic_full,
                callback=self._depth_callback,
                qos_profile=qos_profile_sensor_data,
            )

        if self._pointcloud_enabled and pointcloud_topic is not None:
            pc_topic_full = f"{self._namespace}/{pointcloud_topic.strip('/')}"
            self._pc_sub = self.create_subscription(
                msg_type=PointCloud2,
                topic=pc_topic_full,
                callback=self._pointcloud_callback,
                qos_profile=qos_profile_sensor_data,
            )
        elif enable_pointcloud and pointcloud_topic is not None and pc2 is None:
            self.get_logger().warning("sensor_msgs_py not available; point cloud topic will be ignored.")

        if auto_spin:
            self._start_spinning(wait_for_frame=wait_for_first_frame, timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _depth_callback(self, msg: Image) -> None:
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_map = self._convert_depth_to_meters(depth, msg.encoding)
        with self._lock:
            self._depth_map = depth_map
        self._depth_event.set()

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        if pc2 is None:
            return
        try:
            points = np.array(
                list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)),
                dtype=np.float32,
            )
            if points.size == 0:
                return
            xyz = points[:, :3]
            rgb = points[:, 3].view(np.uint32)
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            colors = np.stack([r, g, b], axis=1).astype(np.uint8)
            with self._lock:
                self._point_cloud = PointCloud(xyz, colors)
        except Exception as e:
            self.get_logger().warning(f"Failed to parse point cloud: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _convert_depth_to_meters(self, depth: np.ndarray, encoding: str) -> NumpyDepthMapType:
        enc = (encoding or "").lower()
        if enc in ("16uc1", "mono16"):
            scale = self._depth_scale if self._depth_scale is not None else 0.001
            return depth.astype(np.float32) * float(scale)
        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Camera interface
    # ------------------------------------------------------------------
    def _grab_images(self) -> None:
        if self._wait_for_first_frame and self._rgb_image is None:
            if not self._rgb_event.wait(timeout=self._frame_timeout):
                raise RuntimeError("No RGB image received within timeout.")
        if self._depth_enabled and self._wait_for_first_frame and self._depth_map is None:
            if not self._depth_event.wait(timeout=self._frame_timeout):
                raise RuntimeError("No depth image received within timeout.")

        with self._lock:
            if self._rgb_image is None:
                raise RuntimeError("No RGB image received yet.")
            self._frame_rgb = self._rgb_image
            if self._depth_enabled:
                if self._depth_map is None:
                    raise RuntimeError("No depth image received yet.")
                self._frame_depth = self._depth_map
            else:
                self._frame_depth = None
            if self._pointcloud_enabled:
                self._frame_point_cloud = self._point_cloud

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        if self._frame_depth is None:
            raise RuntimeError("_grab_images must be called before retrieving depth data")
        return self._frame_depth

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        if self._frame_depth is None:
            raise RuntimeError("_grab_images must be called before retrieving depth data")
        depth = self._frame_depth
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        return depth_color

    def get_colored_point_cloud(self) -> PointCloud:
        self._grab_images()
        return self._retrieve_colored_point_cloud()

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        if not self._pointcloud_enabled:
            raise RuntimeError("Cannot retrieve point cloud if point cloud is disabled")
        if self._pointcloud_enabled and self._frame_point_cloud is not None:
            return self._frame_point_cloud

        # Fallback to Open3D-based computation (slow)
        import open3d as o3d

        if self._frame_rgb is None or self._frame_depth is None:
            raise RuntimeError("_grab_images must be called before retrieving point cloud")

        image_rgb_uint8 = self._frame_rgb
        depth_map = self._frame_depth
        intrinsics = self.intrinsics_matrix()

        image_o3d = o3d.t.geometry.Image(image_rgb_uint8)
        depth_map_o3d = o3d.t.geometry.Image(depth_map)
        rgbd_o3d = o3d.t.geometry.RGBDImage(image_o3d, depth_map_o3d)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d,
            intrinsics,
            depth_scale=1.0,
            depth_max=1000.0,
        )

        return open3d_to_point_cloud(pcd)
