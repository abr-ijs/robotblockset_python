"""ROS 2 helper utilities.

This module defines shared ROS 2 helper utilities used by RobotBlockSet ROS 2
interfaces. It currently provides a lightweight TF client for querying frame
transforms and returning them as pose vectors.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah, Mihael Simonic.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Optional
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
import numpy as np


class TFClient:
    """Helper to encapsulate TF2 Buffer/Listener and provide pose queries."""

    def __init__(self, node: Node, base_frame: Optional[str] = "world", ee_frame: Optional[str] = "ee_frame") -> None:
        """
        Initialize a TF2 client for pose lookups.

        Parameters
        ----------
        node : Node
            ROS 2 node used to create the TF buffer and listener.
        base_frame : str, optional
            Frame to use as the base (default "world").
        ee_frame : str, optional
            Frame to use as the end-effector (default "ee_frame").
        """
        self._node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)
        self.base_frame = base_frame
        self.ee_frame = ee_frame

    def get_pose(self, base_frame: Optional[str] = None, ee_frame: Optional[str] = None, timeout: float = 2.0) -> Optional[np.ndarray]:
        """
        Look up a transform and return it as a pose vector.

        Parameters
        ----------
        base_frame : str, optional
            Override for the base frame.
        ee_frame : str, optional
            Override for the end-effector frame.
        timeout : float, optional
            Lookup timeout in seconds.

        Returns
        -------
        np.ndarray
            Pose vector ``[x, y, z, qw, qx, qy, qz]``.

        Raises
        ------
        RuntimeError
            If the transform cannot be retrieved within the timeout.
        """
        bf = base_frame or self.base_frame
        ef = ee_frame or self.ee_frame
        try:
            tf = self._tf_buffer.lookup_transform(bf, ef, Time(), timeout=Duration(seconds=timeout))
            t = tf.transform.translation
            r = tf.transform.rotation
            return np.array([t.x, t.y, t.z, r.w, r.x, r.y, r.z], dtype=float)
        except Exception as e:
            raise RuntimeError(f"Could not get transform from {bf} to {ef}: {e}")


if __name__ == "__main__":
    # example usage for fr3_link0 to fr3_link8
    import rclpy

    rclpy.init()
    node = rclpy.create_node("tf_client_example")
    tf_client = TFClient(node, base_frame="fr3_link0", ee_frame="fr3_link8")

    # spin until we get the transform
    pose = None
    while pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            pose = tf_client.get_pose()
        except RuntimeError:
            pass

    print(f"End-effector pose: {pose}")

    rclpy.shutdown()
