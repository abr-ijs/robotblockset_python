"""OptiTrack Localization Module.

This module provides functionality for localizing rigid bodies using the OptiTrack NatNet system.
It includes methods to fetch real-time frame data, manage rigid body and marker sets, and perform
transformations between task spaces (e.g., world and optical frames).

The module integrates with the NatNet client to retrieve and process motion capture data.

Copyright (c) 2025- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import time
import numpy as np
from time import sleep
import copy
from typing import Callable, Optional, Union

from robotblockset.rbs_typing import ArrayLike
from robotblockset.optitrack.natnet_client import NatNetClient
from robotblockset.tools import _struct, rbs_object, rbs_type, check_option, isvector
from robotblockset.transformations import map_pose, world2frame


class _default(_struct):
    """
    Default configuration class for the optitrack_localization.

    Attributes
    ----------
    TaskSpace : str
        The task space for pose representation. Default is "World".
    TaskPoseForm : str
        The pose form to use ("Pose", "Position", etc.). Default is "Pose".
    UpdateTime : float
        Time interval between updates. Default is 1.0 seconds.
    """

    def __init__(self) -> None:
        """
        Initialize the default OptiTrack localization settings.

        Returns
        -------
        None
            This constructor initializes the default configuration object in place.
        """
        self.TaskSpace = "World"
        self.TaskPoseForm = "Pose"
        self.UpdateTime = 1.0


class optitrack_localization(rbs_object):
    """
    A class for managing the localization of objects (rigid bodies) using OptiTrack NatNet.

    Attributes
    ----------
    host_ip : str
        IP address of the OptiTrack host.
    client_ip : str
        IP address of the client.
    multicast_ip : str
        Multicast IP address for OptiTrack communication.
    natnet_model : MoCapModel, optional
        Model description from the OptiTrack system.
    frame : MoCapFrame, optional
        Latest frame captured from OptiTrack.
    bodies : list
        List of rigid body names.
    n_bodies : int
        Number of rigid bodies.
    markersets : list
        List of marker set names.
    n_markersets : int
        Number of marker sets.
    n_markers : int
        Total number of markers.
    n_markers_in_set : list
        List with the number of markers in each marker set.
    BodyID : int
        ID for the platform's rigid body.
    last_valid_body_data : RigidBody, optional
        Last valid rigid body data for the platform.
    last_valid_body_frame_id : int, optional
        Last frame ID with valid rigid body data.
    last_valid_body_frame_time : float, optional
        Last frame time with valid rigid body data.
    TWorld : np.ndarray
        Transformation matrix between optical and world frames.
    client : NatNetClient
        The NatNet client used to interface with OptiTrack.
    """

    def __init__(self, host_ip: str = "127.0.0.1", client_ip: str = "127.0.0.1", multicast_ip: str = "239.255.42.99", frame_callback: Optional[Callable[..., None]] = None, body_callback: Optional[Callable[..., None]] = None) -> None:
        """
        Initializes the optitrack_localization class by connecting to the OptiTrack system.

        Parameters
        ----------
        host_ip : str, optional
            IP address of the OptiTrack host. Default is "127.0.0.1".
        client_ip : str, optional
            IP address of the client. Default is "127.0.0.1".
        multicast_ip : str, optional
            Multicast IP address for OptiTrack communication. Default is "239.255.42.99".
        frame_callback : Callable[..., None], optional
            Callback function to process new frames.
        body_callback : Callable[..., None], optional
            Callback function to process rigid body data.
        """
        self.Name = "OptiTrack"
        self.host_ip = host_ip
        self.client_ip = client_ip
        self.multicast_ip = multicast_ip

        self._default = _default()  # Default options

        self.natnet_model = None  # OptiTrack model
        self.frame = None  # OptiTrack captured frame
        self.bodies = {}  # Optitrack bodies
        self.n_bodies = 0  # number of rigid bodies
        self.markersets = {}  # Optitrack marker sets
        self.n_markersets = 0  # number of marker sets
        self.n_markers = 0  # number of all markers
        self.n_markers_in_set = []  # number of markers in each marker set

        self.BodyID = 0  # rigidbody id for platform
        self.last_valid_body_data = None  # last valid rigidbody data for platform
        self.last_valid_body_frame_id = None  # last frame id with valid rigidybody data for platform
        self.last_valid_body_frame_time = None  # last frame time with valid rigidybody data for platform

        self.TWorld = np.eye(4)  # transformation between optical and world frame

        self.last_update = time.time()

        self.client = NatNetClient()
        self.client.localIPAddress = self.client_ip
        self.client.serverIPAddress = self.host_ip
        self.client.multicastAddress = "239.255.42.99"

        if frame_callback is not None:
            self.client.newFrameListener = frame_callback

        if body_callback is not None:
            self.client.rigidBodyListener = body_callback

        self.client.run()

        i = 0
        while self.client.Model is None:
            i += 1
            if i > 3:
                raise ConnectionError("Client did not get the model")
            self.client.GetModel()
            sleep(0.1)
        i = 0
        while self.frame is None:
            i += 1
            if i > 3:
                raise ConnectionError("Client did not get the frame")
            self.GetFrame()
            sleep(0.1)

        self.natnet_model = self.client.GetModelDescription()
        if self.natnet_model.RigidBodyCount < 1:
            self.WarningMessage("No rigid bodies defined")
        else:
            self.n_bodies = self.natnet_model.RigidBodyCount
            self.bodies = self.natnet_model.GetRigidBodyNames()

        if self.natnet_model.MarkerSetCount < 1:
            self.WarningMessage("No marker sets defined")
        else:
            self.n_markersets = self.natnet_model.MarkerSetCount
            self.markersets = self.natnet_model.GetMarkerSetNames()
            self.n_markers_in_set = self.natnet_model.GetMarkerSetMarkerCount()
            self.n_markers = sum(self.n_markers_in_set[:-1])

    def isReady(self) -> bool:
        """
        Checks if the OptiTrack system is ready (connected).

        Returns
        -------
        bool
            True if the system is connected and ready, False otherwise.
        """
        return self.client.isConnected()

    def InitObject(self) -> None:
        """
        Initializes the object for various attributes.

        Returns
        -------
        None
            This method does not return any value. It modifies the internal state of the object.
        """
        pass

    def SetBodyID(self, id: Union[int, str] = 0) -> None:
        """
        Sets the ID of the rigid body to be used for localization.

        Parameters
        ----------
        id : Union[int, str], optional
            The rigid body ID or name to set. Default is 0.

        Raises
        ------
        ValueError
            If the rigid body ID is invalid.
        """
        if isinstance(id, str):
            if id in self.bodies:
                id = self.name2id("body", id)
            else:
                raise ValueError(f"Rigid body {id} does not exist")
        if id < self.frame.RigidBodyCount:
            self.BodyID = id
            self.last_valid_body_data = None
            self.GetFrame()
            if self.last_valid_body_data is None:
                print(f"No valid data for platform Rigidbody {id}")
        else:
            raise ValueError("Wrong RigidBody ID for platform")

    @property
    def x(self) -> np.ndarray:
        """
        Returns the current pose (position and orientation) of the rigid body.

        Returns
        -------
        np.ndarray
            The pose as a 7-element array [position, orientation].
        """
        return copy.deepcopy(self.GetPose(out="x"))

    @property
    def p(self) -> np.ndarray:
        """
        Returns the current position of the rigid body.

        Returns
        -------
        np.ndarray
            The position as a 3-element array.
        """
        return copy.deepcopy(self.GetPose(out="p"))

    @property
    def Q(self) -> np.ndarray:
        """
        Returns the current orientation of the rigid body.

        Returns
        -------
        np.ndarray
            The orientation as a 4-element array (quaternion).
        """
        return copy.deepcopy(self.GetPose(out="Q"))

    @property
    def R(self) -> np.ndarray:
        """
        Returns the current rotation matrix of the rigid body.

        Returns
        -------
        np.ndarray
            The rotation matrix as a 3x3 matrix.
        """
        return copy.deepcopy(self.GetPose(out="R"))

    @property
    def T(self) -> np.ndarray:
        """
        Returns the current transformation matrix of the rigid body.

        Returns
        -------
        np.ndarray
            The transformation matrix as a 4x4 matrix.
        """
        return copy.deepcopy(self.GetPose(state="Actual", task_space="World", out="T"))

    def spatial(self, x: ArrayLike) -> np.ndarray:
        """
        Validates the shape of the input `x` and returns it in an appropriate format.

        Parameters
        ----------
        x : ArrayLike
            The input array representing a spatial quantity, which can be one of the following shapes:
            - (7,) : pose (position and quaternion)
            - (4, 4) : transformation matrix
            - (3,) : position vector
            - (4,) : quaternion
            - (3, 3) : rotation matrix
            - (6,) : twist (linear and angular velocity)
            - (3, 4) : homogeneous matrix without the last row (assumed to be 3x4)

        Returns
        -------
        np.ndarray
            The input `x` in the validated shape, possibly modified if the shape was (3, 4).

        Raises
        ------
        TypeError
            If the input `x` does not have a valid shape.
        """
        x = rbs_type(x)

        # Check for valid shapes
        if x.shape == (7,) or x.shape == (4, 4) or x.shape == (3,) or x.shape == (4,) or x.shape == (3, 3) or x.shape == (6,):
            return x
        elif x.shape == (3, 4):
            x = np.vstack((x, np.array([0, 0, 0, 1])))
            return x
        else:
            raise TypeError("Parameter has not proper shape")

    def SetWorld(self, x: ArrayLike) -> None:
        """
        Sets the transformation between the world frame and the optical frame.

        Parameters
        ----------
        x : ArrayLike
            The transformation matrix or pose to set.
        """
        _x = self.spatial(x)
        if _x.shape == (4, 4):
            _T = _x
        elif isvector(_x, dim=7):
            _T = map_pose(x=_x, out="T")
        else:
            raise ValueError(f"Base pose shape {_x.shape} not supported")
        self.TWorld = _T

    def GetFrameTime(self) -> float:
        """
        Retrieves the time of the current frame if available.

        This method checks if the `frame` attribute is not `None` and returns the time of the current frame.
        If the `frame` is `None`, it returns a default value of 0.

        Returns
        -------
        float
            The time of the current frame. Returns 0 if the `frame` is not set.
        """
        if self.frame is not None:
            return self.frame.Time
        else:
            return 0

    def GetPose(self, out: Optional[str] = None, task_space: Optional[str] = None) -> np.ndarray:
        """
        Gets the pose of the rigid body in the specified task space.

        Parameters
        ----------
        out : str, optional
            The output format ("Pose", "Position", "Quaternion", "RotationMatrix", etc.).
        task_space : str, optional
            The task space frame ("World", "Optical").

        Returns
        -------
        np.ndarray
            The pose in the requested format.
        """
        if out is None:
            out = self._default.TaskPoseForm
        if task_space is None:
            task_space = self._default.TaskSpace
        self.GetFrame()
        _x = self.last_valid_body_data.x
        if check_option(task_space, "World"):
            _x = world2frame(_x, self.TWorld)
        elif check_option(task_space, "Optical"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(x=_x, out=out)

    def GetPos(self, out: str = "p", task_space: Optional[str] = None) -> np.ndarray:
        """
        Get the platform's position in the specified task space.

        Parameters
        ----------
        out : str, optional
            The output form of the position. By default, it returns the position ("p").
            Other possible values: "Position".
        task_space : str, optional
            The task space frame in which to return the position. By default, it is "World".
            Other options may include "Robot" or "Object".

        Returns
        -------
        np.ndarray
            The platform's position, represented as a 3-element array (3,).

        Raises
        ------
        ValueError
            If the `out` parameter is not one of the supported values.
        """
        if out in ["Position", "p"]:
            return self.GetPose(out=out, task_space=task_space)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetPos")

    def GetOri(self, out: str = "Q", task_space: Optional[str] = None) -> np.ndarray:
        """
        Get the platform's orientation in the specified task space.

        Parameters
        ----------
        out : str, optional
            The output form of the orientation. By default, it returns the orientation as a quaternion ("Q").
            Other possible values: "Quaternion", "RotationMatrix", "R".
        task_space : str, optional
            The task space frame in which to return the orientation. By default, it is "World".
            Other options may include "Robot" or "Object".

        Returns
        -------
        np.ndarray
            The platform's orientation, either as a quaternion (4,) or rotation matrix (3, 3).

        Raises
        ------
        ValueError
            If the `out` parameter is not one of the supported values.
        """
        if out in ["Quaternion", "Q", "RotationMatrix", "R"]:
            return self.GetPose(out=out, task_space=task_space)
        else:
            raise ValueError(f"Output form '{out}' not supported in GetOri")

    def GetRigidBodyPose(self, id: Union[int, str] = 0, task_space: Optional[str] = None, out: Optional[str] = None) -> np.ndarray:
        """
        Get the pose of a specific rigid body in the specified task space.

        Parameters
        ----------
        id : Union[int, str], optional
            The ID of the rigid body. By default, it is 0. If a string is provided and it corresponds
            to a body name, it will be converted to the corresponding ID.
        task_space : str, optional
            The task space frame in which to return the pose. By default, it is the task space defined
            in the default settings.
        out : str, optional
            The output form of the pose. By default, it is set to the task pose form defined in the default settings.

        Returns
        -------
        np.ndarray
            The pose of the rigid body, represented as a 7-element array (position + orientation).

        Raises
        ------
        ValueError
            If the `id` is incorrect, or if the rigid body does not exist.
        """
        if out is None:
            out = self._default.TaskPoseForm
        if isinstance(id, str):
            if id in self.bodies:
                id = self.name2id("body", id)
            else:
                raise ValueError(f"Rigid body {id} does not exist")
        if id < self.frame.RigidBodyCount:
            return self.GetRigidBodies(task_space=task_space, out=out)[id]
        else:
            raise ValueError("Wrong RigidBody ID for platform")

    # ----------------------------------------------------------
    def GetModel(self) -> None:
        """
        Fetch the model from OptiTrack.

        This method retrieves the current model from the OptiTrack system via the `client`.
        It waits briefly after fetching the model and then retrieves the model description.

        Returns
        -------
        None
        """
        self.client.GetModel()
        sleep(0.1)
        self.natnet_model = self.client.GetModelDescription()

    def GetFrame(self) -> None:
        """
        Fetch the latest frame from OptiTrack.

        This method retrieves the latest frame of data from the OptiTrack system. It checks if the frame
        contains valid data for the rigid body specified by the `BodyID` and updates relevant fields accordingly.

        Returns
        -------
        None
        """
        self.client.GetFrame()
        self.frame = self.client.Frame
        if self.frame is not None:
            if (self.frame.RigidBodyCount > self.BodyID) and (self.frame.RigidBodies[self.BodyID].Tracked):
                self.last_valid_body_data = self.frame.RigidBodies[self.BodyID]
                self.last_valid_body_frame_id = self.frame.FrameID
                self.last_valid_body_frame_time = self.frame.Time

    def GetRigidBodies(self, task_space: Optional[str] = None, out: str = "x") -> np.ndarray:
        """
        Retrieves the poses of all rigid bodies.

        Parameters
        ----------
        task_space : str, optional
            The task space frame ("World", "Optical").
        out : str, optional
            The output format ("Pose", "Position", "Quaternion", etc.).

        Returns
        -------
        np.ndarray
            The poses of all rigid bodies.
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        self.GetFrame()
        _x = self.frame.RigidBodies_x
        if check_option(task_space, "World"):
            if isvector(_x):
                _x = world2frame(_x, self.TWorld)
            else:
                for i in range(_x.shape[0]):
                    _x[i, :] = world2frame(_x[i, :], self.TWorld)
        elif check_option(task_space, "Optical"):
            pass
        else:
            raise ValueError(f"Task space '{task_space}' not supported in GetPose")
        return map_pose(x=_x, out=out)

    def GetMarkers(self, markerset_id: Optional[Union[int, str]] = None, task_space: Optional[str] = None, out: str = "p") -> Optional[np.ndarray]:
        """
        Retrieves the marker positions for a specified marker set.

        Parameters
        ----------
        markerset_id : Union[int, str], optional
            The marker set ID or name.
        task_space : str, optional
            The task space frame ("World", "Optical").
        out : str, optional
            The output format ("Position", "p").

        Returns
        -------
        np.ndarray or None
            The marker positions in the requested format, or None if unavailable.
        """
        if task_space is None:
            task_space = self._default.TaskSpace
        self.client.GetFrame()
        if isinstance(markerset_id, str):
            if markerset_id in self.markersets:
                id = self.name2id("markerset", markerset_id)
            else:
                raise ValueError(f"MarkerSet {markerset_id} does not exist")
        else:
            id = markerset_id
        try:
            _p = self.client.Frame.MarkerSets[id].Markers_p
            if check_option(task_space, "World"):
                if isvector(_p):
                    _p = world2frame(_p, self.TWorld)
                else:
                    for i in range(_p.shape[0]):
                        _p[i, :] = world2frame(_p[i, :], self.TWorld)
            elif check_option(task_space, "Optical"):
                pass
            else:
                raise ValueError(f"Task space '{task_space}' not supported in GetPose")
            return map_pose(p=_p, out=out)
        except Exception:
            return None

    def name2id(self, typ: str = "body", name: Optional[str] = None) -> Optional[int]:
        """
        Converts the name of an object (rigid body or marker set) to its corresponding ID.

        Parameters
        ----------
        typ : str, optional
            The type of object ("body" or "markerset").
        name : str, optional
            The name of the object.

        Returns
        -------
        Optional[int]
            The ID of the object if it exists, otherwise None.

        Raises
        ------
        ValueError
            If the object type is not supported.
        """
        idx = None
        if name is not None:
            if check_option(typ, "body"):
                if name in self.bodies:
                    idx = self.bodies.index(name)
            elif check_option(typ, "markerset"):
                if name in self.markersets:
                    idx = self.markersets.index(name)
            else:
                raise ValueError(f"Object type '{typ}' not supported in name2id")
        return idx


def receiveNewFrame(frameNumber: int, markerSetCount: int, unlabeledMarkersCount: int, rigidBodyCount: int, skeletonCount: int, labeledMarkerCount: int, timecode: int, timecodeSub: int, timestamp: float, isRecording: bool, trackedModelsChanged: bool) -> None:
    print("Received frame", frameNumber)


def receiveRigidBodyFrame(id: int, position: ArrayLike, rotation: ArrayLike) -> None:
    print("Received frame for rigid body", id)


# if __name__ == "__main__":

if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    opti = optitrack_localization()
    opti.GetFrame()
    for i in range(2):
        sleep(0.5)
        opti.GetFrame()
        print(f"Frame: {opti.frame.FrameID}\nRigidBodies:\n", opti.frame.RigidBodies_x)
        print("Markers:\n", opti.frame.MarkerSets[-1].Markers_p)

    opti.SetWorld([0, 0, 10, 1, 0, 0, 0])
    print(opti.TWorld)
    print(f"Frame: {opti.frame.FrameID} RigidBodies:\n", opti.GetRigidBodies())
