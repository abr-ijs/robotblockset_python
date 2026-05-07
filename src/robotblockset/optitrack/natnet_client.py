"""NatNet client implementation.

Licensed under the Apache License, Version 2.0 (the "License");.
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

OptiTrack NatNet direct depacketization library for Python 3.x

NatNet Client Module
--------------------

This module implements the communication protocol for interacting with the OptiTrack NatNet system.

It provides functionality to unpack and process motion capture data, such as marker sets, rigid bodies,
skeletons, and frame data. The NatNetClient class is the primary interface for connecting to the NatNet
server, sending requests, and receiving motion capture data for real-time analysis.

Copyright (c) 2018 Naturalpoint

Updated 2025 - Leon Zlajpah

  Added Model description classes and Frame classes
  Parser fills received data into Model and Frame
"""

import socket
import struct
from threading import Thread
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from datetime import datetime
from time import perf_counter

from robotblockset.rbs_typing import ArrayLike


class _struct:
    """
    A base class for objects with debugging and messaging capabilities.

    Provides utility methods to convert an object to a dictionary,
    iterate over its attributes, and populate an object from a dictionary.

    Methods
    -------
    asdict() -> dict
        Converts the object's attributes to a dictionary.
    __repr__() -> str
        Returns the string representation of the object as a dictionary.
    __iter__() -> iter
        Iterates over the object's attributes as key-value pairs.
    from_dict(data: dict) -> None
        Sets the attributes of the object from the given dictionary.
    """

    def asdict(self) -> Dict[str, Any]:
        return vars(self)

    def __repr__(self) -> str:
        return self.asdict().__str__()

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for key, value in vars(self).items():
            yield key, value

    def from_dict(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, value)


class RigidBodyDescription(_struct):
    """
    A class describing a rigid body in a motion capture system.

    Attributes
    ----------
    Name : str
        The name of the rigid body.
    ID : int
        The ID of the rigid body.
    ParentID : int
        The ID of the parent rigid body.
    MarkerOffsets : np.ndarray
        An array of offsets for each marker on the rigid body.
    ActiveLabels : list
        List of active labels for the markers on the rigid body.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[int] = None,
        parent_id: Optional[int] = None,
        offsets: Sequence[ArrayLike] = [],
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Initializes a RigidBodyDescription object with the provided name, ID, parent ID, marker offsets, and active labels.

        Parameters
        ----------
        name : str, optional
            The name of the rigid body (default is None).
        id : int, optional
            The ID of the rigid body (default is None).
        parent_id : int, optional
            The ID of the parent rigid body (default is None).
        offsets : Sequence[ArrayLike], optional
            A sequence of marker offsets for the rigid body (default is an empty list).
        labels : Sequence[int], optional
            A sequence of active labels for the markers on the rigid body (default is None, and is initialized as an empty list if not provided).
        """
        self.Name = name
        self.ID = id
        self.ParentID = parent_id
        self.MarkerOffsets = np.array(offsets)
        self.ActiveLabels = labels or []

    @property
    def MarkerCount(self) -> int:
        """
        Returns the number of markers on the rigid body.

        Returns
        -------
        int
            The number of markers on the rigid body, which is equivalent to the length of the MarkerOffsets array.
        """
        return len(self.MarkerOffsets)


class MarkerSetDescription(_struct):
    """
    A class describing a set of markers in a motion capture system.

    Attributes
    ----------
    Name : str
        The name of the marker set.
    Markers : list
        A list of marker names in the set.

    Properties
    ----------
    MarkerCount : int
        Returns the number of markers in the marker set.
    """

    def __init__(self, name: Optional[str] = None, markers: Sequence[str] = []) -> None:
        self.Name = name
        self.Markers = markers

    @property
    def MarkerCount(self) -> int:
        return len(self.Markers)


class SkeletonDescription(_struct):
    """
    A class that describes a skeleton in a motion capture system.

    Attributes
    ----------
    Name : str
        The name of the skeleton.
    ID : int
        The ID of the skeleton.
    RigidBodies : list
        A sequence of `RigidBodyDescription` objects associated with the skeleton.
    """

    def __init__(self, name: Optional[str] = None, id: Optional[int] = None, bodies: Optional[Sequence["RigidBodyDescription"]] = None) -> None:
        """
        Initializes the SkeletonDescription with the given name, ID, and a list of rigid bodies.

        Parameters
        ----------
        name : str, optional
            The name of the skeleton (default is None).
        id : int, optional
            The ID of the skeleton (default is None).
        bodies : Sequence['RigidBodyDescription'], optional
            A sequence of `RigidBodyDescription` objects associated with the skeleton (default is an empty list).
        """
        if bodies is None:
            bodies = []
        self.Name = name
        self.ID = id
        self.RigidBodies = bodies

    @property
    def RigidBodyCount(self) -> int:
        """
        Returns the number of rigid bodies associated with the skeleton.

        Returns
        -------
        int
            The number of rigid bodies.
        """
        return len(self.RigidBodies)

    def add_rigidbody(self, rigidbody: "RigidBodyDescription") -> None:
        """
        Adds a rigid body to the skeleton.

        Parameters
        ----------
        rigidbody : 'RigidBodyDescription'
            The rigid body to be added to the skeleton.
        """
        if rigidbody is not None:
            self.RigidBodies.append(rigidbody)


class MoCapModel(_struct):
    """
    A class representing a motion capture model consisting of rigid bodies, marker sets, and skeletons.

    Attributes
    ----------
    RigidBodies : list
        A list of `RigidBodyDescription` objects.
    MarkerSets : list
        A list of `MarkerSetDescription` objects.
    Skeletons : list
        A list of `SkeletonDescription` objects.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes a MoCapModel with empty lists for rigid bodies, marker sets, and skeletons.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments to initialize model attributes.
        """
        self.RigidBodies = []
        self.MarkerSets = []
        self.Skeletons = []

    def __str__(self) -> str:
        """
        Returns a string representation of the MoCapModel including details about rigid bodies,
        marker sets, and skeletons.

        Returns
        -------
        str
            A string representation of the MoCapModel.
        """
        _tmp = f"MarkerSetCount: {self.MarkerSetCount}\n"
        _tmp += f"RigidBodyCount: {self.RigidBodyCount}\n"
        _tmp += f"SkeletonCount: {self.SkeletonCount}\n"

        for i, x in enumerate(self.RigidBodies):
            _tmp += f"\n  Rigid body {i}:"
            _tmp += f"\n    Name     : {x.Name}"
            _tmp += f"\n    ID       : {x.ID}"
            _tmp += f"\n    ParentID : {x.ParentID}"
            _tmp += f"\n    Markers count: {x.MarkerCount}"
            for j, m in enumerate(x.MarkerOffsets):
                _tmp += f"\n      Marker{j} Offset: {m}"
            for j, a in enumerate(x.ActiveLabels):
                _tmp += f"\n      Marker{j} Active Labels: {a}"

        for i, x in enumerate(self.MarkerSets):
            _tmp += f"\n  Marker set {i}:"
            _tmp += f"\n    Name : {x.Name}"
            _tmp += f"\n    Count: {x.MarkerCount}"
            for j, m in enumerate(x.Markers):
                _tmp += f"\n      Marker{j}: {m}"

        for k, s in enumerate(self.Skeletons):
            _tmp += f"\n  Skeleton {k}:"
            _tmp += f"\n    Name: {s.Name}"
            _tmp += f"\n    ID  : {s.ID}"
            _tmp += f"\n    Rigid body count: {s.RigidBodyCount}"
            for i, x in enumerate(s.RigidBodies):
                _tmp += f"\n      Rigid body {i}:"
                _tmp += f"\n        Name     : {x.Name}"
                _tmp += f"\n        ID       : {x.ID}"
                _tmp += f"\n        ParentID : {x.ParentID}"
                _tmp += f"\n        Markers count: {x.MarkerCount}"
                for j, m in enumerate(x.MarkerOffsets):
                    _tmp += f"\n          Marker{j} Offset: {m}"
                for j, a in enumerate(x.ActiveLabels):
                    _tmp += f"\n          Marker{j} Active Labels: {a}"

        return _tmp

    @property
    def RigidBodyCount(self) -> int:
        """
        Returns the number of rigid bodies in the model.

        Returns
        -------
        int
            The number of rigid bodies.
        """
        return len(self.RigidBodies)

    @property
    def MarkerSetCount(self) -> int:
        """
        Returns the number of marker sets in the model.

        Returns
        -------
        int
            The number of marker sets.
        """
        return len(self.MarkerSets)

    @property
    def SkeletonCount(self) -> int:
        """
        Returns the number of skeletons in the model.

        Returns
        -------
        int
            The number of skeletons.
        """
        return len(self.Skeletons)

    def GetRigidBodyNames(self) -> List[str]:
        """
        Returns a list of names of all rigid bodies in the model.

        Returns
        -------
        list
            A list of rigid body names.
        """
        return [b.Name for b in self.RigidBodies]

    def GetMarkerSetNames(self) -> List[str]:
        """
        Returns a list of names of all marker sets in the model.

        Returns
        -------
        list
            A list of marker set names.
        """
        return [m.Name for m in self.MarkerSets]

    def GetMarkerSetMarkerCount(self) -> List[int]:
        """
        Returns a list of the marker counts for each marker set in the model.

        Returns
        -------
        list
            A list of marker counts for each marker set.
        """
        return [m.MarkerCount for m in self.MarkerSets]

    def add_markerset(self, name: Optional[str] = None, markers: Sequence[str] = []) -> None:
        """
        Adds a new marker set to the model.

        Parameters
        ----------
        name : str, optional
            The name of the marker set (default is None).
        markers : Sequence[str], optional
            A sequence of marker names (default is an empty list).
        """
        _marker_set = MarkerSetDescription(name=name, markers=markers)
        self.MarkerSets.append(_marker_set)

    def add_rigidbody(
        self,
        name: Optional[str] = None,
        id: Optional[int] = None,
        parent_id: Optional[int] = None,
        offsets: Sequence[ArrayLike] = [],
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Adds a new rigid body to the model.

        Parameters
        ----------
        name : str, optional
            The name of the rigid body (default is None).
        id : int, optional
            The ID of the rigid body (default is None).
        parent_id : int, optional
            The ID of the parent rigid body (default is None).
        offsets : Sequence[ArrayLike], optional
            A sequence of offsets for each marker on the rigid body (default is an empty list).
        labels : Sequence[int], optional
            A sequence of active labels for the markers on the rigid body (default is None).
        """
        _rigid_body = RigidBodyDescription(name=name, id=id, parent_id=parent_id, offsets=offsets, labels=labels)
        self.RigidBodies.append(_rigid_body)

    def add_skeleton(self, name: Optional[str] = None, id: Optional[int] = None, bodies: Sequence[RigidBodyDescription] = []) -> None:
        """
        Adds a new skeleton to the model.

        Parameters
        ----------
        name : str, optional
            The name of the skeleton (default is None).
        id : int, optional
            The ID of the skeleton (default is None).
        bodies : Sequence[RigidBodyDescription], optional
            A sequence of rigid bodies associated with the skeleton (default is an empty list).
        """
        _skeleton = SkeletonDescription(name=name, id=id, bodies=bodies)
        self.Skeletons.append(_skeleton)


class Marker(_struct):
    """
    A class representing a marker in a motion capture system.

    Attributes
    ----------
    ID : int
        The unique identifier for the marker.
    p : np.ndarray
        The position of the marker in 3D space (x, y, z).
    Size : float
        The size of the marker.
    Occluded : bool
        Indicates whether the marker is occluded (True/False).
    PointCloudSolved : bool
        Indicates if the point cloud for the marker has been solved (True/False).
    ModelSolved : bool
        Indicates if the marker model has been solved (True/False).
    Residual : float
        The residual error associated with the marker.

    """

    def __init__(
        self,
        id: Optional[int] = None,
        position: ArrayLike = [0, 0, 0],
        size: float = 0,
        occluded: bool = False,
        point_cloud_solved: bool = True,
        model_solved: bool = True,
        residual: float = 0,
    ) -> None:
        """
        Initializes a Marker object with the given parameters.

        Parameters
        ----------
        id : int, optional
            The unique identifier for the marker (default is None).
        position : ArrayLike, optional
            The position of the marker in 3D space (default is [0, 0, 0]).
        size : float, optional
            The size of the marker (default is 0).
        occluded : bool, optional
            Indicates whether the marker is occluded (default is False).
        point_cloud_solved : bool, optional
            Indicates if the point cloud for the marker has been solved (default is True).
        model_solved : bool, optional
            Indicates if the marker model has been solved (default is True).
        residual : float, optional
            The residual error associated with the marker (default is 0).
        """
        self.ID = id
        self.p = np.array(position)
        self.Size = size
        self.Occluded = occluded
        self.PointCloudSolved = point_cloud_solved
        self.ModelSolved = model_solved
        self.Residual = residual

    def __str__(self) -> str:
        """
        Returns a string representation of the Marker object, including its ID, position, size,
        residual, and other states.

        Returns
        -------
        str
            A formatted string representation of the marker.
        """
        _tmp = f"  Marker ID: {self.ID}"
        _tmp += f"\n    Pos: {self.p}"
        _tmp += f"\n    Size: {self.Size}"
        _tmp += f"\n    Residual: {self.Residual}"
        _tmp += f"\n    Occluded: {self.Occluded} PointCloudSolved: {self.PointCloudSolved} ModelSolved: {self.ModelSolved}"
        return _tmp


class MarkerSet(_struct):
    """
    A class representing a set of markers in a motion capture system.

    Attributes
    ----------
    Name : str
        The name of the marker set.
    Markers : list
        A list of `Marker` objects contained in the marker set.
    """

    def __init__(self, name: str = "") -> None:
        """
        Initializes a MarkerSet object with the provided name and an empty list of markers.

        Parameters
        ----------
        name : str, optional
            The name of the marker set (default is an empty string).
        """
        self.Name = name
        self.Markers = []

    def __str__(self) -> str:
        """
        Returns a string representation of the MarkerSet object, including the name and marker details.

        Returns
        -------
        str
            A formatted string representation of the marker set.
        """
        _tmp = f"MarkerSet Name: {self.Name}"
        _tmp += f"\n  Marker Count: {self.MarkerCount}"
        for i, m in enumerate(self.Markers):
            _tmp += f"\n    Marker{i} Position: {m.p}"
        return _tmp

    @property
    def MarkerCount(self) -> int:
        """
        Returns the number of markers in the marker set.

        Returns
        -------
        int
            The number of markers in the marker set.
        """
        return len(self.Markers)

    @property
    def Markers_p(self) -> np.ndarray:
        """
        Returns the positions of all markers as a 2D NumPy array.

        Returns
        -------
        np.ndarray
            A 2D NumPy array containing the positions of all markers in the marker set.
        """
        return np.array([m.p for m in self.Markers])

    def add_marker(self, marker: Optional[Marker] = None) -> None:
        """
        Adds a new marker to the marker set.

        Parameters
        ----------
        marker : Marker, optional
            A `Marker` object to be added to the marker set (default is None).
        """
        if marker is not None:
            self.Markers.append(marker)


class RigidBody(_struct):
    """
    A class representing a rigid body in a motion capture system.

    Attributes
    ----------
    ID : int
        The unique identifier for the rigid body.
    p : np.ndarray
        The position of the rigid body in 3D space (x, y, z).
    Q : np.ndarray
        The orientation of the rigid body represented as a quaternion (w, x, y, z).
    MeanError : float
        The mean error associated with the rigid body.
    Tracked : bool
        Indicates whether the rigid body is being tracked (True/False).
    """

    def __init__(
        self,
        id: Optional[int] = None,
        position: ArrayLike = [0, 0, 0],
        orientation: ArrayLike = [0, 0, 0, 1],
        error: float = 0,
        tracked: bool = True,
    ) -> None:
        """
        Initializes a RigidBody object with the given parameters.

        Parameters
        ----------
        id : int, optional
            The unique identifier for the rigid body (default is None).
        position : ArrayLike, optional
            The position of the rigid body in 3D space (default is [0, 0, 0]).
        orientation : ArrayLike, optional
            The orientation of the rigid body represented as a quaternion (default is [0, 0, 0, 1]).
        error : float, optional
            The mean error associated with the rigid body (default is 0).
        tracked : bool, optional
            Indicates whether the rigid body is tracked (default is True).
        """
        self.ID = id
        self.p = np.array(position)
        self.Q = np.array(orientation)[[3, 0, 1, 2]]  # Reorder quaternion to [w, x, y, z]
        self.MeanError = error
        self.Tracked = tracked

    def __str__(self) -> str:
        """
        Returns a string representation of the RigidBody object, including its ID, position, orientation,
        mean error, and tracking status.

        Returns
        -------
        str
            A formatted string representation of the rigid body.
        """
        _tmp = f"  RigidBody ID: {self.ID}"
        _tmp += f"\n    Position: {self.p}"
        _tmp += f"\n    Orientation: {self.Q}"
        _tmp += f"\n    MeanError: {self.MeanError}"
        _tmp += f"\n    Tracked: {self.Tracked}"
        return _tmp

    @property
    def x(self) -> np.ndarray:
        """
        Returns the position and orientation of the rigid body concatenated as a 1D NumPy array.

        Returns
        -------
        np.ndarray
            A 1D array containing the position and orientation of the rigid body.
        """
        return np.hstack((self.p, self.Q))

    @property
    def tracked(self) -> bool:
        """
        Returns the tracking status of the rigid body.

        Returns
        -------
        bool
            True if the rigid body is tracked, otherwise False.
        """
        return self.Tracked


class Skeleton(_struct):
    """
    A class representing a skeleton in a motion capture system.

    Attributes
    ----------
    ID : int
        The unique identifier for the skeleton.
    RigidBodies : list
        A list of `RigidBody` objects associated with the skeleton.
    """

    def __init__(self, id: int = None) -> None:
        """
        Initializes a Skeleton object with the given ID and an empty list of rigid bodies.

        Parameters
        ----------
        id : int, optional
            The unique identifier for the skeleton (default is None).
        """
        self.ID = id
        self.RigidBodies = []

    @property
    def RigidBodyCount(self) -> int:
        """
        Returns the number of rigid bodies in the skeleton.

        Returns
        -------
        int
            The number of rigid bodies in the skeleton.
        """
        return len(self.RigidBodies)

    def add_rigidbody(self, rigidbody: Optional[RigidBody] = None) -> None:
        """
        Adds a new rigid body to the skeleton.

        Parameters
        ----------
        rigidbody : RigidBody, optional
            A `RigidBody` object to be added to the skeleton (default is None).
        """
        if rigidbody is not None:
            self.RigidBodies.append(rigidbody)

    def __str__(self) -> str:
        """
        Returns a string representation of the Skeleton object, including its ID and details of rigid bodies.

        Returns
        -------
        str
            A formatted string representation of the skeleton.
        """
        _tmp = f"Skeleton ID: {self.ID}"
        _tmp += f"\n  RigidBody Count: {self.RigidBodyCount}"
        for i, rb in enumerate(self.RigidBodies):
            _tmp += f"\n    RigidBody {i}:"
            _tmp += f"\n      ID       : {rb.ID}"
            _tmp += f"\n      Position : {rb.p}"
            _tmp += f"\n      Orientation: {rb.Q}"
        return _tmp


class MoCapFrame(_struct):
    """
    A class representing a single frame of motion capture data.

    Attributes
    ----------
    FrameID : int
        The unique identifier for the frame.
    Timestamp : float
        The timestamp of the frame.
    Timecode : int
        The timecode of the frame.
    TimecodeSub : int
        The sub-timecode of the frame.
    MarkerSets : list
        A list of `MarkerSet` objects associated with the frame.
    LabeledMarkers : list
        A list of `Marker` objects that are labeled in the frame.
    UnlabeledMarkers : list
        A list of positions for unlabeled markers.
    RigidBodies : list
        A list of `RigidBody` objects associated with the frame.
    Skeletons : list
        A list of `Skeleton` objects associated with the frame.
    Time : float
        The wall time associated with the frame.
    """

    def __init__(self, id: Optional[int] = None, ts: Optional[float] = None, timecode: int = 0, timecode_sub: int = 0, wall_time: Optional[float] = None) -> None:
        """
        Initializes a MoCapFrame object with the given parameters.

        Parameters
        ----------
        id : int, optional
            The unique identifier for the frame (default is None).
        ts : float, optional
            The timestamp of the frame (default is None).
        timecode : int, optional
            The timecode of the frame (default is 0).
        timecode_sub : int, optional
            The sub-timecode of the frame (default is 0).
        wall_time : float, optional
            The wall time associated with the frame (default is None).
        """
        self.FrameID = id
        self.Timestamp = ts
        self.Timecode = timecode
        self.TimecodeSub = timecode_sub
        self.MarkerSets = []
        self.LabeledMarkers = []
        self.UnlabeledMarkers = []
        self.RigidBodies = []
        self.Skeletons = []
        self.Time = wall_time

    def __str__(self) -> str:
        """
        Returns a string representation of the MoCapFrame object, including its ID, timestamp,
        timecode, and associated data (e.g., marker sets, rigid bodies, labeled markers).

        Returns
        -------
        str
            A formatted string representation of the motion capture frame.
        """
        _tmp = f"Frame: {self.FrameID}"
        _tmp += f"\nTime: {self.Time}"
        _tmp += f"\nTimestamp: {datetime.fromtimestamp(self.Timestamp)}"
        _tmp += f"\nTimecode: {self.Timecode}"
        _tmp += f"\nTimecodeSub: {self.TimecodeSub}"

        _tmp += f"\nMarkerSets Count: {len(self.MarkerSets)}"
        for ms in self.MarkerSets:
            _tmp += "\n" + ms.__str__()
        _tmp += f"\nRigidBodies Count: {len(self.RigidBodies)}"
        for b in self.RigidBodies:
            _tmp += "\n" + b.__str__()
        _tmp += f"\nLabeledMarkers Count: {len(self.LabeledMarkers)}"
        for m in self.LabeledMarkers:
            _tmp += "\n" + m.__str__()
        _tmp += f"\nUnlabeledMarkers Count: {len(self.UnlabeledMarkers)}"
        for i, m in enumerate(self.UnlabeledMarkers):
            _tmp += f"\n  Marker{i} Position: {m}"
        _tmp += f"\nSkeletons Count: {len(self.Skeletons)}"
        for s in self.Skeletons:
            _tmp += "\n" + s.__str__()

        return _tmp

    @property
    def RigidBodyCount(self) -> int:
        """
        Returns the number of rigid bodies in the frame.

        Returns
        -------
        int
            The number of rigid bodies in the frame.
        """
        return len(self.RigidBodies)

    @property
    def MarkerSetCount(self) -> int:
        """
        Returns the number of marker sets in the frame.

        Returns
        -------
        int
            The number of marker sets in the frame.
        """
        return len(self.MarkerSets)

    @property
    def LabeledMarkersCount(self) -> int:
        """
        Returns the number of labeled markers in the frame.

        Returns
        -------
        int
            The number of labeled markers in the frame.
        """
        return len(self.LabeledMarkers)

    @property
    def UnlabeledMarkersCount(self) -> int:
        """
        Returns the number of unlabeled markers in the frame.

        Returns
        -------
        int
            The number of unlabeled markers in the frame.
        """
        return len(self.UnlabeledMarkers)

    @property
    def SkeletonCount(self) -> int:
        """
        Returns the number of skeletons in the frame.

        Returns
        -------
        int
            The number of skeletons in the frame.
        """
        return len(self.Skeletons)

    @property
    def RigidBodies_x(self) -> np.ndarray:
        """
        Returns the position and orientation (x) of all rigid bodies in the frame.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n, 7) where each row contains position and orientation data of a rigid body.
        """
        return np.atleast_2d([b.x for b in self.RigidBodies])

    @property
    def RigidBodies_tracked(self) -> List[bool]:
        """
        Returns the tracking status of each rigid body in the frame.

        Returns
        -------
        list
            A list of tracking statuses for each rigid body.
        """
        return [b.tracked for b in self.RigidBodies]

    @property
    def RigidBodies_p(self) -> np.ndarray:
        """
        Returns the position of all rigid bodies in the frame.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n, 3) containing the positions of the rigid bodies.
        """
        return np.atleast_2d([b.p for b in self.RigidBodies])

    @property
    def RigidBodies_Q(self) -> np.ndarray:
        """
        Returns the orientation (quaternion) of all rigid bodies in the frame.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n, 4) containing the orientations (quaternions) of the rigid bodies.
        """
        return np.atleast_2d([b.Q for b in self.RigidBodies])

    def add_markerset(self, markerset: Optional[MarkerSet] = None) -> None:
        """
        Adds a marker set to the frame.

        Parameters
        ----------
        markerset : MarkerSet, optional
            A `MarkerSet` object to be added to the frame (default is None).
        """
        if markerset is not None:
            self.MarkerSets.append(markerset)

    def add_labeledmarker(
        self,
        id: Optional[int] = None,
        pos: ArrayLike = [0, 0, 0],
        size: float = 0,
        occluded: bool = False,
        point_cloud_solved: bool = True,
        model_solved: bool = True,
        residual: float = 0,
    ) -> None:
        """
        Adds a labeled marker to the frame.

        Parameters
        ----------
        id : int, optional
            The ID of the labeled marker (default is None).
        pos : ArrayLike, optional
            The position of the labeled marker in 3D space (default is [0, 0, 0]).
        size : float, optional
            The size of the labeled marker (default is 0).
        occluded : bool, optional
            Whether the marker is occluded (default is False).
        point_cloud_solved : bool, optional
            Whether the point cloud is solved (default is True).
        model_solved : bool, optional
            Whether the model is solved (default is True).
        residual : float, optional
            The residual error associated with the labeled marker (default is 0).
        """
        _marker = Marker(id=id, position=pos, size=size, occluded=occluded, point_cloud_solved=point_cloud_solved, model_solved=model_solved, residual=residual)
        self.LabeledMarkers.append(_marker)

    def add_rigidbody(
        self,
        id: Optional[int] = None,
        position: ArrayLike = [0, 0, 0],
        orientation: ArrayLike = [0, 0, 0, 1],
        error: float = 0,
        tracked: bool = True,
    ) -> None:
        """
        Adds a rigid body to the frame.

        Parameters
        ----------
        id : int, optional
            The ID of the rigid body (default is None).
        position : ArrayLike, optional
            The position of the rigid body (default is [0, 0, 0]).
        orientation : ArrayLike, optional
            The orientation (quaternion) of the rigid body (default is [0, 0, 0, 1]).
        error : float, optional
            The mean error associated with the rigid body (default is 0).
        tracked : bool, optional
            Whether the rigid body is being tracked (default is True).
        """
        _rigid_body = RigidBody(id=id, position=position, orientation=orientation, error=error, tracked=tracked)
        self.RigidBodies.append(_rigid_body)

    def add_skeleton(self, skeleton: Optional[Skeleton] = None) -> None:
        """
        Adds a skeleton to the frame.

        Parameters
        ----------
        skeleton : Skeleton, optional
            A `Skeleton` object to be added to the frame (default is None).
        """
        if skeleton is not None:
            self.Skeletons.append(skeleton)


# Client/server message ids
natnet_msgs = {
    "NAT_PING": "0",
    "NAT_PINGRESPONSE": "1",
    "NAT_REQUEST": "2",
    "NAT_RESPONSE": "3",
    "NAT_REQUEST_MODELDEF": "4",
    "NAT_MODELDEF": "5",
    "NAT_REQUEST_FRAMEOFDATA": "6",
    "NAT_FRAMEOFDATA": "7",
    "NAT_MESSAGESTRING": "8",
    "NAT_DISCONNECT": "9",
    "NAT_UNRECOGNIZED_REQUEST": "100",
}
natnet_msgs_id = {v[0]: k for k, v in natnet_msgs.items()}


def trace(*args: Any) -> None:
    pass  # print("".join(map(str, args)))


# Create structs for reading various object types to speed up parsing.
Vector3 = struct.Struct("<fff")
Quaternion = struct.Struct("<ffff")
FloatValue = struct.Struct("<f")
DoubleValue = struct.Struct("<d")


class NatNetClient:
    def __init__(self) -> None:
        # Change this value to the IP address of the NatNet server.
        self.serverIPAddress: str = "127.0.0.1"

        # Change this value to the IP address of your local network interface
        self.localIPAddress: str = "127.0.0.1"

        # This should match the multicast address listed in Motive's streaming settings.
        self.multicastAddress: str = "239.255.42.99"

        # NatNet Command channel
        self.commandPort: int = 1510

        # NatNet Data channel
        self.dataPort: int = 1511

        # Set this to a callback method of your choice to receive per-rigid-body data at each frame.
        self.rigidBodyListener: Optional[Callable[[int, Tuple[float, float, float], Tuple[float, float, float, float]], None]] = None
        self.newFrameListener: Optional[Callable[[int, int, int, int, int, int, int, int, float, bool, bool], None]] = None

        # NatNet stream version. This will be updated to the actual version the server is using during initialization.
        self.__natNetStreamVersion: Tuple[int, int, int, int] = (3, 0, 0, 0)

        self.abort: bool = False
        self._connected: bool = False

        self.Model: Optional[MoCapModel] = None
        self._rigid_body_description: Optional[RigidBodyDescription] = None
        self.Frame: Optional[MoCapFrame] = None
        self._rigid_body: Optional[RigidBody] = None
        self.commandSocket: Optional[socket.socket] = None
        self.dataSocket: Optional[socket.socket] = None

    # Client/server message ids
    NAT_PING = 0
    NAT_PINGRESPONSE = 1
    NAT_REQUEST = 2
    NAT_RESPONSE = 3
    NAT_REQUEST_MODELDEF = 4
    NAT_MODELDEF = 5
    NAT_REQUEST_FRAMEOFDATA = 6
    NAT_FRAMEOFDATA = 7
    NAT_MESSAGESTRING = 8
    NAT_DISCONNECT = 9
    NAT_UNRECOGNIZED_REQUEST = 100

    # Create a data socket to attach to the NatNet stream
    def __createDataSocket(self, port: int) -> socket.socket:
        result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # Internet  # UDP

        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicastAddress) + socket.inet_aton(self.localIPAddress))

        result.bind((self.localIPAddress, port))

        return result

    # Create a command socket to attach to the NatNet stream
    def __createCommandSocket(self) -> socket.socket:
        result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result.bind(("", 0))
        result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        return result

    # Unpack a rigid body object from a data packet
    def __unpackRigidBody(self, data: Union[bytes, memoryview]) -> int:
        offset = 0

        # ID (4 bytes)
        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("ID:", id)

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        trace("\tPosition:", pos[0], ",", pos[1], ",", pos[2])
        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        trace("\tOrientation:", rot[0], ",", rot[1], ",", rot[2], ",", rot[3])
        self._rigid_body = RigidBody(id=id, position=pos, orientation=rot)

        # Send information to any listener.
        if self.rigidBodyListener is not None:
            self.rigidBodyListener(id, pos, rot)

        # RB Marker Data ( Before version 3.0.  After Version 3.0 Marker data is in description )
        if self.__natNetStreamVersion[0] < 3 and self.__natNetStreamVersion[0] != 0:
            # Marker count (4 bytes)
            markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            markerCountRange = range(0, markerCount)
            trace("\tMarker Count:", markerCount)

            # Marker positions
            for i in markerCountRange:
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                trace("\tMarker", i, ":", pos[0], ",", pos[1], ",", pos[2])

            if self.__natNetStreamVersion[0] >= 2:
                # Marker ID's
                for i in markerCountRange:
                    id = int.from_bytes(data[offset : offset + 4], byteorder="little")
                    offset += 4
                    trace("\tMarker ID", i, ":", id)

                # Marker sizes
                for i in markerCountRange:
                    size = FloatValue.unpack(data[offset : offset + 4])
                    offset += 4
                    trace("\tMarker Size", i, ":", size[0])

        if self.__natNetStreamVersion[0] >= 2:
            (markerError,) = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            trace("\tMarker Error:", markerError)
            self._rigid_body.MeanError = markerError

        # Version 2.6 and later
        if ((self.__natNetStreamVersion[0] == 2) and (self.__natNetStreamVersion[1] >= 6)) or self.__natNetStreamVersion[0] > 2 or self.__natNetStreamVersion[0] == 0:
            (param,) = struct.unpack("h", data[offset : offset + 2])
            trackingValid = (param & 0x01) != 0
            offset += 2
            trace("\tTracking Valid:", "True" if trackingValid else "False")
            self._rigid_body.Tracked = trackingValid

        return offset

    # Unpack a skeleton object from a data packet
    def __unpackSkeleton(self, data: Union[bytes, memoryview]) -> int:
        offset = 0

        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("ID:", id)
        _s = Skeleton(id=id)

        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("Rigid Body Count:", rigidBodyCount)
        for j in range(0, rigidBodyCount):
            offset += self.__unpackRigidBody(data[offset:])
            _s.add_rigidbody(self._rigid_body)

        return offset

    # Unpack data from a motion capture frame message
    def __unpackMocapData(self, data: Union[bytes, memoryview]) -> None:
        trace("Begin MoCap Frame\n-----------------\n")

        data = memoryview(data)
        offset = 0

        # Frame number (4 bytes)
        frameNumber = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("Frame #:", frameNumber)
        self.Frame = MoCapFrame(id=frameNumber, wall_time=perf_counter())

        # Marker set count (4 bytes)
        markerSetCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("Marker Set Count:", markerSetCount)

        for i in range(0, markerSetCount):
            # Marker set name
            modelName, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(modelName) + 1
            trace("Marker Set Name:", modelName.decode("utf-8"))
            _ms = MarkerSet(modelName.decode("utf-8"))

            # Marker count (4 bytes)
            markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("Marker Count:", markerCount)

            for j in range(0, markerCount):
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                trace("\tMarker", j, ":", pos[0], ",", pos[1], ",", pos[2])
                _ms.add_marker(Marker(position=pos))

            self.Frame.add_markerset(_ms)

        # Unlabeled markers count (4 bytes)
        unlabeledMarkersCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("Unlabeled Markers Count:", unlabeledMarkersCount)

        for i in range(0, unlabeledMarkersCount):
            pos = Vector3.unpack(data[offset : offset + 12])
            offset += 12
            trace("\tMarker", i, ":", pos[0], ",", pos[1], ",", pos[2])
            self.Frame.UnlabeledMarkers.append(pos)

        # Rigid body count (4 bytes)
        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("Rigid Body Count:", rigidBodyCount)

        for i in range(0, rigidBodyCount):
            offset += self.__unpackRigidBody(data[offset:])
            self.Frame.RigidBodies.append(self._rigid_body)

        # Version 2.1 and later
        skeletonCount = 0
        if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] > 0) or self.__natNetStreamVersion[0] > 2:
            skeletonCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("Skeleton Count:", skeletonCount)
            for i in range(0, skeletonCount):
                offset += self.__unpackSkeleton(data[offset:])

        # Labeled markers (Version 2.3 and later)
        labeledMarkerCount = 0
        if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] > 3) or self.__natNetStreamVersion[0] > 2:
            labeledMarkerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("Labeled Marker Count:", labeledMarkerCount)
            for i in range(0, labeledMarkerCount):
                id = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                size = FloatValue.unpack(data[offset : offset + 4])
                offset += 4
                _m = Marker(id=id, position=pos, size=size)

                # Version 2.6 and later
                if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 6) or self.__natNetStreamVersion[0] > 2:  # or major == 0:
                    (param,) = struct.unpack("h", data[offset : offset + 2])
                    offset += 2
                    occluded = (param & 0x01) != 0
                    pointCloudSolved = (param & 0x02) != 0
                    modelSolved = (param & 0x04) != 0
                    _m.Occluded = occluded
                    _m.PointCloudSolved = pointCloudSolved
                    _m.ModelSolved = modelSolved

                # Version 3.0 and later
                if self.__natNetStreamVersion[0] >= 3:  # or major == 0:
                    (residual,) = FloatValue.unpack(data[offset : offset + 4])
                    offset += 4
                    trace("Residual:", residual)
                    _m.Residual = residual

                self.Frame.LabeledMarkers.append(_m)

        # Force Plate data (version 2.9 and later)
        if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 9) or self.__natNetStreamVersion[0] > 2:
            forcePlateCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("Force Plate Count:", forcePlateCount)
            for i in range(0, forcePlateCount):
                # ID
                forcePlateID = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                trace("Force Plate", i, ":", forcePlateID)

                # Channel Count
                forcePlateChannelCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4

                # Channel Data
                for j in range(0, forcePlateChannelCount):
                    trace("\tChannel", j, ":", forcePlateID)
                    forcePlateChannelFrameCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
                    offset += 4
                    for k in range(0, forcePlateChannelFrameCount):
                        forcePlateChannelVal = int.from_bytes(data[offset : offset + 4], byteorder="little")
                        offset += 4
                        trace("\t\t", forcePlateChannelVal)

        # Device data (version 2.11 and later)
        if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 11) or self.__natNetStreamVersion[0] > 2:
            deviceCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("Device Count:", deviceCount)
            for i in range(0, deviceCount):
                # ID
                deviceID = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                trace("Device", i, ":", deviceID)

                # Channel Count
                deviceChannelCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4

                # Channel Data
                for j in range(0, deviceChannelCount):
                    trace("\tChannel", j, ":", deviceID)
                    deviceChannelFrameCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
                    offset += 4
                    for k in range(0, deviceChannelFrameCount):
                        deviceChannelVal = int.from_bytes(data[offset : offset + 4], byteorder="little")
                        offset += 4
                        trace("\t\t", deviceChannelVal)

        # Timecode
        timecode = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        timecodeSub = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        # Timestamp (increased to double precision in 2.7 and later)
        if (self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 7) or self.__natNetStreamVersion[0] > 2:
            (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
            offset += 8
        else:
            (timestamp,) = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
        self.Frame.Timestamp = timestamp
        self.Frame.Timecode = timecode
        self.Frame.TimecodeSub = timecodeSub

        # Hires Timestamp (Version 3.0 and later)
        if self.__natNetStreamVersion[0] >= 3:  # or major == 0:
            stampCameraExposure = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8
            stampDataReceived = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8
            stampTransmit = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8

        # Frame parameters
        (param,) = struct.unpack("h", data[offset : offset + 2])
        isRecording = (param & 0x01) != 0
        trackedModelsChanged = (param & 0x02) != 0
        offset += 2

        # Send information to any listener.
        if self.newFrameListener is not None:
            self.newFrameListener(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount, labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged)

    # Unpack a marker set description packet
    def __unpackMarkerSetDescription(self, data: Union[bytes, memoryview]) -> int:
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace("Markerset Name:", name.decode("utf-8"))
        _markerset_name = name.decode("utf-8")

        markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        _markers = []
        for i in range(0, markerCount):
            name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(name) + 1
            trace("\tMarker Name:", name.decode("utf-8"))
            _markers.append(name.decode("utf-8"))

        self.Model.MarkerSets.append(MarkerSetDescription(_markerset_name, _markers))
        return offset

    # Unpack a rigid body description packet
    def __unpackRigidBodyDescription(self, data: Union[bytes, memoryview]) -> int:
        offset = 0

        # Version 2.0 or higher
        if self.__natNetStreamVersion[0] >= 2:
            name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(name) + 1
            trace("RigidBody Name:", name.decode("utf-8"))
            _rigidbody_name = name.decode("utf-8")

        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        parentID = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        trace("RigidBody PatentID:", parentID)

        timestamp = Vector3.unpack(data[offset : offset + 12])
        offset += 12

        # Version 3.0 and higher, rigid body marker information contained in description
        if self.__natNetStreamVersion[0] >= 3 or self.__natNetStreamVersion[0] == 0:
            markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            trace("RigidBody Marker Count:", markerCount)

            offsets = []
            markerCountRange = range(0, markerCount)
            for marker in markerCountRange:
                markerOffset = Vector3.unpack(data[offset : offset + 12])
                trace("\tMarker Offset:", markerOffset)
                offset += 12
                offsets.append(markerOffset)
            labels = []
            for marker in markerCountRange:
                activeLabel = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                trace("\tMarker Label:", activeLabel)
                labels.append(activeLabel)

        self._rigid_body_description = RigidBodyDescription(_rigidbody_name, id, parentID, offsets, labels)
        return offset

    # Unpack a skeleton description packet
    def __unpackSkeletonDescription(self, data: Union[bytes, memoryview]) -> int:
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        trace("\tSkeleton Name:", name.decode("utf-8"))
        _skeleton_name = name.decode("utf-8")

        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        bodies = []
        for i in range(0, rigidBodyCount):
            offset += self.__unpackRigidBodyDescription(data[offset:])
            bodies.append(self._rigid_body_description)

        self.Model.add_skeleton(_skeleton_name, id, bodies)
        return offset

    # Unpack a data description packet
    def __unpackDataDescriptions(self, data: Union[bytes, memoryview]) -> None:
        trace("Begin Data Description\n-----------------\n")
        self.Model = MoCapModel()
        offset = 0
        datasetCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        for i in range(0, datasetCount):
            type = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            if type == 0:
                offset += self.__unpackMarkerSetDescription(data[offset:])
            elif type == 1:
                offset += self.__unpackRigidBodyDescription(data[offset:])
                self.Model.RigidBodies.append(self._rigid_body_description)

            elif type == 2:
                offset += self.__unpackSkeletonDescription(data[offset:])

    def __dataThreadFunction(self, socket: socket.socket) -> None:
        while not self.abort:
            # Block for input
            data, addr = socket.recvfrom(32768)  # 32k byte buffer size
            if len(data) > 0:
                self.__processMessage(data)

    def __processMessage(self, data: Union[bytes, memoryview]) -> None:
        trace("Begin Packet\n------------\n")

        messageID = int.from_bytes(data[0:2], byteorder="little")
        trace("Message ID:", messageID)

        packetSize = int.from_bytes(data[2:4], byteorder="little")
        trace("Packet Size:", packetSize)

        offset = 4
        if messageID == self.NAT_FRAMEOFDATA:
            self.__unpackMocapData(data[offset:])
        elif messageID == self.NAT_MODELDEF:
            self.__unpackDataDescriptions(data[offset:])
        elif messageID == self.NAT_PINGRESPONSE:
            offset += 256  # Skip the sending app's Name field
            offset += 4  # Skip the sending app's Version info
            self.__natNetStreamVersion = struct.unpack("BBBB", data[offset : offset + 4])
            offset += 4
        elif messageID == self.NAT_RESPONSE:
            if packetSize == 4:
                commandResponse = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
            else:
                message, separator, remainder = bytes(data[offset:]).partition(b"\0")
                offset += len(message) + 1
                trace("Command response:", message.decode("utf-8"))
        elif messageID == self.NAT_UNRECOGNIZED_REQUEST:
            trace("Received 'Unrecognized request' from server")
        elif messageID == self.NAT_MESSAGESTRING:
            message, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(message) + 1
            trace("Received message from server:", message.decode("utf-8"))
        else:
            trace("ERROR: Unrecognized packet type")

        trace("End Packet\n----------\n")

    def sendCommand(self, command: int, commandStr: str, socket: socket.socket, address: Tuple[str, int]) -> None:
        # Compose the message in our known message format
        if command == self.NAT_REQUEST_MODELDEF or command == self.NAT_REQUEST_FRAMEOFDATA:
            packetSize = 0
            commandStr = ""
        elif command == self.NAT_REQUEST:
            packetSize = len(commandStr) + 1
        elif command == self.NAT_PING:
            commandStr = "Ping"
            packetSize = len(commandStr) + 1

        data = command.to_bytes(2, byteorder="little")
        data += packetSize.to_bytes(2, byteorder="little")

        data += commandStr.encode("utf-8")
        data += b"\0"

        socket.sendto(data, address)

    def run(self) -> None:
        # Create the data socket
        self.dataSocket = self.__createDataSocket(self.dataPort)
        if self.dataSocket is None:
            print("Could not open data channel")
            exit

        # Create the command socket
        self.commandSocket = self.__createCommandSocket()
        if self.commandSocket is None:
            print("Could not open command channel")
            exit

        # Create a separate thread for receiving data packets
        dataThread = Thread(target=self.__dataThreadFunction, args=(self.dataSocket,))
        dataThread.daemon = True
        dataThread.start()

        # Create a separate thread for receiving command packets
        commandThread = Thread(target=self.__dataThreadFunction, args=(self.commandSocket,))
        commandThread.daemon = True
        commandThread.start()

        self._connected = True

        # self.sendCommand(self.NAT_PING, "", self.commandSocket, (self.serverIPAddress, self.commandPort))
        self.sendCommand(self.NAT_REQUEST_MODELDEF, "", self.commandSocket, (self.serverIPAddress, self.commandPort))
        # self.sendCommand( self.NAT_REQUEST, "StartRecording", self.commandSocket, (self.serverIPAddress, self.commandPort) )
        # self.sendCommand( self.NAT_REQUEST, "StopRecording", self.commandSocket, (self.serverIPAddress, self.commandPort) )

        print("NATNET running")

    # ----------------------------------------------------
    def isConnected(self) -> bool:
        """
        Checks if the client is connected to the NatNet server.

        Returns
        -------
        bool
            `True` if the client is connected, `False` otherwise.
        """
        return self._connected

    def GetFrame(self) -> None:
        """
        Checks if a new frame is available and requests a new frame if needed.

        If the current frame is None or too old (older than 1 second),
        the client will resend a NAT request to fetch the latest frame.

        Returns
        -------
        None
        """
        if self.Frame is None or (perf_counter() - self.Frame.Time > 1):
            print("Resend NAT request")
            self.sendCommand(self.NAT_PING, "", self.commandSocket, (self.serverIPAddress, self.commandPort))

    def GetFrameData(self) -> Optional[MoCapFrame]:
        """
        Retrieves the current motion capture frame.

        Returns
        -------
        MoCapFrame or None
            The most recent frame received from the NatNet server, or None if no frame is available yet.
        """
        return self.Frame

    def GetModel(self) -> None:
        """
        Requests the motion capture model description from the NatNet server.

        Returns
        -------
        None
        """
        print("Resend NAT model request")
        self.sendCommand(self.NAT_REQUEST_MODELDEF, "", self.commandSocket, (self.serverIPAddress, self.commandPort))

    def GetModelDescription(self) -> Optional[MoCapModel]:
        """
        Retrieves the current model description.

        Returns
        -------
        MoCapModel or None
            The model description of the motion capture system, or None if unavailable.
        """
        return self.Model
