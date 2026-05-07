"""Socket-based client interface to the MuJoCo server backend.

This module defines the low-level communication layer used to exchange data
with an external MuJoCo simulation server. It includes command and error
constants, lightweight data containers for MuJoCo model and runtime state, and
the `mjInterface` client class used to connect to the server, load models,
control simulation execution, and get or set kinematic, dynamic, sensor, and
visualization data.

Copyright (c) 2024 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

# pylint: disable=invalid-name

import socket
import numpy as np
import struct
from threading import Semaphore

# ----------------------------- MuJoCo API constants -------------------------------------
BUFSZ = 100000

# Communication errors
mjCOM_OK = 0  # success
# Server-to-client errors
mjCOM_BADSIZE = -1  # data has invalid size
mjCOM_BADINDEX = -2  # object has invalid index
mjCOM_BADTYPE = -3  # invalid object type
mjCOM_BADCOMMAND = -4  # unknown command
mjCOM_NOMODEL = -5  # model has not been loaded
mjCOM_CANNOTSEND = -6  # could not send data
mjCOM_CANNOTRECV = -7  # could not receive data
mjCOM_TIMEOUT = -8  # receive timeout
# Client-side errors
mjCOM_NOCONNECTION = -9  # connection not established
mjCOM_CONNECTED = -10  # already connected

# Communication commands
# client-to-server
mjCOM_INFO = 3  # get model info
mjCOM_STEP = 4  # advance simulation in paused mode
mjCOM_UPDATE = 5  # set control, advance, get sensor data
mjCOM_RESET = 6  # reset simulation
mjCOM_PAUSE = 7  # pause simulation
mjCOM_RUN = 8  # run simulation
mjCOM_EQUALITY = 9  # set state of equality constraint
mjCOM_MESSAGE = 10  # show text message
mjCOM_NAME2ID = 11  # convert object name to id
mjCOM_ID2NAME = 12  # convert object id to name
# load model
mjCOM_LOAD = 13  # load MuJoCo model
# get dynamic data
mjCOM_GETSTATE = 16  # state
mjCOM_GETCONTROL = 17  # control
mjCOM_GETAPPLIED = 18  # applied forces
mjCOM_GETONEBODY = 19  # detailed info for one body
mjCOM_GETMOCAP = 20  # mocap
mjCOM_GETDYNAMICS = 21  # output of forward dynamics
mjCOM_GETSENSOR = 22  # sensor data
mjCOM_GETBODY = 23  # body kinematics
mjCOM_GETGEOM = 24  # geom kinematics
mjCOM_GETGEOMSIZE = 25  # geom kinematics
mjCOM_GETSITE = 26  # site kinematics
mjCOM_GETTENDON = 27  # tendon kinematics
mjCOM_GETACTUATOR = 28  # actuator kinematics and force
mjCOM_GETFORCE = 29  # generalized forces
mjCOM_GETCONTACT = 30  # contact info
# set dynamic data
mjCOM_SETSTATE = 48  # state
mjCOM_SETCONTROL = 49  # control
mjCOM_SETAPPLIED = 50  # applied forces
mjCOM_SETONEBODY = 51  # detailed info for one body
mjCOM_SETMOCAP = 52  # mocap
mjCOM_SETGEOMSIZE = 53  # geom size
# get and set rgba static data
mjCOM_GETRGBA = 64  # object rgba
mjCOM_SETRGBA = 65  # object rgba
# simulation control
mjCOM_SCREENSHOT = 66  # make a screenshot request
mjCOM_GETCAMERA = 67  # camera info
mjCOM_GETGLCAMERA = 68  # active (GL) camera info


# Type of geometric shape
mjGEOM_PLANE = 0  # plane
mjGEOM_HFIELD = 1  # height field
mjGEOM_SPHERE = 2  # sphere
mjGEOM_CAPSULE = 3  # capsule
mjGEOM_ELLIPSOID = 4  # ellipsoid
mjGEOM_CYLINDER = 5  # cylinder
mjGEOM_BOX = 6  # box
mjGEOM_MESH = 7  # mesh

# Sensors
# common robotic sensors, attached to a site
mjSENS_TOUCH = 0  # scalar contact normal forces summed over sensor zone
mjSENS_ACCELEROMETER = 1  # 3D linear acceleration, in local frame
mjSENS_VELOCIMETER = 2  # 3D linear velocity, in local frame
mjSENS_GYRO = 3  # 3D angular velocity, in local frame
mjSENS_FORCE = 4  # 3D force between site's body and its parent body
mjSENS_TORQUE = 5  # 3D torque between site's body and its parent body
mjSENS_MAGNETOMETER = 6  # 3D magnetometer
mjSENS_RANGEFINDER = 7  # scalar distance to nearest geom or site along z-axis
# sensors related to scalar joint tendons, actuatorss, tendons, actuators
mjSENS_JOINTPOS = 8  # scalar joint position (hinge and slide only)
mjSENS_JOINTVEL = 9  # scalar joint velocity (hinge and slide only)
mjSENS_TENDONPOS = 10  # scalar tendon position
mjSENS_TENDONVEL = 11  # scalar tendon velocity
mjSENS_ACTUATORPOS = 12  # scalar actuator position
mjSENS_ACTUATORVEL = 13  # scalar actuator velocity
mjSENS_ACTUATORFRC = 14  # scalar actuator force
# sensors related to ball joints
mjSENS_BALLQUAT = 15  # 4D ball joint quaterion
mjSENS_BALLANGVEL = 16  # 3D ball joint angular velocity
# sensors attached to an object h spatial frame: (x)body, geom, site, camerawith spatial frame: (x)body, geom, site, camera
mjSENS_FRAMEPOS = 17  # 3D position
mjSENS_FRAMEQUAT = 18  # 4D unit quaternion orientation
mjSENS_FRAMEXAXIS = 19  # 3D unit vector: x-axis of object's frame
mjSENS_FRAMEYAXIS = 20  # 3D unit vector: y-axis of object's frame
mjSENS_FRAMEZAXIS = 21  # 3D unit vector: z-axis of object's frame
mjSENS_FRAMELINVEL = 22  # 3D linear velocity
mjSENS_FRAMEANGVEL = 23  # 3D angular velocity
mjSENS_FRAMELINACC = 24  # 3D linear acceleration
mjSENS_FRAMEANGACC = 25  # 3D angular acceleration
# sensors related to kinematic srees; attached to a body (which is the subtree root)ubtrees; attached to a body (which is the subtree root)
mjSENS_SUBTREECOM = 26  # 3D center of mass of subtree
mjSENS_SUBTREELINVEL = 27  # 3D linear velocity of subtree
mjSENS_SUBTREEANGMOM = 28  # 3D angular momentum of subtree
# user-defined sensor
mjSENS_USER = 29  # sensor data provided by mjcb_sensor callback

# Type of joint
mjJNT_FREE = 0  # "joint" defining floating body
mjJNT_BALL = 1  # ball joint
mjJNT_SLIDE = 2  # sliding/prismatic joint
mjJNT_HINGE = 3  # hinge joint

# Type of actuator transmission
mjTRN_JOINT = 0  # force on joint
mjTRN_JOINTINPARENT = 1  # force on joint, expressed in parent frame
mjTRN_SLIDERCRANK = 2  # force via slider-crank linkage
mjTRN_TENDON = 3  # force on tendon
mjTRN_SITE = 4  # force on site

# Type of equality constraint
mjEQ_CONNECT = 0  # connect two bodies at a point (ball joint)
mjEQ_WELD = 1  # fix relative position and orientation of two bodies
mjEQ_JOINT = 2  # couple the values of two scalar joints with cubic
mjEQ_TENDON = 3  # couple the lengths of two tendons with cubic
mjEQ_DISTANCE = 4  # fix the contact distance between two geoms


# ----------------------------- MuJoCo API structures ------------------------------------
class mjInfo:
    def __init__(
        self,
        nq,  # number of generalized positions
        nv,  # number of generalized velocities
        na,  # number of actuator activations
        njnt,  # number of joints
        nbody,  # number of bodies
        ngeom,  # number of geoms
        nsite,  # number of sites
        ntendon,  # number of tendons
        nu,  # number of actuators/controls
        neq,  # number of equality constraints
        nkey,  # number of keyframes
        nmocap,  # number of mocap bodies
        nsensor,  # number of sensors
        nsensordata,  # number of elements in sensor data array
        nmat,  # number of materials
        ncam,  # number of cameras
        timestep,  # simulation timestep
        apirate,  # API update rate (same as hxRobotInfo.update_rate)
        sensor_type=None,  # sensor type (mjtSensor)
        sensor_datatype=None,  # type of sensorized object
        sensor_objtype=None,  # type of sensorized object
        sensor_objid=None,  # id of sensorized object
        sensor_dim=None,  # number of (scalar) sensor outputs
        sensor_adr=None,  # address in sensor data array
        sensor_noise=None,  # noise standard deviation
        jnt_type=None,  # joint type (mjtJoint)
        jnt_bodyid=None,  # id of body to which joint belongs
        jnt_qposadr=None,  # address of joint position data in qpos
        jnt_dofadr=None,  # address of joint velocity data in qvel
        jnt_range=None,  # joint range; (0,0): no limits
        geom_type=None,  # geom type (mjtGeom)
        geom_bodyid=None,  # id of body to which geom is attached
        eq_type=None,  # equality constraint type (mjtEq)
        eq_obj1id=None,  # id of constrained object
        eq_obj2id=None,  # id of 2nd constrained object; -1 if not applicable
        actuator_trntype=None,  # transmission type (mjtTrn)
        actuator_trnid=None,  # transmission target id
        actuator_ctrlrange=None,  # actuator control range; (0,0): no limits
    ):
        self.nq = nq
        self.nv = nv
        self.na = na
        self.njnt = njnt
        self.nbody = nbody
        self.ngeom = ngeom
        self.nsite = nsite
        self.ntendon = ntendon
        self.nu = nu
        self.neq = neq
        self.nkey = nkey
        self.nmocap = nmocap
        self.nsensor = nsensor
        self.nsensordata = nsensordata
        self.nmat = nmat
        self.ncam = ncam
        self.timestep = timestep
        self.apirate = apirate
        self.sensor_type = sensor_type if sensor_type is not None else []
        self.sensor_datatype = sensor_datatype if sensor_datatype is not None else []
        self.sensor_objtype = sensor_objtype if sensor_objtype is not None else []
        self.sensor_objid = sensor_objid if sensor_objid is not None else []
        self.sensor_dim = sensor_dim if sensor_dim is not None else []
        self.sensor_adr = sensor_adr if sensor_adr is not None else []
        self.sensor_noise = sensor_noise if sensor_noise is not None else []
        self.jnt_type = jnt_type if jnt_type is not None else []
        self.jnt_bodyid = jnt_bodyid if jnt_bodyid is not None else []
        self.jnt_qposadr = jnt_qposadr if jnt_qposadr is not None else []
        self.jnt_dofadr = jnt_dofadr if jnt_dofadr is not None else []
        self.jnt_range = jnt_range if jnt_range is not None else []
        self.geom_type = geom_type if geom_type is not None else []
        self.geom_bodyid = geom_bodyid if geom_bodyid is not None else []
        self.eq_type = eq_type if eq_type is not None else []
        self.eq_obj1id = eq_obj1id if eq_obj1id is not None else []
        self.eq_obj2id = eq_obj2id if eq_obj2id is not None else []
        self.actuator_trntype = actuator_trntype if actuator_trntype is not None else []
        self.actuator_trnid = actuator_trnid if actuator_trnid is not None else []
        self.actuator_ctrlrange = actuator_ctrlrange if actuator_ctrlrange is not None else []

    def serialize(self):
        data = bytearray(struct.pack("<16i2f", self.nq, self.nv, self.na, self.njnt, self.nbody, self.ngeom, self.nsite, self.ntendon, self.nu, self.neq, self.nkey, self.nmocap, self.nsensor, self.nsensordata, self.nmat, self.ncam, self.timestep, self.apirate))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_type))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_datatype))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_objtype))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_objid))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_dim))
        data.extend(struct.pack(f"{self.nsensor}i", *self.sensor_adr))
        data.extend(struct.pack(f"{self.nsensor}f", *self.sensor_noise))
        data.extend(struct.pack(f"{self.njnt}i", *self.jnt_type))
        data.extend(struct.pack(f"{self.njnt}i", *self.jnt_bodyid))
        data.extend(struct.pack(f"{self.njnt}i", *self.jnt_qposadr))
        data.extend(struct.pack(f"{self.njnt}i", *self.jnt_dofadr))
        for row in self.jnt_range:
            data.extend(struct.pack(f"{len(row)}f", *row))
        data.extend(struct.pack(f"{self.ngeom}i", *self.geom_type))
        data.extend(struct.pack(f"{self.ngeom}i", *self.geom_bodyid))
        data.extend(struct.pack(f"{self.neq}i", *self.eq_type))
        data.extend(struct.pack(f"{self.neq}i", *self.eq_obj1id))
        data.extend(struct.pack(f"{self.neq}i", *self.eq_obj2id))
        data.extend(struct.pack(f"{self.nu}i", *self.actuator_trntype))
        for row in self.actuator_trnid:
            data.extend(struct.pack(f"{len(row)}i", *row))
        for row in self.actuator_ctrlrange:
            data.extend(struct.pack(f"{len(row)}f", *row))
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        off = 0
        (
            nq,
            nv,
            na,
            njnt,
            nbody,
            ngeom,
            nsite,
            ntendon,
            nu,
            neq,
            nkey,
            nmocap,
            nsensor,
            nsensordata,
            nmat,
            ncam,
            timestep,
            apirate,
        ) = struct.unpack("<16i2f", data[off : off + 72])
        off += 72

        sensor_type = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_datatype = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_objtype = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_objid = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_dim = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_adr = list(struct.unpack(f"{nsensor}i", data[off : off + nsensor * 4]))
        off += nsensor * 4
        sensor_noise = np.array(struct.unpack(f"{nsensor}f", data[off : off + nsensor * 4]))
        off += nsensor * 4

        jnt_type = list(struct.unpack(f"{njnt}i", data[off : off + njnt * 4]))
        off += njnt * 4
        jnt_bodyid = list(struct.unpack(f"{njnt}i", data[off : off + njnt * 4]))
        off += njnt * 4
        jnt_qposadr = list(struct.unpack(f"{njnt}i", data[off : off + njnt * 4]))
        off += njnt * 4
        jnt_dofadr = list(struct.unpack(f"{njnt}i", data[off : off + njnt * 4]))
        off += njnt * 4

        jnt_range = []
        for _ in range(njnt):
            row = list(struct.unpack("2f", data[off : off + 8]))
            off += 8
            jnt_range.append(row)
        jnt_range = np.array(jnt_range)

        geom_type = list(struct.unpack(f"{ngeom}i", data[off : off + ngeom * 4]))
        off += ngeom * 4
        geom_bodyid = list(struct.unpack(f"{ngeom}i", data[off : off + ngeom * 4]))
        off += ngeom * 4

        eq_type = list(struct.unpack(f"{neq}i", data[off : off + neq * 4]))
        off += neq * 4
        eq_obj1id = list(struct.unpack(f"{neq}i", data[off : off + neq * 4]))
        off += neq * 4
        eq_obj2id = list(struct.unpack(f"{neq}i", data[off : off + neq * 4]))
        off += neq * 4

        actuator_trntype = list(struct.unpack(f"{nu}i", data[off : off + nu * 4]))
        off += nu * 4

        actuator_trnid = []
        for _ in range(nu):
            row = list(struct.unpack("2i", data[off : off + 8]))
            off += 8
            actuator_trnid.append(row)

        actuator_ctrlrange = []
        for _ in range(nu):
            row = list(struct.unpack("2f", data[off : off + 8]))
            off += 8
            actuator_ctrlrange.append(row)
        actuator_ctrlrange = np.array(actuator_ctrlrange)

        return mjInfo(
            nq,
            nv,
            na,
            njnt,
            nbody,
            ngeom,
            nsite,
            ntendon,
            nu,
            neq,
            nkey,
            nmocap,
            nsensor,
            nsensordata,
            nmat,
            ncam,
            timestep,
            apirate,
            sensor_type,
            sensor_datatype,
            sensor_objtype,
            sensor_objid,
            sensor_dim,
            sensor_adr,
            sensor_noise,
            jnt_type,
            jnt_bodyid,
            jnt_qposadr,
            jnt_dofadr,
            jnt_range,
            geom_type,
            geom_bodyid,
            eq_type,
            eq_obj1id,
            eq_obj2id,
            actuator_trntype,
            actuator_trnid,
            actuator_ctrlrange,
        )

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjState:
    def __init__(self, nq, nv, na, time, qpos=None, qvel=None, act=None):
        self
        self.nq = nq  # number of generalized positions
        self.nv = nv  # number of generalized velocities
        self.na = na  # number of actuator activations
        self.time = time  # simulation time
        self.qpos = qpos if qpos is not None else []  # generalized positions
        self.qvel = qvel if qvel is not None else []  # generalized velocities
        self.act = act if act is not None else []  # actuator activations

    def serialize(self):
        data = struct.pack("iii", self.nq, self.nv, self.na)
        # data += struct.pack("f", self.time) # Not send!
        data += struct.pack(f"{self.nq}f", *self.qpos)
        data += struct.pack(f"{self.nv}f", *self.qvel)
        data += struct.pack(f"{self.na}f", *self.act)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into a mjState instance
        nq, nv, na = struct.unpack("iii", data[:12])
        time = struct.unpack("f", data[12:16])[0]
        off = 16
        qpos = np.array(struct.unpack(f"{nq}f", data[off : off + nq * 4]))
        off += nq * 4
        qvel = np.array(struct.unpack(f"{nv}f", data[off : off + nv * 4]))
        off += nv * 4
        act = np.array(struct.unpack(f"{na}f", data[off : off + na * 4]))
        return mjState(nq, nv, na, time, qpos, qvel, act)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjControl:
    def __init__(self, nu=0, time=0.0, ctrl=None):
        self.nu = nu  # number of actuators
        self.time = time  # simulation time
        self.ctrl = ctrl if ctrl is not None else []  # control signals

    def serialize(self):
        data = bytearray(struct.pack("<i", self.nu))
        # data.extend(struct.pack("f", *self.time))
        data.extend(struct.pack(f"{self.nu}f", *self.ctrl))
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        nu, time = struct.unpack("<if", data[:8])
        ctrl = np.array(struct.unpack(f"{nu}f", data[8:]))
        return mjControl(nu, time, ctrl)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjApplied:
    def __init__(self, nv=0, nbody=0, time=0, qfrc_applied=None, xfrc_applied=None):
        self.nv = nv  # number of generalized velocities
        self.nbody = nbody  # number of bodies
        self.time = time  # simulation time
        self.qfrc_applied = qfrc_applied if qfrc_applied is not None else []  # generalized forces
        self.xfrc_applied = xfrc_applied if xfrc_applied is not None else []  #

    def serialize(self):
        # Serialize the applied attributes into a bytes object
        data = struct.pack("ii", self.nv, self.nbody)
        # data += struct.pack("f", self.time)
        data += struct.pack(f"{len(self.qfrc_applied)}f", *self.qfrc_applied)

        for i in range(self.nbody):
            data += struct.pack(f"{6}f", *self.xfrc_applied[i])

        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjApplied instance
        nv, nbody = struct.unpack("ii", data[:8])
        time = struct.unpack("f", data[8:12])[0]

        qfrc_applied = np.array(struct.unpack(f"{nv}f", data[12 : 12 + nv * 4]))

        xfrc_applied = []
        offset = 12 + nv * 4
        for _ in range(nbody):
            xfrc_applied.append(list(struct.unpack(f"{6}f", data[offset : offset + 6 * 4])))
            offset += 6 * 4
        xfrc_applied = np.array(xfrc_applied)

        return mjApplied(nv, nbody, time, qfrc_applied, xfrc_applied)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjOneBody:
    def __init__(self, bodyid=0, isfloating=0, time=0, linacc=0, angacc=0, contactforce=0, pos=None, quat=None, linvel=None, angvel=None, force=None, torque=None):
        self.bodyid = bodyid  # body id, provided by user
        self.isfloating = isfloating  # 1 if body is floating, 0 otherwise
        self.time = time  # simulation time
        self.linacc = linacc  # linear acceleration
        self.angacc = angacc  # angular acceleration
        self.contactforce = contactforce  # net force from all contacts on this body
        self.pos = pos if pos is not None else []  # position
        self.quat = quat if quat is not None else []  # orientation quaternion
        self.linvel = linvel if linvel is not None else []  # linear velocity
        self.angvel = angvel if angvel is not None else []  # angular velocity
        self.force = force if force is not None else []  # Cartesian force applied to body CoM
        self.torque = torque if torque is not None else []  # Cartesian torque applied to body

    def serialize(self):
        # Serialize the mjOneBody attributes into a bytes object
        data = struct.pack("i", self.bodyid)
        # data = struct.pack("i", self.isfloating)
        # data = struct.pack("f", self.time)
        # data += struct.pack("3f", *self.linacc)
        # data += struct.pack("3f", *self.angacc)
        # data += struct.pack("3f", *self.contactforce)
        data += struct.pack("3f", *self.pos)
        data += struct.pack("4f", *self.quat)
        data += struct.pack("3f", *self.linvel)
        data += struct.pack("3f", *self.angvel)
        data += struct.pack("3f", *self.force)
        data += struct.pack("3f", *self.torque)
        return data

    def deserialize(self, data):
        if not data:
            return None
        # Deserialize the bytes object into an mjOneBody instance
        self.isfloating, self.time = struct.unpack("if", data[:8])
        self.linacc = np.array(struct.unpack("3f", data[8:20]))
        self.angacc = np.array(struct.unpack("3f", data[20:32]))
        self.contactforce = np.array(struct.unpack("3f", data[32:44]))
        self.pos = np.array(struct.unpack("3f", data[44:56]))
        self.quat = np.array(struct.unpack("4f", data[56:72]))
        self.linvel = np.array(struct.unpack("3f", data[72:84]))
        self.angvel = np.array(struct.unpack("3f", data[84:96]))
        self.force = np.array(struct.unpack("3f", data[96:108]))
        self.torque = np.array(struct.unpack("3f", data[108:120]))
        return self

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjMocap:
    def __init__(self, nmocap, time, pos=None, quat=None):
        self.nmocap = nmocap  # number of mocap bodies
        self.time = time  # simulation time
        self.pos = pos if pos is not None else []  # positions
        self.quat = quat if quat is not None else []  # quaternion orientations

    def serialize(self):
        # Serialize the mjMocap attributes into a bytes object
        data = struct.pack("i", self.nmocap)
        # data += struct.pack("f", self.time)
        for i in range(self.nmocap):
            data += struct.pack("3f", *self.pos[i])
        for i in range(self.nmocap):
            data += struct.pack("4f", *self.quat[i])
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjMocap instance
        nmocap, time = struct.unpack("if", data[:8])
        pos = []
        quat = []
        offset = 8
        for i in range(nmocap):
            pos.append(np.array(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
        for i in range(nmocap):
            quat.append(np.array(struct.unpack("4f", data[offset : offset + 16])))
            offset += 16
        pos = np.array(pos)
        quat = np.array(quat)
        return mjMocap(nmocap, time, pos, quat)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjDynamics:
    def __init__(self, nv, na, time, qacc=None, actdot=None):
        self.nv = nv  # number of generalized velocities
        self.na = na  # number of actuator activations
        self.time = time  # simulation time
        self.qacc = qacc if qacc is not None else []  # generalized accelerations
        self.actdot = actdot if actdot is not None else []  # time-derivatives of actuator activations

    def serialize(self):
        # Serialize the mjDynamics attributes into a bytes object
        data = struct.pack("ii", self.nv, self.na)
        data += struct.pack("f", self.time)
        data += struct.pack(f"{len(self.qacc)}f", *self.qacc)
        data += struct.pack(f"{len(self.actdot)}f", *self.actdot)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjDynamics instance
        nv, na = struct.unpack("ii", data[:8])
        time = struct.unpack("f", data[8:12])[0]
        qacc = np.array(struct.unpack(f"{nv}f", data[12 : 12 + nv * 4]))
        actdot = np.array(struct.unpack(f"{na}f", data[12 + nv * 4 :]))
        return mjDynamics(nv, na, time, qacc, actdot)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjSensor:
    def __init__(self, nsensordata, time, sensordata=None):
        self.nsensordata = nsensordata  # size of sensor data array
        self.time = time  # simulation time
        self.sensordata = sensordata if sensordata is not None else []  # sensor data array

    def serialize(self):
        # Serialize the mjSensor attributes into a bytes object
        data = struct.pack("if", self.nsensordata, self.time)
        data += struct.pack(f"{len(self.sensordata)}f", *self.sensordata)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjSensor instance
        nsensordata, time = struct.unpack("if", data[:8])
        sensordata = np.array(struct.unpack(f"{nsensordata}f", data[8:]))
        return mjSensor(nsensordata, time, sensordata)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjBody:
    def __init__(self, nbody, time, pos=None, mat=None):
        self.nbody = nbody  # number of bodies
        self.time = time  # simulation time
        self.pos = pos if pos is not None else []  # positions
        self.mat = mat if mat is not None else []  # frame orientations

    def serialize(self):
        # Serialize the mjBody attributes into a bytes object
        data = struct.pack("if", self.nbody, self.time)
        for i in range(self.nbody):
            data += struct.pack("3f", *self.pos[i])
        for i in range(self.nbody):
            data += struct.pack("9f", *self.mat[i])
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjBody instance
        nbody, time = struct.unpack("if", data[:8])
        pos = []
        mat = []
        offset = 8
        for _ in range(nbody):
            pos.append(list(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
        for _ in range(nbody):
            mat.append(list(struct.unpack("9f", data[offset : offset + 36])))
            offset += 36
        pos = np.array(pos)
        mat = np.array(mat)
        return mjBody(nbody, time, pos, mat)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjGeom:
    def __init__(self, ngeom, time, pos=None, mat=None):
        self.ngeom = ngeom  # number of geoms
        self.time = time  # simulation time
        self.pos = pos if pos is not None else []  # positions
        self.mat = mat if mat is not None else []  # frame orientations

    def serialize(self):
        # Serialize the mjGeom attributes into a bytes object
        data = struct.pack("if", self.ngeom, self.time)
        for i in range(self.ngeom):
            data += struct.pack("3f", *self.pos[i])
        for i in range(self.ngeom):
            data += struct.pack("9f", *self.mat[i])
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjGeom instance
        ngeom, time = struct.unpack("if", data[:8])
        pos = []
        mat = []
        offset = 8
        for _ in range(ngeom):
            pos.append(list(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
        for _ in range(ngeom):
            mat.append(list(struct.unpack("9f", data[offset : offset + 36])))
            offset += 36
        pos = np.array(pos)
        mat = np.array(mat)
        return mjGeom(ngeom, time, pos, mat)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjSite:
    def __init__(self, nsite, time, pos=None, mat=None):
        self.nsite = nsite  # number of sites
        self.time = time  # simulation time
        self.pos = pos if pos is not None else []  # positions
        self.mat = mat if mat is not None else []  # frame orientations

    def serialize(self):
        # Serialize the mjSite attributes into a bytes object
        data = struct.pack("if", self.nsite, self.time)
        for i in range(self.nsite):
            data += struct.pack("3f", *self.pos[i])
        for i in range(self.nsite):
            data += struct.pack("9f", *self.mat[i])
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjSite instance
        nsite, time = struct.unpack("if", data[:8])
        pos = []
        mat = []
        offset = 8
        for _ in range(nsite):
            pos.append(list(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
        for _ in range(nsite):
            mat.append(list(struct.unpack("9f", data[offset : offset + 36])))
            offset += 36
        pos = np.array(pos)
        mat = np.array(mat)
        return mjSite(nsite, time, pos, mat)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjTendon:
    def __init__(self, ntendon, time, length=None, velocity=None):
        self.ntendon = ntendon  # number of tendons
        self.time = time  # simulation time
        self.length = length if length is not None else []  # tendon lengths
        self.velocity = velocity if velocity is not None else []  # tendon velocities

    def serialize(self):
        # Serialize the mjTendon attributes into a bytes object
        data = struct.pack("if", self.ntendon, self.time)
        data += struct.pack(f"{len(self.length)}f", *self.length)
        data += struct.pack(f"{len(self.velocity)}f", *self.velocity)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjTendon instance
        ntendon, time = struct.unpack("if", data[:8])
        length = np.array(struct.unpack(f"{ntendon}f", data[8 : 8 + 4 * ntendon]))
        offset = 8 + 4 * ntendon
        velocity = np.array(struct.unpack(f"{ntendon}f", data[offset:]))
        return mjTendon(ntendon, time, length, velocity)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjActuator:
    def __init__(self, nu, time, length=None, velocity=None, force=None):
        self.nu = nu  # number of actuators
        self.time = time  # simulation time
        self.length = length if length is not None else []  # actuator lengths
        self.velocity = velocity if velocity is not None else []  # actuator velocities
        self.force = force if force is not None else []  # actuator forces

    def serialize(self):
        # Serialize the mjActuator attributes into a bytes object
        data = struct.pack("if", self.nu, self.time)
        data += struct.pack(f"{len(self.length)}f", *self.length)
        data += struct.pack(f"{len(self.velocity)}f", *self.velocity)
        data += struct.pack(f"{len(self.force)}f", *self.force)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjActuator instance
        nu, time = struct.unpack("if", data[:8])
        length = np.array(struct.unpack(f"{nu}f", data[8 : 8 + 4 * nu]))
        offset = 8 + 4 * nu
        velocity = np.array(struct.unpack(f"{nu}f", data[offset : offset + 4 * nu]))
        offset += 4 * nu
        force = np.array(struct.unpack(f"{nu}f", data[offset:]))
        return mjActuator(nu, time, length, velocity, force)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjForce:
    def __init__(self, nv, time, nonconstraint=None, constraint=None):
        self.nv = nv  # number of generalized velocities/forces
        self.time = time  # simulation time
        self.nonconstraint = nonconstraint if nonconstraint is not None else []  # sum of all non-constraint forces
        self.constraint = constraint if constraint is not None else []  # constraint forces (including contacts)

    def serialize(self):
        # Serialize the mjForce attributes into a bytes object
        data = struct.pack("if", self.nv, self.time)
        data += struct.pack(f"{len(self.nonconstraint)}f", *self.nonconstraint)
        data += struct.pack(f"{len(self.constraint)}f", *self.constraint)
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjForce instance
        nv, time = struct.unpack("if", data[:8])
        nonconstraint = np.array(struct.unpack(f"{nv}f", data[8 : 8 + 4 * nv]))
        offset = 8 + 4 * nv
        constraint = np.array(struct.unpack(f"{nv}f", data[offset:]))
        return mjForce(nv, time, nonconstraint, constraint)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjContact:
    def __init__(self, ncon, time, dist=None, pos=None, frame=None, force=None, geom1=None, geom2=None):
        self.ncon = ncon  # number of detected contacts
        self.time = time  # simulation time
        self.dist = dist if dist is not None else []  # contact normal distance
        self.pos = pos if pos is not None else []  # contact position in world frame
        self.frame = frame if frame is not None else []  # contact frame relative to world frame (0-2: normal)
        self.force = force if force is not None else []  # contact force in contact frame
        self.geom1 = geom1 if geom1 is not None else []  # id of 1st contacting geom
        self.geom2 = geom2 if geom2 is not None else []  # id of 2nd contacting geom (force: 1st -> 2nd)

    def serialize(self):
        # Serialize the mjContact attributes into a bytes object
        data = struct.pack("if", self.ncon, self.time)
        data += struct.pack(f"{len(self.dist)}f", *self.dist)
        for i in range(self.ncon):
            data += struct.pack("3f", *self.pos[i])
            data += struct.pack("9f", *self.frame[i])
            data += struct.pack("3f", *self.force[i])
            data += struct.pack("ii", self.geom1[i], self.geom2[i])
        return data

    @staticmethod
    def deserialize(data):
        if not data:
            return None
        # Deserialize the bytes object into an mjContact instance
        ncon, time = struct.unpack("if", data[:8])
        offset = 8
        dist = np.array(struct.unpack(f"{ncon}f", data[offset : offset + 4 * ncon]))
        offset += 4 * ncon
        pos = []
        frame = []
        force = []
        geom1 = []
        geom2 = []
        for _ in range(ncon):
            pos.append(list(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
            frame.append(list(struct.unpack("9f", data[offset : offset + 36])))
            offset += 36
            force.append(list(struct.unpack("3f", data[offset : offset + 12])))
            offset += 12
            geom1_val, geom2_val = struct.unpack("ii", data[offset : offset + 8])
            geom1.append(geom1_val)
            geom2.append(geom2_val)
            offset += 8
        pos = np.array(pos)
        frame = np.array(frame)
        force = np.array(force)
        geom1 = np.array(geom1)
        geom2 = np.array(geom2)
        return mjContact(ncon, time, dist, pos, frame, force, geom1, geom2)

    def __str__(self):
        s = ""
        for field_name in self.__dict__:
            s += "\n  {0:s}: {1:}".format(field_name, self.__dict__[field_name])
        return "{0:s}\n".format(s)


class mjCamera:
    def __init__(self, ncam, time, cam_xpos, cam_xmat, cam_mode, cam_bodyid, cam_targetbodyid, cam_pos, cam_quat, cam_poscom0, cam_pos0, cam_mat0, cam_fovy, cam_ipd):
        self.ncam = ncam  # number of camera bodies
        self.time = time  # simulation time
        self.cam_xpos = cam_xpos if cam_xpos is not None else []  # Cartesian camera position (ncam x 3)
        self.cam_xmat = cam_xmat if cam_xmat is not None else []  # Cartesian camera orientation (ncam x 9)
        self.cam_mode = cam_mode if cam_mode is not None else []  # camera tracking mode (mjtCamLight) (ncam x 1)
        self.cam_bodyid = cam_bodyid if cam_bodyid is not None else []  # id of camera's body (ncam x 1)
        self.cam_targetbodyid = cam_targetbodyid  # id of targeted body; -1: none (ncam x 1)
        self.cam_pos = cam_pos if cam_pos is not None else []  # position rel. to body frame (ncam x 3)
        self.cam_quat = cam_quat if cam_quat is not None else []  # orientation rel. to body frame (ncam x 4)
        self.cam_poscom0 = cam_poscom0 if cam_poscom0 is not None else []  # global position rel. to sub-com in qpos0 (ncam x 3)
        self.cam_pos0 = cam_pos0 if cam_pos0 is not None else []  # global position rel. to body in qpos0 (ncam x 3)
        self.cam_mat0 = cam_mat0 if cam_mat0 is not None else []  # global orientation in qpos0 (ncam x 9)
        self.cam_fovy = cam_fovy if cam_fovy is not None else []  # y-field of view (deg) (ncam x 1)
        self.cam_ipd = cam_ipd if cam_ipd is not None else []  # inter-pupilary distance (ncam x 1)

    def serialize(self):
        data = struct.pack("if", self.ncam, self.time)
        data += struct.pack(f"{self.ncam * 3}f", *self.cam_xpos)
        data += struct.pack(f"{self.ncam * 9}f", *self.cam_xmat)
        data += struct.pack(f"{self.ncam}i", *self.cam_mode)
        data += struct.pack(f"{self.ncam}i", *self.cam_bodyid)
        data += struct.pack(f"{self.ncam}i", *self.cam_targetbodyid)
        data += struct.pack(f"{self.ncam * 3}f", *self.cam_pos)
        data += struct.pack(f"{self.ncam * 4}f", *self.cam_quat)
        data += struct.pack(f"{self.ncam * 3}f", *self.cam_poscom0)
        data += struct.pack(f"{self.ncam * 3}f", *self.cam_pos0)
        data += struct.pack(f"{self.ncam * 9}f", *self.cam_mat0)
        data += struct.pack(f"{self.ncam}f", *self.cam_fovy)
        data += struct.pack(f"{self.ncam}f", *self.cam_ipd)
        return data

    @staticmethod
    def deserialize(data):
        offset = 0
        ncam, time = struct.unpack("if", data[offset : offset + 8])
        offset += 8
        cam_xpos = np.array(struct.unpack(f"{ncam * 3}f", data[offset : offset + ncam * 12]))
        offset += ncam * 12
        cam_xmat = np.array(struct.unpack(f"{ncam * 9}f", data[offset : offset + ncam * 36]))
        offset += ncam * 36
        cam_mode = np.array(struct.unpack(f"{ncam}i", data[offset : offset + ncam * 4]))
        offset += ncam * 4
        cam_bodyid = np.array(struct.unpack(f"{ncam}i", data[offset : offset + ncam * 4]))
        offset += ncam * 4
        cam_targetbodyid = np.array(struct.unpack(f"{ncam}i", data[offset : offset + ncam * 4]))
        offset += ncam * 4
        cam_pos = np.array(struct.unpack(f"{ncam * 3}f", data[offset : offset + ncam * 12]))
        offset += ncam * 12
        cam_quat = np.array(struct.unpack(f"{ncam * 4}f", data[offset : offset + ncam * 16]))
        offset += ncam * 16
        cam_poscom0 = np.array(struct.unpack(f"{ncam * 3}f", data[offset : offset + ncam * 12]))
        offset += ncam * 12
        cam_pos0 = np.array(struct.unpack(f"{ncam * 3}f", data[offset : offset + ncam * 12]))
        offset += ncam * 12
        cam_mat0 = np.array(struct.unpack(f"{ncam * 9}f", data[offset : offset + ncam * 36]))
        offset += ncam * 36
        cam_fovy = np.array(struct.unpack(f"{ncam}f", data[offset : offset + ncam * 4]))
        offset += ncam * 4
        cam_ipd = np.array(struct.unpack(f"{ncam}f", data[offset : offset + ncam * 4]))

        return mjCamera(ncam, time, cam_xpos, cam_xmat, cam_mode, cam_bodyid, cam_targetbodyid, cam_pos, cam_quat, cam_poscom0, cam_pos0, cam_mat0, cam_fovy, cam_ipd)


class mjGLCamera:
    def __init__(self, nglcam, time, fixedcamid, type, trackbodyid, lookat, distance, azimuth, elevation, pos, forward, up, frustum_center, frustum_bottom, frustum_top, frustum_near, frustum_far):
        self.nglcam = nglcam
        self.time = time
        self.fixedcamid = fixedcamid if fixedcamid is not None else []
        self.type = type if type is not None else []
        self.trackbodyid = trackbodyid if trackbodyid is not None else []
        self.lookat = lookat if lookat is not None else []
        self.distance = distance if distance is not None else []
        self.azimuth = azimuth if azimuth is not None else []
        self.elevation = elevation if elevation is not None else []
        self.pos = pos if pos is not None else []
        self.forward = forward if forward is not None else []
        self.up = up if up is not None else []
        self.frustum_center = frustum_center if frustum_center is not None else []
        self.frustum_bottom = frustum_bottom if frustum_bottom is not None else []
        self.frustum_top = frustum_top if frustum_top is not None else []
        self.frustum_near = frustum_near if frustum_near is not None else []
        self.frustum_far = frustum_far if frustum_far is not None else []

    def serialize(self):
        data = struct.pack("if", self.nglcam, self.time)
        data += struct.pack(f"{self.nglcam}i", *self.fixedcamid)
        data += struct.pack(f"{self.nglcam}i", *self.type)
        data += struct.pack(f"{self.nglcam}i", *self.trackbodyid)
        data += struct.pack(f"{self.nglcam * 3}f", *self.lookat)
        data += struct.pack(f"{self.nglcam}f", *self.distance)
        data += struct.pack(f"{self.nglcam}f", *self.azimuth)
        data += struct.pack(f"{self.nglcam}f", *self.elevation)
        data += struct.pack(f"{self.nglcam * 3}f", *self.pos)
        data += struct.pack(f"{self.nglcam * 3}f", *self.forward)
        data += struct.pack(f"{self.nglcam * 3}f", *self.up)
        data += struct.pack(f"{self.nglcam}f", *self.frustum_center)
        data += struct.pack(f"{self.nglcam}f", *self.frustum_bottom)
        data += struct.pack(f"{self.nglcam}f", *self.frustum_top)
        data += struct.pack(f"{self.nglcam}f", *self.frustum_near)
        data += struct.pack(f"{self.nglcam}f", *self.frustum_far)
        return data

    @staticmethod
    def deserialize(data):
        offset = 0
        nglcam, time = struct.unpack("if", data[offset : offset + 8])
        offset += 8
        fixedcamid = np.array(struct.unpack(f"{nglcam}i", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        type = np.array(struct.unpack(f"{nglcam}i", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        trackbodyid = np.array(struct.unpack(f"{nglcam}i", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        lookat = np.array(struct.unpack(f"{nglcam * 3}f", data[offset : offset + nglcam * 12]))
        offset += nglcam * 12
        distance = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        azimuth = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        elevation = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        pos = np.array(struct.unpack(f"{nglcam * 3}f", data[offset : offset + nglcam * 12]))
        offset += nglcam * 12
        forward = np.array(struct.unpack(f"{nglcam * 3}f", data[offset : offset + nglcam * 12]))
        offset += nglcam * 12
        up = np.array(struct.unpack(f"{nglcam * 3}f", data[offset : offset + nglcam * 12]))
        offset += nglcam * 12
        frustum_center = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        frustum_bottom = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        frustum_top = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        frustum_near = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))
        offset += nglcam * 4
        frustum_far = np.array(struct.unpack(f"{nglcam}f", data[offset : offset + nglcam * 4]))

        return mjGLCamera(nglcam, time, fixedcamid, type, trackbodyid, lookat, distance, azimuth, elevation, pos, forward, up, frustum_center, frustum_bottom, frustum_top, frustum_near, frustum_far)


mujoco_semaphore = Semaphore(1)


# ----------------------------- MuJoCo API object ----------------------------------------
class mjInterface(object):
    """Creation of MuJoCo interface"""

    def __init__(self, host="localhost", port=50000):
        self._soc = None
        self._host = host
        self._port = port
        self._connected = 0
        self._lastError = 0
        self.return_codes = {
            0: "OK",
            -1: "BADSIZE",
            -2: "BADINDEX",
            -3: "BADTYPE",
            -4: "BADCOMMAND",
            -5: "NOMODEL",
            -6: "CANNOTSEND",
            -7: "CANNOTRECV",
            -8: "TIMEOUT",
            -9: "NOCONNECTION",
            -10: "CONNECTED",
        }

        self.geom_type = (
            "PLANE",
            "HFIELD",
            "SPHERE",
            "CAPSULE",
            "ELLIPSOID",
            "CYLINDER",
            "BOX",
            "MESH",
        )

        self.sensor_type = (
            "TOUCH",  # scalar contact normal forces summed over sensor zone
            "ACCELEROMETER",  # 3D linear acceleration', in local frame
            "VELOCIMETER",  # 3D linear velocity', in local frame
            "GYRO",  # 3D angular velocity', in local frame
            "FORCE",  # 3D force between site's body and its parent body
            "TORQUE",  # 3D torque between site's body and its parent body
            "MAGNETOMETER",  # 3D magnetometer
            "RANGEFINDER",  # distance geom along the positive size Z-axis
            "JOINTPOS",  # scalar joint position (hinge and slide only)
            "JOINTVEL",  # scalar joint velocity (hinge and slide only)
            "TENDONPOS",  # scalar tendon position
            "TENDONVEL",  # scalar tendon velocity
            "ACTUATORPOS",  # scalar actuator position
            "ACTUATORVEL",  # scalar actuator velocity
            "ACTUATORFRC",  # scalar actuator force
            "BALLQUAT",  # 4D ball joint quaterion
            "BALLANGVEL",  # 3D ball joint angular velocity
            "FRAMEPOS",  # 3D position
            "FRAMEQUAT",  # 4D unit quaternion orientation
            "FRAMEXAXIS",  # 3D unit vector: x-axis of object's frame
            "FRAMEYAXIS",  # 3D unit vector: y-axis of object's frame
            "FRAMEZAXIS",  # 3D unit vector: z-axis of object's frame
            "FRAMELINVEL",  # 3D linear velocity
            "FRAMEANGVEL",  # 3D angular velocity
            "FRAMELINACC",  # 3D linear acceleration
            "FRAMEANGACC",  # 3D angular acceleration
            "SUBTREECOM",  # 3D center of mass of subtree
            "SUBTREELINVEL",  # 3D linear velocity of subtree
            "SUBTREEANGMOM",  # 3D angular momentum of subtree
            "USER",
        )
        self.joint_type = ("FREE", "BALL", "SLIDE", "HINGE")

        self.transmission_type = (
            "JOINT",
            "JOINTINPARENT",
            "SLIDERCRANK",
            "TENDON",
            "SITE",
        )

        self.constraint_type = (
            "CONNECT",  # connect two bodies at a point (ball joint)
            "WELD",  # fix relative position and orientation of two bodies
            "JOINT",  # couple the values of two scalar joints with cubic
            "TENDON",  # couple the lengths of two tendons with cubic
            "DISTANCE",
        )  # fix the contact distance between two geoms

    onebody = mjOneBody()

    def mjSetError(self, err):
        self._lastError = err

    # Socket
    def mjSend(self, cmd, data=None):
        if not self._soc:
            self.mjSetError(mjCOM_CANNOTSEND)
            raise RuntimeError("No communication to server")

        # Flush input buffer
        """
        self._soc.settimeout(0.001)
        try:
            self._soc.recv(1024)
        except socket.timeout:
            pass
        except Exception as e:
            self.mjSetError(mjCOM_CANNOTSEND)
            raise RuntimeError("Send: " + str(e))
        """

        # Prepare buffer
        if not mujoco_semaphore.acquire(timeout=10):
            raise RuntimeError(f"No mujoco_semaphore ({mujoco_semaphore._value})")

        if data:
            _buffer = bytearray(struct.pack("<2i", cmd, len(data)))
            _buffer.extend(data)
        else:
            _buffer = bytearray(struct.pack("<2i", cmd, 0))
        # Try to send
        try:
            self._soc.sendall(_buffer)
        except Exception as e:
            self._soc.close()
            self._soc = None
            self.mjSetError(mjCOM_CANNOTSEND)
            raise RuntimeError("Send: " + str(e))
        else:
            self.mjSetError(mjCOM_OK)

    def mjRecv(self):
        if self._lastError != mjCOM_OK:
            raise RuntimeError(self.return_codes[self._lastError])

        # Get message header, disconnect on error or timeout
        try:
            header = self._soc.recv(8)
        except socket.timeout:
            self._soc.close()
            self._lastError = mjCOM_TIMEOUT
            raise RuntimeError("Timeout - could not get data")
        except Exception as e:
            self._soc.close()
            self._lastError = mjCOM_CANNOTRECV
            raise RuntimeError("Recieve: " + str(e))

        cmd, size = struct.unpack("2i", header)
        self.mjSetError(cmd)

        # Get message data, disconnect on error
        if size > 0 and size <= BUFSZ:
            try:
                data = self._soc.recv(size)
                mujoco_semaphore.release()
                return data
            except Exception as e:
                self._soc.close()
                self.mjSetError(mjCOM_CANNOTRECV)
                raise RuntimeError("Recieve: " + str(e))
        elif size:
            self._soc.close()
            self.mjSetError(mjCOM_BADSIZE)
            raise RuntimeError("Received data has invalid size")
        mujoco_semaphore.release()

    def mjGetAck(self):
        if self._lastError != mjCOM_OK:
            raise RuntimeError(self.return_codes[self._lastError])

        # try to receive acknowledgment
        self._soc.settimeout(0.1)
        try:
            _data = self._soc.recv(8)
        except socket.timeout:
            self._soc.close()
            self.mjSetError(mjCOM_TIMEOUT)
            raise RuntimeError("Timeout - could not get acknowledge data")

        # Check if data was received successfully
        if not _data:
            self._soc.close()
            self.mjSetError(mjCOM_CANNOTRECV)
            raise RuntimeError("Could not get acknowledge data")

        _cmd, _size = struct.unpack("ii", _data)

        # data size should be 0
        if _size != 0:
            self._soc.close()
            self.mjSetError(mjCOM_CANNOTRECV)
            raise RuntimeError("Bad acknowledgement")
        else:
            self.mjSetError(_cmd)
        mujoco_semaphore.release()

    # Get functions
    def mj_get_state(self) -> mjState:
        """Read model state

        Returns
        -------
        -------
        structure mjState
            nq : int
                number of data in qpos
            nv : int
                number of data in qvel
            na : int
                number of data in act
            time : float)
                simulation time
            qpos : array [nq]
                generalized positions
            qvel : array [nv]
                generalized velocities
            act : array [na]
                actuator activations
        """
        self.mjSend(mjCOM_GETSTATE)
        _data = self.mjRecv()
        _out = mjState.deserialize(_data)
        return _out

    def mj_get_control(self) -> mjControl:
        """Read control signals

        Returns
        -------
        --------
        structure mjControl
            nu (int)        : number of data in ctrl
            time (float)    : simulation time
            ctrl[nu] (float): control array
        """
        self.mjSend(mjCOM_GETCONTROL)
        _data = self.mjRecv()
        _out = mjControl.deserialize(_data)
        return _out

    def mj_get_applied(self) -> mjApplied:
        """Read applied forces

        Returns
        -------
        --------
        structure mjApplied
            nv (int)        : number of data in forces
            nbody (int)     : id of body
            time (float)    : simulation time
            qfrc[nv] (float): applied generalized forces
            xfrc[nv] (float): Cartesian F/T applied to body
        """
        self.mjSend(mjCOM_GETAPPLIED)
        _data = self.mjRecv()
        _out = mjApplied.deserialize(_data)
        return _out

    def mj_get_onebody(self, bodyid: int) -> mjOneBody:
        """Read information about one body

        Parameters
        ----------
        -----------
        bodyid (int)  : body id, provided by user

        Returns
        -------
        --------
        structure mjOneBody
           bodyid; (int)    : body id, provided by user
           get only:
           isfloating (int) : 1 if body is floating, 0 otherwise
           time (float)     : simulation time
           linacc[3] (float): linear acceleration
           angacc[3] (float): angular acceleration
           contactforce[3] (float) : net force from all contacts on this body
        get for all bodies; set for floating bodies only:
           pos[3] (float)   : position
           quat[4] (float)  : orientation quaternion
           linvel[3] (float): linear velocity
           angvel[3] (float): angular velocity
        get and set for all bodies:
           force[3] (float) : Cartesian force applied to body CoM
           torque[3] (float): Cartesian torque applied to body
        """
        self.mjSend(mjCOM_GETONEBODY, struct.pack("i", bodyid))
        _data = self.mjRecv()
        self.onebody.deserialize(_data)
        self.onebody.bodyid = bodyid
        return self.onebody

    def mj_get_mocap(self) -> mjMocap:
        """Read mocaps

        Returns
        -------
        --------
        structure mjMocap
            nmocap (int)           : number of mocap bodies
            time (float)           : simulation time
            pos[nmocap][3] (float) : positions
            quat[nmocap][4] (float): quaternion orientations
        """
        self.mjSend(mjCOM_GETMOCAP)
        _data = self.mjRecv()
        _out = mjMocap.deserialize(_data)
        return _out

    def mj_get_dynamics(self) -> mjDynamics:
        """Read forward dynamics

        Returns
        -------
        --------
        structure mjDynamics
            nv (int)          : number of generalized velocities
            na (int)          : number of actuator activations
            time (float)      : simulation time
            qacc[nv] (float)  : generalized accelerations
            actdot[na] (float): time-derivatives of actuator activations
        """
        self.mjSend(mjCOM_GETDYNAMICS)
        _data = self.mjRecv()
        _out = mjDynamics.deserialize(_data)
        return _out

    def mj_get_sensor(self) -> mjSensor:
        """
        Read sensor data from the simulator.

        Use the sensor descriptors available in ``mjInfo`` to decode the
        returned values.

        Returns
        -------
        mjSensor
            Sensor data structure with:

            - ``nsensordata``: number of sensor values
            - ``time``: simulation time
            - ``sensordata``: sensor data array
        """
        self.mjSend(mjCOM_GETSENSOR)
        _data = self.mjRecv()
        _out = mjSensor.deserialize(_data)
        return _out

    def mj_get_body(self) -> mjBody:
        """Read body positions

        Returns
        -------
        --------
        structure mjBody
            nbody (int)          : number of bodies
            time (float)         : simulation time
            pos[nbody][3] (float): positions
            mat[nbody][9] (float): frame orientations
        """
        self.mjSend(mjCOM_GETBODY)
        _data = self.mjRecv()
        _out = mjBody.deserialize(_data)
        return _out

    def mj_get_geom(self) -> mjGeom:
        """Read geom positions

        Returns
        -------
        --------
        structure mjGeom
            ngeom (int)          : number of geoms
            time (float)         : simulation time
            pos[ngeom][3] (float): positions
            mat[ngeom][9] (float): frame orientations
        """
        self.mjSend(mjCOM_GETGEOM)
        _data = self.mjRecv()
        _out = mjGeom.deserialize(_data)
        return _out

    def mj_get_site(self) -> mjSite:
        """Read site positions

        Returns
        -------
        --------
        structure mjSite
            nsite (int)          : number of bodies
            time (float)         : simulation time
            pos[nsite][3] (float): positions
            mat[nsite][9] (float): frame orientations
        """
        self.mjSend(mjCOM_GETSITE)
        _data = self.mjRecv()
        _out = mjSite.deserialize(_data)
        return _out

    def mj_get_tendon(self) -> mjTendon:
        """Read tendons data

        Returns
        -------
        --------
        structure mjTendon
            ntendon (int)            : number of tendons
            time (float)             : simulation time
            length[ntendon] (float)  : tendon lengths
            velocity[ntendon] (float): tendon velocities
        """
        self.mjSend(mjCOM_GETTENDON)
        _data = self.mjRecv()
        _out = mjTendon.deserialize(_data)
        return _out

    def mj_get_actuator(self) -> mjActuator:
        """Read tendons data

        Returns
        -------
        --------
        structure mjTendon
            nu (int)            : number of actuators
            time (float)        : simulation time
            length[nu] (float)  : actuator lengths
            velocity[nu] (float): actuator velocities
            force[nu] (float)   : actuator forces
        """
        self.mjSend(mjCOM_GETACTUATOR, 0)
        _data = self.mjRecv()
        _out = mjActuator.deserialize(_data)
        return _out

    def mj_get_force(self) -> mjForce:
        """Read tendons data

        Returns
        -------
        --------
        structure mjTendon
            nv (int)                 : number of generalized velocities/forces
            time (float)             : simulation time
            nonconstraint[nv] (float): sum of all non-constraint forces
            constraint[nv] (float)   : constraint forces (including contacts)
        """
        self.mjSend(mjCOM_GETFORCE, 0)
        _data = self.mjRecv()
        _out = mjForce.deserialize(_data)
        return _out

    def mj_get_contact(self) -> mjContact:
        """Read tendons data

        Returns
        -------
        --------
        structure mjTendon
            ncon (int)            : number of detected contacts
            time (float)          : simulation time
            dist[ncon             : contact normal distance
            pos[ncon][3] (float)  : contact position in world frame
            frame[ncon][9] (float): contact frame relative to world frame (0-2: normal)
            force[ncon][3] (float): contact force in contact frame
            geom1[ncon] (float)   : id of 1st contacting geom
            geom2[ncon] (float)   : id of 2nd contacting geom (force: 1st -> 2nd)
        """
        self.mjSend(mjCOM_GETCONTACT)
        _data = self.mjRecv()
        _out = mjContact.deserialize(_data)
        return _out

    def mj_get_camera(self) -> mjCamera:
        """Read camera data

        Returns
        -------
        --------
        structure mjCamera
        """
        self.mjSend(mjCOM_GETCAMERA)
        _data = self.mjRecv()
        _out = mjCamera.deserialize(_data)
        return _out

    def mj_get_glcamera(self) -> mjGLCamera:
        """Read GL camera data

        Returns
        -------
        --------
        structure mjGLCamera
        """
        self.mjSend(mjCOM_GETGLCAMERA)
        _data = self.mjRecv()
        _out = mjCamera.deserialize(_data)
        return _out

    # Set functions
    def mj_set_state(self, state: mjState):
        """Set simulator state

        Parameters
        ----------
        -----------
        state : mjState
            structure mjState

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = state.serialize()
        self.mjSend(mjCOM_SETSTATE, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_set_control(self, control: mjControl):
        """Set control signals

        Parameters
        ----------
        -----------
        control : mjControl
            structure mjControl

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = control.serialize()
        self.mjSend(mjCOM_SETCONTROL, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_set_applied(self, applied: mjApplied):
        """Set applied forces

        Parameters
        ----------
        -----------
        applied : mjApplied
            structure mjApplied

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = applied.serialize()
        self.mjSend(mjCOM_SETAPPLIED, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_set_onebody(self, onebody: mjOneBody):
        """Set one body data

        Parameters
        ----------
        -----------
        onebody : mjOneBody
            structure mjOneBody

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = onebody.serialize()
        self.mjSend(mjCOM_SETONEBODY, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_set_mocap(self, mocap: mjMocap):
        """Set mocap positions

        Parameters
        ----------
        -----------
        mocap : mjMocap
            structure mjMocap

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = mocap.serialize()
        self.mjSend(mjCOM_SETMOCAP, _buffer)
        self.mjGetAck()
        return self._lastError

    # RGBA functions
    def mj_get_rgba(self, typ: str, id: int) -> np.array:
        _buffer = struct.pack("20s i", typ.encode("utf-8"), id)
        self.mjSend(mjCOM_GETRGBA, _buffer)
        _data = self.mjRecv()
        if _data:
            rgba = np.array(struct.unpack("4f", _data))
            return rgba
        else:
            self.mjSetError(mjCOM_BADSIZE)
            print(f"No RGBA data for object {typ}:{id}")

    def mj_set_rgba(self, typ: str, id: int, rgba: np.array) -> int:
        clamped_rgba = [min(1, max(0, x)) for x in rgba]
        _buffer = struct.pack("20s i 4f", typ.encode("utf-8"), id, *clamped_rgba)
        self.mjSend(mjCOM_SETRGBA, _buffer)
        self.mjGetAck()
        return self._lastError

    # Command and information functions
    def mj_load(self, model_filename: str = None) -> int:
        """Show text message in simulator

        Parameters
        ----------
        -----------
        message : string
            message, None: clear currently shown message

        Returns
        -------
        --------
        int
            API return code
        """
        if model_filename:
            _buffer = struct.pack(f"{len(model_filename) + 1}s", model_filename.encode("utf-8"))
            self.mjSend(mjCOM_LOAD, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_connect(self):
        """Connect to Haptix simulator

        Parameters
        ----------
        -----------
        ip : string
            host IP

        Returns
        -------
        --------
        int
            API return code
        """
        self._soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._soc.connect((self._host, self._port))
            self._connected = 1
            self.mjSetError(mjCOM_OK)
            return self._lastError
        except Exception as e:
            self.mjSetError(mjCOM_NOCONNECTION)
            raise RuntimeError("Connect: " + str(e))

    def connect(self):
        """Connect to MuJoCo simulator (alias)"""
        return self.mj_connect()

    def mj_close(self):
        """Close connection to Haptix simulator

        Returns
        -------
        --------
        int
            API return code
        """
        if not self._soc:
            self._connected = 0
            return self._connected
        try:
            self._soc.sendall(struct.pack("ii", 0, 0))
            self._soc.close()
            self._connected = 0
            return self._connected
        except Exception as e:
            raise RuntimeError("Close: " + str(e))

    def close(self):
        """Close connection to MuJoCo simulator (alias)"""
        return self.mj_close()

    def mj_result(self):
        """
        Return the last API result code.

        Returns
        -------
        str
            Last result code reported by the MuJoCo interface.
        """
        return self._lastError

    def mj_connected(self):
        """
        Return the connection status.

        Returns
        -------
        int
            ``1`` if connected to the simulator, otherwise ``0``.
        """
        return self._connected

    def connected(self):
        """Returns connection status (alias)"""
        return self._connected

    def mj_info(self):
        """Get static properties of current model

        Returns
        -------
        --------
        myInfo: structure
                nq (int)                        : number of generalized positions
                nv (int)                        : number of generalized velocities
                na (int)                        : number of actuator activations
                njnt (int)                      : number of joints
                nbody (int)                     : number of bodies
                ngeom (int)                     : number of geoms
                nsite (int)                     : number of sites
                ntendon (int)                   : number of tendons
                nu (int)                        : number of actuators/controls
                neq (int)                       : number of equality constraints
                nkey (int)                      : number of keyframes
                nmocap (int)                    : number of mocap bodies
                nsensor (int)                   : number of sensors
                nsensordata (int)               : number of elements in sensor data array
                nmat (int)                      : number of materials
                timestep (float)                : simulation timestep
                apirate (float)                 : API update rate
                sensor_type[nsensor] (int)      : sensor type
                sensor_datatype[nsensor] (int)  : type of sensorized object
                sensor_objtype[nsensor] (int)   : type of sensorized object
                sensor_objid[nsensor] (int)     : id of sensorized object
                sensor_dim[nsensor] (int)       : number of sensor outputs
                sensor_adr[nsensor] (int)       : address in sensor data array
                sensor_noise[nsensor] (float)   : noise standard deviation
                jnt_type[njnt] (int)            : joint type (mjtJoint)
                jnt_bodyid[njnt] (int)          : id of body to which joint belongs
                jnt_qposadr[njnt] (int)         : address of joint position data in qpos
                jnt_dofadr[njnt] (int)          : address of joint velocity data in qvel
                jnt_range[njnt][2] (float)      : joint range  (0,0): no limits
                geom_type[ngeom] (int)          : geom type (mjtGeom)
                geom_bodyid[ngeom] (int)        : id of body to which geom is attached
                eq_type[neq] (int)              : equality constraint type (mjtEq)
                eq_obj1id[neq] (int)            : id of constrained object
                eq_obj2id[neq] (int)            : id of 2nd constrained object  -1 if not applicable
                actuator_trntype[nu] (int)      : transmission type (mjtTrn)
                actuator_trnid[nu][2] (int)     : transmission target id
                actuator_ctrlrange[nu][2](float): actuator control range (0,0): no limits
        """
        self.mjSend(mjCOM_INFO, 0)
        _data = self.mjRecv()
        _info = mjInfo.deserialize(_data)
        return _info

    def mj_step(self) -> int:
        """Advance simulation if paused, no effect if running

        Returns
        -------
        --------
        int
            API return code
        """
        self.mjSend(mjCOM_STEP)
        self.mjGetAck()
        return self._lastError

    def mj_pause(self) -> int:
        """Pause simulation

        Returns
        -------
        --------
        int
            API return code
        """
        self.mjSend(mjCOM_PAUSE)
        self.mjGetAck()
        return self._lastError

    def mj_run(self) -> int:
        """Run simulation

        Returns
        -------
        --------
        int
            API return code
        """
        self.mjSend(mjCOM_RUN)
        self.mjGetAck()
        return self._lastError

    def mj_update(self, control: mjControl) -> int:
        """
        Advance the simulation and refresh sensor data.

        The method sends the control command, advances the simulation, and
        returns the refreshed sensor data structure received from the simulator.

        Parameters
        ----------
        control : mjControl
            Control command structure to send to MuJoCo.

        Returns
        -------
        mjSensor
            Updated sensor data returned by the simulator.
        """
        _buffer = control.serialize()
        self.mjSend(mjCOM_UPDATE, _buffer)
        _data = self.mjRecv()
        _out = mjSensor.deserialize(_data)
        return _out

    def mj_reset(self, keyframe: int = None) -> int:
        """Reset simulation to specified key frame

        Parameters
        ----------
        -----------
        keyframe : int, optional
            key frame; -1: reset to model reference configuration

        Returns
        -------
        --------
        int
            API return code
        """
        if keyframe is None:
            self.mjSend(mjCOM_RESET)
        else:
            _buffer = struct.pack("i", keyframe)
            self.mjSend(mjCOM_RESET, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_equality(self, eqid: int, state: int) -> int:
        """Modify state of specified equality constraint

        Parameters
        ----------
        -----------
        eqid : int
            equality id
        state : int
            equality constraint, 1: enable, 0: disable

        Returns
        -------
        --------
        int
            API return code
        """
        _buffer = struct.pack("2i", eqid, state)
        self.mjSend(mjCOM_EQUALITY, _buffer)
        self.mjGetAck()
        return self._lastError

    def mj_message(self, message: str = None) -> int:
        """Show text message in simulator

        Parameters
        ----------
        -----------
        message : str, optional
            message, None: clear currently shown message

        Returns
        -------
        --------
        int
            API return code
        """
        if message:
            _buffer = struct.pack(f"{len(message) + 1}s", message.encode("utf-8"))
            self.mjSend(mjCOM_MESSAGE, _buffer)
        else:
            self.mjSend(mjCOM_MESSAGE)
        self.mjGetAck()
        return self._lastError

    def mj_name2id(self, typ: str, name: str) -> int:
        """
        Return the id of an object with the specified type and name.

        Valid object types are ``body``, ``geom``, ``site``, ``joint``,
        ``tendon``, ``sensor``, ``actuator``, and ``equality``.

        Parameters
        ----------
        typ : str
            Object type.
        name : str
            Object name.

        Returns
        -------
        int
            Object id. Returns ``-1`` if not found and ``-2`` on error.
        """
        _buffer = struct.pack("100s 100s", typ.encode("utf-8"), name.encode("utf-8"))
        self.mjSend(mjCOM_NAME2ID, _buffer)
        _data = self.mjRecv()
        _out = struct.unpack("i", _data)[0]
        return _out

    def mj_id2name(self, typ: str, id: int) -> str:
        """Returns name of object with specified type and id

        valid object types: body, geom, site, joint, tendon, sensor, actuator, equality

        Parameters
        ----------
        -----------
        typ : str
            object type
        id : int
            object id

        Returns
        -------
        --------
        string
            object name
        """
        _buffer = struct.pack("i 100s", id, typ.encode("utf-8"))
        self.mjSend(mjCOM_ID2NAME, _buffer)
        _data = self.mjRecv()
        idx = _data.find(b"\x00")
        if idx != -1:
            _out = _data[:idx].decode("utf-8")
            return _out
        else:
            return None


def isMuJoCo(scn):
    return isinstance(scn, mjInterface)


# --------------------------------------------------------------------
if __name__ == "__main__":
    # Run MuJoCo and load model
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    scn = mjInterface()
    res = scn.mj_connect()
    print(scn.mj_result())

    if scn.mj_connected():
        print("Connected to the simulator.")
    info = scn.mj_info()
    if not info:
        print("No model loaded - terminating")
    else:
        scn.mj_message("Test")
        print("Model info:\n}", info)
        state = scn.mj_get_state()
        print("State:\n", state)
        # state.qpos = np.array([0, -1, 0, 1, 0, 0.5, 0.7, 0, 0])
        # print(" SET: ", scn.mj_set_state(state))
        # state = scn.mj_get_state()
        # print("State:\n", state)

        control = scn.mj_get_control()
        print("Control:\n ", control)
        control.ctrl[:7] = [0, -1, 0, 1, 0, 0.5, 0.7]
        scn.mj_set_control(control)

        applied = scn.mj_get_applied()
        print("applied:\n ", applied)
        applied.qfrc_applied[2] = 0.1
        scn.mj_set_applied(applied)

        body = scn.mj_get_body()
        print("body:\n ", body)

        id = scn.mj_name2id("body", "Target")
        print("Id of Target:", id)
        if id >= 0:
            onebody = scn.mj_get_onebody(id)
            print("onebody:\n ", onebody)
            if onebody.bodyid >= 0:
                # onebody.bodyid = id
                onebody.pos[1] = 0.2
                onebody.pos[2] = 1.0
                scn.mj_set_onebody(onebody)
            onebody = scn.mj_get_onebody(0)
            print("onebody:\n ", onebody)

        mocap = scn.mj_get_mocap()
        print("mocap:\n ", mocap)

        dynamics = scn.mj_get_dynamics()
        print("dynamics:\n ", dynamics)

        sensor = scn.mj_get_sensor()
        print("sensor:\n ", sensor)

        geom = scn.mj_get_geom()
        print("geom:\n ", geom)

        site = scn.mj_get_site()
        print("site:\n ", site)

        tendon = scn.mj_get_tendon()
        print("tendon:\n ", tendon)

        actuator = scn.mj_get_actuator()
        print("actuator:\n ", actuator)

        force = scn.mj_get_force()
        print("force:\n ", force)

        rgba = scn.mj_get_rgba("geom", 1)
        print(rgba)

        rgba[0] = 1
        scn.mj_set_rgba("geom", 4, rgba)

        scn.mj_message()

        print("Site id: ", scn.mj_name2id("site", "Panda_hand"))
        print("Site name: ", scn.mj_id2name("site", 14))

    scn.mj_close()
