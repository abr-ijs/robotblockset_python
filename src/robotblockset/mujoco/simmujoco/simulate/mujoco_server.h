/*
  C/C++ API for socket-based programmatic access to MuJoCo

  Written by Leon Zlajpah (based on haptix API by Emo Todorov)

  Copyright (C) 2022 IJS
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#pragma once

// DLL export/import
#ifndef MJAPI
#if defined(MJ_STATIC)
    #define MJAPI
#elif defined(MJ_EXPORT)
    #define MJAPI __declspec(dllexport)
#else
    #define MJAPI __declspec(dllimport)
#endif
#endif

// this is a C API
#if defined(__cplusplus)
extern "C" {
#endif


/****************************************************************************************

Native API  (prefix 'mj')

    This API provides more complete access to the simulator and does not make
    assumptions about the model structure, except for the maximum array sizes.
    It aims to maximize the benefits of using the simulator.

    The simple and native APIs can be mixed.
    
    Unlike the simple API where the user is expected to know the sizes of the
    variable-size arrays (by calling hx_robot_info), here the sizes of the
    variable-size arrays are included in the structures containing the arrays.

****************************************************************************************/


//-------------------------------- constants --------------------------------------------

// predefined size for array allocation
#define mjMAXSZ 1000

extern char show_message[101];          // message for simulator
extern char load_filename[201];         // model filename for simulator

// API commands
typedef enum
{
    // client-to-server, extended API
    mjCOM_INFO = 3,                     // get model info
    mjCOM_STEP,                         // advance simulation in paused mode
    mjCOM_UPDATE,                       // set control, advance, get sensor data
    mjCOM_RESET,                        // reset simulation
    mjCOM_PAUSE,                        // pause simulation
    mjCOM_RUN,                          // run simulation
    mjCOM_EQUALITY,                     // set state of equality constraint
    mjCOM_MESSAGE,                      // show text message
    mjCOM_NAME2ID,                      // convert object name to id
    mjCOM_ID2NAME,                      // convert object id to name
    // load model
    mjCOM_LOAD,                         // load MuJoCo model

    // get dynamic data
    mjCOM_GETSTATE = 16,                // state
    mjCOM_GETCONTROL,                   // control
    mjCOM_GETAPPLIED,                   // applied forces
    mjCOM_GETONEBODY,                   // detailed info for one body
    mjCOM_GETMOCAP,                     // mocap
    mjCOM_GETDYNAMICS,                  // output of forward dynamics
    mjCOM_GETSENSOR,                    // sensor data
    mjCOM_GETBODY,                      // body kinematics
    mjCOM_GETGEOM,                      // geom kinematics
    mjCOM_GETGEOMSIZE,                  // geom size
    mjCOM_GETSITE,                      // site kinematics
    mjCOM_GETTENDON,                    // tendon kinematics
    mjCOM_GETACTUATOR,                  // actuator kinematics and force
    mjCOM_GETFORCE,                     // generalized forces
    mjCOM_GETCONTACT,                   // contact info

    // set dynamic data
    mjCOM_SETSTATE = 48,                // state
    mjCOM_SETCONTROL,                   // control
    mjCOM_SETAPPLIED,                   // applied forces
    mjCOM_SETONEBODY,                   // detailed info for one body
    mjCOM_SETMOCAP,                     // mocap
    mjCOM_SETGEOMSIZE,                  // geom size

    // get and set rgba static data
    mjCOM_GETRGBA = 64,                 // object rgba
    mjCOM_SETRGBA,                      // object rgba

    // cameras
    mjCOM_SCREENSHOT,                   // make a screenshot request
    mjCOM_GETCAMERA,                    // camera info
    mjCOM_GETGLCAMERA                   // active (GL) camera info
} mjtComCode;


// API return codes
typedef enum
{
    mjCOM_OK            = 0,            // success

    // server-to-client errors
    mjCOM_BADSIZE       = -1,           // data has invalid size
    mjCOM_BADINDEX      = -2,           // object has invalid index
    mjCOM_BADTYPE       = -3,           // invalid object type
    mjCOM_BADCOMMAND    = -4,           // unknown command
    mjCOM_NOMODEL       = -5,           // model has not been loaded
    mjCOM_CANNOTSEND    = -6,           // could not send data
    mjCOM_CANNOTRECV    = -7,           // could not receive data
    mjCOM_TIMEOUT       = -8,           // receive timeout

    // client-side errors
    mjCOM_NOCONNECTION  = -9,           // connection not established
    mjCOM_CONNECTED     = -10,          // already connected
} mjtResult;

//------------------------- Quantities that can be SEND ----------------------------------

// information about all detected contacts
struct _mxContact
{
    int ncon;                           // number of detected contacts
    float time;                         // simulation time
    float dist[mjMAXSZ];                // contact normal distance
    float pos[mjMAXSZ][3];              // contact position in world frame
    float frame[mjMAXSZ][9];            // contact frame relative to world frame (0-2: normal)  
    float force[mjMAXSZ][3];            // contact force in contact frame   
    int geom1[mjMAXSZ];                 // id of 1st contacting geom    
    int geom2[mjMAXSZ];                 // id of 2nd contacting geom (force: 1st -> 2nd)
};
typedef struct _mxContact mxContact;


//------------------------- Quantities that can be SEND and GET --------------------------

// detailed information about one body
struct _mjOneBody
{
    int bodyid;                         // body id, provided by user

    // get only
    int isfloating;                     // 1 if body is floating, 0 otherwise
    float time;                         // simulation time
    float linacc[3];                    // linear acceleration
    float angacc[3];                    // angular acceleration
    float contactforce[3];              // net force from all contacts on this body

    // get for all bodies; set for floating bodies only
    //  (setting the state of non-floating bodies would require inverse kinematics)
    float pos[3];                       // position
    float quat[4];                      // orientation quaternion
    float linvel[3];                    // linear velocity
    float angvel[3];                    // angular velocity

    // get and set for all bodies 
    //  (modular access to the same data as provided by mjApplied.xfrc_applied)
    float force[3];                     // Cartesian force applied to body CoM
    float torque[3];                    // Cartesian torque applied to body
};
typedef struct _mjOneBody mjOneBody;


//--------------------------- API send/set functions -------------------------------------

// send dynamic data from simulator
mjtResult mx_send_state    (mjModel* m, mjData* d);
mjtResult mx_send_control  (mjModel* m, mjData* d);
mjtResult mx_send_applied  (mjModel* m, mjData* d);
mjtResult mx_send_onebody  (mjModel* m, mjData* d);
mjtResult mx_send_mocap    (mjModel* m, mjData* d);
mjtResult mx_send_dynamics (mjModel* m, mjData* d);
mjtResult mx_send_sensor   (mjModel* m, mjData* d);
mjtResult mx_send_body     (mjModel* m, mjData* d);
mjtResult mx_send_geom     (mjModel* m, mjData* d);
mjtResult mx_send_geomsize (mjModel* m, mjData* d);
mjtResult mx_send_site     (mjModel* m, mjData* d);
mjtResult mx_send_tendon   (mjModel* m, mjData* d);
mjtResult mx_send_actuator (mjModel* m, mjData* d);
mjtResult mx_send_force    (mjModel* m, mjData* d);
mjtResult mx_send_contact  (mjModel* m, mjData* d);
mjtResult mx_send_camera   (mjModel* m, mjData* d);
mjtResult mx_name2id       (mjModel* m);
mjtResult mx_id2name       (mjModel* m);

// set dynamic data in simulator
mjtResult mx_set_state    (mjModel* m, mjData* d);
mjtResult mx_set_control  (mjModel* m, mjData* d);
mjtResult mx_set_applied  (mjModel* m, mjData* d);
mjtResult mx_set_onebody  (mjModel* m, mjData* d);
mjtResult mx_set_mocap    (mjModel* m, mjData* d);
mjtResult mx_set_geomsize (mjModel* m);

// get and set rgba static data in simulator
//  valid object types: geom, site, tendon, material
mjtResult mx_send_rgba    (mjModel* m);
mjtResult mx_set_rgba     (mjModel* m);

// make a request to simulator thread to take a screenshot from active window
//mjtResult mx_req_screenshot (mj::Simulate* sim);
//mjtResult mx_pls_img        (mj::Simulate* sim);
//mjtResult mx_send_GLcamera  (mj::Simulate* sim);

// forward images to client
mjtResult mx_send_img    (unsigned char* rgb, float* depth, unsigned int w, unsigned int h);
mjtResult mx_send_rgb    (unsigned char* rgb, unsigned int w, unsigned int h);
mjtResult mx_send_depth  (unsigned char* rgbd, unsigned int w, unsigned int h);

//--------------------------- API command and information functions ---------------------

// text description of last mjtResult returned by any API function call
const char* mx_last_result(void);

// connect to simulator
mjtResult mx_connect(const char* host = 0, const char* port = 0);

// load model
mjtResult mx_load(void);
char* mx_model_filename(void);

// close connection to simulator
mjtResult mx_close(void);

// check incomming connection to simulator
mjtResult mx_check(mjModel* m, mjData* d, int* run);

// return last result code
mjtResult mx_result(void);

// return 1 if client connected to simulator, 0 otherwise
int mx_connected(void);

// return 1 if listening for clients, 0 otherwise
int mx_available(void);

// get static properties of current model
mjtResult mx_send_info(mjModel* m);

// set control, step if paused or wait for 1/apirate if running, get sensor data
mjtResult mx_update(mjModel* m, mjData* d);

// reset simulation to specified key frame; -1: reset to model reference configuration
mjtResult mx_reset(mjModel* m, mjData* d);

// modify state of specified equality constraint (1: enable, 0: disable)
mjtResult mx_equality(mjModel* m);

// show text message in simulator; NULL: clear currently shown message
mjtResult mj_message(void);
char* mx_message_txt(void);


// install user error handler
void mx_handler(void(*handler)(int));


#if defined(__cplusplus)
}
#endif
