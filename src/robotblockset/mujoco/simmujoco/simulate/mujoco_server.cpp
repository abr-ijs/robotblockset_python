/*
  MuJoCo socket communications

  Written by Emo Todorov

  Copyright (C) 2017 Roboti LLC
  
  Updated by Leon Zlajpah 

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

#include "mujoco/mjmodel.h"
#include "mujoco/mjdata.h"
#include "mujoco/mjrender.h"
#include <mujoco/mujoco.h>
#include "socket.h"
#include "mujoco_server.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#ifdef _WIN32
#else
#include "string.h"
#endif



//------------------------- static global variables -------------------------------------
static mjSocket server_soc;                        // socket object

static const int BUFSZ = 100000;                                // buffer size
static char buffer[BUFSZ+8];                                    // preallocated buffer (plus 2-int header)
static int* cmd = (int*)buffer;                                 // command field
static int* size = ((int*)buffer)+1;                            // size field
static int* data = ((int*)buffer)+2;                            // first data field

static int lastError = 0;                                       // result from last operation

static void (*user_handler)(int) = 0;                           // user error handler

char show_message[101] = "";                                    // message for simulator
char load_filename[201] = "";    // model filename for simulator


//------------------------- static utility functions ------------------------------------
// print buffer 
static int printbuffer(char* src, int n, char* msg)
{
    printf("%s:\n ",msg);
    for(int i = 0;i <n; i++){
        printf("%02X ",(unsigned char)src[i]);
        if (((i+1) % 4)==0) printf("   ");
    }
    printf("\n");
    return 0;
}

/*
static int printfloats(float* src, int n, char* msg)
{
    printf("%s: \n",msg);
    for(int i = 0;i <n; i++){
        printf("%f ",src[i]);
    }
    printf("\n");
    return 0;
}

static int print_mjtNum(mjtNum* src, int n, char* msg)
{
    printf("%s: \n",msg);
    for(int i = 0;i <n; i++){
        printf("%f ",(float)src[i]);
    }
    printf("\n");
    return 0;
}
*/

static void mjSetError(int err)
{
    // save result
    lastError = err;

    // call user handler if set
    if( user_handler && err!=mjCOM_OK )
        user_handler(err);
}


static void mjSend(int message, int sz)
{
    // make sure we have connection
    if( !server_soc.getState() )
    {
        mjSetError(mjCOM_CANNOTSEND);
        return;
    }

    // set header
    *cmd = message;
    *size = sz;

    // try to send
    if( server_soc.sendBuffer(buffer, *size + 2*sizeof(int), 1000) )
    {
        server_soc.clear();
        mjSetError(mjCOM_CANNOTSEND);
    }
    else
        mjSetError(mjCOM_OK);
}



static void mjRecv(void)
{
    // get message header, disconnect on error or timeout
    int res = server_soc.recvBuffer(buffer, 2*sizeof(int), 8);
    if( res )
    {
        server_soc.clear();
        mjSetError(res==mjSOC_TIMEOUT ? mjCOM_TIMEOUT : mjCOM_CANNOTRECV);
        return;
    }

    // get message data, disconnect on error
    if( *size>0 && *size<=BUFSZ )
    {
        if( server_soc.recvBuffer((char*)data, *size, 1000) )
        {
            server_soc.clear();
            mjSetError(mjCOM_CANNOTRECV);
        }
    }
    else if( *size )
    {
        server_soc.clear();
        mjSetError(mjCOM_BADSIZE);
    }
}

// copy float data, return size
template <class T, class U>
static int copydata(T* dst, const U* src, int n)
{
    for(int i = 0;i <n; i++){
        dst[i] = (T)src[i];
    }
    return n;
}

// text description of last mjtResult returned by any API function call
const char* mx_last_result(void)
{
    switch( lastError )
    {
    case mjCOM_OK:
        return "OK";

    case mjCOM_BADSIZE:
        return "Bad data size";

    case mjCOM_BADINDEX:
        return "Bad object index";

    case mjCOM_BADCOMMAND:
        return "Invalid command";

    case mjCOM_NOMODEL:
        return "No model loaded";

    case mjCOM_CANNOTSEND:
        return "Could not send data, disconnecting";

    case mjCOM_CANNOTRECV:
        return "Could not receive data, disconnecting";

    case mjCOM_TIMEOUT:
        return "Socket timeout, disconnecting";

    case mjCOM_NOCONNECTION:
        return "No connection to socket";

    case mjCOM_CONNECTED:
        return "Already connected";

    default:
        return "Unknown error";
    }
}


// require connection
#define REQCON if(!server_soc.getState()) {mjSetError(mjCOM_NOCONNECTION); return (mjtResult)lastError; }


// check maximum size
#define CHECK(x) if((x)>mjMAXSZ) {mjSetError(mjCOM_BADSIZE); return mjCOM_BADSIZE;}


// check buffer size
#define VERIFY(ptr) if((char*)ptr - (char*)data != *size) mjSetError(mjCOM_BADSIZE);

// check buffer size
#define DATASIZE(ptr) (int)((char*)ptr - (char*)data)

//------------------------- Extended API: send ------------------------------------------

// get static properties of current model
mjtResult mx_send_info(mjModel* m)
{
    REQCON

    // copy data
    // sizes
    CHECK( data[0]  = m->nq           );
    CHECK( data[1]  = m->nv           );
    CHECK( data[2]  = m->na           );
    CHECK( data[3]  = m->njnt         );
    CHECK( data[4]  = m->nbody        );
    CHECK( data[5]  = m->ngeom        );
    CHECK( data[6]  = m->nsite        );
    CHECK( data[7]  = m->ntendon      );
    CHECK( data[8]  = m->nu           );
    CHECK( data[9]  = m->neq          );
    CHECK( data[10] = m->nkey         );
    CHECK( data[11] = m->nmocap       );
    CHECK( data[12] = m->nsensor      );
    CHECK( data[13] = m->nsensordata  );
    CHECK( data[14] = m->nmat         );
    CHECK(data[15] = m->ncam          );

    // timing parameters
    float* num = (float*)(data+16);
    num[0] = m->opt.timestep;
    num[1] = 0.0f;

    // sensor descriptions
    int* pint = (int*)(num+2);
    pint += copydata(pint, m->sensor_type,     m->nsensor);
    pint += copydata(pint, m->sensor_datatype, m->nsensor);
    pint += copydata(pint, m->sensor_objtype,  m->nsensor);
    pint += copydata(pint, m->sensor_objid,    m->nsensor);
    pint += copydata(pint, m->sensor_dim,      m->nsensor);
    pint += copydata(pint, m->sensor_adr,      m->nsensor);
    num = (float*)pint;
    num +=  copydata(num,  m->sensor_noise,    m->nsensor);

    // joint properties
    pint = (int*)num;
    pint += copydata(pint, m->jnt_type,     m->njnt);
    pint += copydata(pint, m->jnt_bodyid,   m->njnt);
    pint += copydata(pint, m->jnt_qposadr,  m->njnt);
    pint += copydata(pint, m->jnt_dofadr,   m->njnt);
    num = (float*)pint;
    num += copydata(num,  m->jnt_range, 2*m->njnt);

    // geom properties
    pint = (int*)num;
    pint += copydata(pint, m->geom_type,    m->ngeom);
    pint += copydata(pint, m->geom_bodyid,  m->ngeom);

    // equality constraint properties
    pint += copydata(pint, m->eq_type,      m->neq);
    pint += copydata(pint, m->eq_obj1id,    m->neq);
    pint += copydata(pint, m->eq_obj2id,    m->neq);

    // actuator properties
    pint += copydata(pint, m->actuator_trntype,   m->nu);
    pint += copydata(pint, m->actuator_trnid,   2*m->nu);
    num = (float*)pint;
    num += copydata(num, m->actuator_ctrlrange, 2*m->nu);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_state(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nq );
    CHECK( data[1] = m->nv );
    CHECK( data[2] = m->na );
    
    // data
    float* num = (float*)(data+3);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->qpos,  m->nq);
    num += copydata(num, (mjtNum *)d->qvel,  m->nv);
    num += copydata(num, (mjtNum *)d->act,   m->na);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_control(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nu  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->ctrl,  m->nu);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}


mjtResult mx_send_applied(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nv  );
    CHECK( data[1] = m->nbody  );

    // data
    float* num = (float*)(data+2);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->qfrc_applied,  m->nv);
    num += copydata(num, (mjtNum *)d->xfrc_applied,  6*m->nbody);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_onebody(mjModel* m, mjData* d)
{
    REQCON
    // sizes
    int bodyid = data[0];

    // data
    int* pint = (int*)data;
    *pint = (bodyid>=0 && m->body_jntnum[bodyid]==1 &&  m->jnt_type[m->body_jntadr[bodyid]]==mjJNT_FREE ? 1 : 0);
    
    mjtNum contact_forces[3];
    for(int i = 0;i <3; i++){
        contact_forces[i]=d->cfrc_ext[bodyid*6+3+i]-d->xfrc_applied[bodyid*6+3+i];
    }
    
    float* num = (float*)(pint+1);
    num += copydata(num, (mjtNum *)&d->time,  1);

/*
    int qposadr = -1, qveladr = -1;
    if( bodyid>=0 && m->body_jntnum[bodyid]==1 && m->jnt_type[m->body_jntadr[bodyid]]==mjJNT_FREE )
    {
        // extract the addresses from the joint specification
        qposadr = m->jnt_qposadr[m->body_jntadr[bodyid]];
        qveladr = m->jnt_dofadr[m->body_jntadr[bodyid]];
        num += copydata(num, (mjtNum *)&d->qacc[qveladr],        6 ); 
        num += copydata(num, (mjtNum *)&d->cfrc_ext[bodyid*6+3], 3 );
        num += copydata(num, (mjtNum *)&d->cfrc_ext[bodyid*6],   3 );
        num += copydata(num, (mjtNum *)&d->qpos[qposadr],        7 );
        num += copydata(num, (mjtNum *)&d->qvel[qveladr],        6 );
        num += copydata(num, (mjtNum *)&d->cfrc_int[qveladr],    6 );
    } else {
*/
        num += copydata(num, (mjtNum *)&d->cacc[bodyid*6+3],     3 );  // lin acc
        num += copydata(num, (mjtNum *)&d->cacc[bodyid*6],       3 );  // ang acc
        num += copydata(num, (mjtNum *)&contact_forces,          3 );  // contact force
        num += copydata(num, (mjtNum *)&d->xpos[bodyid*3],       3 );  // pos 
        num += copydata(num, (mjtNum *)&d->xquat[bodyid*4],      4 );  // quat
        num += copydata(num, (mjtNum *)&d->cvel[bodyid*6+3],     3 );  // lin vel
        num += copydata(num, (mjtNum *)&d->cvel[bodyid*6],       3 );  // ang vel
        num += copydata(num, (mjtNum *)&d->cfrc_int[bodyid*6],   6 );  // force/torque
//    }

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_mocap(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nmocap  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->mocap_pos,  3*m->nmocap);
    num += copydata(num, (mjtNum *)d->mocap_quat, 4*m->nmocap);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_dynamics(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nv  );
    CHECK( data[1] = m->na  );

    // data
    float* num = (float*)(data+2);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->qacc,    m->nv);
    num += copydata(num, (mjtNum *)d->act_dot, m->na);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_sensor(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nsensordata  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->sensordata, m->nsensordata);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_body(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nbody  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->xpos, 3*m->nbody);
    num += copydata(num, (mjtNum *)d->xmat, 9*m->nbody);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_geom(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK(data[0] = m->ngeom);

    // data
    float* num = (float*)(data + 1);
    num += copydata(num, (mjtNum*)&d->time, 1);
    num += copydata(num, (mjtNum*)d->geom_xpos, 3 * m->ngeom);
    num += copydata(num, (mjtNum*)d->geom_xmat, 9 * m->ngeom);

    mjSend(0, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_geomsize(mjModel* m, mjData* d)
{
    REQCON

    // id
    int id;
    int* pint = (int*)(data);
    pint += copydata(&id, pint, 1);

    float* num = (float*)data;
    num += copydata(num, m->geom_size + (id) * 3, 3);

    mjSend(0, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_site(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nsite  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->site_xpos, 3*m->nsite);
    num += copydata(num, (mjtNum *)d->site_xmat, 9*m->nsite);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_tendon(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->ntendon  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->ten_length,   m->ntendon);
    num += copydata(num, (mjtNum *)d->ten_velocity, m->ntendon);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_actuator(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nu  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->actuator_length,   m->nu);
    num += copydata(num, (mjtNum *)d->actuator_velocity, m->nu);
    num += copydata(num, (mjtNum *)d->actuator_force,    m->nu);


    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_force(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( data[0] = m->nv  );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->qfrc_smooth,     m->nv);
    num += copydata(num, (mjtNum *)d->qfrc_constraint, m->nv);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_send_contact(mjModel* m, mjData* d)
{
    REQCON

    mjtNum contact_forces[6];
    // sizes
    CHECK( data[0] = d->ncon );

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,  1);
    int* pint = (int*)num;
    for (int i = 0;i < d->ncon; i++) {
        num = ((float*)data)+2+i;
        num += copydata(num, (mjtNum *)&d->contact[i].dist, 1);
        num = ((float*)data)+2+d->ncon+i*3;
        num += copydata(num, (mjtNum *)&d->contact[i].pos, 3);
        num = ((float*)data)+2+d->ncon*4+i*9;
        num += copydata(num, (mjtNum *)&d->contact[i].frame, 9);
        mj_contactForce(m, d, i, contact_forces);
        num = ((float*)data)+2+d->ncon*13+i*3;
        num += copydata(num, contact_forces, 3);

        pint = ((int*)data)+2+d->ncon*16+i;
        pint += copydata(pint, &d->contact[i].geom1, 1);
        pint = ((int*)data)+2+d->ncon*17+i;
        pint += copydata(pint, &d->contact[i].geom2, 1);
    }
    mjSend(mjCOM_OK, DATASIZE(pint));
    return (mjtResult)lastError;
}

mjtResult mx_send_camera(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK(data[0] = m->ncam);
    // CHECK(data[1] = cam->fixedcamid);
    // CHECK(data[2] = cam->trackbodyid);

    // data
    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&d->time,    1);
    num += copydata(num, (mjtNum *)d->cam_xpos, 3*m->ncam);
    num += copydata(num, (mjtNum *)d->cam_xmat, 9*m->ncam);

    int* pint = (int*)num;
    pint += copydata(pint, m->cam_mode,         m->ncam);
    pint += copydata(pint, m->cam_bodyid,       m->ncam);
    pint += copydata(pint, m->cam_targetbodyid, m->ncam);

    num = (float*)pint;
    num += copydata(num, (mjtNum *)m->cam_pos,     3*m->ncam);
    num += copydata(num, (mjtNum *)m->cam_quat,    4*m->ncam);
    num += copydata(num, (mjtNum *)m->cam_poscom0, 3*m->ncam);
    num += copydata(num, (mjtNum *)m->cam_pos0,    3*m->ncam);
    num += copydata(num, (mjtNum *)m->cam_mat0,    9*m->ncam);
    num += copydata(num, (mjtNum *)m->cam_fovy,    m->ncam);
    num += copydata(num, (mjtNum *)m->cam_ipd,     m->ncam);
    // num += copydata(num, (mjtNum *)m->cam_user, m->nuser_cam*m->ncam);
    
    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

/*
mjtResult mx_send_GLcamera(mj::Simulate* sim)
{
    REQCON

    mjvCamera cam = sim->cam;
    mjvScene scn = sim->scn;
    const int nglcam = sizeof(scn.camera)/sizeof(mjvGLCamera);
    // mjvGLCamera* glcam = scn.camera;
    mjvGLCamera glcam[nglcam];

    for (int i = 0; i < nglcam; ++i) {
        glcam[i] = scn.camera[i];
    }

    // sizes
    CHECK(data[0] = nglcam);

    float* num = (float*)(data+1);
    num += copydata(num, (mjtNum *)&sim->d->time,      1);

    // data mjvCamera
    int* pint = (int*)(num);
    pint += copydata(pint, &cam.fixedcamid,           1*nglcam);
    pint += copydata(pint, &cam.type,                 1*nglcam);
    pint += copydata(pint, &cam.trackbodyid,          1*nglcam);

    num = (float*)(pint);
    num += copydata(num, (mjtNum *)cam.lookat,        3*nglcam);
    num += copydata(num, (mjtNum *)&cam.distance,     1*nglcam);
    num += copydata(num, (mjtNum *)&cam.azimuth,      1*nglcam);
    num += copydata(num, (mjtNum *)&cam.elevation,    1*nglcam);

    // data mjvGLCamera
    // camera frame
    num += copydata(num, glcam->pos,          3*nglcam);
    num += copydata(num, glcam->forward,      3*nglcam);
    num += copydata(num, glcam->up,           3*nglcam);

    // camera projection
    num += copydata(num, &glcam->frustum_center,    1*nglcam);
    num += copydata(num, &glcam->frustum_bottom,    1*nglcam);
    num += copydata(num, &glcam->frustum_top,       1*nglcam);
    num += copydata(num, &glcam->frustum_near,      1*nglcam);
    num += copydata(num, &glcam->frustum_far,       1*nglcam);

    
    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}
*/

// return id of object with specified type and name; -1: not found; -2: error
//  allowed object types: body, geom, site, joint, tendon, sensor, actuator, equality
mjtResult mx_name2id(mjModel* m)
{
    REQCON
    
    // get type and name: 100 bytes each
    static char type[101];
    static char name[101];
    char* txt = (char*)data;
    strncpy(type, txt, 100);
    strncpy(name, txt+100, 100);
    data[0]=mj_name2id(m, mju_str2Type(type), name);
 
    mjSend(mjCOM_OK, 4);
    return (mjtResult)lastError;
}


// return name of object with specified type and id; NULL: error
mjtResult mx_id2name(mjModel* m)
{
    REQCON

    // get data: id, 100 bytes for type name
    static char type[101];
    int id = data[0];
    char* txt = (char*)(data+1);
    strncpy(type, txt, 100);
    char* name = (char*)(data);
    if ( mj_id2name(m, mju_str2Type(type), id) ) {
        strncpy(name, mj_id2name(m, mju_str2Type(type), id), 100);
    } else {
        name[0] = 0;
    }
    name[99] = 0;

    mjSend(mjCOM_OK, 100);
    return (mjtResult)lastError;
}


//------------------------- Extended API: set -------------------------------------------

mjtResult mx_set_state(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( m->nq = data[0] );
    CHECK( m->nv = data[1] );
    CHECK( m->na = data[2] );

    // data
    float* num = (float*)(data+3);  
    num += copydata((mjtNum *)d->qpos, num, m->nq);
    num += copydata((mjtNum *)d->qvel, num, m->nv);
    num += copydata((mjtNum *)d->act,  num, m->na);

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

mjtResult mx_set_control(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( m->nu = data[0] );

    // data
    float* num = (float*)(data+1);  
    num += copydata((mjtNum *)d->ctrl,  num, m->nu);

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}


mjtResult mx_set_applied(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( m->nv    = data[0] );
    CHECK( m->nbody = data[1] );

    // data
    float* num = (float*)(data+2);  
    num += copydata((mjtNum *)d->qfrc_applied, num,  m->nv);
    num += copydata((mjtNum *)d->xfrc_applied, num,  6*(m->nbody));

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

mjtResult mx_set_onebody(mjModel* m, mjData* d)
{
    REQCON

    int qposadr = -1, qveladr = -1;
    int bodyid = data[0];

    if( bodyid>=0 && m->body_jntnum[bodyid]==1 && m->jnt_type[m->body_jntadr[bodyid]]==mjJNT_FREE )
    {
        // extract the addresses from the joint specification
        qposadr = m->jnt_qposadr[m->body_jntadr[bodyid]];
        qveladr = m->jnt_dofadr[m->body_jntadr[bodyid]];
    }

    // data
    float* num = (float*)(data+1);  
    num += copydata((mjtNum *)&d->qpos[qposadr],          num, 7 );
    num += copydata((mjtNum *)&d->qvel[qveladr],          num, 6 );
    num += copydata((mjtNum *)&d->xfrc_applied[bodyid*6], num, 6 );

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}


mjtResult mx_set_mocap(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( m->nmocap = data[0] );

    // data
    float* num = (float*)(data+1);  
    num += copydata((mjtNum *)d->mocap_pos,  num, 3*m->nmocap);
    num += copydata((mjtNum *)d->mocap_quat, num, 4*m->nmocap);

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}


mjtResult mx_set_geomsize(mjModel* m)
{
    REQCON

    // id
    int id;
    int* pint = (int*)(data);
    pint += copydata(&id, pint, 1);

    // send size
    float* num = (float*)pint;
    mjtNum* size = (mjtNum*)m->geom_size+(id)*3;
    num += copydata(size, num, 3);

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

//------------------------- Extended API: rgba ------------------------------------------

mjtResult mx_send_rgba(mjModel* m)
{
    REQCON

    // get data: 20 bytes for type name and id
    static char type[21];
    char* txt = (char*)(data);
    strncpy(type, txt, 20);
    int id;
    int* pint = (int*)(txt+20);
    pint += copydata(&id, pint, 1);
    
    int typ = mju_str2Type(type);

    float* num = (float*)data;
    switch (typ)
    {
        case mjOBJ_GEOM:
            num += copydata(num, m->geom_rgba+(id)*4, 4);
        break;
    
        case mjOBJ_SITE:
            num += copydata(num, m->site_rgba+(id)*4, 4);
        break;
        
        case mjOBJ_SKIN:
            num += copydata(num, m->skin_rgba+(id)*4, 4);
        break;
        
        case mjOBJ_TENDON:
            num += copydata(num, m->tendon_rgba+(id)*4, 4);
        break;
    };

    VERIFY(num);
    // confirm data
    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}



mjtResult mx_set_rgba(mjModel* m)
{
    REQCON

    // get data: 20 bytes for type name and id
    static char type[21];
    char* txt = (char*)(data);
    strncpy(type, txt, 20);
    int id;
    int* pint = (int*)(txt+20);
    pint += copydata(&id, pint, 1);
    
    int typ = mju_str2Type(type);

    float* num = (float*)pint;
    switch (typ)
    {
        case mjOBJ_GEOM:
            num += copydata(m->geom_rgba+(id)*4, num, 4);
        break;
    
        case mjOBJ_SITE:
            num += copydata(m->site_rgba+(id)*4, num, 4);
        break;
        
        case mjOBJ_SKIN:
            num += copydata(m->skin_rgba+(id)*4, num, 4);
        break;
        
        case mjOBJ_TENDON:
            num += copydata(m->tendon_rgba+(id)*4, num, 4);
        break;
    };

    VERIFY(num);

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}


//------------------------- Extended API: control -----------------------

mjtResult mx_update(mjModel* m, mjData* d)
{
    REQCON

    // sizes
    CHECK( m->nu = data[0] );

    // data
    float* num = (float*)(data+1);  
    num += copydata((mjtNum *)d->ctrl,  num, m->nu);

    VERIFY(num);

    // sizes
    CHECK( data[0] = m->nsensordata  );

    // data
    num += copydata(num, (mjtNum *)&d->time,  1);
    num += copydata(num, (mjtNum *)d->sensordata, m->nsensordata);

    mjSend(mjCOM_OK, DATASIZE(num));
    return (mjtResult)lastError;
}

mjtResult mx_reset(mjModel* m, mjData* d)
{
    REQCON

    mj_resetData(m, d);

    if( *size==0 )
    {
        mj_forward(m, d);
    } 
    else 
    {
        int i = data[0];
        d->time = m->key_time[i];
        mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
        mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
        mju_copy(d->act, m->key_act+i*m->na, m->na);
        mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
        mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
        mju_copy(d->ctrl, m->key_ctrl+i*m->nv, m->nv);
        mj_forward(m, d);
    }

    // confirm data
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

mjtResult mx_equality(mjModel* m, mjData* d)
{
    REQCON

    d->eq_active[data[0]]=(mjtByte)data[1];

    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

mjtResult mx_message(void)
{
    REQCON

    char* txt = (char*)(data);
    char* msg = (char*)(show_message);
    if (*size==0)
    {
        show_message[0]=0;
    }
    else
    {
        strncpy(msg, txt, 100);
    }

    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

//------------------------- Extended API: command and information -----------------------

// connect to simulator
mjtResult mx_connect(const char* host, const char* port)
{
    static bool firstcall = true;
    // init sockets and clear info only once
    if (server_soc.verbose)
        printf("Init socket (S:%d)... \n", (int)server_soc.soc);
    if( firstcall )
    {
        server_soc.mjInitSockets();
        firstcall = false;
    }

    // do not connect again
    if( server_soc.getState() )
    {
        mjSetError(mjCOM_CONNECTED);
        return (mjtResult)lastError;
    }

    // try to connect for 5 seconds
    mjSetError(mjCOM_OK);
    if (server_soc.verbose)
        printf("Opening server (S:%d) on host %s port %s... \n", (int)server_soc.soc, host, port);

    if( server_soc.connectServer(5000, 1, host, port) )
    {
        if (server_soc.verbose)
            printf("Server listening (S:%d L:%d) ... \n",(int)server_soc.soc,(int)server_soc.listen_soc);
        return mjCOM_OK;
    }
    else
    {
        if (server_soc.verbose)
            printf("Server not opened! \n");
        mjSetError(mjCOM_NOCONNECTION);
        return (mjtResult)lastError;
    }
   
}

// load model
mjtResult mx_load(void)
{
    REQCON

    char* txt = (char*)(data);
    char* fln = (char*)(load_filename);
    if (*size==0)
    {
    }
    else
    {
        strncpy(fln, txt, 100);
    }
    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

/*
mjtResult mx_req_screenshot(mj::Simulate* sim) {
    REQCON

    sim->screenshotrequest.store(true);

    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

mjtResult mx_pls_img(mj::Simulate* sim) {
    REQCON

    sim->snapshotrequest.store(true);

    // mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}

*/

mjtResult mx_send_img(unsigned char* rgb, float* depth, unsigned int w, unsigned int h) {
    REQCON

    mjSend(mjCOM_OK, 0);
    return (mjtResult)lastError;
}


mjtResult mx_send_rgb(unsigned char* rgb, unsigned int w, unsigned int h) {
    REQCON

    // mjSend(mjCOM_OK, DATASIZE());
    return (mjtResult)lastError;
}

mjtResult mx_send_depth(unsigned char* rgbd, unsigned int w, unsigned int h) {
    REQCON

    // mjSend(mjCOM_OK, DATASIZE());
    return (mjtResult)lastError;
}

// check incomming connection to simulator
mjtResult mx_check(mjModel* m, mjData* d, int* run)
{
    char Cmd_str[10] = "Command :";
    char Rsp_str[10] = "Response:";
    if( server_soc.getState() )
    {
        int tmp_r;
        tmp_r=server_soc.waitSocket(server_soc.soc, true, 0);
        if(tmp_r) {
            mjRecv();
            if (server_soc.verbose)
                printbuffer(buffer,*size+8,Cmd_str);
            if (*cmd==mjCOM_LOAD){
                mx_load();
            }
            else if ( m ) {
                if (*cmd==mjCOM_PAUSE){
                    *run=0;
                    mjSend(mjCOM_OK, 0);
                }
                else if (*cmd==mjCOM_RUN){
                    *run=1;
                    mjSend(mjCOM_OK, 0);
                }
                else if (*cmd==mjCOM_INFO){
                    mx_send_info(m);
                }
                else if (*cmd==mjCOM_STEP){
                    if (*run==0) {
                        mj_step(m, d);
                    }
                    mjSend(mjCOM_OK, 0);
                }
                else if (*cmd==mjCOM_UPDATE){
                    mx_update(m,d);
                }
                else if (*cmd==mjCOM_RESET){
                    mx_reset(m,d);
                }
                else if (*cmd==mjCOM_EQUALITY){
                    mx_equality(m,d);
                }
                else if (*cmd==mjCOM_MESSAGE){
                    mx_message();
                }
                else if (*cmd==mjCOM_SCREENSHOT){
                    //mx_pls_img(sim);
                }
                else if (*cmd==mjCOM_NAME2ID){
                    mx_name2id(m);
                }
                else if (*cmd==mjCOM_ID2NAME){
                    mx_id2name(m);
                }
                else if (*cmd==mjCOM_GETSTATE){
                    mx_send_state(m,d);
                    //printf("State time: %12.4f %12.4f\n", d->time - state_time, d->time);
                    //state_time = d->time;
                }
                else if (*cmd==mjCOM_GETCONTROL){
                    mx_send_control(m,d);
                }
                else if (*cmd==mjCOM_GETAPPLIED){
                    mx_send_applied(m,d);
                }
                else if (*cmd==mjCOM_GETONEBODY){
                    mx_send_onebody(m,d);
                }
                else if (*cmd==mjCOM_GETMOCAP){
                    mx_send_mocap(m,d);
                }
                else if (*cmd==mjCOM_GETDYNAMICS){
                    mx_send_dynamics(m,d);
                }
                else if (*cmd==mjCOM_GETSENSOR){
                    mx_send_sensor(m,d);
                }
                else if (*cmd==mjCOM_GETBODY){
                    mx_send_body(m,d);
                }
                else if (*cmd == mjCOM_GETGEOM) {
                    mx_send_geom(m, d);
                }
                else if (*cmd == mjCOM_GETGEOMSIZE) {
                    mx_send_geomsize(m, d);
                }
                else if (*cmd==mjCOM_GETSITE){
                    mx_send_site(m,d);
                }
                else if (*cmd==mjCOM_GETACTUATOR){
                    mx_send_actuator(m,d);
                }
                else if (*cmd==mjCOM_GETFORCE){
                    mx_send_force(m,d);
                }
                else if (*cmd==mjCOM_GETCONTACT){
                    mx_send_contact(m,d);
                }
                else if (*cmd==mjCOM_GETCAMERA){
                    mx_send_camera(m, d);
                    printf("Camera info sent.\n");
                }
                else if (*cmd==mjCOM_GETGLCAMERA){
                    //mx_send_glcamera(m, d);
                    printf("Camera info not sent.\n");
                }
                else if (*cmd==mjCOM_SETSTATE){
                    mx_set_state(m,d);
                }
                else if (*cmd==mjCOM_SETCONTROL){
                    mx_set_control(m,d);
                }
                else if (*cmd==mjCOM_SETAPPLIED){
                    mx_set_applied(m,d);
                }
                else if (*cmd==mjCOM_SETONEBODY){
                    mx_set_onebody(m,d);
                }
                else if (*cmd==mjCOM_SETMOCAP){
                    mx_set_mocap(m,d);
                }
                else if (*cmd==mjCOM_GETTENDON){
                    mx_send_tendon(m,d);
                }
                else if (*cmd==mjCOM_SETGEOMSIZE){
                    mx_set_geomsize(m);
                }
                else if (*cmd==mjCOM_GETRGBA){
                    mx_send_rgba(m);
                }
                else if (*cmd==mjCOM_SETRGBA){
                    mx_set_rgba(m);
                }
                else {
                    mjSend(-1, 0);;
                }
            } else {
                mjSend(mjCOM_NOMODEL, 0);        
            }
            if (server_soc.verbose)
                printbuffer(buffer,*size+8,Rsp_str);
        } else {
            if (m) if (*run==0) mj_forward(m, d);
        }
    } 
    else 
    {
        if (*run==0) mj_forward(m, d);
        server_soc.acceptClient(0);
    }
    return (mjtResult)lastError;
}


// close connection to simulator
mjtResult mx_close(void)
{
    REQCON

    if (server_soc.verbose)
        printf("Server closed\n");
    server_soc.clear();
    mjSetError(mjCOM_OK);
    return mjCOM_OK;
}

// return last result code
mjtResult mx_result(void)
{
    return (mjtResult)lastError;
}


// return 1 if connection to simulator is live, 0 otherwise
int mx_connected(void)
{
    return (int)server_soc.getState();
}

int mx_available(void)
{
    return (int)server_soc.getListenState();
}

char* mx_message_txt(void)
{
    return (char*)show_message;
}


// install user error handler
void mx_handler(void(*handler)(int))
{
    user_handler = handler;
}
