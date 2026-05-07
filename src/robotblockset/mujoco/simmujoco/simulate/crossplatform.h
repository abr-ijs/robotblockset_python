/*
  MuJoCo OS-specific functions

  Written by Emo Todorov

  Copyright (C) 2017 Roboti LLC
  
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


//------------------------- OS-specific include, libraries, defines --------------------

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#include <mmsystem.h>
	
	#define PATHSYMBOL '\\'

#else
	#include <pthread.h>
	#include <semaphore.h>
	#include <sys/types.h>
	#include <errno.h>
	#include <netdb.h>
	#include <unistd.h>
	#include <X11/Xlib.h>
	#include <sys/time.h>

	#define PATHSYMBOL '/'
#endif

// namespace mujoco {

//---------------------------- Timing ---------------------------------------------------

// initialize 1 msec timer (on Windows), record base time
void mjBeginTime(void);

// close timer (on Windows)
void mjEndTime(void);

// get time in milliseconds scince mjBeginTime
int mjGetTime(void);

// get time in microseconds since mjBeginTime
long long int mjGetTimeHR(void);

// sleep for given number of milliseconds
void mjSleep(unsigned int msec);

// }	// namespace mujoco