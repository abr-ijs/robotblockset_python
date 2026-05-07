/*
  Windows implementation of MuJoCo OS-specific functions

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

#include "crossplatform.h"
#include "stdio.h"

#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
#endif

// namespace mujoco {

//---------------------------- Timing ---------------------------------------------------
static bool _tmInitialized = false;
static int _tmBase;

#ifdef _WIN32
static LARGE_INTEGER _tmFreqHR;
static LARGE_INTEGER _tmBaseHR;

// initialize 1 msec timer, record base time
void mjBeginTime(void)
{
	if( _tmInitialized )
		return;

	// initialize multimedia timer
	timeBeginPeriod(1);
	_tmBase = timeGetTime();

	// initialize high performance timer
	QueryPerformanceFrequency(&_tmFreqHR);
	QueryPerformanceCounter(&_tmBaseHR);

	_tmInitialized = true;
}

// close timer
void mjEndTime(void)
{
	if( _tmInitialized )
	{
		timeEndPeriod(1);
		_tmInitialized = false;
	}
}

// get time in milliseconds scince initialization
int mjGetTime(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	return (timeGetTime() - _tmBase);
}

// get time in microseconds scince initialization
long long int mjGetTimeHR(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	LARGE_INTEGER tm;
	QueryPerformanceCounter(&tm);
	return (long long int) (((tm.QuadPart - _tmBaseHR.QuadPart)*1000000)/_tmFreqHR.QuadPart);
}

// sleep for given number of milliseconds
void mjSleep(unsigned int msec)
{
	if( !_tmInitialized )
		mjBeginTime();

	Sleep(msec);
}

#else

unsigned int timeGetTime()
{
	/*
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec/1000;
    */
	struct timespec now;   
	clock_gettime(CLOCK_MONOTONIC, &now);   
	return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0; 
}

// initialize 1 msec timer, record base time
void mjBeginTime(void)
{
	if( _tmInitialized )
		return;

	// initialize multimedia timer
	_tmBase = timeGetTime();

	_tmInitialized = true;
}

// close timer
void mjEndTime(void)
{
	if( _tmInitialized )
	{
		_tmInitialized = false;
	}
}

// get time in milliseconds scince initialization
int mjGetTime(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	return (timeGetTime() - _tmBase);
}

// get time in microseconds scince initialization
long long int mjGetTimeHR(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	return (long long int) (timeGetTime() * 1000.0);
}

// sleep for given number of milliseconds
void mjSleep(unsigned int msec)
{
	if( !_tmInitialized )
		mjBeginTime();

	 usleep(msec * 1000);   // usleep takes sleep time in us 
}	
#endif

// }	// namespace mujoco