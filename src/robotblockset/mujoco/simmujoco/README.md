# simmujoco

## Introduction

RBS **simmujoco** is an extended version of the MuJoCo `simulate` application with an added socket server for communication with external user programs. The socket interface is based on the MuJoCo [HAPTIX](https://roboti.us/book/haptix.html) API and extends its communication protocol for RobotBlockSet use.

## Building From Source

To build MuJoCo from source, you need CMake and a working C++17 compiler. First download the MuJoCo sources from the [MuJoCo releases page](https://github.com/deepmind/mujoco/releases), for example [version 3.3.5](https://github.com/google-deepmind/mujoco/archive/refs/tags/3.3.5.zip). The general MuJoCo build flow is documented in the [official MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/programming/index.html#building-from-source).

The general build steps are:

1. Clone the `mujoco` repository from GitHub.
2. Create a new `build` directory and `cd` into it.
3. Run `cmake $PATH_TO_CLONED_REPO` to configure the build.
4. Run `cmake --build .` or `make` to build.
5. Optionally configure an install target with
   `cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`.
6. Install with `cmake --install .`.

## Preparing the Sources

To build the **simmujoco** the steps are similar. After cloning  the `mujoco` repository from GitHub, it is necessary to patch the original MuJoCo sources to add the socket server and build the modified binaries. 

The current `simmujoco` sources are prepared for [MuJoCo 3.6.0](https://github.com/google-deepmind/mujoco/archive/refs/tags/3.6.0.zip). If you build exactly this version, copy the following files from `<simmujoco>/simulate/` into `<mujoco-3.6.0>/simulate/`:

```text
CMakeLists.txt
crossplatform.cpp
crossplatform.h
main.cc
mujoco_server.cpp
mujoco_server.h
simulate.cc
socket.cpp
socket.h
```

For other nearby MuJoCo versions, copy only the socket-related sources from `<simmujoco>/simulate/` into `<mujoco-3.6.0>/simulate/`:

```text
crossplatform.cpp
crossplatform.h
mujoco_server.cpp
mujoco_server.h
socket.cpp
socket.h
```

Then, patch `CMakeLists.txt`, `simulate.cc`, and `main.cc` to include the server. The folder `<simmujoco>/simulate/` contains the reference originals `simulate.cc.ori` and `main.cc.ori`, which can be compared against the modified files to see the required changes.

## Building Binaries

After the sources are prepared, build the binaries.

### Windows

1. Open a Microsoft Visual Studio native command prompt for x64.
2. `cd` into the `<mujoco-3.6.8>` folder.
3. Create a new `build` directory and `cd` into it.
4. Run:

```text
cmake .. -DCMAKE_INSTALL_PREFIX=../simmujoco
cmake --build . --config Release
cmake --install .
```

The resulting binaries are placed in `<mujoco-3.6.0>\simmujoco\bin` and
`<mujoco-3.6.0>\simmujoco\lib`.

### Linux

1. Clone the `mujoco` repository and `cd` into it.
2. Create a new `build` directory and `cd` into it.
3. Run:

```text
cmake .. -DCMAKE_INSTALL_PREFIX=../simmujoco
cmake --build . --config Release
cmake --install .
```

The resulting binaries are placed in `<mujoco-3.6.0>/simmujoco/bin` and `<mujoco-3.6.0>/simmujoco/lib`.

## Quick Start

### Windows

To start **simmujoco**, run:

```text
simmujoco
```

Or specify the socket host, port, and model to load:

```text
simmujoco localhost 50000 my_model.xml
```

Because **simmujoco** opens a listening socket, Windows will usually display a firewall dialog the first time you run it. Click "Allow access" to create an inbound rule. If you click "Cancel", the application may still work from the local machine but incoming connections from other machines will be blocked.

### Linux

Set execute permissions for `simmujoco`:

```text
chmod +x <mujoco-3.6.0>/simmujoco/bin/simmujoco
```

Then start it with:

```text
<mujoco-3.6.0>/simmujoco/bin/simmujoco
```

Or specify the socket host, port, and model to load:

```text
<mujoco-3.6.0>/simmujoco/bin/simmujoco localhost 50000 my_model.xml
```
