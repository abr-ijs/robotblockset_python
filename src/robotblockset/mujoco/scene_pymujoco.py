"""Scene utilities for interactive Python MuJoCo simulations.

This module provides the `mujoco_scene` helper used by RobotBlockSet MuJoCo
backends to load models, manage the passive viewer, render auxiliary camera
windows, and control simulation stepping and resets.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from time import perf_counter, sleep, time
import threading

try:
    import mujoco
    import mujoco.viewer
    import glfw
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

import numpy as np
from typing import List, Optional, Union
from robotblockset.tools import rbs_object


class mujoco_scene(rbs_object):
    """
    MuJoCo scene manager with viewer and optional camera-window support.

    Attributes
    ----------
    spec : Optional[mujoco.MjSpec]
        MuJoCo specification object when the scene was loaded from XML.
    model : mujoco.MjModel
        Compiled MuJoCo model used by the scene.
    data : mujoco.MjData
        Runtime MuJoCo data associated with `model`.
    pause : bool
        Flag indicating whether simulation stepping is paused.
    synchronized : bool
        Whether simulation stepping is synchronized to the model timestep.
    viewer : Optional[mujoco.viewer.Handle]
        Passive MuJoCo viewer used for the main scene window.
    show_camera : list[Union[str, int]]
        Camera names or IDs rendered in additional windows.
    visual_thread : threading.Thread
        Background thread that owns the viewer loop.
    """

    def __init__(self, model_xml_file: Optional[str] = None, model: Optional[mujoco.MjModel] = None, show_camera: Optional[List[Union[str, int]]] = None, synchronized: bool = True, verbose: int = 0) -> None:
        """Create a MuJoCo scene manager.

        Parameters
        ----------
        model_xml_file : str, optional
            Path to the MuJoCo XML model file to load.
        model : mujoco.MjModel, optional
            Existing MuJoCo model to use instead of loading from XML.
        show_camera : Optional[List[Union[str, int]]], optional
            Camera names or IDs to render in auxiliary windows.
        synchronized : bool, optional
            If `True`, synchronize stepping to the MuJoCo model timestep.
        verbose : int, optional
            Verbosity level used for status messages.

        Returns
        -------
        None
            This constructor initializes the MuJoCo scene object in place.
        """
        super().__init__()
        self._verbose = verbose
        self.Name = "MuJoCo Scene"
        if model is not None:
            self.spec = None
            self.model = model
        elif model_xml_file is not None:
            self.spec = mujoco.MjSpec.from_file(model_xml_file)
            self.model = self.spec.compile()
        else:
            raise ValueError("Either model_xml_file or model must be provided.")
        self.data = mujoco.MjData(self.model)
        self.pause = False
        self.synchronized = synchronized
        self._initial_time = time()
        self._samples = 0
        self.Message("Model loaded successfully.", 2)

        # Store camera options
        self.show_camera = show_camera if show_camera is not None else []

        # Start the visualization thread
        self.viewer = None
        self.visual_thread = threading.Thread(target=self.render, daemon=True)
        self.visual_thread.start()
        self._connected = 1
        self._last_step_time = perf_counter() - self.model.opt.timestep
        self.Message("Viewer started.", 2)

    def _synchro_simulation(self) -> None:
        """
        Synchronize simulation stepping to the configured model timestep.

        Returns
        -------
        None
            This method delays the caller so the next step matches the target timestep.
        """
        next_time = self._last_step_time + self.model.opt.timestep
        self._remaining = next_time - time()
        if self._remaining > 0:
            sleep(self._remaining)
        self._last_step_time = time()

    def start_simulation(self) -> None:
        """
        Start the visualization thread if it is not already running.

        Returns
        -------
        None
            This method launches the background render loop.
        """
        if not self.visual_thread.is_alive():
            self.visual_thread = threading.Thread(target=self.render, daemon=True)
            self.visual_thread.start()
            self.Message("Viewer started.", 2)

    def stop_simulation(self) -> None:
        """
        Stop the visualization thread and close viewer windows.

        Returns
        -------
        None
            This method stops the background render loop.
        """
        self.abort_viewer = True
        if self.visual_thread.is_alive():
            self.visual_thread.join()
            self.Message("Viewer stopped.", 2)

    def restart_simulation(self) -> None:
        """
        Restart the viewer thread and recompile the model when possible.

        Returns
        -------
        None
            This method restarts visualization using the current scene state.
        """
        self.stop_simulation()
        if self.spec is not None:
            self.model, self.data = self.spec.recompile(self.model, self.data)
        self.start_simulation()

    def render(self) -> None:
        """
        Renders the simulation and the camera windows (if specified).

        Runs the simulation and handles rendering for the main viewer and any additional camera windows.
        Also handles input events, stepping the simulation, and updating the viewer's state.

        Returns
        -------
        None
        """
        self.abort_viewer = False

        # Key callback for stopping the viewer
        def key_callback(keycode: int) -> None:
            """
            Callback function for handling key inputs in the viewer.

            Parameters
            ----------
            keycode : int
                The keycode of the pressed key.

            Returns
            -------
            None
            """
            if chr(keycode) == "Q":
                self.abort_viewer = True
            elif chr(keycode) == " ":
                self.pause = not self.pause

        # Create a viewer for the simulation
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)
        with self.viewer as viewer:
            self.Message("Viewer launched, running simulation...", 2)

            # Create additional windows for the cameras
            cam_windows = []
            default_w, default_h = 400, 300  # sensible 4:3 default
            if self.show_camera:
                # Initialize additional windows for each camera
                for cam in self.show_camera:
                    cam_id = cam if isinstance(cam, int) else self.model.camera(cam).id
                    if 0 <= cam_id < self.model.ncam:
                        # Create a GLFW window for the camera
                        win = glfw.create_window(default_w, default_h, f"Camera {cam_id}", None, None)
                        if not win:
                            continue
                        glfw.make_context_current(win)

                        # A separate MjrContext per window (tied to that GL context)
                        ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

                        # A fixed camera pointing to this defined camera
                        cam = mujoco.MjvCamera()
                        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        cam.fixedcamid = cam_id
                        cam_windows.append({"id": cam_id, "win": win, "ctx": ctx, "cam": cam})

            if cam_windows:
                scene = mujoco.MjvScene(self.model, maxgeom=20_000)
                opt = mujoco.MjvOption()
                opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                # Example: hide labels/contacts if you like
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT.value] = 0
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE.value] = 1

            while viewer.is_running() and not self.abort_viewer and (not cam_windows or any(not glfw.window_should_close(cw["win"]) for cw in cam_windows)):
                if self.synchronized:
                    expected_time = self._samples * self.model.opt.timestep
                    time_diff = expected_time - (time() - self._initial_time)
                    if time_diff > 0:
                        sleep(time_diff)
                    self._samples += 1

                # Sync with the main viewer (main simulation)
                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    pass

                # Poll events once per loop
                glfw.poll_events()

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                if not self.pause:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    # # Render each camera window
                    for cw in cam_windows:
                        win = cw["win"]
                        if glfw.window_should_close(win):
                            continue
                        glfw.make_context_current(win)
                        fbw, fbh = glfw.get_framebuffer_size(win)

                        # Update the scene and render the camera view
                        mujoco.mjv_updateScene(self.model, self.data, opt, None, cw["cam"], mujoco.mjtCatBit.mjCAT_ALL, scene)
                        mujoco.mjr_render(mujoco.MjrRect(0, 0, fbw, fbh), scene, cw["ctx"])
                        glfw.swap_buffers(win)

                alive = []
                for cw in cam_windows:
                    if glfw.window_should_close(cw["win"]):
                        glfw.destroy_window(cw["win"])
                        self.Message(f"Camera {cw['id']} view closed.", 2)
                    else:
                        alive.append(cw)
                cam_windows = alive

            for cw in cam_windows:
                glfw.destroy_window(cw["win"])
                self.Message(f"Camera {cw['id']} view closed.", 2)
            self.Message("Viewer closed.", 2)

    def mj_pause(self) -> None:
        """
        Pause simulation stepping.

        Returns
        -------
        None
            This method sets the internal pause flag.
        """
        self.pause = True

    def mj_run(self) -> None:
        """
        Resume simulation stepping.

        Returns
        -------
        None
            This method clears the internal pause flag.
        """
        self.pause = False

    def mj_wait(self, wait: float = 0) -> None:
        """
        Wait for a specified duration.

        Parameters
        ----------
        wait : float, optional
            Amount of simulated time (in seconds) to advance.

        Returns
        -------
        None
            This method blocks until the requested amount of simulated time has elapsed.
        """
        _t0 = self.data.time
        while self.data.time < _t0 + wait:
            sleep(self.model.opt.timestep)

    def mj_reset(self, keyframe: Optional[int] = None) -> None:
        """
        Resets the simulation state to its initial state, or to a specified keyframe.

        Parameters
        ----------
        keyframe : int, optional
            The index of the keyframe to reset to. If None, the simulation is reset to the initial state.

        Returns
        -------
        None
        """
        if keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)

    def mj_capture_camera(self, camera: Union[int, str] = -1, scene_option: Optional[mujoco.MjvOption] = None, **kwargs: object) -> np.ndarray:
        """
        Captures an image from a specified camera in the simulation.

        Parameters
        ----------
        camera : Union[int, str], optional
            The ID or name of the camera to capture. Default is -1, which captures the default camera.
        scene_option : mujoco.MjvOption, optional
            Rendering options to customize the scene appearance. Default is None.
        **kwargs : object
            Additional keyword arguments passed to `mujoco.Renderer`.

        Returns
        -------
        np.ndarray
            The rendered image as a NumPy array (RGB format).
        """
        with mujoco.Renderer(self.model, **kwargs) as renderer:
            renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
            frame = renderer.render()
            return frame
