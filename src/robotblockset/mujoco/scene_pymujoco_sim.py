"""Scene utilities for synchronous Python MuJoCo simulations.

This module provides the `mujoco_scene` helper used by RobotBlockSet synchronous
MuJoCo backends to load models, manage the viewer, render auxiliary camera
windows, and control simulation stepping and resets.

Copyright (c) 2025 Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
from time import perf_counter, sleep, time
from typing import List, Optional, Union

try:
    import mujoco
    import mujoco.viewer
    import glfw
except Exception as e:
    raise e from RuntimeError("MuJoCo not installed. \nYou can install MuJoCo through pip:\n   pip install mujoco")

from robotblockset.tools import rbs_object


class mujoco_scene(rbs_object):
    """
    MuJoCo scene manager with viewer support for synchronous backends.

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
    viewer : Optional[mujoco.viewer.Handle]
        Passive MuJoCo viewer used for the main scene window.
    show_camera : list[Union[str, int]]
        Camera names or IDs rendered in additional windows.
    cam_windows : list[dict[str, object]]
        Metadata for additional GLFW camera windows.
    """

    def __init__(self, model_xml_file: Optional[str] = None, model: Optional[mujoco.MjModel] = None, show_viewer: bool = True, show_camera: Optional[List[Union[str, int]]] = None, verbose: int = 0) -> None:
        """Create a synchronous MuJoCo scene manager.

        Parameters
        ----------
        model_xml_file : str, optional
            Path to the MuJoCo XML model file to load.
        model : mujoco.MjModel, optional
            Existing MuJoCo model to use instead of loading from XML.
        show_viewer : bool, optional
            If `True`, open the main passive MuJoCo viewer.
        show_camera : Optional[List[Union[str, int]]], optional
            Camera names or IDs to render in auxiliary windows.
        verbose : int, optional
            Verbosity level used for status messages.

        Returns
        -------
        None
            This constructor initializes the synchronous MuJoCo scene object in place.
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
        mujoco.mj_forward(self.model, self.data)
        self.pause = False
        self._initial_time = time()
        self._samples = 0
        self.Message("Model loaded successfully.", 2)

        self._connected = 1
        self._last_step_time = perf_counter() - self.model.opt.timestep

        self.viewer = None
        self._viewer_active = show_viewer
        self.cam_windows = []
        self.show_camera = show_camera if show_camera is not None else []

        self.start_simulation()

        self.Message("Viewer started.", 2)

    def _synchro_simulation(self) -> None:
        """Synchronize simulation stepping to the configured model timestep.

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
        Start the viewer and auxiliary camera windows.

        Returns
        -------
        None
            This method creates the passive viewer and any requested camera windows.
        """
        if self._viewer_active:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
                        self.cam_windows.append({"id": cam_id, "win": win, "ctx": ctx, "cam": cam})

        if self.cam_windows:
            self.scene = mujoco.MjvScene(self.model, maxgeom=20_000)
            self.opt = mujoco.MjvOption()
            self.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            # Example: hide labels/contacts if you like
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT.value] = 0
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE.value] = 1

    def stop_simulation(self) -> None:
        """
        Stop rendering by closing the passive viewer.

        Returns
        -------
        None
            This method closes the main viewer window.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def restart_simulation(self) -> None:
        """
        Restart rendering and recompile the model when possible.

        Returns
        -------
        None
            This method refreshes the scene after stopping the viewer.
        """
        self.stop_simulation()
        if self.spec is not None:
            self.model, self.data = self.spec.recompile(self.model, self.data)
        self.start_simulation()

    def update_cam_windows(self) -> None:
        """Render all active auxiliary camera windows.

        Returns
        -------
        None
            This method updates and prunes additional GLFW camera windows.
        """
        if not self.cam_windows:
            return

        alive = []
        for cw in self.cam_windows:
            win = cw["win"]
            if glfw.window_should_close(win):
                glfw.destroy_window(win)
                self.cam_windows.remove(cw)
                continue
            else:
                alive.append(cw)

            # Get framebuffer viewport
            glfw.make_context_current(win)
            fb_w, fb_h = glfw.get_framebuffer_size(win)
            viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)

            # Update scene and render
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, cw["cam"], mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, cw["ctx"])

            # Swap OpenGL buffers (blocking vsync)
            glfw.swap_buffers(win)
            # Process window events (keyboard, mouse, etc.)
            glfw.poll_events()
        self.cam_windows = alive

    def mj_step(self) -> None:
        """
        Advances the simulation by one time step.

        Returns
        -------
        None
            This method steps the MuJoCo simulation and refreshes viewers.
        """
        if not self.pause:
            mujoco.mj_step(self.model, self.data)
            if self._viewer_active and self.viewer.is_running():
                self.viewer.sync()
            self.update_cam_windows()

    def mj_forward(self) -> None:
        """
        Advances the simulation by using forward dynamics.

        Returns
        -------
        None
            This method recomputes forward dynamics and refreshes viewers.
        """
        if not self.pause:
            mujoco.mj_forward(self.model, self.data)
            if self._viewer_active and self.viewer.is_running():
                self.viewer.sync()
            self.update_cam_windows()

    def mj_pause(self) -> None:
        """Pause simulation stepping.

        Returns
        -------
        None
            This method sets the internal pause flag.
        """
        self.pause = True

    def mj_run(self) -> None:
        """Resume simulation stepping.

        Returns
        -------
        None
            This method clears the internal pause flag.
        """
        self.pause = False

    def mj_wait(self, wait: float = 0) -> None:
        """
        Advance the MuJoCo simulation for a specified duration.

        This method performs repeated simulation steps until the internal
        simulation time (`self.data.time`) has advanced by the given
        amount. Unlike a passive delay (e.g., ``time.sleep``), this
        function actively progresses the MuJoCo physics simulation.

        Parameters
        ----------
        wait : float, optional
            Amount of simulated time (in seconds) to advance.
            A value of ``0`` (default) performs no additional steps.

        Notes
        -----
        - Simulation progresses by repeatedly calling ``self.mj_step``.
        - The function returns only after the MuJoCo model time exceeds
          the initial time plus ``wait``.
        - This does *not* block real time; the speed of advancement
          depends on computation speed and simulation complexity.

        Returns
        -------
        None
            This method blocks until the requested amount of simulated time has elapsed.
        """
        _t0 = self.data.time
        while self.data.time < _t0 + wait:
            self.mj_step()

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
        mujoco.mj_forward(self.model, self.data)

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
