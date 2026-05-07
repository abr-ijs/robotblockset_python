"""Robotiq gripper interface via pyRobotiqGripper.

High-level interface for controlling Robotiq 2F grippers through the
`pyrobotiqgripper` package.

Copyright (c) 2026 Jozef Stefan Institute

Authors: Leon Zlajpah
"""

from typing import Any, Optional

try:
    from pyrobotiqgripper import RobotiqGripper  # pyright: ignore[reportMissingImports]
except Exception as e:
    raise e from RuntimeError("Python bindings for the Robotiq gripper are not installed.\n" "You can install them through pip:\n" "  pip install pyrobotiqgripper")

from robotblockset.grippers import gripper


class robotiq_gripper(RobotiqGripper, gripper):
    """RobotBlockset wrapper around `pyrobotiqgripper.RobotiqGripper`.

    The upstream library communicates with the gripper over Modbus RTU and uses
    a serial port connection. Width commands are expressed in meters in the
    RobotBlockset API and are converted to millimeters for the backend.
    """

    def __init__(
        self,
        name: str = "robotiq_gripper",
        portname: str = "auto",
        slave_address: int = 9,
        activate: bool = True,
        calibrate: bool = False,
        width_min: float = 0.0,
        width_max: float = 0.085,
        **kwargs: Any,
    ) -> None:
        """Initialize the Robotiq gripper interface.

        Parameters
        ----------
        name : str, optional
            Name identifier for the gripper instance.
        portname : str, optional
            Serial port used by `pyrobotiqgripper`. `"auto"` enables the package
            auto-detection logic.
        slave_address : int, optional
            Modbus slave address of the gripper, usually `9`.
        activate : bool, optional
            If `True`, activate the gripper during initialization.
        calibrate : bool, optional
            If `True`, calibrate the gripper using `width_min` and `width_max`.
        width_min : float, optional
            Closed gripper width in meters used for calibration.
        width_max : float, optional
            Open gripper width in meters used for calibration.
        **kwargs : Any
            Additional keyword arguments reserved for future extensions.
        """
        gripper.__init__(self)
        RobotiqGripper.__init__(self, com_port=portname, device_id=slave_address, **kwargs)

        self.Name = name
        self.portname = portname
        self.slave_address = slave_address
        self._width_grasp = width_min
        self._width = width_max
        self._width_min = width_min
        self._width_max = width_max
        self._state = -1
        self._speed = 255
        self._speed_max = 255
        self._force = 255
        self._force_max = 255

        self.connect()

        if activate:
            self.Activate()
        if calibrate:
            self.Calibrate(width_min=width_min, width_max=width_max)

        self._update_state()
        self.Message("Created", 2)

    @property
    def width(self) -> float:
        """Get the current gripper width in meters."""
        return self.GetWidth()

    def Activate(self) -> None:
        """Activate the physical gripper."""
        self.activate()

    def Calibrate(self, width_min: float = 0.0, width_max: float = 0.085) -> None:
        """Calibrate the gripper for mm-based commands and feedback."""
        self.calibrate(width_min * 1000.0, width_max * 1000.0)
        self._width_min = width_min
        self._width_max = width_max
        self._update_state()

    def GetWidth(self) -> float:
        """Return the current opening width in meters."""
        self._update_state()
        return self._width

    def is_grasped(self) -> bool:
        """Return `True` if the backend reports contact while closing."""
        self._refresh_registers()
        gobj = self.status.get("gOBJ")
        return gobj == 2

    def GetState(self) -> str:
        """Return the gripper state as a human-readable string."""
        self._update_state()
        return gripper.GetState(self)

    def Grasp(self, width: float = 0.0, speed: int = 255, force: int = 255, **kwargs: Any) -> bool:
        """Close the gripper to `width` and report whether an object was detected."""
        del kwargs
        success = self.Move(width=width, speed=speed, force=force)
        if success and self.is_grasped():
            self._state = 1
        return success

    def Move(self, width: float, speed: int = 255, force: int = 255) -> bool:
        """Move the gripper to the requested width in meters."""
        target_width = min(max(width, self._width_min), self._width_max)
        target_mm = target_width * 1000.0
        target_bits = self._width_to_bits(target_width)

        self._state = -1
        self._speed = int(max(0, min(speed, self._speed_max)))
        self._force = int(max(0, min(force, self._force_max)))

        try:
            if self._supports_mm():
                self.move_mm(target_mm, self._speed, self._force)
            else:
                self.move(target_bits, self._speed, self._force)
        except Exception:
            self.Message("Gripper move failed", 2)
            raise

        self._update_state(target_width=target_width)
        return True

    def Open(self, speed: int = 255, force: int = 255, **kwargs: Any) -> bool:
        """Open the gripper to its calibrated maximum width."""
        del kwargs
        return self.Move(self._width_max, speed=speed, force=force)

    def Close(self, speed: int = 255, force: int = 255, **kwargs: Any) -> bool:
        """Close the gripper."""
        del kwargs
        return self.Move(self._width_min, speed=speed, force=force)

    def Homing(self, **kwargs: Any) -> bool:
        """Open the gripper after optionally re-running activation."""
        reactivate = kwargs.pop("reactivate", False)
        if reactivate:
            self.Activate()
        return self.Open(**kwargs)

    def _supports_mm(self) -> bool:
        """Return `True` when the backend can accept mm commands."""
        return bool(getattr(self, "isCalibrated", False))

    def _refresh_registers(self) -> None:
        """Refresh cached backend registers when the API exposes them."""
        if hasattr(self, "readStatus"):
            self.readStatus()

    def _update_state(self, target_width: Optional[float] = None) -> None:
        """Refresh cached width and infer the RobotBlockset state."""
        self._refresh_registers()

        try:
            if self._supports_mm() and hasattr(self, "getPositionmm"):
                self._width = float(self.getPositionmm()) / 1000.0
            elif hasattr(self, "getPosition"):
                self._width = self._bits_to_width(int(self.getPosition()))
        except Exception:
            pass

        if target_width is None:
            target_width = self._width

        eps = 0.002
        if self.is_grasped() or self._width <= (self._width_min + eps):
            self._state = 1
        elif self._width >= (self._width_max - eps):
            self._state = 0
        elif abs(self._width - target_width) <= eps:
            self._state = -1

    def _width_to_bits(self, width: float) -> int:
        """Convert width in meters to the Robotiq 0..255 position scale."""
        span = self._width_max - self._width_min
        if span <= 0:
            return 0
        normalized = (self._width_max - width) / span
        return int(round(max(0.0, min(normalized, 1.0)) * 255.0))

    def _bits_to_width(self, bits: int) -> float:
        """Convert the Robotiq 0..255 position scale to meters."""
        clamped_bits = max(0, min(int(bits), 255))
        span = self._width_max - self._width_min
        if span <= 0:
            return self._width_min
        normalized = 1.0 - (clamped_bits / 255.0)
        return self._width_min + normalized * span
