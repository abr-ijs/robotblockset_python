"""SpaceNavigator and 3Dconnexion HID interface utilities.

This module provides low-level support for reading 6-DoF input devices such as
the 3Dconnexion SpaceNavigator over HID. It defines device mappings, callback
helpers, configuration containers, and utility functions used to decode
translation, rotation, and button events from supported devices.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

from __future__ import annotations

import copy
import threading
import timeit
from collections import namedtuple
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from easyhid import Enumeration, HIDException


# clock for timing
high_acc_clock = timeit.default_timer

# axis mappings are specified as:
# [channel, byte1, byte2, scale]; scale is usually just -1 or 1 and multiplies the result by this value
# (but per-axis scaling can also be achieved by setting this value)
# byte1 and byte2 are indices into the HID array indicating the two bytes to read to form the value for this axis
# For the SpaceNavigator, these are consecutive bytes following the channel number.
AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

# button states are specified as:
# [channel, data byte,  bit of byte, index to write to]
# If a message is received on the specified channel, the value of the data byte is set in the button bit array
ButtonSpec = namedtuple("ButtonSpec", ["channel", "byte", "bit"])

# Simple HID code to read data from the 3dconnexion devices
# convert two 8 bit bytes to a signed 16 bit integer


def to_int16(y1: int, y2: int) -> int:
    """
    Convert two unsigned bytes into a signed 16-bit integer.

    Parameters
    ----------
    y1 : int
        Low byte of the HID value.
    y2 : int
        High byte of the HID value.

    Returns
    -------
    int
        Signed 16-bit integer reconstructed from ``y1`` and ``y2``.
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


# tuple for 6DOF results
SpaceNavigator = namedtuple("SpaceNavigator", ["t", "x", "y", "z", "roll", "pitch", "yaw", "buttons"])


class ButtonState(list):
    """List-like container that stores decoded button states."""

    def __int__(self) -> int:
        """
        Convert the button state vector to a bit-packed integer.

        Returns
        -------
        int
            Integer whose bits encode the button states.
        """
        return sum((b << i) for (i, b) in enumerate(reversed(self)))


class ButtonCallback:
    """
    Configuration block for button-triggered callbacks.

    Parameters
    ----------
    buttons : Union[int, List[int]]
        Button index or list of button indices that must be active to trigger
        the callback.
    callback : Callable[[SpaceNavigator, ButtonState, Union[int, List[int]]], None]
        Function called when the requested button combination is active.
    """

    def __init__(self, buttons: Union[int, List[int]], callback: Callable[["SpaceNavigator", "ButtonState", Union[int, List[int]]], None]) -> None:
        """
        Initialize a button callback configuration.

        Parameters
        ----------
        buttons : Union[int, List[int]]
            Button index or button combination to monitor.
        callback : Callable[[SpaceNavigator, ButtonState, Union[int, List[int]]], None]
            Function invoked when the requested button state is active.
        """
        self.buttons = buttons
        self.callback = callback


class DofCallback:
    """
    Configuration block for per-axis DoF callbacks.

    Parameters
    ----------
    axis : str
        Name of the monitored axis. Must be one of ``"x"``, ``"y"``, ``"z"``,
        ``"roll"``, ``"pitch"``, or ``"yaw"``.
    callback : Callable[[SpaceNavigator, float], None]
        Function invoked when the axis exceeds the configured filter threshold
        in the positive direction.
    sleep : float, optional
        Minimum time in seconds between consecutive callback invocations.
    callback_minus : Callable[[SpaceNavigator, float], None], optional
        Optional function invoked when the axis exceeds the filter threshold in
        the negative direction.
    filter : float, optional
        Absolute threshold that the axis value must exceed before a callback is
        triggered.
    """

    def __init__(self, axis: str, callback: Callable[["SpaceNavigator", float], None], sleep: float = 0.0, callback_minus: Optional[Callable[["SpaceNavigator", float], None]] = None, filter: float = 0.0) -> None:
        """
        Initialize a DoF callback configuration.

        Parameters
        ----------
        axis : str
            Name of the monitored axis.
        callback : Callable[[SpaceNavigator, float], None]
            Function invoked for positive axis motion.
        sleep : float, optional
            Minimum delay between callback invocations.
        callback_minus : Callable[[SpaceNavigator, float], None], optional
            Optional function invoked for negative axis motion.
        filter : float, optional
            Absolute motion threshold used to suppress small inputs.
        """
        self.axis = axis
        self.callback = callback
        self.sleep = sleep
        self.callback_minus = callback_minus
        self.filter = filter


class Config:
    """
    Container for validated SpaceMouse callback configuration.

    Parameters
    ----------
    callback : Callable[[SpaceNavigator], None], optional
        Generic callback invoked for every processed event.
    dof_callback : Callable[[SpaceNavigator], None], optional
        Callback invoked when any DoF value changes.
    dof_callback_arr : List[DofCallback], optional
        List of per-axis callback configurations.
    button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
        Callback invoked when button states change.
    button_callback_arr : List[ButtonCallback], optional
        List of button-combination callback configurations.
    """

    def __init__(
        self,
        callback: Optional[Callable[["SpaceNavigator"], None]] = None,
        dof_callback: Optional[Callable[["SpaceNavigator"], None]] = None,
        dof_callback_arr: Optional[List[DofCallback]] = None,
        button_callback: Optional[Callable[["SpaceNavigator", ButtonState], None]] = None,
        button_callback_arr: Optional[List[ButtonCallback]] = None,
    ) -> None:
        """
        Initialize and validate a callback configuration bundle.

        Parameters
        ----------
        callback : Callable[[SpaceNavigator], None], optional
            Generic event callback.
        dof_callback : Callable[[SpaceNavigator], None], optional
            Callback invoked when DoF state changes.
        dof_callback_arr : List[DofCallback], optional
            Per-axis callback configurations.
        button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
            Callback invoked when button state changes.
        button_callback_arr : List[ButtonCallback], optional
            Button callback configurations.
        """
        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr


class DeviceSpec(object):
    """
    Runtime representation of a supported 3Dconnexion device.

    The object stores the HID identifiers, axis and button mappings, decoded
    state, and active callbacks associated with a single device model.
    """

    def __init__(
        self,
        name: str,
        hid_id: List[int],
        led_id: Optional[List[int]],
        mappings: Dict[str, AxisSpec],
        button_mapping: List[ButtonSpec],
        axis_scale: float = 350.0,
    ) -> None:
        """
        Initialize a device specification and its runtime state.

        Parameters
        ----------
        name : str
            Human-readable device name.
        hid_id : List[int]
            Vendor and product identifier pair.
        led_id : List[int], optional
            Optional LED usage identifier pair.
        mappings : Dict[str, AxisSpec]
            Mapping from logical axis names to HID byte locations.
        button_mapping : List[ButtonSpec]
            Mapping from logical button indices to HID message bits.
        axis_scale : float, optional
            Scale factor used to normalize raw HID axis values.
        """
        self.name = name
        self.hid_id = hid_id
        self.led_id = led_id
        self.__mappings = mappings
        self.button_mapping = button_mapping
        self.axis_scale = axis_scale
        self.__bytes_to_read = self.__get_num_bytes_to_read()
        self.dof_changed = False
        self.buttons_changed = False

        # self.led_usage = hid.get_full_usage_id(led_id[0], led_id[1])
        # initialise to a vector of 0s for each state
        self.dict_state = {
            "t": -1,
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "buttons": ButtonState([0] * len(self.button_mapping)),
        }
        # initialise to a vector for button_callback_arr timer
        self.dict_state_last = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        self.tuple_state = SpaceNavigator(**self.dict_state)

        # start in disconnected state
        self.device = None
        self.callback = None
        self.dof_callback = None
        self.dof_callback_arr = None
        self.button_callback = None
        self.button_callback_arr = None
        self.set_nonblocking_loop = True

    def __get_num_bytes_to_read(self) -> int:
        """
        Compute the largest HID packet size needed for configured axis mappings.

        Returns
        -------
        int
            Number of bytes that must be read to decode all mapped axes.
        """
        byte_indices = []
        for value in self.__mappings.values():
            byte_indices.extend([value.byte1, value.byte2])

        return max(byte_indices) + 1

    def describe_connection(self) -> str:
        """
        Return a human-readable description of the current device connection.

        Returns
        -------
        str
            Description of the device model and connection state.
        """
        if self.device is None:
            return f"{self.name} [disconnected]"
        else:
            return f"{self.name} connected to {self.vendor_name} {self.product_name} version: {self.version_number} [serial: {self.serial_number}]"

    @property
    def mappings(self) -> Dict[str, AxisSpec]:
        """
        Get the configured logical-axis mapping.

        Returns
        -------
        Dict[str, AxisSpec]
            Mapping from axis names to HID byte descriptors.
        """
        return self.__mappings

    @mappings.setter
    def mappings(self, val: Dict[str, AxisSpec]) -> None:
        """
        Set the logical-axis mapping and recompute the read packet size.

        Parameters
        ----------
        val : Dict[str, AxisSpec]
            New mapping from axis names to HID byte descriptors.
        """
        self.__mappings = val
        self.__bytes_to_read = self.__get_num_bytes_to_read()

    @property
    def connected(self) -> bool:
        """
        Check whether the HID device is currently open.

        Returns
        -------
        bool
            ``True`` if a device handle is present, otherwise ``False``.
        """
        return self.device is not None

    @property
    def state(self) -> Optional["SpaceNavigator"]:
        """
        Return the current device state.

        Returns
        -------
        Optional[SpaceNavigator]
            Most recent decoded state, or ``None`` if the device is not open.
        """
        return self.read()

    def open(self) -> None:
        """
        Open the underlying HID device and cache its metadata.

        Raises
        ------
        Exception
            If the HID device cannot be opened.
        """
        if self.device:
            try:
                self.device.open()
            except HIDException as e:
                raise Exception("Failed to open device") from e

        # copy in product details
        self.product_name = self.device.product_string
        self.vendor_name = self.device.manufacturer_string
        self.version_number = self.device.release_number
        # doesn't seem to work on 3dconnexion devices...
        # serial number will be a byte string, we convert to a hex id
        self.serial_number = "".join(["%02X" % ord(char) for char in self.device.serial_number])

    # def set_led(self, state):
    #     """Set the LED state to state (True or False)"""
    #     if self.connected:
    #         reports = self.device.find_output_reports()
    #         for report in reports:
    #             if self.led_usage in report:
    #                 report[self.led_usage] = state
    #                 report.send()

    def close(self) -> None:
        """
        Close the underlying HID device if it is open.

        Returns
        -------
        None
        """
        if self.connected:
            self.device.close()
            self.device = None

    def read(self) -> Optional["SpaceNavigator"]:
        """
        Read and decode the current SpaceMouse state.

        Returns
        -------
        Optional[SpaceNavigator]
            Current decoded device state, or ``None`` if the device is not open.
        """
        if not self.connected:
            return None
        # read bytes from SpaceMouse
        ret = self.device.read(self.__bytes_to_read)
        # test for nonblocking read
        if ret:
            self.process(ret)
        return self.tuple_state

    def process(self, data: Sequence[int]) -> None:
        """
        Update the device state from one HID report and dispatch callbacks.

        Parameters
        ----------
        data : Sequence[int]
            Raw HID report bytes for a single input event.

        Returns
        -------
        None
        """
        button_changed = False
        dof_changed = False

        for name, (chan, b1, b2, flip) in self.__mappings.items():
            if data[0] == chan:
                dof_changed = True
                # check if b1 or b2 is over the length of the data
                if b1 < len(data) and b2 < len(data):
                    self.dict_state[name] = flip * to_int16(data[b1], data[b2]) / float(self.axis_scale)

        for button_index, (chan, byte, bit) in enumerate(self.button_mapping):
            if data[0] == chan:
                button_changed = True
                # update the button vector
                mask = 1 << bit
                self.dict_state["buttons"][button_index] = 1 if (data[byte] & mask) != 0 else 0
        self.dof_changed = dof_changed
        self.buttons_changed = button_changed

        self.dict_state["t"] = high_acc_clock()

        # must receive both parts of the 6DOF state before we return the state dictionary
        if len(self.dict_state) == 8:
            self.tuple_state = SpaceNavigator(**self.dict_state)

        # call any attached callbacks
        if self.callback:
            self.callback(self.tuple_state)

        # only call the DOF callback if the DOF state actually changed
        if self.dof_callback and dof_changed:
            self.dof_callback(self.tuple_state)

        # only call the DoF callback_arr if the specific DoF state actually changed
        if self.dof_callback_arr and dof_changed:
            # foreach all callbacks (ButtonCallback)
            for block_dof_callback in self.dof_callback_arr:
                now = high_acc_clock()
                axis_name = block_dof_callback.axis
                if now >= self.dict_state_last[axis_name] + block_dof_callback.sleep:
                    axis_val = self.dict_state[axis_name]
                    # is minus callback defined
                    if block_dof_callback.callback_minus:
                        # is axis value greater than filter
                        if axis_val > block_dof_callback.filter:
                            block_dof_callback.callback(self.tuple_state, axis_val)
                        elif axis_val < -block_dof_callback.filter:
                            block_dof_callback.callback_minus(self.tuple_state, axis_val)
                    elif axis_val > block_dof_callback.filter or axis_val < -block_dof_callback.filter:
                        block_dof_callback.callback(self.tuple_state, axis_val)
                    self.dict_state_last[axis_name] = now

        # only call the button callback if the button state actually changed
        if self.button_callback and button_changed:
            self.button_callback(self.tuple_state, self.tuple_state.buttons)

        # only call the button callback_arr if the specific button state actually changed
        if self.button_callback_arr and button_changed:
            # foreach all callbacks (ButtonCallback)
            for block_button_callback in self.button_callback_arr:
                run = True
                # are buttons list
                if type(block_button_callback.buttons) is list:
                    for button_id in block_button_callback.buttons:
                        if not self.tuple_state.buttons[button_id]:
                            run = False

                # is one button
                elif isinstance(block_button_callback.buttons, int):
                    if not self.tuple_state.buttons[block_button_callback.buttons]:
                        run = False
                # call callback
                if run:
                    block_button_callback.callback(self.tuple_state, self.tuple_state.buttons, block_button_callback.buttons)

    def config_set(self, config: "Config") -> None:
        """
        Apply a validated callback configuration to the device.

        Parameters
        ----------
        config : Config
            Callback configuration bundle to apply.
        """

        self.callback = config.callback
        self.dof_callback = config.dof_callback
        self.dof_callback_arr = config.dof_callback_arr
        self.button_callback = config.button_callback
        self.button_callback_arr = config.button_callback_arr

    def config_set_sep(
        self,
        callback: Optional[Callable[["SpaceNavigator"], None]] = None,
        dof_callback: Optional[Callable[["SpaceNavigator"], None]] = None,
        dof_callback_arr: Optional[List[DofCallback]] = None,
        button_callback: Optional[Callable[["SpaceNavigator", ButtonState], None]] = None,
        button_callback_arr: Optional[List[ButtonCallback]] = None,
    ) -> None:
        """
        Set and validate device callback configuration from separate arguments.

        Parameters
        ----------
        callback : Callable[[SpaceNavigator], None], optional
            Generic event callback.
        dof_callback : Callable[[SpaceNavigator], None], optional
            Callback invoked when DoF state changes.
        dof_callback_arr : List[DofCallback], optional
            Per-axis callback configurations.
        button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
            Callback invoked when button state changes.
        button_callback_arr : List[ButtonCallback], optional
            Button callback configurations.
        """

        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)

        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr

    def config_remove(self) -> None:
        """
        Remove all active callback configuration from the device.

        Returns
        -------
        None
        """

        self.callback = None
        self.dof_callback = None
        self.dof_callback_arr = None
        self.button_callback = None
        self.button_callback_arr = None


# the IDs for the supported devices
# Each ID maps a device name to a DeviceSpec object
device_specs = {
    "SpaceMouse Enterprise": DeviceSpec(
        name="SpaceMouse Enterprise",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC633],
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            # ButtonSpec(channel=3, byte=5, bit=0),
            # ButtonSpec(channel=3, byte=5, bit=1),
            # ButtonSpec(channel=3, byte=5, bit=2),
            # ButtonSpec(channel=3, byte=5, bit=3),
            # ButtonSpec(channel=3, byte=5, bit=4),
            # ButtonSpec(channel=3, byte=5, bit=5),
            # ButtonSpec(channel=3, byte=5, bit=6),
            # ButtonSpec(channel=3, byte=5, bit=7),
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=3, bit=0),  # 5
            ButtonSpec(channel=3, byte=3, bit=1),  # 6
            ButtonSpec(channel=3, byte=3, bit=2),  # 7
            ButtonSpec(channel=3, byte=3, bit=3),  # 8
            ButtonSpec(channel=3, byte=3, bit=4),  # 9
            ButtonSpec(channel=3, byte=3, bit=5),  # 10
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
            ButtonSpec(channel=3, byte=1, bit=2),  # T IN SQUARE
            ButtonSpec(channel=3, byte=1, bit=4),  # R IN SQUARE
            ButtonSpec(channel=3, byte=1, bit=5),  # F IN SQUARE
            ButtonSpec(channel=3, byte=2, bit=0),  # SQUARE WITH ROTATING ARROWS
            ButtonSpec(channel=3, byte=2, bit=2),  # ISO1
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=2),  # LOCK
        ],
        axis_scale=350.0,
    ),
    "SpaceExplorer": DeviceSpec(
        name="SpaceExplorer",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC627],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=2, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=1, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=1, bit=7),  # ALT
            ButtonSpec(channel=3, byte=1, bit=0),  # 1
            ButtonSpec(channel=3, byte=1, bit=1),  # 2
            ButtonSpec(channel=3, byte=2, bit=3),  # PANEL
            ButtonSpec(channel=3, byte=2, bit=2),  # FIT
            ButtonSpec(channel=3, byte=2, bit=5),  # -
            ButtonSpec(channel=3, byte=2, bit=4),  # +
            ButtonSpec(channel=3, byte=1, bit=2),  # T
            ButtonSpec(channel=3, byte=1, bit=3),  # L
            ButtonSpec(channel=3, byte=1, bit=5),  # F
            ButtonSpec(channel=3, byte=2, bit=6),  # 2D
            ButtonSpec(channel=3, byte=1, bit=4),  # R
        ],
        axis_scale=350.0,
    ),
    "SpaceNavigator": DeviceSpec(
        name="SpaceNavigator",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC626],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse USB": DeviceSpec(
        name="SpaceMouseUSB",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC641],
        # LED HID usage code pair
        led_id=None,
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=None, byte=None, bit=None),  # No buttons
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Compact": DeviceSpec(
        name="SpaceMouse Compact",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC635],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Pro Wireless": DeviceSpec(
        name="SpaceMouse Pro Wireless",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC632],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Pro": DeviceSpec(
        name="SpaceMouse Pro",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC62B],
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Wireless": DeviceSpec(
        name="SpaceMouse Wireless",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC62E],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],  # FIT
        axis_scale=350.0,
    ),
    "SpaceMouse Wireless [NEW]": DeviceSpec(
        name="SpaceMouse Wireless [NEW]",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC63A],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],  # FIT
        axis_scale=350.0,
    ),
    "3Dconnexion Universal Receiver": DeviceSpec(
        name="3Dconnexion Universal Receiver",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC652],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpacePilot": DeviceSpec(
        name="SpacePilot",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC625],
        # LED HID usage code pair
        led_id=None,
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # 1
            ButtonSpec(channel=3, byte=1, bit=1),  # 2
            ButtonSpec(channel=3, byte=1, bit=2),  # 3
            ButtonSpec(channel=3, byte=1, bit=3),  # 4
            ButtonSpec(channel=3, byte=1, bit=4),  # 5
            ButtonSpec(channel=3, byte=1, bit=5),  # 6
            ButtonSpec(channel=3, byte=1, bit=6),  # T
            ButtonSpec(channel=3, byte=1, bit=7),  # L
            ButtonSpec(channel=3, byte=2, bit=0),  # R
            ButtonSpec(channel=3, byte=2, bit=1),  # F
            ButtonSpec(channel=3, byte=2, bit=2),  # Esc
            ButtonSpec(channel=3, byte=2, bit=3),  # Alt
            ButtonSpec(channel=3, byte=2, bit=4),  # Shift
            ButtonSpec(channel=3, byte=2, bit=5),  # Ctrl
            ButtonSpec(channel=3, byte=2, bit=6),  # Fit
            ButtonSpec(channel=3, byte=2, bit=7),  # Panel
            ButtonSpec(channel=3, byte=3, bit=0),  # Zoom -
            ButtonSpec(channel=3, byte=3, bit=1),  # Zoom +
            ButtonSpec(channel=3, byte=3, bit=2),  # Dom
            ButtonSpec(channel=3, byte=3, bit=3),  # 3D Lock
            ButtonSpec(channel=3, byte=3, bit=4),  # Config
        ],
        axis_scale=350.0,
    ),
    "SpacePilot Pro": DeviceSpec(
        name="SpacePilot Pro",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC629],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=3, bit=1),  # 1
            ButtonSpec(channel=3, byte=3, bit=2),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=3, bit=0),  # 5
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=4, bit=6),  # -
            ButtonSpec(channel=3, byte=4, bit=5),  # +
            ButtonSpec(channel=3, byte=4, bit=4),  # DOMINANT
            ButtonSpec(channel=3, byte=4, bit=3),  # PAN/ZOOM
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=2, bit=2),  # ISO
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
}

# [For the SpaceNavigator]
# The HID data is in the format
# [id, a, b, c, d, e, f]
# each pair (a,b), (c,d), (e,f) is a 16 bit signed value representing the absolute device state [from -350 to 350]

# if id==1, then the mapping is
# (a,b) = y translation
# (c,d) = x translation
# (e,f) = z translation

# if id==2 then the mapping is
# (a,b) = x tilting (roll)
# (c,d) = y tilting (pitch)
# (d,e) = z tilting (yaw)

# if id==3 then the mapping is
# a = button. Bit 1 = button 1, bit 2 = button 2

# Each movement of the device always causes two HID events, one
# with id 1 and one with id 2, to be generated, one after the other.


supported_devices = list(device_specs.keys())
_active_device = None


def close() -> None:
    """
    Close the module-level active device, if one is open.

    Returns
    -------
    None
    """
    if _active_device is not None:
        _active_device.close()


def read() -> Optional["SpaceNavigator"]:
    """
    Return the current state of the active navigation controller.

    Returns
    -------
    Optional[SpaceNavigator]
        Current SpaceNavigator state with ``{t, x, y, z, pitch, yaw, roll, button}``,
        or ``None`` if no device is open.
    """
    return _active_device.read() if _active_device is not None else None


def list_devices() -> List[str]:
    """
    Return the names of connected supported devices.

    Returns
    -------
    List[str]
        Names of connected devices known to this module.
    """
    devices = []
    try:
        hid = Enumeration()
    except AttributeError as e:
        raise Exception("HID API is probably not installed. " "Look at https://spacemouse.kubaandrysek.cz for details.") from e

    all_hids = hid.find()

    if all_hids:
        for device in all_hids:
            devices.extend(device_name for device_name, spec in device_specs.items() if (device.vendor_id == spec.hid_id[0] and device.product_id == spec.hid_id[1]))
    return devices


def list_available_devices() -> List[Tuple[str, int, int]]:
    """
    Return all supported device definitions known to the module.

    Returns
    -------
    List[Tuple[str, int, int]]
        Tuples of ``(device_name, vendor_id, product_id)``.
    """
    return [(device_name, spec.hid_id[0], spec.hid_id[1]) for device_name, spec in device_specs.items()]


def list_all_hid_devices() -> List[Tuple[str, str, int, int]]:
    """
    Return all HID devices detected by the backend.

    Returns
    -------
    List[Tuple[str, str, int, int]]
        Tuples of ``(product_string, manufacturer_string, vendor_id, product_id)``.
    """
    try:
        hid = Enumeration()
    except AttributeError as e:
        raise Exception("HID API is probably not installed." "Look at https://spacemouse.kubaandrysek.cz for details.") from e

    return [(device.product_string, device.manufacturer_string, device.vendor_id, device.product_id) for device in hid.find()]


def openCfg(config: "Config", set_nonblocking_loop: bool = True, device: Optional[str] = None, DeviceNumber: int = 0) -> Optional["DeviceSpec"]:
    """
    Open a device using a prebuilt :class:`Config` instance.

    Parameters
    ----------
    config : Config
        Validated callback configuration bundle.
    set_nonblocking_loop : bool, optional
        If ``True``, configure the HID device for non-blocking reads.
    device : str, optional
        Preferred device name. If omitted, the first supported connected device
        is used.
    DeviceNumber : int, optional
        Index of the matching device to open when multiple devices of the same
        type are present.

    Returns
    -------
    Optional[DeviceSpec]
        Opened device wrapper, or ``None`` if no supported device was opened.
    """

    return open(config.callback, config.dof_callback, config.dof_callback_arr, config.button_callback, config.button_callback_arr, set_nonblocking_loop, device=device, DeviceNumber=DeviceNumber)


def open(
    callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback_arr: Optional[List[DofCallback]] = None,
    button_callback: Optional[Callable[["SpaceNavigator", ButtonState], None]] = None,
    button_callback_arr: Optional[List[ButtonCallback]] = None,
    set_nonblocking_loop: bool = True,
    device: Optional[str] = None,
    path: Optional[str] = None,
    DeviceNumber: int = 0,
) -> Optional["DeviceSpec"]:
    """
    Open a supported 3Dconnexion device and make it the active module device.

    The returned device becomes the target of the module-level `read()` and
    `close()` helpers. When working with multiple devices, use the returned
    :class:`DeviceSpec` instance directly instead of the module-level helpers.

    Parameters
    ----------
    callback : Callable[[SpaceNavigator], None], optional
        Generic event callback invoked after processing an input report.
    dof_callback : Callable[[SpaceNavigator], None], optional
        Callback invoked when any DoF state changes.
    dof_callback_arr : List[DofCallback], optional
        Per-axis callback configurations.
    button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
        Callback invoked when button state changes.
    button_callback_arr : List[ButtonCallback], optional
        Button callback configurations.
    set_nonblocking_loop : bool, optional
        If ``True``, configure the HID device for non-blocking reads.
    device : str, optional
        Device name to open. If omitted, the first connected supported device is
        selected.
    path : str, optional
        Optional HID path override.
    DeviceNumber : int, optional
        Index of the matching device to open when multiple devices of the same
        type are present.

    Returns
    -------
    Optional[DeviceSpec]
        Opened device wrapper, or ``None`` if no supported device was opened.
    """
    # only used if the module-level functions are used
    global _active_device

    # if no device name specified, look for any matching device and choose the first
    if device is None:
        all_devices = list_devices()
        if len(all_devices) > 0:
            device = all_devices[0]
        else:
            raise Exception("No found any connected or supported devices.")

    found_devices = []
    hid = Enumeration()
    all_hids = hid.find()
    if all_hids:
        for dev in all_hids:
            if path:
                dev.path = path
            spec = device_specs[device]
            if dev.vendor_id == spec.hid_id[0] and dev.product_id == spec.hid_id[1]:
                found_devices.append({"Spec": spec, "HIDDevice": dev})
                print(f"{device} found")

    else:
        print("No HID devices detected")
        return None

    if not found_devices:
        print("No supported devices found")
        return None
    else:
        if len(found_devices) <= DeviceNumber:
            DeviceNumber = 0

        if len(found_devices) > DeviceNumber:
            # Check that the input configuration has the correct components
            # Raise an exception if it encounters incorrect component.
            check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
            # create a copy of the device specification
            spec = found_devices[DeviceNumber]["Spec"]
            dev = found_devices[DeviceNumber]["HIDDevice"]
            new_device = copy.deepcopy(spec)
            new_device.device = dev

            # set the callbacks
            new_device.callback = callback
            new_device.dof_callback = dof_callback
            new_device.dof_callback_arr = dof_callback_arr
            new_device.button_callback = button_callback
            new_device.button_callback_arr = button_callback_arr
            # open the device
            new_device.open()
            # set nonblocking/blocking mode
            new_device.set_nonblocking_loop = set_nonblocking_loop
            dev.set_nonblocking(set_nonblocking_loop)

            _active_device = new_device
            return new_device

    print("Unknown error occured.")
    return None


def check_config(
    callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback_arr: Optional[List[DofCallback]] = None,
    button_callback: Optional[Callable[["SpaceNavigator", ButtonState], None]] = None,
    button_callback_arr: Optional[List[ButtonCallback]] = None,
) -> None:
    """
    Validate a callback configuration.

    Parameters
    ----------
    callback : Callable[[SpaceNavigator], None], optional
        Generic event callback.
    dof_callback : Callable[[SpaceNavigator], None], optional
        DoF state-change callback.
    dof_callback_arr : List[DofCallback], optional
        Per-axis callback configurations.
    button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
        Button state-change callback.
    button_callback_arr : List[ButtonCallback], optional
        Button callback configurations.

    Raises
    ------
    Exception
        If any callback configuration entry is invalid.
    """
    if dof_callback_arr and check_dof_callback_arr(dof_callback_arr):
        pass
    if button_callback_arr and check_button_callback_arr(button_callback_arr):
        pass


def check_button_callback_arr(button_callback_arr: List[ButtonCallback]) -> List[ButtonCallback]:
    """
    Validate a list of button callback configurations.

    Parameters
    ----------
    button_callback_arr : List[ButtonCallback]
        Callback configuration objects to validate.

    Returns
    -------
    List[ButtonCallback]
        The validated callback list.

    Raises
    ------
    Exception
        If any entry has an invalid type or callback signature.
    """

    # foreach ButtonCallback
    for num, butt_call in enumerate(button_callback_arr):
        if not isinstance(butt_call, ButtonCallback):
            raise Exception(f"'ButtonCallback[{num}]' is not instance of 'ButtonCallback'")
        if type(butt_call.buttons) is int:
            pass
        elif type(butt_call.buttons) is list:
            for xnum, butt in enumerate(butt_call.buttons):
                if type(butt) is not int:
                    raise Exception(f"'ButtonCallback[{num}]:buttons[{xnum}]' is not type int or list of int")
        else:
            raise Exception(f"'ButtonCallback[{num}]:buttons' is not type int or list of int")
        if not callable(butt_call.callback):
            raise Exception(f"'ButtonCallback[{num}]:callback' is not callable")
    return button_callback_arr


def check_dof_callback_arr(dof_callback_arr: List[DofCallback]) -> List[DofCallback]:
    """
    Validate a list of DoF callback configurations.

    Parameters
    ----------
    dof_callback_arr : List[DofCallback]
        Callback configuration objects to validate.

    Returns
    -------
    List[DofCallback]
        The validated callback list.

    Raises
    ------
    Exception
        If any entry has an invalid axis, type, or callback configuration.
    """

    # foreach DofCallback
    for num, dof_call in enumerate(dof_callback_arr):
        if not isinstance(dof_call, DofCallback):
            raise Exception(f"'DofCallback[{num}]' is not instance of 'DofCallback'")
            # has the correct axis name
        if dof_call.axis not in ["x", "y", "z", "roll", "pitch", "yaw"]:
            raise Exception(f"'DofCallback[{num}]:axis' is not string from ['x', 'y', 'z', 'roll', 'pitch', 'yaw']")

            # is callback callable
        if not callable(dof_call.callback):
            raise Exception(f"'DofCallback[{num}]:callback' is not callable")

            # is sleep type float
        if type(dof_call.sleep) is not float:
            raise Exception(f"'DofCallback[{num}]:sleep' is not type float")

            # is callback_minus callable
        if dof_call.callback_minus is not None and not callable(dof_call.callback_minus):
            raise Exception(f"'DofCallback[{num}]:callback_minus' is not callable")

            # is filter type float
        if type(dof_call.filter) is not float:
            raise Exception(f"'DofCallback[{num}]:filter' is not type float")
    return dof_callback_arr


def config_set(config: "Config") -> None:
    """
    Apply a configuration bundle to the active module-level device.

    Parameters
    ----------
    config : Config
        Validated callback configuration bundle.
    """

    if _active_device is not None:
        _active_device.config_set(config)


def config_set_sep(
    callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback: Optional[Callable[["SpaceNavigator"], None]] = None,
    dof_callback_arr: Optional[List[DofCallback]] = None,
    button_callback: Optional[Callable[["SpaceNavigator", ButtonState], None]] = None,
    button_callback_arr: Optional[List[ButtonCallback]] = None,
) -> None:
    """
    Apply callback configuration to the active device from separate arguments.

    Parameters
    ----------
    callback : Callable[[SpaceNavigator], None], optional
        Generic event callback.
    dof_callback : Callable[[SpaceNavigator], None], optional
        DoF state-change callback.
    dof_callback_arr : List[DofCallback], optional
        Per-axis callback configurations.
    button_callback : Callable[[SpaceNavigator, ButtonState], None], optional
        Button state-change callback.
    button_callback_arr : List[ButtonCallback], optional
        Button callback configurations.
    """

    if _active_device is not None:
        _active_device.config_set_sep(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)


def config_remove() -> None:
    """
    Remove all callback configuration from the active module-level device.

    Returns
    -------
    None
    """

    if _active_device is not None:
        _active_device.config_remove()


def print_state(state: Optional["SpaceNavigator"]) -> None:
    """
    Print the decoded DoF state to standard output.

    Parameters
    ----------
    state : SpaceNavigator, optional
        Decoded SpaceMouse state to print.
    """
    if state:
        print(" ".join(["%4s %+.2f" % (k, getattr(state, k)) for k in ["x", "y", "z", "roll", "pitch", "yaw", "t"]]))


def silent_callback(state: Optional["SpaceNavigator"]) -> None:
    """
    Ignore the provided state.

    Parameters
    ----------
    state : SpaceNavigator, optional
        Decoded SpaceMouse state. The value is ignored.
    """
    pass


def print_buttons(state: "SpaceNavigator", buttons: ButtonState) -> None:
    """
    Print the decoded button state to standard output.

    Parameters
    ----------
    state : SpaceNavigator
        Current decoded SpaceMouse state.
    buttons : ButtonState
        Button-state vector to print.
    """
    # simple default button callback
    print("[", " ".join(["%2d, " % buttons[k] for k in range(len(buttons))]), "]")


# def toggle_led(state, buttons):
#     print("".join(["buttons=", str(buttons)]))
#     # Switch on the led on left push, off on right push
#     if buttons[0] == 1:
#         set_led(1)
#     if buttons[1] == 1:
#         set_led(0)


# def set_led(state):
#     if _active_device:
#         _active_device.set_led(state)


class spacenavigator(object):
    """
    Convenience wrapper that maintains the latest SpaceMouse state in a thread.

    The wrapper opens a device, polls it continuously in the background, and
    exposes the latest motion vector and button state through simple properties.
    """

    def __init__(self, device: Optional[str] = None, path: Optional[str] = None, DeviceNumber: int = 0, **kwargs: Any) -> None:
        """
        Open a SpaceMouse device and start the background polling thread.

        Parameters
        ----------
        device : str, optional
            Preferred device name. If omitted, the first supported connected
            device is used.
        path : str, optional
            Optional HID path override.
        DeviceNumber : int, optional
            Index of the matching device to open.
        **kwargs : Any
            Additional keyword arguments forwarded to :func:`open`.

        Raises
        ------
        RuntimeError
            If no supported device could be opened.
        """
        self.device = open(device=device, path=path, DeviceNumber=DeviceNumber, **kwargs)
        if self.device is None:
            raise RuntimeError("Failed to open SpaceMouse device")
        self._x = np.zeros(6, dtype=np.float32)  # Initialize pose displacement
        self._buttons = [0, 0]  # Initialize button states
        self._t0 = high_acc_clock()  # Record the start time
        self._last_update_state: float = -100  # Last state update timestamp
        self._last_update_buttons: float = -100  # Last buttons update timestamp
        self._stop_event = threading.Event()  # Used to stop the thread
        self._state_thread = threading.Thread(target=self._update_state_in_background, daemon=True)
        sleep(1)
        self._state_thread.start()  # Start the background thread

    def _update_state_in_background(self) -> None:
        """
        Poll the device continuously in the background thread.

        Returns
        -------
        None
        """
        while not self._stop_event.is_set():  # The thread will run until stop_event is set
            self.GetState()
            sleep(0.001)  # Sleep to prevent excessive CPU usage

    @property
    def t(self) -> float:
        """
        Get elapsed time since the last reset.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        self.GetState()
        return self._t - self._t0

    @property
    def x(self) -> np.ndarray:
        """
        Get the latest decoded 6-DoF motion vector.

        Returns
        -------
        np.ndarray
            Vector ``[x, y, z, roll, pitch, yaw]``.
        """
        self.GetState()
        return self._x

    @property
    def buttons(self) -> List[int]:
        """
        Get the latest decoded button states.

        Returns
        -------
        List[int]
            Current button-state vector.
        """
        self.GetState()
        return self._buttons

    def GetState(self) -> None:
        """
        Poll the device once and update cached motion and button state.

        Returns
        -------
        None
        """
        self._state = self.device.read()
        self._t = high_acc_clock()
        if self.device.dof_changed:
            self._x = np.array([getattr(self._state, k) for k in ["x", "y", "z", "roll", "pitch", "yaw"]])
            self._last_update_state = self._t
            self.device.dof_changed = False
        if self.device.buttons_changed:
            self._buttons = getattr(self._state, "buttons")
            self._last_update_buttons = self._t
            self.device.buttons_changed = False

    def Close(self) -> None:
        """
        Stop the background polling thread and close the device.

        Returns
        -------
        None
        """
        self._stop_event.set()  # Signal the thread to stop
        self.device.close()

    def ResetTime(self) -> None:
        """
        Reset the wrapper time origin used by :attr:`t`.

        Returns
        -------
        None
        """
        self._t0 = high_acc_clock()


if __name__ == "__main__":

    np.set_printoptions(formatter={"float": "{: 0.4f}".format})
    ginp = spacenavigator()

    t0 = -1
    while True:
        # ginp.GetState()
        if ginp._last_update_state > t0 or ginp._last_update_buttons > t0:
            print("Status: ", ginp.x, ginp.buttons)
            t0 = ginp._t
        sleep(0.01)
        if ginp.t > 60:
            ginp._stop_event.set()
            break
