"""Cytation camera driver using Aravis (GenICam/USB3 Vision).

Standalone: no PLR dependencies except numpy. ``CytationMicroscope`` delegates camera
operations here; Aravis (via PyGObject) talks to the camera over USB3 Vision/GenICam.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

try:
  import numpy as np
except ImportError:
  np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Aravis is an optional dependency. It requires:
#   - System library: brew install aravis (macOS) or apt install libaravis-dev (Linux)
#   - Python bindings: pip install PyGObject
# If not installed, AravisCamera.setup() will raise ImportError with instructions.
try:
  # PyGObject ships no stubs, and it is absent entirely wherever the
  # cytation-microscopy extra is not installed, so both codes can fire.
  import gi  # type: ignore[import-untyped,import-not-found]

  gi.require_version("Aravis", "0.8")
  from gi.repository import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    Aravis,
  )

  HAS_ARAVIS = True
except (ImportError, ValueError):
  HAS_ARAVIS = False
  Aravis = None  # type: ignore[assignment]

# Number of pre-allocated buffers for the Aravis stream.
# For single-frame software-triggered capture, 5 is more than sufficient.
_BUFFER_COUNT = 5


@dataclass
class CameraInfo:
  """Discovery result for a GenICam-compatible camera.

  Returned by AravisCamera.enumerate_cameras() and get_device_info().
  Contains identification and connection metadata — no hardware access needed
  after discovery.
  """

  serial_number: str
  model_name: str
  vendor: str
  firmware_version: str
  connection_type: str  # "USB3" for the Cytation 5 BlackFly


class AravisCamera:
  """Cytation camera access through the vendor-independent USB3 Vision protocol.

  This class wraps all Aravis/GenICam operations for single-frame image
  acquisition with software triggering.

  GenICam primer for non-camera-experts:
    GenICam is a standard that defines how camera features (exposure, gain,
    trigger, etc.) are named and accessed. Every compliant camera publishes
    an XML file describing its features as "nodes." Aravis reads this XML
    and lets you get/set node values by name (e.g., "ExposureTime", "Gain").
    Aravis provides a Python API to access these nodes via PyGObject.

  Buffer management:
    Aravis uses a producer-consumer model with pre-allocated buffers.
    During setup(), we allocate a small pool of buffers and push them to
    the stream. When we trigger a capture, Aravis fills a buffer with image
    data. We pop the buffer, copy the data to a numpy array, and push the
    buffer back to the pool for reuse. The copy is necessary because the
    buffer memory is owned by Aravis and will be reused.

  Usage:
    camera = AravisCamera()
    await camera.setup("12345678")
    await camera.set_exposure(10.0)  # 10 ms
    image = await camera.trigger()    # numpy array, camera's monochrome format
    await camera.stop()
  """

  def __init__(self) -> None:
    self._camera: Optional[Any] = None  # Aravis.Camera
    self._device: Optional[Any] = None  # Aravis.Device
    self._stream: Optional[Any] = None  # Aravis.Stream
    self._serial_number: Optional[str] = None
    self._acquiring: bool = False
    self._width: int = 0
    self._height: int = 0
    self._payload_size: int = 0
    self._capture_lock = asyncio.Lock()

  @property
  def width(self) -> int:
    """Image width in pixels (read-only, from camera default)."""
    return self._width

  @property
  def height(self) -> int:
    """Image height in pixels (read-only, from camera default)."""
    return self._height

  @property
  def camera(self) -> Any:
    """The ``Aravis.Camera`` bound at setup. Raises before setup()."""
    if self._camera is None:
      raise RuntimeError("Camera is not initialized. Call setup() first.")
    return self._camera

  @property
  def device(self) -> Any:
    """The ``Aravis.Device`` bound at setup. Raises before setup()."""
    if self._device is None:
      raise RuntimeError("Camera is not initialized. Call setup() first.")
    return self._device

  @property
  def stream(self) -> Any:
    """The ``Aravis.Stream`` allocated at setup. Raises before setup()."""
    if self._stream is None:
      raise RuntimeError("Camera is not initialized. Call setup() first.")
    return self._stream

  async def setup(self, serial_number: Optional[str] = None) -> None:
    """Connect to camera, configure software trigger, allocate buffers.

    The software trigger configuration:
    TriggerSelector=FrameStart, TriggerSource=Software, TriggerMode=On.

    Args:
        serial_number: Camera serial number (e.g., "12345678"). If None, uses
            the first available camera. Use enumerate_cameras() to discover
            available cameras.

    Raises:
        ImportError: If Aravis/PyGObject is not installed.
        RuntimeError: If camera cannot be found or connected.
    """
    if not HAS_ARAVIS:
      raise ImportError(
        "Aravis is not installed. Install it with:\n"
        "  macOS:   brew install aravis && pip install PyGObject\n"
        "  Linux:   sudo apt-get install libaravis-dev gobject-introspection "
        "&& pip install PyGObject\n"
        "  Windows: pacman -S mingw-w64-x86_64-aravis (in MSYS2)"
      )

    self._serial_number = serial_number

    Aravis.update_device_list()
    n_devices = Aravis.get_n_devices()

    if n_devices == 0:
      raise RuntimeError("No cameras found.")

    device_id_to_connect = None
    if serial_number is None:
      # Use the first available camera.
      device_id_to_connect = Aravis.get_device_id(0)
    else:
      for i in range(n_devices):
        dev_serial = Aravis.get_device_serial_nbr(i) or ""
        dev_id = Aravis.get_device_id(i) or ""
        if serial_number in (dev_serial, dev_id):
          device_id_to_connect = dev_id
          break

    if device_id_to_connect is None:
      raise RuntimeError(
        f"Camera with serial '{serial_number}' not found. "
        f"Available devices: {[Aravis.get_device_id(i) for i in range(n_devices)]}"
      )

    try:
      self._camera = Aravis.Camera.new(device_id_to_connect)
    except Exception as e:
      raise RuntimeError(
        f"Failed to connect to camera '{device_id_to_connect}'. "
        f"Is the camera in use by another process? Error: {e}"
      ) from e

    self._device = self.camera.get_device()

    # Aravis disables other trigger selectors and configures the appropriate edge
    # and frame-rate settings for the camera model.
    self.camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
    self.camera.set_trigger("Software")

    # Read image dimensions and payload size from camera.
    self._width = self.camera.get_region()[2]  # x, y, width, height
    self._height = self.camera.get_region()[3]
    self._payload_size = self.camera.get_payload()

    # BlackFly/Flea3 cameras need a delay after trigger mode change.
    await asyncio.sleep(1)

    # Create stream and pre-allocate buffer pool.
    # Aravis requires buffers to be pushed to the stream before acquisition.
    # We allocate a small pool (5 buffers) — for single-frame software
    # trigger, we only use one at a time but the pool prevents starvation.
    self._stream = self.camera.create_stream(None, None)
    for _ in range(_BUFFER_COUNT):
      self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload_size))

    logger.info(
      "AravisCamera: Connected to %s (SN: %s), %dx%d",
      self.device.get_string_feature_value("DeviceModelName"),
      serial_number,
      self._width,
      self._height,
    )

  def start_acquisition(self) -> None:
    """Begin camera acquisition (no-op if already acquiring).

    After this call, the
    camera is ready to receive software triggers and produce image buffers.
    """
    camera = self.camera
    if self._acquiring:
      return
    camera.start_acquisition()
    self._acquiring = True

  def stop_acquisition(self) -> None:
    """End camera acquisition (no-op if not acquiring).

    Stop camera acquisition.
    """
    camera = self.camera
    if not self._acquiring:
      return
    camera.stop_acquisition()
    self._acquiring = False

  async def trigger(self, timeout_ms: int = 5000) -> np.ndarray:
    """Capture a single frame: software trigger → grab.

    Acquisition must already be started via start_acquisition().
    Start/stop acquisition bracket the capture loop, not each individual frame.

    Args:
        timeout_ms: Maximum time to wait for image buffer, in milliseconds.
            Default 5000 (5 seconds).

    Returns:
        numpy.ndarray: Image as a 2D array (height × width). Mono8 returns uint8;
            unpacked Mono10, Mono12, and Mono16 return uint16.

    Raises:
        RuntimeError: If the camera is not acquiring or a frame is incomplete.
        TimeoutError: If no frame arrives before the deadline.
    """
    _ = self.camera
    if not self._acquiring:
      raise RuntimeError("Camera is not acquiring. Call start_acquisition() first.")

    if timeout_ms <= 0:
      raise ValueError("timeout_ms must be positive")
    async with self._capture_lock:
      stream = self.stream
      self.camera.software_trigger()
      pending = asyncio.create_task(asyncio.to_thread(stream.timeout_pop_buffer, timeout_ms * 1000))
      try:
        buffer = await asyncio.shield(pending)
      except asyncio.CancelledError:
        buffer = await pending
        if buffer is not None:
          stream.push_buffer(buffer)
        raise
      if buffer is None:
        raise TimeoutError(f"Camera capture timed out after {timeout_ms}ms")
      try:
        if buffer.get_status() != Aravis.BufferStatus.SUCCESS:
          raise RuntimeError(f"Camera returned an incomplete frame: {buffer.get_status()}")
        pixel_format = buffer.get_image_pixel_format()
        # PFNC unpacked monochrome formats. Mono10/12 occupy 16 bits per pixel.
        formats = {
          0x01080001: "u1",
          0x01100003: "<u2",
          0x01100005: "<u2",
          0x01100007: "<u2",
        }
        if pixel_format not in formats:
          raise NotImplementedError(f"Unsupported camera pixel format: 0x{pixel_format:08x}")
        dtype = np.dtype(formats[pixel_format])
        width, height = buffer.get_image_width(), buffer.get_image_height()
        padding_x, _ = buffer.get_image_padding()
        stride = width * dtype.itemsize + padding_x
        data = buffer.get_image_data()
        if len(data) < height * stride:
          raise RuntimeError("Camera image payload is shorter than its dimensions")
        return np.ndarray(
          (height, width), dtype=dtype, buffer=data, strides=(stride, dtype.itemsize)
        ).copy()
      finally:
        stream.push_buffer(buffer)

  async def stop(self) -> None:
    """Release camera and free all Aravis resources.

    Safe to call at any point —
    stops acquisition if active, resets trigger mode, releases camera.
    """
    try:
      if self._acquiring and self._camera is not None:
        self.stop_acquisition()

      if self._device is not None:
        try:
          self._device.set_string_feature_value("TriggerMode", "Off")
        except Exception:
          pass  # Camera may already be disconnected
    finally:
      self._camera = None
      self._device = None
      self._stream = None
      self._serial_number = None
      self._acquiring = False
      self._width = 0
      self._height = 0
      self._payload_size = 0

  def get_pixel_max(self) -> int:
    """Return the maximum code value of the active unpacked monochrome format."""
    maxima = {0x01080001: 255, 0x01100003: 1023, 0x01100005: 4095, 0x01100007: 65535}
    pixel_format = self.camera.get_pixel_format()
    if pixel_format not in maxima:
      raise NotImplementedError(f"Unsupported camera pixel format: 0x{pixel_format:08x}")
    return maxima[pixel_format]

  def get_exposure_limits(self) -> tuple[float, float]:
    """Return the camera's allowed exposure interval in milliseconds."""
    low, high = self.camera.get_exposure_time_bounds()
    return float(low) / 1000, float(high) / 1000

  def get_gain_limits(self) -> tuple[float, float]:
    """Return the camera's allowed analog gain interval in dB."""
    low, high = self.camera.get_gain_bounds()
    return float(low), float(high)

  async def set_exposure(self, exposure_ms: float) -> None:
    """Set exposure time in milliseconds.

    Disables auto-exposure first, then sets the GenICam ExposureTime node.
    GenICam uses microseconds internally; this method handles the conversion.

    Args:
        exposure_ms: Exposure time in milliseconds (e.g., 10.0 = 10 ms).
    """
    low, high = self.get_exposure_limits()
    if not math.isfinite(exposure_ms) or not low <= exposure_ms <= high:
      raise ValueError(f"Exposure must be between {low} and {high} ms")
    camera = self.camera
    # Disable auto-exposure before setting manual value.
    self.device.set_string_feature_value("ExposureAuto", "Off")
    # GenICam ExposureTime is in microseconds.
    exposure_us = exposure_ms * 1000.0
    camera.set_exposure_time(exposure_us)

  async def get_exposure(self) -> float:
    """Read current exposure time in milliseconds (from hardware, not cached)."""
    exposure_us = self.camera.get_exposure_time()
    return float(exposure_us) / 1000.0

  async def set_gain(self, gain: float) -> None:
    """Set gain value.

    Disables auto-gain first, then sets the GenICam Gain node.
    The gain range depends on the camera model (typically 0-30 for BlackFly).

    Args:
        gain: Gain value (e.g., 1.0).
    """
    low, high = self.get_gain_limits()
    if not math.isfinite(gain) or not low <= gain <= high:
      raise ValueError(f"Gain must be between {low} and {high} dB")
    camera = self.camera
    self.device.set_string_feature_value("GainAuto", "Off")
    camera.set_gain(gain)

  async def set_auto_gain(self, mode: str) -> None:
    """Set auto-gain mode.

    Args:
        mode: One of "off", "once", "continuous". Maps to GenICam
            GainAuto node values: Off, Once, Continuous.
    """
    _ = self.camera
    mode_map = {"off": "Off", "once": "Once", "continuous": "Continuous"}
    aravis_mode = mode_map.get(mode.lower())
    if aravis_mode is None:
      raise ValueError(f"Invalid auto-gain mode '{mode}'. Use 'off', 'once', or 'continuous'.")
    self.device.set_string_feature_value("GainAuto", aravis_mode)

  async def get_gain(self) -> float:
    """Read current gain value (from hardware, not cached)."""
    return float(self.camera.get_gain())

  async def set_auto_exposure(self, mode: str) -> None:
    """Set auto-exposure mode.

    Args:
        mode: One of "off", "once", "continuous". Maps to GenICam
            ExposureAuto node values: Off, Once, Continuous.
    """
    _ = self.camera
    mode_map = {"off": "Off", "once": "Once", "continuous": "Continuous"}
    aravis_mode = mode_map.get(mode.lower())
    if aravis_mode is None:
      raise ValueError(f"Invalid auto-exposure mode '{mode}'. Use 'off', 'once', or 'continuous'.")
    self.device.set_string_feature_value("ExposureAuto", aravis_mode)

  async def set_pixel_format(self, fmt: Optional[int] = None) -> None:
    """Set pixel format. Default is Mono8.

    Must be called before start_acquisition(). The format value is an
    Aravis pixel format constant (e.g., Aravis.PIXEL_FORMAT_MONO_8).

    Args:
        fmt: Aravis pixel format constant. If None, uses Mono8.
    """
    camera = self.camera
    if self._acquiring:
      raise RuntimeError("Stop acquisition before changing the pixel format")
    if fmt is None:
      if HAS_ARAVIS:
        fmt = Aravis.PIXEL_FORMAT_MONO_8
      else:
        return
    camera.set_pixel_format(fmt)
    self._width, self._height = camera.get_region()[2:]
    self._payload_size = camera.get_payload()
    self._stream = None
    self._stream = camera.create_stream(None, None)
    for _ in range(_BUFFER_COUNT):
      self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload_size))

  def get_device_info(self) -> CameraInfo:
    """Read camera identification from GenICam nodes.

    Returns model name, serial number, vendor, and firmware version
    without requiring acquisition to be active.

    Returns:
        CameraInfo with fields populated from the camera's GenICam XML.
    """
    device = self.device
    return CameraInfo(
      serial_number=device.get_string_feature_value("DeviceSerialNumber"),
      model_name=device.get_string_feature_value("DeviceModelName"),
      vendor=device.get_string_feature_value("DeviceVendorName"),
      firmware_version=device.get_string_feature_value("DeviceFirmwareVersion"),
      connection_type="USB3",
    )

  @staticmethod
  def enumerate_cameras() -> list[CameraInfo]:
    """List all connected GenICam-compatible cameras.

    Uses Aravis device enumeration — finds cameras across USB3 Vision
    and GigE Vision transports (though this driver targets USB3 only).

    Returns:
        List of CameraInfo for each detected camera. Empty list if none
        found (not an error — matches PLR's graceful enumeration pattern).

    Raises:
        ImportError: If Aravis/PyGObject is not installed.
    """
    if not HAS_ARAVIS:
      raise ImportError(
        "Aravis is not installed. Install it with:\n"
        "  macOS:   brew install aravis && pip install PyGObject\n"
        "  Linux:   sudo apt-get install libaravis-dev gobject-introspection "
        "&& pip install PyGObject\n"
        "  Windows: pacman -S mingw-w64-x86_64-aravis (in MSYS2)"
      )

    Aravis.update_device_list()
    n_devices = Aravis.get_n_devices()
    cameras: list[CameraInfo] = []

    for i in range(n_devices):
      # Parse info from device ID string without opening the camera.
      # Opening the camera here would lock the USB device and prevent
      # a subsequent setup() from connecting (GObject doesn't release
      # the USB handle reliably on del).
      device_id = Aravis.get_device_id(i)
      # device_id format varies: "USB3Vision-vendor-model-serial" or similar
      serial = Aravis.get_device_serial_nbr(i) or device_id
      model = Aravis.get_device_model(i) or "Unknown"
      vendor = Aravis.get_device_vendor(i) or "Unknown"
      protocol = Aravis.get_device_protocol(i) or "USB3"

      info = CameraInfo(
        serial_number=serial,
        model_name=model,
        vendor=vendor,
        firmware_version="",  # Not available without opening camera
        connection_type=protocol,
      )
      cameras.append(info)

    return cameras
