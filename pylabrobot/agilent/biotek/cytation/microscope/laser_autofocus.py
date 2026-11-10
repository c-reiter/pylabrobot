"""Reference-based Cytation laser autofocus using the USB3 Vision camera.

The laser reflects from the plate bottom. Calibrate at a visually focused sample
plane for each objective and plate type; subsequent corrections track that plane.
No instrument calibration or EEPROM values are modified.
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Optional, Tuple

from .models import Image, ImagingMode, Objective

if TYPE_CHECKING:
  from .microscope import CytationMicroscope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaserFocusCalibration:
  """Laser spot position and displacement per mm for a known sample focus."""

  objective: Objective
  imaging_mode: ImagingMode
  focal_position: float
  spot: Tuple[float, float]
  slope: Tuple[float, float]
  exposure: float
  current: float


def laser_spot_position(image: Image, pixel_max: int) -> Tuple[float, float]:
  """Locate an unsaturated reflection, rejecting dark frames and isolated hot pixels.

  Coordinates are camera pixels (x, y). A local intensity-weighted centroid avoids
  background noise biasing the measurement across the full sensor.
  """
  import numpy as np

  if image.ndim != 2 or image.size == 0:
    raise ValueError("Laser autofocus requires a nonempty monochrome image")
  data = image.astype(float)
  background = float(np.median(data))
  peak = float(data.max())
  if peak >= pixel_max * 0.999:
    raise RuntimeError("Laser reflection is saturated; reduce calibration exposure")
  if peak - background < max(5, pixel_max * 0.02):
    raise RuntimeError("No usable laser reflection detected")
  row, column = np.unravel_index(np.argmax(data), data.shape)
  y0, x0 = max(0, int(row) - 64), max(0, int(column) - 64)
  window = data[y0 : int(row) + 65, x0 : int(column) + 65] - background
  candidates = window >= (peak - background) * 0.3
  # Follow the connected reflection instead of clipping it to a peak-centered box.
  # Speckle can shift the brightest pixel without moving the reflection itself.
  # Separate reflections from the two plate surfaces must not share a centroid.
  selected = np.zeros_like(candidates)
  pending = [(int(row) - y0, int(column) - x0)]
  while pending:
    y, x = pending.pop()
    if not (0 <= y < window.shape[0] and 0 <= x < window.shape[1]):
      continue
    if selected[y, x] or not candidates[y, x]:
      continue
    selected[y, x] = True
    pending.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
  if np.count_nonzero(selected) < 9:
    raise RuntimeError("Laser reflection is too small to distinguish from hot pixels")
  if selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any():
    raise RuntimeError("Laser reflection is clipped or too large for reliable localization")
  weights = np.where(selected, window, 0)
  total = weights.sum()
  ys, xs = np.indices(weights.shape)
  return float(x0 + (xs * weights).sum() / total), float(y0 + (ys * weights).sum() / total)


class CytationLaserAutofocus:
  """Laser focus calibration and bounded correction for the installed laser cube."""

  def __init__(self, microscope: CytationMicroscope) -> None:
    self.microscope = microscope
    self.calibration: Optional[LaserFocusCalibration] = None

  @property
  def available(self) -> bool:
    """Whether discovery found a laser autofocus cube."""
    return self.microscope._laser_slot is not None

  @property
  def _slot(self) -> int:
    """Require a discovered laser cube before sending controller commands."""
    slot = self.microscope._laser_slot
    if slot is None:
      raise RuntimeError("No laser autofocus cube is configured")
    return slot

  async def _peripheral(self, message: str, response_length: int = 5) -> bytes:
    """Exchange a laser LED-controller frame through Cytation peripheral port 2."""
    parameters = f"02{len(message):03d}1{response_length:03d}{message}".encode("ascii")
    response = await self.microscope.driver._exchange(
      b"\x0c", parameters, data_length=response_length, timeout=20
    )
    if len(response) != response_length or response[:1] != b"\x06":
      raise ValueError(f"Malformed laser controller response: {response!r}")
    if response[-4:] != b"0000":
      raise RuntimeError(f"Laser controller returned status {response[-4:]!r}")
    return response[1:-4]

  async def _read_current(self) -> float:
    """Average four low-range ADC samples, returning current in mA."""
    readings = []
    for _ in range(4):
      data = (await self._peripheral("a1", response_length=10)).strip()
      if not data.isdigit():
        raise ValueError(f"Malformed laser current ADC value: {data!r}")
      readings.append(int(data))
    return sum(readings) / len(readings) * 2.048 / (10 * 65535) * 1000

  async def _enable(self, current: float) -> None:
    """Regulate current against the monitor before routing it to the laser diode.

    Cube 1225010 has a 30 mA absolute maximum. The requested range leaves margin
    below that maximum; adjustment occurs with the laser disconnected from its drive.
    """
    if not math.isfinite(current) or not 1 <= current <= 29:
      raise ValueError("Laser current must be finite and between 1 and 29 mA")
    slot = self._slot
    await self.microscope.led_off()
    await self._peripheral("h0")  # Low-current amplifier range.
    await self._peripheral("w1")  # Monitor the drive before connecting the diode.
    conversion = 6553.6 * 5.02 * 2 / 1000
    code = max(0, int(current * conversion - 9.315))
    for _ in range(8):
      await self._peripheral(f"c{code:05d}")
      measured = await self._read_current()
      if 0 < measured < 30 and abs(measured - current) <= max(0.2, current * 0.02):
        await self._peripheral(f"l{slot}")
        logger.debug("Laser enabled at monitored current %.3f mA", measured)
        return
      if measured <= 0:
        raise RuntimeError("Laser current monitor returned zero while calibrating the drive")
      code = min(int(29 * conversion), max(0, round(code * current / measured)))
    raise RuntimeError("Laser drive current did not converge; illumination remains off")

  async def _set_focus(self, position: float) -> None:
    """Move fine focus using the laser cube's optical compensation."""
    m = self.microscope
    if not math.isfinite(position) or position < 0:
      raise ValueError("Laser focus position must be finite and nonnegative")
    raw_counts = position / m.focus_resolution
    if not math.isfinite(raw_counts) or raw_counts >= 10**m.position_digits:
      raise ValueError("Laser focus position is outside the instrument's position field")
    counts = round(raw_counts)
    if counts >= 10**m.position_digits:
      raise ValueError("Laser focus position is outside the instrument's position field")
    await m.driver.send_command("i", f"F{self._slot}{counts:0{m.position_digits}d}", timeout=20)
    m._focal_height = None

  async def _measure(self) -> Tuple[float, float]:
    """Average five spot centroids to reduce frame noise."""
    m = self.microscope
    positions = [
      laser_spot_position(await m.acquire_image(), m.camera.get_pixel_max()) for _ in range(5)
    ]
    return sum(p[0] for p in positions) / len(positions), sum(p[1] for p in positions) / len(
      positions
    )

  @asynccontextmanager
  async def _session(self, exposure: float, current: float) -> AsyncIterator[None]:
    """Temporarily configure laser optics, restoring camera settings and leaving LEDs off."""
    m = self.microscope
    mode = m._imaging_mode
    old_intensity = m._led_intensity
    if m._objective is None or mode is None:
      raise ValueError("Select an objective and imaging mode before using laser autofocus")
    old_exposure = m._exposure if m._exposure is not None else await m.get_exposure()
    old_gain = m._gain if m._gain is not None else await m.camera.get_gain()
    old_focus = m._focal_height
    was_acquiring = m._acquiring
    old_format = m.camera.camera.get_pixel_format()
    try:
      # Preserve subpixel centroid precision using the camera's full intensity depth.
      if was_acquiring:
        m.stop_acquisition()
      await m.camera.set_pixel_format(0x01100007)  # Unpacked Mono16.
      await m.led_off()
      for command in ("P1101", f"P0d{self._slot:02d}", "P1001"):
        await m.driver.send_command("Y", command)
      await m.set_exposure(exposure)
      await m.set_gain(0)
      m.start_acquisition()
      await self._enable(current)
      yield
    finally:
      try:
        await m.led_off()
        m._imaging_mode = None
        await m.set_imaging_mode(mode, led_intensity=old_intensity)
        await m.led_off()
        await m.set_exposure(old_exposure)
        await m.set_gain(old_gain)
        if old_focus is not None:
          await m.set_focus(old_focus)
      finally:
        m.stop_acquisition()
        await m.camera.set_pixel_format(old_format)
        if was_acquiring:
          m.start_acquisition()

  async def calibrate(
    self, focal_position: float, span: float = 0.2, exposure: float = 5, current: float = 12
  ) -> LaserFocusCalibration:
    """Calibrate at a sample plane focused by image autofocus or visual inspection.

    Positions and span are in mm, exposure in ms, current in mA. The sample must
    remain at the same location during calibration. Repeat for another objective,
    imaging mode or plate type. The three-point scan must remain inside safe focus
    travel. No permanent instrument settings are written.
    """
    if (
      not all(math.isfinite(v) for v in (focal_position, span))
      or span <= 0
      or focal_position < span
    ):
      raise ValueError("Calibration requires a positive span and focal_position >= span")
    if not math.isfinite(exposure) or exposure <= 0:
      raise ValueError("Laser exposure must be finite and positive")
    if not math.isfinite(current) or not 1 <= current <= 29:
      raise ValueError("Laser current must be finite and between 1 and 29 mA")
    m = self.microscope
    async with m.driver._operation():
      objective, mode = m._objective, m._imaging_mode
      if objective is None or mode is None:
        raise ValueError("Select an objective and imaging mode before calibration")
      self.calibration = None
      await m.set_focus(focal_position)
      async with self._session(exposure, current):
        await self._set_focus(focal_position - span)
        lower = await self._measure()
        await self._set_focus(focal_position)
        center = await self._measure()
        await self._set_focus(focal_position + span)
        upper = await self._measure()
      slope = ((upper[0] - lower[0]) / (2 * span), (upper[1] - lower[1]) / (2 * span))
      magnitude = math.hypot(*slope)
      if magnitude * span < 2:
        raise RuntimeError("Laser spot displacement is too small for a reliable calibration")
      residual = math.hypot(
        center[0] - (lower[0] + upper[0]) / 2, center[1] - (lower[1] + upper[1]) / 2
      )
      if residual > max(0.5, magnitude * span * 0.1):
        raise RuntimeError("Laser spot did not move linearly; reflection selection is unreliable")
      calibration = LaserFocusCalibration(
        objective, mode, focal_position, center, slope, exposure, current
      )
      self.calibration = calibration
      logger.info(
        "Laser focus calibrated at %.4f mm with slope %.2f pixels/mm", focal_position, magnitude
      )
      return calibration

  async def autofocus(
    self,
    start: Optional[float] = None,
    max_correction: float = 0.5,
    tolerance: float = 0.005,
    max_rounds: int = 6,
  ) -> float:
    """Return to the calibrated sample plane and retain the corrected focus in mm.

    ``max_correction`` bounds total movement around ``start`` (default: calibration
    focus). Calibration is specific to the objective, imaging mode and plate type.
    A missing, clipped or ambiguous laser spot raises rather than moving blindly.
    """
    calibration = self.calibration
    if calibration is None:
      raise RuntimeError("Calibrate laser autofocus at a focused sample plane first")
    m = self.microscope
    if m._objective != calibration.objective or m._imaging_mode != calibration.imaging_mode:
      raise ValueError("Laser calibration does not match the selected objective and imaging mode")
    origin = calibration.focal_position if start is None else start
    if not all(math.isfinite(v) for v in (origin, max_correction, tolerance)):
      raise ValueError("Laser focus bounds and tolerance must be finite")
    if origin < 0 or max_correction <= 0 or tolerance <= 0 or max_rounds < 1:
      raise ValueError("Laser autofocus requires nonnegative start and positive search limits")
    sx, sy = calibration.slope
    norm = sx * sx + sy * sy
    if not math.isfinite(norm) or norm <= 0:
      raise ValueError("Laser calibration slope must be finite and nonzero")
    position = origin
    async with m.driver._operation():
      async with self._session(calibration.exposure, calibration.current):
        for _ in range(max_rounds):
          await self._set_focus(position)
          x, y = await self._measure()
          dx, dy = x - calibration.spot[0], y - calibration.spot[1]
          correction = -(dx * sx + dy * sy) / norm
          # A transverse shift suggests a different reflection or displaced optics.
          transverse = abs(dx * sy - dy * sx) / math.sqrt(norm)
          if transverse > 3:
            raise RuntimeError("Laser reflection moved outside the calibrated focus trajectory")
          if abs(correction) <= tolerance:
            break
          position += correction
          if position < 0 or abs(position - origin) > max_correction:
            raise RuntimeError("Laser focus correction exceeds the configured travel bound")
        else:
          raise RuntimeError("Laser autofocus did not converge within max_rounds")
      await m.set_focus(position)
      return position
