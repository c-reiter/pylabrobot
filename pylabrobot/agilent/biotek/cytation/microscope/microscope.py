"""The Cytation's microscope.

Owns the AravisCamera and all imaging state/logic: optics control (filter wheel,
objectives, focus, LED, stage positioning), camera control (exposure, gain,
acquisition), and capture orchestration (tiling, retry).

Optics commands go over the plate reader's serial connection (``send_command``); frames
come from the AravisCamera (GenICam/USB3 Vision).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

from pylabrobot.agilent.biotek.cytation.microscope.models import (
  AutoExposure,
  AutoFocus,
  Exposure,
  FocalPosition,
  Gain,
  Image,
  ImagingMode,
  ImagingResult,
  Objective,
)
from pylabrobot.resources.plate import Plate

from .aravis_camera import AravisCamera
from .laser_autofocus import CytationLaserAutofocus
from .configuration import (
  CytationObjectiveConfiguration,
  parse_focus_resolution,
  parse_objective_configuration,
  parse_position_digits,
)

if TYPE_CHECKING:
  from pylabrobot.agilent.biotek.cytation.cytation5 import Cytation5

logger = logging.getLogger(__name__)

# Color brightfield is not a fixed imaging mode. Instead of a single illuminated acquisition, the
# instrument illuminates the sample sequentially with three separate LED colors and the host
# combines the resulting mono frames into one RGB image. These are the firmware LED codes for the
# three colors, in the order they are cycled (red, green, blue), discovered by reverse-engineering
# a Gen5 color-brightfield capture: per color the firmware sends `i L0{code}10` (LED on) followed
# by `i l{code}` (strobe) before each camera trigger. White brightfield (code 5) is used only for
# setup/focus, never in the cycle.
COLOR_BRIGHTFIELD_LED_CODES = (6, 7, 8)


@dataclass
class CytationImagingConfig:
  """Imaging configuration for the Cytation.

  ``camera_pixel_size`` is sensor pixel pitch in mm. When omitted, use the known
  camera model's nominal pitch. ``optical_relay_magnification`` accounts for any
  relay optics between objective and camera; the standard Cytation path is 1×.
  Supply a measured effective pitch for quantitative montage calibration.
  """

  camera_serial_number: Optional[str] = None
  filters: Optional[List[Optional[ImagingMode]]] = None
  objectives: Optional[List[Optional[Objective]]] = None
  max_image_read_attempts: int = 3
  image_read_delay: float = 0.3
  camera_pixel_size: Optional[float] = None
  optical_relay_magnification: float = 1.0


class CytationMicroscope:
  """The Cytation's microscope.

  Owns the AravisCamera and all imaging state, and drives the optics over the plate
  reader's serial connection.
  """

  def __init__(
    self,
    driver: Cytation5,
    camera_serial: Optional[str] = None,
    imaging_config: Optional[CytationImagingConfig] = None,
    use_cam: bool = True,
  ) -> None:
    self.driver = driver
    self.camera = AravisCamera()
    self._camera_serial = camera_serial
    self._use_cam = use_cam
    self.imaging_config = imaging_config or CytationImagingConfig(
      camera_serial_number=camera_serial
    )

    # Imaging state
    self._filters: Optional[List[Optional[ImagingMode]]] = self.imaging_config.filters
    self._objectives: Optional[List[Optional[Objective]]] = self.imaging_config.objectives
    self._laser_slot: Optional[int] = None
    self.laser = CytationLaserAutofocus(self)
    self._objective_configurations: List[Optional[CytationObjectiveConfiguration]] = []
    self._exposure: Optional[Exposure] = None
    self._focal_height: Optional[FocalPosition] = None
    self._focus_resolution: Optional[float] = None
    self._position_digits: Optional[int] = None
    self._gain: Optional[Gain] = None
    self._imaging_mode: Optional[ImagingMode] = None
    self._led_intensity = 10
    self._row: Optional[int] = None
    self._column: Optional[int] = None
    self._pos_x: Optional[float] = None
    self._pos_y: Optional[float] = None
    self._objective: Optional[Objective] = None
    self._acquiring = False

  async def _on_setup(self) -> None:
    """Connect camera (if use_cam) and load filters/objectives from firmware."""
    self._clear_imaging_state()
    self.laser.calibration = None
    if self._use_cam:
      serial = self._camera_serial or (
        self.imaging_config.camera_serial_number if self.imaging_config else None
      )
      await self.camera.setup(serial_number=serial)
      logger.info("Camera connected: %s", self.camera.get_device_info())

    if self.imaging_config.filters is None:
      await self._load_filters()
    if self.imaging_config.objectives is None:
      await self._load_objectives()
    self._focus_resolution = parse_focus_resolution(await self.driver.query_capability(2098))
    self._position_digits = parse_position_digits(await self.driver.query_capability(2102))

  async def _on_stop(self) -> None:
    self._clear_imaging_state()
    try:
      await self.camera.stop()
    except Exception:
      logger.exception("Error stopping camera")

  # ─── Properties ─────────────────────────────────────────────────────

  @property
  def position_digits(self) -> int:
    """Number of characters in a motor position, reported by the instrument."""
    if self._position_digits is None:
      raise RuntimeError("Motor position format not loaded. Call setup() first.")
    return self._position_digits

  @property
  def focus_resolution(self) -> float:
    """Fine-focus millimeters per motor step, reported by the instrument."""
    if self._focus_resolution is None:
      raise RuntimeError("Focus resolution not loaded. Call setup() first.")
    return self._focus_resolution

  @property
  def filters(self) -> List[Optional[ImagingMode]]:
    if self._filters is None:
      raise RuntimeError("Filters not loaded. Call setup() first.")
    return self._filters

  @property
  def objectives(self) -> List[Optional[Objective]]:
    if self._objectives is None:
      raise RuntimeError("Objectives not loaded. Call setup() first.")
    return self._objectives

  def _clear_sample_position(self) -> None:
    """Forget positions affected by tray movement or a change of plate geometry."""
    self._row = None
    self._column = None
    self._pos_x = None
    self._pos_y = None
    self._focal_height = None

  def _clear_imaging_state(self) -> None:
    self._exposure = None
    self._focal_height = None
    self._gain = None
    self._imaging_mode = None
    self._row = None
    self._column = None
    self._pos_x = None
    self._pos_y = None
    self._objective = None
    self._acquiring = False

  # ─── Filter / Objective Discovery ────────────────────────────────────

  async def _load_filters(self) -> None:
    """Discover installed filter cube positions from firmware."""
    cytation_code2imaging_mode = {
      1225121: ImagingMode.C377_647,
      1225123: ImagingMode.C400_647,
      1225113: ImagingMode.C469_593,
      1225109: ImagingMode.ACRIDINE_ORANGE,
      1225107: ImagingMode.CFP,
      1225118: ImagingMode.CFP_FRET_V2,
      1225110: ImagingMode.CFP_YFP_FRET,
      1225119: ImagingMode.CFP_YFP_FRET_V2,
      1225112: ImagingMode.CHLOROPHYLL_A,
      1225105: ImagingMode.CY5,
      1225114: ImagingMode.CY5_5,
      1225106: ImagingMode.CY7,
      1225100: ImagingMode.DAPI,
      1225101: ImagingMode.GFP,
      1225116: ImagingMode.GFP_CY5,
      1225122: ImagingMode.OXIDIZED_ROGFP2,
      1225111: ImagingMode.PROPIDIUM_IODIDE,
      1225103: ImagingMode.RFP,
      1225117: ImagingMode.RFP_CY5,
      1225115: ImagingMode.TAG_BFP,
      1225102: ImagingMode.TEXAS_RED,
      1225104: ImagingMode.YFP,
    }

    self._filters = []
    self._laser_slot = None
    for slot in range(1, 5):
      configuration = await self.driver.send_command("i", f"q{slot}")
      assert configuration is not None
      parts = configuration.decode().strip().split(" ")
      if len(parts) == 1:
        self._filters.append(None)
      else:
        cytation_code = int(parts[0])
        if cytation_code == 1225010:
          self._laser_slot = slot
        self._filters.append(cytation_code2imaging_mode.get(cytation_code, None))

    logger.info("Loaded filters: %s", self._filters)

  @property
  def objective_configurations(self) -> List[Optional[CytationObjectiveConfiguration]]:
    """Return configured turret slots, retaining labels for unknown objective models."""
    return list(self._objective_configurations)

  async def _load_objectives(self) -> None:
    """Read all six objective slots from the objective turret's configuration."""
    configurations: List[Optional[CytationObjectiveConfiguration]] = []
    for slot in range(1, 7):
      response = await self.driver.send_command("i", f"o{slot}")
      if response is None:
        raise RuntimeError("Failed to load objective configuration")
      configuration = parse_objective_configuration(response[:-5])
      if configuration is not None and configuration.objective is None:
        logger.warning(
          "Unrecognized objective in slot %d: %s %s",
          slot,
          configuration.manufacturer,
          configuration.part_number,
        )
      configurations.append(configuration)
    self._objective_configurations = configurations
    self._objectives = [item.objective if item is not None else None for item in configurations]
    logger.info("Configured objectives: %s", self._objectives)

  # ─── Camera Control ──────────────────────────────────────────────────

  async def set_exposure(self, exposure: Exposure) -> None:
    """Set camera exposure time in ms."""
    async with self.driver._operation():
      if exposure == self._exposure:
        return

      if isinstance(exposure, str):
        if exposure == "machine-auto":
          await self.camera.set_auto_exposure("continuous")
          self._exposure = "machine-auto"
          return
        raise ValueError("exposure must be a number or 'machine-auto'")
      await self.camera.set_auto_exposure("off")
      await self.camera.set_exposure(float(exposure))
      self._exposure = exposure

  async def set_gain(self, gain: Gain) -> None:
    """Set camera gain."""
    async with self.driver._operation():
      if gain == self._gain:
        return

      if gain == "machine-auto":
        await self.camera.set_auto_gain("continuous")
      else:
        await self.camera.set_gain(float(gain))

      self._gain = gain

  async def set_auto_exposure(self, auto_exposure: Literal["off", "once", "continuous"]) -> None:
    """Set camera auto-exposure mode."""
    await self.camera.set_auto_exposure(auto_exposure)

  def start_acquisition(self) -> None:
    """Start camera acquisition (buffered streaming)."""
    self.camera.start_acquisition()
    self._acquiring = True

  def stop_acquisition(self) -> None:
    """Stop camera acquisition."""
    if self._acquiring:
      self.camera.stop_acquisition()
      self._acquiring = False

  async def acquire_image(self) -> Image:
    """Trigger camera and read image, with retry on failure."""
    config = self.imaging_config
    for attempt in range(config.max_image_read_attempts):
      try:
        return await self.camera.trigger(timeout_ms=5000)
      except Exception:
        if attempt < config.max_image_read_attempts - 1:
          logger.warning("Failed to get image, retrying...")
          self.stop_acquisition()
          self.start_acquisition()
          await asyncio.sleep(config.image_read_delay)
        else:
          raise
    raise TimeoutError("max_image_read_attempts reached")

  async def _set_color_brightfield_led(self, led_code: int, intensity: int) -> None:
    """Switch color-brightfield illumination to a single LED color and strobe it.

    Unlike other modes, color brightfield has no static LED state: the instrument turns on one of
    the three LED colors (``i L0{led_code}{intensity}``), strobes it (``i l{led_code}``), then the
    camera is triggered, repeating once per color. See ``COLOR_BRIGHTFIELD_LED_CODES``.
    """
    if not 1 <= intensity <= 10:
      raise ValueError("intensity must be between 1 and 10")
    intensity_str = str(intensity).zfill(2)
    await self.driver.send_command("i", f"L0{led_code}{intensity_str}")
    await self.driver.send_command("i", f"l{led_code}")

  async def _acquire_color_brightfield_image(self, led_intensity: int) -> Image:
    """Acquire one color-brightfield image.

    Captures a mono frame under each of the three brightfield LED colors and stacks them into an
    ``(H, W, 3)`` RGB image. A single exposure/gain (set by the caller) is used for all three
    channels; Gen5's per-color auto-exposure is not replicated.
    """
    import numpy as np

    channels: List[Image] = []
    for led_code in COLOR_BRIGHTFIELD_LED_CODES:
      await self._set_color_brightfield_led(led_code, intensity=led_intensity)
      channels.append(await self.acquire_image())
    return np.stack(channels, axis=-1)

  async def _acquire_analysis_image(self) -> Image:
    """Use the selected illumination, including all RGB channels, for automatic searches."""
    if self._imaging_mode == ImagingMode.COLOR_BRIGHTFIELD:
      return await self._acquire_color_brightfield_image(self._led_intensity)
    return await self.acquire_image()

  async def auto_expose(self, settings: Optional[AutoExposure] = None) -> float:
    """Find and retain a bounded exposure, returning milliseconds.

    Illumination, gain, well and focus must already be configured. Without custom
    settings, place the image's 99th percentile between 60% and 80% of the
    monochrome pixel range, within the camera's limits and at most 16 frames.
    Custom settings can evaluate a region or a different fluorescence target.
    Failure to meet the requested exposure criterion raises RuntimeError.
    """
    import numpy as np

    camera_low, camera_high = self.camera.get_exposure_limits()
    if settings is None:

      async def evaluate(image: Image) -> Literal["higher", "lower", "good"]:
        """Avoid clipping the brightest structures without amplifying isolated hot pixels."""
        if image.dtype.kind != "u":
          raise ValueError("Automatic exposure requires unsigned images")
        level = float(np.percentile(image, 99)) / self.camera.get_pixel_max()
        return "higher" if level < 0.6 else "lower" if level > 0.8 else "good"

      settings = AutoExposure(evaluate, max_rounds=16, low=camera_low, high=camera_high)
    low, high = settings.low, settings.high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
      raise ValueError("Auto-exposure bounds must be finite and 0 < low <= high")
    if low < camera_low or high > camera_high:
      raise ValueError("Auto-exposure bounds must lie within the camera's exposure limits")
    if settings.max_rounds < 1:
      raise ValueError("Auto-exposure max_rounds must be positive")
    was_acquiring = self._acquiring
    async with self.driver._operation():
      try:
        if not was_acquiring:
          self.start_acquisition()
        for _ in range(settings.max_rounds):
          exposure = math.sqrt(low) * math.sqrt(high)
          await self.set_exposure(exposure)
          result = await settings.evaluate_exposure(await self._acquire_analysis_image())
          if result == "good":
            return await self.get_exposure()
          if result == "higher":
            low = exposure
          elif result == "lower":
            high = exposure
          else:
            raise ValueError(f"Invalid exposure evaluation: {result!r}")
        raise RuntimeError("No acceptable exposure found within the configured bounds and rounds")
      finally:
        if not was_acquiring:
          self.stop_acquisition()

  async def autofocus(self, settings: AutoFocus) -> float:
    """Find and retain the best image focus inside a bounded interval in mm.

    Scan nine evenly spaced positions, then refine around the highest score.
    The evaluator must return a finite score, with higher values indicating better
    focus. Exposure and gain must be fixed before calling. Flat scores and peaks
    at the interval boundary raise instead of claiming a successful focus.
    ``timeout`` is checked between captures; an in-progress motor/camera operation
    finishes or reaches its own timeout before the search stops.
    """
    low, high, tolerance = settings.low, settings.high, settings.tolerance
    if not all(math.isfinite(value) for value in (low, high, tolerance, settings.timeout)):
      raise ValueError("Autofocus bounds, tolerance and timeout must be finite")
    if not 0 <= low < high or tolerance <= 0 or settings.timeout <= 0:
      raise ValueError("Autofocus requires 0 <= low < high and positive tolerance and timeout")
    if high / self.focus_resolution >= 10**self.position_digits:
      raise ValueError("Autofocus interval exceeds the instrument's focus position field")
    if self._exposure in (None, "machine-auto") or self._gain in (None, "machine-auto"):
      raise ValueError("Set a fixed exposure and gain before autofocus")
    tolerance = max(tolerance, self.focus_resolution)
    scores: dict[float, float] = {}
    deadline = time.monotonic() + settings.timeout

    async def evaluate(position: float) -> None:
      """Capture a previously unmeasured position and reject invalid evaluation scores."""
      if time.monotonic() >= deadline:
        raise TimeoutError("Autofocus search timed out")
      if position in scores:
        return
      await self.set_focus(position)
      score = float(settings.evaluate_focus(await self._acquire_analysis_image()))
      if not math.isfinite(score):
        raise ValueError("Focus evaluator returned a non-finite score")
      scores[position] = score

    async def search() -> float:
      """Bracket the strongest coarse peak and refine both sides to motor precision."""
      for i in range(9):
        await evaluate(low + (high - low) * i / 8)
      if max(scores.values()) == min(scores.values()):
        raise RuntimeError("Autofocus found no image contrast change across the interval")
      while True:
        positions = sorted(scores)
        best = max(positions, key=lambda position: scores[position])
        index = positions.index(best)
        if index == 0 or index == len(positions) - 1:
          raise RuntimeError("Best focus is at the search boundary; choose a wider interval")
        left, right = positions[index - 1], positions[index + 1]
        if right - left <= tolerance:
          await self.set_focus(best)
          return best
        await evaluate((left + best) / 2)
        await evaluate((best + right) / 2)

    was_acquiring = self._acquiring
    async with self.driver._operation():
      try:
        if not was_acquiring:
          self.start_acquisition()
        return await search()
      finally:
        if not was_acquiring:
          self.stop_acquisition()

  async def get_exposure(self) -> float:
    """Get current exposure time in ms."""
    return await self.camera.get_exposure()

  # ─── Optics Control (serial protocol) ────────────────────────────────

  def _imaging_mode_code(self, mode: ImagingMode) -> int:
    """Get filter wheel position index for an imaging mode."""
    if mode in (
      ImagingMode.BRIGHTFIELD,
      ImagingMode.PHASE_CONTRAST,
      ImagingMode.COLOR_BRIGHTFIELD,
    ):
      # Color brightfield uses the brightfield light path (filter cube 5); its three colors are
      # selected via LED codes 6/7/8 during acquisition, not here.
      return 5
    for i, f in enumerate(self.filters):
      if f == mode:
        return i + 1
    raise ValueError(f"Mode {mode} not found in filters: {self.filters}")

  def _objective_code(self, objective: Objective) -> int:
    """Get turret position index for an objective."""
    for i, o in enumerate(self.objectives):
      if o == objective:
        return i + 1
    raise ValueError(f"Objective {objective} not found: {self.objectives}")

  async def set_imaging_mode(self, mode: ImagingMode, led_intensity: int = 10) -> None:
    """Set filter wheel position and LED."""
    if not isinstance(led_intensity, int) or not 1 <= led_intensity <= 10:
      raise ValueError("led_intensity must be an integer between 1 and 10")
    async with self.driver._operation():
      self._led_intensity = led_intensity
      if mode == self._imaging_mode:
        # Color brightfield has no single LED state to (re)enable; its colors are cycled per frame
        # during acquisition (see _acquire_color_brightfield_image).
        if mode != ImagingMode.COLOR_BRIGHTFIELD:
          await self.led_on(intensity=led_intensity)
        return

      await self.led_off()
      filter_index = self._imaging_mode_code(mode)

      if mode == ImagingMode.PHASE_CONTRAST:
        # Position 20 selects the configured annulus for the current objective.
        await self.driver.send_command("Y", "P1120")
        await self.driver.send_command("Y", "P0d05")
        await self.driver.send_command("Y", "P1002")
      elif mode in (ImagingMode.BRIGHTFIELD, ImagingMode.COLOR_BRIGHTFIELD):
        await self.driver.send_command("Y", "P1101")
        await self.driver.send_command("Y", "P0d05")
        await self.driver.send_command("Y", "P1002")
      else:
        await self.driver.send_command("Y", "P1101")
        await self.driver.send_command("Y", f"P0d{filter_index:02}")
        await self.driver.send_command("Y", "P1001")

      # Color brightfield has no single LED state; its colors are cycled per frame during
      # acquisition (see _acquire_color_brightfield_image).
      self._imaging_mode = mode
      self._focal_height = None
      if mode != ImagingMode.COLOR_BRIGHTFIELD:
        await self.led_on(intensity=led_intensity)

  async def set_objective(self, objective: Objective) -> None:
    """Rotate objective turret to the specified objective."""
    async with self.driver._operation():
      if objective == self._objective:
        return
      obj_code = self._objective_code(objective)
      await self.driver.send_command("Y", f"P0e{obj_code:02}", timeout=60)
      self._objective = objective
      self._imaging_mode = None  # force re-set after objective change
      self._focal_height = None
      self._pos_x, self._pos_y = None, None

  async def set_focus(self, focal_position: FocalPosition) -> None:
    """Move focus motor to the specified height (mm)."""
    if focal_position == "machine-auto":
      await self.laser.autofocus()
      return

    position = float(focal_position)
    if not math.isfinite(position) or position < 0:
      raise ValueError("focal_position must be a finite, nonnegative height in mm")

    if focal_position == self._focal_height:
      return

    counts = position / self.focus_resolution
    if not math.isfinite(counts) or counts >= 10**self.position_digits:
      raise ValueError("focal_position exceeds the instrument's focus position field")
    focus_integer = int(counts + 0.5)
    if focus_integer >= 10**self.position_digits:
      raise ValueError("focal_position exceeds the instrument's focus position field")
    focus_str = f"{focus_integer:0{self.position_digits}d}"

    if self._imaging_mode is None:
      raise ValueError("Imaging mode not set. Call set_imaging_mode() first.")
    imaging_mode_code = self._imaging_mode_code(self._imaging_mode)
    await self.driver.send_command("i", f"F{imaging_mode_code}{focus_str}")

    self._focal_height = focal_position

  async def led_on(self, intensity: int = 10) -> None:
    """Turn on LED at specified intensity (1-10)."""
    if not 1 <= intensity <= 10:
      raise ValueError("intensity must be between 1 and 10")
    if self._imaging_mode is None:
      raise ValueError("Imaging mode not set. Call set_imaging_mode() first.")
    imaging_mode_code = self._imaging_mode_code(self._imaging_mode)
    intensity_str = str(intensity).zfill(2)
    await self.driver.send_command("i", f"L0{imaging_mode_code}{intensity_str}")

  async def led_off(self) -> None:
    """Turn off LED."""
    await self.driver.send_command("i", "L0001")

  def _stage_counts(self, position: float) -> int:
    """Quantize a stage offset in mm to the Cytation's 10.16 µm motor steps."""
    if not math.isfinite(position):
      raise ValueError("Stage offsets must be finite")
    raw_counts = position / 0.01016
    if not math.isfinite(raw_counts) or abs(raw_counts) >= 10**self.position_digits:
      raise ValueError("Stage offset exceeds the instrument's position field")
    counts = round(raw_counts)
    if len(f"{counts:0{self.position_digits}d}") > self.position_digits:
      raise ValueError("Stage offset exceeds the instrument's position field")
    return counts

  async def select(self, row: int, column: int) -> None:
    """Select a well using one-based row and column numbers and center the stage."""
    if not isinstance(row, int) or not isinstance(column, int):
      raise ValueError("Row and column must be integers")
    if not 1 <= row <= 99 or not 1 <= column <= 99:
      raise ValueError("Row and column must be between 1 and 99")
    async with self.driver._operation():
      if row == self._row and column == self._column and self._pos_x is not None:
        return
      await self.driver.send_command("Y", f"W6{row:02d}{column:02d}", timeout=50)
      self._row, self._column = row, column
      self._pos_x, self._pos_y = 0.0, 0.0

  async def set_position(self, x: float, y: float) -> None:
    """Move to offsets in mm from the selected well center.

    Targets are quantized before calculating relative moves, so repeated sub-step
    requests cannot accumulate rounding error. An unknown position is re-centered
    before applying offsets.
    """
    x_counts, y_counts = self._stage_counts(x), self._stage_counts(y)
    if self._row is None or self._column is None:
      raise ValueError("Row and column not set. Call select() first.")
    async with self.driver._operation():
      if self._pos_x is None or self._pos_y is None:
        await self.select(self._row, self._column)
      assert self._pos_x is not None and self._pos_y is not None
      dx = x_counts - self._stage_counts(self._pos_x)
      dy = y_counts - self._stage_counts(self._pos_y)
      for delta in (dx, dy):
        if len(f"{delta:0{self.position_digits}d}") > self.position_digits:
          raise ValueError("Relative stage movement exceeds the position field")
      try:
        if dx:
          await self.driver.send_command("Y", f"O00{dx:0{self.position_digits}d}", timeout=50)
        if dy:
          await self.driver.send_command("Y", f"O01{dy:0{self.position_digits}d}", timeout=50)
      except BaseException:
        self._pos_x, self._pos_y = None, None
        raise
      self._pos_x, self._pos_y = x_counts * 0.01016, y_counts * 0.01016
      if dx or dy:
        await asyncio.sleep(0.1)

  async def get_focus_reference(self, at_position: bool = False) -> Optional[float]:
    """Read the calibrated focus reference in mm without moving the focus motor.

    By default this returns the objective/filter pair's factory reference. With
    ``at_position=True``, query its calibration at the selected well and offset.
    A reference is a starting point for focusing, not a measurement of the sample.
    Returns None when the instrument has no valid calibration for this combination.
    """
    if self._objective is None or self._imaging_mode is None:
      raise ValueError("Set an objective and imaging mode before querying focus calibration")
    row = column = x = y = 0
    if at_position:
      if self._row is None or self._column is None or self._pos_x is None or self._pos_y is None:
        raise ValueError("Select a well and position before querying its focus calibration")
      row, column = self._row, self._column
      x, y = self._stage_counts(self._pos_x), self._stage_counts(self._pos_y)
    objective = self._objective_code(self._objective)
    cube = self._imaging_mode_code(self._imaging_mode)
    mode = 6 if at_position else 0
    response = await self.driver.send_command(
      "Y",
      f"Z{objective}{cube}{mode}{row:02d}{column:02d}"
      f"{y:0{self.position_digits}d}{x:0{self.position_digits}d}",
      timeout=20,
    )
    if response is None or len(response) != self.position_digits + 6:
      raise ValueError(f"Malformed focus calibration response: {response!r}")
    if response[:1] not in (b"0", b"1") or not response[1:-5].isdigit():
      raise ValueError(f"Malformed focus calibration response: {response!r}")
    return int(response[1:-5]) * self.focus_resolution if response[:1] == b"1" else None

  def get_field_of_view(self, objective: Objective) -> Tuple[float, float]:
    """Return image width and height in mm using camera dimensions and optical scale.

    The Grasshopper GS3-U3-14S5M has 6.45 µm sensor pixels. Other cameras require
    ``camera_pixel_size`` in the imaging configuration. Calibration takes precedence
    over nominal sensor pitch. The camera must be connected to read its image size.
    """
    pitch = self.imaging_config.camera_pixel_size
    relay = self.imaging_config.optical_relay_magnification
    if pitch is None:
      model = self.camera.get_device_info().model_name.removeprefix("Grasshopper3 ")
      pitches = {"GS3-U3-14S5M": 0.00645, "GS3-U3-14S5M-C": 0.00645}
      if model not in pitches:
        raise ValueError(f"Set camera_pixel_size to calibrate the field of view of {model}")
      pitch = pitches[model]
    if not math.isfinite(pitch) or pitch <= 0 or not math.isfinite(relay) or relay <= 0:
      raise ValueError("Pixel size and optical relay magnification must be finite and positive")
    scale = pitch / (objective.magnification * relay)
    if self.camera.width <= 0 or self.camera.height <= 0:
      raise RuntimeError("Connect the camera before calculating its field of view")
    return self.camera.width * scale, self.camera.height * scale

  @staticmethod
  def _tile_positions(
    coverage: Union[Literal["full"], Tuple[int, int]],
    well_size: Tuple[float, float],
    field_of_view: Tuple[float, float],
    center: Tuple[float, float],
    overlap: float,
  ) -> Tuple[Tuple[int, int], List[Tuple[float, float]]]:
    """Plan a centered row-major tile grid, retaining the camera's aspect ratio."""
    width, height = field_of_view
    if not all(math.isfinite(v) and v > 0 for v in (width, height, *well_size)):
      raise ValueError("Field of view and well dimensions must be finite and positive")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
      raise ValueError("overlap must be finite and in [0, 1)")
    if not all(math.isfinite(v) for v in center):
      raise ValueError("Capture center coordinates must be finite")
    dx, dy = width * (1 - overlap), height * (1 - overlap)
    if coverage == "full":
      rows = max(1, math.ceil((well_size[1] - height) / dy) + 1)
      cols = max(1, math.ceil((well_size[0] - width) / dx) + 1)
    else:
      rows, cols = coverage
      if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        raise ValueError("coverage must specify positive integer rows and columns")
    positions = [
      (center[0] + (column - (cols - 1) / 2) * dx, center[1] - (row - (rows - 1) / 2) * dy)
      for row in range(rows)
      for column in range(cols)
    ]
    return (rows, cols), positions

  async def capture(
    self,
    row: int,
    column: int,
    mode: ImagingMode,
    objective: Objective,
    exposure_time: Union[Exposure, AutoExposure],
    focal_height: Union[FocalPosition, AutoFocus],
    gain: Gain,
    plate: Plate,
    led_intensity: int = 10,
    coverage: Union[Literal["full"], Tuple[int, int]] = (1, 1),
    center_position: Optional[Tuple[float, float]] = None,
    overlap: Optional[float] = None,
    auto_stop_acquisition: bool = True,
  ) -> ImagingResult:
    """Capture an image of a well.

    Args:
      led_intensity: LED intensity (1-10).
      coverage: Image tiling coverage. ``"full"`` for full-well montage, or a
        ``(rows, cols)`` tuple for a specific tile grid. ``(1, 1)`` is a single image.
      center_position: Center position of the capture area as ``(x_mm, y_mm)`` relative
        to the well center. If None, centers on the well.
      overlap: Fractional overlap between tiles (0 <= overlap < 1) for montage stitching.
        If None, no overlap. Only used when coverage produces multiple tiles.
      auto_stop_acquisition: Whether to automatically stop image acquisition after capture.
    """
    if not 1 <= row <= plate.num_items_y or not 1 <= column <= plate.num_items_x:
      raise ValueError("Capture row and column must identify a well on the supplied plate")
    center = center_position if center_position is not None else (0.0, 0.0)
    first_well = plate.get_item(0)
    # A single frame needs no physical-scale calibration.
    field_of_view = (1.0, 1.0) if coverage == (1, 1) else self.get_field_of_view(objective)
    grid, positions = self._tile_positions(
      coverage,
      (first_well.get_size_x(), first_well.get_size_y()),
      field_of_view,
      center,
      overlap if overlap is not None else 0.0,
    )

    async with self.driver._operation():
      await self.driver._prepare_microscopy(plate)

      if not self._acquiring:
        self.start_acquisition()

      try:
        await self.set_objective(objective)
        await self.set_imaging_mode(mode, led_intensity=led_intensity)
        await self.select(row, column)
        await self.set_position(*center)
        await self.set_gain(gain)
        if isinstance(focal_height, AutoFocus):
          if isinstance(exposure_time, AutoExposure) or exposure_time == "machine-auto":
            raise ValueError("Image autofocus requires a fixed exposure")
          await self.set_exposure(exposure_time)
          await self.autofocus(focal_height)
        else:
          await self.set_focus(focal_height)
          if isinstance(exposure_time, AutoExposure):
            await self.auto_expose(exposure_time)
          elif exposure_time == "machine-auto":
            await self.auto_expose()
          else:
            await self.set_exposure(exposure_time)

        images: List[Image] = []
        actual_positions: List[Tuple[float, float]] = []
        for x_pos, y_pos in positions:
          await self.set_position(x=x_pos, y=y_pos)
          assert self._pos_x is not None and self._pos_y is not None
          actual_positions.append((self._pos_x, self._pos_y))
          t0 = time.time()
          if mode == ImagingMode.COLOR_BRIGHTFIELD:
            images.append(await self._acquire_color_brightfield_image(led_intensity=led_intensity))
          else:
            images.append(await self.acquire_image())
          t1 = time.time()
          logger.debug("[cytation] acquired image in %.2f seconds", t1 - t0)
      finally:
        try:
          await self.led_off()
        except Exception:
          logger.exception("Failed to turn off LED during cleanup")
        if auto_stop_acquisition:
          self.stop_acquisition()

      exposure_ms = await self.get_exposure()
      assert self._focal_height is not None
      focal_height_val = float(self._focal_height)

      return ImagingResult(
        images=images,
        exposure_time=exposure_ms,
        focal_height=focal_height_val,
        tile_positions=actual_positions,
        tile_grid=grid,
      )
