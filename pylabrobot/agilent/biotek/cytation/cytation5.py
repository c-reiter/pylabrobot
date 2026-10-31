from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from pylabrobot.agilent.biotek._reader import BioTekError
from pylabrobot.agilent.biotek.results import FluorescenceResult
from pylabrobot.resources import Plate, Well

from pylabrobot.agilent.biotek.cytation.base import _CytationBase
from pylabrobot.agilent.biotek.cytation.microscope.microscope import (
  CytationImagingConfig,
  CytationMicroscope,
)


logger = logging.getLogger(__name__)


class Cytation5(_CytationBase):
  """Agilent BioTek Cytation 5. Adds a camera, reached through :attr:`microscope`."""

  _model_name = "Agilent BioTek Cytation 5"

  def __init__(
    self,
    name: str,
    camera_serial: Optional[str] = None,
    device_id: Optional[str] = None,
    imaging_config: Optional[CytationImagingConfig] = None,
    use_cam: bool = True,
    port: Optional[str] = None,
  ):
    super().__init__(name=name, device_id=device_id, port=port)

    self._tray_closed = False
    self.microscope = CytationMicroscope(
      driver=self,
      camera_serial=camera_serial,
      imaging_config=imaging_config,
      use_cam=use_cam,
    )

  @property
  def supports_heating(self) -> bool:
    """The Cytation 5 has an incubation heater."""
    return True

  @property
  def temperature_range(self) -> Tuple[Optional[float], Optional[float]]:
    """Heating reaches 65 °C; the lowest regulated temperature depends on ambient."""
    return None, 65.0

  async def _temperature_parameters(self, temperature: float, gradient: float) -> str:
    """Encode firmware-selected temperature and top/bottom gradient fields."""
    tenths = (await self.query_capability(2126))[:1] == b"1"
    gradient_capability = await self.query_capability(2088)
    has_gradient = gradient_capability[:1] == b"1"
    resolution = 0.1 if tenths else 1.0
    values = (temperature, gradient)
    if any(not math.isclose(value / resolution, round(value / resolution)) for value in values):
      raise ValueError(f"This firmware requires temperature and gradient steps of {resolution} °C")
    parameters = f"{round(temperature / resolution):0{3 if tenths else 2}d}"
    if has_gradient:
      limit_field = gradient_capability[1:2]
      if not limit_field.isdigit():
        raise ValueError(f"Malformed temperature-gradient capability: {gradient_capability!r}")
      limit = int(limit_field) or 3
      if not 0 <= gradient <= limit:
        raise ValueError(f"Temperature gradient must be between 0 and {limit} °C")
      parameters += f"{round(gradient / resolution):0{2 if tenths else 1}d}"
    elif gradient != 0:
      raise ValueError("This firmware does not support a temperature gradient")
    return parameters

  async def set_temperature(self, temperature: float, gradient: float = 0) -> Optional[bytes]:
    """Set incubation in °C, with an optional top/bottom temperature gradient.

    Regulation is specified from ambient + 4 °C to 65 °C. Lowering the target
    allows passive cooling. Firmware determines the supported resolution and
    gradient range. This acknowledges the target without waiting for equilibrium.
    """
    if not math.isfinite(temperature) or not 0 < temperature <= 65:
      raise ValueError("Cytation 5 temperature must be finite and in (0, 65] °C")
    if not math.isfinite(gradient) or gradient < 0:
      raise ValueError("Temperature gradient must be finite and nonnegative")
    async with self._operation():
      parameters = await self._temperature_parameters(temperature, gradient)
      return await self.send_command("g", parameters, timeout=40)

  async def stop_heating_or_cooling(self) -> Optional[bytes]:
    """Disable incubation using this firmware's temperature field lengths."""
    async with self._operation():
      return await self.send_command("g", await self._temperature_parameters(0, 0), timeout=40)

  async def read_time_resolved_fluorescence(
    self,
    plate: Plate,
    wells: List[Well],
    excitation_wavelength: int,
    emission_wavelength: int,
    focal_height: float,
    delay: float = 0.0001,
    integration_time: float = 0.0003,
  ) -> List[FluorescenceResult]:
    """Read flash-excited, time-resolved fluorescence through the monochromators.

    Uses ten flashes, 20 nm bandwidths and gain 100. Timing is performed by the
    instrument after each excitation flash, rather than by host-side sleeps.
    This operation has not yet been validated on hardware.

    Args:
      plate: Plate to read.
      wells: Wells to read.
      excitation_wavelength: Excitation wavelength in nm.
      emission_wavelength: Emission wavelength in nm.
      focal_height: Focal height in mm.
      delay: Delay after each flash, in seconds, from 0 to 0.002.
      integration_time: Collection duration in seconds, from 0.00002 to 0.002.
        Both timings must be whole multiples of one microsecond.
    """
    timing = (delay, integration_time)
    if any(not math.isfinite(value) for value in timing):
      raise ValueError("Fluorescence timing must be finite")
    if not 0 <= delay <= 0.002 or not 0.00002 <= integration_time <= 0.002:
      raise ValueError("Delay must be 0–0.002 s and integration time 0.00002–0.002 s")
    if any(
      not math.isclose(value * 1_000_000, round(value * 1_000_000), abs_tol=1e-8)
      for value in timing
    ):
      raise ValueError("Fluorescence timing requires whole microsecond steps")
    async with self._operation():
      capability = await self.query_capability(2006)
      if capability[:1] != b"1" or capability[2:3] != b"1":
        raise NotImplementedError("This Cytation has no monochromator time-resolved fluorescence")
      logger.warning("Cytation time-resolved fluorescence has not yet been validated on hardware")
      return await self._read_fluorescence(
        plate,
        wells,
        excitation_wavelength,
        emission_wavelength,
        focal_height,
        delay_microseconds=round(delay * 1_000_000),
        collection_microseconds=round(integration_time * 1_000_000),
      )

  def _reset_state(self) -> None:
    """Discard optics and plate position assumptions after disconnecting."""
    super()._reset_state()
    self._tray_closed = False
    self.microscope._clear_imaging_state()

  async def setup(self) -> None:
    """Connect the controller and optional camera, releasing both on failure."""
    try:
      await super().setup()
      await self.microscope._on_setup()
    except BaseException:
      await self.microscope._on_stop()
      await super().stop()
      raise

  async def stop(self) -> None:
    """Switch off illumination and release the camera and controller."""
    try:
      if self._version is not None:
        try:
          await self.microscope.led_off()
        except Exception:
          logger.exception("Failed to switch off illumination while stopping Cytation 5")
    finally:
      try:
        await self.microscope._on_stop()
      finally:
        await super().stop()

  async def _prepare_plate_read(self) -> None:
    """Disable microscopy illumination and forget stage positions before an assay."""
    self._tray_closed = False
    await self.microscope.led_off()
    self.microscope.stop_acquisition()
    self.microscope._clear_imaging_state()

  async def open(self, slow: bool = False) -> None:
    """Extend the loading tray and invalidate the microscope's sample position."""
    async with self._operation():
      await self.microscope.led_off()
      await super().open(slow=slow)
      self._tray_closed = False
      self.microscope.laser.calibration = None
      self.microscope._clear_sample_position()

  async def close(self, plate: Optional[Plate] = None, slow: bool = False) -> None:
    """Retract the tray using the supplied plate geometry."""
    async with self._operation():
      await super().close(plate=plate, slow=slow)
      self._tray_closed = True
      self.microscope._clear_sample_position()

  async def set_plate(self, plate: Plate) -> Optional[bytes]:
    """Load plate geometry, cycling the tray if the firmware requires it.

    Cytation 5 can return 0x5B00 when a height change requires the carrier outside
    the instrument. For that status only, extend the tray, retry the geometry and
    retract it. Keep the tray area clear when changing plate geometry.
    """
    async with self._operation():
      if plate is self._plate:
        return None
      try:
        response = await super().set_plate(plate)
      except BioTekError as exc:
        if exc.code != 0x5B00:
          raise
        logger.info("Cytation requires the tray outside to update plate geometry")
        await self.open()
        response = await super().set_plate(plate)
        await self.close()
        self._plate = plate
      self.microscope._clear_sample_position()
      self.microscope.laser.calibration = None
      return response

  async def _prepare_microscopy(self, plate: Plate) -> None:
    """Retract the tray before imaging when its state is unknown or an assay has run."""
    async with self._operation():
      if not self._tray_closed:
        await self.close(plate=plate)
      else:
        await self.set_plate(plate)

  async def home(self) -> Optional[bytes]:
    """Home the imager axes and extend the carrier; reclose before microscopy."""
    async with self._operation():
      await self.microscope.led_off()
      response = await super().home()
      self._tray_closed = False
      self.microscope.laser.calibration = None
      self.microscope._clear_imaging_state()
      return response
