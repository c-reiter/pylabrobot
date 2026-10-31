from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

from pylabrobot.agilent.biotek._reader import BioTekError
from pylabrobot.resources import Plate

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

  async def set_temperature(self, temperature: float) -> Optional[bytes]:
    """Set the incubation target in °C, up to 65 °C.

    Regulation is specified from ambient + 4 °C. Lowering the target allows
    passive cooling; this instrument does not have active cooling. Use
    ``stop_heating_or_cooling()`` to disable the heater.
    """
    if not math.isfinite(temperature) or not 0 < temperature <= 65:
      raise ValueError("Cytation 5 temperature must be finite and in (0, 65] °C")
    return await self.send_command("g", f"{round(temperature * 1000):05d}")

  def _reset_state(self) -> None:
    """Discard optics and plate position assumptions after disconnecting."""
    super()._reset_state()
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

  async def open(self, slow: bool = False) -> None:
    """Extend the loading tray and invalidate the microscope's sample position."""
    async with self._operation():
      await self.microscope.led_off()
      await super().open(slow=slow)
      self.microscope._clear_sample_position()

  async def close(self, plate: Optional[Plate] = None, slow: bool = False) -> None:
    """Retract the tray using the supplied plate geometry."""
    async with self._operation():
      await super().close(plate=plate, slow=slow)
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
      return response
