import asyncio
import enum
import logging
import math
from abc import ABCMeta
from datetime import datetime
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from pylabrobot.agilent.biotek._reader import BioTekHandshakeError, _BioTekReader
from pylabrobot.agilent.biotek.plate_reader_protocol import (
  encode_absorbance,
  encode_fluorescence,
  encode_luminescence,
  parse_assay_response,
  parse_response,
  validate_detector_settings,
  validate_fluorescence_settings,
)
from pylabrobot.agilent.biotek.results import (
  AbsorbanceResult as AbsorbanceResult,
)
from pylabrobot.agilent.biotek.results import (
  FluorescenceResult as FluorescenceResult,
)
from pylabrobot.agilent.biotek.results import (
  LuminescenceResult as LuminescenceResult,
)
from pylabrobot.io.serial import Serial
from pylabrobot.resources import Plate, Well

logger = logging.getLogger(__name__)


class BioTekPlateReaderDriver(_BioTekReader, metaclass=ABCMeta):
  """RDOP22 assays and plate geometry shared by Cytation and Synergy H1 readers."""

  def __init__(
    self,
    timeout: float = 20,
    device_id: Optional[str] = None,
    human_readable_device_name: str = "Agilent BioTek",
    port: Optional[str] = None,
  ) -> None:
    super().__init__(timeout, device_id, human_readable_device_name, port=port)

    self._plate: Optional[Plate] = None
    self._shaking = False
    self._shaking_task: Optional[asyncio.Task] = None
    self._slow_mode: Optional[bool] = None

  def _non_overlapping_rectangles(
    self,
    points: Iterable[Tuple[int, int]],
  ) -> List[Tuple[int, int, int, int]]:
    """Find non-overlapping rectangles that cover all given points."""
    pts = set(points)
    rects = []

    while pts:
      r0, c0 = min(pts)
      c1 = c0
      while (r0, c1 + 1) in pts:
        c1 += 1
      r1 = r0
      while all((r1 + 1, c) in pts for c in range(c0, c1 + 1)):
        r1 += 1

      rects.append((r0, c0, r1, c1))
      for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
          pts.discard((r, c))

    rects.sort()
    return rects

  def _reset_state(self) -> None:
    """Forget the plate geometry and motion settings associated with the connection."""
    super()._reset_state()
    self._plate = None
    self._slow_mode = None
    self._shaking = False

  async def _configure_serial(self) -> None:
    """Reset the adapter before applying the RDOP22 readers' serial settings."""
    if not isinstance(self.io, Serial):
      await self.io.usb_reset()
    await super()._configure_serial()

  async def _initialize_reader(self) -> None:
    """Identify the reader, retrying at the alternate supported baud rate."""
    try:
      await super()._initialize_reader()
    except (TimeoutError, BioTekHandshakeError):
      if isinstance(self.io, Serial):
        await self.io.stop()
        self.io.baudrate = 38_461
        await self.io.setup()
      else:
        await self.io.set_baudrate(38_461)
      await super()._initialize_reader()

  @property
  def abs_wavelength_range(self) -> tuple:
    return (230, 999)

  @property
  def focal_height_range(self) -> tuple:
    return (4.5, 13.88)

  @property
  def excitation_range(self) -> tuple:
    return (250, 700)

  @property
  def emission_range(self) -> tuple:
    return (250, 700)

  @property
  def supports_heating(self) -> bool:
    return False

  @property
  def supports_cooling(self) -> bool:
    return False

  @property
  def supports_active_cooling(self) -> bool:
    return self.supports_cooling

  @property
  def temperature_range(self) -> Tuple[Optional[float], Optional[float]]:
    max_temp = 45.0 if self.supports_heating else None
    min_temp = 4.0 if self.supports_cooling else None
    return (min_temp, max_temp)

  async def _purge_buffers(self) -> None:
    """Clear pending adapter data before an RDOP22 command."""
    self._receive_buffer.clear()
    if isinstance(self.io, Serial):
      await self.io.reset_input_buffer()
      await self.io.reset_output_buffer()
      return
    for _ in range(6):
      await self.io.usb_purge_rx_buffer()
    await self.io.usb_purge_tx_buffer()

  async def _require_rdop22(self) -> None:
    """Reject legacy readers before sending RDOP22 geometry or assay commands."""
    if (await self.query_capability(2000))[:1] != b"1":
      raise NotImplementedError(
        "This reader reports no RDOP22 support (capability 2000). "
        "Legacy BioTek V/S reads are not implemented by this driver."
      )

  async def set_slow_mode(self, slow: bool) -> None:
    """Select carrier speed when supported; ordinary movement needs no extension."""
    async with self._operation():
      if self._slow_mode == slow:
        return
      if (await self.query_capability(2138))[:1] != b"1":
        if slow:
          raise NotImplementedError("This BioTek reader does not support carrier speed selection")
        self._slow_mode = False
        return
      response = await self.send_command("&", "S1" if slow else "S0")
      parse_response(response, "& S")
      self._slow_mode = slow

  def clear_plate(self) -> None:
    """Forget the plate geometry the firmware last had loaded (e.g. after closing the tray)."""
    self._plate = None

  async def open(self, slow: bool = False):
    """Drive the loading tray out.

    Args:
      slow: Move at the reduced carrier speed.
    """
    async with self._operation():
      await self.set_slow_mode(slow)
      await super().open()

  async def close(self, plate: Optional[Plate] = None, slow: bool = False):
    """Drive the loading tray in.

    Args:
      plate: Labware being carried in. Its geometry is sent to the firmware before the close
        motion so the carrier accounts for the labware height; without it a tall plate can jam
        the tray.
      slow: Move at the reduced carrier speed.
    """
    # Closing invalidates whatever plate geometry the firmware last had loaded.
    async with self._operation():
      self.clear_plate()
      await self.set_slow_mode(slow)
      if plate is not None:
        await self.set_plate(plate)
      await super().close()

  async def home(self) -> Optional[bytes]:
    """Home the reader axes through the imager-controller command."""
    return await self.send_command("i", "x")

  async def set_temperature(self, temperature: float):
    if not self.supports_heating and not self.supports_cooling:
      raise NotImplementedError(f"{self.__class__.__name__} does not support temperature control.")

    tmin, tmax = self.temperature_range
    if (tmin is not None and temperature < tmin) or (tmax is not None and temperature > tmax):
      raise ValueError(
        f"{self.__class__.__name__}: "
        f"Requested temperature {temperature}°C is outside supported range "
        f"{tmin}-{tmax}°C"
      )
    async with self._operation():
      current_temperature = await self.request_current_temperature()
      if temperature < current_temperature and not self.supports_cooling:
        raise ValueError(f"{self.__class__.__name__}: Cooling is not supported.")
      if temperature > current_temperature and not self.supports_heating:
        raise ValueError(f"{self.__class__.__name__}: Heating is not supported.")

      return await self.send_command("g", f"{int(temperature * 1000):05}")

  async def stop_heating_or_cooling(self):
    return await self.send_command("g", "00000")

  async def deactivate(self):
    return await self.stop_heating_or_cooling()

  def _parse_body(self, body: bytes) -> List[List[Optional[float]]]:
    assert self._plate is not None, "Plate must be set before reading data"
    plate = self._plate
    start_index = 22
    end_index = body.rindex(b"\r\n")
    num_rows = plate.num_items_y
    rows = body[start_index:end_index].split(b"\r\n,")[:num_rows]

    parsed_data: Dict[Tuple[int, int], float] = {}
    for row in rows:
      values = row.split(b",")
      grouped_values = [values[i : i + 3] for i in range(0, len(values), 3)]

      for group in grouped_values:
        assert len(group) == 3
        row_index = int(group[0].decode()) - 1
        column_index = int(group[1].decode()) - 1
        raw_value = group[2].decode()
        value = float("nan") if "*" in raw_value else float(raw_value)
        parsed_data[(row_index, column_index)] = value

    result: List[List[Optional[float]]] = [
      [None for _ in range(plate.num_items_x)] for _ in range(plate.num_items_y)
    ]
    for (row_idx, col_idx), value in parsed_data.items():
      result[row_idx][col_idx] = value
    return result

  async def set_plate(self, plate: Plate) -> Optional[bytes]:
    async with self._operation():
      if plate is self._plate:
        return None
      await self._require_rdop22()

      rows = plate.num_items_y
      columns = plate.num_items_x

      bottom_right_well = plate.get_item(plate.num_items - 1)
      assert bottom_right_well.location is not None
      bottom_right_well_center = bottom_right_well.location + bottom_right_well.get_anchor(
        x="c", y="c"
      )
      top_left_well = plate.get_item(0)
      assert top_left_well.location is not None
      top_left_well_center = top_left_well.location + top_left_well.get_anchor(x="c", y="c")

      plate_size_y = plate.get_size_y()
      plate_size_x = plate.get_size_x()
      plate_size_z = plate.get_size_z()
      if plate.lid is not None:
        plate_size_z += plate.lid.get_size_z() - plate.lid.nesting_z_height

      top_left_well_center_y = plate.get_size_y() - top_left_well_center.y
      bottom_right_well_center_y = plate.get_size_y() - bottom_right_well_center.y

      cmd = (
        f"{rows:02}"
        f"{columns:02}"
        f"{int(top_left_well_center_y * 100):05}"
        f"{int(bottom_right_well_center_y * 100):05}"
        f"{int(top_left_well_center.x * 100):05}"
        f"{int(bottom_right_well_center.x * 100):05}"
        f"{int(plate_size_y * 100):05}"
        f"{int(plate_size_x * 100):05}"
        f"{int(plate_size_z * 100):04}"
        "\x03"
      )

      resp = await self.send_command("y", cmd, timeout=20)
      parse_response(resp, "y")
      self._plate = plate
      return resp

  def _get_min_max_row_col_tuples(
    self, wells: List[Well], plate: Plate
  ) -> List[Tuple[int, int, int, int]]:
    plates = set(well.parent for well in wells)
    if len(plates) != 1 or plates.pop() != plate:
      raise ValueError("All wells must be in the specified plate")
    return self._non_overlapping_rectangles((well.get_row(), well.get_column()) for well in wells)

  async def _temperature_after_read(self) -> Optional[float]:
    """Read optional result metadata, closing a timed-out stream before returning."""
    try:
      return await self.request_current_temperature()
    except TimeoutError:
      await self._disconnect()
      return None

  async def _prepare_plate_read(self) -> None:
    """Allow an integrated microscope to release optics before a plate-reader assay."""

  async def read_absorbance(
    self,
    plate: Plate,
    wells: List[Well],
    wavelength: int,
  ) -> List[AbsorbanceResult]:
    """Read endpoint absorbance using the reader's negotiated RDOP22 layout.

    Capabilities 2056 and 2064 select the optical record fields. Legacy readers
    that report no RDOP22 support are rejected before geometry or assay setup.
    """
    min_abs, max_abs = self.abs_wavelength_range
    if not (min_abs <= wavelength <= max_abs):
      raise ValueError(f"{self.__class__.__name__}: wavelength must be within {min_abs}-{max_abs}")

    rectangles = self._get_min_max_row_col_tuples(wells, plate)
    async with self._operation():
      await self._require_rdop22()
      extra_read_field = (await self.query_capability(2056))[:1] == b"1"
      extended_optics = (await self.query_capability(2064))[:1] == b"1"

      logger.info(
        "[BioTek %s] read_absorbance: plate=%s, wavelength=%dnm, wells=%d",
        self.io.port if isinstance(self.io, Serial) else self.io.device_id,
        plate.name,
        wavelength,
        len(wells),
      )
      await self.set_plate(plate)
      await self._prepare_plate_read()

      all_data: List[List[Optional[float]]] = [
        [None for _ in range(plate.num_items_x)] for _ in range(plate.num_items_y)
      ]

      for min_row, min_col, max_row, max_col in rectangles:
        cmd = encode_absorbance(
          bounds=(min_row + 1, min_col + 1, max_row + 1, max_col + 1),
          wavelength=wavelength,
          extended_optics=extended_optics,
          extra_read_field=extra_read_field,
        )
        parse_assay_response(await self.send_command("D", cmd))

        resp = await self.send_command("O")
        parse_response(resp, "O")
        if resp != b"\x060000\x03":
          raise ValueError(f"BioTek O: expected ACK and status, received {resp!r}")

        body = await self._read_until(b"\x03", timeout=60 * 3)
        assert body is not None
        parsed_data = self._parse_body(body)
        for r in range(plate.num_items_y):
          for c in range(plate.num_items_x):
            if parsed_data[r][c] is not None:
              all_data[r][c] = parsed_data[r][c]

      temp = await self._temperature_after_read()

      return [
        AbsorbanceResult(
          wavelength=wavelength,
          data=all_data,
          temperature=temp,
          timestamp=datetime.now(),
        )
      ]

  async def read_luminescence(
    self,
    plate: Plate,
    wells: List[Well],
    focal_height: float,
    integration_time: float = 1,
    *,
    gain: int = 135,
    dynamic_range: Literal["standard", "extended"] = "extended",
  ) -> List[LuminescenceResult]:
    """Read luminescence.

    Args:
      plate: Plate to read.
      wells: Wells to read.
      focal_height: Focal height in mm.
      integration_time: Integration time in seconds, from 0.2 to 60 in steps of 0.2.
      gain: PMT sensitivity, 0 or an integer from 25 to 255.
      dynamic_range: Standard or extended result range.
    """
    min_fh, max_fh = self.focal_height_range
    if not (min_fh <= focal_height <= max_fh):
      raise ValueError(f"{self.__class__.__name__}: focal height must be within {min_fh}-{max_fh}")

    if not math.isfinite(integration_time) or not 0.2 <= integration_time <= 60:
      raise ValueError("Integration time must be between 0.2 and 60 seconds")
    if not math.isclose(integration_time * 5, round(integration_time * 5), abs_tol=1e-8):
      raise ValueError("Integration time must be a multiple of 0.2 seconds")
    integration_ticks = round(integration_time * 50)
    validate_detector_settings(gain, dynamic_range)

    async with self._operation():
      logger.info(
        "[BioTek %s] read_luminescence: plate=%s, wells=%d",
        self.io.port if isinstance(self.io, Serial) else self.io.device_id,
        plate.name,
        len(wells),
      )
      await self.set_plate(plate)
      await self._prepare_plate_read()

      cmd = f"3{14220 + int(1000 * focal_height)}\x03"
      parse_response(await self.send_command("t", cmd), "t")

      all_data: List[List[Optional[float]]] = [
        [None for _ in range(plate.num_items_x)] for _ in range(plate.num_items_y)
      ]
      for min_row, min_col, max_row, max_col in self._get_min_max_row_col_tuples(wells, plate):
        cmd = encode_luminescence(
          bounds=(min_row + 1, min_col + 1, max_row + 1, max_col + 1),
          integration_ticks=integration_ticks,
          gain=gain,
          dynamic_range=dynamic_range,
        )
        parse_assay_response(await self.send_command("D", cmd))

        resp = await self.send_command("O")
        parse_response(resp, "O")
        if resp != b"\x060000\x03":
          raise ValueError(f"BioTek O: expected ACK and status, received {resp!r}")

        timeout = 60 + integration_time * (2 * 60 + 10)
        body = await self._read_until(b"\x03", timeout=timeout)
        assert body is not None
        parsed_data = self._parse_body(body)
        for r in range(plate.num_items_y):
          for c in range(plate.num_items_x):
            if parsed_data[r][c] is not None:
              all_data[r][c] = parsed_data[r][c]

      temp = await self._temperature_after_read()

      return [
        LuminescenceResult(
          data=all_data,
          temperature=temp,
          timestamp=datetime.now(),
        )
      ]

  async def _read_fluorescence(
    self,
    plate: Plate,
    wells: List[Well],
    excitation_wavelength: int,
    emission_wavelength: int,
    focal_height: float,
    delay_microseconds: int = 0,
    collection_microseconds: int = 0,
    *,
    gain: int = 100,
    dynamic_range: Literal["standard", "extended"] = "extended",
    read_from: Literal["top", "bottom"] = "top",
    excitation_bandwidth: int = 20,
    emission_bandwidth: int = 20,
    flashes: int = 10,
    lamp_energy: Literal["low", "high"] = "high",
  ) -> List[FluorescenceResult]:
    """Execute a monochromator endpoint read with optional hardware-gated timing."""
    validate_fluorescence_settings(
      gain, dynamic_range, read_from, excitation_bandwidth, emission_bandwidth, flashes, lamp_energy
    )
    min_fh, max_fh = self.focal_height_range
    if not (min_fh <= focal_height <= max_fh):
      raise ValueError(f"{self.__class__.__name__}: focal height must be within {min_fh}-{max_fh}")

    min_ex, max_ex = self.excitation_range
    if not (min_ex <= excitation_wavelength <= max_ex):
      raise ValueError(
        f"{self.__class__.__name__}: excitation wavelength must be {min_ex}-{max_ex}"
      )

    min_em, max_em = self.emission_range
    if not (min_em <= emission_wavelength <= max_em):
      raise ValueError(f"{self.__class__.__name__}: emission wavelength must be {min_em}-{max_em}")

    async with self._operation():
      logger.info(
        "[BioTek %s] read_fluorescence: plate=%s, excitation=%dnm, emission=%dnm, wells=%d",
        self.io.port if isinstance(self.io, Serial) else self.io.device_id,
        plate.name,
        excitation_wavelength,
        emission_wavelength,
        len(wells),
      )
      if excitation_bandwidth != 20 or emission_bandwidth != 20:
        capability = await self.query_capability(2108)
        if capability[:1] != b"1":
          raise NotImplementedError("This reader does not support variable monochromator bandwidth")
        minimum = 3 if capability[1:2] == b"2" else 9
        if min(excitation_bandwidth, emission_bandwidth) < minimum:
          raise ValueError(f"This reader requires bandwidths between {minimum} and 50 nm")
      await self.set_plate(plate)
      await self._prepare_plate_read()

      if read_from == "top":
        cmd = f"{614220 + int(1000 * focal_height)}\x03"
        parse_response(await self.send_command("t", cmd), "t")

      all_data: List[List[Optional[float]]] = [
        [None for _ in range(plate.num_items_x)] for _ in range(plate.num_items_y)
      ]
      for min_row, min_col, max_row, max_col in self._get_min_max_row_col_tuples(wells, plate):
        cmd = encode_fluorescence(
          bounds=(min_row + 1, min_col + 1, max_row + 1, max_col + 1),
          excitation_wavelength=excitation_wavelength,
          emission_wavelength=emission_wavelength,
          delay_microseconds=delay_microseconds,
          collection_microseconds=collection_microseconds,
          gain=gain,
          dynamic_range=dynamic_range,
          read_from=read_from,
          excitation_bandwidth=excitation_bandwidth,
          emission_bandwidth=emission_bandwidth,
          flashes=flashes,
          lamp_energy=lamp_energy,
        )
        parse_assay_response(await self.send_command("D", cmd))

        resp = await self.send_command("O")
        parse_response(resp, "O")
        if resp != b"\x060000\x03":
          raise ValueError(f"BioTek O: expected ACK and status, received {resp!r}")

        body = await self._read_until(b"\x03", timeout=60 * 2)
        assert body is not None
        parsed_data = self._parse_body(body)
        for r in range(plate.num_items_y):
          for c in range(plate.num_items_x):
            if parsed_data[r][c] is not None:
              all_data[r][c] = parsed_data[r][c]

      temp = await self._temperature_after_read()

      return [
        FluorescenceResult(
          excitation_wavelength=excitation_wavelength,
          emission_wavelength=emission_wavelength,
          data=all_data,
          temperature=temp,
          timestamp=datetime.now(),
        )
      ]

  async def read_fluorescence(
    self,
    plate: Plate,
    wells: List[Well],
    excitation_wavelength: int,
    emission_wavelength: int,
    focal_height: float,
    *,
    gain: int = 100,
    dynamic_range: Literal["standard", "extended"] = "extended",
    read_from: Literal["top", "bottom"] = "top",
    excitation_bandwidth: int = 20,
    emission_bandwidth: int = 20,
    flashes: int = 10,
    lamp_energy: Literal["low", "high"] = "high",
  ) -> List[FluorescenceResult]:
    """Read endpoint fluorescence through the monochromator optical path.

    Args:
      plate: Plate to read.
      wells: Wells to read.
      excitation_wavelength: Excitation wavelength in nm.
      emission_wavelength: Emission wavelength in nm.
      focal_height: Top-probe height in mm; ignored for bottom reads.
      gain: PMT sensitivity, 0 or an integer from 25 to 255. Compare results only
        with the same gain and range settings.
      dynamic_range: Standard or extended result range.
      read_from: Top or bottom monochromator optics.
      excitation_bandwidth: Excitation bandwidth in nm; 9–50 on this Cytation 5.
      emission_bandwidth: Emission bandwidth in nm; 9–50 on this Cytation 5.
      flashes: Excitation flashes per well, from 1 to 255.
      lamp_energy: Low or high excitation flash energy.
    """
    return await self._read_fluorescence(
      plate,
      wells,
      excitation_wavelength,
      emission_wavelength,
      focal_height,
      gain=gain,
      dynamic_range=dynamic_range,
      read_from=read_from,
      excitation_bandwidth=excitation_bandwidth,
      emission_bandwidth=emission_bandwidth,
      flashes=flashes,
      lamp_energy=lamp_energy,
    )

  async def _abort(self) -> None:
    """Abort an active cycle and consume its DLE/status completion frame."""
    await self.send_command("x", wait_for_response=False)
    response = await self._read_until(b"\x03", timeout=30)
    if len(response) != 6 or response[:1] != b"\x10":
      raise ValueError(f"BioTek abort: expected DLE and status, received {response!r}")
    self._check_status(response[-5:])

  class ShakeType(enum.IntEnum):
    LINEAR = 0
    ORBITAL = 1

  async def _start_shaking_cycle(self, shake_type: ShakeType, frequency: int) -> None:
    """Start one firmware-limited sixteen-minute shaking cycle."""
    async with self._operation():
      await self._prepare_plate_read()
      cmd = f"0033010101010100002000000013960{shake_type.value}{frequency}01"
      checksum = (sum(cmd.encode()) + 73) % 100
      parse_response(await self.send_command("D", f"{cmd}{checksum:02d}\x03"), "D")
      response = await self.send_command("O")
      parse_response(response, "O")
      if response != b"\x060000\x03":
        raise ValueError(f"BioTek O: expected ACK and status, received {response!r}")

  async def shake(self, shake_type: ShakeType, frequency: int) -> None:
    """Start continuous shaking, reporting startup errors to the caller.

    Args:
      shake_type: Linear or orbital shaking.
      frequency: Speed setting: 6 = 360 CPM, 5 = 410 CPM, 4 = 493 CPM,
        3 = 567 CPM, 2 = 731 CPM, 1 = 1096 CPM.
    """
    if not isinstance(shake_type, self.ShakeType):
      raise ValueError("shake_type must be a ShakeType")
    if isinstance(frequency, bool) or not isinstance(frequency, int) or not 1 <= frequency <= 6:
      raise ValueError("Frequency must be an integer between 1 and 6")

    async def renew_shaking() -> None:
      """Renew the firmware cycle until stopped, logging later connection failures."""
      try:
        while self._shaking:
          await asyncio.sleep(16 * 60)
          if self._shaking:
            await self._start_shaking_cycle(shake_type, frequency)
      except Exception:
        self._shaking = False
        logger.exception("Continuous shaking stopped after a reader error")

    async with self._operation():
      await self.stop_shaking()
      await self._start_shaking_cycle(shake_type, frequency)
      self._shaking = True
      self._shaking_task = asyncio.create_task(renew_shaking())

  async def stop_shaking(self) -> None:
    """Abort shaking and release the background task under the shared connection lock."""
    async with self._operation():
      if self._shaking:
        await self._abort()
        self._shaking = False
      if self._shaking_task is not None:
        task = self._shaking_task
        self._shaking_task = None
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

  async def _before_stop(self) -> None:
    """Stop background shaking before closing or reopening the connection."""
    await self.stop_shaking()
