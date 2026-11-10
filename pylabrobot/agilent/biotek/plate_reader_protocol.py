"""Framing and endpoint assay records for the BioTek RDOP22 protocol."""

from typing import Literal, Optional, Tuple


def parse_response(response: Optional[bytes], command: str) -> bytes:
  """Check the hexadecimal status of a reply and return its data, excluding ACK."""
  if response is None or len(response) < 5 or response[-1:] != b"\x03":
    raise ValueError(f"BioTek {command}: incomplete response: {response!r}")
  status = response[-5:-1]
  if any(c not in b"0123456789ABCDEFabcdef" for c in status):
    raise ValueError(f"BioTek {command}: invalid status: {status!r}")
  if int(status, 16) != 0:
    raise RuntimeError(f"BioTek {command}: instrument status 0x{status.decode()}")
  data = response[:-5]
  return data[1:] if data.startswith(b"\x06") else data


def parse_assay_response(response: Optional[bytes]) -> bytes:
  """Validate a D reply: length3 + data + checksum3 + status4 + ETX.

  The length counts everything after the length field, including status and ETX.
  The checksum is the unsigned byte sum of the data alone.
  """
  framed = parse_response(response, "D")
  if len(framed) < 6 or not framed[:3].isdigit() or not framed[-3:].isdigit():
    raise ValueError("BioTek D: invalid length/checksum fields")
  if int(framed[:3]) != len(framed) + 2:
    raise ValueError("BioTek D: response length mismatch")
  data = framed[3:-3]
  if int(framed[-3:]) != sum(data) % 256:
    raise ValueError("BioTek D: response checksum mismatch")
  return data


def encode_assay(body: str) -> str:
  """Frame a D assay with length4 and checksum3; no ETX follows the checksum."""
  length = len(body) + 3
  if length > 9999:
    raise ValueError("BioTek assay exceeds the four-digit length field")
  checksum = sum(body.encode("ascii")) % 256
  return f"{length:04}{body}{checksum:03}"


def encode_absorbance(
  bounds: Tuple[int, int, int, int],
  wavelength: int,
  extended_optics: bool,
  extra_read_field: bool,
) -> str:
  """Encode a single-wavelength endpoint assay with one-based, inclusive bounds.

  Capability 2064 controls the optical-path byte, and 2056 controls an extra
  three-digit read field. These capabilities are independent of firmware major
  version. The record selects normal plate mode and eight flashes.
  """
  min_row, min_col, max_row, max_col = bounds
  if not (1 <= min_row <= max_row <= 99 and 1 <= min_col <= max_col <= 99):
    raise ValueError("BioTek assay bounds must be ordered and between 1 and 99")
  if not 1 <= wavelength <= 9999:
    raise ValueError("BioTek wavelength must fit four decimal digits")
  header = f"01{min_row:02}{min_col:02}{max_row:02}{max_col:02}0001200100001"
  read_event = "1001000001"
  optics = "0" + ("6" if extended_optics else "") + "00008"
  optics += ("000" if extra_read_field else "") + f"{wavelength:04}"
  return encode_assay(header + read_event + optics)


def validate_detector_settings(gain: int, dynamic_range: Literal["standard", "extended"]) -> None:
  """Validate the PMT sensitivity and result range before any instrument movement."""
  if type(gain) is not int or (gain != 0 and not 25 <= gain <= 255):
    raise ValueError("Gain must be an integer: 0 or between 25 and 255")
  if dynamic_range not in ("standard", "extended"):
    raise ValueError("Dynamic range must be 'standard' or 'extended'")


def validate_fluorescence_settings(
  gain: int,
  dynamic_range: Literal["standard", "extended"],
  read_from: Literal["top", "bottom"],
  excitation_bandwidth: int,
  emission_bandwidth: int,
  flashes: int,
  lamp_energy: Literal["low", "high"],
) -> None:
  """Check monochromator settings; the device-specific bandwidth limit is separate."""
  validate_detector_settings(gain, dynamic_range)
  if read_from not in ("top", "bottom"):
    raise ValueError("Read direction must be 'top' or 'bottom'")
  for bandwidth in (excitation_bandwidth, emission_bandwidth):
    if type(bandwidth) is not int or not 3 <= bandwidth <= 50:
      raise ValueError("Bandwidth must be an integer between 3 and 50 nm")
  if type(flashes) is not int or not 1 <= flashes <= 255:
    raise ValueError("Flashes must be an integer between 1 and 255")
  if lamp_energy not in ("low", "high"):
    raise ValueError("Lamp energy must be 'low' or 'high'")


def encode_fluorescence(
  bounds: Tuple[int, int, int, int],
  excitation_wavelength: int,
  emission_wavelength: int,
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
) -> str:
  """Encode the monochromator endpoint layout, including gated flash measurements."""
  validate_fluorescence_settings(
    gain, dynamic_range, read_from, excitation_bandwidth, emission_bandwidth, flashes, lamp_energy
  )
  min_row, min_col, max_row, max_col = bounds
  if not (1 <= min_row <= max_row <= 99 and 1 <= min_col <= max_col <= 99):
    raise ValueError("BioTek assay bounds must be ordered and between 1 and 99")
  if not (1 <= excitation_wavelength <= 9999 and 1 <= emission_wavelength <= 9999):
    raise ValueError("BioTek wavelengths must fit four decimal digits")
  if not 0 <= delay_microseconds <= 2000:
    raise ValueError("Fluorescence delay must be 0–2000 microseconds")
  if collection_microseconds != 0 and not 20 <= collection_microseconds <= 2000:
    raise ValueError("Fluorescence collection must be 20–2000 microseconds")
  if delay_microseconds and not collection_microseconds:
    raise ValueError("A fluorescence delay requires a collection duration")
  header = f"01{min_row:02}{min_col:02}{max_row:02}{max_col:02}0001200100001"
  read_event = "1001000001"
  path = "5" if read_from == "top" else "4"
  energy = "2" if lamp_energy == "high" else "0"
  range_code = "1" if dynamic_range == "extended" else "0"
  optics = (
    f"3{path}0{flashes:04}{excitation_bandwidth:03}{emission_bandwidth:03}0"
    f"{excitation_wavelength:04}000{emission_wavelength:04}000"
    f"00000{delay_microseconds:05}{collection_microseconds:05}{energy}{gain:03}{range_code}"
  )
  return encode_assay(header + read_event + optics)


def encode_luminescence(
  bounds: Tuple[int, int, int, int],
  integration_ticks: int,
  *,
  gain: int = 135,
  dynamic_range: Literal["standard", "extended"] = "extended",
  emission_wavelength: Optional[int] = None,
  emission_bandwidth: int = 20,
) -> str:
  """Encode broadband or top-monochromator luminescence in 20 ms ticks."""
  min_row, min_col, max_row, max_col = bounds
  validate_detector_settings(gain, dynamic_range)
  if not (1 <= min_row <= max_row <= 99 and 1 <= min_col <= max_col <= 99):
    raise ValueError("BioTek assay bounds must be ordered and between 1 and 99")
  if not 1 <= integration_ticks <= 3000:
    raise ValueError("BioTek luminescence integration must be 1–3000 ticks")
  if emission_wavelength is not None and (
    type(emission_wavelength) is not int or not 1 <= emission_wavelength <= 9999
  ):
    raise ValueError("Emission wavelength must be an integer fitting four digits")
  if type(emission_bandwidth) is not int or not 3 <= emission_bandwidth <= 50:
    raise ValueError("Emission bandwidth must be an integer between 3 and 50 nm")
  if emission_wavelength is None and emission_bandwidth != 20:
    raise ValueError("Emission bandwidth requires a selected emission wavelength")
  header = f"01{min_row:02}{min_col:02}{max_row:02}{max_col:02}0001200100001"
  read_event = "1001000001"
  range_code = "1" if dynamic_range == "extended" else "0"
  path = "3" if emission_wavelength is None else "5"
  emission = -3 if emission_wavelength is None else emission_wavelength
  optics = (
    f"2{path}0{integration_ticks:04}020{emission_bandwidth:03}0-001000{emission:04}000"
    f"0000000000000000{gain:03}{range_code}"
  )
  return encode_assay(header + read_event + optics)
