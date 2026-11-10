"""Decode the Cytation 5's fixed-width optical configuration records."""

import math
from dataclasses import dataclass
from typing import Optional

from .models import Objective


def parse_position_digits(capability: bytes) -> int:
  """Decode capability 2102, the width of signed motor position fields."""
  if len(capability) != 17:
    raise ValueError(f"Malformed Cytation position capability: {capability!r}")
  if capability[:1] == b"0":
    return 5
  if capability[:1] != b"1" or capability[1:2] not in b"56789":
    raise ValueError(f"Unsupported Cytation position capability: {capability!r}")
  return int(capability[1:2])


def parse_focus_resolution(capability: bytes) -> float:
  """Decode capability 2098 as millimeters per fine-focus motor step."""
  if len(capability) != 17 or capability[:1] != b"1":
    raise ValueError(f"Missing Cytation focus capability: {capability!r}")
  mode = capability[1:2]
  if mode in (b"0", b"1"):
    field = capability[3:9]
  elif mode == b"2":
    field = capability[3:12]
  elif mode == b"4":
    field = capability[10:16]
  else:
    raise ValueError(f"Unsupported Cytation focus capability: {capability!r}")
  try:
    resolution = float(field.rstrip(b"\x00 ")) / 1000
  except ValueError as exc:
    raise ValueError(f"Malformed Cytation focus resolution: {field!r}") from exc
  if not math.isfinite(resolution) or resolution <= 0:
    raise ValueError(f"Invalid Cytation focus resolution: {field!r}")
  return resolution


_OBJECTIVES = {
  "uplsapo 40x2": Objective.O_40X_PL_APO,
  "lucplfln 60x": Objective.O_60X_PL_FL,
  "uplfln 4x": Objective.O_4X_PL_FL,
  "lucplfln 20xph": Objective.O_20X_PL_FL_Phase,
  "lucplfln 40xph": Objective.O_40X_PL_FL_Phase,
  "u plan": Objective.O_2_5X_PL_ACH_Meiji,
  "uplfln 10xph": Objective.O_10X_PL_FL_Phase,
  "plapon 1.25x": Objective.O_1_25X_PL_APO,
  "uplfln 10x": Objective.O_10X_PL_FL,
  "uplfln 60xoi": Objective.O_60X_OIL_PL_FL,
  "pln 4x": Objective.O_4X_PL_ACH,
  "pln 40x": Objective.O_40X_PL_ACH,
  "lucplfln 40x": Objective.O_40X_PL_FL,
  "ec-h-plan/2x": Objective.O_2X_PL_ACH_Motic,
  "uplfln 100xo2": Objective.O_100X_OIL_PL_FL,
  "uplfln 4xph": Objective.O_4X_PL_FL_Phase,
  "lucplfln 20x": Objective.O_20X_PL_FL,
  "pln 20x": Objective.O_20X_PL_ACH,
  "fluar 2.5x/0.12": Objective.O_2_5X_FL_Zeiss,
  "uplsapo 100xo": Objective.O_100X_OIL_PL_APO,
  "plapon 60xo": Objective.O_60X_OIL_PL_APO,
  "uplsapo 20x": Objective.O_20X_PL_APO,
}


def _decode_label(field: bytes) -> str:
  """Decode decimal character codes relative to ASCII space, padded with spaces."""
  encoded = field.rstrip(b" ")
  if len(encoded) % 2 or (encoded and not encoded.isdigit()):
    raise ValueError(f"Malformed Cytation objective label: {field!r}")
  return "".join(chr(int(encoded[i : i + 2]) + 32) for i in range(0, len(encoded), 2))


@dataclass(frozen=True)
class CytationObjectiveConfiguration:
  """An objective turret slot as configured in the instrument.

  A configured slot is not physical proof that the objective is currently fitted.
  Unknown part numbers retain their original labels and have ``objective=None``.
  An empty turret slot is represented separately by ``None``.
  """

  objective: Optional[Objective]
  manufacturer: str
  part_number: str
  magnification: float
  numerical_aperture: float
  phase_ring_position: int


def parse_objective_configuration(payload: bytes) -> Optional[CytationObjectiveConfiguration]:
  """Parse the 75-byte data field returned by ``i o<slot>`` (without status/ETX)."""
  if len(payload) != 75 or not payload[:10].isdigit() or not payload[74:].isdigit():
    raise ValueError(f"Malformed Cytation objective configuration: {payload!r}")
  manufacturer = _decode_label(payload[10:42])
  part_number = _decode_label(payload[42:74])
  magnification = int(payload[:5]) / 1000
  numerical_aperture = int(payload[5:10]) / 1000
  phase_ring_position = int(payload[74:])
  if magnification == 0:
    if manufacturer or part_number or numerical_aperture or phase_ring_position:
      raise ValueError(f"Inconsistent empty Cytation objective configuration: {payload!r}")
    return None
  if not manufacturer or not part_number:
    raise ValueError(f"Missing Cytation objective identification: {payload!r}")
  return CytationObjectiveConfiguration(
    objective=_OBJECTIVES.get(part_number.lower()),
    manufacturer=manufacturer,
    part_number=part_number,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    phase_ring_position=phase_ring_position,
  )
