"""Optical discovery using replies recorded from Cytation 5 firmware 1.12."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from .configuration import (
  parse_focus_resolution,
  parse_objective_configuration,
  parse_position_digits,
)
from .microscope import CytationMicroscope
from .models import ImagingMode, Objective


class TestOpticalConfiguration(unittest.IsolatedAsyncioTestCase):
  """Keep objective slots, condenser positions, and filter positions distinct."""

  def setUp(self) -> None:
    """Load captured query responses without connecting to hardware."""
    records = json.loads(
      (Path(__file__).parent / "test_data" / "cytation5_1_12_optics.json").read_text()
    )
    self.responses = {item["parameter"]: bytes.fromhex(item["response_hex"]) for item in records}
    self.driver = MagicMock()
    self.driver.version = "1.12"
    self.driver.send_command = AsyncMock(
      side_effect=lambda command, parameter: self.responses[parameter]
    )
    self.microscope = CytationMicroscope(self.driver, use_cam=False)

  async def test_reads_six_objective_records_on_cytation5_firmware_one(self) -> None:
    """The firmware major version does not identify the Cytation model."""
    await self.microscope._load_objectives()
    self.assertEqual(
      self.microscope.objectives,
      [
        Objective.O_4X_PL_FL_Phase,
        Objective.O_10X_PL_FL_Phase,
        Objective.O_20X_PL_FL_Phase,
        None,
        Objective.O_4X_PL_ACH,
        None,
      ],
    )
    self.assertEqual(
      [call.args for call in self.driver.send_command.await_args_list],
      [("i", f"o{slot}") for slot in range(1, 7)],
    )
    self.assertEqual(
      [
        item.phase_ring_position if item is not None else None
        for item in self.microscope.objective_configurations
      ],
      [2, 3, 3, None, 1, None],
    )

  async def test_reads_fluorescence_filter_slots_without_shifting_laser_slot(self) -> None:
    """The laser cube does not displace fluorescence cube indices."""
    await self.microscope._load_filters()
    self.assertEqual(
      self.microscope.filters,
      [None, ImagingMode.DAPI, ImagingMode.RFP, ImagingMode.TEXAS_RED],
    )
    self.assertEqual(self.microscope._imaging_mode_code(ImagingMode.DAPI), 2)

  def test_objective_record_preserves_optical_parameters(self) -> None:
    """Decode labels and optical values separately from the trailing status."""
    config = parse_objective_configuration(self.responses["o2"][:-5])
    assert config is not None
    self.assertEqual(config.manufacturer, "OLYMPUS")
    self.assertEqual(config.part_number, "UPLFLN 10XPH")
    self.assertEqual(config.magnification, 10)
    self.assertEqual(config.numerical_aperture, 0.3)
    self.assertEqual(config.phase_ring_position, 3)

  def test_truncated_record_and_invalid_labels_are_rejected(self) -> None:
    """Do not silently classify malformed inventory as an empty turret slot."""
    payload = self.responses["o1"][:-5]
    for invalid in (payload[:-1], payload + b"0", payload[:42] + b"ZZ" + payload[44:]):
      with self.subTest(payload=invalid), self.assertRaises(ValueError):
        parse_objective_configuration(invalid)

  def test_unknown_objective_retains_its_identification(self) -> None:
    """Unrecognized optics remain inspectable instead of looking like an empty slot."""
    payload = self.responses["o1"][:-5]
    config = parse_objective_configuration(payload[:42] + b"33" + b" " * 30 + payload[74:])
    assert config is not None
    self.assertIsNone(config.objective)
    self.assertEqual(config.part_number, "A")

  def test_focus_resolution_uses_firmware_units(self) -> None:
    """The microscope reports micrometers per step, while PLR uses millimeters."""
    self.assertAlmostEqual(parse_focus_resolution(b"11,00.094\x000000000"), 0.000094)

  def test_position_width_is_read_from_capability(self) -> None:
    """Motor width is a firmware capability, not a model/version heuristic."""
    self.assertEqual(parse_position_digits(b"16000000000000000"), 6)
    self.assertEqual(parse_position_digits(b"17000000000000000"), 7)
    self.assertEqual(parse_position_digits(b"00000000000000000"), 5)
    with self.assertRaises(ValueError):
      parse_position_digits(b"1x000000000000000")

  def test_invalid_focus_resolution_is_rejected(self) -> None:
    """Never substitute an arbitrary conversion for missing calibration."""
    for reply in (b"00000000000000000", b"11,00.000\x000000000", b"11,notnum\x000000000"):
      with self.subTest(reply=reply), self.assertRaises(ValueError):
        parse_focus_resolution(reply)
