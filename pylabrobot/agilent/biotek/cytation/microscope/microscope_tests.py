import unittest
import unittest.mock

import pytest

pytest.importorskip("numpy")

import numpy as np

from pylabrobot.agilent.biotek.cytation.microscope.microscope import (
  COLOR_BRIGHTFIELD_LED_CODES,
  CytationMicroscope,
)
from pylabrobot.agilent.biotek.cytation.microscope.models import ImagingMode, Objective


def _make_backend(version: str = "2.07"):
  """A microscope with a fully mocked serial driver (no hardware)."""
  driver = unittest.mock.MagicMock()
  driver.version = version
  driver.send_command = unittest.mock.AsyncMock()
  backend = CytationMicroscope(driver=driver, use_cam=False)
  return backend, driver


class TestFocus(unittest.IsolatedAsyncioTestCase):
  """Focus positions occupy seven decimal digits, including below 100000 counts."""

  async def test_focus_command_has_fixed_width(self):
    """Encode both sides of the six-digit count boundary as complete frames."""
    backend, driver = _make_backend()
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    for height, command in (
      (4.5, "F50047876"),
      (7.0, "F50074473"),
      (9.4, "F50100007"),
      (13.5, "F50143627"),
    ):
      with self.subTest(height=height):
        await backend.set_focus(height)
        driver.send_command.assert_awaited_with("i", command)
        self.assertEqual(backend._focal_height, height)

  async def test_refused_focus_is_not_cached(self):
    """A refused move must be retried rather than skipped as already complete."""
    backend, driver = _make_backend()
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    driver.send_command.side_effect = RuntimeError("BioTek reported error 0x570F")
    for _ in range(2):
      with self.assertRaisesRegex(RuntimeError, "570F"):
        await backend.set_focus(7.0)
    self.assertIsNone(backend._focal_height)
    self.assertEqual(driver.send_command.await_count, 2)

  async def test_invalid_positions_do_not_send_commands(self):
    """Invalid numbers must never reach the motor's ASCII position field."""
    backend, driver = _make_backend()
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    for height in (-1, float("nan"), float("inf"), float("-inf"), 1e308, 1000):
      with self.subTest(height=height):
        with self.assertRaises(ValueError):
          await backend.set_focus(height)
    driver.send_command.assert_not_awaited()

  async def test_objective_change_requires_focus_and_position_to_be_reapplied(self):
    """Objective motion invalidates focus and within-well positioning caches."""
    backend, driver = _make_backend()
    backend._objectives = [Objective.O_4X_PL_FL, Objective.O_10X_PL_FL_Phase]
    backend._objective = Objective.O_4X_PL_FL
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    backend._pos_x, backend._pos_y = 0, 0
    await backend.set_focus(7.0)
    await backend.set_objective(Objective.O_10X_PL_FL_Phase)
    self.assertIsNone(backend._focal_height)
    self.assertIsNone(backend._pos_x)
    self.assertIsNone(backend._pos_y)
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    await backend.set_focus(7.0)
    self.assertEqual(
      driver.send_command.await_args_list,
      [
        unittest.mock.call("i", "F50074473"),
        unittest.mock.call("Y", "P0e02", timeout=60),
        unittest.mock.call("i", "F50074473"),
      ],
    )


class TestColorBrightfield(unittest.IsolatedAsyncioTestCase):
  """Color brightfield illuminates with three LEDs (codes 6/7/8) and stacks the frames."""

  def test_imaging_mode_code_uses_brightfield_light_path(self):
    backend, _ = _make_backend()
    # Color brightfield shares the brightfield filter cube (5); colors come from the LEDs.
    self.assertEqual(backend._imaging_mode_code(ImagingMode.COLOR_BRIGHTFIELD), 5)

  async def test_acquire_cycles_three_leds_and_stacks_rgb(self):
    backend, driver = _make_backend()
    frame = np.zeros((4, 5), dtype=np.uint8)
    backend.acquire_image = unittest.mock.AsyncMock(return_value=frame)

    image = await backend._acquire_color_brightfield_image(led_intensity=10)

    # One mono frame per color, stacked into an (H, W, 3) RGB image.
    self.assertEqual(image.shape, (4, 5, 3))
    self.assertEqual(backend.acquire_image.await_count, 3)

    # Per color, in cycle order: LED on (L0{code}10) then strobe (l{code}).
    expected = []
    for code in COLOR_BRIGHTFIELD_LED_CODES:
      expected.append(unittest.mock.call("i", f"L0{code}10"))
      expected.append(unittest.mock.call("i", f"l{code}"))
    self.assertEqual(driver.send_command.await_args_list, expected)

  async def test_set_imaging_mode_matches_brightfield_setup_without_static_led(self):
    backend, driver = _make_backend(version="2.07")
    backend._imaging_mode = None

    await backend.set_imaging_mode(ImagingMode.COLOR_BRIGHTFIELD, led_intensity=10)

    sent = [call.args for call in driver.send_command.await_args_list]
    # Same optics setup as plain brightfield (filter cube 5).
    for cmd in (("Y", "P1101"), ("Y", "P0d05"), ("Y", "P1002")):
      self.assertIn(cmd, sent)
    # No single LED is enabled (colors are cycled during acquisition): only led_off (L0001).
    self.assertEqual([p for (ch, p) in sent if ch == "i"], ["L0001"])
    self.assertEqual(backend._imaging_mode, ImagingMode.COLOR_BRIGHTFIELD)

  async def test_color_brightfield_not_supported_on_cytation1(self):
    backend, _ = _make_backend(version="1.05")
    backend._imaging_mode = None
    with self.assertRaises(NotImplementedError):
      await backend.set_imaging_mode(ImagingMode.COLOR_BRIGHTFIELD, led_intensity=10)


if __name__ == "__main__":
  unittest.main()
