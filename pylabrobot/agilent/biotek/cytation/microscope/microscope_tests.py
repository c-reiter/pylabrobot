import unittest
import unittest.mock
from typing import Any

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
  backend._focus_resolution = 0.000094
  backend._position_digits = 7
  return backend, driver


class TestFocus(unittest.IsolatedAsyncioTestCase):
  """Use the firmware's fixed-width position format at every focal height."""

  async def test_six_digit_firmware_uses_reported_position_width(self):
    """Cytation 5 firmware 1.12 uses six characters instead of seven."""
    backend, driver = _make_backend(version="1.12")
    backend._position_digits = 6
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    await backend.set_focus(7.75)
    driver.send_command.assert_awaited_once_with("i", "F5082447")

  async def test_focus_command_has_fixed_width(self):
    """Encode both sides of the six-digit count boundary as complete frames."""
    backend, driver = _make_backend()
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    for height, command in (
      (4.5, "F50047872"),
      (7.0, "F50074468"),
      (9.4, "F50100000"),
      (13.5, "F50143617"),
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
        unittest.mock.call("i", "F50074468"),
        unittest.mock.call("Y", "P0e02", timeout=60),
        unittest.mock.call("i", "F50074468"),
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

  async def test_cytation5_firmware_one_uses_six_objective_motor_addresses(self):
    """Firmware 1.x is also used by Cytation 5 and must retain its optics commands."""
    for mode in (ImagingMode.COLOR_BRIGHTFIELD, ImagingMode.PHASE_CONTRAST, ImagingMode.DAPI):
      with self.subTest(mode=mode):
        backend, driver = _make_backend(version="1.12")
        backend._imaging_mode = None
        backend._objective = None
        backend._objectives = [Objective.O_4X_PL_ACH]
        backend._filters = [None, ImagingMode.DAPI, None, None]
        await backend.set_objective(Objective.O_4X_PL_ACH)
        await backend.set_imaging_mode(mode)
        driver.send_command.assert_any_await("Y", "P0e01", timeout=60)
        ring = "P1120" if mode == ImagingMode.PHASE_CONTRAST else "P1101"
        driver.send_command.assert_any_await("Y", ring)
        cube = "P0d02" if mode == ImagingMode.DAPI else "P0d05"
        driver.send_command.assert_any_await("Y", cube)


if __name__ == "__main__":
  unittest.main()


class TestStage(unittest.IsolatedAsyncioTestCase):
  """Stage targets retain motor-step precision and use firmware field widths."""

  async def test_offsets_quantize_targets_without_drift(self):
    backend, driver = _make_backend()
    backend._position_digits = 6
    await backend.select(1, 2)
    for x in (0.004, 0.008, 0.012, 0.016, 0.020, 0.0):
      await backend.set_position(x, 0)
    self.assertEqual(
      driver.send_command.await_args_list,
      [
        unittest.mock.call("Y", "W60102", timeout=50),
        unittest.mock.call("Y", "O00000001", timeout=50),
        unittest.mock.call("Y", "O00000001", timeout=50),
        unittest.mock.call("Y", "O00-00002", timeout=50),
      ],
    )
    self.assertEqual((backend._pos_x, backend._pos_y), (0, 0))

  async def test_partial_movement_invalidates_both_coordinates(self):
    backend, driver = _make_backend()
    await backend.select(1, 1)
    driver.send_command.side_effect = [b"0000\x03", RuntimeError("motor error")]
    with self.assertRaisesRegex(RuntimeError, "motor error"):
      await backend.set_position(1, 1)
    self.assertIsNone(backend._pos_x)
    self.assertIsNone(backend._pos_y)

  async def test_unknown_position_is_recentered(self):
    backend, driver = _make_backend()
    backend._row, backend._column = 1, 2
    await backend.set_position(0.01016, 0)
    self.assertEqual(
      driver.send_command.await_args_list,
      [
        unittest.mock.call("Y", "W60102", timeout=50),
        unittest.mock.call("Y", "O000000001", timeout=50),
      ],
    )

  async def test_invalid_targets_never_move_stage(self):
    backend, driver = _make_backend()
    for x in (float("nan"), float("inf"), 1e308):
      with self.assertRaises(ValueError):
        await backend.set_position(x, 0)
    for row, column in ((0, 1), (1, -1), (100, 1), (1.5, 1)):
      with self.assertRaises(ValueError):
        await backend.select(row, column)
    driver.send_command.assert_not_awaited()

  async def test_focus_calibration_query(self):
    backend, driver = _make_backend()
    backend._position_digits = 6
    backend._objectives = [Objective.O_4X_PL_FL_Phase]
    backend._objective = Objective.O_4X_PL_FL_Phase
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    driver.send_command.return_value = b"10824470000\x03"
    self.assertAlmostEqual(await backend.get_focus_reference(), 7.750018)
    driver.send_command.assert_awaited_once_with(
      "Y", "Z150" + "0000" + "000000" + "000000", timeout=20
    )
    driver.send_command.return_value = b"00824470000\x03"
    self.assertIsNone(await backend.get_focus_reference())


class TestAutomaticImaging(unittest.IsolatedAsyncioTestCase):
  """Bound automatic searches and never report an unmeasured focus or exposure."""

  async def test_auto_exposure_converges_using_frames(self):
    from .models import AutoExposure

    backend, _ = _make_backend()
    backend.camera = unittest.mock.MagicMock()
    backend.camera.get_exposure_limits.return_value = (0.1, 1000)
    backend.camera.set_auto_exposure = unittest.mock.AsyncMock()
    backend.camera.set_exposure = unittest.mock.AsyncMock()
    backend.get_exposure = unittest.mock.AsyncMock(side_effect=lambda: backend._exposure)
    backend.acquire_image = unittest.mock.AsyncMock()
    evaluate = unittest.mock.AsyncMock(side_effect=["higher", "lower", "good"])
    result = await backend.auto_expose(AutoExposure(evaluate, 3, 1, 100))
    self.assertAlmostEqual(result, 17.7827941004)
    self.assertEqual(evaluate.await_count, 3)
    self.assertFalse(backend._acquiring)

  async def test_auto_exposure_failure_stops_owned_acquisition(self):
    from .models import AutoExposure

    backend, _ = _make_backend()
    backend.camera = unittest.mock.MagicMock()
    backend.camera.get_exposure_limits.return_value = (0.1, 1000)
    backend.camera.set_auto_exposure = unittest.mock.AsyncMock()
    backend.camera.set_exposure = unittest.mock.AsyncMock()
    backend.acquire_image = unittest.mock.AsyncMock()
    evaluate = unittest.mock.AsyncMock(return_value="higher")
    with self.assertRaisesRegex(RuntimeError, "No acceptable exposure"):
      await backend.auto_expose(AutoExposure(evaluate, 3, 1, 100))
    self.assertEqual(evaluate.await_count, 3)
    self.assertFalse(backend._acquiring)

  async def test_autofocus_brackets_and_refines_peak(self):
    from .models import AutoFocus

    backend, _ = _make_backend()
    backend.camera = unittest.mock.MagicMock()
    backend._imaging_mode = ImagingMode.BRIGHTFIELD
    backend._exposure, backend._gain = 10, 0
    backend.acquire_image = unittest.mock.AsyncMock(side_effect=lambda: backend._focal_height)
    result = await backend.autofocus(
      AutoFocus(lambda z: float(-((z - 4.31) ** 2)), 10, 4, 5, 0.001)
    )
    self.assertAlmostEqual(result, 4.31, delta=0.001)
    self.assertEqual(backend._focal_height, result)
    self.assertFalse(backend._acquiring)

  async def test_autofocus_rejects_flat_and_out_of_range_peaks(self):
    from .models import AutoFocus

    for evaluate in (lambda z: 1, lambda z: z):
      backend, _ = _make_backend()
      backend.camera = unittest.mock.MagicMock()
      backend._imaging_mode = ImagingMode.BRIGHTFIELD
      backend._exposure, backend._gain = 10, 0
      backend.acquire_image = unittest.mock.AsyncMock(side_effect=lambda: backend._focal_height)
      with self.assertRaises(RuntimeError):
        await backend.autofocus(AutoFocus(evaluate, 10, 4, 5))
      self.assertFalse(backend._acquiring)


class TestMontage(unittest.IsolatedAsyncioTestCase):
  """Use rectangular camera dimensions and overlap for stage tiling."""

  def test_grasshopper_field_of_view(self):
    backend, _ = _make_backend()
    backend.camera = unittest.mock.MagicMock()
    backend.camera.width, backend.camera.height = 1384, 1036
    backend.camera.get_device_info.return_value.model_name = "Grasshopper3 GS3-U3-14S5M"
    width, height = backend.get_field_of_view(Objective.O_4X_PL_FL_Phase)
    self.assertAlmostEqual(width, 2.2317)
    self.assertAlmostEqual(height, 1.67055)

  def test_full_coverage_uses_rows_for_height_and_columns_for_width(self):
    grid, positions = CytationMicroscope._tile_positions("full", (6, 4), (2, 1), (0, 0), 0.25)
    self.assertEqual(grid, (5, 4))
    self.assertEqual(len(positions), 20)
    self.assertEqual(positions[0], (-2.25, 1.5))
    self.assertEqual(positions[-1], (2.25, -1.5))

  def test_invalid_overlap_and_coverage_are_rejected(self):
    for overlap in (-0.1, 1, float("nan")):
      with self.assertRaises(ValueError):
        CytationMicroscope._tile_positions((1, 2), (6, 6), (2, 1), (0, 0), overlap)
    invalid_coverages: Any = ((0, 1), (1, 2.5), (-1, 2))
    for coverage in invalid_coverages:
      with self.assertRaises(ValueError):
        CytationMicroscope._tile_positions(coverage, (6, 6), (2, 1), (0, 0), 0)

  async def test_capture_returns_grid_and_stops_illumination(self):
    from pylabrobot.resources import cor_96_wellplate_360uL_Fb

    backend, driver = _make_backend()
    driver.set_plate = unittest.mock.AsyncMock()
    backend._objectives = [Objective.O_4X_PL_FL_Phase]
    backend.camera = unittest.mock.MagicMock()
    backend.camera.width, backend.camera.height = 1384, 1036
    backend.camera.get_device_info.return_value.model_name = "Grasshopper3 GS3-U3-14S5M"
    backend.camera.set_auto_exposure = unittest.mock.AsyncMock()
    backend.camera.set_exposure = unittest.mock.AsyncMock()
    backend.camera.set_gain = unittest.mock.AsyncMock()
    backend.camera.get_exposure = unittest.mock.AsyncMock(return_value=10)
    backend.camera.trigger = unittest.mock.AsyncMock(return_value=np.ones((5, 5), dtype=np.uint8))
    result = await backend.capture(
      1,
      1,
      ImagingMode.BRIGHTFIELD,
      Objective.O_4X_PL_FL_Phase,
      10,
      3.28,
      0,
      cor_96_wellplate_360uL_Fb("plate"),
      coverage=(1, 2),
      overlap=0.2,
    )
    self.assertEqual(result.tile_grid, (1, 2))
    self.assertEqual(len(result.images), 2)
    self.assertEqual(len(result.tile_positions), 2)
    driver.send_command.assert_awaited_with("i", "L0001")
    backend.camera.stop_acquisition.assert_called_once()
