"""Laser reflection estimation, current limits and bounded focus correction."""

import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

pytest.importorskip("numpy")

import numpy as np

from .laser_autofocus import CytationLaserAutofocus, LaserFocusCalibration, laser_spot_position
from .models import ImagingMode, Objective


def _laser():
  """Create a laser subsystem with mocked hardware and a known optical configuration."""
  m = MagicMock()
  m._objective = Objective.O_4X_PL_FL_Phase
  m._imaging_mode = ImagingMode.BRIGHTFIELD
  m._laser_slot = 1
  m.led_off = AsyncMock()
  m.set_focus = AsyncMock()
  m.driver._exchange = AsyncMock(return_value=b"\x060000")
  return CytationLaserAutofocus(m), m


class TestLaserSpot(unittest.TestCase):
  """Subpixel centroids must reject missing, clipped and isolated-pixel signals."""

  def test_centroid_ignores_background(self):
    y, x = np.indices((200, 200))
    image = (3 + 100 * np.exp(-((x - 91.3) ** 2 + (y - 103.7) ** 2) / 50)).astype(np.uint8)
    cx, cy = laser_spot_position(image, 255)
    self.assertAlmostEqual(cx, 91.3, delta=0.1)
    self.assertAlmostEqual(cy, 103.7, delta=0.1)

  def test_secondary_plate_reflection_does_not_bias_centroid(self):
    y, x = np.indices((100, 100))
    primary = 100 * np.exp(-((x - 50.2) ** 2 + (y - 60.3) ** 2) / 18)
    secondary = 60 * np.exp(-((x - 49) ** 2 + (y - 30) ** 2) / 40)
    image = (primary + secondary).astype(np.uint8)
    cx, cy = laser_spot_position(image, 255)
    self.assertAlmostEqual(cx, 50.2, delta=0.15)
    self.assertAlmostEqual(cy, 60.3, delta=0.15)

  def test_unusable_images_raise(self):
    dark = np.zeros((100, 100), dtype=np.uint8)
    saturated = dark.copy()
    saturated[50:60, 50:60] = 255
    hot = dark.copy()
    hot[50, 50] = 100
    for image in (dark, saturated, hot):
      with self.assertRaises(RuntimeError):
        laser_spot_position(image, 255)

  def test_broad_reflection_is_not_clipped_by_an_off_center_peak(self):
    """Speckle changes the brightest pixel without translating the whole reflection."""
    y, x = np.indices((140, 140))
    image = (100 * np.exp(-((x - 70.2) ** 2 / 72 + (y - 75.3) ** 2 / 288))).astype(np.uint8)
    image[65, 70] = 150
    cx, cy = laser_spot_position(image, 255)
    self.assertAlmostEqual(cx, 70.2, delta=0.1)
    self.assertAlmostEqual(cy, 75.3, delta=0.1)


class TestLaserControl(unittest.IsolatedAsyncioTestCase):
  """Verify current before diode enable and enforce a finite focus search."""

  async def test_current_feedback_precedes_laser_enable(self):
    laser, m = _laser()
    with patch.object(laser, "_read_current", new_callable=AsyncMock) as current:
      current.side_effect = [13, 12]
      await laser._enable(12)
    frames = [c.args[1] for c in m.driver._exchange.await_args_list]
    self.assertEqual(frames[:2], [b"020021005h0", b"020021005w1"])
    self.assertEqual(frames[-1], b"020021005l1")
    self.assertEqual(sum(b"c" in frame for frame in frames), 2)

  async def test_current_monitor_failure_never_enables_diode(self):
    laser, m = _laser()
    with patch.object(laser, "_read_current", new_callable=AsyncMock, return_value=0):
      with self.assertRaisesRegex(RuntimeError, "returned zero"):
        await laser._enable(12)
    self.assertFalse(any(c.args[1].endswith(b"l1") for c in m.driver._exchange.await_args_list))
    for current in (0, 30, float("nan")):
      with self.assertRaises(ValueError):
        await laser._enable(current)

  async def test_adc_framing_and_units(self):
    laser, m = _laser()
    m.driver._exchange.return_value = b"\x06 40940000"
    self.assertAlmostEqual(await laser._read_current(), 12.7939452201)
    m.driver._exchange.assert_awaited_with(b"\x0c", b"020021010a1", data_length=10, timeout=20)

  async def test_calibration_and_offset_correction(self):
    laser, m = _laser()
    position = 0.0

    async def move(z):
      nonlocal position
      position = z

    async def measure():
      return 660.0, 520.0 - 25.0 * position

    with (
      patch.object(laser, "_session", return_value=MagicMock()),
      patch.object(laser, "_set_focus", new_callable=AsyncMock, side_effect=move),
      patch.object(laser, "_measure", new_callable=AsyncMock, side_effect=measure),
    ):
      calibration = await laser.calibrate(3.28)
      self.assertEqual(calibration.slope, (0, -25))
      result = await laser.autofocus(start=3.43)
    self.assertAlmostEqual(result, 3.28)
    m.set_focus.assert_awaited_with(result)

  async def test_out_of_bounds_correction_stops_before_moving(self):
    laser, m = _laser()
    laser.calibration = LaserFocusCalibration(
      m._objective, m._imaging_mode, 3.28, (660, 438), (0, -25), 5, 12
    )
    with (
      patch.object(laser, "_session", return_value=MagicMock()),
      patch.object(laser, "_set_focus", new_callable=AsyncMock) as move,
      patch.object(laser, "_measure", new_callable=AsyncMock, return_value=(660, 400)),
    ):
      with self.assertRaisesRegex(RuntimeError, "travel bound"):
        await laser.autofocus(max_correction=0.5)
      self.assertEqual(move.await_args_list, [call(3.28)])

  async def test_laser_session_restores_camera_format_after_failure(self):
    laser, m = _laser()
    m._exposure, m._gain, m._focal_height = 25, 2, 3.28
    m._acquiring = True
    m.camera.camera.get_pixel_format.return_value = 0x01080001
    m.camera.set_pixel_format = AsyncMock()
    m.set_exposure, m.set_gain = AsyncMock(), AsyncMock()
    m.set_imaging_mode = AsyncMock()
    m.driver.send_command = AsyncMock()
    with patch.object(laser, "_enable", new_callable=AsyncMock):
      with self.assertRaisesRegex(RuntimeError, "reflection missing"):
        async with laser._session(5, 12):
          raise RuntimeError("reflection missing")
    self.assertEqual(
      m.camera.set_pixel_format.await_args_list, [call(0x01100007), call(0x01080001)]
    )
    m.set_exposure.assert_awaited_with(25)
    m.set_gain.assert_awaited_with(2)
    m.set_focus.assert_awaited_with(3.28)
    self.assertEqual(m.start_acquisition.call_count, 2)
