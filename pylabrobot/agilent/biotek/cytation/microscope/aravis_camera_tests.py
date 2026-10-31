"""Frame decoding and buffer ownership without a connected camera."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy")

import numpy as np

from .aravis_camera import AravisCamera


class TestAravisFrames(unittest.IsolatedAsyncioTestCase):
  """Reject incomplete frames and return every received buffer to the stream."""

  def setUp(self) -> None:
    """Supply a two-by-two camera frame with a mocked Aravis stream."""
    self.driver = AravisCamera()
    self.driver._camera = MagicMock()
    self.driver._device = MagicMock()
    self.driver._stream = MagicMock()
    self.driver._acquiring = True
    self.buffer = MagicMock()
    self.buffer.get_status.return_value = 0
    self.buffer.get_image_pixel_format.return_value = 0x01080001
    self.buffer.get_image_width.return_value = 2
    self.buffer.get_image_height.return_value = 2
    self.buffer.get_image_padding.return_value = (0, 0)
    self.buffer.get_image_data.return_value = bytes([1, 2, 3, 4])
    self.driver._stream.timeout_pop_buffer.return_value = self.buffer
    aravis = MagicMock()
    aravis.BufferStatus.SUCCESS = 0
    self.patch = patch("pylabrobot.agilent.biotek.cytation.microscope.aravis_camera.Aravis", aravis)
    self.patch.start()
    self.addCleanup(self.patch.stop)

  async def test_mono16_retains_pixel_depth_and_skips_row_padding(self) -> None:
    """High-bit-depth frames must not be reshaped as eight-bit pixels."""
    self.buffer.get_image_pixel_format.return_value = 0x01100007
    self.buffer.get_image_padding.return_value = (2, 0)
    self.buffer.get_image_data.return_value = np.array(
      [1000, 4095, 0, 32768, 65535, 0], dtype="<u2"
    ).tobytes()
    frame = await self.driver.trigger()
    np.testing.assert_array_equal(frame, [[1000, 4095], [32768, 65535]])
    self.assertEqual(frame.dtype, np.uint16)
    self.driver.stream.push_buffer.assert_called_once_with(self.buffer)

  async def test_incomplete_frame_is_rejected_and_recycled(self) -> None:
    """An Aravis transport error must not become a plausible scientific image."""
    self.buffer.get_status.return_value = 3
    with self.assertRaisesRegex(RuntimeError, "incomplete frame"):
      await self.driver.trigger()
    self.driver.stream.push_buffer.assert_called_once_with(self.buffer)

  async def test_short_payload_is_rejected_and_recycled(self) -> None:
    """A decoding failure must not exhaust the stream's finite buffer pool."""
    self.buffer.get_image_data.return_value = b"\x01"
    with self.assertRaisesRegex(RuntimeError, "shorter"):
      await self.driver.trigger()
    self.driver.stream.push_buffer.assert_called_once_with(self.buffer)

  async def test_missing_frame_raises_timeout(self) -> None:
    """No buffer is available to return when the camera misses a trigger."""
    self.driver.stream.timeout_pop_buffer.return_value = None
    with self.assertRaises(TimeoutError):
      await self.driver.trigger()
    self.driver.stream.push_buffer.assert_not_called()

  async def test_pixel_format_cannot_change_during_acquisition(self) -> None:
    """Changing payload size while streaming would invalidate allocated buffers."""
    with self.assertRaisesRegex(RuntimeError, "Stop acquisition"):
      await self.driver.set_pixel_format(0x01100007)
    self.driver.camera.set_pixel_format.assert_not_called()
