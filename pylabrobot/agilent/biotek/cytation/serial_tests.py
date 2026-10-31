"""Exercise the Cytation serial transport without a connected instrument."""

import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from pylabrobot.agilent.biotek.cytation import Cytation5
from pylabrobot.io.serial import Serial


class TestCytationSerial(unittest.IsolatedAsyncioTestCase):
  """Windows COM ports use the same protocol as the direct FTDI connection."""

  async def test_setup_and_query_through_serial(self) -> None:
    """Read identity and serial number without any FTDI-specific operations."""
    reader = Cytation5("reader", port="COM9", use_cam=False)
    assert isinstance(reader.io, Serial)
    self.assertEqual(reader.io.stopbits, 2)
    self.assertTrue(reader.io.rtscts)
    transport = MagicMock(spec=Serial)
    transport.setup = AsyncMock()
    transport.stop = AsyncMock()
    transport.reset_input_buffer = AsyncMock()
    transport.reset_output_buffer = AsyncMock()
    transport.write = AsyncMock(return_value=None)
    response = b"\x061320200  Version 1.12   0000\x03\x0617100919        0000\x03"
    transport.read = AsyncMock(side_effect=[bytes([value]) for value in response])
    reader.io = transport
    with patch.object(reader.microscope, "_on_setup", new_callable=AsyncMock):
      await reader.setup()
    self.assertEqual(reader.version, "1.12")
    self.assertEqual(await reader.request_serial_number(), "17100919")
    self.assertEqual(transport.write.await_args_list, [call(b"e"), call(b"C")])
    await reader.stop()
    transport.stop.assert_awaited_once()

  def test_rejects_ambiguous_transport(self) -> None:
    """A COM port and a USB identifier must not silently override each other."""
    with self.assertRaisesRegex(ValueError, "either a serial port or an FTDI device_id"):
      Cytation5("reader", port="COM9", device_id="17100919", use_cam=False)
