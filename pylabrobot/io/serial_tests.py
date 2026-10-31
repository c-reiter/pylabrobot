import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from pylabrobot.io.serial import Serial, find_serial_ports


class TestSerialWrites(unittest.IsolatedAsyncioTestCase):
  """A short host write must not be reported as a complete device command."""

  async def test_short_write_raises_before_recording_success(self) -> None:
    """Preserve the failure so a driver can close the incomplete exchange."""
    connection = Serial("test device", port="test-port")
    connection._ser = MagicMock()
    connection._ser.write.return_value = 2
    with ThreadPoolExecutor(max_workers=1) as executor:
      connection._executor = executor
      with patch("pylabrobot.io.serial.capturer.record") as record:
        with self.assertRaisesRegex(IOError, "sent 2 of 4 bytes"):
          await connection.write(b"test")
        record.assert_not_called()
    connection._executor = None


class TestSerialDiscovery(unittest.TestCase):
  def test_find_serial_ports_matches_multiple_vid_pid_pairs(self) -> None:
    ports = [
      types.SimpleNamespace(device="/dev/tty-one", vid=0x1234, pid=0x0001),
      types.SimpleNamespace(device="/dev/tty-two", vid=0x1234, pid=0x0002),
      types.SimpleNamespace(device="/dev/tty-other", vid=0x9999, pid=0x0001),
    ]
    serial_module = types.SimpleNamespace(
      tools=types.SimpleNamespace(
        list_ports=types.SimpleNamespace(comports=lambda: ports),
      )
    )

    with (
      patch("pylabrobot.io.serial.HAS_SERIAL", True),
      patch("pylabrobot.io.serial.serial", serial_module, create=True),
    ):
      discovered = find_serial_ports({(0x1234, 0x0001), (0x1234, 0x0002)})

    self.assertEqual(discovered, ["/dev/tty-one", "/dev/tty-two"])

  def test_find_serial_ports_reports_missing_pyserial(self) -> None:
    with patch("pylabrobot.io.serial.HAS_SERIAL", False):
      with self.assertRaisesRegex(RuntimeError, "pyserial is not installed"):
        find_serial_ports({(0x1234, 0x0001)})
