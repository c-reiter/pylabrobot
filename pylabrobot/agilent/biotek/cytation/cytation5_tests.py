"""Cytation controller/camera lifecycle and plate-height recovery."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pylabrobot.agilent.biotek._reader import BioTekError
from pylabrobot.agilent.biotek.cytation import Cytation5
from pylabrobot.agilent.biotek.plate_reader_base import BioTekPlateReaderDriver
from pylabrobot.resources import cor_96_wellplate_360uL_Fb


class TestCytationLifecycle(unittest.IsolatedAsyncioTestCase):
  """Release resources on errors and only recover the documented geometry status."""

  async def test_plate_height_recovery_cycles_tray(self):
    reader = Cytation5("reader", use_cam=False)
    reader._slow_mode = False
    require = patch.object(reader, "_require_rdop22", new_callable=AsyncMock)
    require.start()
    self.addCleanup(require.stop)
    command_patch = patch.object(reader, "send_command", new_callable=AsyncMock)
    send_command = command_patch.start()
    self.addCleanup(command_patch.stop)
    send_command.side_effect = [
      BioTekError(0x5B00, "y"),
      b"0000\x03",
      b"\x060000\x03",
      b"0000\x03",
      b"\x060000\x03",
    ]
    plate = cor_96_wellplate_360uL_Fb("plate")
    reader.microscope._row, reader.microscope._column = 1, 1
    await reader.set_plate(plate)
    self.assertEqual([c.args[0] for c in send_command.await_args_list], ["y", "i", "J", "y", "A"])
    self.assertIs(reader._plate, plate)
    self.assertIsNone(reader.microscope._row)

  async def test_other_geometry_errors_do_not_move_tray(self):
    reader = Cytation5("reader", use_cam=False)
    require = patch.object(reader, "_require_rdop22", new_callable=AsyncMock)
    require.start()
    self.addCleanup(require.stop)
    command_patch = patch.object(reader, "send_command", new_callable=AsyncMock)
    send_command = command_patch.start()
    self.addCleanup(command_patch.stop)
    send_command.side_effect = BioTekError(0x5A00, "y")
    reader.io = MagicMock()
    reader.io.stop = AsyncMock()
    with self.assertRaises(BioTekError) as result:
      await reader.set_plate(cor_96_wellplate_360uL_Fb("plate"))
    self.assertEqual(result.exception.code, 0x5A00)
    self.assertEqual(send_command.await_count, 1)
    self.assertIsNone(reader._plate)

  async def test_camera_setup_failure_releases_both_connections(self):
    reader = Cytation5("reader")
    setup_patch = patch.object(reader.microscope, "_on_setup", new_callable=AsyncMock)
    setup_patch.start().side_effect = RuntimeError("camera missing")
    self.addCleanup(setup_patch.stop)
    stop_patch = patch.object(reader.microscope, "_on_stop", new_callable=AsyncMock)
    camera_stop = stop_patch.start()
    self.addCleanup(stop_patch.stop)
    with (
      patch.object(BioTekPlateReaderDriver, "setup", new_callable=AsyncMock),
      patch.object(BioTekPlateReaderDriver, "stop", new_callable=AsyncMock) as stop,
    ):
      with self.assertRaisesRegex(RuntimeError, "camera missing"):
        await reader.setup()
      stop.assert_awaited_once()
    camera_stop.assert_awaited_once()
