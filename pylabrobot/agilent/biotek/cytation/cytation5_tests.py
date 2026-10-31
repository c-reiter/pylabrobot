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

  async def test_capture_preparation_recloses_after_assay(self):
    reader = Cytation5("reader", use_cam=False)
    plate = cor_96_wellplate_360uL_Fb("plate")
    with (
      patch.object(reader, "close", new_callable=AsyncMock) as close,
      patch.object(reader, "set_plate", new_callable=AsyncMock) as geometry,
      patch.object(reader.microscope, "led_off", new_callable=AsyncMock),
    ):
      reader._tray_closed = True
      await reader._prepare_microscopy(plate)
      geometry.assert_awaited_once_with(plate)
      close.assert_not_awaited()
      await reader._prepare_plate_read()
      await reader._prepare_microscopy(plate)
      close.assert_awaited_once_with(plate=plate)


class TestCytationTemperature(unittest.IsolatedAsyncioTestCase):
  """Heater targets can be reduced without requiring active cooling."""

  async def test_heating_range_and_passive_cooling(self):
    reader = Cytation5("reader", use_cam=False)
    with (
      patch.object(reader, "query_capability", new_callable=AsyncMock) as capability,
      patch.object(reader, "send_command", new_callable=AsyncMock) as command,
    ):
      capability.side_effect = lambda code: b"0" * 17 if code == 2126 else b"1402" + b"0" * 13
      await reader.set_temperature(37)
      command.assert_awaited_with("g", "370", timeout=40)
      await reader.set_temperature(25)
      command.assert_awaited_with("g", "250", timeout=40)
      await reader.set_temperature(65)
      command.assert_awaited_with("g", "650", timeout=40)
      await reader.set_temperature(30, gradient=2)
      command.assert_awaited_with("g", "302", timeout=40)
      await reader.stop_heating_or_cooling()
      command.assert_awaited_with("g", "000", timeout=40)
    self.assertTrue(reader.supports_heating)
    self.assertFalse(reader.supports_active_cooling)

  async def test_invalid_targets_do_not_send_commands(self):
    reader = Cytation5("reader", use_cam=False)
    with patch.object(reader, "send_command", new_callable=AsyncMock) as command:
      for target in (-1, 0, 65.01, float("nan"), float("inf")):
        with self.assertRaises(ValueError):
          await reader.set_temperature(target)
      command.assert_not_awaited()

  async def test_extended_temperature_fields(self):
    reader = Cytation5("reader", use_cam=False)
    with (
      patch.object(reader, "query_capability", new_callable=AsyncMock) as capability,
      patch.object(reader, "send_command", new_callable=AsyncMock) as command,
    ):
      capability.side_effect = lambda code: (
        b"1" + b"0" * 16 if code == 2126 else b"1402" + b"0" * 13
      )
      await reader.set_temperature(30.5, gradient=1.2)
      command.assert_awaited_with("g", "30512", timeout=40)

  async def test_fractional_target_rejected_by_integer_firmware(self):
    reader = Cytation5("reader", use_cam=False)
    reader.io = MagicMock()
    reader.io.stop = AsyncMock()
    with (
      patch.object(reader, "query_capability", new_callable=AsyncMock, return_value=b"0" * 17),
      patch.object(reader, "send_command", new_callable=AsyncMock) as command,
    ):
      with self.assertRaisesRegex(ValueError, "steps of 1.0"):
        await reader.set_temperature(30.5)
      command.assert_not_awaited()


class TestCytationTimeResolvedFluorescence(unittest.IsolatedAsyncioTestCase):
  """Validate flash timing and capability selection without accessing a reader."""

  async def test_timing_conversion_and_capability(self) -> None:
    """Public seconds become whole microseconds in the gated endpoint request."""
    reader = Cytation5("reader", use_cam=False)
    plate = cor_96_wellplate_360uL_Fb("plate")
    with (
      patch.object(reader, "query_capability", new_callable=AsyncMock) as capability,
      patch.object(reader, "_read_fluorescence", new_callable=AsyncMock) as read,
    ):
      capability.return_value = b"101" + b"0" * 14
      read.return_value = []
      result = await reader.read_time_resolved_fluorescence(
        plate, plate["A3"], 360, 620, 6, delay=0.0001, integration_time=0.0005
      )
      self.assertEqual(result, [])
      capability.assert_awaited_once_with(2006)
      read.assert_awaited_once_with(
        plate,
        plate["A3"],
        360,
        620,
        6,
        delay_microseconds=100,
        collection_microseconds=500,
      )

  async def test_unavailable_monochromator_trf_does_not_read(self) -> None:
    """Filter-only TRF support does not imply monochromator TRF support."""
    reader = Cytation5("reader", use_cam=False)
    plate = cor_96_wellplate_360uL_Fb("plate")
    with (
      patch.object(reader, "query_capability", new_callable=AsyncMock) as capability,
      patch.object(reader, "_read_fluorescence", new_callable=AsyncMock) as read,
    ):
      for data in (b"0" * 17, b"110" + b"0" * 14):
        capability.return_value = data
        with self.assertRaises(NotImplementedError):
          await reader.read_time_resolved_fluorescence(plate, plate["A3"], 360, 620, 6)
      read.assert_not_awaited()

  async def test_invalid_timings_do_not_query_hardware(self) -> None:
    """Reject out-of-range, nonfinite and sub-microsecond requests before I/O."""
    reader = Cytation5("reader", use_cam=False)
    plate = cor_96_wellplate_360uL_Fb("plate")
    with patch.object(reader, "query_capability", new_callable=AsyncMock) as capability:
      for delay, duration in (
        (-0.000001, 0.0003),
        (0.002001, 0.0003),
        (0, 0),
        (0, 0.000019),
        (0, 0.002001),
        (0.0000005, 0.0003),
        (0, 0.0003005),
        (float("nan"), 0.0003),
        (0, float("inf")),
      ):
        with self.subTest(delay=delay, duration=duration):
          with self.assertRaises(ValueError):
            await reader.read_time_resolved_fluorescence(
              plate, plate["A3"], 360, 620, 6, delay, duration
            )
      capability.assert_not_awaited()
