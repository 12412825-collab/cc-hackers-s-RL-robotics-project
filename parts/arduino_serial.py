"""Fail-safe DonkeyCar <-> Arduino serial bridge.

Protocol (newline terminated ASCII):
  Pi -> Arduino: C,<sequence>,<steering>,<throttle>
  Arduino -> Pi: T,<sequence>,<left_ticks>,<right_ticks>,<speed_mps>
"""

import logging
import threading
import time


logger = logging.getLogger(__name__)


class ArduinoSerialBridge:
    def __init__(self, port, baudrate=115200, timeout=0.05,
                 command_hz=20.0, stale_after=0.5, serial_factory=None):
        if serial_factory is None:
            try:
                import serial
            except ImportError as exc:
                raise ImportError(
                    "Arduino serial control requires pyserial: pip install pyserial"
                ) from exc
            serial_factory = serial.Serial
        self.serial = serial_factory(port=port, baudrate=baudrate,
                                     timeout=timeout)
        self.command_period = 1.0 / float(command_hz)
        self.stale_after = float(stale_after)
        self.sequence = 0
        self.last_write = 0.0
        self.last_telemetry = 0.0
        self.speed = 0.0
        self.left_ticks = 0
        self.right_ticks = 0
        self.connected = False
        self._running = True
        self._lock = threading.Lock()

    @staticmethod
    def _clip(value):
        value = 0.0 if value is None else float(value)
        return max(-1.0, min(1.0, value))

    def update(self):
        while self._running:
            try:
                raw = self.serial.readline()
                if not raw:
                    continue
                fields = raw.decode("ascii", errors="replace").strip().split(",")
                if len(fields) != 6 or fields[0] != "T":
                    continue
                _, _seq, left, right, speed, status = fields
                with self._lock:
                    self.left_ticks = int(left)
                    self.right_ticks = int(right)
                    self.speed = float(speed)
                    self.connected = status == "OK"
                    self.last_telemetry = time.monotonic()
            except (OSError, ValueError) as exc:
                logger.warning("Arduino telemetry error: %s", exc)
                time.sleep(0.05)

    def run_threaded(self, steering, throttle):
        now = time.monotonic()
        if now - self.last_write >= self.command_period:
            self.sequence += 1
            command = "C,{},{:.4f},{:.4f}\n".format(
                self.sequence, self._clip(steering), self._clip(throttle))
            try:
                self.serial.write(command.encode("ascii"))
                self.last_write = now
            except OSError as exc:
                logger.error("Arduino command write failed: %s", exc)
                self.connected = False
        with self._lock:
            alive = self.connected and now - self.last_telemetry <= self.stale_after
            speed = self.speed if alive else 0.0
            return speed, self.left_ticks, self.right_ticks, alive

    def shutdown(self):
        self._running = False
        try:
            self.serial.write(b"C,0,0.0000,0.0000\n")
            self.serial.flush()
        finally:
            self.serial.close()
