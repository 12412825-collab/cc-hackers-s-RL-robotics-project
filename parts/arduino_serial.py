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


class ArduinoSerial:
    """Library Assistant differential-drive bridge from the main branch."""

    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.angle = 0.0
        self.throttle = 0.0
        self.us_left = 999.0
        self.us_center = 999.0
        self.us_right = 999.0
        self.connect()

    def connect(self):
        try:
            import serial

            self.serial = serial.Serial(
                self.port, self.baudrate, timeout=0.1)
            time.sleep(2)
            logger.info(
                "Connected to Arduino on %s at %s",
                self.port, self.baudrate)
            self.serial.write(b"ODOM_RESET\n")
        except Exception as exc:
            logger.error("Failed to connect to Arduino: %s", exc)
            self.serial = None

    def run(self, angle, throttle):
        angle = 0.0 if angle is None else float(angle)
        throttle = 0.0 if throttle is None else float(throttle)

        left_throttle = throttle + angle
        right_throttle = throttle - angle
        max_throttle = max(abs(left_throttle), abs(right_throttle))
        if max_throttle > 1.0:
            left_throttle /= max_throttle
            right_throttle /= max_throttle

        if self.serial and self.serial.is_open:
            left_pwm = int(left_throttle * 255)
            right_pwm = int(right_throttle * 255)
            self.serial.write(
                f"DRIVE_{left_pwm}_{right_pwm}\n".encode("ascii"))
            self.serial.write(b"CHECK\n")
            line = self.serial.readline().decode("ascii").strip()
            if line.startswith("ODOM:") and "|US:" in line:
                try:
                    for part in line.split("|"):
                        if part.startswith("US:"):
                            values = part[3:].split(",")
                            self.us_left = float(values[0])
                            self.us_center = float(values[1])
                            self.us_right = float(values[2])
                except (ValueError, IndexError):
                    logger.debug("Ignoring malformed Arduino response: %s", line)

        return self.us_left, self.us_center, self.us_right

    def shutdown(self):
        if self.serial and self.serial.is_open:
            self.serial.write(b"STOP\n")
            self.serial.close()
            logger.info("Arduino Serial connection closed.")
