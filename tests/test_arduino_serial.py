from parts.arduino_serial import ArduinoSerialBridge


class FakeSerial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = []
        self.closed = False

    def readline(self):
        return b""

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_command_is_clipped_and_shutdown_stops_vehicle():
    serial = FakeSerial()
    bridge = ArduinoSerialBridge("test", command_hz=1000,
                                 serial_factory=lambda **kwargs: serial)
    bridge.run_threaded(2.0, -2.0)
    assert serial.writes[-1].decode().endswith(",1.0000,-1.0000\n")
    bridge.shutdown()
    assert serial.writes[-1] == b"C,0,0.0000,0.0000\n"
    assert serial.closed
