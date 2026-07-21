"""Raspberry Pi + Arduino Mega hardware profile.

Start with:
    python manage.py drive --myconfig=myconfig.pi.py
"""

SIMULATOR = "none"
DONKEY_GYM = False
USE_TRAINING_CONSOLE = False

# Arduino owns the actuators. DonkeyCar's local GPIO/PCA9685 drivetrain must
# remain disabled to prevent two controllers from driving the same hardware.
DRIVE_TRAIN_TYPE = "MOCK"
HAVE_ARDUINO_SERIAL = True
ARDUINO_SERIAL_PORT = "/dev/ttyACM0"
ARDUINO_SERIAL_BAUD = 115200
ARDUINO_SERIAL_TIMEOUT = 0.05
ARDUINO_COMMAND_HZ = 20.0
ARDUINO_TELEMETRY_STALE_SEC = 0.5

USE_VELOCITY_CONTROL = True
MAX_LINEAR_VELOCITY = 0.20
MAX_ANGULAR_VELOCITY = 1.50

USE_MULTI_MODAL = True
HAVE_ODOM = False  # Arduino bridge publishes enc/speed directly.
ENABLE_ENCODER = True
ENCODER_TYPE = "Arduino"
ENCODER_MM_PER_TICK = 10.2102  # pi*65mm / 20 pulses; measure before training.
ENCODER_MAX_SPEED = 2.0

# Enable these only after their physical drivers are verified.
HAVE_IMU = False
ENABLE_IMU = False
ENABLE_OBSTACLE = False
RESIDUAL_RL = False
