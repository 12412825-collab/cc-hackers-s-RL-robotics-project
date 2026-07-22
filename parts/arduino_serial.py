import serial
import time
import logging

logger = logging.getLogger(__name__)

class ArduinoSerial:
    """
    DonkeyCar Part that converts steering (angle) and throttle into differential 
    drive commands and sends them to the Arduino over USB Serial.
    """
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.angle = 0.0
        self.throttle = 0.0
        
        # Sensor data from Arduino
        self.us_left = 999.0
        self.us_center = 999.0
        self.us_right = 999.0
        
        self.connect()

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)  # Wait for Arduino to reset
            logger.info(f"Connected to Arduino on {self.port} at {self.baudrate}")
            self.serial.write(b"ODOM_RESET\n")
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.serial = None

    def run(self, angle, throttle):
        """
        DonkeyCar vehicle loop function.
        Receives standard steering/throttle, sends to Arduino, and returns sensor data.
        """
        if angle is None: angle = 0.0
        if throttle is None: throttle = 0.0
        
        # Differential drive mixing
        # angle > 0 means turn right (left motor faster)
        left_throttle = throttle + angle
        right_throttle = throttle - angle
        
        # Normalize to [-1.0, 1.0] if it exceeds
        max_t = max(abs(left_throttle), abs(right_throttle))
        if max_t > 1.0:
            left_throttle /= max_t
            right_throttle /= max_t
            
        # Convert to PWM [-255, 255]
        left_pwm = int(left_throttle * 255)
        right_pwm = int(right_throttle * 255)
        
        if self.serial and self.serial.is_open:
            # Send Drive command
            cmd = f"DRIVE_{left_pwm}_{right_pwm}\n"
            self.serial.write(cmd.encode('ascii'))
            
            # Poll for sensor data occasionally (DonkeyCar runs at ~20Hz)
            # In a real app we might thread this, but for now we poll
            self.serial.write(b"CHECK\n")
            line = self.serial.readline().decode('ascii').strip()
            
            # Parse CHECK response: ODOM:...|US:left,center,right|IMU:...
            if line.startswith("ODOM:") and "|US:" in line:
                try:
                    parts = line.split("|")
                    for p in parts:
                        if p.startswith("US:"):
                            us_vals = p[3:].split(',')
                            self.us_left = float(us_vals[0])
                            self.us_center = float(us_vals[1])
                            self.us_right = float(us_vals[2])
                except Exception as e:
                    pass # Ignore malformed lines

        return self.us_left, self.us_center, self.us_right

    def shutdown(self):
        if self.serial and self.serial.is_open:
            self.serial.write(b"STOP\n")
            self.serial.close()
            logger.info("Arduino Serial connection closed.")
