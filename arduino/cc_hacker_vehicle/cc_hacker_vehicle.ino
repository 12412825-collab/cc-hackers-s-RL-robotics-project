#include <Servo.h>

// Project wiring from README/README.md.
constexpr uint8_t LEFT_IN1 = 5;
constexpr uint8_t LEFT_IN2 = 6;
constexpr uint8_t RIGHT_IN1 = 9;
constexpr uint8_t RIGHT_IN2 = 10;
constexpr uint8_t STEERING_PIN = 7;
constexpr uint8_t LEFT_ENCODER_PIN = 2;
constexpr uint8_t RIGHT_ENCODER_PIN = 3;

constexpr unsigned long BAUD = 115200;
constexpr unsigned long WATCHDOG_MS = 500;
constexpr unsigned long TELEMETRY_MS = 50;
constexpr float METRES_PER_TICK = 0.0102102f; // pi*0.065m / 20
constexpr uint8_t MAX_MOTOR_PWM = 80;         // Safe first-test limit.
constexpr int SERVO_LEFT_US = 1100;           // Calibrate on the real vehicle.
constexpr int SERVO_CENTER_US = 1500;
constexpr int SERVO_RIGHT_US = 1900;

volatile long leftTicks = 0;
volatile long rightTicks = 0;
Servo steeringServo;
unsigned long lastCommandMs = 0;
unsigned long lastTelemetryMs = 0;
long previousMeanTicks = 0;
unsigned long previousSpeedMs = 0;
unsigned long sequence = 0;

void leftTick() { ++leftTicks; }
void rightTick() { ++rightTicks; }

void setMotor(uint8_t in1, uint8_t in2, float value) {
  value = constrain(value, -1.0f, 1.0f);
  int pwm = static_cast<int>(fabs(value) * MAX_MOTOR_PWM);
  analogWrite(in1, value >= 0 ? pwm : 0);
  analogWrite(in2, value < 0 ? pwm : 0);
}

void stopVehicle() {
  analogWrite(LEFT_IN1, 0); analogWrite(LEFT_IN2, 0);
  analogWrite(RIGHT_IN1, 0); analogWrite(RIGHT_IN2, 0);
  steeringServo.writeMicroseconds(SERVO_CENTER_US);
}

void applyCommand(float steering, float throttle) {
  steering = constrain(steering, -1.0f, 1.0f);
  int pulse = SERVO_CENTER_US;
  if (steering < 0) pulse += static_cast<int>((SERVO_CENTER_US - SERVO_LEFT_US) * steering);
  else pulse += static_cast<int>((SERVO_RIGHT_US - SERVO_CENTER_US) * steering);
  steeringServo.writeMicroseconds(pulse);
  setMotor(LEFT_IN1, LEFT_IN2, throttle);
  setMotor(RIGHT_IN1, RIGHT_IN2, throttle);
}

void readCommand() {
  static char line[64];
  static uint8_t length = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      line[length] = '\0';
      char *kind = strtok(line, ",");
      char *seqText = strtok(nullptr, ",");
      char *steeringText = strtok(nullptr, ",");
      char *throttleText = strtok(nullptr, ",");
      char *extra = strtok(nullptr, ",");
      if (kind && strcmp(kind, "C") == 0 && seqText && steeringText &&
          throttleText && !extra) {
        sequence = strtoul(seqText, nullptr, 10);
        float steering = atof(steeringText);
        float throttle = atof(throttleText);
        applyCommand(steering, throttle);
        lastCommandMs = millis();
      }
      length = 0;
    } else if (c != '\r' && length < sizeof(line) - 1) {
      line[length++] = c;
    }
  }
}

void sendTelemetry() {
  unsigned long now = millis();
  if (now - lastTelemetryMs < TELEMETRY_MS) return;
  noInterrupts(); long left = leftTicks; long right = rightTicks; interrupts();
  long meanTicks = (left + right) / 2;
  unsigned long dtMs = now - previousSpeedMs;
  float speed = dtMs ? (meanTicks - previousMeanTicks) * METRES_PER_TICK * 1000.0f / dtMs : 0.0f;
  Serial.print("T,"); Serial.print(sequence); Serial.print(',');
  Serial.print(left); Serial.print(','); Serial.print(right); Serial.print(',');
  Serial.print(speed, 4); Serial.println(",OK");
  previousMeanTicks = meanTicks; previousSpeedMs = now; lastTelemetryMs = now;
}

void setup() {
  pinMode(LEFT_IN1, OUTPUT); pinMode(LEFT_IN2, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT); pinMode(RIGHT_IN2, OUTPUT);
  pinMode(LEFT_ENCODER_PIN, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_PIN), leftTick, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_PIN), rightTick, RISING);
  steeringServo.attach(STEERING_PIN);
  stopVehicle(); Serial.begin(BAUD);
  lastCommandMs = previousSpeedMs = millis();
}

void loop() {
  readCommand();
  if (millis() - lastCommandMs > WATCHDOG_MS) stopVehicle();
  sendTelemetry();
}
