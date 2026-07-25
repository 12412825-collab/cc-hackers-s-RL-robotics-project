// ============================================================
//  AdvancedModel — 最终底层固件
//  树莓派通过串口发指令，Arduino 负责所有硬件驱动
// ============================================================

#include <AFMotor.h>
#include <Wire.h>

// ==================== 电机 ====================
// 左侧: M1(端口1) + M4(端口4)
// 右侧: M2(端口2) + M3(端口3)
AF_DCMotor motorL1(1);
AF_DCMotor motorL2(4);
AF_DCMotor motorR1(2);
AF_DCMotor motorR2(3);

#define MIN_SPEED 0
#define MAX_SPEED 255

// ==================== 编码器 ====================
#define ENC_L_PIN 19
#define ENC_R_PIN 18
volatile long encL = 0, encR = 0;

#define WHEEL_DIAMETER_MM  65.0
#define COUNTS_PER_REV     4
const float MM_PER_COUNT = PI * WHEEL_DIAMETER_MM / COUNTS_PER_REV;

// ==================== 超声波 ====================
#define TRIG_L 25
#define ECHO_L 24
#define TRIG_C 23
#define ECHO_C 22
#define TRIG_R 27
#define ECHO_R 26

// ==================== IMU MPU6500 ====================
// I2C: SDA=pin20, SCL=pin21, AD0→GND → 0x68
#define MPU_ADDR 0x68
float gxBias = 0.6988, gyBias = 1.1523, gzBias = 0.5106;
float angleZ = 0.0;
unsigned long lastImuUs = 0;

// ==================== 传感器缓存 ====================
float usL = 999, usC = 999, usR = 999;
int   usIdx = 0;  // 轮询索引

// ==================== 前馈补偿 ====================
int feedforward = 15;  // 正=左快→减速左加速右；负=右快

// ==================== 电机状态 ====================
int  curLspeed = 0, curRspeed = 0;
long lastDriveMs = 0;
#define DRIVE_TIMEOUT_MS 2000  // 2秒无新指令自动停（安全保护）

// ============================================================
//  超声波测距
// ============================================================
float readUS(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long dur = pulseIn(echo, HIGH, 30000);
  if (dur == 0) return 999;
  return dur * 0.0343 / 2.0;
}

// ============================================================
//  电机控制 (负值=后退, 正=前进, 0=松开)
// ============================================================
void setMotors(int lSpd, int rSpd) {
  // 应用前馈补偿（同向时生效，前进后退分别处理符号）
  if ((lSpd > 0 && rSpd > 0) || (lSpd < 0 && rSpd < 0)) {
    int ff = (lSpd > 0) ? feedforward : -feedforward;
    lSpd = constrain(lSpd - ff, -MAX_SPEED, MAX_SPEED);
    rSpd = constrain(rSpd + ff, -MAX_SPEED, MAX_SPEED);
  }

  // 左侧
  uint8_t lDir;
  int lAbs = abs(lSpd);
  if (lSpd > 0)      lDir = FORWARD;
  else if (lSpd < 0) lDir = BACKWARD;
  else               lDir = RELEASE;
  lAbs = constrain(lAbs, MIN_SPEED, MAX_SPEED);
  motorL1.setSpeed(lAbs); motorL1.run(lDir);
  motorL2.setSpeed(lAbs); motorL2.run(lDir);

  // 右侧
  uint8_t rDir;
  int rAbs = abs(rSpd);
  if (rSpd > 0)      rDir = FORWARD;
  else if (rSpd < 0) rDir = BACKWARD;
  else               rDir = RELEASE;
  rAbs = constrain(rAbs, MIN_SPEED, MAX_SPEED);
  motorR1.setSpeed(rAbs); motorR1.run(rDir);
  motorR2.setSpeed(rAbs); motorR2.run(rDir);

  curLspeed = lSpd; curRspeed = rSpd;
  lastDriveMs = millis();
}

void stopMotors() {
  motorL1.run(RELEASE); motorL2.run(RELEASE);
  motorR1.run(RELEASE); motorR2.run(RELEASE);
  curLspeed = 0; curRspeed = 0;
}

// ============================================================
//  IMU 读取 + 角度积分
// ============================================================
void readIMU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x43);  // GYRO_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)6);

  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  float gzCal = gz / 131.0 - gzBias;

  unsigned long now = micros();
  if (lastImuUs == 0) { lastImuUs = now; return; }
  float dt = (now - lastImuUs) / 1000000.0;
  lastImuUs = now;

  // 死区过滤
  if (abs(gzCal) > 0.5) {
    angleZ += gzCal * dt;
  }
}

// ============================================================
//  读取所有编码器返回距离 (mm)
// ============================================================
float getDistL() {
  noInterrupts();
  long c = encL;
  interrupts();
  return c * MM_PER_COUNT;
}

float getDistR() {
  noInterrupts();
  long c = encR;
  interrupts();
  return c * MM_PER_COUNT;
}

float getDistAvg() {
  noInterrupts();
  long l = encL, r = encR;
  interrupts();
  return (l + r) / 2.0 * MM_PER_COUNT;
}

// ============================================================
//  CHECK 响应 — 一帧返回全部传感器
// ============================================================
void printCheck() {
  // ODOM:左距离,右距离,左计数,右计数
  // US:左,中,右 (cm)
  // IMU:角度Z(°)
  noInterrupts();
  long lc = encL, rc = encR;
  interrupts();

  float distL = lc * MM_PER_COUNT;
  float distR = rc * MM_PER_COUNT;

  Serial.print("ODOM:");
  Serial.print(distL, 1); Serial.print(",");
  Serial.print(distR, 1); Serial.print(",");
  Serial.print(lc); Serial.print(",");
  Serial.print(rc);

  Serial.print("|US:");
  Serial.print(usL, 1); Serial.print(",");
  Serial.print(usC, 1); Serial.print(",");
  Serial.print(usR, 1);

  Serial.print("|IMU:");
  Serial.print(angleZ, 2);

  Serial.println();
}

// ============================================================
//  串口命令处理
// ============================================================
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "CHECK") {
    printCheck();
  }
  else if (cmd == "STOP") {
    stopMotors();
    Serial.println("OK:STOP");
  }
  else if (cmd == "ODOM_RESET") {
    noInterrupts();
    encL = 0; encR = 0;
    interrupts();
    angleZ = 0;
    Serial.println("OK:RESET");
  }
  else if (cmd.startsWith("DRIVE_")) {
    // 格式: DRIVE_left_right  例如 DRIVE_200_150 或 DRIVE_-200_200
    String rest = cmd.substring(6);
    int us1 = rest.indexOf('_');
    if (us1 == -1) { Serial.println("ERR:FORMAT"); return; }
    int lSpd = rest.substring(0, us1).toInt();
    int rSpd = rest.substring(us1 + 1).toInt();
    setMotors(lSpd, rSpd);
    Serial.print("OK:DRIVE "); Serial.print(lSpd); Serial.print(" "); Serial.println(rSpd);
  }
  else {
    Serial.print("ERR:UNKNOWN "); Serial.println(cmd);
  }
}

// ============================================================
//  setup
// ============================================================
void setup() {
  Serial.begin(115200);  // 高速串口
  delay(1000);

  // 编码器
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), isrEncL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), isrEncR, RISING);

  // 超声波
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_C, OUTPUT); pinMode(ECHO_C, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);

  // IMU
  Wire.begin();
  Wire.setClock(400000);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);  // 唤醒
  Wire.endTransmission();
  delay(100);

  // 电机初始化
  motorL1.setSpeed(0); motorL1.run(RELEASE);
  motorL2.setSpeed(0); motorL2.run(RELEASE);
  motorR1.setSpeed(0); motorR1.run(RELEASE);
  motorR2.setSpeed(0); motorR2.run(RELEASE);

  Serial.println("READY");
  Serial.println("Commands: DRIVE_l_r | CHECK | STOP | ODOM_RESET");

  // ===== 自动测试：DRIVE_150_200 跑 3 秒 =====
  delay(1500);
  Serial.println(">>> AUTO_TEST: DRIVE_150_200 (3s)");
  setMotors(150,150);
  delay(1000);
  stopMotors();
  delay(1000);
  setMotors(-150,-150);
  delay(1000);
  stopMotors();
  Serial.println(">>> AUTO_TEST_DONE");
}

// ============================================================
//  loop
// ============================================================
void loop() {
  // ---- 串口命令 ----
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }

  // ---- 超声波轮询（每次读1个，3轮覆盖全部） ----
  static unsigned long lastUs = 0;
  if (millis() - lastUs > 60) {
    lastUs = millis();
    switch (usIdx) {
      case 0: usL = readUS(TRIG_L, ECHO_L); break;
      case 1: usC = readUS(TRIG_C, ECHO_C); break;
      case 2: usR = readUS(TRIG_R, ECHO_R); break;
    }
    usIdx = (usIdx + 1) % 3;
  }

  // ---- IMU 角度积分 ----
  readIMU();

  // ---- 安全：超时自动停 ----
  if ((curLspeed != 0 || curRspeed != 0) && millis() - lastDriveMs > DRIVE_TIMEOUT_MS) {
    stopMotors();
    Serial.println("AUTO_STOP:TIMEOUT");
  }
}

// ============================================================
//  编码器中断
// ============================================================
void isrEncL() { encL++; }
void isrEncR() { encR++; }
