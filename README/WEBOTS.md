# Webots differential-drive integration

This project keeps DonkeyCar as its runtime and replaces DonkeySim with a
Webots differential-drive adapter. Commands inside the robot boundary use
physical units:

- `linear/velocity`: metres/second
- `angular/velocity`: radians/second
- residual SAC output: angular-velocity correction only

## Control flow

```text
user or base pilot (normalized steering/throttle)
    -> VelocityDriveMode (v_base, omega_base)
    -> omega = omega_base + residual_omega
    -> WebotsAdapter
    -> left/right wheel angular velocity
```

The same `v/omega` command is converted back to normalized DonkeyCar channels
at a compatibility boundary. Existing Tub recording and real differential
drivetrain parts therefore continue to work.

## Required Webots Robot devices

The Robot or PROTO must have `supervisor TRUE` and the following device names.
Names can be overridden in `myconfig.py`.

| Device | Default name |
|---|---|
| left RotationalMotor | `left wheel motor` |
| right RotationalMotor | `right wheel motor` |
| left PositionSensor | `left wheel sensor` |
| right PositionSensor | `right wheel sensor` |
| Camera | `camera` |
| Accelerometer | `accelerometer` |
| Gyro | `gyro` |
| front DistanceSensor | `front distance sensor` |

The supplied 4WD model uses four motor and four encoder names, configured by
`WEBOTS_LEFT_MOTORS`, `WEBOTS_RIGHT_MOTORS`, `WEBOTS_LEFT_ENCODERS`, and
`WEBOTS_RIGHT_ENCODERS`. Front and rear wheels on the same side receive the
same target speed. Odometry uses the mean encoder rate on each side.

Set the Robot's `controller` field to `donkey_webots`. Add the project
`simulation/controllers` directory to the Webots controller search path, or
copy/link the `donkey_webots` controller directory into the Webots project.

Webots must use a Python interpreter containing this project's dependencies
(`donkeycar`, NumPy, OpenCV, and PyTorch when residual RL is enabled).

## Calibration before training

Replace the provisional values in `myconfig.py` with real measurements:

1. `WHEEL_RADIUS`
2. `WHEEL_SEPARATION`
3. `MAX_WHEEL_SPEED`
4. `MAX_LINEAR_VELOCITY`
5. `MAX_ANGULAR_VELOCITY`
6. motor acceleration/torque and robot mass in the Webots model
7. camera pose, field of view, resolution, and update rate

`WEBOTS_DISTANCE_TO_CM` must agree with the DistanceSensor lookup table.

## Running

Open `simulation/worlds/four_wheel_track.wbt` as a Webots project and start the
simulation. Webots launches
`simulation/controllers/donkey_webots/donkey_webots.py`. To load a base pilot,
set `DONKEY_MODEL_PATH` in the controller environment to the model path.

Keep `RESIDUAL_RL=False` until a policy trained for the new `v/omega` action
semantics is available. Old residual checkpoints represented normalized
steering corrections and are not physically equivalent.

## Current boundary

This integration supplies a closed Webots/DonkeyCar control loop and sensor
channels. A true online SAC environment still needs explicit episode reset,
reward, termination, and track-centre projection. Do not treat the existing
offline trainer (zero reward and identical next-state) as closed-loop RL.

## Model assumptions from available specifications

The supplied `FourWheelRobot.proto` uses the known values directly: 255 mm
length, 170 mm width, 65 mm tyre diameter, 26 mm tyre width, 150 mm wheelbase,
130 mm wheel separation, and 0.8 kg mass including the battery.

Unknown parameters are provisional: 55 mm chassis height, 105 mm camera
height, 60 degree horizontal camera FOV, 12 rad/s maximum wheel speed, and
0.12 Nm motor torque. These values are intentionally exposed in the PROTO or
`myconfig.py` and must be calibrated before quantitative Sim-to-Real claims.
