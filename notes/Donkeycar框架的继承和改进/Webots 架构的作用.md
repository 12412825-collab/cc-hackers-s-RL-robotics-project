1. Establish a Differential Chassis that Matches the Real Car
Currently DonkeySim receives:

```text
steering
throttle
```

In Webots you should establish:

```text
left_wheel_motor
right_wheel_motor
```

The robot model needs to be configured with:

- Wheelbase
- Wheel radius
- Robot mass
- Center of mass and inertia
- Maximum wheel speed
- Motor acceleration or torque
- Friction between tires and ground
- Front/rear universal wheels or support structure

The control layer should ideally use:

```text
action = [v, ω]
```

Which is converted to left and right wheel angular velocities:

```text
ω_left  = (v - ω × wheel_separation / 2) / wheel_radius
ω_right = (v + ω × wheel_separation / 2) / wheel_radius
```

This way both Webots and the real car can use the same policy interface.

### 2. Simulate Real Sensors

Webots can provide simulated data for the current `parts/sensors.py`:

| Current Observation | Webots Device or Calculation Method |
| ----------------- | -------------------- |
| `cam/image_array` | Camera               |
| Left wheel speed  | Left wheel PositionSensor differential |
| Right wheel speed | Right wheel PositionSensor differential |
| Linear velocity   | Combined left and right wheel speeds |
| Angular velocity  | IMU/Gyro or calculated from wheel speeds |
| IMU Acceleration  | Accelerometer        |
| IMU Angular Vel   | Gyro                 |
| Obstacle distance | DistanceSensor, LiDAR |
| Ground truth pose | Supervisor           |
| CTE               | Calculated from robot pos and track centerline |

Webots natively supports differential wheels and encoder modeling, making it a better substitute for the current simulation approach which is primarily Ackermann vehicles. [Webots Official Project](https://github.com/cyberbotics/webots)

### 3. Provide a True Closed-loop RL Environment

The offline training in current [residual_rl.py (line 879)](C:\\Users\\Sichang Yang\\Downloads\\cc+hacker final\\parts\\residual_rl.py:879) has:

```python
next_state = state
reward = 0
done = False
```

Webots should provide real state transitions:

```python
observation = env.reset()

while not terminated:
    action = policy(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)
```

Where every step genuinely advances the Webots simulation to get the next frame image, wheel speeds, IMU, position, and collision status. This is the closed-loop interaction SAC needs.

### 4. Provide Automatic Evaluation

The Webots Supervisor can obtain simulation ground truth, but this data is only used for rewards and evaluation, and doesn\'t need to be policy input:

- Distance from track centerline
- Heading error
- Distance traveled
- Reaching goal status
- Collisions
- Out of bounds
- Completion time
- Average speed
- Control smoothness

### 5. Domain Randomization and Sim-to-Real

During training you can randomly change:

- Lighting intensity, direction, and color
- Camera noise, exposure, and field of view
- Ground texture
- Tire friction
- Robot mass
- Response differences between left and right motors
- Wheel radius and wheelbase errors
- Control delay
- Sensor noise and frame drops
- Obstacle positions

These randomizations can reduce the issue of the policy only adapting to a single simulation scenario.