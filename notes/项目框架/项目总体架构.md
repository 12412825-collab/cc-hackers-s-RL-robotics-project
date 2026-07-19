

- Person A: Vision, deep learning, reinforcement learning, path planning, model deployment.
- Person B: Raspberry Pi, chassis, motors, sensors, communications, ROS, drivers.

What is truly needed is to establish a set of **Vibe Coding (AI collaborative development)** processes, instead of just treating AI as a code generator.

---

# I. It is recommended that the whole project adopts a four-layer architecture

Do not understand the project as just one Python program.

It is recommended to split it into four layers.

```text
                 Algorithm Layer (You)
────────────────────────────────────
Vision Model
Object Detection
Path Planning
Reinforcement Learning
Residual RL

                ↑↓↓↓

────────────────────────────────────
Control Layer
PID
Pure Pursuit
MPC
ROS Node

                ↑↓↓↓

────────────────────────────────────
Robot Layer (Classmate)
Raspberry Pi
STM32
Motors
Encoders
IMU
Camera

                ↑↓↓↓

────────────────────────────────────
Hardware Layer
Chassis
Servos
ESC (Electronic Speed Controller)
Power Supply
```