# How exactly should the simulation system be built?

This is the most important question of the whole project, and also where many robotics labs are most prone to take detours.

When many students first start doing reinforcement learning for robots, they think: "I should first open VSCode, and then start writing simulation code."

In reality, genuine robotics labs almost never do this.

A mature digital twin system comes about **70% from existing open-source ecosystems, and 30% from your own innovative development**.

What you truly need to write yourself is not the entire robot, but the innovative part of your paper.

Therefore, I recommend the following development approach for the entire project.

```text
               GitHub Open Source Projects
                      │
      ┌───────────────┼───────────────┐
      │               │               │
   Robot Model   Gazebo Environment  ROS2 Comm
      │               │               │
      └───────────────┼───────────────┘
                      │
               Digital Twin
                      │
          Gymnasium Environment
                      │
             Your RL Algorithm
                      │
               Raspberry Pi
                      │
                 Arduino Mega
                      │
                  Real Car
```

Note a very important concept here:

> **GitHub is responsible for the "robot platform", and your code is responsible for the "algorithm innovation".**

Never reverse this.

---

# Why is it not recommended to write everything yourself?

Many people doing reinforcement learning robots for the first time will have this thought:

> "I\'ll open Pygame, draw a robot myself, and simulate the physics myself."

Theoretically possible.

But practically, this means you need to complete yourself:

- Robot kinematics/dynamics
- Differential drive model
- Collision detection
- Map loading
- Camera rendering
- Lighting simulation
- Sensor noise
- Time synchronization
- RL interfaces

There are already plenty of mature solutions for these things.

If you develop everything from scratch, one person might need several months or longer, and the final effect is often worse than Gazebo.

Therefore, for research projects, **the ROI of writing the entire simulation platform yourself is very low**.

---

# Why can\'t you rely entirely on GitHub either?

Many students will then go to the other extreme:

> "Then why don\'t I just find a Reinforcement Learning Car project and run it?"

The problem is, your car is not a DonkeyCar, nor a TurtleBot, nor an F1TENTH.

What you are currently using is:

- Arduino Mega2560
- Four-wheel TT motor chassis
- Later adding a Raspberry Pi
- Camera
- Deep learning vision module
- Your own designed RL policy

There is hardly any GitHub project that exactly matches your hardware.

If you just copy one directly, you will later spend even more time trying to modify the reward function, observation space, or robot model.

So the truly reasonable method is:

> **Reuse mature frameworks, and complete the parts related to your paper yourself.**

---

# I recommend adopting a "building block" development approach

Don\'t look for an "all-in-one repository".

You should look for different modules separately and then combine them.

The entire project can be split into six parts.

## Part 1: Robot Model

This part includes:

- Chassis dimensions
- Tire dimensions
- Motors
- Mass
- Inertia
- Sensor installation positions

It\'s recommended to use URDF/Xacro descriptions directly for this.

There are already tons of mobile robot models on GitHub you can refer to.

Later you just need to modify the dimensions and parameters.

For example:

```text
base_link

├── chassis

├── wheel_fl

├── wheel_fr

├── wheel_rl

├── wheel_rr

├── camera

├── imu

└── encoder
```

There is almost no paper innovation here, so there is no need to develop it from scratch.

---

## Part 2: Gazebo World (Simulation World)

This is a part many people ignore.

A digital twin is not just the robot.

It also includes the environment the robot is in.

For example:

```text
World

├── Floor

├── Walls

├── Track

├── Obstacles

├── Lighting

├── Cameras

└── Friction Parameters
```

It is recommended to design this yourself.

Because later, reinforcement learning will need a lot of Domain Randomization.

For example:

Every training episode:

- Floor color changes
- Lighting changes
- Wall texture changes
- Obstacle position changes

These will all improve the Sim2Real capability.

---

# Part 3: ROS2

This layer is actually the communication layer.

For example:

```text
Camera Topic

↓

Vision Node

↓

RL Node

↓

Action

↓

Motor Controller
```

ROS2 already finishes this for you:

- Topics
- Services
- Messages
- Clock
- Launch

Thus, there is no need to reinvent the communication system.

---

# Part 4: Gymnasium Environment

It\'s highly recommended to write this part entirely yourself.

Because it directly determines your paper.

For example:

```text
Observation

Action

Reward

Done
```

In environments written by others:

The Reward is different.

The Observation is different.

The Action Space is different.

Then the paper will have no innovation.

Therefore, this should be entirely controlled by you.