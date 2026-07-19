Residual Reinforcement Learning (**Residual RL**) is a control framework that combines **traditional control methods** with **deep reinforcement learning**. In recent years, it has been widely applied in areas such as mobile robots, robotic arms, drones, and autonomous driving.

For your robot car project (Camera + Deep Learning + Reinforcement Learning + Multi-sensors), Residual RL is a highly recommended method because it can leverage the stability of traditional controllers while utilizing the adaptive capabilities of reinforcement learning.

---

# I. Why Propose Residual Reinforcement Learning?

In normal reinforcement learning, the agent needs to **learn the entire control policy from scratch**.

For example, a robot car needs to learn:

- How to stay in the center of the lane;
- How to turn;
- How to avoid obstacles;
- How to control speed;
- How to reach the target location.

If it relies entirely on reinforcement learning, the robot needs to undergo a massive amount of trial and error to master these skills.

This brings several problems:

## (1) Slow Learning Speed

Reinforcement learning inherently relies on continuous exploration of the environment.

For a real robot, it might take hundreds of thousands or even millions of interactions to learn a good policy, and real robots cannot bear such high training costs.

---

## (2) Safety Risks in Exploration

To find the optimal policy, reinforcement learning constantly tries new actions, such as:

- Turn left?
- Turn right?
- Accelerate?
- Sharp turn?
- Should I crash to see what happens?

In a simulation environment, these explorations are fine; but on a real robot, frequent collisions will damage the equipment.

---

## (3) Existing Controllers are Already Good

Many robots actually already possess mature control algorithms, such as:

- PID
- MPC (Model Predictive Control)
- Pure Pursuit
- Stanley
- DWA
- A* Path Planning

These algorithms are capable of basic navigation tasks.

If reinforcement learning is forced to relearn the entire control policy, it is equivalent to discarding existing experience, which is not only inefficient but might also lead to degraded performance.

---

Therefore, researchers proposed a new idea:

> **Instead of having reinforcement learning relearn the entire controller, have it make corrections based on an existing controller.**

This is the core idea of Residual Reinforcement Learning.

---

# II. The Meaning of Residual

The idea of Residual originally comes from **ResNet (Residual Neural Network)**.

In a normal neural network, we hope to learn a complete mapping:

$y=F(x)$

ResNet, however, does not learn the entire mapping, but learns the difference (Residual) between the input and the output:

$y=x+F(x)$

That is to say:

> **Output = Original Result + A Correction Amount**

Residual RL introduces this idea into reinforcement learning control.

---

Normal reinforcement learning directly learns:

$u=\pi(s)$

Where:

- $s$ is the State
- $\pi$ is the policy network
- $u$ is the control action

Whereas Residual RL learns:

$u=u_{base}+u_{RL}$

Where:

- $u_{base}$: Traditional controller output (PID, MPC, etc.)
- $u_{RL}$: The residual control amount learned by reinforcement learning

Therefore:

> **Reinforcement learning is not responsible for the entire control, but only for correcting the traditional controller.**

---

# III. Mathematical Expression

Assume the robot\'s current state is $s$.

The traditional controller calculates:

$u_{base}=Controller(s)$

The reinforcement learning network outputs:

$u_{RL}=f_{\theta}(s)$

The final control amount sent to the robot for execution is:

$u=u_{base}+u_{RL}$

This can also be written as:

$\pi(s)=Controller(s)+f_{\theta}(s)$

Where:

$Controller$ is the traditional control algorithm;
$f_{\theta}$ is the neural network;
$\theta$ are the network parameters.

Therefore, what reinforcement learning truly learns is:

> **How much the traditional controller is lacking.**

---

# IV. Overall Framework of Residual RL

```text
                Camera
                   │
                 CNN
                   │
              State Features
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
Traditional Controller      RL Policy Network
     (PID/MPC)               │
         │                   │
   Base Action          Residual Action
         └─────────┬─────────┘
                   ▼
          Final Action
                   │
                   ▼
                Robot
```

The final control amount consists of two parts:

$Final\ Action = Base\ Action + Residual\ Action$

---

# V. A Robot Car Example

Assume the robot is driving along the center of a lane.

The traditional PID controller calculates:

> Steering Angle = **15°**

At this time, the camera finds:

- There is a puddle ahead;
- There are obstacles on the side of the road;
- The optimal route should veer slightly to the left.

The reinforcement learning network outputs:

> Residual = **+3°**

So the final control amount becomes:

$15^\circ+3^\circ=18^\circ$

You can observe:

PID ensures the robot can drive stably, while reinforcement learning is only responsible for fine adjustments.

---

# VI. How Does the Network Learn?

During training, the robot first acquires various sensor information, such as:

- Camera Image
- IMU
- Encoder
- LiDAR (if any)

This information forms the state:

$State=(Image,IMU,Encoder,\cdots)$

Then:

The traditional controller outputs:

$u_{base}$

Reinforcement learning outputs:

$\Delta u$

Final execution:

$u=u_{base}+\Delta u$

After the robot executes the action, the environment returns a reward (Reward), for example:

- Whether it reached the target;
- Whether a collision occurred;
- Whether it stayed in the center of the lane;
- Whether the control was smooth;
- Whether the energy consumption was low.

Reinforcement learning continuously updates the network parameters:

$\theta$

While the traditional controller remains unchanged.

Therefore:

> **RL learns *when* it needs to correct the traditional controller.**

---

# VII. Reward Function Design

Robot navigation usually combines multiple objectives.

For example:

$Reward = Goal + Center + Velocity - Collision - Energy$

A simple design could be written as:

$R= 1.0\times Goal + 0.5\times Center + 0.2\times Velocity - 10\times Collision$

Where:

- Goal: Whether it is close to the target;
- Center: The error from the lane center;
- Velocity: Maintaining a reasonable speed;
- Collision: Collision penalty;
- Energy: Energy consumption or the severity of actions.

Residual RL automatically learns:

- When to correct more;
- When to correct less;
- How to correct to get the highest reward.

---

# VIII. Why is Residual RL Easier to Train?

Normal reinforcement learning needs to learn the full action space.

For example:

Steering range:

$[-30^\circ,30^\circ]$

Residual RL only needs to learn:

$[-5^\circ,5^\circ]$

Because:

The traditional controller has already done most of the work.

Reinforcement learning only needs to learn a small range of corrections.

Therefore:

- The search space is smaller;
- The convergence speed is faster;
- The sample efficiency is higher;
- It is easier to obtain a stable policy.

---

# IX. Advantages of Residual RL

## (1) Faster Training Speed

Reinforcement learning doesn\'t need to relearn all control laws; it just learns how to optimize the existing controller.

---

## (2) Much Safer

Even if the reinforcement learning output is:

$u_{RL}=0$

The robot can still rely on the traditional controller to operate normally.

It will not completely lose control just because the reinforcement learning hasn\'t finished training.

---

## (3) Higher Stability

The traditional controller is responsible for:

- Ensuring system stability;
- Maintaining basic control performance.

Reinforcement learning is responsible for:

- Optimizing the control effect;
- Improving environmental adaptability.

The two complement each other, so the control is much more stable.

---

## (4) Higher Sample Efficiency

Reinforcement learning doesn\'t need to relearn:

- How to go straight;
- How to stop;
- How to slow down;

It only needs to learn:

> **When to make corrections.**

---

# X. Disadvantages of Residual RL

Of course, it also has certain limitations.

## (1) The Base Controller Cannot Be Too Bad

If the PID itself has a very large error, for example:

> The path error has reached 50 cm

Then the small corrections from reinforcement learning will find it hard to fully compensate for the controller\'s own problems.

---

## (2) Easily Restricted by the Base Policy

If the traditional controller can never bypass a certain obstacle,

Residual RL might also fail to learn a completely new control policy due to its limited correction range.

---

## (3) Residual Weights Need Adjusting

Many practical systems adopt:

$u=u_{base}+\alpha u_{RL}$

Where:

$\alpha$ is the residual weight.

If:

- $\alpha$ is too large, reinforcement learning can easily disrupt system stability;
- $\alpha$ is too small, reinforcement learning will barely have any effect.

Therefore, it needs to be adjusted according to the task.

---

# XI. Typical Applications

In recent years, a massive number of robotics papers have adopted Residual RL.

For example:

| Application Scenario | Base Controller | Residual RL Role |
|---|---|---|
| Robot Grasping | Inverse Kinematics | Correct grasping trajectory |
| Autonomous Driving | Pure Pursuit | Correct steering control |
| Mobile Robots | MPC | Optimize path tracking |
| Drones | PID | Correct attitude control |
| Robotic Arms | Impedance Control | Improve operational precision |

Currently, it has become an important research direction for **traditional control + deep learning integration** in the robotics field.

---

# XII. How to Combine With Your Car Project

For your robot car, you can adopt the following overall architecture.

```text
Camera ─────────────┐
                    │
IMU ────────────────┤
                    │
Encoder ────────────┤
                    ▼
             State Feature Extraction
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
   Pure Pursuit / MPC / PID     RL Policy Network
          │                    │
      Base Action        Residual Action
          └────────┬───────────┘
                   ▼
             Final Control (Speed, Steering)
                   │
                   ▼
                Robot
                   │
                   ▼
      Reward (Obstacle Avoidance, Path Tracking, Smoothness)
```

In this system:

- The **Visual Branch** is responsible for extracting road, obstacle, and target information;
- The **Traditional Controller** ensures the robot has basic navigation capabilities;
- **Residual RL** dynamically corrects the control amount based on visual and sensor information;
- The **Reward Function** comprehensively considers safety, trajectory error, speed, and control smoothness to continuously optimize the policy.

This architecture provides both high safety and good learning capability, making it very suitable for real robot deployment.

---

# XIII. Conclusion

The core idea of Residual Reinforcement Learning can be summarized in one sentence:

> **Reinforcement learning no longer learns control from scratch, but learns a "Residual" based on an existing controller, i.e., how to optimally correct the traditional controller.**

Compared to traditional reinforcement learning, it has the following advantages:

- **Faster training speed**: Simpler learning objective, smaller search space;
- **Higher safety**: The traditional controller always provides stable baseline control;
- **Higher sample efficiency**: No need to relearn basic control laws;
- **Easier engineering deployment**: Can be directly combined with mature control algorithms like PID, MPC, Pure Pursuit.

For your **visual navigation robot car** project, Residual RL can excellently integrate **deep learning visual perception, sensor fusion, traditional control, and reinforcement learning optimization**, and it is also one of the most engineering-valuable and promising technological routes in current robotics reinforcement learning research.