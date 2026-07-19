It should be placed in the final layer.

For example

```text
Camera
↓

MobileNet

↓

Feature

↓

Fusion

↓

Policy

↓

Action
```

Imitation Learning Stage

```text
Loss

= Human steering angle

vs

Network output
```

After training is complete

**The network structure remains completely unchanged.**

Just swap out the Loss.

It becomes

```text
Reward

↓

PPO updates Policy
```

So:

You are not rewriting a model.

Just

**Changing the training method.**