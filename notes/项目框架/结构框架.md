          Camera
             │
     MobileNetV2/EfficientNet
             │
        Image Feature
             │
             ├────────────┐
             │            │
      IMU / Encoder / Ultrasonic
             │
      Sensor Feature (MLP)
             │
             └────Fusion─────► Policy Network
                               │
                     Steering + Throttle