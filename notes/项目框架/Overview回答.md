
1. Primarily uses RGB images captured by an onboard camera mounted on the mobile robot, PEG images (uint8, 120×160×3, BGR); `user/angle`: Steering wheel angle; `user/throttle`: Throttle value.
2. DL Model
Cascaded Dual-Model Architecture. Base Model is KerasLinear, TensorFlow/Keras,
in RL part Residual Model is SAC, PyTorch
