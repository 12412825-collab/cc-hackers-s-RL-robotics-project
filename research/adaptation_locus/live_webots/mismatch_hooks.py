"""Backward-compatible facade over the Step-3 MismatchLayer. """

from __future__ import annotations

from .mismatch import MismatchConfig, MismatchLayer, make_mismatch


class MismatchHooks:
    """Legacy mutable-looking API used by Step-2 code; delegates to MismatchLayer."""

    def __init__(self, imu_bias_rad_s: float = 0.0, motor_delta: float = 0.0):
        if abs(imu_bias_rad_s) < 1e-15 and abs(motor_delta) < 1e-15:
            cfg = make_mismatch("none")
        elif abs(motor_delta) < 1e-15:
            cfg = make_mismatch("imu_bias", imu_bias_rad_s=imu_bias_rad_s)
        elif abs(imu_bias_rad_s) < 1e-15:
            cfg = make_mismatch("motor_asymmetry", motor_delta=motor_delta)
        else:
            raise ValueError("Step-3 forbids simultaneous M1 and M2 in one config")
        self._layer = MismatchLayer(cfg)

    @property
    def imu_bias_rad_s(self) -> float:
        return self._layer.config.effective_imu_bias_rad_s

    @imu_bias_rad_s.setter
    def imu_bias_rad_s(self, value: float) -> None:
        v = float(value)
        d = self._layer.config.motor_delta
        if abs(v) < 1e-15 and abs(d) < 1e-15:
            self._layer.set_config(make_mismatch("none"))
        else:
            self._layer.set_config(make_mismatch("imu_bias", imu_bias_rad_s=v))

    @property
    def motor_delta(self) -> float:
        return float(self._layer.config.motor_delta)

    @motor_delta.setter
    def motor_delta(self, value: float) -> None:
        d = float(value)
        b = self._layer.config.effective_imu_bias_rad_s
        if abs(d) < 1e-15 and abs(b) < 1e-15:
            self._layer.set_config(make_mismatch("none"))
        else:
            self._layer.set_config(make_mismatch("motor_asymmetry", motor_delta=d))

    def observe_imu_yaw_rate(self, true_gyro_yaw_rate_rad_s: float) -> float:
        return self._layer.apply_imu_bias(true_gyro_yaw_rate_rad_s).observed_imu_yaw_rate_rad_s

    def apply_motor_gains(self, left_cmd: float, right_cmd: float) -> tuple[float, float]:
        r = self._layer.apply_motor_gains(left_cmd, right_cmd)
        return r.applied_left_rad_s, r.applied_right_rad_s

    def is_nominal(self) -> bool:
        return self._layer.config.is_nominal()

    def set_config(self, config: MismatchConfig) -> None:
        self._layer.set_config(config)

    @property
    def layer(self) -> MismatchLayer:
        return self._layer
