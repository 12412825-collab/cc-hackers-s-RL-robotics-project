"""Backward-compatible facade over the MismatchLayer. """

from __future__ import annotations

from .mismatch import MismatchConfig, MismatchLayer, make_mismatch


class MismatchHooks:
    def __init__(
        self,
        imu_bias_rad_s: float = 0.0,
        motor_delta: float = 0.0,
        fixed_heading_bias_rad: float = 0.0,
        gyro_rate_bias_rad_s: float = 0.0,
    ):
        rate = gyro_rate_bias_rad_s if abs(gyro_rate_bias_rad_s) > 0 else imu_bias_rad_s
        if abs(fixed_heading_bias_rad) > 0:
            cfg = make_mismatch(
                "fixed_heading_bias", fixed_heading_bias_rad=fixed_heading_bias_rad
            )
        elif abs(rate) < 1e-15 and abs(motor_delta) < 1e-15:
            cfg = make_mismatch("none")
        elif abs(motor_delta) < 1e-15:
            cfg = make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=rate)
        elif abs(rate) < 1e-15 and abs(fixed_heading_bias_rad) < 1e-15:
            cfg = make_mismatch("motor_asymmetry", motor_delta=motor_delta)
        else:
            raise ValueError("Cannot combine observation and motor mismatches")
        self._layer = MismatchLayer(cfg)

    @property
    def imu_bias_rad_s(self) -> float:
        return self._layer.config.effective_gyro_rate_bias_rad_s

    @imu_bias_rad_s.setter
    def imu_bias_rad_s(self, value: float) -> None:
        v = float(value)
        d = self._layer.config.motor_delta
        if abs(v) < 1e-15 and abs(d) < 1e-15:
            self._layer.set_config(make_mismatch("none"))
        else:
            self._layer.set_config(make_mismatch("gyro_rate_bias", gyro_rate_bias_rad_s=v))

    @property
    def motor_delta(self) -> float:
        return float(self._layer.config.motor_delta)

    @motor_delta.setter
    def motor_delta(self, value: float) -> None:
        d = float(value)
        if abs(d) < 1e-15:
            self._layer.set_config(make_mismatch("none"))
        else:
            self._layer.set_config(make_mismatch("motor_asymmetry", motor_delta=d))

    def observe_imu_yaw_rate(self, true_gyro_yaw_rate_rad_s: float) -> float:
        return self._layer.apply_gyro_rate_bias(
            true_gyro_yaw_rate_rad_s
        ).observed_imu_yaw_rate_rad_s

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
