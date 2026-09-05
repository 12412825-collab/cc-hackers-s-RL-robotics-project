"""Live Webots episode environment for Phase-1A-R research adapter."""

from __future__ import annotations

from typing import Any, Optional

from .controller import HeadingPController, ResidualHook
from .estimator import HeadingEstimator
from .heading_source import GyroHeadingIntegrator
from .mismatch import MismatchConfig, angle_diff_rad, make_mismatch
from .mismatch_hooks import MismatchHooks
from .plant_backend import LiveWebotsBackend
from .types import ControllerObservation, LiveObservation, PrivilegedEvalState

# Known spawn heading from world config (not Supervisor runtime readout).
SPAWN_HEADING_RAD = 0.0
HEADING_SOURCE = "gyro_integration"


class LiveWebotsEnv:
    """Gym-like API over live Webots ODE + historical control building blocks."""

    def __init__(
        self,
        cruise_linear_velocity: float = 0.12,
        heading_kp: float = 2.0,
        max_angular_velocity: float = 1.50,
        max_linear_velocity: float = 0.20,
        residual_scale: float = 0.75,
        fusion_weight: float = 0.85,
        max_steps: int = 80,
        segment_length_m: float = 1.0,
        corridor_half_width_m: float = 0.25,
        max_heading_error_rad: float = 0.785,
        backend: Optional[LiveWebotsBackend] = None,
        mismatch: Optional[MismatchConfig] = None,
        spawn_heading_rad: float = SPAWN_HEADING_RAD,
    ):
        self.backend = backend or LiveWebotsBackend()
        self.cruise_v = float(cruise_linear_velocity)
        self.max_steps = int(max_steps)
        self.segment_length_m = float(segment_length_m)
        self.corridor_half_width_m = float(corridor_half_width_m)
        self.max_heading_error_rad = float(max_heading_error_rad)
        self.heading_kp = float(heading_kp)
        self.max_angular_velocity = float(max_angular_velocity)
        self.spawn_heading_rad = float(spawn_heading_rad)

        self.gyro_heading = GyroHeadingIntegrator(self.backend.dt, self.spawn_heading_rad)
        self.encoder_heading = GyroHeadingIntegrator(self.backend.dt, self.spawn_heading_rad)
        # Contrast integrator for secondary path diagnostics (integrates biased rate)
        self._biased_rate_heading = GyroHeadingIntegrator(
            self.backend.dt, self.spawn_heading_rad
        )

        self.estimator = HeadingEstimator(
            fusion_weight=fusion_weight, dt=self.backend.dt
        )
        self.estimator.enable_adaptation(False)
        self.base_controller = HeadingPController(
            kp=heading_kp, omega_max=max_angular_velocity
        )
        self.residual = ResidualHook(
            scale_rad_s=residual_scale,
            max_angular_velocity=max_angular_velocity,
            max_linear_velocity=max_linear_velocity,
        )
        self.mismatch = MismatchHooks()
        self.set_mismatch(mismatch or make_mismatch("none"))
        self.seed = 0
        self.episode = 0
        self.condition = "nominal_A0_none"
        self._step_count = 0
        self._start_x = 0.0
        self._log: list[dict[str, Any]] = []
        self._phi0: dict[str, float] = {}
        self.open_loop_omega: Optional[float] = None

    def set_mismatch(self, config: MismatchConfig) -> None:
        self.mismatch.set_config(config)
        self.mismatch.layer.assert_no_cross_contamination()

    def close(self) -> None:
        self.backend.apply_wheel_speeds(0.0, 0.0)

    def reset(
        self,
        seed: int = 0,
        condition: str = "nominal_A0_none",
        mismatch: Optional[MismatchConfig] = None,
    ) -> ControllerObservation:
        self.seed = int(seed)
        self.condition = str(condition)
        self.episode += 1
        self._step_count = 0
        self._log = []
        if mismatch is not None:
            self.set_mismatch(mismatch)

        self.estimator.enable_adaptation(False)
        self.residual.enable_adaptation(False)
        self.residual.reset()
        assert abs(self.residual.residual_omega_rad_s) < 1e-15
        assert self.estimator.adaptation_enabled is False

        self.backend.reset_physics_state()
        sens = self.backend.read_sensors()

        # Reset heading integrators from known spawn config (not Supervisor).
        self.gyro_heading.reset(self.spawn_heading_rad)
        self.encoder_heading.reset(self.spawn_heading_rad)
        self._biased_rate_heading.reset(self.spawn_heading_rad)

        self.estimator.unlock()
        self.estimator.reset(heading0=self.spawn_heading_rad)
        self._phi0 = self.estimator.get_params()
        self.estimator.lock()

        self._start_x = sens.true_position_m[0]
        return self._controller_obs_from_state(
            sens,
            raw_heading=self.gyro_heading.heading_rad,
            observed_heading=self.gyro_heading.heading_rad,
            enc_heading=self.encoder_heading.heading_rad,
            raw_rate=sens.gyro_yaw_rate_rad_s,
            obs_rate=sens.gyro_yaw_rate_rad_s,
            rate_bias=0.0,
            fixed_bias=0.0,
        )

    def _encoder_yaw_rate(self, left_rad_s: float, right_rad_s: float) -> float:
        r = self.backend.kinematics.wheel_radius
        b = self.backend.kinematics.wheel_separation
        return (r * (right_rad_s - left_rad_s)) / b

    def _controller_obs_from_state(
        self,
        sens,
        *,
        raw_heading: float,
        observed_heading: float,
        enc_heading: float,
        raw_rate: float,
        obs_rate: float,
        rate_bias: float,
        fixed_bias: float,
    ) -> ControllerObservation:
        return ControllerObservation(
            sim_time_s=sens.sim_time_s,
            imu_accel_g=list(sens.accel_g),
            imu_gyro_deg_s=list(sens.gyro_deg_s),
            raw_imu_yaw_rate_rad_s=float(raw_rate),
            observed_imu_yaw_rate_rad_s=float(obs_rate),
            gyro_rate_bias_rad_s=float(rate_bias),
            raw_heading_rad=float(raw_heading),
            fixed_heading_bias_rad=float(fixed_bias),
            observed_heading_rad=float(observed_heading),
            encoder_heading_rad=float(enc_heading),
            encoder_left_rad_s=sens.left_rad_s,
            encoder_right_rad_s=sens.right_rad_s,
            encoder_speed_m_s=sens.speed_m_s,
            distance_cm=sens.distance_cm,
            heading_est_rad=float(self.estimator.heading_est_rad),
            heading_source=HEADING_SOURCE,
            estimator_params=self.estimator.get_params(),
        )

    def step(self, control_action: float = 0.0) -> tuple[ControllerObservation, dict, bool]:
        del control_action
        if self.backend.terminated:
            raise RuntimeError("Webots simulation terminated")
        if self.estimator.adaptation_enabled:
            raise RuntimeError("estimator adaptation must stay OFF")
        if abs(self.residual.residual_omega_rad_s) > 1e-15:
            raise RuntimeError("residual must stay 0")

        sens = self.backend.read_sensors()
        raw_rate = float(sens.gyro_yaw_rate_rad_s)
        enc_yaw_rate = self._encoder_yaw_rate(sens.left_rad_s, sens.right_rad_s)

        # Always integrate RAW gyro for raw_heading (sensor pipeline, no Supervisor).
        raw_heading = self.gyro_heading.update(raw_rate)
        enc_heading = self.encoder_heading.update(enc_yaw_rate)

        cfg = self.mismatch.layer.config
        rate_iv = self.mismatch.layer.apply_gyro_rate_bias(raw_rate)
        # Primary: fixed heading bias on raw integrated heading
        head_iv = self.mismatch.layer.apply_fixed_heading_bias(raw_heading)

        if cfg.type in ("gyro_rate_bias", "imu_bias"):
            # Secondary path: controller sees heading from biased-rate integration
            observed_heading = self._biased_rate_heading.update(
                rate_iv.observed_imu_yaw_rate_rad_s
            )
            observed_rate = rate_iv.observed_imu_yaw_rate_rad_s
            fixed_bias = 0.0
            self.estimator.unlock()
            heading_est = self.estimator.update(observed_rate, enc_yaw_rate)
            self.estimator.lock()
        else:
            # Primary / nominal / motor: heading-space path
            observed_heading = head_iv.observed_heading_rad
            observed_rate = raw_rate  # rate channel uncorrupted for primary M1
            fixed_bias = head_iv.fixed_heading_bias_rad
            self._biased_rate_heading.heading_rad = raw_heading  # keep in sync
            self.estimator.unlock()
            heading_est = self.estimator.update_from_headings(
                observed_heading, enc_heading
            )
            self.estimator.lock()

        if self.estimator.get_params() != self._phi0:
            raise RuntimeError("estimator φ mutated under adaptation OFF")

        if self.open_loop_omega is None:
            base_omega = self.base_controller(heading_est)
        else:
            base_omega = float(self.open_loop_omega)
        v, omega_final, residual_omega = self.residual.combine(self.cruise_v, base_omega)
        assert abs(residual_omega) < 1e-15

        cmd_l, cmd_r = self.backend.kinematics.run(v, omega_final)
        motor = self.mismatch.layer.apply_motor_gains(cmd_l, cmd_r)
        self.backend.apply_wheel_speeds(motor.applied_left_rad_s, motor.applied_right_rad_s)

        intervention = {
            "raw_imu_yaw_rate_rad_s": raw_rate,
            "gyro_rate_bias_rad_s": rate_iv.gyro_rate_bias_rad_s,
            "observed_imu_yaw_rate_rad_s": observed_rate,
            "raw_heading_rad": raw_heading,
            "fixed_heading_bias_rad": fixed_bias,
            "observed_heading_rad": observed_heading,
            "encoder_heading_rad": enc_heading,
            "heading_offset_rad": angle_diff_rad(observed_heading, raw_heading),
            "heading_source": HEADING_SOURCE,
            "requested_left_rad_s": motor.requested_left_rad_s,
            "requested_right_rad_s": motor.requested_right_rad_s,
            "motor_gain_left": motor.motor_gain_left,
            "motor_gain_right": motor.motor_gain_right,
            "applied_left_rad_s": motor.applied_left_rad_s,
            "applied_right_rad_s": motor.applied_right_rad_s,
            "clipped_left": motor.clipped_left,
            "clipped_right": motor.clipped_right,
            "pre_physics_true_position_m": list(sens.true_position_m),
            "pre_physics_true_yaw_rad": float(sens.true_yaw_rad),
            "estimator_params": dict(self._phi0),
            "heading_kp": self.heading_kp,
            "residual_omega_rad_s": residual_omega,
        }

        if not self.backend.step_physics():
            raise RuntimeError("Webots simulation terminated during step")

        sens2 = self.backend.read_sensors()
        self._step_count += 1

        ctrl = self._controller_obs_from_state(
            sens2,
            raw_heading=raw_heading,
            observed_heading=observed_heading,
            enc_heading=enc_heading,
            raw_rate=raw_rate,
            obs_rate=observed_rate,
            rate_bias=rate_iv.gyro_rate_bias_rad_s,
            fixed_bias=fixed_bias,
        )
        ctrl.heading_est_rad = float(heading_est)

        priv = PrivilegedEvalState(
            true_position_m=list(sens2.true_position_m),
            true_yaw_rad=float(sens2.true_yaw_rad),
            true_linear_speed_m_s=float(sens2.true_speed_m_s),
        )
        track_err = float(sens2.true_yaw_rad)
        progress = abs(sens2.true_position_m[0] - self._start_x)
        lateral = abs(sens2.true_position_m[2])
        done = self._step_count >= self.max_steps
        success = (
            progress >= self.segment_length_m * 0.5
            and abs(track_err) < self.max_heading_error_rad
            and lateral < self.corridor_half_width_m
        )

        obs = LiveObservation(
            controller=ctrl,
            privileged=priv,
            base_omega_rad_s=float(base_omega),
            residual_omega_rad_s=float(residual_omega),
            final_omega_rad_s=float(omega_final),
            linear_velocity_m_s=float(v),
            cmd_left_rad_s=float(cmd_l),
            cmd_right_rad_s=float(cmd_r),
            motor_gain_left=float(motor.motor_gain_left),
            motor_gain_right=float(motor.motor_gain_right),
            applied_left_rad_s=float(motor.applied_left_rad_s),
            applied_right_rad_s=float(motor.applied_right_rad_s),
            clipped_left=bool(motor.clipped_left),
            clipped_right=bool(motor.clipped_right),
            tracking_error_rad=track_err,
            episode=self.episode,
            seed=self.seed,
            condition=self.condition,
            mismatch_type=cfg.type,
            success=bool(success and done),
            done=done,
        )
        row = obs.to_log_row()
        row["intervention"] = intervention
        self._log.append(row)
        info = {
            "privileged_eval_only": True,
            "true_yaw_rad": priv.true_yaw_rad,
            "true_position_m": priv.true_position_m,
            "progress_m": progress,
            "residual_omega_rad_s": residual_omega,
            "estimator_params": self.estimator.get_params(),
            "intervention": intervention,
            "mismatch": cfg.to_dict(),
            "clip_fraction": self.mismatch.layer.clip_fraction,
            "plant": self.backend.provenance(),
        }
        return ctrl, info, done

    def get_metrics(self) -> dict[str, Any]:
        if not self._log:
            return {}
        yaws = [row["true_yaw_rad"] for row in self._log]
        residuals = [row["residual_control_omega_rad_s"] for row in self._log]
        offsets = [
            abs(angle_diff_rad(row["observed_heading_rad"], row["raw_heading_rad"]))
            for row in self._log
        ]
        return {
            "n_steps": len(self._log),
            "mean_abs_true_yaw_rad": float(sum(abs(y) for y in yaws) / len(yaws)),
            "final_true_yaw_rad": float(yaws[-1]),
            "mean_residual_abs": float(sum(abs(r) for r in residuals) / len(residuals)),
            "mean_abs_heading_offset_rad": float(sum(offsets) / len(offsets)),
            "final_position_m": self._log[-1]["true_position_m"],
            "estimator_params_final": self._log[-1]["estimator_parameters"],
            "clip_fraction": self.mismatch.layer.clip_fraction,
            "mismatch": self.mismatch.layer.config.to_dict(),
            "heading_source": HEADING_SOURCE,
            "plant": self.backend.provenance(),
        }

    def get_log(self) -> list[dict[str, Any]]:
        return list(self._log)

    def assert_controller_obs_firewall(self, ctrl: ControllerObservation) -> None:
        data = ctrl.to_dict()
        forbidden = ("true_yaw", "true_position", "privileged", "supervisor")
        blob = str(data).lower()
        for key in forbidden:
            if key in blob:
                raise AssertionError(f"ControllerObservation leaks privileged key: {key}")
        if data.get("heading_source") != HEADING_SOURCE:
            raise AssertionError("heading_source must be gyro_integration")
