"""Live Webots episode environment for Phase-1A-R research adapter."""

from __future__ import annotations

import math
from typing import Any, Optional

from .controller import HeadingPController, ResidualHook
from .estimator import HeadingEstimator
from .mismatch_hooks import MismatchHooks
from .plant_backend import LiveWebotsBackend
from .types import ControllerObservation, LiveObservation, PrivilegedEvalState


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
    ):
        self.backend = backend or LiveWebotsBackend()
        self.cruise_v = float(cruise_linear_velocity)
        self.max_steps = int(max_steps)
        self.segment_length_m = float(segment_length_m)
        self.corridor_half_width_m = float(corridor_half_width_m)
        self.max_heading_error_rad = float(max_heading_error_rad)

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
        self.seed = 0
        self.episode = 0
        self.condition = "nominal_A0_none"
        self._step_count = 0
        self._start_x = 0.0
        self._log: list[dict[str, Any]] = []

    def close(self) -> None:
        self.backend.apply_wheel_speeds(0.0, 0.0)

    def reset(self, seed: int = 0, condition: str = "nominal_A0_none") -> ControllerObservation:
        self.seed = int(seed)
        self.condition = str(condition)
        self.episode += 1
        self._step_count = 0
        self._log = []

        # Adaptation OFF; residual OFF; mismatch nominal
        self.estimator.enable_adaptation(False)
        self.estimator.lock()  # prevent accidental φ writes during episode
        self.residual.enable_adaptation(False)
        self.residual.reset()
        if not self.mismatch.is_nominal():
            # Step 2 validation may temporarily set zeros explicitly
            pass

        self.backend.reset_physics_state()
        sens = self.backend.read_sensors()
        # Estimator init: do NOT seed from Supervisor true yaw (firewall).
        # Start at 0; closed-loop regulates estimated heading.
        self.estimator.unlock()
        self.estimator.reset(heading0=0.0)
        self.estimator.lock()

        self._start_x = sens.true_position_m[0]
        return self._controller_obs(sens)

    def _encoder_yaw_rate(self, left_rad_s: float, right_rad_s: float) -> float:
        # ω ≈ (v_r - v_l) / B ; v = r * wheel_ω
        r = 0.0325
        b = 0.130
        return (r * (right_rad_s - left_rad_s)) / b

    def _controller_obs(self, sens) -> ControllerObservation:
        imu_obs = self.mismatch.observe_imu_yaw_rate(sens.gyro_yaw_rate_rad_s)
        return ControllerObservation(
            sim_time_s=sens.sim_time_s,
            imu_accel_g=list(sens.accel_g),
            imu_gyro_deg_s=list(sens.gyro_deg_s),
            imu_gyro_yaw_rate_rad_s=float(imu_obs),
            encoder_left_rad_s=sens.left_rad_s,
            encoder_right_rad_s=sens.right_rad_s,
            encoder_speed_m_s=sens.speed_m_s,
            distance_cm=sens.distance_cm,
            heading_est_rad=float(self.estimator.heading_est_rad),
            estimator_params=self.estimator.get_params(),
        )

    def step(self, control_action: float = 0.0) -> tuple[ControllerObservation, dict, bool]:
        """One control cycle. control_action ignored while residual adaptation OFF."""
        del control_action  # residual stays 0 in Step 2
        if self.backend.terminated:
            raise RuntimeError("Webots simulation terminated")

        sens = self.backend.read_sensors()
        imu_obs = self.mismatch.observe_imu_yaw_rate(sens.gyro_yaw_rate_rad_s)
        enc_yaw = self._encoder_yaw_rate(sens.left_rad_s, sens.right_rad_s)

        # Estimator update (params frozen)
        self.estimator.unlock()
        heading_est = self.estimator.update(imu_obs, enc_yaw)
        self.estimator.lock()

        base_omega = self.base_controller(heading_est)
        v, omega_final, residual_omega = self.residual.combine(self.cruise_v, base_omega)

        cmd_l, cmd_r = self.backend.kinematics.run(v, omega_final)
        app_l, app_r = self.mismatch.apply_motor_gains(cmd_l, cmd_r)
        self.backend.apply_wheel_speeds(app_l, app_r)

        if not self.backend.step_physics():
            raise RuntimeError("Webots simulation terminated during step")

        sens2 = self.backend.read_sensors()
        self._step_count += 1

        ctrl = self._controller_obs(sens2)
        # Recompute heading display after update already applied
        ctrl.heading_est_rad = float(self.estimator.heading_est_rad)

        priv = PrivilegedEvalState(
            true_position_m=list(sens2.true_position_m),
            true_yaw_rad=float(sens2.true_yaw_rad),
            true_linear_speed_m_s=float(sens2.true_speed_m_s),
        )
        # Tracking error for metrics uses privileged yaw (eval only)
        track_err = float(sens2.true_yaw_rad)  # regulate to 0 heading
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
            applied_left_rad_s=float(app_l),
            applied_right_rad_s=float(app_r),
            tracking_error_rad=track_err,
            episode=self.episode,
            seed=self.seed,
            condition=self.condition,
            success=bool(success and done),
            done=done,
        )
        self._log.append(obs.to_log_row())
        info = {
            "privileged_eval_only": True,
            "true_yaw_rad": priv.true_yaw_rad,
            "true_position_m": priv.true_position_m,
            "progress_m": progress,
            "residual_omega_rad_s": residual_omega,
            "estimator_params": self.estimator.get_params(),
            "plant": self.backend.provenance(),
        }
        return ctrl, info, done

    def get_metrics(self) -> dict[str, Any]:
        if not self._log:
            return {}
        yaws = [row["true_yaw_rad"] for row in self._log]
        residuals = [row["residual_control_omega_rad_s"] for row in self._log]
        return {
            "n_steps": len(self._log),
            "mean_abs_true_yaw_rad": float(sum(abs(y) for y in yaws) / len(yaws)),
            "final_true_yaw_rad": float(yaws[-1]),
            "mean_residual_abs": float(sum(abs(r) for r in residuals) / len(residuals)),
            "final_position_m": self._log[-1]["true_position_m"],
            "estimator_params_final": self._log[-1]["estimator_parameters"],
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
