import unittest

from parts.differential_drive import (
    DifferentialDriveKinematics,
    VelocityDriveMode,
    VelocityToNormalizedControl,
)


class DifferentialDriveTests(unittest.TestCase):
    def test_straight_motion_has_equal_wheel_rates(self):
        mixer = DifferentialDriveKinematics(0.05, 0.20)
        left, right = mixer.run(0.5, 0.0)
        self.assertAlmostEqual(left, 10.0)
        self.assertAlmostEqual(right, 10.0)

    def test_rotation_has_opposite_wheel_rates(self):
        mixer = DifferentialDriveKinematics(0.05, 0.20)
        left, right = mixer.run(0.0, 2.0)
        self.assertAlmostEqual(left, -4.0)
        self.assertAlmostEqual(right, 4.0)

    def test_saturation_preserves_curvature(self):
        mixer = DifferentialDriveKinematics(0.05, 0.20, max_wheel_speed=5.0)
        left, right = mixer.run(0.5, 1.0)
        self.assertAlmostEqual(max(abs(left), abs(right)), 5.0)
        self.assertAlmostEqual(right / left, 1.5)

    def test_residual_changes_only_angular_velocity(self):
        mode = VelocityDriveMode(0.6, 2.5, ai_throttle_mult=0.5)
        linear, angular = mode.run(
            'local', 0.0, 0.0, 0.2, 0.8, residual_omega=0.4)
        self.assertAlmostEqual(linear, 0.24)
        self.assertAlmostEqual(angular, 0.9)

    def test_legacy_conversion_round_trip(self):
        mode = VelocityDriveMode(0.6, 2.5)
        legacy = VelocityToNormalizedControl(0.6, 2.5)
        linear, angular = mode.run('user', -0.4, 0.75, None, None)
        steering, throttle = legacy.run(linear, angular)
        self.assertAlmostEqual(steering, -0.4)
        self.assertAlmostEqual(throttle, 0.75)


if __name__ == '__main__':
    unittest.main()

