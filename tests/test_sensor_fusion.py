import unittest

import numpy as np

from parts.sensors import SensorConfig, SensorFusion


class SensorFusionTests(unittest.TestCase):
    def test_single_run_produces_one_normalized_observation(self):
        config = SensorConfig(
            ENABLE_ENCODER=True,
            ENABLE_IMU=True,
            ENABLE_OBSTACLE=True,
            ENCODER_MAX_SPEED=2.0,
            IMU_ACCEL_RANGE=2.0,
            IMU_GYRO_RANGE=200.0,
            OBSTACLE_MIN_DIST=0.0,
            OBSTACLE_MAX_DIST=100.0,
        )
        fusion = SensorFusion(config)
        observation = fusion.run(
            1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 100.0,
            100.0,
        )
        self.assertEqual(observation.shape, (9,))
        self.assertEqual(fusion.update_count, 1)
        self.assertTrue(np.all(observation >= -1.0))
        self.assertTrue(np.all(observation <= 1.0))
        self.assertAlmostEqual(observation[0], 0.5)
        self.assertAlmostEqual(observation[8], 1.0)


if __name__ == '__main__':
    unittest.main()

