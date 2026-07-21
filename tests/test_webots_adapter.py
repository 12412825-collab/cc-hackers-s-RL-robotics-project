import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakeMotor:
    def __init__(self):
        self.velocity = None

    def setPosition(self, position):
        self.position = position

    def setVelocity(self, velocity):
        self.velocity = velocity


class FakeEncoder:
    def __init__(self):
        self.value = 0.0

    def enable(self, timestep):
        self.timestep = timestep

    def getValue(self):
        return self.value


class FakeVectorSensor:
    def __init__(self, values):
        self.values = values

    def enable(self, timestep):
        self.timestep = timestep

    def getValues(self):
        return self.values


class FakeDistanceSensor(FakeEncoder):
    pass


class FakeCamera:
    def enable(self, timestep):
        self.timestep = timestep

    def getImage(self):
        return np.zeros((2, 3, 4), dtype=np.uint8).tobytes()

    def getHeight(self):
        return 2

    def getWidth(self):
        return 3


class FakeNode:
    def __init__(self):
        self.position = [0.1, 0.03, 0.02]

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]

    def getField(self, name):
        node = self
        class Field:
            def setSFVec3f(self, value):
                node.position = list(value)
            def setSFRotation(self, value):
                pass
        return Field()

    def resetPhysics(self):
        pass


class FakeSupervisor:
    devices = {}

    def __init__(self):
        motors = [
            'front left wheel motor', 'rear left wheel motor',
            'front right wheel motor', 'rear right wheel motor']
        encoders = [
            'front left wheel sensor', 'rear left wheel sensor',
            'front right wheel sensor', 'rear right wheel sensor']
        self.devices = {name: FakeMotor() for name in motors}
        self.devices.update({name: FakeEncoder() for name in encoders})
        self.devices['camera'] = FakeCamera()
        self.devices['accelerometer'] = FakeVectorSensor([0.0, 9.80665, 0.0])
        self.devices['gyro'] = FakeVectorSensor([0.0, 0.0, 1.0])
        self.devices['front distance sensor'] = FakeDistanceSensor()
        self.devices['front distance sensor'].value = 0.5

    def getBasicTimeStep(self):
        return 50

    def getDevice(self, name):
        return self.devices[name]

    def getSelf(self):
        if not hasattr(self, "node"):
            self.node = FakeNode()
        return self.node

    def step(self, timestep):
        return 0


class WebotsAdapterTests(unittest.TestCase):
    def test_four_motors_receive_left_right_kinematic_commands(self):
        fake_controller = types.ModuleType('controller')
        fake_controller.Supervisor = FakeSupervisor
        config = SimpleNamespace(
            WEBOTS_TIMESTEP_MS=50,
            WHEEL_RADIUS=0.0325,
            WHEEL_SEPARATION=0.130,
            MAX_WHEEL_SPEED=12.0,
            WEBOTS_LEFT_MOTOR='unused',
            WEBOTS_RIGHT_MOTOR='unused',
            WEBOTS_LEFT_MOTORS=[
                'front left wheel motor', 'rear left wheel motor'],
            WEBOTS_RIGHT_MOTORS=[
                'front right wheel motor', 'rear right wheel motor'],
            WEBOTS_LEFT_ENCODERS=[
                'front left wheel sensor', 'rear left wheel sensor'],
            WEBOTS_RIGHT_ENCODERS=[
                'front right wheel sensor', 'rear right wheel sensor'],
            WEBOTS_CAMERA='camera',
            WEBOTS_ACCELEROMETER='accelerometer',
            WEBOTS_GYRO='gyro',
            WEBOTS_DISTANCE_SENSOR='front distance sensor',
            WEBOTS_DISTANCE_TO_CM=100.0,
            WEBOTS_CTE_AXIS='z',
            WEBOTS_TRACK_CENTER=0.0,
            IMAGE_H=2,
            IMAGE_W=3,
            IMAGE_DEPTH=3,
        )

        with patch.dict(sys.modules, {'controller': fake_controller}):
            from simulation.webots_adapter import WebotsAdapter
            adapter = WebotsAdapter(config)
            outputs = adapter.run(0.2, 1.0)

        expected_left = (0.2 - 1.0 * 0.065) / 0.0325
        expected_right = (0.2 + 1.0 * 0.065) / 0.0325
        self.assertTrue(all(motor.velocity == expected_left
                            for motor in adapter.left_motors))
        self.assertTrue(all(motor.velocity == expected_right
                            for motor in adapter.right_motors))
        self.assertEqual(len(outputs), 16)
        self.assertEqual(outputs[0].shape, (2, 3, 3))
        self.assertAlmostEqual(outputs[10], 50.0)
        self.assertAlmostEqual(outputs[15], 0.02)

    def test_geofence_reset_does_not_publish_exploded_state(self):
        fake_controller = types.ModuleType('controller')
        fake_controller.Supervisor = FakeSupervisor
        config = SimpleNamespace(
            WEBOTS_TIMESTEP_MS=50, WHEEL_RADIUS=0.0325,
            WHEEL_SEPARATION=0.130, MAX_WHEEL_SPEED=12.0,
            WEBOTS_LEFT_MOTOR='unused', WEBOTS_RIGHT_MOTOR='unused',
            WEBOTS_LEFT_MOTORS=['front left wheel motor', 'rear left wheel motor'],
            WEBOTS_RIGHT_MOTORS=['front right wheel motor', 'rear right wheel motor'],
            WEBOTS_LEFT_ENCODERS=['front left wheel sensor', 'rear left wheel sensor'],
            WEBOTS_RIGHT_ENCODERS=['front right wheel sensor', 'rear right wheel sensor'],
            WEBOTS_CAMERA='camera', WEBOTS_ACCELEROMETER='accelerometer',
            WEBOTS_GYRO='gyro', WEBOTS_DISTANCE_SENSOR='front distance sensor',
            WEBOTS_DISTANCE_TO_CM=100.0, WEBOTS_CTE_AXIS='z',
            WEBOTS_TRACK_CENTER=0.0, WEBOTS_GEOFENCE_M=45.0,
            WEBOTS_MIN_HEIGHT_M=-0.25, IMAGE_H=2, IMAGE_W=3, IMAGE_DEPTH=3)
        with patch.dict(sys.modules, {'controller': fake_controller}):
            from simulation.webots_adapter import WebotsAdapter
            adapter = WebotsAdapter(config)
            adapter.self_node.position = [0.1, 0.03, -1000.0]
            outputs = adapter.run(0.2, 0.0)
        self.assertEqual(outputs[13], 0.02)
        self.assertEqual(outputs[14], 0.0)
        self.assertEqual(outputs[15], 0.02)

    def test_dashboard_reset_request_restores_initial_pose(self):
        fake_controller = types.ModuleType('controller')
        fake_controller.Supervisor = FakeSupervisor
        config = SimpleNamespace(
            WEBOTS_TIMESTEP_MS=50, WHEEL_RADIUS=0.0325,
            WHEEL_SEPARATION=0.130, MAX_WHEEL_SPEED=12.0,
            WEBOTS_LEFT_MOTOR='unused', WEBOTS_RIGHT_MOTOR='unused',
            WEBOTS_LEFT_MOTORS=['front left wheel motor', 'rear left wheel motor'],
            WEBOTS_RIGHT_MOTORS=['front right wheel motor', 'rear right wheel motor'],
            WEBOTS_LEFT_ENCODERS=['front left wheel sensor', 'rear left wheel sensor'],
            WEBOTS_RIGHT_ENCODERS=['front right wheel sensor', 'rear right wheel sensor'],
            WEBOTS_CAMERA='camera', WEBOTS_ACCELEROMETER='accelerometer',
            WEBOTS_GYRO='gyro', WEBOTS_DISTANCE_SENSOR='front distance sensor',
            WEBOTS_DISTANCE_TO_CM=100.0, WEBOTS_CTE_AXIS='z',
            WEBOTS_TRACK_CENTER=0.0, WEBOTS_GEOFENCE_M=45.0,
            WEBOTS_MIN_HEIGHT_M=-0.25, IMAGE_H=2, IMAGE_W=3, IMAGE_DEPTH=3)
        with patch.dict(sys.modules, {'controller': fake_controller}):
            from simulation.webots_adapter import WebotsAdapter
            from parts.simulation_control import request_reset
            adapter = WebotsAdapter(config)
            adapter.self_node.position = [4.0, 0.03, 2.0]
            request_reset()
            outputs = adapter.run(0.2, 0.0)
        self.assertEqual(outputs[11:14], (0.1, 0.03, 0.02))
        self.assertEqual(outputs[14], 0.0)


if __name__ == '__main__':
    unittest.main()

