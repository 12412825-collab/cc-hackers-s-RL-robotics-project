import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class WebotsAssetTests(unittest.TestCase):
    def test_proto_contains_four_motors_and_matching_sensors(self):
        proto = (ROOT / 'simulation' / 'protos' /
                 'FourWheelRobot.proto').read_text(encoding='utf-8')
        for position in ('front left', 'front right', 'rear left', 'rear right'):
            self.assertIn(f'name "{position} wheel motor"', proto)
            self.assertIn(f'name "{position} wheel sensor"', proto)
        self.assertEqual(proto.count('{'), proto.count('}'))
        self.assertEqual(proto.count('['), proto.count(']'))
        self.assertIn('type "sonar"', proto)
        self.assertIn('aperture 0.10', proto)
        self.assertNotIn('minRange', proto)

    def test_world_references_robot_proto(self):
        world = (ROOT / 'simulation' / 'worlds' /
                 'four_wheel_track.wbt').read_text(encoding='utf-8')
        self.assertIn('EXTERNPROTO "../protos/FourWheelRobot.proto"', world)
        self.assertIn('FourWheelRobot {', world)
        self.assertEqual(world.count('{'), world.count('}'))


if __name__ == '__main__':
    unittest.main()
