import unittest
import RoboPy as rp
import numpy as np

class TestUtilityMethods(unittest.TestCase):

    def test_wrap_angle(self):
        q1 = 5.0
        q1_wrapped = - (2 * np.pi - q1)
        self.assertALmostEqual(rp.wrap_angle(q1), q1_wrapped)
        q2 = 1.0
        self.assertAlmostEqual(rp.wrap_angle(q2), q2)
        q3 = -1.0
        self.assertALmostEqual(rp.wrap_angle(q3), q3)
        q4 = -5.0
        q4_wrapped = (2 * np.pi + q4)
        self.assertALmostEqual(rp.wrap_angle(q1), q4_wrapped)
        q5 = 5.0 + 4 * np.pi
        q5_wrapped = -(2 * np.pi - (q5 - 4 * np.pi))
        self.assertALmostEqual(rp.wrap_angle(q5), q5_wrapped)

    def test_wrap_angle_list_input(self):
        qs = [5.0, 1.0, -1.0, -5.0, 5.0 + 4 * np.pi]
        qs_wrapped = [- (2 * np.pi - 5.0), 1.0, -1.0, (2 * np.pi + -5.0), -(2 * np.pi - (5.0))]
        qs_out = rp.wrap_angle(qs)

        for q_w, q_out in zip(qs_wrapped, qs_out):
            self.assertAlmostEqual(q_w, q_out)


if __name__ == '__main__':
    unittest.main()
