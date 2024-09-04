import unittest

from utility import (
    LatencyUtility,
    LatencyStepUtility,
    SquareLatencyUtility,
    SLOPenalty,
    SLOStrictPenalty,
    SLOLinearPenalty,
    SLOStrictLinearPenalty,
)


class LatencyUtilityTest(unittest.TestCase):

    def setUp(self):
        target_latency = 0.8
        self.util_fn = LatencyUtility(target_latency)

    def test_float(self):
        self.assertEqual(self.util_fn(0.0), 1)
        self.assertEqual(self.util_fn(0.8), 1)
        self.assertEqual(self.util_fn(0.7), 1)
        self.assertAlmostEqual(self.util_fn(1.2), 0.666666666)
        self.assertAlmostEqual(self.util_fn(1.6), 0.5)

    def test_nparray(self):
        self.assertAlmostEqual(self.util_fn([0.0, 0.0]), 1)
        self.assertAlmostEqual(self.util_fn([0.0, 1.6]), 0.75)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6]), 0.65)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6, 0.0]), 0.76666667)


class LatencyStepUtilityTest(unittest.TestCase):

    def setUp(self):
        target_latency = 0.8
        self.util_fn = LatencyStepUtility(target_latency)

    def test_float(self):
        self.assertEqual(self.util_fn(0.0), 1)
        self.assertEqual(self.util_fn(0.8), 1)
        self.assertEqual(self.util_fn(0.7), 1)
        self.assertAlmostEqual(self.util_fn(1.2), 0)
        self.assertAlmostEqual(self.util_fn(1.6), 0)

    def test_nparray(self):
        self.assertAlmostEqual(self.util_fn([0.0, 0.0]), 1)
        self.assertAlmostEqual(self.util_fn([0.0, 1.6]), 0.5)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6]), 0)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6, 0.0]), 0.3333333)


class SquareLatencyUtilityTest(unittest.TestCase):

    def setUp(self):
        target_latency = 0.8
        self.util_fn = SquareLatencyUtility(target_latency)

    def test_float(self):
        self.assertEqual(self.util_fn(0.0), 1)
        self.assertEqual(self.util_fn(0.8), 1)
        self.assertEqual(self.util_fn(0.7), 1)
        self.assertAlmostEqual(self.util_fn(1.2), 0.444444444)
        self.assertAlmostEqual(self.util_fn(1.6), 0.25)

    def test_nparray(self):
        self.assertAlmostEqual(self.util_fn([0.0, 0.0]), 1)
        self.assertAlmostEqual(self.util_fn([0.0, 1.6]), 0.625)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6]), 0.445)
        self.assertAlmostEqual(self.util_fn([1.0, 1.6, 0.0]), 0.63)


class SLOLPenaltyTest(unittest.TestCase):

    def test_cuts(self):
        penalty_fn = SLOPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.01), 0.0)
        self.assertAlmostEqual(penalty_fn(0.05), 0.25)
        self.assertAlmostEqual(penalty_fn(0.1), 0.5)

    def test_middle_points(self):
        penalty_fn = SLOPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.005), 0.0)
        self.assertAlmostEqual(penalty_fn(0.01), 0.0)
        self.assertAlmostEqual(penalty_fn(0.03), 0.25)
        self.assertAlmostEqual(penalty_fn(0.05), 0.25)
        self.assertAlmostEqual(penalty_fn(0.075), 0.5)
        self.assertAlmostEqual(penalty_fn(0.1), 0.5)
        self.assertAlmostEqual(penalty_fn(0.55), 1.0)


class SLOLinearPenaltyTest(unittest.TestCase):

    def test_cuts(self):
        penalty_fn = SLOLinearPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.01), 0.0)
        self.assertAlmostEqual(penalty_fn(0.05), 0.25)
        self.assertAlmostEqual(penalty_fn(0.1), 0.5)

    def test_middle_points(self):
        penalty_fn = SLOLinearPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.005), 0.0)
        self.assertAlmostEqual(penalty_fn(0.01), 0.0)
        self.assertAlmostEqual(penalty_fn(0.03), 0.125)
        self.assertAlmostEqual(penalty_fn(0.05), 0.25)
        self.assertAlmostEqual(penalty_fn(0.075), 0.375)
        self.assertAlmostEqual(penalty_fn(0.1), 0.5)
        self.assertAlmostEqual(penalty_fn(0.55), 0.75)


class SLOLStrictPenaltyTest(unittest.TestCase):

    def test_cuts(self):
        penalty_fn = SLOStrictPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.001), 0.0)
        self.assertAlmostEqual(penalty_fn(0.03), 0.25)
        self.assertAlmostEqual(penalty_fn(0.08), 0.5)

    def test_middle_points(self):
        penalty_fn = SLOStrictPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.0005), 0.0)
        self.assertAlmostEqual(penalty_fn(0.001), 0.0)
        self.assertAlmostEqual(penalty_fn(0.0155), 0.25)
        self.assertAlmostEqual(penalty_fn(0.03), 0.25)
        self.assertAlmostEqual(penalty_fn(0.055), 0.5)
        self.assertAlmostEqual(penalty_fn(0.08), 0.5)
        self.assertAlmostEqual(penalty_fn(0.54), 1.0)


class SLOStrictLinearPenaltyTest(unittest.TestCase):

    def test_cuts(self):
        penalty_fn = SLOStrictLinearPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.001), 0.0)
        self.assertAlmostEqual(penalty_fn(0.03), 0.25)
        self.assertAlmostEqual(penalty_fn(0.08), 0.5)

    def test_middle_points(self):
        penalty_fn = SLOStrictLinearPenalty()
        self.assertAlmostEqual(penalty_fn(0), 0.0)
        self.assertAlmostEqual(penalty_fn(0.0005), 0.0)
        self.assertAlmostEqual(penalty_fn(0.001), 0.0)
        self.assertAlmostEqual(penalty_fn(0.0155), 0.125)
        self.assertAlmostEqual(penalty_fn(0.03), 0.25)
        self.assertAlmostEqual(penalty_fn(0.055), 0.375)
        self.assertAlmostEqual(penalty_fn(0.08), 0.5)
        self.assertAlmostEqual(penalty_fn(0.54), 0.75)


if __name__ == "__main__":
    unittest.main()
