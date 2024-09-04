import itertools
import unittest

import numpy.testing as npt

from data import utils


class TestCalculateAggregate(unittest.TestCase):

    def test_case1(self):
        values = [1, 2, 3]
        tss = [0.1, 0.2, 1.2]
        out = utils.calculate_aggregate(values=values, tss=tss)
        npt.assert_array_equal(out, [1.5, 3])

    def test_case2(self):
        values = [1, 2, 3, 4]
        tss = [0.1, 0.2, 1.2, 3.0]
        out = utils.calculate_aggregate(values=values, tss=tss)
        npt.assert_array_equal(out, [1.5, 3, 0, 4])

    def test_case3(self):
        values = itertools.chain(*[[1, 2], [3], [4]])
        tss = itertools.chain(*[[0.1, 0.2], [1.2], [3.0]])
        out = utils.calculate_aggregate(values=values, tss=tss)
        npt.assert_array_equal(out, [1.5, 3, 0, 4])


if __name__ == "__main__":
    unittest.main()
