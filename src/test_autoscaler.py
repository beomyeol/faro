import unittest
from datetime import datetime, timedelta

import numpy.testing as npt

from autoscale import Action
from autoscaler import Autoscaler, HybridAutoscaler, calculate_num_replicas


class UtilFuncTest(unittest.TestCase):

    def test_calculate_num_replicas(self):
        self.assertEqual(
            3,
            calculate_num_replicas(
                Action(head_name="serve-cluster1-ray-head",
                       deployment_name="classifier",
                       target=3),
                current_replicas=1, min_replicas=1, max_replicas=10))
        self.assertEqual(
            4,
            calculate_num_replicas(
                Action(head_name="serve-cluster1-ray-head",
                       deployment_name="classifier",
                       delta=3),
                current_replicas=1, min_replicas=1, max_replicas=10))
        self.assertEqual(
            1,
            calculate_num_replicas(
                Action(head_name="serve-cluster1-ray-head",
                       deployment_name="classifier",
                       delta=-2),
                current_replicas=1, min_replicas=1, max_replicas=10))
        self.assertEqual(
            6,
            calculate_num_replicas(
                Action(head_name="serve-cluster1-ray-head",
                       deployment_name="classifier",
                       factor=3),
                current_replicas=2, min_replicas=1, max_replicas=10))
        self.assertEqual(
            1,
            calculate_num_replicas(
                Action(head_name="serve-cluster1-ray-head",
                       deployment_name="classifier",
                       delta=-2),
                current_replicas=3, min_replicas=1, max_replicas=10))


class AutoscalerTest(unittest.TestCase):

    def test_sort_actions_by_delta(self):
        num_replicas = {
            "serve-cluster1-ray-head": {"classifier": 1},
            "serve-cluster2-ray-head": {"classifier": 3},
            "serve-cluster3-ray-head": {"classifier": 2},
            "serve-cluster4-ray-head": {"classifier": 4},
        }
        action1 = Action(head_name="serve-cluster1-ray-head",
                         deployment_name="classifier",
                         target=4)  # delta: 3
        action2 = Action(head_name="serve-cluster2-ray-head",
                         deployment_name="classifier",
                         target=1)  # delta: -2
        action3 = Action(head_name="serve-cluster3-ray-head",
                         deployment_name="classifier",
                         delta=1)  # delta: 1
        action4 = Action(head_name="serve-cluster4-ray-head",
                         deployment_name="classifier",
                         delta=-3)  # delta: -3
        actions = [action1, action2, action3, action4]
        expected = [action4, action2, action3, action1]
        output = Autoscaler.sort_actions_by_delta(
            actions, num_replicas, 1, 10)
        self.assertEqual(expected, output)

        only_downscale_expected = [action2, action4, action1, action3]
        only_downscale_output = Autoscaler.sort_actions_by_delta(
            actions, num_replicas, 1, 10, only_downscale_first=True)
        self.assertEqual(only_downscale_expected, only_downscale_output)


class HybridAutoscalerTest(unittest.TestCase):

    def parse_datetime(self, datetime_str):
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S,%f").timestamp()

    def test_merge_input_rates(self):
        first_ts = self.parse_datetime("2023-03-27 19:09:31,537") - \
            self.parse_datetime("2023-03-27 19:09:11,437") + \
            8.360539674758911
        input_rates = [
            (self.parse_datetime("2023-03-27 19:09:01,414"), None),
            (self.parse_datetime("2023-03-27 19:09:11,437"),
             [[7, 20, 20, 19, 20, 19, 20, 18, 21], 8.360539674758911]),
            (self.parse_datetime("2023-03-27 19:09:21,453"),
             [[20, 19, 18, 21, 20, 19, 20, 20, 20, 19], 9.98317813873291]),
            (self.parse_datetime("2023-03-27 19:09:31,537"),
             [[1, 20, 20, 20, 20, 18, 20, 20, 20, 19, 21], 10.013054847717285])
        ]
        expected = [
            [7, 20, 20, 19, 20, 19, 20, 18, 21, 20, 19, 18, 21, 20, 19,
                20, 20, 20, 20, 20, 20, 20, 20, 18, 20, 20, 20, 19, 21],
            first_ts
        ]
        merged = HybridAutoscaler.merge_input_rates(input_rates)
        npt.assert_array_almost_equal(merged[0], expected[0])
        self.assertAlmostEqual(merged[1], expected[1])

    def test_aggregate_counts(self):
        input_data = [
            7, 20, 20, 19, 20, 19, 20, 18, 21, 20, 19, 18, 21, 20, 19, 20,
            20, 20, 20, 20, 20, 20, 20, 18, 20, 20, 20, 19, 21]
        expected = [66, 98, 98, 99, 98, 100]
        output = HybridAutoscaler.aggregate_counts(input_data, 5)
        npt.assert_array_equal(expected, output)

        expected2 = [262, 297]
        output2 = HybridAutoscaler.aggregate_counts(input_data, 15)
        npt.assert_array_equal(expected2, output2)

        input_data_small = [7, 20, 20, 19, 20, 19]
        expected_small = [105]
        output_small = HybridAutoscaler.aggregate_counts(input_data_small, 15)
        npt.assert_array_equal(expected_small, output_small)


if __name__ == "__main__":
    unittest.main()
