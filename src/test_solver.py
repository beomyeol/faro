import unittest

import numpy as np
import numpy.testing as npt

from solver import (
    Solver,
    SolverWithDrop,
    MMcQueue,
    _get_mdc_latency,
)
from utility import LatencyUtility


class MMCQueueTest(unittest.TestCase):

    def test_percentiles(self):
        # data source: https://www.sciencedirect.com/science/article/pii/S1434841105001846
        # rho = 0.8
        arrival = 8
        departure = 1.0
        capacity = 10.0
        queue = MMcQueue(arrival, departure, capacity)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(80), 0.358, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(90), 0.704, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(95), 1.051, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(99), 1.856, places=3)

        # rho = 0.95
        arrival = 9.5
        departure = 1.0
        capacity = 10.0
        queue = MMcQueue(arrival, departure, capacity)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(80), 2.836, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(90), 4.222, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(95), 5.608, places=3)
        self.assertAlmostEqual(
            queue.getPercentileQueueTime(99), 8.827, places=3)


class MDCLatencyTest(unittest.TestCase):

    def test_stable(self):
        input_rates = np.array([19.0, 19.0])
        processing_times = np.array([0.15, 0.15])
        x = np.array([3.0, 3.0])
        latencies = _get_mdc_latency(x, input_rates, processing_times, 90)
        self.assertEqual(latencies.ndim, 1)
        npt.assert_almost_equal(latencies, [1.252, 1.252], decimal=3)

    def test_unstable_1(self):
        input_rates = np.array([20.0])
        processing_times = np.array([0.15])
        x = np.array([3.0])
        latencies = _get_mdc_latency(x, input_rates, processing_times, 90)
        self.assertEqual(latencies.ndim, 1)
        npt.assert_array_almost_equal(latencies, [1.318], decimal=3)

    def test_unstable_2(self):
        input_rates = np.array([30.0])
        processing_times = np.array([0.15])
        x = np.array([3.0])
        latencies = _get_mdc_latency(x, input_rates, processing_times, 90)
        self.assertEqual(latencies.ndim, 1)
        npt.assert_almost_equal(latencies, [1.978], decimal=3)

    def test_stable_2d(self):
        input_rates = np.array([[19.0, 17.0, 17.0], [19.0, 17.0, 17.0]])
        processing_times = np.array([0.15, 0.15])
        x = np.array([3.0, 3.0])
        latencies = _get_mdc_latency(x, input_rates, processing_times, 90)
        npt.assert_almost_equal(
            latencies, [[1.252, 0.481, 0.481], [1.252, 0.481, 0.481]], decimal=3)

    def test_partially_stable_2d(self):
        input_rates = np.array([[30.0, 17.0], [30.0, 17.0]])
        processing_times = np.array([0.15, 0.15])
        x = np.array([3.0, 3.0])
        latencies = _get_mdc_latency(x, input_rates, processing_times, 90)
        npt.assert_almost_equal(
            latencies, [[1.978, 0.481], [1.978, 0.481]], decimal=3)


class SolverTest(unittest.TestCase):

    def create_solver(self, *args, **kwargs):
        return Solver(
            resource_limit=32,
            *args, **kwargs
        )

    def test_1d_input_rate(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla")
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=26,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=33,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        self.assertAlmostEqual(value, [1.0, 1.0])
        self.assertEqual(sol["cluster1"], 7)
        self.assertEqual(sol["cluster2"], 9)

    def test_1d_input_rate_with_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=26,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=33,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_array_almost_equal(value, [1.0, 1.0])
        self.assertEqual(sol["cluster1"], 5)
        self.assertEqual(sol["cluster2"], 6)

    def test_2d_input_rate(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla")
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        self.assertAlmostEqual(value, [1.0, 1.0])
        self.assertEqual(sol["cluster1"], 7)
        self.assertEqual(sol["cluster2"], 9)

    def test_2d_input_rate_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        self.assertEqual(sol["cluster1"], 5)
        self.assertEqual(sol["cluster2"], 6)

    def test_2d_input_rate_mdc_difflen_rates(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        self.assertEqual(sol["cluster1"], 5)
        self.assertEqual(sol["cluster2"], 6)

    def test_reshape_x(self):
        old_x = np.array([1.0, 2.0, 2.0])
        x = np.array([2.0, 1.0, 3.0])
        x = Solver._reshape_x(x, old_x, upscale_overhead=2, dim=4)
        npt.assert_almost_equal(
            x,
            np.array([
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
            ])
        )

    def test_calculate_utility_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(np.array([5.0, 6.0]))
        npt.assert_almost_equal(utilities, np.array([1.0, 1.0]))

    def test_calculate_utility_mdc_with_overhead(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95,
            upscale_overhead=1)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(np.array([5.0, 6.0]))
        npt.assert_almost_equal(utilities, np.array([0.677, 0.675]), decimal=3)

    def test_calculate_utility(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(np.array([7.0, 9.0]))
        npt.assert_almost_equal(utilities, np.array([1.0, 1.0]))

    def test_calculate_utility_with_overhead(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla",
            upscale_overhead=1)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(np.array([7.0, 9.0]))
        npt.assert_almost_equal(utilities, np.array([0.718, 0.707]), decimal=3)


class SolverWithDropTest(unittest.TestCase):

    def create_solver(self, *args, **kwargs):
        return SolverWithDrop(
            resource_limit=32, drop_integrality=True,
            *args, **kwargs
        )

    def test_1d_input_rate(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="de")
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=26,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=33,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        self.assertAlmostEqual(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (7, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (9, 1), decimal=3)

    def test_1d_input_rate_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="de", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=26,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=33,
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        import time
        sol, value = solver.solve()
        self.assertAlmostEqual(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (5, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (6, 1), decimal=3)

    def test_2d_input_rate(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla")
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (7, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (9, 1), decimal=3)

    def test_2d_input_rate_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (5, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (6, 1), decimal=3)

    def test_2d_input_rate_mdc_difflen_rates(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (5, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (6, 1), decimal=3)

    def test_2d_input_rate_mdc_difflen_rates_with_zeros(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[0],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        sol, value = solver.solve()
        npt.assert_almost_equal(value, [1.0, 1.0])
        npt.assert_almost_equal(sol["cluster1"], (5, 1), decimal=3)
        npt.assert_almost_equal(sol["cluster2"], (1, 1), decimal=3)

    def test_calculate_utility_mdc(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(
            np.array([5.0, 6.0, 100.0, 100.0]))
        npt.assert_almost_equal(utilities, np.array([1.0, 1.0]))

    def test_calculate_utility_mdc_with_overhead(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95,
            upscale_overhead=1)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(
            np.array([5.0, 6.0, 100.0, 100.0]))
        npt.assert_almost_equal(utilities, np.array([0.677, 0.675]), decimal=3)

    def test_calculate_utility(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla", mdc_percentile=95)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(
            np.array([7.0, 9.0, 100.0, 100.0]))
        npt.assert_almost_equal(utilities, np.array([1.0, 1.0]))

    def test_calculate_utility_with_overhead(self):
        util_func = LatencyUtility(
            slo_target=592,  # 148 * 4
        )
        solver = self.create_solver(
            adjust=True, min_max=False, method="cobyla",
            upscale_overhead=1)
        solver.add_deployment(
            key="cluster1",
            processing_time=148,
            input_rate=[26, 20, 21],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver.add_deployment(
            key="cluster2",
            processing_time=148,
            input_rate=[33, 28, 10],
            weight=1.0,
            util_func=util_func,
            current_num_replicas=1,
            resource_per_replica=1,
        )
        solver._prepare()
        utilities = solver.calculate_utility(
            np.array([7.0, 9.0, 100.0, 100.0]))
        npt.assert_almost_equal(utilities, np.array([0.718, 0.707]), decimal=3)


if __name__ == "__main__":
    unittest.main()
