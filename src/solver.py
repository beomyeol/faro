import logging
import time
from typing import List, Optional, Union, Iterable

import scipy.optimize
import numpy as np
import numba
import pandas as pd

from utility import SLOStrictPenalty, SLOStrictLinearPenalty

_LOGGER = logging.getLogger(__name__)
_PRECISION = 4
_EPSILON = 1e-4

_OPT_OPTIONS = {
    # 'cobyla': {'rhobeg': 3.0, 'tol': 1e-3, 'maxiter': 2000},
    'cobyla': {'rhobeg': 3.0},
    'slsqp': {'eps': 1.0},
}


def _vectorized_objective(x, objective):
    if len(x.shape) == 1:
        return objective(x)
    num = x.shape[-1]
    return [objective(x[:, i]) for i in range(num)]


@numba.experimental.jitclass([
    ("_arrival", numba.float64),
    ("_departure", numba.float64),
    ("_capacity", numba.float64),
    ("_rou", numba.float64),
    ("_finalTerm", numba.float64),
    ("_p0", numba.float64),
    ("_pc", numba.float64),
    ("_probSum", numba.float64),
])
class MMcQueue:
    def __init__(self, arrival, departure, capacity):
        """
        Given the parameter of one M/M/c/c Queue,
        initialize the queue with these parameter and calculate some parameters.
        `_rou`:     Server Utilization
        `_p0`:      Probability of that there is no packets in the queue
        `_pc`:      Probability of that there is exactly `capacity` packets in the queue,
                    that is, all the server is busy.
        `_probSum`:  p0 + p1 + p2 + ... pc - pc
        `_finalTerm`: 1/(c!) * (arrival / departure)^c
        """
        if capacity * departure <= arrival:
            raise ValueError(
                "This Queue is unstable with the Input Parameters!!!")
        self._arrival = float(arrival)
        self._departure = float(departure)
        self._capacity = capacity
        self._rou = self._arrival / self._departure / self._capacity

        # init the parameter as if the capacity == 0
        powerTerm = 1.0
        factorTerm = 1.0
        preSum = 1.0
        # Loop through `1` to `self._capacity` to get each term and preSum
        for i in range(1, int(self._capacity) + 1):
            powerTerm *= self._arrival / self._departure
            factorTerm /= i
            preSum += powerTerm * factorTerm
        self._finalTerm = powerTerm * factorTerm
        preSum -= self._finalTerm
        self._p0 = 1.0 / (preSum + self._finalTerm / (1 - self._rou))
        self._pc = self._finalTerm * self._p0
        self._probSum = preSum * self._p0

    @property
    def arrival(self):
        return self._arrival

    @property
    def departure(self):
        return self._departure

    @property
    def capacity(self):
        return self._capacity

    def getQueueProb(self):
        """
        Return the probability when a packet comes, it needs to queue in the buffer.
        That is, P(W>0) = 1 - P(N < c)
        Also known as Erlang-C function
        """
        return 1.0 - self._probSum

    def getAvgQueueTime(self):
        """
        Return the average time of packets spending in the queue
        """
        return self.getQueueProb() / (self._capacity * self._departure - self._arrival)

    def getAvgResponseTime(self):
        """
        Return the average time of packets spending in the system (in service and in the queue)
        """
        return self.getAvgQueueTime() + 1.0 / self._departure

    def getPercentileQueueTime(self, percentile):
        queue_prob = self.getQueueProb()
        if queue_prob == 0.0:
            queue_prob = _EPSILON  # avoid division by zero
        return max(
            0,
            (self.getAvgQueueTime() / queue_prob *
             np.log((100 * queue_prob) / (100 - percentile))))


@numba.njit(cache=True)
def _get_mdc_latency_2d(x, input_rates, processing_times, percentile, max_rho):
    rho = input_rates / x * processing_times
    weights = np.ones_like(rho)
    for i, j in zip(*np.nonzero(rho > max_rho)):
        expected_input = max_rho * x[i, j] / processing_times[i, 0]
        weights[i, j] = expected_input / input_rates[i, j]
        input_rates[i, j] = expected_input
        rho[i, j] = expected_input / x[i, j] * processing_times[i, 0]
    assert np.all(rho <= max_rho + _EPSILON)
    latencies = np.zeros_like(input_rates, dtype=np.float64)
    for i in range(latencies.shape[0]):
        for j in range(latencies.shape[1]):
            queue = MMcQueue(
                input_rates[i, j], 1 / processing_times[i, 0], x[i, j])
            # approximate with the M/M/c queue
            # https://www.sciencedirect.com/science/article/pii/S1434841105001846
            if percentile >= 0:
                latencies[i, j] = (
                    queue.getPercentileQueueTime(percentile) / 2 + processing_times[i, 0])
            else:
                latencies[i, j] = (
                    queue.getAvgQueueTime() / 2 + processing_times[i, 0])
    return latencies / weights


@numba.njit(cache=True)
def _get_mdc_latency(x, input_rates, processing_times, percentile, max_rho=0.95):
    if np.any(x <= 0):
        # COBYLA suggested the case where x <= 0.
        # We cannot calculate latency since the M/D/C is not well defined.
        # return np.inf
        return np.ones_like(input_rates) * np.inf
    if input_rates.ndim == 1:
        input_rates = input_rates.reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[-1] != input_rates.shape[-1]:
        # x = np.repeat(x, input_rates.shape[-1], axis=1)
        # not numba compatible
        new_x = np.empty((x.shape[0], input_rates.shape[-1]), dtype=np.float64)
        for i in range(input_rates.shape[-1]):
            new_x[:, i] = x[:, 0]
        x = new_x
    latencies = _get_mdc_latency_2d(
        x, input_rates.copy(), processing_times.reshape(-1, 1), percentile, max_rho)
    return latencies


@numba.njit(cache=True)
def _estimate_latency(x: np.ndarray, input_rates, processing_times, mdc_percentile, max_rho):
    if mdc_percentile > 0:
        # processing time unit is ms. divide them by 1e3
        expected_latencies = _get_mdc_latency(
            x, input_rates, processing_times/1e3, mdc_percentile, max_rho)
        # convert it to ms
        expected_latencies *= 1e3
    else:
        expected_latencies = input_rates / x * processing_times
    return expected_latencies


@numba.njit(cache=True)
def _reshape_x(x: np.ndarray, old_x: np.ndarray, upscale_overhead: int, dim: int):
    # current available input (dim) is too small to apply upscale_overhead
    if upscale_overhead == 0 or upscale_overhead >= dim:
        return x.reshape(-1, 1)

    # x = x.reshape(-1, 1)
    # old_x = old_x.reshape(-1, 1)
    # mask = np.repeat(x > old_x, upscale_overhead, axis=1)
    # x = np.repeat(x, dim, axis=1)
    # x[:, :upscale_overhead][mask] = np.repeat(
    #     old_x, upscale_overhead, axis=1)[mask]
    # return x

    mask = x > old_x
    x_vec = np.empty((x.shape[0], dim), dtype=np.float64)
    for i in range(dim):
        x_vec[:, i] = x
        if i < upscale_overhead:
            x_vec[:, i][mask] = old_x[mask]
    return x_vec


@numba.njit(cache=True)
def _calculate_utility(x,
                       old_x,
                       upscale_overhead,
                       input_rates,
                       processing_times,
                       mdc_percentile,
                       max_rho,
                       util_type,
                       slo_targets):
    if input_rates.ndim == 2:
        x = _reshape_x(x, old_x, upscale_overhead, input_rates.shape[-1])
    expected_latencies = _estimate_latency(
        x, input_rates, processing_times, mdc_percentile, max_rho)

    retval = np.empty(len(expected_latencies), dtype=np.float64)
    for i in range(len(expected_latencies)):
        latency = expected_latencies[i]
        latency = np.clip(latency, a_min=1e-4, a_max=None)
        if util_type == "latency":
            retval[i] = np.mean(
                np.clip(slo_targets[i] / latency, a_min=None, a_max=1.0))
        elif util_type == "latency_step":
            retval[i] = np.mean((latency <= slo_targets[i]).astype(np.float32) * 1.0)
        else:
            # not implemented
            retval[i] = 0.0
    return retval


@numba.njit(cache=True)
def _calculate_penalized_utility(x,
                                 old_replicas,
                                 upscale_overhead,
                                 input_rates,
                                 processing_times,
                                 mdc_percentile,
                                 max_rho,
                                 util_type,
                                 slo_targets,
                                 linear_penalty,
                                 drop_integrality,
                                 coeffs,
                                 threshold_penalty_tuples):
    replica_x = x[:len(old_replicas)]
    accept_rate_x = x[len(old_replicas):]
    if drop_integrality:
        accept_rate_x = accept_rate_x / 100
    utility = _calculate_utility(
        replica_x, old_replicas, upscale_overhead, input_rates,
        processing_times, mdc_percentile, max_rho, util_type, slo_targets)
    for i in range(len(utility)):
        violation_rate = max(0.0, min(1 - accept_rate_x[i], 1.0))
        penalty = 1
        for j in range(len(threshold_penalty_tuples)):
            threshold, v = threshold_penalty_tuples[j]
            if violation_rate <= threshold:
                if linear_penalty:
                    coeff, intercept = coeffs[j]
                    penalty = coeff * violation_rate + intercept
                else:
                    penalty = v
                break
        utility[i] *= (1 - penalty)
    return utility


class Solver:

    def __init__(self, resource_limit, adjust: bool, min_max: bool,
                 method: Union[str, List[str]] = "slsqp",
                 changes_weight: float = 0.0,
                 drop_integrality: bool = False,
                 drop_weight: float = 0.0,
                 linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None,
                 upscale_overhead: int = 0,
                 max_rho: float = 0.95,
                 num_children: Optional[int] = None,
                 random_split: bool = False,
                 util_type: str = "latency"):
        self.keys = []
        self.processing_times = []
        self.input_rates = []
        self.weights = []
        self.util_type = util_type
        self.slo_targets = []
        self.num_replicas_list = []
        self.resource_per_replicas = []
        self.resource_limit = resource_limit
        self.adjust = adjust
        self.min_max = min_max
        self.changes_weight = changes_weight
        if isinstance(method, str):
            self.methods = [method]
        else:
            self.methods = method
        self.num_replicas = None
        assert not drop_integrality
        assert drop_weight == 0.0
        assert not linear_penalty
        assert num_children is None
        assert not random_split
        self.mdc_percentile = mdc_percentile
        self.utility_weight = 1.0 if utility_weight is None else utility_weight
        self.upscale_overhead = upscale_overhead
        self.max_rho = max_rho

        _LOGGER.info("Solver configs. %s", self.config_str())

    def config_str(self):
        return ", ".join([
            f"adjust={self.adjust}",
            f"min_max={self.min_max}",
            f"mdc_percentile={self.mdc_percentile}",
            f"utility_weight={self.utility_weight}",
            f"upscale_overhead={self.upscale_overhead}",
            f"max_rho={self.max_rho}",
            f"resource_limit={self.resource_limit}",
            f"util_type={self.util_type}",
        ])

    def add_deployment(self,
                       key,
                       processing_time,
                       input_rate: Union[int, Iterable[int]],
                       weight,
                       slo_target,
                       current_num_replicas,
                       resource_per_replica):
        self.keys.append(key)
        self.processing_times.append(processing_time)
        self.input_rates.append(input_rate)
        self.weights.append(weight)
        self.slo_targets.append(slo_target)
        self.num_replicas_list.append(current_num_replicas)
        self.resource_per_replicas.append(resource_per_replica)

    def availalbe_resources(self):
        return self.resource_limit - np.dot(self.resource_per_replicas, self.num_replicas)

    def get_upscale_overhead(self):
        if self.availalbe_resources() > 0:
            # TODO: fix this later
            return int(self.upscale_overhead // 2)
        else:
            return self.upscale_overhead

    def calculate_utility(self, x: np.array):
        return _calculate_utility(
            x, self.num_replicas, self.get_upscale_overhead(), self.input_rates,
            self.processing_times, self.mdc_percentile, self.max_rho,
            self.util_type, self.slo_targets)

    def apply_weights(self, utility):
        return self.utility_weight * self.weights * utility

    def get_changes(self, x):
        return self.changes_weight * np.count_nonzero(self.num_replicas != np.ceil(x))

    def objective(self, x):
        return -np.sum(self.apply_weights(self.calculate_utility(x))) + self.get_changes(x)

    def objective_minmax(self, x):
        utilities = self.calculate_utility(x)
        min_max = np.min(utilities) - np.max(utilities)
        return -np.sum(self.apply_weights(utilities)) - min_max + self.get_changes(x)

    def objective_minmax2(self, x):
        utilities = self.calculate_utility(x)
        # change range from [-1, 0] to [0, 1]
        min_max = np.min(utilities) - np.max(utilities) + 1
        return -np.sum(self.apply_weights(utilities)) - min_max + self.get_changes(x)

    def objective_minmax3(self, x):
        utilities = self.calculate_utility(x)
        # change range from [-1, 0] to [0, 1] and scale
        min_max = (np.min(utilities) - np.max(utilities) + 1) * len(utilities)
        return -np.sum(self.apply_weights(utilities)) - min_max + self.get_changes(x)

    def get_used_resources(self, x):
        return np.dot(self.resource_per_replicas, x)

    def get_min_replicas(self, i):
        return 1

    def make_feasible(self, x: np.array, objective):
        x = np.ceil(x)  # .astype(int)
        used_resource = self.get_used_resources(x)
        while used_resource > self.resource_limit:
            _LOGGER.info("Used resource (%s) > limit (%s). replicas=%s",
                         used_resource, self.resource_limit, x)
            # remove the replica that has lowest objective value decrease
            candidates = []
            for i in range(len(x)):
                new_x = x.copy()
                if new_x[i] <= self.get_min_replicas(i):
                    continue
                new_x[i] -= 1
                new_objective = np.round(objective(new_x), _PRECISION)
                change = abs(new_x[i] - self.num_replicas[i])
                candidates.append(((new_objective, change), new_x))

            # update with the minimum objective and
            # mimium change from the current setup (tie-breaker)
            _, x = sorted(candidates, key=lambda x: x[0])[0]
            used_resource = self.get_used_resources(x)

        return x

    def adjust_solution(self, x, objective):
        objective_val = objective(x)
        new_x = np.copy(x)
        for idx in range(len(x)):
            while new_x[idx] > self.get_min_replicas(idx):
                new_x[idx] -= 1
                new_objective_val = objective(new_x)
                if new_objective_val - objective_val > _EPSILON:
                    new_x[idx] += 1
                    break
                objective_val = new_objective_val
        if not np.all(x == new_x):
            _LOGGER.info("Change num replicas from %s to %s", x, new_x)
            x = self.try_allocate_remaining_resources(new_x, objective)
        return x

    def try_allocate_remaining_resources(self, x, objective):
        """Checks by allocating all remaining resources to each job."""
        avail_resources = self.resource_limit - self.get_used_resources(x)
        assert avail_resources > 0

        objective_val = objective(x)
        new_x = np.copy(x)
        for idx in range(len(x)):
            allocatable_num_replicas = int(
                avail_resources / self.resource_per_replicas[idx])
            new_x[idx] += allocatable_num_replicas
            new_objective_val = objective(new_x)
            if objective_val - new_objective_val > _EPSILON:
                # better solution found
                _LOGGER.info("Better solution found: %s (%.2f) from (%.2f)",
                             new_x, new_objective_val, objective_val)
                if self.adjust:
                    new_x = self.adjust_solution(new_x, objective)
                break
            else:
                new_x[idx] -= allocatable_num_replicas

        return new_x

    def ineq_constraints(self, x):
        constraints = [
            # sum resource(x_i) <= resource limit
            self.resource_limit - self.get_used_resources(x)
        ]
        # x_i >= 1
        constraints += list(x - 1)
        return np.array(constraints)

    def _prepare(self):
        # handle the case where input rate lengths are not the same
        self.input_rates = pd.DataFrame(self.input_rates).fillna(0).values
        self.keys = np.array(self.keys)
        self.resource_per_replicas = np.array(self.resource_per_replicas)
        self.weights = np.array(self.weights)
        self.processing_times = np.array(self.processing_times)
        self.slo_targets = np.array(self.slo_targets)
        if self.input_rates.ndim == 2:
            self.processing_times = self.processing_times.reshape(-1, 1)
        if len(self.num_replicas_list):
            self.num_replicas = np.array(
                self.num_replicas_list, dtype=np.float64)
            self.num_replicas_list = []

    def _solve(self):
        dict_constraints = [{'type': 'ineq', 'fun': self.ineq_constraints}]
        bounds = scipy.optimize.Bounds(
            lb=np.ones(len(self.keys)),
            ub=np.ones(len(self.keys)) * self.resource_limit,
            keep_feasible=True)
        constraints = [
            scipy.optimize.LinearConstraint(
                self.resource_per_replicas, lb=0, ub=self.resource_limit),
        ]

        if self.min_max:
            if self.min_max == 2:
                objective = self.objective_minmax2
            elif self.min_max == 3:
                objective = self.objective_minmax3
            else:
                objective = self.objective_minmax
        else:
            objective = self.objective

        x = self.num_replicas
        for method in self.methods:
            t = time.time()
            if method == "de":
                args = (objective,)
                sol = scipy.optimize.differential_evolution(
                    _vectorized_objective, bounds, args, workers=1,
                    vectorized=True, mutation=(0.5, 1.0), maxiter=1000, x0=x,
                    integrality=np.ones(len(self.keys), dtype=bool),
                    disp=False, constraints=constraints, updating='deferred',
                )
            else:
                sol = scipy.optimize.minimize(
                    objective, x, method=method,
                    constraints=dict_constraints,
                    options=_OPT_OPTIONS.get(method, None))
            _LOGGER.info(
                "Optimizer (%s) solution (%s): %s val=%.2f, elapsed_time=%.2f",
                method, sol.success, sol.x, sol.fun, time.time() - t)

            # adjust next replicas
            x = self.make_feasible(sol.x, objective)

            if self.adjust:
                x = self.adjust_solution(x, objective)

            _LOGGER.info("objective: %.2f", objective(x))

        next_replicas = {}
        for key, n in zip(self.keys, x):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            next_replicas[key] = int(n)
        return next_replicas, self.calculate_utility(x)

    def solve(self):
        self._prepare()
        return self._solve()


class SolverWithDrop(Solver):

    def __init__(self, resource_limit, adjust: bool, min_max: bool,
                 method: Union[str, List[str]] = "slsqp",
                 changes_weight: float = 0.0, drop_integrality: bool = False,
                 drop_weight: float = 0.0, linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None,
                 upscale_overhead: int = 0,
                 max_rho: float = 0.95,
                 num_children: Optional[int] = None,
                 random_split: bool = False,
                 util_type: str = "latency"):
        super().__init__(
            resource_limit, adjust, min_max, method, changes_weight,
            mdc_percentile=mdc_percentile, utility_weight=utility_weight,
            upscale_overhead=upscale_overhead, max_rho=max_rho,
            num_children=num_children, util_type=util_type)
        self.linear_penalty = linear_penalty
        if self.linear_penalty:
            penalty_func = SLOStrictLinearPenalty()
            self.coeffs = np.array(penalty_func.coeffs, dtype=np.float64)
        else:
            penalty_func = SLOStrictPenalty()
            self.coeffs = np.zeros_like(
                penalty_func.threshold_penalty_tuples, dtype=np.float64)
        self.threshold_penalty_tuples = np.array(
            penalty_func.threshold_penalty_tuples)

        self.accept_rates_list = []
        self.drop_integrality = drop_integrality
        self.drop_weight = drop_weight
        assert not random_split

    def add_deployment(self,
                       key,
                       processing_time,
                       input_rate,
                       weight,
                       slo_target,
                       current_num_replicas,
                       resource_per_replica,
                       accept_rate: float = 1.0):
        super().add_deployment(
            key, processing_time, input_rate, weight, slo_target,
            current_num_replicas, resource_per_replica)
        self.accept_rates_list.append(accept_rate)

    def get_accept_rates(self, x):
        accept_rate_x = x[len(self.keys):]
        if self.drop_integrality:
            accept_rate_x = accept_rate_x / 100
        return accept_rate_x

    def get_accept_rate_term(self, x):
        if self.drop_weight == 0.0:
            return 0.0
        return self.drop_weight * np.sum(self.get_accept_rates(x))

    def objective(self, x):
        # maximize the accept rate
        return super().objective(x) - self.get_accept_rate_term(x)

    def objective_minmax(self, x):
        # maximize the accept rate
        return super().objective_minmax(x) - self.get_accept_rate_term(x)

    def objective_minmax2(self, x):
        return super().objective_minmax2(x) - self.get_accept_rate_term(x)

    def objective_minmax3(self, x):
        return super().objective_minmax3(x) - self.get_accept_rate_term(x)

    def calculate_utility(self, x: np.array):
        return _calculate_penalized_utility(
            x, self.num_replicas, self.get_upscale_overhead(), self.input_rates,
            self.processing_times, self.mdc_percentile, self.max_rho,
            self.util_type, self.slo_targets, self.linear_penalty,
            self.drop_integrality, self.coeffs, self.threshold_penalty_tuples)
        # num_jobs = len(self.keys)
        # replica_x = x[:num_jobs]
        # accept_rate_x = self.get_accept_rates(x)
        # utility = super().calculate_utility(replica_x)
        # return [util * (1 - penalty_func(1 - accept_rate))
        #         for util, penalty_func, accept_rate
        #         in zip(utility, self.penalty_funcs, accept_rate_x)]

    def make_feasible(self, x: np.array, objective):
        """Makes the solution feasible.

        This does NOT change the accept rates.
        """
        num_jobs = len(self.keys)
        accept_rate_x = x[num_jobs:]

        def objective_wrapper(replica_x):
            return objective(np.hstack([replica_x, accept_rate_x]))

        new_replica_x = super().make_feasible(x[:num_jobs], objective_wrapper)
        return np.hstack([new_replica_x, accept_rate_x])

    def adjust_solution(self, x, objective):
        """Minimizes the number of replicas without decreasing the objective value.

        This does NOT change the accept rates.
        """
        num_jobs = len(self.keys)
        accept_rate_x = x[num_jobs:]

        def objective_wrapper(replica_x):
            return objective(np.hstack([replica_x, accept_rate_x]))

        new_replica_x = super().adjust_solution(
            x[:num_jobs], objective_wrapper)
        return np.hstack([new_replica_x, accept_rate_x])

    def try_allocate_remaining_resources(self, x, objective):
        """Checks by allocating all remaining resources to each job.

        This does NOT change the accept rates.
        """
        num_jobs = len(self.keys)
        accept_rate_x = x[num_jobs:]

        def objective_wrapper(replica_x):
            return objective(np.hstack([replica_x, accept_rate_x]))

        new_replica_x = super().try_allocate_remaining_resources(
            x[:num_jobs], objective_wrapper)
        return np.hstack([new_replica_x, accept_rate_x])

    def ineq_constraints(self, x):
        num_jobs = len(self.keys)
        replica_x = x[:num_jobs]
        accept_rate_x = x[num_jobs:]
        constraints = [
            # sum resource(x_i) <= resource limit
            self.resource_limit - np.dot(self.resource_per_replicas, replica_x)
        ]

        # Bounds as constraints
        # x_i >= 1
        constraints += list(replica_x - 1)
        if self.drop_integrality:
            # 0 <= d_i * 100 <= 100
            constraints += list(accept_rate_x)
            constraints += list(100 - accept_rate_x)
        else:
            # 0 <= d_i <= 1
            constraints += list(accept_rate_x)
            constraints += list(1 - accept_rate_x)

        return np.array(constraints)

    def get_changes(self, x):
        return self.changes_weight * np.count_nonzero(self.num_replicas != np.ceil(x[:len(self.keys)]))

    def _solve(self):
        dict_constraints = [
            {
                'type': 'ineq',
                'fun': self.ineq_constraints,
            }
        ]
        bounds = scipy.optimize.Bounds(
            lb=np.hstack([np.ones(len(self.keys)), np.zeros(len(self.keys))]),
            ub=np.hstack([
                np.ones(len(self.keys)) * self.resource_limit,
                np.ones(len(self.keys)) * (100 if self.drop_integrality else 1)]),
            keep_feasible=True)

        def const(x):
            return np.expand_dims(np.dot(self.resource_per_replicas, x[:len(self.keys)]), axis=0)
        constraints = [
            scipy.optimize.NonlinearConstraint(
                const,
                lb=0, ub=self.resource_limit),
        ]

        accept_rates = np.array(self.accept_rates_list)
        if self.drop_integrality:
            accept_rates *= 100

        if self.min_max:
            if self.min_max == 2:
                objective = self.objective_minmax2
            elif self.min_max == 3:
                objective = self.objective_minmax3
            else:
                objective = self.objective_minmax
        else:
            objective = self.objective

        x = np.hstack([self.num_replicas, accept_rates])
        for method in self.methods:
            t = time.time()
            if method == "de":
                args = (objective,)
                if self.drop_integrality:
                    integrality = np.ones(len(self.keys)*2, dtype=bool)
                else:
                    integrality = np.hstack([
                        np.ones(len(self.keys), dtype=bool),  # num replicas
                        np.zeros(len(self.keys), dtype=bool),  # drop rates
                    ])
                sol = scipy.optimize.differential_evolution(
                    _vectorized_objective, bounds, args, workers=1,
                    vectorized=True, mutation=(0.5, 1.0), maxiter=1000, x0=x,
                    integrality=integrality,
                    disp=False, constraints=constraints, updating='deferred',
                )
            else:
                sol = scipy.optimize.minimize(
                    objective, x, method=method,
                    constraints=dict_constraints,
                    options=_OPT_OPTIONS.get(method, None))
            _LOGGER.info(
                "Optimizer (%s) solution (%s): %s val=%.2f, elapsed_time=%.2f",
                method, sol.success, sol.x, sol.fun, time.time() - t)

            # adjust next replicas
            x = self.make_feasible(sol.x, objective)

            if self.adjust:
                x = self.adjust_solution(x, objective)

            _LOGGER.info("objective: %.2f", objective(x))

        next_sol = {}
        for i, key in enumerate(self.keys):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            next_sol[key] = (
                np.round(x[i]).astype(int),
                x[len(self.keys) + i] * (0.01 if self.drop_integrality else 1))
        return next_sol, self.calculate_utility(x)

    def solve(self):
        self._prepare()
        return self._solve()


def _merge(v: np.ndarray, num_children, aggr=np.sum):
    split = np.array_split(v, num_children, axis=0)
    return np.concatenate([[aggr(v, axis=0)] for v in split], axis=0), split


def _split(sol: Union['HierarchicalSolver', 'HierarchicalSolverWithDrop'],
           random_split: bool):
    if len(sol.keys) <= sol.num_children:
        # no need to split. clear all split variables
        sol.index = None
        sol.keys_split = None
        sol.input_rates_split = None
        sol.weights_split = None
        sol.processing_times_split = None
        sol.num_replicas_split = None
        sol.resource_per_replicas_split = None
        sol.slo_targets_split = None
        return None
    else:
        # aggregate the original inputs
        # TODO: think of better split strategy since it can affect the quality
        if random_split:
            idxes = np.random.permutation(len(sol.keys))
        else:
            idxes = np.arange(len(sol.keys))

        sol.keys_split = np.array_split(sol.keys[idxes], sol.num_children)
        sol.keys = [f"{sol.prefix}{i}" for i in range(
            len(sol.keys_split))]
        sol.index = {key: i for i, key in enumerate(sol.keys)}
        sol.input_rates, sol.input_rates_split = _merge(
            sol.input_rates[idxes], sol.num_children)
        sol.weights, sol.weights_split = _merge(
            sol.weights[idxes], sol.num_children, aggr=np.mean)
        sol.processing_times, sol.processing_times_split = _merge(
            sol.processing_times[idxes], sol.num_children, aggr=np.mean)
        sol.num_replicas, sol.num_replicas_split = _merge(
            sol.num_replicas[idxes], sol.num_children)
        sol.resource_per_replicas, sol.resource_per_replicas_split = _merge(
            sol.resource_per_replicas[idxes], sol.num_children, aggr=np.mean)
        sol.slo_targets, sol.slo_targets_split = _merge(
            sol.slo_targets[idxes], sol.num_children, aggr=np.mean)
        return idxes


def _split_resources(child_resources: np.ndarray, total_resources: int):
    child_resource_limits = child_resources / \
        child_resources.sum() * total_resources
    child_resource_limits, frac = np.divmod(child_resource_limits, 1)
    child_resource_limits = child_resource_limits.astype(int)
    remainder = total_resources - child_resource_limits.sum()
    idxes = np.argsort(-frac)[:remainder]
    child_resource_limits[idxes] += 1
    if child_resource_limits.sum() != total_resources:
        raise ValueError(
            "Remained resources are not allocated correctly")
    return child_resource_limits


class HierarchicalSolver(Solver):

    def __init__(self, resource_limit, adjust: bool, min_max: bool,
                 method: Union[str, List[str]] = "slsqp",
                 changes_weight: float = 0.0,
                 drop_integrality: bool = False,
                 drop_weight: float = 0.0,
                 linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None,
                 upscale_overhead: int = 0,
                 max_rho: float = 0.95,
                 num_children: int = 10,
                 random_split: bool = False,
                 util_type: str = "latency",
                 prefix: str = "group"):
        self.prefix = prefix
        self.num_children = num_children
        self.keys_split = None
        self.input_rates_split = None
        self.weights_split = None
        self.processing_times_split = None
        self.num_replicas_split = None
        self.resource_per_replicas_split = None
        self.slo_targets_split = None
        self.random_split = random_split
        super().__init__(
            resource_limit, adjust, min_max, method, changes_weight,
            drop_integrality, drop_weight, linear_penalty,
            mdc_percentile=mdc_percentile, utility_weight=utility_weight,
            upscale_overhead=upscale_overhead, max_rho=max_rho,
            util_type=util_type)

    def config_str(self):
        return ", ".join([
            super().config_str(),
            f"num_children={self.num_children}",
            f"prefix={self.prefix}",
        ])

    def create_child_solver(self, resource_limit, keys, input_rates, weights, processing_times, num_replicas, resource_per_replicas, slo_targets, prefix):
        child = HierarchicalSolver(
            resource_limit, self.adjust, self.min_max, self.methods,
            self.changes_weight, mdc_percentile=self.mdc_percentile,
            utility_weight=self.utility_weight,
            upscale_overhead=self.upscale_overhead, max_rho=self.max_rho,
            num_children=self.num_children, random_split=self.random_split,
            util_type=self.util_type, prefix=f"{prefix}_group")

        for i in range(len(keys)):
            child.add_deployment(
                keys[i], processing_times[i], input_rates[i], weights[i],
                slo_targets[i], num_replicas[i], resource_per_replicas[i])

        return child

    def ineq_constraints(self, x):
        if self.index is None:
            return super().ineq_constraints(x)

        constraints = [
            # sum resource(x_i) <= resource limit
            self.resource_limit - self.get_used_resources(x)
        ]
        # x_i >= #jobs for each group
        for i in range(len(x)):
            constraints.append(x[i] - len(self.keys_split[i]))
        return np.array(constraints)

    def get_min_replicas(self, i):
        if self.index is None:
            return super().get_min_replicas(i)
        # each job should have at least 1 replica
        return len(self.keys_split[i])

    def _prepare(self):
        super()._prepare()
        _split(self, self.random_split)

    def solve(self):
        # solve for groups
        self._prepare()
        group_next_replicas, group_next_utility = super()._solve()
        _LOGGER.info("Group solutions: %s, utility=%s",
                     group_next_replicas, group_next_utility)

        next_replicas = {}
        next_utility = []

        if self.index is None:
            # no split happened
            next_replicas = group_next_replicas
            next_utility = group_next_utility
        else:
            # split happened
            remained_group = []
            remained_resources = self.resource_limit
            child_resources = []

            for key, num_replicas in group_next_replicas.items():
                i = self.index[key]
                child_keys = self.keys_split[i]
                if len(child_keys) == 1:
                    # no need to run for a single job group
                    next_replicas[child_keys[0]] = num_replicas
                    next_utility.append(group_next_utility[i])
                    remained_resources -= int(
                        num_replicas * self.resource_per_replicas[i])
                    continue
                else:
                    remained_group.append((key, num_replicas))
                    child_resources.append(int(
                        num_replicas * self.resource_per_replicas[i]))

            # split the remained resources proportional to each group's used resources
            child_resource_limits = _split_resources(
                np.array(child_resources), remained_resources)

            for (key, num_replicas), child_resource_limit in zip(remained_group, child_resource_limits):
                i = self.index[key]
                child = self.create_child_solver(
                    resource_limit=child_resource_limit,
                    keys=self.keys_split[i],
                    input_rates=self.input_rates_split[i],
                    weights=self.weights_split[i],
                    processing_times=self.processing_times_split[i],
                    num_replicas=self.num_replicas_split[i],
                    resource_per_replicas=self.resource_per_replicas_split[i],
                    slo_targets=self.slo_targets_split[i],
                    prefix=key)
                child_next_replicas, child_next_utility = child.solve()
                next_replicas.update(child_next_replicas)
                next_utility.extend(child_next_utility)

        return next_replicas, next_utility


class HierarchicalSolverWithDrop(SolverWithDrop):

    def __init__(self, resource_limit, adjust: bool, min_max: bool,
                 method: Union[str, List[str]] = "slsqp",
                 changes_weight: float = 0.0, drop_integrality: bool = False,
                 drop_weight: float = 0.0, linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None,
                 upscale_overhead: int = 0,
                 max_rho: float = 0.95,
                 num_children: int = 10,
                 random_split: bool = False,
                 util_type: str = "latency",
                 prefix: str = "group"):
        self.index = None
        self.prefix = prefix
        self.num_children = num_children
        self.random_split = random_split
        self.keys_split = None
        self.input_rates_split = None
        self.weights_split = None
        self.processing_times_split = None
        self.num_replicas_split = None
        self.resource_per_replicas_split = None
        self.slo_targets_split = None
        self.accept_rates_list_split = None
        self.random_split: bool = False
        super().__init__(
            resource_limit, adjust, min_max, method, changes_weight,
            mdc_percentile=mdc_percentile, utility_weight=utility_weight,
            upscale_overhead=upscale_overhead, max_rho=max_rho,
            drop_integrality=drop_integrality, drop_weight=drop_weight,
            linear_penalty=linear_penalty, util_type=util_type)

    def config_str(self):
        return ", ".join([
            super().config_str(),
            f"num_children={self.num_children}",
            f"prefix={self.prefix}",
        ])

    def create_child_solver(self, resource_limit, keys, input_rates, weights,
                            processing_times, num_replicas,
                            resource_per_replicas, slo_targets,
                            accept_rates, prefix):
        child = HierarchicalSolverWithDrop(
            resource_limit, self.adjust, self.min_max, self.methods,
            self.changes_weight, self.drop_integrality, self.drop_weight,
            self.linear_penalty, self.mdc_percentile, self.utility_weight,
            self.upscale_overhead, self.max_rho, self.num_children,
            self.random_split, self.util_type, f"{prefix}_group")

        for i in range(len(keys)):
            child.add_deployment(
                keys[i], processing_times[i], input_rates[i], weights[i],
                slo_targets[i], num_replicas[i], resource_per_replicas[i],
                accept_rates[i])

        return child

    def ineq_constraints(self, x):
        if self.index is None:
            return super().ineq_constraints(x)

        num_jobs = len(self.keys)
        replica_x = x[:num_jobs]
        accept_rate_x = x[num_jobs:]
        constraints = [
            # sum resource(x_i) <= resource limit
            self.resource_limit - np.dot(self.resource_per_replicas, replica_x)
        ]

        # Bounds as constraints
        # x_i >= #jobs for each group
        for i in range(num_jobs):
            constraints.append(replica_x[i] - len(self.keys_split[i]))

        if self.drop_integrality:
            # 0 <= d_i * 100 <= 100
            constraints += list(accept_rate_x)
            constraints += list(100 - accept_rate_x)
        else:
            # 0 <= d_i <= 1
            constraints += list(accept_rate_x)
            constraints += list(1 - accept_rate_x)

        return np.array(constraints)

    def get_min_replicas(self, i):
        if self.index is None:
            return super().get_min_replicas(i)
        # each job should have at least 1 replica
        return len(self.keys_split[i])

    def _prepare(self):
        super()._prepare()
        self.accept_rates_list = np.array(self.accept_rates_list)
        idxes = _split(self, self.random_split)
        # split others
        if self.index is None:
            self.accept_rates_list_split = None
        else:
            self.accept_rates_list, self.accept_rates_list_split = _merge(
                self.accept_rates_list[idxes], self.num_children, aggr=np.mean)

    def solve(self):
        # solve for groups
        self._prepare()
        group_next_sol, group_next_utility = super()._solve()
        _LOGGER.info("Group solutions: %s, utility=%s",
                     group_next_sol, group_next_utility)

        next_sol = {}
        next_utility = []

        if self.index is None:
            # no split happened
            next_sol = group_next_sol
            next_utility = group_next_utility
        else:
            # split happened
            remained_group = []
            remained_resources = self.resource_limit
            child_resources = []

            for key, (num_replicas, accept_rate) in group_next_sol.items():
                i = self.index[key]
                child_keys = self.keys_split[i]
                if len(child_keys) == 1:
                    # no need to run for a single job group
                    next_sol[child_keys[0]] = (num_replicas, accept_rate)
                    next_utility.append(group_next_utility[i])
                    remained_resources -= int(
                        num_replicas * self.resource_per_replicas[i])
                    continue
                else:
                    remained_group.append((key, num_replicas))
                    child_resources.append(int(
                        num_replicas * self.resource_per_replicas[i]))

            # split the remained resources proportional to each group's used resources
            child_resource_limits = _split_resources(
                np.array(child_resources), remained_resources)

            for (key, num_replicas), child_resource_limit in zip(remained_group, child_resource_limits):
                i = self.index[key]
                child = self.create_child_solver(
                    resource_limit=child_resource_limit,
                    keys=self.keys_split[i],
                    input_rates=self.input_rates_split[i],
                    weights=self.weights_split[i],
                    processing_times=self.processing_times_split[i],
                    num_replicas=self.num_replicas_split[i],
                    resource_per_replicas=self.resource_per_replicas_split[i],
                    slo_targets=self.slo_targets_split[i],
                    accept_rates=self.accept_rates_list_split[i],
                    prefix=key)
                child_next_sol, child_next_utility = child.solve()
                next_sol.update(child_next_sol)
                next_utility.extend(child_next_utility)

        return next_sol, next_utility


def create_solver(with_drop, num_children=None, **kwargs):
    if with_drop:
        kwargs["drop_weight"] = 0.0
        kwargs["drop_integrality"] = True
        kwargs["linear_penalty"] = True
        if num_children is not None and num_children > 1:
            solver_cls = HierarchicalSolverWithDrop
        else:
            solver_cls = SolverWithDrop
            num_children = None
    else:
        if num_children is not None and num_children > 1:
            solver_cls = HierarchicalSolver
        else:
            solver_cls = Solver
            num_children = None

    _LOGGER.info("Solver: %s", solver_cls.__name__)

    return solver_cls(num_children=num_children, **kwargs)


def run_solver(solver, input_rates, num_replicas, processing_time, target_latency):
    for i, (input_rate, num_replica) in enumerate(zip(input_rates, num_replicas)):
        solver.add_deployment(
            f"cluster{i}",
            processing_time,
            input_rate,
            1,
            target_latency,
            num_replica,
            1,
        )

    t = time.time()
    sol, value = solver.solve()
    elapsed_time = time.time() - t
    v = []
    for i in range(len(num_replicas)):
        v.append(sol[f"cluster{i}"])
    return v, np.sum(value), elapsed_time


if __name__ == "__main__":
    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(logging.INFO), format=format)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--method", type=str, default="cobyla")
    parser.add_argument("--num-children", type=int)
    parser.add_argument("--with-drop", action="store_true")
    parser.add_argument("--processing-time", type=int, default=180)
    parser.add_argument("--target-latency", type=int, default=720)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--resource-limit", type=int, default=32)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--util-type", type=str, default="latency")
    parser.add_argument("--minmax", type=int)
    parser.add_argument("--mdc-percentile", type=int, default=99)
    parser.add_argument("--upscale-overhead", type=int, default=1)
    parser.add_argument("--random-split", action="store_true",
                        help="Random split for hierarchical solver")

    args = parser.parse_args()
    _LOGGER.info("args: %s", args)

    samples = pd.read_pickle(args.path)
    _LOGGER.info("Loaded num_samples: %s", len(samples))

    solver_kwargs = {
        "resource_limit": args.resource_limit,
        "adjust": True,
        "min_max": args.minmax,
        "method": args.method,
        "mdc_percentile": args.mdc_percentile,
        "util_type": args.util_type,
        "upscale_overhead": args.upscale_overhead,
        "num_children": args.num_children,
        "random_split": args.random_split,
    }

    for i,  input_rates in enumerate(samples):
        input_rates = np.concatenate(
            [input_rates for _ in range(args.factor)], axis=0) / 60
        _LOGGER.info("sample %d shape=%s", i, input_rates.shape)
        num_replicas = np.ones(
            input_rates.shape[0], dtype=int) * args.num_replicas

        v, value, elapsed_time = run_solver(
            solver=create_solver(with_drop=args.with_drop, **solver_kwargs),
            input_rates=input_rates,
            num_replicas=num_replicas,
            processing_time=args.processing_time,
            target_latency=args.target_latency,
        )

        _LOGGER.info("elapsed time: %.2f", elapsed_time)
        _LOGGER.info("replicas (%d): %s", len(v), v)
        _LOGGER.info("utility_sum: %.2f", value)
