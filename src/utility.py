from typing import Any, Callable, Union

import numpy as np


class SLOPenalty:

    def __init__(self, threshold_penalty_tuples=None):
        if threshold_penalty_tuples is None:
            threshold_penalty_tuples = [(0.01, 0), (0.05, 0.25), (0.1, 0.5)]
        self.threshold_penalty_tuples = threshold_penalty_tuples
        # check whether thresholds/penalties are sorted in acending order
        thresholds, penalties = zip(*threshold_penalty_tuples)
        thresholds, penalties = list(thresholds), list(penalties)
        assert thresholds == sorted(thresholds)
        assert penalties == sorted(penalties)

    def __call__(self, violation_rate):
        for threshold, penalty in self.threshold_penalty_tuples:
            if violation_rate <= threshold:
                return penalty
        return 1


class SLOStrictPenalty(SLOPenalty):

    def __init__(self):
        # Setting 0 for the 0 penalty threshold prevents the optimizer
        # from achieving good results.
        super().__init__(threshold_penalty_tuples=[
            (0.001, 0), (0.03, 0.25), (0.08, 0.5)])


class SLOLinearPenalty:

    def __init__(self, threshold_penalty_tuples=None):
        if threshold_penalty_tuples is None:
            threshold_penalty_tuples = [(0.01, 0), (0.05, 0.25), (0.1, 0.5)]
        self.threshold_penalty_tuples = threshold_penalty_tuples
        if self.threshold_penalty_tuples[-1] != (1.0, 1.0):
            self.threshold_penalty_tuples.append((1.0, 1.0))
        # check whether thresholds/penalties are sorted in acending order
        thresholds, penalties = zip(*threshold_penalty_tuples)
        thresholds, penalties = list(thresholds), list(penalties)
        assert thresholds == sorted(thresholds)
        assert penalties == sorted(penalties)

        self.coeffs = []
        prev_threshold = 0
        prev_penalty = 0
        for threshold, penalty in self.threshold_penalty_tuples:
            coeff = (penalty - prev_penalty) / (threshold - prev_threshold)
            intercept = (
                (prev_penalty * threshold - penalty * prev_threshold) /
                (threshold - prev_threshold))
            self.coeffs.append((coeff, intercept))
            prev_penalty = penalty
            prev_threshold = threshold

    def __call__(self, violation_rate):
        violation_rate = float(np.clip(violation_rate, a_min=0, a_max=1))
        for i, (threshold, _) in enumerate(self.threshold_penalty_tuples):
            if violation_rate <= threshold:
                coeff, intercept = self.coeffs[i]
                return coeff * violation_rate + intercept
        return 1


class SLOStrictLinearPenalty(SLOLinearPenalty):

    def __init__(self):
        # Setting 0 for the 0 penalty threshold prevents the optimizer
        # from achieving good results.
        super().__init__(
            threshold_penalty_tuples=[(0.001, 0), (0.03, 0.25), (0.08, 0.5)])


class LatencyStepUtility:

    def __init__(self, slo_target: float, max_value: float = 1.0):
        self.target_latency = slo_target
        self.max_value = max_value

    def __call__(self, latency: Union[float, np.array]) -> float:
        latency = np.array(latency)
        return np.mean((latency <= self.target_latency).astype(float) * self.max_value)


class LatencyUtility:

    def __init__(self, slo_target: float, max_value: float = 1.0, alpha=1.0):
        self.target_latency = slo_target
        self.alpha = alpha
        self.max_value = max_value

    def __call__(self, latency: Union[float, np.array]) -> float:
        latency = np.array(latency)
        # set very low latency to zero latency to avoid division by zero
        latency = np.clip(latency, a_min=1e-4, a_max=None)
        return np.mean(
            np.clip(np.power(self.target_latency / latency, self.alpha),
                    a_min=None, a_max=1.0) * self.max_value)


class SquareLatencyUtility(LatencyUtility):

    def __init__(self, slo_target: float, max_value: float = 1.0):
        super().__init__(slo_target=slo_target, max_value=max_value, alpha=2)


class CubicLatencyUtility(LatencyUtility):

    def __init__(self, slo_target: float, max_value: float = 1.0):
        super().__init__(slo_target=slo_target, max_value=max_value, alpha=3)


class SQRTLatencyUtility(LatencyUtility):

    def __init__(self, slo_target: float, max_value: float = 1.0):
        super().__init__(slo_target=slo_target, max_value=max_value, alpha=1/2)


class CBRTLatencyUtility(LatencyUtility):

    def __init__(self, slo_target: float, max_value: float = 1.0):
        super().__init__(slo_target=slo_target, max_value=max_value, alpha=1/3)


class LatencyUtilityWithPenalty:

    def __init__(self, slo_target, threshold_penalty_tuples):
        self.utility_fn = LatencyUtility(slo_target)
        self.penalty_fn = SLOPenalty(threshold_penalty_tuples)

    def __call__(self, latency: float):
        return 1 - self.penalty_fn(1 - self.utility_fn(latency))


class Pow100LatencyUtility(LatencyUtility):

    def __init__(self, slo_target: float, max_value: float = 1.0):
        super().__init__(slo_target=slo_target, max_value=max_value, alpha=100)


_UTILITY_FUNC_CLASSES = {
    "latency": LatencyUtility,
    "square_latency": SquareLatencyUtility,
    "cubic_latency": CubicLatencyUtility,
    "latency_step": LatencyStepUtility,
    "sqrt_latency": SQRTLatencyUtility,
    "cbrt_latency": CBRTLatencyUtility,
    "latency_with_penalty": LatencyUtilityWithPenalty,
    "pow100_latency": Pow100LatencyUtility,
}


def create_utility_func(type_name: str, *args, **kwargs) -> Callable[[Any], float]:
    return _UTILITY_FUNC_CLASSES[type_name](*args, **kwargs)
