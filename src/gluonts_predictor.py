import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from gluonts.model.predictor import Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset

from predictor import BasePredictor

_LOGGER = logging.getLogger(__name__)

_MODEL_CACHE = {}


class GluonTSPredictor(BasePredictor):

    def __init__(self, model_path: str, window_size_s: int,
                 head_name: str, deployment_name: str,
                 num_samples: Optional[int] = None, quantile: float = -1,
                 unit_time_s: int = 1, device: str = "cuda"):
        super().__init__(unit_time_s=unit_time_s)
        self.model = self.load_model(model_path, device)
        assert window_size_s % unit_time_s == 0
        self.window_size = window_size_s // unit_time_s
        self.head_name = head_name
        self.deployment_name = deployment_name
        self.num_samples = num_samples
        self.quantile = quantile
        if unit_time_s == 1.0:
            self.freq = pd.offsets.Second()
        elif unit_time_s == 60.0:
            self.freq = pd.offsets.Minute()
        else:
            raise ValueError(f"invalid unit time: {unit_time_s}")

        _LOGGER.info(
            "(%s:%s) GluonTSPredictor window_size=%d, num_samples=%s, "
            "quantile=%s, path=%s",
            head_name, deployment_name, self.window_size, num_samples,
            quantile, model_path)

        if self.window_size > self.pred_len:
            raise ValueError("window size ({}) > pred length (%d)".format(
                self.window_size, self.pred_len))

    @staticmethod
    def load_model(path, device: str = "cpu") -> PyTorchPredictor:
        path = Path(path).resolve()
        _LOGGER.info("Loading model from %s", path)

        if path in _MODEL_CACHE:
            _LOGGER.info("Use cached model")
            return _MODEL_CACHE[path]

        model = Predictor.deserialize(path, device=device)
        _MODEL_CACHE[path] = model

        return model

    @property
    def context_len(self):
        return self.model.network.model.context_length

    @property
    def pred_len(self):
        return self.model.prediction_length

    def predict(self, input_metrics, current_ts: float) -> Optional[List[int]]:
        if not self.update_history(input_metrics, current_ts):
            return None
        input_dataset = ListDataset(
            [{"start": pd.Period.now(freq=self.freq) - self.context_len,
              "target": self.history}], freq=self.freq)
        pred = list(self.model.predict(
            input_dataset,
            num_samples=self.num_samples))[0]
        if self.quantile < 0:
            return np.clip(np.round(pred.mean), a_min=1, a_max=None)
        else:
            return np.clip(np.round(pred.quantile(self.quantile)), a_min=1, a_max=None)
