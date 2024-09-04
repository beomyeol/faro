import logging
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.dataprocessing.transformers import Scaler
import torch
import numpy as np

from predictor import BasePredictor

_LOGGER = logging.getLogger(__name__)


_MODEL_CACHE = {}


class DartsPredictor(BasePredictor):

    def __init__(self, model_path: str, window_size_s: int,
                 head_name: str, deployment_name: str, device: str = "cpu",
                 num_samples: int = 1, quantile: float = 0.8,
                 unit_time_s: int = 1):
        super().__init__(unit_time_s=unit_time_s)
        self.model, self.scaler = self.load_model(
            model_path, map_location=torch.device(device))
        assert window_size_s % unit_time_s == 0
        self.unit_time_s = unit_time_s
        self.window_size = window_size_s // unit_time_s
        self.head_name = head_name
        self.deployment_name = deployment_name
        self.num_samples = num_samples
        self.quantile = quantile

        self.scaler_idx = None
        if head_name.startswith("cluster"):
            idx_str = head_name[len("cluster"):]
            if idx_str.endswith("-ray-head"):
                idx_str = idx_str[:-len("-ray-head")]
            self.scaler_idx = int(idx_str) - 1
        elif head_name.find("serve-cluster") > 0:
            begin = head_name.find("serve-cluster") + len("serve-cluster")
            end = head_name.find("-", begin)
            self.scaler_idx = int(head_name[begin:end]) - 1
        if len(self.scaler._fitted_params) == 1:
            # individually trained model
            self.scaler_idx = 0

        if self.scaler_idx is None:
            raise RuntimeError("Cannot infer scaler index...")

        _LOGGER.info("(%s:%s) num_samples=%d, quantile=%s, scaler_idx=%d",
                     head_name, deployment_name, num_samples, quantile,
                     self.scaler_idx)

        if self.window_size > self.model.output_chunk_length:
            _LOGGER.warning(
                "window size (%d) > pred length (%d)",
                self.window_size, self.model.output_chunk_length)

    def update_quantile(self, quantile):
        _LOGGER.info("update quantile %s -> %s", self.quantile, quantile)
        self.quantile = quantile

    def transform(self, series: TimeSeries):
        return self.scaler.ts_transform(
            series, self.scaler._fitted_params[self.scaler_idx])

    def inverse_transform(self, series: TimeSeries):
        return self.scaler.ts_inverse_transform(
            series, self.scaler._fitted_params[self.scaler_idx])

    @staticmethod
    def load_model(path, map_location=None) -> Tuple[ForecastingModel, Scaler]:
        _LOGGER.info("Loading model from %s", path)

        path = Path(path).resolve()

        if path in _MODEL_CACHE:
            _LOGGER.info("Use cached model")
            return _MODEL_CACHE[path]

        scaler_path = path.with_name("scaler.pt")
        scaler = torch.load(scaler_path)

        with open(path, "rb") as fin:
            model = torch.load(fin, map_location=map_location)

        ckpt_path = path.with_name(path.name + ".ckpt")
        if ckpt_path.exists():
            model.model = model.model.__class__.load_from_checkpoint(ckpt_path)
            model.trainer = None

        model.trainer_params = {
            "enable_progress_bar": False,
            "logger": False,
        }

        if map_location == torch.device("cuda"):
            _LOGGER.info("using cuda")
            model.trainer_params["accelerator"] = "cuda"
            model.trainer_params["devices"] = [0]

        _MODEL_CACHE[path] = model, scaler

        return model, scaler

    @property
    def context_len(self):
        return self.model.input_chunk_length

    @property
    def pred_len(self):
        return self.model.output_chunk_length

    def predict(self, input_metrics, current_ts: float) -> Optional[List[int]]:
        if self.window_size == 0:
            return None
        if not self.update_history(input_metrics, current_ts):
            return None
        _LOGGER.info("[%.3f] (%s:%s) history: %s", current_ts, self.head_name,
                     self.deployment_name, self.history)
        series = self.transform(TimeSeries.from_values(self.history))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_raw = self.model.predict(n=self.window_size, series=series,
                                          num_samples=self.num_samples)
            if pred_raw.n_samples > 1:
                pred_raw = pred_raw.quantile(self.quantile)
            pred = self.inverse_transform(pred_raw)
        return np.clip(pred.univariate_values()[:self.window_size], a_min=1, a_max=None)
