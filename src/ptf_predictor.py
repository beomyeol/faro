import logging
from math import ceil
from typing import List, Optional
import warnings

import pandas as pd
import numpy as np
import pytorch_forecasting as ptf

from predictor import BasePredictor

_LOGGER = logging.getLogger(__name__)


class PyTorchForcastingPredictor(BasePredictor):

    def __init__(self, model_type: str, model_path: str, window_size_s: int,
                 head_name: str, deployment_name: str, unit_time_s: int = 1):
        super().__init__(unit_time_s=unit_time_s)
        self.model = self.load_model(model_type, model_path)
        self.window_size_s = window_size_s
        self.head_name = head_name
        self.deployment_name = deployment_name

        if window_size_s > self.model.hparams.prediction_length:
            raise ValueError("window size ({}) > pred length ({})".format(
                window_size_s, self.model.hparams.prediction_length
            ))

    @staticmethod
    def load_model(model_type: str, model_path: str) -> ptf.BaseModel:
        _LOGGER.info("Loading model (%s) from %s", model_type, model_path)
        if model_type == "nhits":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ptf.NHiTS.load_from_checkpoint(model_path)
        elif model_type == "deepar":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ptf.DeepAR.load_from_checkpoint(model_path)
        else:
            raise ValueError("Unknown type: " + model_type)

    @property
    def context_len(self):
        return self.model.hparams.context_length

    @property
    def pred_len(self):
        return self.model.hparams.prediction_length

    def predict(self, input_metrics, current_ts: float) -> Optional[List[int]]:
        if not self.update_history(input_metrics, current_ts):
            return None
        # _LOGGER.info("[%.3f] (%s:%s) history: %s", current_ts, self.head_name,
        #              self.deployment_name, self.history)
        x = np.concatenate([self.history, np.zeros(self.pred_len)])
        df = pd.DataFrame(
            dict(value=x, group=np.repeat(0, len(x)), time_idx=np.arange(len(x)))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(df).numpy().reshape(-1)
        return pred[:self.window_size_s]
