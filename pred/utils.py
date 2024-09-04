import numpy as np
import pandas as pd

import torch

from darts.utils.likelihood_models import (
    LaplaceLikelihood,
    PoissonLikelihood,
    GaussianLikelihood,
    NegativeBinomialLikelihood,
)

# from gluonts.torch.distributions.distribution_output import (
#     NegativeBinomialOutput,
#     NormalOutput,
#     StudentTOutput,
# )


def build_dataset(counts):
    values = []
    groups = []
    time_idxs = []
    for i, ts in enumerate(counts):
        values.append(ts)
        groups.append(np.repeat(i, len(ts)))
        time_idxs.append(np.arange(len(ts)))

    data = pd.DataFrame(
        dict(
            value=np.concatenate(values),
            group=np.concatenate(groups),
            time_idx=np.concatenate(time_idxs),
        )
    )
    data = data.astype(dict(value=np.float32))
    return data


def build_dataset_from_df(df, check_validity=True):
    values = []
    groups = []
    time_idxs = []
    for fname, d in df.groupby("hash_func"):
        sorted_d = d.sort_values("day")
        max_day = sorted_d.day.max()
        min_day = sorted_d.day.min()
        if check_validity:
            assert len(sorted_d.day) == (max_day - min_day + 1)
        counts = pd.concat(sorted_d.counts.tolist(), ignore_index=True)
        values.append(counts.tolist())
        groups.append(np.repeat(fname, len(counts)))
        time_idxs.append(counts.index.tolist())

    data = pd.DataFrame(
        dict(
            value=np.concatenate(values),
            group=np.concatenate(groups),
            time_idx=np.concatenate(time_idxs),
        )
    )
    data = data.astype(dict(value=np.float32))
    return data


_LIKELIHOODS = {
    "laplace": LaplaceLikelihood,
    "poisson": PoissonLikelihood,
    "gaussian": GaussianLikelihood,
    "negbinomial": NegativeBinomialLikelihood,
}


def get_likelihood(name: str, *args, **kwargs):
    return _LIKELIHOODS[name](*args, **kwargs)


_OPTIMS = {
    "adam": torch.optim.Adam,
    "radam": torch.optim.RAdam,
}


def get_optim_cls(name: str):
    return _OPTIMS[name.lower()]


# _DISTR_OUTPUTS = {
#     "student": StudentTOutput,
#     "negbinomial": NegativeBinomialOutput,
#     "gaussian": NormalOutput,
# }


# def get_distr_output(name: str):
#     return _DISTR_OUTPUTS[name]()
