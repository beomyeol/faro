from typing import List
import pickle
from collections import defaultdict

import pandas as pd


def parse_path(path):
    cpu_pattern = "_cpus/"
    cpu_pattern_begin = path.find(cpu_pattern)
    num_cpus = path[path.rfind("/", 0, cpu_pattern_begin)+1:cpu_pattern_begin]
    target_str = path[cpu_pattern_begin + len(cpu_pattern):]
    split: List[str] = target_str.split("/")

    config_dict = {}
    config_dict["num_cpus"] = num_cpus
    if split[0].startswith("poisson"):
        config_dict["poisson"] = split[0][split[0].rfind("_")+1:]
        split.pop(0)
    config_dict["policy"] = split[0]
    policy = config_dict["policy"]
    if policy[-1].isdigit() and policy[policy.rfind("_")+1].isdigit():
        idx = int(policy[policy.rfind("_")+1:])
        config_dict["poisson"] = idx
        config_dict["policy"] = policy = policy[:policy.rfind("_")]
    else:
        config_dict.setdefault("poisson", 0)
        # config_dict["poisson"] = 0
    for v in split[1:]:
        if v.startswith("pred_"):
            config_dict["pred_aggr"] = v
        elif v.startswith("window_"):
            config_dict["ws"] = v[len("window_"):]
        elif v.startswith("idle_timeout_"):
            config_dict["idle_timeout"] = v[len("idle_timeout_"):]
        elif v.endswith("_latency"):
            config_dict["latency_estimator"] = v
        elif v in ["nhits", "deepar"]:
            config_dict["model"] = v
        elif "model_" in v:
            config_dict["model_config"] = v
        else:
            raise ValueError(v)
    return config_dict


def config_dict_to_str_list(config_dict):
    retval = []
    retval.append(config_dict["num_cpus"])
    retval.append(config_dict.get("poisson", ""))
    retval.append(config_dict["policy"])
    retval.append(config_dict.get("pred_aggr", ""))
    retval.append(config_dict.get("ws", ""))
    retval.append(config_dict.get("idle_timeout", ""))
    retval.append(config_dict.get("latency_estimator", ""))
    retval.append(config_dict.get("model", ""))
    retval.append(config_dict.get("model_config", ""))
    return retval


def parse_stats(path):
    with open(path, "rb") as f:
        stats_dict = pickle.load(f)

    transposed = defaultdict(list)

    for path, stats in stats_dict.items():
        transposed["path"].append(str(path))
        if "avg_slo_rate_serve-cluster0" not in stats:
            # index starts from 1. renumber this
            new_stats = {}
            for k, v in stats.items():
                pos = k.rfind("cluster")
                if pos > 0:
                    idx = int(k[pos+len("cluster"):])
                    k = k[:pos] + f"cluster{idx-1}"
                new_stats[k] = v
            stats = new_stats
        for k, v in stats.items():
            transposed[k].append(v)

    df = pd.DataFrame(transposed)

    slo_rate_keys = []
    i = 0
    while f"slo_rate_serve-cluster{i}" in df:
        slo_rate_keys.append(f"slo_rate_serve-cluster{i}")
        i += 1

    i = 0
    while f"slo_rate_cluster{i}" in df:
        slo_rate_keys.append(f"slo_rate_cluster{i}")
        i += 1

    df.insert(1, "avg_total_slo_rate", df.pop("avg_total_slo_rate"))
    df.insert(1, "avg_total_utility", df.pop("avg_total_utility"))
    # df.insert(1, "avg_effective_utility_mean", df.pop("avg_effective_utility_mean"))
    df.insert(1, "utility", df.pop("utility"))

    df_wo_count = df.loc[:, ~df.columns.str.contains("count")]
    head_line = [
        "num_cpus",
        "poisson",
        "policy",
        "pred_aggr",
        "ws",
        "idle_timeout",
        "latency_estimator",
        "model",
        "model_config",
    ]
    head_line.extend([v.replace("_", " ")
                     for v in df_wo_count.columns.values[1:]])
    configs = {k: v for k, v in zip(head_line, zip(
        *[config_dict_to_str_list(parse_path(v.values[0])) for _, v in df_wo_count.iterrows()]))}
    config_df = pd.DataFrame(configs)
    return pd.concat([config_df, df_wo_count.drop(columns=["path"])], axis=1)


# M/D/c COBLYA normal (final, compact, real)
REAL_POLICIES = [
    "fairshare",
    "oneshot",
    "aiad",
    "mark",
    "faro_fair",
    "faro_sum",
    "faro_fair_sum",
    # "faro_fair_sum_old",
    "faro_penalty_sum",
    "faro_penalty_fair_sum",
]
# sim_policies = [
#     "fairshare",

#     "oneshot",
#     "aiad",
#     "hybrid_ai_pred_darts", # mark
#     "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_pureminmax", # faro fair
#     "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust",  # faro sum
#     "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_minmax3", # faro fair sum
#     "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust", # faro penalty sum
#     "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust_minmax3", # faro penalty fair sum
# ]

SIM_POLICIES_APP = [
    "fairshare",
    "oneshot",
    "aiad",
    "hybrid_ai_pred_darts",  # mark
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_pureminmax_app",  # faro fair
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_app",  # faro sum
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_minmax3_app",  # faro fair sum
    # faro penalty sum
    "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust_app",
    # faro penalty fair sum
    "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust_minmax3_app",
]

SIM_POLICIES = [
    "fairshare",
    "oneshot",
    "aiad",
    "hybrid_ai_pred_darts",  # mark
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_pureminmax",  # faro fair
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust",  # faro sum
    "hybrid_ai_pred_darts_cobyla_utility_latency_coldstart_mdc0.99_adjust_minmax3",  # faro fair sum
    "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust", # faro penalty sum
    "hybrid_ai_pred_darts_cobyla_drop_linear_int_utility_latency_coldstart_mdc0.99_adjust_minmax3", # faro penalty fair sum
]

LABELS = [
    "FairShare",
    "Oneshot",
    "AIAD",

    "MArk/Cocktail/Barista",
    "Faro-Fair",
    "Faro-Sum",
    "Faro-FairSum",
    "Faro-PenaltySum",
    "Faro-PenaltyFairSum",
]

REAL_LABELS = {
    policy: label for policy, label in zip(REAL_POLICIES, LABELS)
}
SIM_LABELS = {
    policy: label for policy, label in zip(SIM_POLICIES, LABELS)
}
SIM_LABELS_APP = {
    policy: label for policy, label in zip(SIM_POLICIES_APP, LABELS)
}
MODEL_LABELS = {}


def get_stat_dfs(df, policies, labels, model_labels):
    target_df = df[
        # & (df.latency_estimator != "max_latency")
        (df.latency_estimator != "p80_latency")
        # & (df.pred_aggr != "pred_max") & (df.pred_aggr != "pred_p80")
        # & (df.pred_aggr != "pred_max")
        & (df.pred_aggr != "pred_p80") & (df.pred_aggr != "pred_avg")
        # & (df.pred_aggr != "pred_p80") & (df.pred_aggr != "pred_avg") & (df.pred_aggr != "pred_none")
        # & (df.model_config != "model_8")
        & (df.model_config != "model_4") & (df.model_config != "model_5") & (df.model_config != "model_6")
        & (df.model_config != "model_10")
        & (df.ws != "300s") & (df.ws != "480s")
        & (df.idle_timeout != "5s")
        & (~df.policy.str.endswith("utility_latency"))
        & (~df.policy.str.contains("scale"))
        & (~df.policy.str.contains("w_current"))
        & (~df.policy.str.contains("sqrt"))
        & (~df.policy.str.contains("cbrt"))
        & (~df.policy.str.contains("weighted"))
    ].copy()

    cluster_utils = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                cluster_utils[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = target[(
                    target.ws == ws) & (target.model_config == model_config)].loc[:, ["avg_total_utility"]].values.reshape(-1)
    print("cluster_utils")
    for k, v in cluster_utils.items():
        print(k, len(v))
    cluster_util_df = pd.DataFrame(cluster_utils)

    cluster_effective_utils = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                cluster_effective_utils[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = target[(
                    target.ws == ws) & (target.model_config == model_config)].loc[:, ["avg_total_effective_utility"]].values.reshape(-1)
    print("cluster_effective_utils")
    for k, v in cluster_effective_utils.items():
        print(k, len(v))
    cluster_effective_util_df = pd.DataFrame(cluster_effective_utils)

    cluster_slo = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                cluster_slo[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = target[(
                    target.ws == ws) & (target.model_config == model_config)].loc[:, ["avg_total_slo_rate"]].values.reshape(-1)
    print("cluster_slo")
    for k, v in cluster_slo.items():
        print(k, len(v))
    cluster_slo_df = pd.DataFrame(cluster_slo)

    avg_utils = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                avg_utils[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = target[(target.ws == ws) & (target.model_config == model_config)].loc[:, [
                    x for x in df.columns if x.startswith("avg_util_") and x != "avg_utility_mean"]].mean(axis=0).values.reshape(-1)
    print("avg_utils")
    for k, v in avg_utils.items():
        print(k, len(v))
    avg_util_df = pd.DataFrame(avg_utils)

    slo_rates = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                slo_rates[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = 1 - target[(target.ws == ws) & (
                    target.model_config == model_config)].loc[:, [x for x in df.columns if x.startswith("avg_slo_rate_")]].mean(axis=0).values.reshape(-1)

    print("slo_rates")
    for k, v in slo_rates.items():
        print(k, len(v))
    slo_rates_df = pd.DataFrame(slo_rates)

    avg_effective_utils = {}
    for policy in policies:
        target = target_df[target_df.policy == policy]
        for model_config in sorted(target.model_config.unique()):
            for ws in sorted(target.ws.unique()):
                avg_effective_utils[f"{labels[policy]} {ws} {model_labels.get(model_config, model_config)}".strip()] = target[(target.ws == ws) & (
                    target.model_config == model_config)].loc[:, [x for x in df.columns if x.startswith("avg_effective_util_")]].mean(axis=0).values.reshape(-1)

    print("avg_effective_utils")
    for k, v in avg_effective_utils.items():
        print(k, len(v))
    avg_effective_util_df = pd.DataFrame(avg_effective_utils)

    return cluster_util_df, cluster_slo_df, cluster_effective_util_df, avg_util_df, slo_rates_df, avg_effective_util_df

# merged_slo_rates


def merge(real_series, sim_series):
    return real_series.to_frame("Cluster").merge(sim_series.rename("Simulation"), left_index=True, right_index=True)


def trim_column_names(df):
    new_names = {column: column.split(" ")[0] for column in df.columns}
    df.rename(columns=new_names, inplace=True)
