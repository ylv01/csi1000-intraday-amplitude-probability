from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


DATASETS: dict[str, dict[str, str]] = {
    "csi1000": {
        "daily": "CSI1000_daily_amplitude_20190102_20260724.csv",
        "minute": "000852_1m_20190101_latest.csv",
        "label": "中证1000",
    },
    "csi500": {
        "daily": "CSI500_daily.csv",
        "minute": "CSI500_1m.csv",
        "label": "中证500",
    },
    "csi300": {
        "daily": "CSI300_daily.csv",
        "minute": "CSI300_1m.csv",
        "label": "沪深300",
    },
    "chinext": {
        "daily": "ChiNext_daily.csv",
        "minute": "ChiNext_1m.csv",
        "label": "创业板指",
    },
    "sse50": {
        "daily": "SSE50_daily.csv",
        "minute": "SSE50_1m.csv",
        "label": "上证50",
    },
}

ALPHAS_SCREEN = (0.10, 0.50, 0.90)
ALPHAS_FULL = tuple(round(x, 2) for x in np.arange(0.10, 0.91, 0.10))
OUTER_YEARS = (2022, 2023, 2024, 2025)
EXCLUDED_DYNAMIC_DATES = (pd.Timestamp("2020-04-20"),)
TIME_FEATURES = (
    "bar_index",
    "elapsed_fraction",
    "remaining_fraction",
    "is_preopen",
    "is_afternoon",
    "minutes_since_open",
    "minutes_to_close",
)


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.025
    max_estimators: int = 1500
    early_stopping_rounds: int = 120
    early_stop_days: int = 120
    max_depth: int = 4
    num_leaves: int = 12
    min_child_samples: int = 240
    min_split_gain: float = 0.01
    max_bin: int = 63
    feature_fraction: float = 0.75
    bagging_fraction: float = 0.80
    bagging_freq: int = 1
    lambda_l1: float = 0.30
    lambda_l2: float = 3.0
    random_state: int = 20260727
    min_group_improvement: float = 0.02
    min_years_improved: int = 3

    def lgb_params(self, alpha: float, n_estimators: int | None = None) -> dict[str, Any]:
        return {
            "boosting_type": "gbdt",
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "n_estimators": n_estimators or self.max_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "min_split_gain": self.min_split_gain,
            "max_bin": self.max_bin,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "deterministic": True,
            "force_col_wise": True,
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }


def _read_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    daily["date"] = daily["date"].dt.normalize()
    return daily


def _read_minute(path: Path) -> tuple[pd.DataFrame, int]:
    minute = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    minute["date"] = minute["datetime"].dt.normalize()
    invalid = (
        minute["high"].lt(minute[["open", "close", "low"]].max(axis=1))
        | minute["low"].gt(minute[["open", "close", "high"]].min(axis=1))
    )
    invalid_count = int(invalid.sum())
    if invalid_count:
        minute["high"] = minute[["open", "high", "low", "close"]].max(axis=1)
        minute["low"] = minute[["open", "high", "low", "close"]].min(axis=1)
    return minute, invalid_count


def audit_inputs(data_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    base_timestamps: pd.Series | None = None

    for name, spec in DATASETS.items():
        daily_path = data_dir / spec["daily"]
        minute_path = data_dir / spec["minute"]
        if not daily_path.exists() or not minute_path.exists():
            continue

        daily = _read_daily(daily_path)
        minute, invalid_count = _read_minute(minute_path)
        bars = minute.groupby("date").size()
        timestamps = minute["datetime"].reset_index(drop=True)
        if base_timestamps is None:
            base_timestamps = timestamps
            aligned = True
        else:
            aligned = bool(base_timestamps.equals(timestamps))

        prev_close = minute.groupby("date", sort=False)["close"].shift(1)
        prev_close = prev_close.fillna(minute["open"])
        internal_gap = (np.log(minute["open"] / prev_close).abs() * 100).replace(
            [np.inf, -np.inf], np.nan
        )
        max_gap_idx = internal_gap.idxmax()
        max_gap = float(internal_gap.loc[max_gap_idx])

        rows.append(
            {
                "dataset": name,
                "label": spec["label"],
                "daily_rows": len(daily),
                "minute_rows": len(minute),
                "start": str(daily["date"].min().date()),
                "end": str(daily["date"].max().date()),
                "daily_duplicates": int(daily["date"].duplicated().sum()),
                "minute_duplicates": int(minute["datetime"].duplicated().sum()),
                "missing_values": int(daily.isna().sum().sum() + minute.isna().sum().sum()),
                "min_bars_per_day": int(bars.min()),
                "max_bars_per_day": int(bars.max()),
                "invalid_ohlc_repaired": invalid_count,
                "timestamps_aligned": aligned,
                "largest_internal_gap_pct": max_gap,
                "largest_internal_gap_at": str(minute.loc[max_gap_idx, "datetime"]),
            }
        )
        if invalid_count:
            anomalies.append(
                {
                    "dataset": name,
                    "type": "minute_ohlc_envelope",
                    "count": invalid_count,
                    "action": "high/low expanded to contain open and close",
                }
            )
        if max_gap > 1.0:
            anomalies.append(
                {
                    "dataset": name,
                    "type": "intraday_level_break",
                    "count": 1,
                    "at": str(minute.loc[max_gap_idx, "datetime"]),
                    "magnitude_pct": max_gap,
                    "action": "exclude the complete trading date from dynamic modelling",
                }
            )

    anomalies.append(
        {
            "dataset": "all",
            "type": "known_cross_index_level_break",
            "count": 1,
            "at": "2020-04-20",
            "action": "exclude the complete trading date from dynamic modelling",
        }
    )
    return pd.DataFrame(rows), anomalies


def _minute_derived(minute: pd.DataFrame) -> pd.DataFrame:
    minute = minute.copy()
    minute["bar_index"] = minute.groupby("date", sort=False).cumcount() + 1
    previous_close = minute.groupby("date", sort=False)["close"].shift(1)
    minute["log_return"] = np.log(minute["close"] / previous_close)
    first_bar = minute["bar_index"].eq(1)
    minute.loc[first_bar, "log_return"] = np.log(
        minute.loc[first_bar, "close"] / minute.loc[first_bar, "open"]
    )
    minute["log_return"] = minute["log_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minute["abs_log_return"] = minute["log_return"].abs()
    minute["squared_return"] = minute["log_return"].pow(2)
    minute["positive_squared_return"] = minute["squared_return"].where(
        minute["log_return"].gt(0), 0.0
    )
    minute["negative_squared_return"] = minute["squared_return"].where(
        minute["log_return"].lt(0), 0.0
    )
    previous_abs = minute.groupby("date", sort=False)["abs_log_return"].shift(1).fillna(0.0)
    minute["bipower_term"] = minute["abs_log_return"] * previous_abs
    return minute


def _daily_signals(daily: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    day = daily.set_index("date").copy()
    day["amp"] = (day["high"] - day["low"]) / day["pre_close"] * 100
    day["ret"] = (day["close"] / day["pre_close"] - 1) * 100
    day["abs_ret"] = day["ret"].abs()
    day["gap"] = (day["open"] / day["pre_close"] - 1) * 100
    day["gap_abs"] = day["gap"].abs()
    day["oc_ret"] = (day["close"] / day["open"] - 1) * 100
    day["range_log"] = np.log(day["high"] / day["low"]) * 100
    price_range = (day["high"] - day["low"]).replace(0, np.nan)
    day["clv"] = (day["close"] - day["low"]) / price_range
    day["body"] = (day["close"] - day["open"]).abs() / day["pre_close"] * 100
    day["upper_wick"] = (
        day["high"] - day[["open", "close"]].max(axis=1)
    ) / day["pre_close"] * 100
    day["lower_wick"] = (
        day[["open", "close"]].min(axis=1) - day["low"]
    ) / day["pre_close"] * 100
    day["log_money"] = np.log1p(day["money"].clip(lower=0))
    day["log_volume"] = np.log1p(day["volume"].clip(lower=0))

    grouped = minute.groupby("date", sort=False)
    rv_variance = grouped["squared_return"].sum() * 10000
    bipower = grouped["bipower_term"].sum() * (math.pi / 2.0) * 10000
    intraday = pd.DataFrame(index=rv_variance.index)
    intraday["rv"] = np.sqrt(rv_variance.clip(lower=0))
    intraday["rv_up"] = np.sqrt(
        (grouped["positive_squared_return"].sum() * 10000).clip(lower=0)
    )
    intraday["rv_down"] = np.sqrt(
        (grouped["negative_squared_return"].sum() * 10000).clip(lower=0)
    )
    intraday["down_variance_share"] = (
        grouped["negative_squared_return"].sum()
        / grouped["squared_return"].sum().replace(0, np.nan)
    )
    intraday["jump"] = np.sqrt((rv_variance - bipower).clip(lower=0))
    intraday["max_abs_1m"] = grouped["abs_log_return"].max() * 100
    intraday["volume_cv"] = grouped["volume"].std() / grouped["volume"].mean().replace(0, np.nan)
    intraday["zero_volume_share"] = grouped["volume"].apply(lambda x: float(x.eq(0).mean()))

    for label, mask in {
        "rv_first30": minute["bar_index"].le(30),
        "rv_last30": minute["bar_index"].gt(210),
        "rv_am": minute["bar_index"].le(120),
        "rv_pm": minute["bar_index"].gt(120),
    }.items():
        variance = minute.loc[mask].groupby("date")["squared_return"].sum() * 10000
        intraday[label] = np.sqrt(variance.clip(lower=0))

    return day.join(intraday, how="left")


def _historical_features(day: pd.DataFrame, prefix: str, full: bool) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}

    if full:
        lag_spec = {
            "amp": (1, 2, 3, 5, 10, 20),
            "ret": (1, 2, 3, 5, 10),
            "abs_ret": (1, 2, 3, 5, 10),
            "gap": (1, 2, 5, 10),
            "gap_abs": (1, 2, 5, 10),
            "oc_ret": (1, 2, 5),
            "range_log": (1, 2, 5),
            "clv": (1, 2, 5),
            "body": (1, 2, 5),
            "upper_wick": (1, 2, 5),
            "lower_wick": (1, 2, 5),
            "rv": (1, 2, 3, 5, 10),
            "rv_up": (1, 2, 5),
            "rv_down": (1, 2, 5),
            "down_variance_share": (1, 2, 5),
            "jump": (1, 2, 5),
            "max_abs_1m": (1, 2, 5),
            "rv_first30": (1, 2, 5),
            "rv_last30": (1, 2, 5),
            "rv_am": (1, 2, 5),
            "rv_pm": (1, 2, 5),
            "volume_cv": (1, 5),
            "zero_volume_share": (1, 5),
        }
    else:
        lag_spec = {
            "amp": (1, 2, 5),
            "ret": (1, 2, 5),
            "abs_ret": (1, 2, 5),
            "gap_abs": (1, 5),
            "rv": (1, 2, 5),
            "rv_down": (1, 5),
            "down_variance_share": (1,),
            "jump": (1,),
            "max_abs_1m": (1,),
            "rv_last30": (1,),
        }

    for signal, lags in lag_spec.items():
        for lag in lags:
            values[f"{prefix}hist_{signal}_lag{lag}"] = day[signal].shift(lag)

    rolling_spec = {
        "amp": (5, 20, 60, 120) if full else (5, 20, 60),
        "rv": (5, 20, 60) if full else (5, 20),
        "abs_ret": (5, 20),
        "jump": (20,),
    }
    for signal, windows in rolling_spec.items():
        shifted = day[signal].shift(1)
        for window in windows:
            rolling = shifted.rolling(window, min_periods=max(3, window // 2))
            values[f"{prefix}hist_{signal}_mean{window}"] = rolling.mean()
            if full or signal in {"amp", "rv"}:
                values[f"{prefix}hist_{signal}_std{window}"] = rolling.std()

    amp_shifted = day["amp"].shift(1)
    for window in ((20, 60, 120) if full else (20, 60)):
        rolling = amp_shifted.rolling(window, min_periods=max(5, window // 2))
        values[f"{prefix}hist_amp_median{window}"] = rolling.median()
        values[f"{prefix}hist_amp_q90_{window}"] = rolling.quantile(0.90)

    rv_shifted = day["rv"].shift(1)
    values[f"{prefix}hist_amp_ratio5_60"] = (
        amp_shifted.rolling(5, min_periods=3).mean()
        / amp_shifted.rolling(60, min_periods=30).mean()
    )
    values[f"{prefix}hist_rv_ratio5_20"] = (
        rv_shifted.rolling(5, min_periods=3).mean()
        / rv_shifted.rolling(20, min_periods=10).mean()
    )
    values[f"{prefix}hist_trend5"] = (day["close"].shift(1) / day["close"].shift(6) - 1) * 100
    values[f"{prefix}hist_trend20"] = (
        day["close"].shift(1) / day["close"].shift(21) - 1
    ) * 100

    money = day["log_money"].shift(1)
    volume = day["log_volume"].shift(1)
    values[f"{prefix}hist_money_rel20"] = money - money.rolling(20, min_periods=10).mean()
    values[f"{prefix}hist_volume_rel20"] = volume - volume.rolling(20, min_periods=10).mean()
    if full:
        values[f"{prefix}hist_money_rel60"] = money - money.rolling(60, min_periods=30).mean()
        values[f"{prefix}hist_volume_rel60"] = volume - volume.rolling(60, min_periods=30).mean()

    result = pd.DataFrame(values, index=day.index)
    return result.replace([np.inf, -np.inf], np.nan)


def _snapshot_features(
    day: pd.DataFrame,
    minute: pd.DataFrame,
    prefix: str,
    is_target: bool,
) -> pd.DataFrame:
    minute = minute.copy()
    minute = minute.merge(
        day[["pre_close", "amp"]].rename(columns={"amp": "final_amp"}),
        left_on="date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    grouped = minute.groupby("date", sort=False)
    minute["cum_high"] = grouped["high"].cummax()
    minute["cum_low"] = grouped["low"].cummin()
    minute["cum_rv_variance"] = grouped["squared_return"].cumsum()
    minute["cum_up_variance"] = grouped["positive_squared_return"].cumsum()
    minute["cum_down_variance"] = grouped["negative_squared_return"].cumsum()
    minute["cum_abs_return"] = grouped["abs_log_return"].cumsum()
    minute["cum_money"] = grouped["money"].cumsum()
    minute["cum_volume"] = grouped["volume"].cumsum()
    minute["first_open"] = grouped["open"].transform("first")
    minute["current_amp"] = (
        (minute["cum_high"] - minute["cum_low"]) / minute["pre_close"] * 100
    )
    minute["current_return"] = (minute["close"] / minute["pre_close"] - 1) * 100
    minute["current_rv"] = np.sqrt(minute["cum_rv_variance"].clip(lower=0)) * 100
    minute["current_down_share"] = (
        minute["cum_down_variance"] / minute["cum_rv_variance"].replace(0, np.nan)
    )
    minute["current_efficiency"] = (
        np.log(minute["close"] / minute["first_open"]).abs()
        / minute["cum_abs_return"].replace(0, np.nan)
    )
    current_range = (minute["cum_high"] - minute["cum_low"]).replace(0, np.nan)
    minute["current_clv"] = (minute["close"] - minute["cum_low"]) / current_range

    new_high = minute["high"].ge(minute["cum_high"] - 1e-12)
    new_low = minute["low"].le(minute["cum_low"] + 1e-12)
    last_high_bar = minute["bar_index"].where(new_high).groupby(minute["date"]).ffill()
    last_low_bar = minute["bar_index"].where(new_low).groupby(minute["date"]).ffill()
    minute["bars_since_high"] = minute["bar_index"] - last_high_bar
    minute["bars_since_low"] = minute["bar_index"] - last_low_bar

    for window in (5, 15, 30):
        old_close = grouped["close"].shift(window)
        minute[f"return_{window}m"] = np.log(minute["close"] / old_close) * 100
        rolling_var = (
            minute.groupby("date", sort=False)["squared_return"]
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        minute[f"rv_{window}m"] = np.sqrt(rolling_var.clip(lower=0)) * 100
        minute[f"range_expansion_{window}m"] = (
            minute["current_amp"]
            - minute.groupby("date", sort=False)["current_amp"].shift(window)
        )

    snapshot = minute.loc[
        minute["bar_index"].mod(5).eq(0) & minute["bar_index"].le(235)
    ].copy()
    historical_cum_money = snapshot.groupby("bar_index", sort=False)["cum_money"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).median()
    )
    historical_cum_volume = snapshot.groupby("bar_index", sort=False)["cum_volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).median()
    )
    snapshot["cum_money_ratio20"] = snapshot["cum_money"] / historical_cum_money
    snapshot["cum_volume_ratio20"] = snapshot["cum_volume"] / historical_cum_volume

    current_map = {
        "current_amp": "amp",
        "current_return": "return",
        "current_rv": "rv",
        "current_down_share": "down_share",
        "current_efficiency": "efficiency",
        "current_clv": "clv",
        "bars_since_high": "bars_since_high",
        "bars_since_low": "bars_since_low",
        "cum_money_ratio20": "cum_money_ratio20",
        "cum_volume_ratio20": "cum_volume_ratio20",
    }
    for window in (5, 15, 30):
        current_map[f"return_{window}m"] = f"return_{window}m"
        current_map[f"rv_{window}m"] = f"rv_{window}m"
        current_map[f"range_expansion_{window}m"] = f"range_expansion_{window}m"

    keep = ["date", "bar_index"] + list(current_map)
    if is_target:
        keep += ["final_amp"]
    snapshot = snapshot[keep].rename(
        columns={source: f"{prefix}cur_{dest}" for source, dest in current_map.items()}
    )

    preopen = pd.DataFrame({"date": day.index, "bar_index": 0})
    for output_name in current_map.values():
        column = f"{prefix}cur_{output_name}"
        preopen[column] = np.nan if "ratio20" in output_name else 0.0
    if is_target:
        preopen["final_amp"] = preopen["date"].map(day["amp"])

    snapshot = pd.concat([preopen, snapshot], ignore_index=True)
    history = _historical_features(day, prefix=prefix, full=is_target)
    snapshot = snapshot.merge(
        history,
        left_on="date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return snapshot.sort_values(["date", "bar_index"]).reset_index(drop=True)


def build_feature_panel(
    data_dir: str | Path,
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    external_datasets: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    cache_path = Path(cache_path) if cache_path else None
    metadata_path = cache_path.with_suffix(".metadata.json") if cache_path else None
    requested_external = tuple(external_datasets or ())
    unknown = set(requested_external) - (set(DATASETS) - {"csi1000"})
    if unknown:
        raise ValueError(f"Unknown external datasets: {sorted(unknown)}")
    active_datasets = ("csi1000",) + requested_external

    audit, anomalies = audit_inputs(data_dir)
    if cache_path and cache_path.exists() and metadata_path and metadata_path.exists() and not rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("active_datasets", ())) == active_datasets:
            panel = pd.read_pickle(cache_path)
            groups = {key: list(value) for key, value in metadata["feature_groups"].items()}
            return panel, groups, audit, anomalies

    panels: dict[str, pd.DataFrame] = {}
    for name, spec in DATASETS.items():
        if name not in active_datasets:
            continue
        daily_path = data_dir / spec["daily"]
        minute_path = data_dir / spec["minute"]
        if not daily_path.exists() or not minute_path.exists():
            continue
        print(f"[features] building {name}", flush=True)
        daily = _read_daily(daily_path)
        minute, _ = _read_minute(minute_path)
        minute = _minute_derived(minute)
        day = _daily_signals(daily, minute)
        panels[name] = _snapshot_features(
            day=day,
            minute=minute,
            prefix=f"{name}_",
            is_target=name == "csi1000",
        )

    if "csi1000" not in panels:
        raise FileNotFoundError("CSI 1000 target files are required")

    panel = panels.pop("csi1000")
    for name, external in panels.items():
        panel = panel.merge(
            external,
            on=["date", "bar_index"],
            how="left",
            validate="one_to_one",
        )
        for signal in ("amp", "return", "rv", "down_share"):
            target_column = f"csi1000_cur_{signal}"
            external_column = f"{name}_cur_{signal}"
            if target_column in panel and external_column in panel:
                panel[f"relative_{name}_{signal}"] = (
                    panel[target_column] - panel[external_column]
                )

    panel["is_preopen"] = panel["bar_index"].eq(0).astype(np.int8)
    panel["elapsed_fraction"] = panel["bar_index"] / 240.0
    panel["remaining_fraction"] = 1.0 - panel["elapsed_fraction"]
    panel["is_afternoon"] = panel["bar_index"].gt(120).astype(np.int8)
    panel["minutes_since_open"] = panel["bar_index"]
    panel["minutes_to_close"] = 240 - panel["bar_index"]
    panel["current_amp"] = panel["csi1000_cur_amp"].fillna(0.0)
    panel["remaining_amp"] = (panel["final_amp"] - panel["current_amp"]).clip(lower=0.0)
    panel["target_log_remaining"] = np.log1p(panel["remaining_amp"])
    panel["year"] = panel["date"].dt.year
    panel["excluded_anomaly"] = panel["date"].isin(EXCLUDED_DYNAMIC_DATES)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    base_suffixes = {
        "cur_amp",
        "cur_return",
        "cur_rv",
        "cur_down_share",
        "cur_efficiency",
        "cur_clv",
        "cur_bars_since_high",
        "cur_bars_since_low",
        "cur_cum_money_ratio20",
        "cur_cum_volume_ratio20",
        "cur_return_5m",
        "cur_return_15m",
        "cur_return_30m",
        "cur_rv_5m",
        "cur_rv_15m",
        "cur_rv_30m",
        "cur_range_expansion_5m",
        "cur_range_expansion_15m",
        "cur_range_expansion_30m",
        "hist_amp_lag1",
        "hist_amp_lag2",
        "hist_amp_lag3",
        "hist_amp_lag5",
        "hist_amp_lag10",
        "hist_amp_lag20",
        "hist_amp_mean5",
        "hist_amp_mean20",
        "hist_amp_mean60",
        "hist_amp_mean120",
        "hist_amp_median20",
        "hist_amp_median60",
        "hist_amp_std20",
        "hist_amp_std60",
        "hist_amp_q90_60",
        "hist_amp_ratio5_60",
        "hist_ret_lag1",
        "hist_ret_lag2",
        "hist_ret_lag5",
        "hist_abs_ret_lag1",
        "hist_abs_ret_lag5",
        "hist_abs_ret_mean5",
        "hist_abs_ret_mean20",
        "hist_gap_lag1",
        "hist_gap_abs_lag1",
        "hist_clv_lag1",
        "hist_body_lag1",
        "hist_rv_lag1",
        "hist_rv_lag2",
        "hist_rv_lag5",
        "hist_rv_mean5",
        "hist_rv_mean20",
        "hist_rv_mean60",
        "hist_rv_std20",
        "hist_rv_ratio5_20",
        "hist_rv_down_lag1",
        "hist_down_variance_share_lag1",
        "hist_jump_lag1",
        "hist_max_abs_1m_lag1",
        "hist_rv_first30_lag1",
        "hist_rv_last30_lag1",
        "hist_money_rel20",
        "hist_volume_rel20",
        "hist_trend5",
        "hist_trend20",
    }
    feature_groups: dict[str, list[str]] = {
        "base": [
            f"csi1000_{suffix}"
            for suffix in sorted(base_suffixes)
            if f"csi1000_{suffix}" in panel.columns
        ]
        + ["bar_index", "is_preopen"]
    }
    external_suffixes = {
        "cur_amp",
        "cur_return",
        "cur_rv",
        "cur_down_share",
        "cur_cum_money_ratio20",
        "cur_return_15m",
        "cur_rv_15m",
        "cur_range_expansion_15m",
        "hist_amp_lag1",
        "hist_amp_mean20",
        "hist_amp_std20",
        "hist_rv_lag1",
        "hist_rv_mean20",
        "hist_ret_lag1",
        "hist_money_rel20",
        "hist_trend5",
    }
    for name in panels:
        group_columns = [
            f"{name}_{suffix}"
            for suffix in sorted(external_suffixes)
            if f"{name}_{suffix}" in panel.columns
        ]
        group_columns.extend(
            column
            for column in (
                f"relative_{name}_amp",
                f"relative_{name}_return",
                f"relative_{name}_rv",
            )
            if column in panel.columns
        )
        feature_groups[name] = group_columns

    protected = {
        "date",
        "final_amp",
        "current_amp",
        "remaining_amp",
        "target_log_remaining",
        "year",
        "excluded_anomaly",
    }
    for group, columns in feature_groups.items():
        feature_groups[group] = sorted(set(columns) - protected)

    panel = panel.sort_values(["date", "bar_index"]).reset_index(drop=True)
    if cache_path and metadata_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cache_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(),
                    "active_datasets": active_datasets,
                    "feature_groups": feature_groups,
                    "rows": len(panel),
                    "columns": len(panel.columns),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return panel, feature_groups, audit, anomalies


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    return float(mean_pinball_loss(y_true, y_pred, alpha=alpha))


def _fit_with_early_stopping(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
    config: ModelConfig,
) -> tuple[int, lgb.LGBMRegressor]:
    unique_dates = np.sort(train["date"].unique())
    if len(unique_dates) <= config.early_stop_days + 60:
        raise ValueError("Not enough dates for the configured early-stopping window")
    validation_dates = set(unique_dates[-config.early_stop_days :])
    fit_mask = ~train["date"].isin(validation_dates)
    valid_mask = ~fit_mask
    model = lgb.LGBMRegressor(**config.lgb_params(alpha))
    model.fit(
        train.loc[fit_mask, features].astype(np.float32),
        train.loc[fit_mask, "target_log_remaining"].astype(np.float32),
        eval_set=[
            (
                train.loc[valid_mask, features].astype(np.float32),
                train.loc[valid_mask, "target_log_remaining"].astype(np.float32),
            )
        ],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(
                config.early_stopping_rounds,
                first_metric_only=True,
                min_delta=1e-6,
                verbose=False,
            ),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(model.best_iteration_ or config.max_estimators)
    return best_iteration, model


def _refit(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
    iterations: int,
    config: ModelConfig,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**config.lgb_params(alpha, n_estimators=iterations))
    model.fit(
        train[features].astype(np.float32),
        train["target_log_remaining"].astype(np.float32),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def walkforward_quantiles(
    panel: pd.DataFrame,
    features: list[str],
    alphas: Iterable[float],
    config: ModelConfig,
    years: Iterable[int] = OUTER_YEARS,
    label: str = "model",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, list[int]]]:
    usable = panel.loc[~panel["excluded_anomaly"]].copy()
    alphas = tuple(float(alpha) for alpha in alphas)
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    iterations: dict[float, list[int]] = {alpha: [] for alpha in alphas}

    for year in years:
        train = usable.loc[usable["year"].lt(year)]
        test = usable.loc[usable["year"].eq(year)]
        if train.empty or test.empty:
            continue
        fold = test[["date", "bar_index", "final_amp", "current_amp", "target_log_remaining"]].copy()
        fold["fold_year"] = year
        print(
            f"[walkforward] {label} year={year} train_days={train['date'].nunique()} "
            f"test_days={test['date'].nunique()} features={len(features)}",
            flush=True,
        )
        for alpha in alphas:
            best_iteration, _ = _fit_with_early_stopping(train, features, alpha, config)
            model = _refit(train, features, alpha, best_iteration, config)
            column = f"q{int(round(alpha * 100)):02d}_log"
            fold[column] = model.predict(test[features].astype(np.float32))
            iterations[alpha].append(best_iteration)
            metric_rows.append(
                {
                    "label": label,
                    "year": year,
                    "alpha": alpha,
                    "pinball": _pinball(
                        test["target_log_remaining"].to_numpy(),
                        fold[column].to_numpy(),
                        alpha,
                    ),
                    "best_iteration": best_iteration,
                    "features": len(features),
                    "train_days": train["date"].nunique(),
                    "test_days": test["date"].nunique(),
                }
            )
        all_predictions.append(fold)

    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(metric_rows), iterations


def _score_metrics(metrics: pd.DataFrame) -> tuple[float, pd.Series]:
    by_year = metrics.groupby("year")["pinball"].mean()
    return float(by_year.mean()), by_year


def split_base_feature_families(base_features: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {
        "dynamic_core": [],
        "historical_amplitude": [],
        "historical_realized_volatility": [],
        "historical_price_shape": [],
        "liquidity": [],
    }
    for feature in base_features:
        if feature in {"bar_index", "is_preopen"}:
            families["dynamic_core"].append(feature)
        elif "cum_money_ratio" in feature or "cum_volume_ratio" in feature:
            families["liquidity"].append(feature)
        elif "hist_money_" in feature or "hist_volume_" in feature:
            families["liquidity"].append(feature)
        elif feature.startswith("csi1000_cur_"):
            families["dynamic_core"].append(feature)
        elif feature.startswith("csi1000_hist_amp_"):
            families["historical_amplitude"].append(feature)
        elif any(
            token in feature
            for token in (
                "hist_rv_",
                "hist_jump_",
                "hist_max_abs_1m_",
                "hist_down_variance_",
            )
        ):
            families["historical_realized_volatility"].append(feature)
        else:
            families["historical_price_shape"].append(feature)
    return {key: sorted(value) for key, value in families.items() if value}


def forward_select_base_families(
    panel: pd.DataFrame,
    base_features: list[str],
    config: ModelConfig,
    report_dir: Path,
) -> tuple[list[str], list[str], pd.DataFrame]:
    families = split_base_feature_families(base_features)
    selected_families = ["dynamic_core"]
    current_features = list(families["dynamic_core"])
    remaining = [name for name in families if name != "dynamic_core"]
    cache: dict[tuple[str, ...], tuple[float, pd.Series, pd.DataFrame]] = {}
    screening_rows: list[dict[str, Any]] = []

    def evaluate(selected: list[str]) -> tuple[float, pd.Series, pd.DataFrame]:
        key = tuple(sorted(selected))
        if key in cache:
            return cache[key]
        features = sorted({item for family in selected for item in families[family]})
        label = "base_families:" + "+".join(selected)
        _, metrics, _ = walkforward_quantiles(
            panel,
            features,
            ALPHAS_SCREEN,
            config,
            label=label,
        )
        score, by_year = _score_metrics(metrics)
        cache[key] = (score, by_year, metrics)
        screening_rows.extend(dict(row) for _, row in metrics.iterrows())
        pd.DataFrame(screening_rows).to_csv(
            report_dir / "base_factor_family_screening_folds.csv", index=False
        )
        return cache[key]

    current_score, current_by_year, _ = evaluate(selected_families)
    print(
        f"[base-selection] starting={selected_families} score={current_score:.8f}",
        flush=True,
    )
    while remaining:
        candidates: list[tuple[float, str, pd.Series]] = []
        for family in remaining:
            score, by_year, _ = evaluate(selected_families + [family])
            candidates.append((score, family, by_year))
            print(
                f"[base-selection] candidate={family} score={score:.8f}",
                flush=True,
            )
        best_score, best_family, best_by_year = min(candidates, key=lambda item: item[0])
        improvement = (current_score - best_score) / current_score
        comparable = current_by_year.index.intersection(best_by_year.index)
        years_improved = int(
            (best_by_year.loc[comparable] < current_by_year.loc[comparable]).sum()
        )
        accepted = bool(
            improvement >= config.min_group_improvement
            and years_improved >= config.min_years_improved
        )
        decision = {
            "step": len(selected_families),
            "candidate": best_family,
            "score_before": current_score,
            "score_after": best_score,
            "relative_improvement": improvement,
            "years_improved": years_improved,
            "accepted": accepted,
        }
        with (report_dir / "base_factor_family_decisions.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
        if not accepted:
            print(
                f"[base-selection] stop: best={best_family} improvement={improvement:.2%} "
                f"years_improved={years_improved}",
                flush=True,
            )
            break
        selected_families.append(best_family)
        remaining.remove(best_family)
        current_score = best_score
        current_by_year = best_by_year
        current_features = sorted(
            {item for family in selected_families for item in families[family]}
        )
        print(
            f"[base-selection] accepted={best_family} improvement={improvement:.2%}",
            flush=True,
        )
    return selected_families, current_features, pd.DataFrame(screening_rows)


def forward_select_feature_groups(
    panel: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: ModelConfig,
    report_dir: Path,
    base_features: list[str] | None = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    selected_groups: list[str] = []
    base_features = list(base_features or feature_groups["base"])
    current_features = list(base_features)
    cache: dict[tuple[str, ...], tuple[float, pd.Series, pd.DataFrame]] = {}
    screening_rows: list[dict[str, Any]] = []

    def evaluate(groups: list[str]) -> tuple[float, pd.Series, pd.DataFrame]:
        key = tuple(sorted(groups))
        if key in cache:
            return cache[key]
        features = list(base_features)
        for group in groups:
            features.extend(feature_groups[group])
        features = sorted(set(features))
        label = "base" if not groups else "base+" + "+".join(groups)
        _, metrics, _ = walkforward_quantiles(
            panel,
            features,
            ALPHAS_SCREEN,
            config,
            label=label,
        )
        score, by_year = _score_metrics(metrics)
        cache[key] = (score, by_year, metrics)
        for _, row in metrics.iterrows():
            screening_rows.append(dict(row))
        pd.DataFrame(screening_rows).to_csv(
            report_dir / "feature_group_screening_folds.csv", index=False
        )
        return cache[key]

    current_score, current_by_year, _ = evaluate([])
    remaining = [name for name in feature_groups if name != "base"]
    print(f"[selection] base score={current_score:.8f}", flush=True)

    while remaining:
        candidates: list[tuple[float, str, pd.Series]] = []
        for group in remaining:
            score, by_year, _ = evaluate(selected_groups + [group])
            candidates.append((score, group, by_year))
            print(
                f"[selection] candidate={group} groups={selected_groups + [group]} "
                f"score={score:.8f}",
                flush=True,
            )
        best_score, best_group, best_by_year = min(candidates, key=lambda item: item[0])
        relative_improvement = (current_score - best_score) / current_score
        comparable = current_by_year.index.intersection(best_by_year.index)
        years_improved = int(
            (best_by_year.loc[comparable] < current_by_year.loc[comparable]).sum()
        )
        decision = {
            "step": len(selected_groups) + 1,
            "candidate": best_group,
            "score_before": current_score,
            "score_after": best_score,
            "relative_improvement": relative_improvement,
            "years_improved": years_improved,
            "accepted": bool(
                relative_improvement >= config.min_group_improvement
                and years_improved >= config.min_years_improved
            ),
        }
        with (report_dir / "feature_group_decisions.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")

        if not decision["accepted"]:
            print(
                f"[selection] stop: best={best_group} improvement={relative_improvement:.2%} "
                f"years_improved={years_improved}",
                flush=True,
            )
            break
        selected_groups.append(best_group)
        remaining.remove(best_group)
        current_score = best_score
        current_by_year = best_by_year
        current_features = sorted(
            set(base_features)
            | {feature for group in selected_groups for feature in feature_groups[group]}
        )
        print(
            f"[selection] accepted={best_group} improvement={relative_improvement:.2%}",
            flush=True,
        )

    screening = pd.DataFrame(screening_rows)
    return selected_groups, current_features, screening


def staged_external_dataset_selection(
    data_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
    early_features: list[str],
    late_features: list[str],
    config: ModelConfig,
    cutoff_bar: int,
    rebuild_features: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    """Build and test one external dataset at a time.

    Only datasets accepted by the walk-forward gate are carried into the next
    construction stage. Rejected candidate panels remain diagnostic artifacts
    and are never passed to the final model fit.
    """
    base_panel, _, _, _ = build_feature_panel(
        data_dir,
        cache_path=artifact_dir / "dynamic_feature_panel_base.pkl",
        rebuild=rebuild_features,
        external_datasets=(),
    )
    expert_state: dict[str, dict[str, Any]] = {
        "early": {
            "features": list(early_features),
            "panel": base_panel.loc[base_panel["bar_index"].le(cutoff_bar)],
        },
        "late": {
            "features": list(late_features),
            "panel": base_panel.loc[base_panel["bar_index"].gt(cutoff_bar)],
        },
    }
    for expert_name, state in expert_state.items():
        _, metrics, _ = walkforward_quantiles(
            state["panel"],
            state["features"],
            ALPHAS_SCREEN,
            config,
            label=f"{expert_name}:base_only",
        )
        state["score"], state["by_year"] = _score_metrics(metrics)

    accepted: dict[str, list[str]] = {"early": [], "late": []}
    decision_rows: list[dict[str, Any]] = []
    candidate_order = ("csi500", "csi300", "chinext", "sse50")

    for candidate in candidate_order:
        active_external = sorted(
            set(accepted["early"]) | set(accepted["late"]) | {candidate}
        )
        stage_name = "_".join(active_external)
        candidate_panel, candidate_groups, _, _ = build_feature_panel(
            data_dir,
            cache_path=artifact_dir / f"candidate_panel_{stage_name}.pkl",
            rebuild=rebuild_features,
            external_datasets=active_external,
        )
        print(
            f"[dataset-stage] candidate={candidate} active_build={active_external}",
            flush=True,
        )
        for expert_name, mask in (
            ("early", candidate_panel["bar_index"].le(cutoff_bar)),
            ("late", candidate_panel["bar_index"].gt(cutoff_bar)),
        ):
            state = expert_state[expert_name]
            candidate_features = sorted(
                set(state["features"]) | set(candidate_groups[candidate])
            )
            _, metrics, _ = walkforward_quantiles(
                candidate_panel.loc[mask],
                candidate_features,
                ALPHAS_SCREEN,
                config,
                label=f"{expert_name}:add_{candidate}",
            )
            candidate_score, candidate_by_year = _score_metrics(metrics)
            improvement = (state["score"] - candidate_score) / state["score"]
            comparable = state["by_year"].index.intersection(candidate_by_year.index)
            years_improved = int(
                (
                    candidate_by_year.loc[comparable]
                    < state["by_year"].loc[comparable]
                ).sum()
            )
            is_accepted = bool(
                improvement >= config.min_group_improvement
                and years_improved >= config.min_years_improved
            )
            decision_rows.append(
                {
                    "stage": candidate,
                    "expert": expert_name,
                    "active_datasets_built": ",".join(active_external),
                    "score_before": state["score"],
                    "score_after": candidate_score,
                    "relative_improvement": improvement,
                    "years_improved": years_improved,
                    "accepted": is_accepted,
                }
            )
            if is_accepted:
                accepted[expert_name].append(candidate)
                state["features"] = candidate_features
                state["score"] = candidate_score
                state["by_year"] = candidate_by_year
            print(
                f"[dataset-stage] expert={expert_name} candidate={candidate} "
                f"improvement={improvement:.2%} years={years_improved} "
                f"accepted={is_accepted}",
                flush=True,
            )
        pd.DataFrame(decision_rows).to_csv(
            report_dir / "staged_external_dataset_decisions.csv", index=False
        )

    retained_union = sorted(set(accepted["early"]) | set(accepted["late"]))
    final_panel, _, _, _ = build_feature_panel(
        data_dir,
        cache_path=artifact_dir / "dynamic_feature_panel.pkl",
        rebuild=True,
        external_datasets=retained_union,
    )
    retained_path = artifact_dir / "retained_external_datasets.json"
    retained_path.write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        final_panel,
        list(expert_state["early"]["features"]),
        list(expert_state["late"]["features"]),
        accepted,
    )


def _calibration_offsets(
    oof: pd.DataFrame,
    alphas: Iterable[float],
) -> dict[str, dict[str, float]]:
    offsets: dict[str, dict[str, float]] = {}
    for bar_index, group in oof.groupby("bar_index"):
        checkpoint: dict[str, float] = {}
        for alpha in alphas:
            column = f"q{int(round(alpha * 100)):02d}_log"
            residual = group["target_log_remaining"] - group[column]
            checkpoint[str(alpha)] = float(residual.quantile(alpha))
        offsets[str(int(bar_index))] = checkpoint
    return offsets


def _apply_quantile_predictions(
    frame: pd.DataFrame,
    alphas: Iterable[float],
    offsets: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    alphas = tuple(float(alpha) for alpha in alphas)
    raw_columns = [f"q{int(round(alpha * 100)):02d}_log" for alpha in alphas]
    adjusted = result[raw_columns].to_numpy(dtype=float)
    if offsets:
        for row_index, bar_index in enumerate(result["bar_index"].astype(int)):
            checkpoint = offsets.get(str(bar_index), {})
            for alpha_index, alpha in enumerate(alphas):
                adjusted[row_index, alpha_index] += checkpoint.get(str(alpha), 0.0)
    adjusted = np.sort(adjusted, axis=1)
    remaining_quantiles = np.maximum(np.expm1(adjusted), 0.0)
    final_quantiles = result["current_amp"].to_numpy()[:, None] + remaining_quantiles
    for alpha_index, alpha in enumerate(alphas):
        suffix = f"q{int(round(alpha * 100)):02d}"
        result[f"{suffix}_log"] = adjusted[:, alpha_index]
        result[f"{suffix}_remaining"] = remaining_quantiles[:, alpha_index]
        result[f"{suffix}_final_amp"] = final_quantiles[:, alpha_index]
    return result


def _empirical_baseline(
    panel: pd.DataFrame,
    evaluation_years: Iterable[int],
    alphas: Iterable[float],
    history_days: int = 252,
) -> pd.DataFrame:
    usable = panel.loc[~panel["excluded_anomaly"]].copy()
    result = usable[
        ["date", "bar_index", "year", "target_log_remaining"]
    ].copy()
    for alpha in alphas:
        result[f"q{int(round(alpha * 100)):02d}_log"] = usable.groupby(
            "bar_index", sort=False
        )["target_log_remaining"].transform(
            lambda values: values.shift(1)
            .rolling(history_days, min_periods=min(60, history_days // 2))
            .quantile(alpha)
        )
    return result.loc[result["year"].isin(tuple(evaluation_years))].drop(
        columns="year"
    ).reset_index(drop=True)


def _summarize_predictions(
    predictions: pd.DataFrame,
    alphas: Iterable[float],
    split: str,
) -> dict[str, float | int | str]:
    alphas = tuple(float(alpha) for alpha in alphas)
    q50 = "q50_final_amp"
    q10 = f"q{int(round(min(alphas) * 100)):02d}_final_amp"
    q90 = f"q{int(round(max(alphas) * 100)):02d}_final_amp"
    pinballs = []
    for alpha in alphas:
        column = f"q{int(round(alpha * 100)):02d}_log"
        pinballs.append(
            _pinball(
                predictions["target_log_remaining"].to_numpy(),
                predictions[column].to_numpy(),
                alpha,
            )
        )
    return {
        "split": split,
        "rows": len(predictions),
        "days": predictions["date"].nunique(),
        "mean_pinball_log": float(np.mean(pinballs)),
        "approx_crps_log": float(2 * np.mean(pinballs)),
        "p50_mae_amp_pct": float(
            mean_absolute_error(predictions["final_amp"], predictions[q50])
        ),
        "p10_p90_coverage": float(
            (
                predictions["final_amp"].ge(predictions[q10])
                & predictions["final_amp"].le(predictions[q90])
            ).mean()
        ),
        "p10_p90_mean_width": float((predictions[q90] - predictions[q10]).mean()),
    }


def _feature_importance(
    models: dict[str, lgb.LGBMRegressor],
    features: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for alpha, model in models.items():
        rows.append(
            pd.DataFrame(
                {
                    "feature": features,
                    "gain": model.booster_.feature_importance(importance_type="gain"),
                    "split": model.booster_.feature_importance(importance_type="split"),
                    "alpha": float(alpha),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    total_gain = result.groupby("feature")["gain"].transform("sum").replace(0, np.nan)
    result["gain_share_within_all_heads"] = result["gain"] / total_gain
    return result.sort_values(["alpha", "gain"], ascending=[True, False])


def _write_audit_report(
    path: Path,
    audit: pd.DataFrame,
    anomalies: list[dict[str, Any]],
    panel: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    selected_groups: list[str],
    selected_features: list[str],
    screening: pd.DataFrame,
    summaries: list[dict[str, Any]],
    iterations: dict[float, list[int]],
) -> None:
    screen_summary = (
        screening.groupby(["label", "year"])["pinball"].mean().reset_index()
        if not screening.empty
        else pd.DataFrame()
    )
    lines = [
        "# 中证1000动态振幅概率模型：数据与模型审计报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 模型选择数据：2019—2025",
        "- 最终封存测试：2026",
        "- 动态频率：盘前一次；09:35至14:55每5分钟一次",
        "- 目标：最终振幅减去当前已实现振幅的剩余扩张量（log1p变换）",
        "- 核心模型：共享时点的多分位数LightGBM",
        "",
        "## 数据审计",
        "",
        audit.to_markdown(index=False),
        "",
        "### 已识别异常及处理",
        "",
        "```json",
        json.dumps(anomalies, ensure_ascii=False, indent=2),
        "```",
        "",
        f"- 动态样本行数：{len(panel):,}",
        f"- 交易日数：{panel['date'].nunique():,}",
        f"- 每日预测节点数：{panel.groupby('date').size().median():.0f}",
        f"- 从训练和评估剔除的日期：{', '.join(str(x.date()) for x in EXCLUDED_DYNAMIC_DATES)}",
        "",
        "## 因子族",
        "",
    ]
    for group, columns in feature_groups.items():
        lines.append(f"- `{group}`：{len(columns)}个候选因子")
    lines.extend(
        [
            "",
            f"- 最终保留外部因子族：{', '.join(selected_groups) if selected_groups else '无'}",
            f"- 最终模型因子数：{len(selected_features)}",
            "",
            "## 因子族走步筛选",
            "",
            screen_summary.to_markdown(index=False) if not screen_summary.empty else "无",
            "",
            "## 最佳迭代数",
            "",
            "```json",
            json.dumps({str(key): value for key, value in iterations.items()}, indent=2),
            "```",
            "",
            "## 预测表现",
            "",
            pd.DataFrame(summaries).to_markdown(index=False),
            "",
            "## 防泄漏约束",
            "",
            "- 所有训练、早停和外层验证均按完整交易日切分。",
            "- 当日分钟特征只使用当前预测时点及以前的数据。",
            "- 历史日频与分钟聚合因子全部至少滞后一个交易日。",
            "- 2026数据不参与因子族筛选、超参选择、树数选择或概率校准。",
            "- 分位数预测在输出前执行单调重排，并按预测时点使用2019—2025走步残差校准。",
            "",
            "## 解释限制",
            "",
            "- 同一天的多个5分钟快照高度相关，不能视为独立交易日。",
            "- 2026当前只覆盖至数据文件的最后日期，属于阶段性封存测试。",
            "- 外部指数高度相关，只有达到预设走步改进门槛的因子族才进入最终模型。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _checkpoint_metrics(
    predictions: pd.DataFrame,
    alphas: Iterable[float],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bar_index, group in predictions.groupby("bar_index"):
        summary = _summarize_predictions(group, alphas, split)
        summary["bar_index"] = int(bar_index)
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("bar_index")


def _fit_final_expert(
    panel: pd.DataFrame,
    features: list[str],
    alphas: Iterable[float],
    iterations: dict[float, list[int]],
    config: ModelConfig,
) -> tuple[dict[str, lgb.LGBMRegressor], dict[float, int]]:
    train = panel.loc[panel["year"].le(2025) & ~panel["excluded_anomaly"]].copy()
    models: dict[str, lgb.LGBMRegressor] = {}
    chosen: dict[float, int] = {}
    for alpha in alphas:
        alpha = float(alpha)
        candidates = iterations.get(alpha, [])
        if not candidates:
            best_iteration, _ = _fit_with_early_stopping(
                train, features, alpha, config
            )
            candidates = [best_iteration]
        chosen_iteration = int(np.median(candidates))
        chosen[alpha] = chosen_iteration
        models[str(alpha)] = _refit(
            train, features, alpha, chosen_iteration, config
        )
    return models, chosen


def _predict_expert(
    panel: pd.DataFrame,
    features: list[str],
    models: dict[str, lgb.LGBMRegressor],
    alphas: Iterable[float],
) -> pd.DataFrame:
    output = panel[
        ["date", "bar_index", "final_amp", "current_amp", "target_log_remaining"]
    ].copy()
    for alpha in alphas:
        alpha = float(alpha)
        output[f"q{int(round(alpha * 100)):02d}_log"] = models[str(alpha)].predict(
            panel[features].astype(np.float32)
        )
    return output


def predict_feature_rows(
    bundle: dict[str, Any],
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Predict calibrated amplitude quantiles from an already-built feature panel."""
    alphas = tuple(float(value) for value in bundle["alphas"])
    cutoff_bar = int(bundle["cutoff_bar"])
    pieces: list[pd.DataFrame] = []
    for expert_name, mask in (
        ("early", feature_rows["bar_index"].le(cutoff_bar)),
        ("late", feature_rows["bar_index"].gt(cutoff_bar)),
    ):
        subset = feature_rows.loc[mask]
        if subset.empty:
            continue
        expert = bundle["experts"][expert_name]
        pieces.append(
            _predict_expert(
                subset,
                list(expert["features"]),
                expert["models"],
                alphas,
            )
        )
    raw = pd.concat(pieces, ignore_index=True).sort_values(["date", "bar_index"])
    return _apply_quantile_predictions(
        raw,
        alphas,
        offsets=bundle.get("calibration_offsets"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_routed_audit_report(
    path: Path,
    audit: pd.DataFrame,
    anomalies: list[dict[str, Any]],
    panel: pd.DataFrame,
    early_features: list[str],
    late_features: list[str],
    early_iterations: dict[float, list[int]],
    late_iterations: dict[float, list[int]],
    summaries: list[dict[str, Any]],
    early_base_decisions: list[dict[str, Any]],
    early_external_decisions: list[dict[str, Any]],
    late_external_decisions: list[dict[str, Any]],
) -> None:
    lines = [
        "# 中证1000动态振幅概率模型：数据与模型审计报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 任务：AI测算中证当天振幅范围的动态概率模型",
        "- 模型选择与概率校准数据：2019—2025",
        "- 完全封存测试：2026",
        "- 预测频率：09:25盘前一次；09:35至14:55每5分钟一次",
        "- 目标：`log1p(最终振幅 - 当前已实现振幅)`",
        "- 模型包：时间路由的多分位数LightGBM（一个可交付模型文件）",
        "- 路由：09:25—10:30使用早盘专家；10:35—14:55使用盘中专家",
        "",
        "## 一、数据审计",
        "",
        audit.to_markdown(index=False),
        "",
        "### 异常及处理",
        "",
        "```json",
        json.dumps(anomalies, ensure_ascii=False, indent=2),
        "```",
        "",
        f"- 完整交易日：{panel['date'].nunique():,}",
        f"- 动态样本行数：{len(panel):,}",
        f"- 每日预测节点：{int(panel.groupby('date').size().median())}",
        f"- 剔除动态建模日期：{', '.join(str(x.date()) for x in EXCLUDED_DYNAMIC_DATES)}",
        "",
        "## 二、因子数量控制",
        "",
        f"- 早盘专家：{len(early_features)}个因子",
        f"- 盘中专家：{len(late_features)}个因子",
        "- 早盘保留：当日动态核心 + 历史价格形态。",
        "- 盘中保留：当日动态核心。",
        "- 中证500、沪深300、创业板指、上证50均逐组测试；没有达到预设的2%平均改进门槛，因此未进入最终模型。",
        "",
        "### 早盘基础因子族决策",
        "",
        "```json",
        json.dumps(early_base_decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "### 早盘外部指数决策",
        "",
        "```json",
        json.dumps(early_external_decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "### 盘中外部指数决策",
        "",
        "```json",
        json.dumps(late_external_decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 三、早停和树数",
        "",
        "- 每个外层折在训练期末保留120个交易日用于早停。",
        "- 最大树数1500；连续120棵树无有效改善则停止。",
        "- 外层验证使用完整年度，早停集不与外层年度重叠。",
        "- 最终树数取多个走步折最佳迭代数的中位数，再用完整2019—2025重训。",
        "",
        "### 早盘专家各折最佳迭代数",
        "",
        "```json",
        json.dumps({str(k): v for k, v in early_iterations.items()}, indent=2),
        "```",
        "",
        "### 盘中专家各折最佳迭代数",
        "",
        "```json",
        json.dumps({str(k): v for k, v in late_iterations.items()}, indent=2),
        "```",
        "",
        "## 四、模型表现",
        "",
        pd.DataFrame(summaries).to_markdown(index=False),
        "",
        "说明：2019—2025结果全部来自按年走步的样本外预测；2026仅在模型结构、因子、树数和校准方法冻结后运行。",
        "",
        "## 五、防泄漏检查",
        "",
        "- 同一天全部5分钟快照始终属于同一数据划分。",
        "- 日频和整日分钟聚合因子至少滞后一个交易日。",
        "- 当日动态因子仅累计至当前预测节点。",
        "- 同期成交额基准使用相同分钟节点的过去20个交易日中位数，并先滞后一天。",
        "- 2026不参与因子筛选、路由选择、早停、树数选择或概率校准。",
        "",
        "## 六、已知限制",
        "",
        "- 同日快照高度相关，独立样本量仍以交易日数计。",
        "- 当前封存测试只覆盖2026年数据文件已有日期。",
        "- 当前外部数据仅包含其他现货指数；尚无股指期货、市场宽度或隔夜行情。",
        "- 2020-04-20存在跨指数午间口径断层，因此整日剔除而非事后平滑。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_routed_training_pipeline(
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    report_dir: str | Path = "reports",
    rebuild_features: bool = False,
    reselect: bool = False,
    cutoff_bar: int = 60,
) -> dict[str, Any]:
    """Train the single deliverable model bundle with early/late expert routing."""
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig()

    base_panel, base_feature_groups, audit, anomalies = build_feature_panel(
        data_dir,
        cache_path=artifact_dir / "dynamic_feature_panel_base.pkl",
        rebuild=rebuild_features,
        external_datasets=(),
    )
    audit.to_csv(report_dir / "data_audit.csv", index=False)
    (report_dir / "data_anomalies.json").write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    early_feature_path = artifact_dir / "final_early_features.txt"
    late_feature_path = artifact_dir / "final_late_features.txt"
    if reselect or not early_feature_path.exists() or not late_feature_path.exists():
        early_panel = base_panel.loc[base_panel["bar_index"].le(cutoff_bar)].copy()
        late_panel = base_panel.loc[base_panel["bar_index"].gt(cutoff_bar)].copy()
        early_base_dir = report_dir / "early_selection"
        early_base_dir.mkdir(parents=True, exist_ok=True)

        _, early_base_features, _ = forward_select_base_families(
            early_panel, base_feature_groups["base"], config, early_base_dir
        )
        base_families = split_base_feature_families(base_feature_groups["base"])
        late_base_features = base_families["dynamic_core"]
        panel, early_features, late_features, retained_external = (
            staged_external_dataset_selection(
                data_dir,
                artifact_dir,
                report_dir,
                early_base_features,
                late_base_features,
                config,
                cutoff_bar,
                rebuild_features=rebuild_features,
            )
        )
        early_feature_path.write_text("\n".join(early_features), encoding="utf-8")
        late_feature_path.write_text("\n".join(late_features), encoding="utf-8")
    else:
        early_features = early_feature_path.read_text(encoding="utf-8").splitlines()
        late_features = late_feature_path.read_text(encoding="utf-8").splitlines()
        retained_path = artifact_dir / "retained_external_datasets.json"
        retained_external = (
            json.loads(retained_path.read_text(encoding="utf-8"))
            if retained_path.exists()
            else {"early": [], "late": []}
        )
        retained_union = sorted(
            set(retained_external.get("early", []))
            | set(retained_external.get("late", []))
        )
        panel, _, _, _ = build_feature_panel(
            data_dir,
            cache_path=artifact_dir / "dynamic_feature_panel.pkl",
            rebuild=rebuild_features,
            external_datasets=retained_union,
        )

    early_panel = panel.loc[panel["bar_index"].le(cutoff_bar)].copy()
    late_panel = panel.loc[panel["bar_index"].gt(cutoff_bar)].copy()
    print(
        f"[routed-final] early_features={len(early_features)} "
        f"late_features={len(late_features)} cutoff={cutoff_bar}",
        flush=True,
    )

    early_oof, early_metrics, early_iterations = walkforward_quantiles(
        early_panel,
        early_features,
        ALPHAS_FULL,
        config,
        label="early_expert",
    )
    late_oof, late_metrics, late_iterations = walkforward_quantiles(
        late_panel,
        late_features,
        ALPHAS_FULL,
        config,
        label="late_expert",
    )
    oof = pd.concat([early_oof, late_oof], ignore_index=True).sort_values(
        ["date", "bar_index"]
    )
    oof_metrics = pd.concat([early_metrics, late_metrics], ignore_index=True)
    oof_metrics.to_csv(report_dir / "final_walkforward_metrics.csv", index=False)

    offsets = _calibration_offsets(oof, ALPHAS_FULL)
    (artifact_dir / "quantile_calibration.json").write_text(
        json.dumps(offsets, indent=2), encoding="utf-8"
    )
    oof_raw = _apply_quantile_predictions(oof, ALPHAS_FULL, offsets=None)
    oof_raw.to_csv(report_dir / "walkforward_predictions_2022_2025.csv", index=False)

    early_models, early_final_iterations = _fit_final_expert(
        early_panel, early_features, ALPHAS_FULL, early_iterations, config
    )
    late_models, late_final_iterations = _fit_final_expert(
        late_panel, late_features, ALPHAS_FULL, late_iterations, config
    )

    test_2026 = panel.loc[
        panel["year"].eq(2026) & ~panel["excluded_anomaly"]
    ].copy()
    early_test = test_2026.loc[test_2026["bar_index"].le(cutoff_bar)]
    late_test = test_2026.loc[test_2026["bar_index"].gt(cutoff_bar)]
    prediction_raw = pd.concat(
        [
            _predict_expert(early_test, early_features, early_models, ALPHAS_FULL),
            _predict_expert(late_test, late_features, late_models, ALPHAS_FULL),
        ],
        ignore_index=True,
    ).sort_values(["date", "bar_index"])
    prediction_2026 = _apply_quantile_predictions(
        prediction_raw, ALPHAS_FULL, offsets=offsets
    )
    prediction_2026.to_csv(report_dir / "predictions_2026_dynamic.csv", index=False)

    empirical_raw = _empirical_baseline(panel, (2026,), ALPHAS_FULL)
    empirical = empirical_raw.merge(
        test_2026[["date", "bar_index", "final_amp", "current_amp"]],
        on=["date", "bar_index"],
        how="left",
        validate="one_to_one",
    )
    empirical = _apply_quantile_predictions(empirical, ALPHAS_FULL, offsets=None)
    empirical.to_csv(report_dir / "predictions_2026_empirical_baseline.csv", index=False)

    summaries = [
        _summarize_predictions(oof_raw, ALPHAS_FULL, "walkforward_2022_2025_raw"),
        _summarize_predictions(
            prediction_2026, ALPHAS_FULL, "sealed_2026_calibrated"
        ),
        _summarize_predictions(
            empirical, ALPHAS_FULL, "sealed_2026_empirical_baseline"
        ),
    ]
    pd.DataFrame(summaries).to_csv(report_dir / "model_summary.csv", index=False)
    checkpoint_metrics = pd.concat(
        [
            _checkpoint_metrics(oof_raw, ALPHAS_FULL, "walkforward_2022_2025_raw"),
            _checkpoint_metrics(
                prediction_2026, ALPHAS_FULL, "sealed_2026_calibrated"
            ),
            _checkpoint_metrics(
                empirical, ALPHAS_FULL, "sealed_2026_empirical_baseline"
            ),
        ],
        ignore_index=True,
    )
    checkpoint_metrics.to_csv(report_dir / "metrics_by_checkpoint.csv", index=False)

    importance_frames: list[pd.DataFrame] = []
    for expert, models, features in (
        ("early", early_models, early_features),
        ("late", late_models, late_features),
    ):
        importance = _feature_importance(models, features)
        importance["expert"] = expert
        importance_frames.append(importance)
    pd.concat(importance_frames, ignore_index=True).to_csv(
        report_dir / "feature_importance.csv", index=False
    )

    bundle = {
        "model_name": "CSI1000_dynamic_routed_multi_quantile_lightgbm",
        "created_at": datetime.now().isoformat(),
        "alphas": ALPHAS_FULL,
        "cutoff_bar": cutoff_bar,
        "experts": {
            "early": {
                "features": early_features,
                "models": early_models,
                "final_iterations": early_final_iterations,
            },
            "late": {
                "features": late_features,
                "models": late_models,
                "final_iterations": late_final_iterations,
            },
        },
        "calibration_offsets": offsets,
        "config": asdict(config),
        "excluded_dynamic_dates": [str(x.date()) for x in EXCLUDED_DYNAMIC_DATES],
        "training_end": "2025-12-31",
        "test_start": "2026-01-01",
        "external_groups_tested": ["csi500", "csi300", "chinext", "sse50"],
        "external_groups_retained": retained_external,
    }
    model_path = artifact_dir / "csi1000_dynamic_probability_model.joblib"
    joblib.dump(bundle, model_path, compress=3)

    _write_routed_audit_report(
        report_dir / "data_and_model_audit.md",
        audit,
        anomalies,
        panel,
        early_features,
        late_features,
        early_iterations,
        late_iterations,
        summaries,
        _read_jsonl(report_dir / "early_selection" / "base_factor_family_decisions.jsonl"),
        _read_jsonl(report_dir / "early_external_selection" / "feature_group_decisions.jsonl"),
        _read_jsonl(report_dir / "late_external_selection" / "feature_group_decisions.jsonl"),
    )

    result = {
        "model_path": str(model_path),
        "audit_report": str(report_dir / "data_and_model_audit.md"),
        "cutoff_bar": cutoff_bar,
        "early_feature_count": len(early_features),
        "late_feature_count": len(late_features),
        "external_groups_retained": retained_external,
        "early_final_iterations": early_final_iterations,
        "late_final_iterations": late_final_iterations,
        "summaries": summaries,
    }
    (report_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_training_pipeline(
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    report_dir: str | Path = "reports",
    rebuild_features: bool = False,
    skip_selection: bool = False,
) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig()
    panel, feature_groups, audit, anomalies = build_feature_panel(
        data_dir,
        cache_path=artifact_dir / "dynamic_feature_panel.pkl",
        rebuild=rebuild_features,
    )
    audit.to_csv(report_dir / "data_audit.csv", index=False)
    (report_dir / "data_anomalies.json").write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if skip_selection:
        selected_base_families = list(split_base_feature_families(feature_groups["base"]))
        base_screening = pd.DataFrame()
        selected_groups: list[str] = []
        selected_features = list(feature_groups["base"])
        screening = pd.DataFrame()
    else:
        base_decision_path = report_dir / "base_factor_family_decisions.jsonl"
        if base_decision_path.exists():
            base_decision_path.unlink()
        selected_base_families, selected_base_features, base_screening = (
            forward_select_base_families(
                panel, feature_groups["base"], config, report_dir
            )
        )
        decision_path = report_dir / "feature_group_decisions.jsonl"
        if decision_path.exists():
            decision_path.unlink()
        selected_groups, selected_features, screening = forward_select_feature_groups(
            panel,
            feature_groups,
            config,
            report_dir,
            base_features=selected_base_features,
        )

    print(
        f"[final] selected_groups={selected_groups} features={len(selected_features)}",
        flush=True,
    )
    oof, final_metrics, iterations = walkforward_quantiles(
        panel,
        selected_features,
        ALPHAS_FULL,
        config,
        label="final_selected",
    )
    final_metrics.to_csv(report_dir / "final_walkforward_metrics.csv", index=False)
    offsets = _calibration_offsets(oof, ALPHAS_FULL)
    (artifact_dir / "quantile_calibration.json").write_text(
        json.dumps(offsets, indent=2), encoding="utf-8"
    )
    oof_calibrated = _apply_quantile_predictions(oof, ALPHAS_FULL, offsets=None)
    oof_calibrated.to_csv(report_dir / "walkforward_predictions_2022_2025.csv", index=False)

    train = panel.loc[
        panel["year"].le(2025) & ~panel["excluded_anomaly"]
    ].copy()
    test_2026 = panel.loc[
        panel["year"].eq(2026) & ~panel["excluded_anomaly"]
    ].copy()
    final_models: dict[str, lgb.LGBMRegressor] = {}
    final_iterations: dict[float, int] = {}
    for alpha in ALPHAS_FULL:
        alpha_iterations = iterations.get(float(alpha), [])
        if not alpha_iterations:
            best_iteration, _ = _fit_with_early_stopping(
                train, selected_features, alpha, config
            )
            alpha_iterations = [best_iteration]
        chosen_iterations = int(np.median(alpha_iterations))
        final_iterations[float(alpha)] = chosen_iterations
        final_models[str(alpha)] = _refit(
            train, selected_features, alpha, chosen_iterations, config
        )

    prediction_base = test_2026[
        ["date", "bar_index", "final_amp", "current_amp", "target_log_remaining"]
    ].copy()
    for alpha in ALPHAS_FULL:
        column = f"q{int(round(alpha * 100)):02d}_log"
        prediction_base[column] = final_models[str(alpha)].predict(
            test_2026[selected_features].astype(np.float32)
        )
    calibrated_2026 = _apply_quantile_predictions(
        prediction_base, ALPHAS_FULL, offsets=offsets
    )
    calibrated_2026.to_csv(report_dir / "predictions_2026_dynamic.csv", index=False)

    empirical_raw = _empirical_baseline(panel, (2026,), ALPHAS_FULL)
    empirical = empirical_raw.merge(
        test_2026[["date", "bar_index", "final_amp", "current_amp"]],
        on=["date", "bar_index"],
        how="left",
        validate="one_to_one",
    )
    empirical = _apply_quantile_predictions(empirical, ALPHAS_FULL, offsets=None)
    empirical.to_csv(report_dir / "predictions_2026_empirical_baseline.csv", index=False)

    summaries = [
        _summarize_predictions(oof_calibrated, ALPHAS_FULL, "walkforward_2022_2025_raw"),
        _summarize_predictions(calibrated_2026, ALPHAS_FULL, "sealed_2026_calibrated"),
        _summarize_predictions(empirical, ALPHAS_FULL, "sealed_2026_empirical_baseline"),
    ]
    pd.DataFrame(summaries).to_csv(report_dir / "model_summary.csv", index=False)

    importance = _feature_importance(final_models, selected_features)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)

    bundle = {
        "model_name": "CSI1000_dynamic_multi_quantile_lightgbm",
        "created_at": datetime.now().isoformat(),
        "alphas": ALPHAS_FULL,
        "features": selected_features,
        "selected_external_groups": selected_groups,
        "models": final_models,
        "calibration_offsets": offsets,
        "final_iterations": final_iterations,
        "config": asdict(config),
        "excluded_dynamic_dates": [str(x.date()) for x in EXCLUDED_DYNAMIC_DATES],
        "training_end": "2025-12-31",
        "test_start": "2026-01-01",
    }
    joblib.dump(bundle, artifact_dir / "csi1000_dynamic_probability_model.joblib", compress=3)

    _write_audit_report(
        report_dir / "data_and_model_audit.md",
        audit,
        anomalies,
        panel,
        feature_groups,
        selected_groups,
        selected_features,
        screening,
        summaries,
        iterations,
    )
    result = {
        "selected_groups": selected_groups,
        "selected_base_families": selected_base_families,
        "selected_feature_count": len(selected_features),
        "summaries": summaries,
        "final_iterations": final_iterations,
        "model_path": str(artifact_dir / "csi1000_dynamic_probability_model.joblib"),
        "audit_report": str(report_dir / "data_and_model_audit.md"),
    }
    (report_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
