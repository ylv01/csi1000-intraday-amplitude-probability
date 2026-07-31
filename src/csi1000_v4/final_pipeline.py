from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .feature_pipeline import ModelConfig, _pinball, build_feature_panel


V4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = V4_ROOT / "config" / "final_config.json"
ROUTE_ORDER = ("preopen", "early", "late")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantile_suffix(alpha: float) -> str:
    return f"q{int(round(float(alpha) * 100)):02d}"


def route_mask(panel: pd.DataFrame, route: str) -> pd.Series:
    if route == "preopen":
        return panel["bar_index"].eq(0)
    if route == "early":
        return panel["bar_index"].between(5, 60)
    if route == "late":
        return panel["bar_index"].gt(60)
    raise KeyError(f"Unknown route: {route}")


def read_feature_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_feature_sets(
    feature_groups: dict[str, list[str]],
    config_dir: str | Path,
) -> dict[str, list[str]]:
    config_dir = Path(config_dir)
    early_base = read_feature_list(config_dir / "base_early_features.txt")
    late_base = read_feature_list(config_dir / "base_late_features.txt")
    external = sorted(
        set(feature_groups["sse50"]) | set(feature_groups["csi300"])
    )
    early_all = sorted(set(early_base) | set(external))
    late_all = sorted(set(late_base) | set(external))
    feature_sets = {
        "preopen": sorted(
            feature for feature in early_all if "_hist_" in feature
        ),
        "early": sorted(set(early_all) - {"is_preopen"}),
        "late": sorted(set(late_all) - {"is_preopen"}),
    }
    validate_feature_sets(feature_sets)
    return feature_sets


def validate_feature_sets(feature_sets: dict[str, list[str]]) -> None:
    if set(feature_sets) != set(ROUTE_ORDER):
        raise ValueError("Feature sets must contain preopen, early and late")
    preopen = feature_sets["preopen"]
    if not preopen:
        raise ValueError("Preopen feature set is empty")
    forbidden = [
        feature
        for feature in preopen
        if "_cur_" in feature
        or feature.startswith("relative_")
        or feature in {"bar_index", "is_preopen"}
    ]
    if forbidden:
        raise ValueError(
            f"Current-day information found in preopen features: {forbidden}"
        )
    if any("_hist_" not in feature for feature in preopen):
        raise ValueError("Every preopen feature must be historical and lagged")


def select_validation_windows(
    dates: np.ndarray,
    window_days: int,
    window_count: int,
) -> dict[str, np.ndarray]:
    dates = np.sort(np.asarray(dates))
    required = int(window_days) * int(window_count)
    if len(dates) <= required:
        raise ValueError("Not enough dates for multi-window tree selection")
    recent = dates[-required:]
    return {
        f"W{index + 1}": recent[
            index * window_days : (index + 1) * window_days
        ]
        for index in range(window_count)
    }


def materialize_model_config(
    base: ModelConfig,
    structure: dict[str, float | int],
    fitted_rows: int,
    config: dict[str, Any],
) -> ModelConfig:
    learning_rate = float(structure["learning_rate"])
    early_stop = config["early_stopping"]
    return replace(
        base,
        learning_rate=learning_rate,
        max_estimators=max(
            50,
            int(
                math.ceil(
                    float(early_stop["initial_max_boosting_time"])
                    / learning_rate
                )
            ),
        ),
        early_stopping_rounds=max(
            20,
            int(
                math.ceil(
                    float(early_stop["patience_boosting_time"])
                    / learning_rate
                )
            ),
        ),
        max_depth=int(structure["max_depth"]),
        num_leaves=int(structure["num_leaves"]),
        min_child_samples=max(
            2,
            int(
                math.ceil(
                    float(structure["min_child_ratio"]) * int(fitted_rows)
                )
            ),
        ),
        feature_fraction=float(structure["feature_fraction"]),
        lambda_l2=float(structure["lambda_l2"]),
    )


def simulate_early_stopping(
    losses: np.ndarray | list[float],
    patience: int,
    min_delta: float,
) -> dict[str, float | int | bool]:
    best_loss = math.inf
    best_iteration = 0
    last_improvement = 0
    stop_iteration = len(losses)
    stopped = False
    for iteration, value in enumerate(losses, start=1):
        loss = float(value)
        if loss < best_loss - float(min_delta):
            best_loss = loss
            best_iteration = iteration
            last_improvement = iteration
        if iteration - last_improvement >= int(patience):
            stop_iteration = iteration
            stopped = True
            break
    return {
        "iteration": int(best_iteration),
        "validation_pinball": float(best_loss),
        "stop_iteration": int(stop_iteration),
        "stopped_before_cap": bool(stopped),
    }


def _fit_validation_curve(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    alpha: float,
    model_config: ModelConfig,
) -> np.ndarray:
    model = lgb.LGBMRegressor(**model_config.lgb_params(alpha))
    model.fit(
        fit[features].astype(np.float32),
        fit["target_log_remaining"].astype(np.float32),
        eval_set=[
            (
                valid[features].astype(np.float32),
                valid["target_log_remaining"].astype(np.float32),
            )
        ],
        eval_metric="quantile",
        callbacks=[lgb.log_evaluation(0)],
    )
    return np.asarray(
        model.evals_result_["valid_0"]["quantile"],
        dtype=float,
    )


def _select_iteration_for_window(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    alpha: float,
    structure: dict[str, float | int],
    config: dict[str, Any],
    base: ModelConfig,
) -> tuple[dict[str, float | int | bool], ModelConfig]:
    model_config = materialize_model_config(
        base,
        structure,
        len(fit),
        config,
    )
    learning_rate = float(structure["learning_rate"])
    absolute_cap = int(
        math.ceil(
            float(config["early_stopping"]["absolute_max_boosting_time"])
            / learning_rate
        )
    )
    current_cap = int(model_config.max_estimators)
    while True:
        current = replace(model_config, max_estimators=current_cap)
        curve = _fit_validation_curve(
            fit,
            valid,
            features,
            alpha,
            current,
        )
        selected = simulate_early_stopping(
            curve,
            current.early_stopping_rounds,
            float(config["early_stopping"]["min_delta"]),
        )
        if bool(selected["stopped_before_cap"]) or current_cap >= absolute_cap:
            selected["curve_cap"] = len(curve)
            selected["absolute_cap_reached"] = bool(
                not selected["stopped_before_cap"]
            )
            return selected, current
        current_cap = min(current_cap * 2, absolute_cap)


def select_multiwindow_iterations(
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    early_stop = config["early_stopping"]
    dates = np.sort(train["date"].unique())
    windows = select_validation_windows(
        dates,
        int(early_stop["window_days"]),
        int(early_stop["window_count"]),
    )
    base = ModelConfig()
    iteration_map: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for route in ROUTE_ORDER:
        route_train = train.loc[route_mask(train, route)].copy()
        structure = config["route_structures"][route]
        features = feature_sets[route]
        iteration_map[route] = {}
        for alpha in config["alphas"]:
            suffix = quantile_suffix(alpha)
            selected_by_window: list[int] = []
            for window, validation_dates in windows.items():
                validation_set = set(validation_dates)
                first_validation_date = validation_dates[0]
                fit = route_train.loc[
                    route_train["date"].lt(first_validation_date)
                ].copy()
                valid = route_train.loc[
                    route_train["date"].isin(validation_set)
                ].copy()
                selected, concrete = _select_iteration_for_window(
                    fit=fit,
                    valid=valid,
                    features=features,
                    alpha=float(alpha),
                    structure=structure,
                    config=config,
                    base=base,
                )
                selected_iteration = int(selected["iteration"])
                selected_by_window.append(selected_iteration)
                rows.append(
                    {
                        "route": route,
                        "alpha": float(alpha),
                        "window": window,
                        "validation_start": pd.Timestamp(
                            validation_dates[0]
                        ),
                        "validation_end": pd.Timestamp(
                            validation_dates[-1]
                        ),
                        "fit_days": int(fit["date"].nunique()),
                        "fit_rows": len(fit),
                        "validation_days": int(valid["date"].nunique()),
                        "validation_rows": len(valid),
                        "min_child_samples": concrete.min_child_samples,
                        "patience": concrete.early_stopping_rounds,
                        "min_delta": float(early_stop["min_delta"]),
                        "selected_iteration": selected_iteration,
                        "validation_pinball": selected[
                            "validation_pinball"
                        ],
                        "stop_iteration": selected["stop_iteration"],
                        "curve_cap": selected["curve_cap"],
                        "absolute_cap_reached": selected[
                            "absolute_cap_reached"
                        ],
                    }
                )
            iteration_map[route][suffix] = int(
                np.median(selected_by_window)
            )
    return iteration_map, pd.DataFrame(rows)


def locked_iterations(
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    iteration_map = {
        route: {
            suffix: int(value)
            for suffix, value in config["locked_iterations"][route].items()
        }
        for route in ROUTE_ORDER
    }
    rows = [
        {
            "route": route,
            "alpha": float(alpha),
            "window": "locked_multiwindow_median",
            "selected_iteration": iteration_map[route][
                quantile_suffix(alpha)
            ],
            "min_delta": float(config["early_stopping"]["min_delta"]),
        }
        for route in ROUTE_ORDER
        for alpha in config["alphas"]
    ]
    return iteration_map, pd.DataFrame(rows)


def _fit_full_model(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
    model_config: ModelConfig,
    iterations: int,
) -> lgb.LGBMRegressor:
    concrete = replace(model_config, max_estimators=int(iterations))
    model = lgb.LGBMRegressor(
        **concrete.lgb_params(alpha, n_estimators=int(iterations))
    )
    model.fit(
        train[features].astype(np.float32),
        train["target_log_remaining"].astype(np.float32),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def _tail_aware_cdf(
    quantile_values: np.ndarray,
    alphas: np.ndarray,
    threshold: float,
) -> float:
    values = np.asarray(quantile_values, dtype=float)
    levels = np.asarray(alphas, dtype=float)
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        if threshold < unique_values[0]:
            return 0.0
        if threshold > unique_values[0]:
            return 1.0
        return 0.5
    unique_levels = np.asarray(
        [levels[values.eq(value)].max() for value in unique_values]
        if isinstance(values, pd.Series)
        else [
            levels[np.isclose(values, value, rtol=0.0, atol=1e-12)].max()
            for value in unique_values
        ],
        dtype=float,
    )
    if threshold < unique_values[0]:
        width = max(unique_values[1] - unique_values[0], 1e-8)
        cdf = unique_levels[0] + (
            threshold - unique_values[0]
        ) * (unique_levels[1] - unique_levels[0]) / width
        return float(np.clip(cdf, 0.0, unique_levels[0]))
    if threshold > unique_values[-1]:
        width = max(unique_values[-1] - unique_values[-2], 1e-8)
        cdf = unique_levels[-1] + (
            threshold - unique_values[-1]
        ) * (unique_levels[-1] - unique_levels[-2]) / width
        return float(np.clip(cdf, unique_levels[-1], 1.0))
    return float(np.interp(threshold, unique_values, unique_levels))


def exceedance_probabilities(
    final_quantiles: np.ndarray,
    alphas: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> np.ndarray:
    quantiles = np.asarray(final_quantiles, dtype=float)
    levels = np.asarray(alphas, dtype=float)
    output = np.empty((len(quantiles), len(thresholds)), dtype=float)
    for row_index, values in enumerate(quantiles):
        for threshold_index, threshold in enumerate(thresholds):
            output[row_index, threshold_index] = 1.0 - _tail_aware_cdf(
                values,
                levels,
                float(threshold),
            )
    return np.clip(output, 0.0, 1.0)


def predict_feature_rows(
    bundle: dict[str, Any],
    feature_rows: pd.DataFrame,
    include_internal: bool = False,
) -> pd.DataFrame:
    alphas = [float(value) for value in bundle["alphas"]]
    thresholds = [
        float(value) for value in bundle["probability_thresholds_pct"]
    ]
    frames: list[pd.DataFrame] = []
    for route in ROUTE_ORDER:
        subset = feature_rows.loc[route_mask(feature_rows, route)].copy()
        if subset.empty:
            continue
        expert = bundle["routes"][route]
        missing = sorted(set(expert["features"]) - set(subset.columns))
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required features for {route}: "
                + ", ".join(missing[:10])
            )
        raw_columns = [
            expert["models"][quantile_suffix(alpha)].predict(
                subset[expert["features"]].astype(np.float32)
            )
            for alpha in alphas
        ]
        raw_matrix = np.column_stack(raw_columns)
        sorted_matrix = np.sort(raw_matrix, axis=1)
        keep = [
            column
            for column in (
                "date",
                "bar_index",
                "current_amp",
                "remaining_amp",
                "target_log_remaining",
                "final_amp",
            )
            if column in subset.columns
        ]
        result = subset[keep].copy()
        result.insert(0, "route", route)
        current_amp = result["current_amp"].to_numpy(dtype=float)
        final_quantiles = np.empty_like(sorted_matrix)
        for index, alpha in enumerate(alphas):
            suffix = quantile_suffix(alpha)
            remaining = np.maximum(np.expm1(sorted_matrix[:, index]), 0.0)
            final_quantiles[:, index] = current_amp + remaining
            if include_internal:
                result[f"{suffix}_log_raw"] = raw_matrix[:, index]
                result[f"{suffix}_log"] = sorted_matrix[:, index]
            result[f"{suffix}_final_amp"] = final_quantiles[:, index]
        probabilities = exceedance_probabilities(
            final_quantiles,
            alphas,
            thresholds,
        )
        for index, threshold in enumerate(thresholds):
            label = str(threshold).replace(".", "_")
            result[f"prob_final_amp_gt_{label}pct"] = probabilities[:, index]
        frames.append(result)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "bar_index"])
        .reset_index(drop=True)
    )


def _evaluate_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alphas = [float(value) for value in config["alphas"]]
    selection_alphas = {
        float(value) for value in config["selection_alphas"]
    }
    metric_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for route in ROUTE_ORDER:
        group = predictions.loc[predictions["route"].eq(route)].copy()
        if group.empty:
            continue
        y = group["target_log_remaining"].to_numpy(dtype=float)
        raw_matrix = np.column_stack(
            [
                group[f"{quantile_suffix(alpha)}_log_raw"].to_numpy()
                for alpha in alphas
            ]
        )
        crossing = np.any(np.diff(raw_matrix, axis=1) < 0.0, axis=1)
        for alpha in alphas:
            suffix = quantile_suffix(alpha)
            metric_rows.append(
                {
                    "route": route,
                    "alpha": alpha,
                    "is_selection_quantile": alpha in selection_alphas,
                    "raw_pinball": _pinball(
                        y,
                        group[f"{suffix}_log_raw"].to_numpy(),
                        alpha,
                    ),
                    "sorted_pinball": _pinball(
                        y,
                        group[f"{suffix}_log"].to_numpy(),
                        alpha,
                    ),
                }
            )
        route_metrics = pd.DataFrame(
            metric_rows
        ).loc[lambda frame: frame["route"].eq(route)]
        actual_final = (
            group["current_amp"].to_numpy(dtype=float)
            + group["remaining_amp"].to_numpy(dtype=float)
        )
        q10 = group["q10_final_amp"].to_numpy(dtype=float)
        q50 = group["q50_final_amp"].to_numpy(dtype=float)
        q90 = group["q90_final_amp"].to_numpy(dtype=float)
        route_rows.append(
            {
                "route": route,
                "route_label": config["route_labels"][route],
                "days": int(group["date"].nunique()),
                "checkpoints": int(group["bar_index"].nunique()),
                "rows": len(group),
                "q3_raw_pinball": float(
                    route_metrics.loc[
                        route_metrics["is_selection_quantile"],
                        "raw_pinball",
                    ].mean()
                ),
                "q3_sorted_pinball": float(
                    route_metrics.loc[
                        route_metrics["is_selection_quantile"],
                        "sorted_pinball",
                    ].mean()
                ),
                "all9_raw_pinball": float(
                    route_metrics["raw_pinball"].mean()
                ),
                "all9_sorted_pinball": float(
                    route_metrics["sorted_pinball"].mean()
                ),
                "p50_mae_amp_pct_points": float(
                    np.mean(np.abs(actual_final - q50))
                ),
                "p10_p90_coverage": float(
                    np.mean((actual_final >= q10) & (actual_final <= q90))
                ),
                "quantile_crossing_row_rate": float(np.mean(crossing)),
            }
        )
    per_quantile = pd.DataFrame(metric_rows)
    per_route = pd.DataFrame(route_rows)
    weights = per_route["checkpoints"].to_numpy(dtype=float)
    summary_rows = [
        {
            "metric": "primary_route_equal_q3_raw_pinball",
            "value": float(per_route["q3_raw_pinball"].mean()),
        },
        {
            "metric": "route_equal_q3_sorted_pinball",
            "value": float(per_route["q3_sorted_pinball"].mean()),
        },
        {
            "metric": "route_equal_all9_raw_pinball",
            "value": float(per_route["all9_raw_pinball"].mean()),
        },
        {
            "metric": "route_equal_all9_sorted_pinball",
            "value": float(per_route["all9_sorted_pinball"].mean()),
        },
        {
            "metric": "checkpoint_equal_all9_raw_pinball",
            "value": float(
                np.average(per_route["all9_raw_pinball"], weights=weights)
            ),
        },
        {
            "metric": "checkpoint_equal_all9_sorted_pinball",
            "value": float(
                np.average(per_route["all9_sorted_pinball"], weights=weights)
            ),
        },
    ]
    return per_quantile, per_route, pd.DataFrame(summary_rows)


def run_training(
    data_dir: str | Path,
    output_dir: str | Path = V4_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    panel_cache: str | Path | None = None,
    rebuild_features: bool = False,
    reselect_iterations: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    report_dir = output_dir / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if panel_cache is None:
        panel_cache = artifact_dir / "feature_panel_cache.pkl"

    panel, feature_groups, audit, anomalies = build_feature_panel(
        data_dir=data_dir,
        cache_path=panel_cache,
        rebuild=rebuild_features,
        external_datasets=config["external_datasets"],
    )
    panel = panel.loc[~panel["excluded_anomaly"]].copy()
    feature_sets = build_feature_sets(
        feature_groups,
        Path(config_path).resolve().parent,
    )
    train_start = pd.Timestamp(config["train_start"])
    train_end = pd.Timestamp(config["train_end"])
    evaluation_start = pd.Timestamp(config["evaluation_start"])
    evaluation_end = pd.Timestamp(config["evaluation_end"])
    train = panel.loc[
        panel["date"].between(train_start, train_end)
    ].copy()
    evaluation = panel.loc[
        panel["date"].between(evaluation_start, evaluation_end)
    ].copy()
    if train.empty:
        raise ValueError("Training panel is empty for the configured period")

    if reselect_iterations:
        iteration_map, iteration_report = select_multiwindow_iterations(
            train,
            feature_sets,
            config,
        )
        iteration_source = "recomputed_multiwindow_median"
    else:
        iteration_map, iteration_report = locked_iterations(config)
        iteration_source = "locked_multiwindow_median"

    base = ModelConfig()
    routes: dict[str, dict[str, Any]] = {}
    structure_rows: list[dict[str, Any]] = []
    for route in ROUTE_ORDER:
        route_train = train.loc[route_mask(train, route)].copy()
        features = feature_sets[route]
        structure = config["route_structures"][route]
        full_config = materialize_model_config(
            base,
            structure,
            len(route_train),
            config,
        )
        models: dict[str, lgb.LGBMRegressor] = {}
        for alpha in config["alphas"]:
            suffix = quantile_suffix(alpha)
            models[suffix] = _fit_full_model(
                train=route_train,
                features=features,
                alpha=float(alpha),
                model_config=full_config,
                iterations=iteration_map[route][suffix],
            )
        routes[route] = {
            "features": features,
            "models": models,
            "iterations": iteration_map[route],
            "structure": structure,
            "min_child_samples_refit": full_config.min_child_samples,
            "train_rows": len(route_train),
            "train_days": int(route_train["date"].nunique()),
        }
        structure_rows.append(
            {
                "route": route,
                **structure,
                "min_child_samples_refit": full_config.min_child_samples,
                "train_rows": len(route_train),
                "train_days": int(route_train["date"].nunique()),
                "features": len(features),
            }
        )

    bundle = {
        "model_name": config["model_name"],
        "model_version": "v4",
        "target": config["target"],
        "trained_from": config["train_start"],
        "trained_through": config["train_end"],
        "external_datasets": config["external_datasets"],
        "alphas": config["alphas"],
        "selection_alphas": config["selection_alphas"],
        "probability_thresholds_pct": config[
            "probability_thresholds_pct"
        ],
        "route_definitions": config["route_definitions"],
        "route_labels": config["route_labels"],
        "routes": routes,
        "early_stopping": config["early_stopping"],
        "iteration_source": iteration_source,
        "evaluation_definition": config["evaluation"],
        "tail_probability_method": (
            "quantile-CDF linear interpolation with linear tail extrapolation"
        ),
    }
    model_path = artifact_dir / "csi1000_v4_final_model.joblib"
    joblib.dump(bundle, model_path, compress=3)

    active = {"csi1000", *config["external_datasets"]}
    audit.loc[audit["dataset"].isin(active)].to_csv(
        report_dir / "data_audit.csv",
        index=False,
    )
    active_anomalies = [
        item for item in anomalies if item.get("dataset") in active
    ]
    (report_dir / "data_anomalies.json").write_text(
        json.dumps(active_anomalies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(structure_rows).to_csv(
        report_dir / "selected_configuration.csv",
        index=False,
    )
    iteration_report.to_csv(
        report_dir / "tree_iteration_selection.csv",
        index=False,
    )
    feature_rows = [
        {"route": route, "feature": feature}
        for route, features in feature_sets.items()
        for feature in features
    ]
    pd.DataFrame(feature_rows).to_csv(
        report_dir / "feature_manifest.csv",
        index=False,
    )

    evaluation_outputs: dict[str, Any] = {}
    if not evaluation.empty:
        predictions_internal = predict_feature_rows(
            bundle,
            evaluation,
            include_internal=True,
        )
        per_quantile, per_route, summary = _evaluate_predictions(
            predictions_internal,
            config,
        )
        per_quantile.to_csv(
            report_dir / "quantile_metrics.csv",
            index=False,
        )
        per_route.to_csv(report_dir / "route_metrics.csv", index=False)
        summary.to_csv(report_dir / "model_summary.csv", index=False)
        internal_columns = [
            column
            for column in predictions_internal
            if column.endswith("_log") or column.endswith("_log_raw")
        ]
        predictions_internal.drop(columns=internal_columns).to_csv(
            report_dir / "evaluation_predictions.csv",
            index=False,
        )
        evaluation_outputs = {
            "per_quantile": per_quantile,
            "per_route": per_route,
            "summary": summary,
        }

    protocol = {
        "model_name": config["model_name"],
        "data_sources": ["csi1000", *config["external_datasets"]],
        "target_index": "csi1000",
        "train_period": [
            config["train_start"],
            config["train_end"],
        ],
        "evaluation_period": [
            config["evaluation_start"],
            config["evaluation_end"],
        ],
        "routes": config["route_definitions"],
        "preopen_uses_current_day_open_or_gap": False,
        "structure_selection_quantiles": config["selection_alphas"],
        "primary_evaluation": config["evaluation"]["primary"],
        "tree_iteration_source": iteration_source,
        "early_stopping": config["early_stopping"],
        "environment": {
            "python": "3.9.25",
            "lightgbm": "4.6.0",
            "numpy": "2.0.2",
            "pandas": "2.3.3",
            "scikit_learn": "1.6.1",
            "joblib": "1.5.3",
        },
    }
    (report_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "bundle": bundle,
        "model_path": model_path,
        "evaluation": evaluation_outputs,
        "iteration_report": iteration_report,
        "structure_report": pd.DataFrame(structure_rows),
    }
