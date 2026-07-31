from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


V4_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from csi1000_v4.final_pipeline import (
    exceedance_probabilities,
    route_mask,
    select_validation_windows,
    simulate_early_stopping,
    validate_feature_sets,
)


def test_routes_are_exhaustive_and_non_overlapping() -> None:
    panel = pd.DataFrame({"bar_index": [0, 5, 60, 65, 120, 125, 235]})
    masks = {
        route: route_mask(panel, route)
        for route in ("preopen", "early", "late")
    }
    total = sum(mask.astype(int) for mask in masks.values())
    assert total.eq(1).all()
    assert masks["preopen"].sum() == 1
    assert masks["early"].sum() == 2
    assert masks["late"].sum() == 4


def test_preopen_rejects_current_day_features() -> None:
    valid = {
        "preopen": ["csi1000_hist_amp_lag1"],
        "early": ["bar_index", "csi1000_cur_amp"],
        "late": ["bar_index", "csi1000_cur_amp"],
    }
    validate_feature_sets(valid)
    invalid = {key: list(value) for key, value in valid.items()}
    invalid["preopen"].append("csi1000_cur_return")
    try:
        validate_feature_sets(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Current-day preopen feature was accepted")


def test_validation_windows_are_equal_and_non_overlapping() -> None:
    windows = select_validation_windows(
        np.arange(500),
        window_days=120,
        window_count=3,
    )
    assert [len(value) for value in windows.values()] == [120, 120, 120]
    assert windows["W1"][0] == 140
    assert windows["W3"][-1] == 499
    assert not set(windows["W1"]).intersection(windows["W2"])


def test_early_stopping_uses_historical_best() -> None:
    selected = simulate_early_stopping(
        [1.0, 0.9, 0.91, 0.92],
        patience=2,
        min_delta=0.0,
    )
    assert selected["iteration"] == 2
    assert selected["stop_iteration"] == 4


def test_exceedance_probabilities_are_monotone_in_threshold() -> None:
    final_quantiles = np.asarray(
        [
            [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
            [0.8, 1.0, 1.1, 1.3, 1.5, 1.9, 2.5, 3.0, 4.0],
        ]
    )
    probabilities = exceedance_probabilities(
        final_quantiles,
        np.arange(0.1, 1.0, 0.1),
        [1.0, 1.5, 2.0, 3.0, 4.0],
    )
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all(np.diff(probabilities, axis=1) <= 1e-12)
