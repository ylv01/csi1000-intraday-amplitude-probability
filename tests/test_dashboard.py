from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest


V4_ROOT = Path(__file__).resolve().parents[1]
REPLAY_FILE = V4_ROOT / "artifacts" / "historical_replay_predictions.joblib"


def test_historical_replay_artifact_is_complete_and_monotone() -> None:
    payload = joblib.load(REPLAY_FILE)
    frame = payload["frame"]

    assert payload["date_start"] == "2019-01-02"
    assert payload["date_end"] == "2026-07-24"
    assert payload["days"] == frame["date"].nunique() == 1832
    assert payload["rows"] == len(frame) == 87_936
    assert frame.groupby("date").size().eq(48).all()
    assert not frame.isna().any().any()

    quantile_columns = [f"q{level}_final_amp" for level in range(10, 100, 10)]
    probability_columns = [
        "prob_final_amp_gt_1_0pct",
        "prob_final_amp_gt_1_5pct",
        "prob_final_amp_gt_2_0pct",
        "prob_final_amp_gt_3_0pct",
        "prob_final_amp_gt_4_0pct",
    ]
    assert np.all(np.diff(frame[quantile_columns].to_numpy(), axis=1) >= 0.0)
    assert np.all(np.diff(frame[probability_columns].to_numpy(), axis=1) <= 0.0)


def test_streamlit_dashboard_renders_without_exception() -> None:
    app = AppTest.from_file(str(V4_ROOT / "app.py"), default_timeout=60).run()

    assert not app.exception
    assert len(app.tabs) == 5
    assert len(app.metric) >= 10
    assert len(app.dataframe) >= 5
    assert len(app.get("download_button")) == 1

    app.selectbox[0].select(2019).run()
    assert not app.exception
    assert app.selectbox[0].value == 2019
    assert str(app.selectbox[1].value.date()) == "2019-12-31"
