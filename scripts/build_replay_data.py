from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


V4_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from csi1000_v4.feature_pipeline import build_feature_panel
from csi1000_v4.final_pipeline import (
    load_config,
    predict_feature_rows,
)


def build_replay_data(
    data_dir: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    panel_cache: str | Path | None = None,
    rebuild_features: bool = False,
) -> dict:
    config = load_config(config_path)
    bundle = joblib.load(model_path)
    panel, _, _, _ = build_feature_panel(
        data_dir=data_dir,
        cache_path=panel_cache,
        rebuild=rebuild_features,
        external_datasets=bundle["external_datasets"],
    )
    start = pd.Timestamp(config["train_start"])
    end = pd.Timestamp(config["evaluation_end"])
    panel = panel.loc[
        panel["date"].between(start, end) & ~panel["excluded_anomaly"]
    ].copy()
    predictions = predict_feature_rows(bundle, panel)
    keep = [
        "date",
        "route",
        "bar_index",
        "current_amp",
        "final_amp",
        *[
            f"q{value:02d}_final_amp"
            for value in range(10, 100, 10)
        ],
        *[
            column
            for column in predictions.columns
            if column.startswith("prob_final_amp_gt_")
        ],
    ]
    replay = predictions[keep].copy()
    replay["date"] = pd.to_datetime(replay["date"]).dt.normalize()
    replay["period_label"] = replay["date"].le(
        pd.Timestamp(config["train_end"])
    ).map(
        {
            True: "历史回放",
            False: "模型评价",
        }
    )
    replay = replay.sort_values(["date", "bar_index"]).reset_index(drop=True)
    payload = {
        "model_name": bundle["model_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "amplitude_definition": "(High - Low) / PreClose * 100",
        "date_start": str(replay["date"].min().date()),
        "date_end": str(replay["date"].max().date()),
        "days": int(replay["date"].nunique()),
        "rows": len(replay),
        "checkpoints_per_complete_day": 48,
        "frame": replay,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path, compress=3)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用V4最终模型生成2019—2026历史回放数据。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=V4_ROOT / "data",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=V4_ROOT
        / "artifacts"
        / "csi1000_v4_final_model.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V4_ROOT
        / "artifacts"
        / "historical_replay_predictions.joblib",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=V4_ROOT / "config" / "final_config.json",
    )
    parser.add_argument(
        "--panel-cache",
        type=Path,
        default=V4_ROOT / "artifacts" / "feature_panel_cache.pkl",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_replay_data(
        data_dir=args.data_dir,
        model_path=args.model,
        output_path=args.output,
        config_path=args.config,
        panel_cache=args.panel_cache,
        rebuild_features=args.rebuild_features,
    )
    print(
        "历史回放数据："
        f"{result['date_start']} 至 {result['date_end']}，"
        f"{result['days']}个交易日，{result['rows']}行"
    )
    print(f"输出：{args.output.resolve()}")
