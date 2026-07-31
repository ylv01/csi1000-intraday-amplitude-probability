from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


V4_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from csi1000_v4.feature_pipeline import build_feature_panel
from csi1000_v4.final_pipeline import predict_feature_rows


def _read_feature_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    frame = pd.read_csv(path)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用V4模型生成P10至P90和指定振幅超越概率。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=V4_ROOT / "artifacts" / "csi1000_v4_final_model.joblib",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--feature-file",
        type=Path,
        help="已经构造好的CSV或Pickle特征面板。",
    )
    source.add_argument(
        "--data-dir",
        type=Path,
        help="从原始CSV构造研究批量特征。",
    )
    parser.add_argument(
        "--panel-cache",
        type=Path,
        default=V4_ROOT / "artifacts" / "feature_panel_cache.pkl",
        help="特征缓存路径；不存在时自动从V4/data构建。",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="只输出指定日期，格式YYYY-MM-DD。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V4_ROOT / "reports" / "latest_predictions.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle = joblib.load(args.model)
    if args.feature_file is not None:
        panel = _read_feature_file(args.feature_file)
    else:
        data_dir = args.data_dir or (V4_ROOT / "data")
        panel, _, _, _ = build_feature_panel(
            data_dir=data_dir,
            cache_path=args.panel_cache,
            rebuild=False,
            external_datasets=bundle["external_datasets"],
        )
    if args.date:
        selected_date = pd.Timestamp(args.date)
        panel = panel.loc[pd.to_datetime(panel["date"]).eq(selected_date)]
    predictions = predict_feature_rows(bundle, panel)
    if predictions.empty:
        raise ValueError("没有可输出的预测行")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"预测行数：{len(predictions)}")
    print(f"输出：{args.output.resolve()}")
