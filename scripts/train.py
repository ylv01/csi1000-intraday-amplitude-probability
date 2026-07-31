from __future__ import annotations

import argparse
import sys
from pathlib import Path


V4_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from csi1000_v4.final_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练中证1000日内振幅V4最终模型。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=V4_ROOT / "data",
        help="原始CSV数据目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=V4_ROOT,
        help="模型和报告输出目录。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=V4_ROOT / "config" / "final_config.json",
    )
    parser.add_argument(
        "--panel-cache",
        type=Path,
        default=None,
        help="可选的特征面板缓存路径。",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="忽略已有特征缓存并重新构建。",
    )
    parser.add_argument(
        "--reselect-iterations",
        action="store_true",
        help="重新执行三个120日窗口选树；默认使用锁定的最终树数。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        panel_cache=args.panel_cache,
        rebuild_features=args.rebuild_features,
        reselect_iterations=args.reselect_iterations,
    )
    print(f"模型：{result['model_path'].resolve()}")
    if result["evaluation"]:
        print("\n最终评价")
        print(result["evaluation"]["summary"].to_string(index=False))
        print("\n分路由评价")
        print(result["evaluation"]["per_route"].to_string(index=False))
