from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V4_ROOT = Path(__file__).resolve().parents[1]
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from csi1000_v4.feature_pipeline import audit_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成V4使用数据的数据审计报告。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=V4_ROOT / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=V4_ROOT / "reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit, anomalies = audit_inputs(args.data_dir)
    active = {"csi1000", "sse50", "csi300"}
    audit = audit.loc[audit["dataset"].isin(active)].copy()
    anomalies = [
        item for item in anomalies if item.get("dataset") in active
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "data_audit.csv", index=False)
    (args.output_dir / "data_anomalies.json").write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
