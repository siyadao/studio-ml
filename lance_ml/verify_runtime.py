"""通过真实模型 Worker 校验 Rust 与 Python 基线推理一致性。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib

from .action_schema import DATA_DIR, ML_ROOT
from .dataset import load_jsonl, model_text


DEFAULT_ARTIFACT_DIR = ML_ROOT / "artifacts" / "baseline"


def parse_args() -> argparse.Namespace:
    """parse_args 解析跨语言运行时检查参数。"""

    parser = argparse.ArgumentParser(description="校验 Rust 智能路由 Worker")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    """main 对每条脱敏样本比较 Rust、Python 与期望 Top-1。"""

    args = parse_args()
    records = load_jsonl(args.data_dir.resolve() / f"{args.split}.jsonl")
    bundle = joblib.load(args.artifact_dir.resolve() / "baseline.joblib")
    features = bundle["vectorizer"].transform(model_text(record) for record in records)
    python_predictions = bundle["classifier"].predict(features)

    expected_matches = 0
    python_matches = 0
    with subprocess.Popen(
        [str(args.worker.resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as worker:
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("模型 Worker 标准输入输出不可用")
        for generation, (record, python_prediction) in enumerate(
            zip(records, python_predictions, strict=True), start=1
        ):
            context = record.get("context", {})
            request = {
                "version": 1,
                "generation": generation,
                "text": record["text"],
                "active_mode": context.get("active_mode", "workspace"),
                "has_selection": bool(context.get("has_selection", False)),
            }
            worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                stderr = worker.stderr.read() if worker.stderr else ""
                raise RuntimeError(f"模型 Worker 提前退出: {stderr}")
            response = json.loads(line)
            if response.get("error"):
                raise RuntimeError(response["error"])
            rust_prediction = response["candidates"][0]["action_id"]
            expected_matches += int(rust_prediction == record["tool"])
            python_matches += int(rust_prediction == python_prediction)
        worker.stdin.close()

    total = len(records)
    print(
        f"split={args.split} samples={total} "
        f"rust_expected_top1={expected_matches / total:.6f} "
        f"rust_python_top1_agreement={python_matches / total:.6f}"
    )
    if python_matches != total:
        raise RuntimeError(f"Rust 与 Python Top-1 有 {total - python_matches} 条不一致")


if __name__ == "__main__":
    main()
