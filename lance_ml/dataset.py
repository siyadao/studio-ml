"""训练数据读写、模型输入和统一离线指标。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """load_jsonl 读取并校验非空 JSONL 数据集。"""

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"数据集为空: {path}")
    return records


def load_manifest(data_dir: Path) -> dict[str, Any]:
    """load_manifest 读取清单并验证三个数据切分的内容哈希。"""

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(data_dir / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"数据文件哈希不匹配: {filename}")
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """write_json 以稳定格式写入 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """sha256_file 返回文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_text(record: dict[str, Any]) -> str:
    """model_text 将非敏感应用上下文编码到模型输入。"""

    context = record.get("context", {})
    mode = str(context.get("active_mode", "workspace"))
    selection = "1" if context.get("has_selection", False) else "0"
    return f"[MODE_{mode}] [SELECTION_{selection}] {record['text']}"


def calculate_metrics(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    probabilities: np.ndarray,
    risk_by_label: dict[str, str],
) -> dict[str, Any]:
    """calculate_metrics 计算 Top 1、Top 3、弃权和高风险误触发指标。"""

    if probabilities.shape != (len(records), len(labels)):
        raise ValueError("概率矩阵尺寸与数据集或标签不一致")
    label_array = np.asarray(labels)
    top_count = min(3, len(labels))
    top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :top_count]
    predicted = label_array[top_indices[:, 0]]
    expected = np.asarray([record["tool"] for record in records])

    top1 = float(np.mean(predicted == expected))
    top3 = float(
        np.mean(
            [truth in label_array[indexes] for truth, indexes in zip(expected, top_indices)]
        )
    )
    abstain_mask = expected == "abstain"
    abstain_recall = (
        float(np.mean(predicted[abstain_mask] == "abstain"))
        if np.any(abstain_mask)
        else 1.0
    )

    high_risk_false_triggers = 0
    for truth, guess in zip(expected, predicted):
        truth_risk = risk_by_label.get(str(truth), "read_only")
        guess_risk = risk_by_label.get(str(guess), "read_only")
        if truth_risk in {"read_only", "write"} and guess_risk in {"high", "critical"}:
            high_risk_false_triggers += 1

    counts = Counter(str(value) for value in expected)
    correct = defaultdict(int)
    for truth, guess in zip(expected, predicted):
        if truth == guess:
            correct[str(truth)] += 1
    per_tool_recall = {
        label: correct[label] / count for label, count in sorted(counts.items())
    }

    return {
        "samples": len(records),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "abstain_recall": abstain_recall,
        "high_risk_false_triggers": high_risk_false_triggers,
        "per_tool_recall": per_tool_recall,
    }


def count_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    """count_by 统计数据集中指定字段的样本数。"""

    return dict(sorted(Counter(str(record[key]) for record in records).items()))
