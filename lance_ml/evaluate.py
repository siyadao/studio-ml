"""统一评估字符基线和 Encoder。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from .action_schema import DATA_DIR, ML_ROOT, load_action_schema
from .dataset import calculate_metrics, load_jsonl, load_manifest, model_text, write_json
from .model import EncoderDataset, load_encoder, select_device
from .train import evaluate_encoder


def parse_args() -> argparse.Namespace:
    """parse_args 解析统一评估参数。"""

    parser = argparse.ArgumentParser(description="评估智能工具路由模型")
    parser.add_argument("--model", choices=("baseline", "encoder"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def evaluate_baseline(
    artifact_dir: Path,
    records: list[dict],
    risk_by_label: dict[str, str],
) -> dict:
    """evaluate_baseline 加载本地 joblib 基线并计算指标。"""

    bundle = joblib.load(artifact_dir / "baseline.joblib")
    features = bundle["vectorizer"].transform(model_text(record) for record in records)
    probabilities = bundle["classifier"].predict_proba(features)
    return calculate_metrics(records, bundle["labels"], probabilities, risk_by_label)


def main() -> None:
    """main 校验数据版本并评估指定模型。"""

    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = load_manifest(data_dir)
    schema = load_action_schema(data_dir / "actions.json")
    records = load_jsonl(data_dir / f"{args.split}.jsonl")
    risk_by_label = {action.id: action.risk for action in schema.actions}
    risk_by_label["abstain"] = "read_only"
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir
        else ML_ROOT / "artifacts" / args.model
    )
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("模型与当前数据集版本不一致")
    if metadata["schema_sha256"] != schema.sha256:
        raise ValueError("模型与当前动作 Schema 不一致")

    if args.model == "baseline":
        metrics = evaluate_baseline(artifact_dir, records, risk_by_label)
    else:
        device = select_device(args.device)
        model, tokenizer, config = load_encoder(artifact_dir, device)
        dataset = EncoderDataset(records, tokenizer, config)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        metrics = evaluate_encoder(
            model, loader, records, config, risk_by_label, device
        )
        metrics["device"] = str(device)

    output = {
        "model": args.model,
        "split": args.split,
        "dataset_version": manifest["dataset_version"],
        "schema_version": schema.schema_version,
        "metrics": metrics,
    }
    write_json(artifact_dir / f"evaluation-{args.split}.json", output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
