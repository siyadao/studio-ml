"""训练多任务小型 Transformer Encoder。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .action_schema import DATA_DIR, ML_ROOT, load_action_schema
from .dataset import calculate_metrics, load_jsonl, load_manifest, model_text, sha256_file, write_json
from .model import (
    EncoderConfig,
    EncoderDataset,
    RoutingTokenizer,
    ToolRoutingEncoder,
    build_labels,
    select_device,
)


DEFAULT_ARTIFACT_DIR = ML_ROOT / "artifacts" / "encoder"
DEFAULT_SEED = 20260901


def parse_args() -> argparse.Namespace:
    """parse_args 解析 Encoder 训练参数。"""

    parser = argparse.ArgumentParser(description="训练智能工具路由 Encoder")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=8_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-size", type=int, default=1_536)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """set_seed 固定 Python、NumPy 与 PyTorch 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_encoder(
    model: ToolRoutingEncoder,
    loader: DataLoader[dict[str, torch.Tensor]],
    records: list[dict[str, Any]],
    config: EncoderConfig,
    risk_by_label: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    """evaluate_encoder 计算分类、弃权与槽位指标。"""

    model.eval()
    probability_batches = []
    slot_expected = []
    slot_predicted = []
    abstain_index = config.tool_labels.index("abstain")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tool_logits, slot_logits, abstain_logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(tool_logits, dim=-1)
            abstain_probabilities = torch.sigmoid(abstain_logits)
            probabilities[:, abstain_index] = torch.maximum(
                probabilities[:, abstain_index], abstain_probabilities
            )
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            probability_batches.append(probabilities.cpu().numpy())

            targets = batch["slot_targets"]
            predictions = slot_logits.argmax(dim=-1).cpu()
            active = targets != -100
            slot_expected.extend(targets[active].tolist())
            slot_predicted.extend(predictions[active].tolist())

    probabilities = np.concatenate(probability_batches, axis=0)
    metrics = calculate_metrics(
        records, config.tool_labels, probabilities, risk_by_label
    )
    non_o_labels = list(range(1, len(config.slot_labels)))
    metrics["slot_macro_f1"] = (
        float(
            f1_score(
                slot_expected,
                slot_predicted,
                labels=non_o_labels,
                average="macro",
                zero_division=0,
            )
        )
        if non_o_labels
        else 1.0
    )
    return metrics


def save_model(model: ToolRoutingEncoder, path: Path) -> None:
    """save_model 以连续 CPU 张量写入安全检查点。"""

    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, path)


def main() -> None:
    """main 训练 Encoder 并保存验证集最佳检查点。"""

    args = parse_args()
    set_seed(args.seed)
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(data_dir)
    schema = load_action_schema(data_dir / "actions.json")
    train_records = load_jsonl(data_dir / "train.jsonl")
    validation_records = load_jsonl(data_dir / "validation.jsonl")

    corpus_path = artifact_dir / "tokenizer-corpus.txt"
    corpus_path.write_text(
        "\n".join(model_text(record) for record in train_records) + "\n",
        encoding="utf-8",
    )
    tokenizer = RoutingTokenizer.train(
        corpus_path,
        artifact_dir / "tokenizer",
        vocab_size=args.vocab_size,
    )
    corpus_path.unlink()

    tool_labels, slot_labels = build_labels(schema)
    config = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
        tool_labels=tool_labels,
        slot_labels=slot_labels,
    )
    if config.hidden_size % config.num_heads != 0:
        raise ValueError("hidden-size 必须能被 heads 整除")
    write_json(artifact_dir / "config.json", config.to_dict())

    train_dataset = EncoderDataset(train_records, tokenizer, config)
    validation_dataset = EncoderDataset(validation_records, tokenizer, config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = select_device(args.device)
    model = ToolRoutingEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    tool_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    abstain_loss_fn = nn.BCEWithLogitsLoss()
    risk_by_label = {action.id: action.risk for action in schema.actions}
    risk_by_label["abstain"] = "read_only"

    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    stale_epochs = 0
    model_path = artifact_dir / "model.safetensors"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tool_targets = batch["tool_target"].to(device)
            slot_targets = batch["slot_targets"].to(device)
            abstain_targets = batch["abstain_target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            tool_logits, slot_logits, abstain_logits = model(input_ids, attention_mask)
            loss = (
                tool_loss_fn(tool_logits, tool_targets)
                + 0.5
                * slot_loss_fn(slot_logits.reshape(-1, len(slot_labels)), slot_targets.reshape(-1))
                + 0.25 * abstain_loss_fn(abstain_logits, abstain_targets)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        metrics = evaluate_encoder(
            model,
            validation_loader,
            validation_records,
            config,
            risk_by_label,
            device,
        )
        score = metrics["top1_accuracy"]
        print(
            f"epoch={epoch} loss={total_loss / len(train_loader):.4f} "
            f"top1={score:.4f} top3={metrics['top3_accuracy']:.4f} "
            f"abstain={metrics['abstain_recall']:.4f} slot_f1={metrics['slot_macro_f1']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            stale_epochs = 0
            save_model(model, model_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_metrics is None:
        raise RuntimeError("训练未产生有效检查点")
    metadata = {
        "model_type": "multitask_transformer_encoder",
        "dataset_version": manifest["dataset_version"],
        "schema_version": schema.schema_version,
        "schema_sha256": schema.sha256,
        "random_seed": args.seed,
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "training_parameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "vocab_size_requested": args.vocab_size,
            "max_length": args.max_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "ffn_size": args.ffn_size,
            "dropout": args.dropout,
        },
        "best_checkpoint": str(model_path),
        "validation_metrics": best_metrics,
        "model_sha256": sha256_file(model_path),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    print(
        f"dataset={manifest['dataset_version']} schema={schema.schema_version} seed={args.seed} "
        f"parameters={metadata['parameter_count']} checkpoint={model_path} "
        f"sha256={metadata['model_sha256']}"
    )


if __name__ == "__main__":
    main()
