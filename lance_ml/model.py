"""SentencePiece 编码与多任务 Transformer Encoder。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import sentencepiece as spm
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset

from .action_schema import ActionSchema
from .dataset import model_text


@dataclass(frozen=True)
class EncoderConfig:
    """EncoderConfig 保存可复现构建模型所需的全部结构参数。"""

    vocab_size: int
    max_length: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_size: int
    dropout: float
    pad_id: int
    tool_labels: tuple[str, ...]
    slot_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """to_dict 返回可 JSON 序列化的模型配置。"""

        payload = asdict(self)
        payload["tool_labels"] = list(self.tool_labels)
        payload["slot_labels"] = list(self.slot_labels)
        return payload

    @classmethod
    def from_path(cls, path: Path) -> "EncoderConfig":
        """from_path 从 JSON 文件加载模型配置。"""

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tool_labels"] = tuple(payload["tool_labels"])
        payload["slot_labels"] = tuple(payload["slot_labels"])
        return cls(**payload)


class RoutingTokenizer:
    """RoutingTokenizer 封装 SentencePiece 与槽位字符范围对齐。"""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @classmethod
    def train(
        cls,
        corpus_path: Path,
        model_prefix: Path,
        *,
        vocab_size: int,
    ) -> "RoutingTokenizer":
        """train 从训练集文本建立中英文混合 SentencePiece 词表。"""

        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=vocab_size,
                model_type="unigram",
                character_coverage=0.9995,
                byte_fallback=True,
                hard_vocab_limit=False,
                normalization_rule_name="identity",
                minloglevel=2,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                user_defined_symbols=[
                    "<JWT>",
                    "<URL>",
                    "<USERNAME>",
                    "<SECRET>",
                    "[MODE_workspace]",
                    "[MODE_editor]",
                    "[MODE_quick_access]",
                    "[SELECTION_0]",
                    "[SELECTION_1]",
                ],
            )
        except RuntimeError as error:
            if "Vocabulary size is smaller than required_chars" in str(error):
                raise ValueError(
                    f"vocab-size={vocab_size} 小于当前语料和 byte fallback 的最小词表要求"
                ) from error
            raise
        return cls(model_prefix.with_suffix(".model"))

    def encode(
        self,
        text: str,
        *,
        max_length: int,
        slot_spans: Sequence[dict[str, Any]],
        slot_to_id: dict[str, int],
    ) -> tuple[list[int], list[int], list[int]]:
        """encode 生成定长 token、注意力掩码和 BIO 槽位标签。"""

        proto = self.processor.encode(text, return_type="proto")
        pieces = list(proto.pieces)
        byte_offsets = any(piece.end > len(text) for piece in pieces)
        spans = []
        for slot in slot_spans:
            start = int(slot["start"])
            end = int(slot["end"])
            if byte_offsets:
                start = len(text[:start].encode("utf-8"))
                end = len(text[:end].encode("utf-8"))
            spans.append((str(slot["name"]), start, end))

        piece_ids = [int(piece.id) for piece in pieces[: max_length - 2]]
        piece_labels: list[int] = []
        started: set[str] = set()
        for piece in pieces[: max_length - 2]:
            label = slot_to_id["O"]
            for name, start, end in spans:
                if piece.end > start and piece.begin < end:
                    prefix = "I" if name in started else "B"
                    label = slot_to_id[f"{prefix}-{name}"]
                    started.add(name)
                    break
            piece_labels.append(label)

        input_ids = [self.processor.bos_id(), *piece_ids, self.processor.eos_id()]
        slot_ids = [-100, *piece_labels, -100]
        attention_mask = [1] * len(input_ids)
        padding = max_length - len(input_ids)
        input_ids.extend([self.processor.pad_id()] * padding)
        attention_mask.extend([0] * padding)
        slot_ids.extend([-100] * padding)
        return input_ids, attention_mask, slot_ids

    @property
    def vocab_size(self) -> int:
        """vocab_size 返回实际训练得到的词表大小。"""

        return int(self.processor.vocab_size())

    @property
    def pad_id(self) -> int:
        """pad_id 返回 SentencePiece 填充 token ID。"""

        return int(self.processor.pad_id())


class ToolRoutingEncoder(nn.Module):
    """ToolRoutingEncoder 联合输出工具、槽位和弃权判断。"""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_id
        )
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.tool_head = nn.Linear(config.hidden_size, len(config.tool_labels))
        self.slot_head = nn.Linear(config.hidden_size, len(config.slot_labels))
        self.abstain_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward 对定长 token 序列执行多任务推理。"""

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
        hidden = self.final_norm(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = self.dropout(pooled)
        return (
            self.tool_head(pooled),
            self.slot_head(hidden),
            self.abstain_head(pooled).squeeze(-1),
        )

    def parameter_count(self) -> int:
        """parameter_count 返回模型总参数量。"""

        return sum(parameter.numel() for parameter in self.parameters())


class EncoderDataset(Dataset[dict[str, torch.Tensor]]):
    """EncoderDataset 将 JSONL 样本预编码为训练张量。"""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: RoutingTokenizer,
        config: EncoderConfig,
    ) -> None:
        self.records = records
        self.tool_to_id = {label: index for index, label in enumerate(config.tool_labels)}
        self.slot_to_id = {label: index for index, label in enumerate(config.slot_labels)}
        self.encoded = []
        for record in records:
            text = model_text(record)
            prefix_length = len(text) - len(record["text"])
            spans = [
                {
                    **slot,
                    "start": int(slot["start"]) + prefix_length,
                    "end": int(slot["end"]) + prefix_length,
                }
                for slot in record["slots"]
            ]
            input_ids, attention_mask, slot_ids = tokenizer.encode(
                text,
                max_length=config.max_length,
                slot_spans=spans,
                slot_to_id=self.slot_to_id,
            )
            self.encoded.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "slot_targets": torch.tensor(slot_ids, dtype=torch.long),
                    "tool_target": torch.tensor(
                        self.tool_to_id[record["tool"]], dtype=torch.long
                    ),
                    "abstain_target": torch.tensor(
                        1.0 if record["tool"] == "abstain" else 0.0,
                        dtype=torch.float32,
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encoded[index]


def build_labels(schema: ActionSchema) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """build_labels 从权威动作字段生成工具和 BIO 槽位标签。"""

    tools = ("abstain", *sorted(action.id for action in schema.actions))
    slot_names = sorted(
        {name for action in schema.actions for name in action.parameters}
    )
    slots = ("O", *[label for name in slot_names for label in (f"B-{name}", f"I-{name}")])
    return tools, slots


def select_device(requested: str) -> torch.device:
    """select_device 依据真实后端可用性选择训练设备。"""

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("请求了 MPS，但当前进程无法访问 MPS 设备")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前进程无法访问 CUDA 设备")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(artifact_dir: Path, device: torch.device) -> tuple[
    ToolRoutingEncoder, RoutingTokenizer, EncoderConfig
]:
    """load_encoder 从 safetensors 产物恢复 Encoder。"""

    config = EncoderConfig.from_path(artifact_dir / "config.json")
    tokenizer = RoutingTokenizer(artifact_dir / "tokenizer.model")
    model = ToolRoutingEncoder(config)
    model.load_state_dict(load_file(artifact_dir / "model.safetensors"))
    model.to(device)
    model.eval()
    return model, tokenizer, config
