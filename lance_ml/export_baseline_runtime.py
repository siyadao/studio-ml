"""导出供 Rust 模型 Worker 使用的紧凑字符 n-gram 基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import joblib
import numpy as np

from .action_schema import DATA_DIR, ML_ROOT, load_action_schema
from .dataset import load_manifest


MODEL_MAGIC = b"SYMODEL2"
MODEL_FORMAT_VERSION = 1
DEFAULT_ARTIFACT_DIR = ML_ROOT / "artifacts" / "baseline"
DEFAULT_OUTPUT = ML_ROOT / "artifacts" / "runtime" / "smart-input-baseline.bin"


def parse_args() -> argparse.Namespace:
    """parse_args 解析运行时模型导出参数。"""

    parser = argparse.ArgumentParser(description="导出 Rust 智能路由运行时模型")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_u16_string(output, value: str) -> None:
    """write_u16_string 写入带 u16 长度的 UTF-8 字符串。"""

    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFF:
        raise ValueError("运行时模型字符串超过 u16 长度")
    output.write(struct.pack("<H", len(encoded)))
    output.write(encoded)


def main() -> None:
    """main 校验训练产物并导出稳定的 little-endian 二进制模型。"""

    args = parse_args()
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output_path = args.output.resolve()

    manifest = load_manifest(data_dir)
    schema = load_action_schema(data_dir / "actions.json")
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("基线模型与当前数据版本不一致")
    if metadata["schema_sha256"] != schema.sha256:
        raise ValueError("基线模型与当前动作 Schema 不一致")

    bundle = joblib.load(artifact_dir / "baseline.joblib")
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    labels = [str(label) for label in classifier.classes_]
    expected_labels = sorted(["abstain", *(action.id for action in schema.actions)])
    if labels != expected_labels:
        raise ValueError("基线模型标签与权威动作 Schema 不一致")

    vocabulary = vectorizer.vocabulary_
    vocabulary_size = len(vocabulary)
    ordered_terms: list[str | None] = [None] * vocabulary_size
    for term, index in vocabulary.items():
        if not 0 <= index < vocabulary_size or ordered_terms[index] is not None:
            raise ValueError("TF-IDF 词表索引不连续或重复")
        ordered_terms[index] = str(term)
    if any(term is None for term in ordered_terms):
        raise ValueError("TF-IDF 词表存在空洞")

    idf = np.asarray(vectorizer.idf_, dtype="<f4")
    if idf.shape != (vocabulary_size,):
        raise ValueError("TF-IDF IDF 维度不符合词表")

    estimators = classifier.calibrated_classifiers_
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(MODEL_MAGIC)
        output.write(
            struct.pack(
                "<IIIIII",
                MODEL_FORMAT_VERSION,
                schema.schema_version,
                vectorizer.ngram_range[0],
                vectorizer.ngram_range[1],
                vocabulary_size,
                len(labels),
            )
        )
        write_u16_string(output, manifest["dataset_version"])
        write_u16_string(output, schema.sha256)
        for term, term_idf in zip(ordered_terms, idf, strict=True):
            write_u16_string(output, term or "")
            output.write(struct.pack("<f", float(term_idf)))
        for label in labels:
            write_u16_string(output, label)
        output.write(struct.pack("<I", len(estimators)))

        expected_shape = (len(labels), vocabulary_size)
        for calibrated in estimators:
            coefficients = np.asarray(calibrated.estimator.coef_, dtype="<f4")
            intercepts = np.asarray(calibrated.estimator.intercept_, dtype="<f4")
            if coefficients.shape != expected_shape:
                raise ValueError("分类器权重维度不符合标签和词表")
            if intercepts.shape != (len(labels),):
                raise ValueError("分类器截距维度不符合标签")
            if len(calibrated.calibrators) != len(labels):
                raise ValueError("分类器校准器数量不符合标签")

            output.write(intercepts.tobytes(order="C"))
            for calibrator in calibrated.calibrators:
                output.write(struct.pack("<ff", float(calibrator.a_), float(calibrator.b_)))
            output.write(coefficients.tobytes(order="C"))

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(digest + "\n", encoding="ascii")
    print(
        f"runtime_model={output_path} bytes={output_path.stat().st_size} "
        f"schema={schema.schema_version} dataset={manifest['dataset_version']} sha256={digest}"
    )


if __name__ == "__main__":
    main()
