"""导出并验证工具路由 Encoder 的 ONNX INT8 产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from .action_schema import ML_ROOT
from .dataset import sha256_file, write_json
from .model import load_encoder


DEFAULT_ARTIFACT_DIR = ML_ROOT / "artifacts" / "encoder"
MAX_MODEL_BYTES = 40 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """parse_args 解析 ONNX 导出参数。"""

    parser = argparse.ArgumentParser(description="导出智能工具路由 ONNX 模型")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--quantize", choices=("none", "int8"), default="int8")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def validate_runtime(path: Path, max_length: int) -> dict[str, list[int]]:
    """validate_runtime 使用 CPUExecutionProvider 验证真实 ONNX 推理。"""

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_ids = np.zeros((1, max_length), dtype=np.int64)
    attention_mask = np.zeros((1, max_length), dtype=np.int64)
    input_ids[:, :4] = np.asarray([1, 4, 5, 2], dtype=np.int64)
    attention_mask[:, :4] = 1
    outputs = session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    return {
        output.name: list(value.shape)
        for output, value in zip(session.get_outputs(), outputs)
    }


def decompose_gemm(model: onnx.ModelProto) -> None:
    """decompose_gemm 将线性头规范化为可动态量化的 MatMul 与 Add。"""

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        if node.op_type != "Gemm":
            nodes.append(node)
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("transA", 0) != 0:
            raise ValueError(f"暂不支持带 transA 的 Gemm: {node.name}")
        if float(attributes.get("alpha", 1.0)) != 1.0:
            raise ValueError(f"暂不支持带 alpha 缩放的 Gemm: {node.name}")
        if len(node.input) >= 3 and float(attributes.get("beta", 1.0)) != 1.0:
            raise ValueError(f"暂不支持带 beta 缩放的 Gemm: {node.name}")

        weight_name = node.input[1]
        if attributes.get("transB", 0) != 0:
            initializer = initializers.get(weight_name)
            if initializer is None:
                raise ValueError(f"Gemm 权重不是常量 initializer: {node.name}")
            weight_name = f"{weight_name}.transposed"
            transposed = numpy_helper.from_array(
                numpy_helper.to_array(initializer).T.copy(), name=weight_name
            )
            model.graph.initializer.append(transposed)
            initializers[weight_name] = transposed

        matmul_output = node.output[0] if len(node.input) < 3 else f"{node.output[0]}.matmul"
        nodes.append(
            helper.make_node(
                "MatMul",
                [node.input[0], weight_name],
                [matmul_output],
                name=f"{node.name}.matmul",
            )
        )
        if len(node.input) >= 3:
            nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_output, node.input[2]],
                    list(node.output),
                    name=f"{node.name}.bias",
                )
            )
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    used_initializers = {name for item in nodes for name in item.input}
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_initializers
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained_initializers)


def main() -> None:
    """main 导出 FP32 ONNX，并按需生成和验证 INT8 模型。"""

    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    model, _, config = load_encoder(artifact_dir, torch.device("cpu"))
    torch.backends.mha.set_fastpath_enabled(False)
    input_ids = torch.zeros((2, config.max_length), dtype=torch.long)
    attention_mask = torch.zeros((2, config.max_length), dtype=torch.long)
    input_ids[:, :4] = torch.tensor([1, 4, 5, 2], dtype=torch.long)
    attention_mask[:, :4] = 1
    batch = torch.export.Dim("batch", min=1, max=8)

    fp32_path = artifact_dir / "model.onnx"
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["tool_logits", "slot_logits", "abstain_logits"],
        dynamic_shapes={
            "input_ids": {0: batch},
            "attention_mask": {0: batch},
        },
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        optimize=False,
    )
    onnx_model = onnx.load(fp32_path)
    decompose_gemm(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, fp32_path)
    fp32_shapes = validate_runtime(fp32_path, config.max_length)

    exported_path = fp32_path
    int8_shapes = None
    if args.quantize == "int8":
        int8_path = artifact_dir / "model.int8.onnx"
        preprocessed_path = artifact_dir / "model.preprocessed.onnx"
        quant_pre_process(
            input_model=fp32_path,
            output_model_path=preprocessed_path,
            auto_merge=True,
            skip_symbolic_shape=True,
        )
        quantize_dynamic(
            model_input=str(preprocessed_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        preprocessed_path.unlink()
        onnx.checker.check_model(onnx.load(int8_path))
        int8_shapes = validate_runtime(int8_path, config.max_length)
        if int8_path.stat().st_size >= MAX_MODEL_BYTES:
            raise ValueError(
                f"INT8 模型超过 40 MB 目标: {int8_path.stat().st_size / 1024 / 1024:.2f} MB"
            )
        exported_path = int8_path

    training_metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "dataset_version": training_metadata["dataset_version"],
        "schema_version": training_metadata["schema_version"],
        "schema_sha256": training_metadata["schema_sha256"],
        "source_model_sha256": sha256_file(artifact_dir / "model.safetensors"),
        "opset": args.opset,
        "max_length": config.max_length,
        "quantization": args.quantize,
        "fp32": {
            "path": str(fp32_path),
            "bytes": fp32_path.stat().st_size,
            "sha256": sha256_file(fp32_path),
            "output_shapes": fp32_shapes,
        },
        "exported": {
            "path": str(exported_path),
            "bytes": exported_path.stat().st_size,
            "sha256": sha256_file(exported_path),
            "output_shapes": int8_shapes or fp32_shapes,
        },
    }
    write_json(artifact_dir / "export-metadata.json", metadata)
    print(
        f"dataset={metadata['dataset_version']} schema={metadata['schema_version']} "
        f"model={exported_path} bytes={metadata['exported']['bytes']} "
        f"sha256={metadata['exported']['sha256']}"
    )


if __name__ == "__main__":
    main()
