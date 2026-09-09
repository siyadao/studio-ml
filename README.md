# studio-ml

思芽岛智能工具路由模型的独立训练工程。这里维护动作 Schema 快照、脱敏数据集、字符 n-gram 基线、Encoder 训练与 ONNX 导出；Rust 推理 Worker 和随应用发布的模型仍由 [`siyadao/lance-studio`](https://github.com/siyadao/lance-studio) 维护。

## 环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

## 数据与 Schema

仓库内的 `data/actions.json` 是从主应用权威动作注册表导出的版本化快照。普通数据生成只读取并严格校验该快照：

```bash
uv run python -m lance_ml.generate_dataset
```

动作注册表变化后，显式指定本机 `lance-studio` 仓库来刷新快照并重新生成数据：

```bash
uv run python -m lance_ml.generate_dataset \
  --refresh-actions \
  --studio-root ../lance-studio
```

## 基线模型

```bash
uv run python -m lance_ml.train_baseline
uv run python -m lance_ml.evaluate --model baseline
uv run python -m lance_ml.export_baseline_runtime
```

默认导出到 `artifacts/runtime/smart-input-baseline.bin`。要更新主应用内的模型，必须显式指定输出路径：

```bash
uv run python -m lance_ml.export_baseline_runtime \
  --output ../lance-studio/assets/models/smart-input-baseline.bin
```

与真实 Rust Worker 对照前，先在主应用仓库构建 Worker，再传入它的路径：

```bash
(cd ../lance-studio && cargo build --locked --bin siyadao-model-worker)
uv run python -m lance_ml.verify_runtime \
  --worker ../lance-studio/target/debug/siyadao-model-worker
```

## Encoder

```bash
uv run python -m lance_ml.train --device auto
uv run python -m lance_ml.evaluate --model encoder
uv run python -m lance_ml.export_onnx --quantize int8
```

训练产物、检查点和本地缓存位于 `artifacts/`、`checkpoints/` 等忽略目录，不进入 Git。动作协议、生成器和脱敏数据集纳入版本控制。
