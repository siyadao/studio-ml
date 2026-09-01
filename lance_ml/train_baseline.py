"""训练字符 n-gram 工具路由基线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .action_schema import DATA_DIR, ML_ROOT, load_action_schema
from .dataset import (
    calculate_metrics,
    load_jsonl,
    load_manifest,
    model_text,
    sha256_file,
    write_json,
)


DEFAULT_ARTIFACT_DIR = ML_ROOT / "artifacts" / "baseline"
DEFAULT_SEED = 20260901


def parse_args() -> argparse.Namespace:
    """parse_args 解析基线训练参数。"""

    parser = argparse.ArgumentParser(description="训练字符 n-gram 工具路由基线")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--calibration-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """main 训练、校准并保存基线与验证指标。"""

    args = parse_args()
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    schema = load_action_schema(data_dir / "actions.json")
    train_records = load_jsonl(data_dir / "train.jsonl")
    validation_records = load_jsonl(data_dir / "validation.jsonl")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    train_features = vectorizer.fit_transform(model_text(record) for record in train_records)
    base_classifier = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=args.seed,
        solver="lbfgs",
    )
    classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        method="sigmoid",
        cv=args.calibration_folds,
        n_jobs=-1,
    )
    classifier.fit(train_features, [record["tool"] for record in train_records])

    validation_features = vectorizer.transform(
        model_text(record) for record in validation_records
    )
    probabilities = classifier.predict_proba(validation_features)
    labels = [str(label) for label in classifier.classes_]
    risk_by_label = {action.id: action.risk for action in schema.actions}
    risk_by_label["abstain"] = "read_only"
    metrics = calculate_metrics(
        validation_records, labels, probabilities, risk_by_label
    )

    model_path = artifact_dir / "baseline.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": labels,
            "risk_by_label": risk_by_label,
        },
        model_path,
        compress=3,
    )
    metadata = {
        "model_type": "char_ngram_logistic_regression",
        "dataset_version": manifest["dataset_version"],
        "schema_version": schema.schema_version,
        "schema_sha256": schema.sha256,
        "random_seed": args.seed,
        "training_parameters": {
            "analyzer": "char",
            "ngram_range": [2, 5],
            "min_df": 2,
            "max_features": args.max_features,
            "classifier": "logistic_regression",
            "classifier_c": 4.0,
            "class_weight": "balanced",
            "calibration": "sigmoid",
            "calibration_folds": args.calibration_folds,
        },
        "best_checkpoint": str(model_path),
        "validation_metrics": metrics,
        "labels": labels,
        "model_sha256": sha256_file(model_path),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    print(
        f"dataset={manifest['dataset_version']} schema={schema.schema_version} seed={args.seed} "
        f"checkpoint={model_path} top1={metrics['top1_accuracy']:.4f} "
        f"top3={metrics['top3_accuracy']:.4f} sha256={metadata['model_sha256']}"
    )


if __name__ == "__main__":
    main()
