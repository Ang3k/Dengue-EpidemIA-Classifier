from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dengue_pipeline.datasets import (  # noqa: E402
    load_ml_years,
    split_features_target,
)
from dengue_pipeline.features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
)
from dengue_pipeline.paths import (  # noqa: E402
    ENSEMBLE_CONFIG_PATH,
    EXPECTED_SPLIT_ROWS,
    MODEL_FIGURES_DIR,
    MODEL_MANIFEST_PATH,
    MODELS_DIR,
    TEST_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
)


METRICS_DIR = PROJECT_ROOT / "reports" / "metrics" / "modeling"
MODEL_FILES = {
    "mlp": "mlp.joblib",
    "xgboost": "xgboost.joblib",
    "lightgbm": "lightgbm.joblib",
}
EVALUATION_FIGURES_DIR = MODEL_FIGURES_DIR / "evaluation"


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(features))
    return values[:, 1] if values.ndim == 2 else values.reshape(-1)


def threshold_metrics(
    model_name: str,
    split: str,
    target: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype("int8")
        tn, fp, fn, tp = confusion_matrix(
            target,
            predictions,
            labels=[0, 1],
        ).ravel()
        rows.append(
            {
                "model": model_name,
                "split": split,
                "threshold": float(threshold),
                "accuracy": accuracy_score(target, predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    target,
                    predictions,
                ),
                "precision": precision_score(
                    target,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    target,
                    predictions,
                    zero_division=0,
                ),
                "specificity": tn / max(tn + fp, 1),
                "f1": f1_score(target, predictions, zero_division=0),
                "predicted_positive_rate": predictions.mean(),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(metrics: pd.DataFrame) -> pd.Series:
    # Youden J = sensibilidade + especificidade - 1, que é monotônico com a
    # balanced accuracy. Escolhe um ponto de operação equilibrado em vez de
    # maximizar F1 (que, sob a prevalência de 2020, empurrava o limiar pra
    # baixo e fazia o modelo prever "confirmado" pra quase todo mundo).
    return metrics.sort_values(
        ["balanced_accuracy", "f1"],
        ascending=False,
    ).iloc[0]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(manifest: dict) -> dict:
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        entry = manifest.get("models", {}).get(name, {})
        if entry.get("file") != filename:
            raise RuntimeError(f"{name} filename differs from model manifest")
        if not path.exists() or entry.get("sha256") != file_sha256(path):
            raise RuntimeError(f"{name} SHA-256 differs from model manifest")
        loaded[name] = joblib.load(path)
    return loaded


def save_evaluation_figures(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    confusion_rows: pd.DataFrame,
    curve_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    EVALUATION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    comparison = test_metrics.set_index("model")[metric_names]
    ax = comparison.plot.bar(figsize=(11, 6), ylim=(0, 1))
    ax.set(title="Métricas no teste final de 2021", xlabel="", ylabel="Valor")
    ax.legend(loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(
        EVALUATION_FIGURES_DIR / "model_metrics_comparison.png",
        dpi=160,
    )
    plt.close()

    selected = validation_metrics[validation_metrics["selected"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in validation_metrics.groupby("model"):
        ax.plot(group["threshold"], group["f1"], label=name)
    ax.scatter(
        selected["threshold"],
        selected["f1"],
        color="black",
        zorder=5,
        label="selecionado em 2020",
    )
    ax.set(
        title="Seleção de limiar exclusivamente na validação de 2020",
        xlabel="Limiar",
        ylabel="F1",
        ylim=(0, 1),
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        EVALUATION_FIGURES_DIR / "threshold_analysis.png",
        dpi=160,
    )
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for axis, name in zip(axes.ravel(), curve_scores):
        matrix = (
            confusion_rows[confusion_rows["model"] == name]
            .pivot(index="actual", columns="predicted", values="count")
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            .to_numpy()
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set(title=name, xlabel="Predito", ylabel="Real")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                )
    plt.tight_layout()
    plt.savefig(
        EVALUATION_FIGURES_DIR / "confusion_matrices.png",
        dpi=160,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, scores in curve_scores.items():
        false_positive, true_positive, _ = roc_curve(y_test, scores)
        ax.plot(false_positive, true_positive, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(
        title="Curvas ROC no teste final de 2021",
        xlabel="Taxa de falsos positivos",
        ylabel="Taxa de verdadeiros positivos",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(EVALUATION_FIGURES_DIR / "roc_curves.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, scores in curve_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=name)
    ax.set(
        title="Curvas precisão-recall no teste final de 2021",
        xlabel="Recall",
        ylabel="Precisão",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        EVALUATION_FIGURES_DIR / "precision_recall_curves.png",
        dpi=160,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate on 2020 and evaluate once on 2021."
    )
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--ensemble-threshold",
        type=float,
        default=None,
        help=(
            "Fixa o limiar do ensemble (ponto de operação de deploy) em vez de "
            "selecioná-lo por balanced accuracy. Ex.: 0.2 para alta sensibilidade."
        ),
    )
    args = parser.parse_args()
    if args.ensemble_threshold is not None and not 0 < args.ensemble_threshold < 1:
        parser.error("--ensemble-threshold deve estar entre 0 e 1")

    manifest = json.loads(
        MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if (
        manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or manifest.get("feature_columns") != list(MODEL_FEATURE_COLUMNS)
        or manifest.get("periods")
        != {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": list(TEST_YEARS),
        }
        or manifest.get("row_counts") != EXPECTED_SPLIT_ROWS
    ):
        raise RuntimeError("Model manifest feature schema is incompatible")

    validation_dataset = load_ml_years(VALIDATION_YEARS)
    if len(validation_dataset) != EXPECTED_SPLIT_ROWS["validation"]:
        raise RuntimeError("Validation dataset row count is not official")
    X_validation, y_validation = split_features_target(validation_dataset)
    thresholds = np.arange(
        args.threshold_step,
        0.951,
        args.threshold_step,
    ).round(4)

    models = load_models(manifest)
    validation_scores = {
        name: positive_probability(model, X_validation)
        for name, model in models.items()
    }
    validation_frames = []
    selected_rows = []
    recalls = {}
    selected_thresholds = {}
    for name in MODEL_FILES:
        metrics = threshold_metrics(
            name,
            "validation",
            y_validation.to_numpy(),
            validation_scores[name],
            thresholds,
        )
        selected = select_threshold(metrics)
        metrics["selected"] = np.isclose(
            metrics["threshold"],
            selected["threshold"],
        )
        validation_frames.append(metrics)
        selected_thresholds[name] = float(selected["threshold"])
        recalls[name] = float(selected["recall"])
        selected_rows.append(
            {
                "model": name,
                "selection_split": "validation",
                "rule": "max_balanced_accuracy",
                **selected.to_dict(),
            }
        )

    recall_total = sum(recalls.values())
    weights = {
        name: recall / recall_total
        for name, recall in recalls.items()
    }
    ensemble_validation = sum(
        validation_scores[name] * weights[name]
        for name in weights
    )
    ensemble_validation_metrics = threshold_metrics(
        "ensemble",
        "validation",
        y_validation.to_numpy(),
        ensemble_validation,
        thresholds,
    )
    ensemble_selected = select_threshold(ensemble_validation_metrics)
    if args.ensemble_threshold is not None:
        ensemble_threshold = round(float(args.ensemble_threshold), 4)
        threshold_rule = "manual_operating_point"
    else:
        ensemble_threshold = float(ensemble_selected["threshold"])
        threshold_rule = "max_balanced_accuracy"
    ensemble_validation_metrics["selected"] = np.isclose(
        ensemble_validation_metrics["threshold"],
        ensemble_threshold,
    )
    validation_frames.append(ensemble_validation_metrics)
    selected_rows.append(
        {
            "model": "ensemble",
            "selection_split": "validation",
            "rule": threshold_rule,
            **ensemble_selected.to_dict(),
            "threshold": ensemble_threshold,
        }
    )

    ensemble_config = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selection_period": list(VALIDATION_YEARS),
        "test_period": list(TEST_YEARS),
        "threshold_rule": threshold_rule,
        "weight_rule": "normalized_validation_recall",
        "threshold": ensemble_threshold,
        "weights": weights,
        "model_manifest_sha256": file_sha256(MODEL_MANIFEST_PATH),
    }

    # Only after every decision is frozen do we open the final 2021 test set.
    del validation_dataset, X_validation, y_validation
    del validation_scores, ensemble_validation
    gc.collect()
    test_dataset = load_ml_years(TEST_YEARS)
    if len(test_dataset) != EXPECTED_SPLIT_ROWS["test"]:
        raise RuntimeError("Test dataset row count is not official")
    X_test, y_test = split_features_target(test_dataset)
    test_scores = {
        name: positive_probability(model, X_test)
        for name, model in models.items()
    }
    ensemble_test = sum(
        test_scores[name] * weights[name]
        for name in weights
    )
    test_metric_rows = []
    confusion_rows = []
    classification_rows = []
    for name, scores in {
        **test_scores,
        "ensemble": ensemble_test,
    }.items():
        threshold = (
            ensemble_threshold
            if name == "ensemble"
            else selected_thresholds[name]
        )
        metric = threshold_metrics(
            name,
            "test",
            y_test.to_numpy(),
            scores,
            np.asarray([threshold]),
        ).iloc[0].to_dict()
        metric.update(
            {
                "roc_auc": roc_auc_score(y_test, scores),
                "pr_auc": average_precision_score(y_test, scores),
                "weight": weights.get(name, 1.0),
            }
        )
        test_metric_rows.append(metric)
        predictions = (scores >= threshold).astype("int8")
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["discarded", "confirmed"],
            output_dict=True,
            zero_division=0,
        )
        for label, values in report.items():
            if isinstance(values, dict):
                classification_rows.append(
                    {"model": name, "label": label, **values}
                )
        for actual, predicted, count in (
            (0, 0, metric["tn"]),
            (0, 1, metric["fp"]),
            (1, 0, metric["fn"]),
            (1, 1, metric["tp"]),
        ):
            confusion_rows.append(
                {
                    "model": name,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(count),
                }
            )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(validation_frames, ignore_index=True).to_csv(
        METRICS_DIR / "threshold_metrics.csv",
        index=False,
    )
    pd.DataFrame(selected_rows).to_csv(
        METRICS_DIR / "selected_thresholds.csv",
        index=False,
    )
    test_metrics_frame = pd.DataFrame(test_metric_rows)
    test_metrics_frame.to_csv(
        METRICS_DIR / "model_metrics.csv",
        index=False,
    )
    confusion_frame = pd.DataFrame(confusion_rows)
    confusion_frame.to_csv(
        METRICS_DIR / "confusion_matrices.csv",
        index=False,
    )
    pd.DataFrame(classification_rows).to_csv(
        METRICS_DIR / "classification_report.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "period_start": "2021-01",
                "period_end": "2021-12",
                "n_samples": len(y_test),
                "n_features": X_test.shape[1],
                "n_confirmed": int(y_test.sum()),
                "n_discarded": int((y_test == 0).sum()),
                "positive_rate": float(y_test.mean()),
            }
        ]
    ).to_csv(METRICS_DIR / "test_set_summary.csv", index=False)

    temporary = ENSEMBLE_CONFIG_PATH.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(ensemble_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ENSEMBLE_CONFIG_PATH)
    save_evaluation_figures(
        pd.concat(validation_frames, ignore_index=True),
        test_metrics_frame,
        confusion_frame,
        {**test_scores, "ensemble": ensemble_test},
        y_test.to_numpy(),
    )
    print(test_metrics_frame.to_string(index=False))
    print(f"Ensemble configuration written to {ENSEMBLE_CONFIG_PATH}")


if __name__ == "__main__":
    main()
