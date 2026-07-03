from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from threading import Event, Thread
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil


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
from dengue_pipeline.models import (  # noqa: E402
    GradientBoostingDiseaseClassifier,
    MLPDiseaseClassifier,
)
from dengue_pipeline.paths import (  # noqa: E402
    EXPECTED_SPLIT_ROWS,
    MODEL_FIGURES_DIR,
    MODEL_MANIFEST_PATH,
    MODELS_DIR,
    TRAIN_YEARS,
    VALIDATION_YEARS,
)


DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "dengue_manifest.json"
MAX_TRAINING_RSS_GIB = 28.0


class PeakMemoryMonitor:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = Event()
        self.peak_rss_bytes = self._process.memory_info().rss
        self._thread = Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self.peak_rss_bytes / (1024**3)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                self._process.memory_info().rss,
            )
            time.sleep(0.25)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_joblib_dump(model, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    joblib.dump(model, temporary)
    temporary.replace(destination)


def save_feature_importance(name: str, importance) -> None:
    MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = importance.head(30).sort_values()
    ax = top.plot.barh(figsize=(10, 8))
    ax.set(
        title=f"{name.upper()} — 30 features mais importantes (2020)",
        xlabel="Importância",
        ylabel="",
    )
    plt.tight_layout()
    plt.savefig(
        MODEL_FIGURES_DIR / f"{name}_feature_importance.png",
        dpi=160,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dengue models with the 2014-2020 temporal split."
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--tuning-sample-size", type=int, default=100_000)
    parser.add_argument(
        "--mlp-hidden",
        type=str,
        default="1024,512,256,128",
        help="Larguras das camadas ocultas da MLP, separadas por vírgula.",
    )
    parser.add_argument("--mlp-learning-rate", type=float, default=5e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=16_384)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device da MLP e do XGBoost (cuda/cpu).",
    )
    parser.add_argument(
        "--lgbm-device",
        type=str,
        default="cpu",
        help=(
            "Device do LightGBM (cpu/gpu). O GPU do LGBM (OpenCL) nem sempre é "
            "mais rápido que a CPU e limita max_bin a 255; teste antes."
        ),
    )
    args = parser.parse_args()
    mlp_hidden = tuple(int(size) for size in args.mlp_hidden.split(",") if size)

    memory_monitor = PeakMemoryMonitor()
    memory_monitor.start()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading training years:", TRAIN_YEARS, flush=True)
    train_dataset = load_ml_years(TRAIN_YEARS)
    print("Loading validation years:", VALIDATION_YEARS, flush=True)
    validation_dataset = load_ml_years(VALIDATION_YEARS)

    X_train, y_train = split_features_target(train_dataset)
    X_validation, y_validation = split_features_target(validation_dataset)
    if X_train["local_density"].isna().all() or X_train["local_positivity"].isna().all():
        raise RuntimeError(
            "local_density/local_positivity estão vazias. Rode "
            "scripts/augment_local_density.py depois de prepare_dengue_data.py "
            "e antes de treinar."
        )
    train_rows = len(train_dataset)
    validation_rows = len(validation_dataset)
    if train_rows != EXPECTED_SPLIT_ROWS["train"]:
        raise RuntimeError(
            f"Training row count mismatch: {train_rows:,} != "
            f"{EXPECTED_SPLIT_ROWS['train']:,}"
        )
    if validation_rows != EXPECTED_SPLIT_ROWS["validation"]:
        raise RuntimeError(
            f"Validation row count mismatch: {validation_rows:,} != "
            f"{EXPECTED_SPLIT_ROWS['validation']:,}"
        )
    del train_dataset, validation_dataset
    gc.collect()

    models = {
        "mlp": MLPDiseaseClassifier(
            hidden_layers=mlp_hidden,
            embedding_dropout=0.1,
            hidden_dropout=args.mlp_dropout,
            batch_size=args.mlp_batch_size,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            max_epochs=args.max_epochs,
            patience=args.mlp_patience,
            device=args.device,
            random_state=42,
        ),
        "xgboost": GradientBoostingDiseaseClassifier(
            model="xgb",
            fast_train=False,
            device=args.device,
        ),
        "lightgbm": GradientBoostingDiseaseClassifier(
            model="lgbm",
            fast_train=False,
            device=args.lgbm_device,
        ),
    }

    models["mlp"].fit(
        X_train,
        y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )
    atomic_joblib_dump(models["mlp"], MODELS_DIR / "mlp.joblib")
    save_feature_importance(
        "mlp",
        models["mlp"].permutation_feature_importance(
            X_validation,
            y_validation,
            sample_size=2_000,
            n_repeats=3,
        ),
    )

    for name in ("xgboost", "lightgbm"):
        models[name].fit(
            X_train,
            y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            n_trials=args.n_trials,
            tuning_sample_size=args.tuning_sample_size,
            tuning_validation_size=args.tuning_sample_size,
        )
        atomic_joblib_dump(models[name], MODELS_DIR / f"{name}.joblib")
        save_feature_importance(name, models[name].feature_importance())

    peak_rss_gib = memory_monitor.stop()
    if peak_rss_gib >= MAX_TRAINING_RSS_GIB:
        raise MemoryError(
            f"Training used {peak_rss_gib:.2f} GiB RSS; "
            f"limit is {MAX_TRAINING_RSS_GIB:.0f} GiB"
        )

    manifest = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "periods": {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": [2021],
        },
        "row_counts": {
            "train": train_rows,
            "validation": validation_rows,
            "test": EXPECTED_SPLIT_ROWS["test"],
        },
        "peak_training_rss_gib": round(peak_rss_gib, 3),
        "data_manifest_sha256": file_sha256(DATA_MANIFEST_PATH),
        "models": {
            name: {
                "file": f"{name}.joblib",
                "sha256": file_sha256(MODELS_DIR / f"{name}.joblib"),
            }
            for name in models
        },
    }
    temporary = MODEL_MANIFEST_PATH.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(MODEL_MANIFEST_PATH)
    print(f"Model manifest written to {MODEL_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
