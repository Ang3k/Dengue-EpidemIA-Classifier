from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .features import (
    DATASET_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from .paths import (
    TEST_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
    ml_dataset_path,
)


def load_ml_years(years: Iterable[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = ml_dataset_path(int(year))
        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found for {year}: {path}. "
                "Run scripts/prepare_dengue_data.py first."
            )
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("At least one dataset year is required")
    return pd.concat(frames, ignore_index=True)


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = (
        set(DATASET_METADATA_COLUMNS)
        | set(MODEL_FEATURE_COLUMNS)
    ) - set(dataset.columns)
    if missing:
        raise ValueError(f"Processed dataset columns missing: {sorted(missing)}")

    features = dataset.loc[:, MODEL_FEATURE_COLUMNS].astype("float32")
    target = dataset["final_classification"].astype("int8")
    return features, target


def load_temporal_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": load_ml_years(TRAIN_YEARS),
        "validation": load_ml_years(VALIDATION_YEARS),
        "test": load_ml_years(TEST_YEARS),
    }


__all__ = [
    "load_ml_years",
    "load_temporal_splits",
    "split_features_target",
]
