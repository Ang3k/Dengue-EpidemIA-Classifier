from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    DATASET_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SYMPTOM_COLUMNS,
    build_ml_dataset,
)
from .paths import PROJECT_ROOT
from .sinan_mappings import (
    BINARY_CLASSIFICATION_LABELS,
    EDUCATION_LABELS,
    PREGNANCY_LABELS,
    RACE_LABELS,
    REVERSE_CBO_MAP,
    SEX_LABELS,
    UF_LABELS,
    standardize_columns,
)


LEGACY_POSITIVE_CLASSIFICATIONS = frozenset({1, 2, 3, 4, 10, 11, 12})
MODERN_POSITIVE_CLASSIFICATIONS = frozenset({10, 11, 12})
NEGATIVE_CLASSIFICATIONS = frozenset({5})
IGNORED_CLASSIFICATIONS = frozenset({0, 8, 9})

REQUIRED_STANDARDIZED_COLUMNS = frozenset(
    {
        "age",
        "sex",
        "pregnancy_status",
        "race",
        "education_level",
        "occupation_code",
        "residence_state",
        "residence_municipality",
        "residence_health_region",
        "disease_code",
        "notification_date",
        "notification_year",
        "notification_epi_week",
        "notif_municipality",
        "notif_health_region",
        "health_facility",
        "symptom_onset_date",
        "symptom_epi_week",
        *SYMPTOM_COLUMNS,
        "hospitalized",
        "hospital_state",
        "final_classification",
    }
)

ANALYSIS_COLUMNS = (
    "source_year",
    "age_years",
    "sex",
    "sex_label",
    "pregnancy_status",
    "pregnancy_status_label",
    "race",
    "race_label",
    "education_level",
    "education_level_label",
    "occupation_code",
    "occupation_name",
    "residence_state",
    "residence_state_label",
    "residence_municipality",
    "residence_health_region",
    "disease_code",
    "notification_date",
    "notification_year",
    "notification_month",
    "notification_day",
    "notification_epi_week",
    "notif_municipality",
    "notif_health_region",
    "health_facility",
    "symptom_onset_date",
    "days_to_notification",
    "symptom_epi_year",
    "symptom_epi_week_number",
    *SYMPTOM_COLUMNS,
    "hospitalized",
    "hospital_state",
    "hospital_state_label",
    "final_classification_code",
    "final_classification",
    "final_classification_label",
)


def positive_classifications_for_year(year: int) -> frozenset[int]:
    if 2014 <= year <= 2016:
        return LEGACY_POSITIVE_CLASSIFICATIONS
    if 2017 <= year <= 2021:
        return MODERN_POSITIVE_CLASSIFICATIONS
    raise ValueError(f"Unsupported dengue source year: {year}")


def harmonize_final_classification(
    values: pd.Series,
    source_year: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    positives = positive_classifications_for_year(source_year)
    result = pd.Series(pd.NA, index=values.index, dtype="Int8")
    result.loc[numeric.isin(NEGATIVE_CLASSIFICATIONS)] = 0
    result.loc[numeric.isin(positives)] = 1
    return result


def classification_counts(values: pd.Series) -> dict[str, int]:
    normalized = values.astype("string").fillna("").str.strip()
    normalized = normalized.replace({"<NA>": "", "nan": ""})
    return {
        str(code): int(count)
        for code, count in normalized.value_counts(dropna=False).items()
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _map_numeric(values: pd.Series, mapping: dict) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").map(mapping)


def _parse_age_years(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    units = np.floor(numeric / 1000)
    amounts = numeric.mod(1000)
    years = pd.Series(np.nan, index=values.index, dtype="float64")
    years.loc[units.eq(4)] = amounts.loc[units.eq(4)]
    years.loc[units.eq(3)] = amounts.loc[units.eq(3)] / 12
    years.loc[units.eq(2)] = amounts.loc[units.eq(2)] / 365
    years.loc[units.eq(1)] = amounts.loc[units.eq(1)] / 8760
    direct_age = numeric.between(0, 130) & ~units.isin([1, 2, 3, 4])
    years.loc[direct_age] = numeric.loc[direct_age]
    return years.where(years.between(0, 130))


class DengueDataCleaner:
    """Stateless, chunk-oriented transformation for SINAN dengue extracts."""

    def __init__(self, arquivos=None):
        if arquivos is None:
            self.arquivos: tuple[Path, ...] = ()
        elif isinstance(arquivos, (str, Path)):
            self.arquivos = (Path(arquivos),)
        else:
            self.arquivos = tuple(Path(path) for path in arquivos)

    @staticmethod
    def validate_raw_schema(frame: pd.DataFrame) -> None:
        standardized = {
            standardize_columns(pd.DataFrame(columns=[column])).columns[0]
            for column in frame.columns
        }
        missing = REQUIRED_STANDARDIZED_COLUMNS - standardized
        if missing:
            raise ValueError(
                "Required SINAN columns are missing: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def transformar_analise_chunk(
        raw_frame: pd.DataFrame,
        source_year: int,
    ) -> pd.DataFrame:
        positive_classifications_for_year(source_year)
        raw_frame = raw_frame.copy()
        raw_frame.columns = raw_frame.columns.astype(str).str.upper()
        frame = standardize_columns(raw_frame)

        missing = REQUIRED_STANDARDIZED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"Year {source_year} is missing required columns: "
                f"{sorted(missing)}"
            )

        target_code = _numeric(frame, "final_classification")
        target = harmonize_final_classification(
            frame["final_classification"],
            source_year,
        )
        valid = target.notna()
        frame = frame.loc[valid].copy()
        target_code = target_code.loc[valid]
        target = target.loc[valid].astype("int8")

        notification_date = pd.to_datetime(
            frame["notification_date"],
            errors="coerce",
        )
        symptom_onset_date = pd.to_datetime(
            frame["symptom_onset_date"],
            errors="coerce",
        )
        symptom_epi_week = _numeric(frame, "symptom_epi_week")

        output = pd.DataFrame(index=frame.index)
        output["source_year"] = np.int16(source_year)
        output["age_years"] = _parse_age_years(frame["age"])
        output["sex"] = frame["sex"].astype("string").str.upper()
        output["sex_label"] = output["sex"].map(SEX_LABELS)

        for column in (
            "pregnancy_status",
            "race",
            "education_level",
            "occupation_code",
            "residence_state",
            "residence_municipality",
            "residence_health_region",
            "notification_epi_week",
            "notif_municipality",
            "notif_health_region",
            "health_facility",
            "hospitalized",
            "hospital_state",
        ):
            output[column] = _numeric(frame, column)

        output["pregnancy_status_label"] = _map_numeric(
            frame["pregnancy_status"],
            PREGNANCY_LABELS,
        )
        output["race_label"] = _map_numeric(frame["race"], RACE_LABELS)
        output["education_level_label"] = _map_numeric(
            frame["education_level"],
            EDUCATION_LABELS,
        )
        output["occupation_name"] = _numeric(
            frame,
            "occupation_code",
        ).map(REVERSE_CBO_MAP)
        output["residence_state_label"] = _map_numeric(
            frame["residence_state"],
            UF_LABELS,
        )

        output["disease_code"] = frame["disease_code"].astype("string")
        output["notification_date"] = notification_date
        output["notification_year"] = np.int16(source_year)
        output["notification_month"] = notification_date.dt.month.astype(
            "Int8"
        )
        output["notification_day"] = notification_date.dt.day.astype("Int8")
        output["symptom_onset_date"] = symptom_onset_date
        output["days_to_notification"] = (
            notification_date - symptom_onset_date
        ).dt.days
        output["symptom_epi_year"] = (
            symptom_epi_week.floordiv(100).astype("Int16")
        )
        output["symptom_epi_week_number"] = (
            symptom_epi_week.mod(100).astype("Int8")
        )

        for symptom in SYMPTOM_COLUMNS:
            output[symptom] = _numeric(frame, symptom).astype("Float32")

        output["hospital_state_label"] = _map_numeric(
            frame["hospital_state"],
            UF_LABELS,
        )
        output["final_classification_code"] = target_code.astype("int8")
        output["final_classification"] = target
        output["final_classification_label"] = target.map(
            BINARY_CLASSIFICATION_LABELS
        )

        return output.loc[:, ANALYSIS_COLUMNS].reset_index(drop=True)

    @staticmethod
    def transformar_ml(
        analysis_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        return build_ml_dataset(analysis_frame)

    def carregar(self, arquivo: str | Path | None = None) -> pd.DataFrame:
        path = Path(arquivo) if arquivo is not None else None
        if path is None:
            if len(self.arquivos) != 1:
                raise ValueError(
                    "carregar() requires one explicit file in the "
                    "chunk-oriented pipeline"
                )
            path = self.arquivos[0]

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        return pd.read_parquet(path)

    def transformar_analise(
        self,
        raw_frame: pd.DataFrame | None = None,
        source_year: int | None = None,
    ) -> pd.DataFrame:
        if raw_frame is None:
            raw_frame = self.carregar()
        if source_year is None:
            years = pd.to_numeric(
                standardize_columns(raw_frame)["notification_year"],
                errors="coerce",
            ).dropna().unique()
            if len(years) != 1:
                raise ValueError("source_year is required for mixed-year data")
            source_year = int(years[0])
        return self.transformar_analise_chunk(raw_frame, source_year)

    @staticmethod
    def salvar_df(df: pd.DataFrame, caminho_saida: str | Path) -> None:
        path = Path(caminho_saida)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


__all__ = [
    "ANALYSIS_COLUMNS",
    "DATASET_METADATA_COLUMNS",
    "DengueDataCleaner",
    "IGNORED_CLASSIFICATIONS",
    "MODEL_FEATURE_COLUMNS",
    "NEGATIVE_CLASSIFICATIONS",
    "classification_counts",
    "harmonize_final_classification",
    "positive_classifications_for_year",
]
