from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "2.2.0"

# Janela (em semanas epidemiológicas) usada para medir a circulação local
# recente de dengue. Só olha o passado (semanas anteriores) para não vazar.
LOCAL_DENSITY_WINDOW_WEEKS = 4

SYMPTOM_COLUMNS = (
    "fever",
    "myalgia",
    "headache",
    "rash",
    "vomiting",
    "nausea",
    "back_pain",
    "conjunctivitis",
    "arthritis",
    "joint_pain",
    "petechiae",
    "retro_orbital_pain",
)

IMPORTANT_SYMPTOMS = ("rash", "retro_orbital_pain")

INTERACTION_COLUMNS = tuple(
    f"{symptom_a}_and_{symptom_b}"
    for symptom_a, symptom_b in combinations(SYMPTOM_COLUMNS, 2)
)

SEX_FEATURES = ("sex_Female", "sex_Ignored", "sex_Male")
RACE_CODES = (1, 2, 3, 4, 5, 9)
RACE_FEATURES = tuple(f"race_{code}" for code in RACE_CODES)

BASE_FEATURE_COLUMNS = (
    "age_years",
    "education_level",
    "occupation_code",
    "residence_state",
    "residence_municipality",
    "residence_health_region",
    "days_to_notification",
    *SYMPTOM_COLUMNS,
    *SEX_FEATURES,
    *RACE_FEATURES,
    "notification_month_sin",
    "notification_month_cos",
    "symptom_epi_week_number_sin",
    "symptom_epi_week_number_cos",
    "symptom_month_sin",
    "symptom_month_cos",
)

MODEL_FEATURE_COLUMNS = (
    *BASE_FEATURE_COLUMNS,
    *INTERACTION_COLUMNS,
    "number_of_symptoms",
    "number_of_important_symptoms",
    "number_of_reported_symptoms",
    "pregnancy",
    "pregnancy_informed",
    "local_density",
    "local_positivity",
)

DATASET_METADATA_COLUMNS = (
    "notification_year",
    "notification_month",
    "final_classification",
)

EDUCATION_ORDINAL_MAP = {
    0: 1,
    1: 2,
    2: 2,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 5,
    8: 5,
    9: 0,
    10: 0,
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _date(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce")


def _normalized_symptom(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame, column)
    result = pd.Series(np.nan, index=frame.index, dtype="float32")
    result.loc[values.eq(1).fillna(False)] = 1.0
    result.loc[values.isin([0, 2]).fillna(False)] = 0.0
    return result


def compute_local_density(frame: pd.DataFrame) -> pd.Series:
    """Circulação local recente: notificações do mesmo município nas semanas
    epidemiológicas anteriores (janela de ``LOCAL_DENSITY_WINDOW_WEEKS``).

    Leakage-safe: usa apenas contagens de notificações (não rótulos) e apenas
    de semanas passadas, dentro do mesmo ano. Vetorizado. Retorna log1p da
    contagem para comprimir a faixa dinâmica.
    """
    year = _numeric(frame, "notification_year").astype("float64")
    municipality = _numeric(frame, "residence_municipality").astype("float64")
    # notification_epi_week vem como código AAAASS (ex.: 202005). A semana do
    # ano (1-53) é o resto por 100.
    week = (_numeric(frame, "notification_epi_week") % 100).astype("float64")

    keys = pd.DataFrame({"y": year, "m": municipality, "w": week})
    valid = keys.notna().all(axis=1)
    counts = (
        keys[valid]
        .groupby(["y", "m", "w"])
        .size()
        .rename("n")
        .reset_index()
    )
    if counts.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float32")

    weeks = np.arange(1.0, 54.0)
    grid = counts.pivot_table(
        index=["y", "m"],
        columns="w",
        values="n",
        fill_value=0.0,
    ).reindex(columns=weeks, fill_value=0.0)
    prior = sum(
        grid.shift(k, axis=1, fill_value=0.0)
        for k in range(1, LOCAL_DENSITY_WINDOW_WEEKS + 1)
    )
    prior_long = prior.stack()
    prior_long.index = prior_long.index.set_names(["y", "m", "w"])

    lookup = pd.MultiIndex.from_arrays(
        [year.to_numpy(), municipality.to_numpy(), week.to_numpy()],
        names=["y", "m", "w"],
    )
    density = prior_long.reindex(lookup).to_numpy()
    return pd.Series(np.log1p(density), index=frame.index, dtype="float32")


def build_local_density_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabela de serving: densidade típica por (município, semana), média entre
    os anos disponíveis. A API usa isto para uma notificação isolada, já que a
    densidade não é derivável de uma única linha.
    """
    density = compute_local_density(frame)
    table = pd.DataFrame(
        {
            "residence_municipality": _numeric(
                frame, "residence_municipality"
            ),
            "epi_week_of_year": _numeric(frame, "notification_epi_week") % 100,
            "local_density": density,
        }
    ).dropna()
    return (
        table.groupby(["residence_municipality", "epi_week_of_year"])[
            "local_density"
        ]
        .mean()
        .reset_index()
    )


def compute_local_positivity(frame: pd.DataFrame) -> pd.Series:
    """Taxa de confirmação local recente: dos casos do mesmo município nas
    semanas anteriores, a fração confirmada (confirmados / rotulados).

    Leakage-safe: usa apenas semanas passadas, dentro do mesmo ano. A
    confirmação do próprio caso nunca entra. NaN quando não há casos rotulados
    anteriores.
    """
    year = _numeric(frame, "notification_year").astype("float64")
    municipality = _numeric(frame, "residence_municipality").astype("float64")
    week = (_numeric(frame, "notification_epi_week") % 100).astype("float64")
    code = _numeric(frame, "final_classification_code")
    confirmed = code.isin([10, 11, 12]).astype("float64")
    labeled = code.isin([10, 11, 12, 5]).astype("float64")

    keys = pd.DataFrame(
        {"y": year, "m": municipality, "w": week, "c": confirmed, "n": labeled}
    )
    valid = keys[["y", "m", "w"]].notna().all(axis=1)
    agg = (
        keys[valid]
        .groupby(["y", "m", "w"])
        .agg(c=("c", "sum"), n=("n", "sum"))
    )
    if agg.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float32")

    weeks = np.arange(1.0, 54.0)
    confirmed_grid = agg["c"].unstack(fill_value=0.0).reindex(
        columns=weeks, fill_value=0.0
    )
    labeled_grid = agg["n"].unstack(fill_value=0.0).reindex(
        columns=weeks, fill_value=0.0
    )
    confirmed_prior = sum(
        confirmed_grid.shift(k, axis=1, fill_value=0.0)
        for k in range(1, LOCAL_DENSITY_WINDOW_WEEKS + 1)
    )
    labeled_prior = sum(
        labeled_grid.shift(k, axis=1, fill_value=0.0)
        for k in range(1, LOCAL_DENSITY_WINDOW_WEEKS + 1)
    )
    rate = confirmed_prior / labeled_prior.where(labeled_prior > 0)
    rate_long = rate.stack()
    rate_long.index = rate_long.index.set_names(["y", "m", "w"])

    lookup = pd.MultiIndex.from_arrays(
        [year.to_numpy(), municipality.to_numpy(), week.to_numpy()],
        names=["y", "m", "w"],
    )
    return pd.Series(
        rate_long.reindex(lookup).to_numpy(),
        index=frame.index,
        dtype="float32",
    )


def build_local_positivity_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabela de serving: taxa de confirmação típica por (município, semana),
    média entre os anos disponíveis.
    """
    positivity = compute_local_positivity(frame)
    table = pd.DataFrame(
        {
            "residence_municipality": _numeric(
                frame, "residence_municipality"
            ),
            "epi_week_of_year": _numeric(frame, "notification_epi_week") % 100,
            "local_positivity": positivity,
        }
    ).dropna()
    return (
        table.groupby(["residence_municipality", "epi_week_of_year"])[
            "local_positivity"
        ]
        .mean()
        .reset_index()
    )


def build_model_features(
    frame: pd.DataFrame,
    local_density: pd.Series | np.ndarray | None = None,
    local_positivity: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the exact model matrix used by training and online inference.

    This transformation is deliberately stateless. Any learned preprocessing
    (categorical vocabularies and numerical medians) belongs to each fitted
    model and must only be learned from the training period.

    ``local_density`` é injetada de fora (calculada offline sobre o ano inteiro
    no treino, ou consultada num lookup no serving), pois depende de contexto
    de grupo e não pode ser derivada de uma linha isolada. Se ausente, fica NaN
    e os modelos de árvore lidam nativamente; a MLP imputa pela mediana.
    """

    data: dict[str, pd.Series] = {}

    data["age_years"] = _numeric(frame, "age_years")
    education = _numeric(frame, "education_level")
    data["education_level"] = education.map(EDUCATION_ORDINAL_MAP)

    for column in (
        "occupation_code",
        "residence_state",
        "residence_municipality",
        "residence_health_region",
    ):
        data[column] = _numeric(frame, column)

    notification_date = _date(frame, "notification_date")
    symptom_onset_date = _date(frame, "symptom_onset_date")

    notification_month = _numeric(frame, "notification_month")
    notification_month = notification_month.fillna(notification_date.dt.month)

    symptom_week = _numeric(frame, "symptom_epi_week_number")
    if "symptom_epi_week" in frame:
        raw_week = _numeric(frame, "symptom_epi_week")
        symptom_week = symptom_week.fillna(raw_week.mod(100))

    symptom_month = symptom_onset_date.dt.month.astype("float64")

    days = _numeric(frame, "days_to_notification")
    calculated_days = (notification_date - symptom_onset_date).dt.days
    days = days.fillna(calculated_days)
    data["days_to_notification"] = days.where(days.between(0, 90))

    symptoms = pd.DataFrame(
        {
            column: _normalized_symptom(frame, column)
            for column in SYMPTOM_COLUMNS
        },
        index=frame.index,
    )
    data.update(
        {column: symptoms[column] for column in SYMPTOM_COLUMNS}
    )

    sex = (
        frame["sex"].astype("string").str.upper()
        if "sex" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    data["sex_Female"] = sex.eq("F").astype("float32")
    data["sex_Ignored"] = (~sex.isin(["F", "M"])).astype("float32")
    data["sex_Male"] = sex.eq("M").astype("float32")

    race = _numeric(frame, "race")
    for code in RACE_CODES:
        data[f"race_{code}"] = race.eq(code).astype("float32")

    data["notification_month_sin"] = np.sin(
        2 * np.pi * notification_month / 12
    )
    data["notification_month_cos"] = np.cos(
        2 * np.pi * notification_month / 12
    )
    data["symptom_epi_week_number_sin"] = np.sin(
        2 * np.pi * symptom_week / 53
    )
    data["symptom_epi_week_number_cos"] = np.cos(
        2 * np.pi * symptom_week / 53
    )
    data["symptom_month_sin"] = np.sin(2 * np.pi * symptom_month / 12)
    data["symptom_month_cos"] = np.cos(2 * np.pi * symptom_month / 12)

    for symptom_a, symptom_b in combinations(SYMPTOM_COLUMNS, 2):
        data[f"{symptom_a}_and_{symptom_b}"] = (
            symptoms[symptom_a] * symptoms[symptom_b]
        )

    data["number_of_symptoms"] = symptoms.fillna(0).sum(axis=1)
    data["number_of_important_symptoms"] = (
        symptoms[list(IMPORTANT_SYMPTOMS)].fillna(0).sum(axis=1)
    )
    data["number_of_reported_symptoms"] = symptoms.notna().sum(axis=1)

    pregnancy_status = _numeric(frame, "pregnancy_status")
    data["pregnancy"] = pregnancy_status.isin([1, 2, 3, 4]).astype(
        "float32"
    )
    data["pregnancy_informed"] = pregnancy_status.isin(
        [1, 2, 3, 4, 5]
    ).astype("float32")

    def _injected(values, name):
        if values is None:
            return np.full(len(frame), np.nan, dtype="float32")
        array = np.asarray(values, dtype="float32").reshape(-1)
        if len(array) != len(frame):
            raise ValueError(f"{name} length must match the number of rows")
        return array

    data["local_density"] = pd.Series(
        _injected(local_density, "local_density"),
        index=frame.index,
        dtype="float32",
    )
    data["local_positivity"] = pd.Series(
        _injected(local_positivity, "local_positivity"),
        index=frame.index,
        dtype="float32",
    )

    features = pd.DataFrame(data, index=frame.index)
    missing = set(MODEL_FEATURE_COLUMNS) - set(features.columns)
    extra = set(features.columns) - set(MODEL_FEATURE_COLUMNS)
    if missing or extra:
        raise RuntimeError(
            "Feature schema mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    return features.loc[:, MODEL_FEATURE_COLUMNS].astype("float32")


def build_ml_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(DATASET_METADATA_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset metadata columns missing: {sorted(missing)}")

    metadata = frame.loc[:, DATASET_METADATA_COLUMNS].copy()
    metadata["notification_year"] = pd.to_numeric(
        metadata["notification_year"], errors="raise"
    ).astype("int16")
    metadata["notification_month"] = pd.to_numeric(
        metadata["notification_month"], errors="raise"
    ).astype("int8")
    metadata["final_classification"] = pd.to_numeric(
        metadata["final_classification"], errors="raise"
    ).astype("int8")
    return pd.concat([metadata, build_model_features(frame)], axis=1)
