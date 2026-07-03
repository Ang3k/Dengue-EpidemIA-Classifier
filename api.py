"""
API de predição de dengue.

Rode com:
    .venv\Scripts\python -m uvicorn api:app --reload

Treino e inferência usam o mesmo construtor de features sem estado. Todo
pré-processamento aprendido fica dentro do artefato de cada modelo e é ajustado
somente nos anos de treino.
"""

from datetime import date, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dengue_pipeline.features import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    SYMPTOM_COLUMNS,
    build_model_features,
    compute_local_density,
    compute_local_positivity,
)
from dengue_pipeline.paths import (
    ENSEMBLE_CONFIG_PATH,
    EXPECTED_SPLIT_ROWS,
    LOCAL_DENSITY_LOOKUP_PATH,
    LOCAL_POSITIVITY_LOOKUP_PATH,
    MODEL_MANIFEST_PATH,
    SIMULATION_POOL_PATH,
    SIMULATION_SOURCE_PARQUET,
    TEST_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
)
from dengue_pipeline.sinan_mappings import (
    DENGUE_CLASSIFICATION_LABELS,
    EDUCATION_LABELS,
    PREGNANCY_LABELS,
    SEX_LABELS,
    RACE_LABELS,
    UF_ABBR_LABELS,
    UF_LABELS,
)
from dengue_pipeline.cbo_map import CBO_MAP
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carregar modelos e o pré-processamento salvos
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "artifacts" / "models"

MODELOS_DISPONIVEIS = {
    "mlp":                 "mlp.joblib",
    "xgboost":             "xgboost.joblib",
    "lightgbm":            "lightgbm.joblib",
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        logger.exception("Não foi possível carregar %s", path)
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_manifest = _load_json(MODEL_MANIFEST_PATH)
ensemble_config = _load_json(ENSEMBLE_CONFIG_PATH)
try:
    ENSEMBLE_WEIGHTS = {
        str(name): float(weight)
        for name, weight in ensemble_config.get("weights", {}).items()
    }
    ENSEMBLE_THRESHOLD = float(ensemble_config.get("threshold", 0.5))
    ensemble_values_valid = (
        set(ENSEMBLE_WEIGHTS) == set(MODELOS_DISPONIVEIS)
        and all(
            np.isfinite(weight) and weight > 0
            for weight in ENSEMBLE_WEIGHTS.values()
        )
        and np.isfinite(ENSEMBLE_THRESHOLD)
        and 0 <= ENSEMBLE_THRESHOLD <= 1
    )
except (AttributeError, TypeError, ValueError):
    ENSEMBLE_WEIGHTS = {}
    ENSEMBLE_THRESHOLD = 0.5
    ensemble_values_valid = False

model_manifest_compatible = (
    model_manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
    and model_manifest.get("feature_columns") == list(MODEL_FEATURE_COLUMNS)
    and model_manifest.get("periods", {}).get("train") == list(TRAIN_YEARS)
    and model_manifest.get("periods", {}).get("validation")
    == list(VALIDATION_YEARS)
    and model_manifest.get("periods", {}).get("test") == list(TEST_YEARS)
    and model_manifest.get("row_counts") == EXPECTED_SPLIT_ROWS
)
ensemble_config_compatible = (
    ensemble_config.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
    and ensemble_config.get("selection_period") == [2020]
    and ensemble_config.get("test_period") == [2021]
    and set(ensemble_config.get("weights", {})) == set(MODELOS_DISPONIVEIS)
    and ensemble_values_valid
    and MODEL_MANIFEST_PATH.exists()
    and ensemble_config.get("model_manifest_sha256")
    == _sha256_file(MODEL_MANIFEST_PATH)
)
artifact_set_compatible = (
    model_manifest_compatible and ensemble_config_compatible
)

modelos = {}
erros_carregamento = {}
for nome, arquivo in MODELOS_DISPONIVEIS.items():
    caminho = MODELS_DIR / arquivo
    if not artifact_set_compatible:
        erros_carregamento[nome] = (
            "manifesto ausente ou incompatível com o esquema de features "
            f"{FEATURE_SCHEMA_VERSION}"
        )
        continue
    if not caminho.exists():
        erros_carregamento[nome] = f"arquivo não encontrado: {caminho}"
        logger.warning("Modelo %s não encontrado em %s", nome, caminho)
        continue

    try:
        model_entry = model_manifest.get("models", {}).get(nome, {})
        if model_entry.get("file") != arquivo:
            raise ValueError("arquivo difere do manifesto do modelo")
        if model_entry.get("sha256") != _sha256_file(caminho):
            raise ValueError("SHA-256 difere do manifesto do modelo")
        modelo = joblib.load(caminho)
        if nome == "xgboost":
            modelo_interno = getattr(modelo, "model", None)
            if hasattr(modelo_interno, "set_params"):
                modelo_interno.set_params(
                    device=os.getenv("XGBOOST_DEVICE", "cpu")
                )
        elif nome == "mlp" and hasattr(modelo, "device"):
            modelo.device = os.getenv("MLP_DEVICE", "auto")
        feature_names = list(
            getattr(modelo, "feature_names", None)
            or getattr(modelo, "feature_names_in_", [])
        )
        if feature_names != list(MODEL_FEATURE_COLUMNS):
            raise ValueError(
                "features do modelo diferem de MODEL_FEATURE_COLUMNS"
            )
        modelos[nome] = modelo
        logger.info("Modelo %s carregado", nome)
    except Exception as exc:
        erros_carregamento[nome] = str(exc)
        logger.exception("Não foi possível carregar o modelo %s", nome)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="API Dengue", version="1.0.0")

origens_cors = [
    origem.strip()
    for origem in os.getenv(
        "DENGUE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origem.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origens_cors,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schema de entrada — campos que o frontend já envia
# ---------------------------------------------------------------------------

def _semana_epidemiologica(dia: date) -> int:
    """Semana epidemiológica (convenção MMWR/SINAN: semana começa no domingo,
    a semana 1 é a que contém a maioria — a quarta-feira — no ano novo).

    Validado contra o SINAN real (100% de concordância). Necessário porque a
    densidade/positividade locais dependem da semana da notificação, e o
    frontend envia a data, não a semana.
    """
    wday = (dia.weekday() + 1) % 7  # domingo = 0
    quarta = dia + timedelta(days=(3 - wday))  # quarta-feira da semana epi
    ano = quarta.year
    jan1 = date(ano, 1, 1)
    jan1_wday = (jan1.weekday() + 1) % 7
    primeira_quarta = jan1 + timedelta(days=(3 - jan1_wday) % 7)
    return (quarta - primeira_quarta).days // 7 + 1


class DadosPaciente(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Paciente
    age_years: float | None = Field(default=None, ge=0, le=130)
    sex: Literal["M", "F", "I"] | None = None
    pregnancy_status: Literal[1, 2, 3, 4, 5, 6, 9] | None = None
    race: Literal[1, 2, 3, 4, 5, 9] | None = None
    education_level: int | None = Field(default=None, ge=0, le=10)
    occupation_code: str | None = Field(
        default=None,
        pattern=r"^(?:0|\d{5,6})$",
    )

    # Residência
    residence_state: int | None = None
    residence_municipality: int | None = Field(default=None, ge=0)
    residence_health_region: int | None = Field(default=None, ge=0)

    # Notificação / datas
    notification_date: date | None = None
    notification_year: int | None = Field(default=None, ge=1900, le=2100)
    notification_month: int | None = Field(default=None, ge=1, le=12)
    notification_epi_week: int | None = Field(default=None, ge=1)
    notif_municipality: int | None = Field(default=None, ge=0)
    notif_health_region: int | None = Field(default=None, ge=0)
    health_facility: int | None = Field(default=None, ge=0)

    # Início dos sintomas
    symptom_onset_date: date | None = None
    days_to_notification: float | None = Field(default=None, ge=0, le=90)
    symptom_epi_year: int | None = Field(default=None, ge=1900, le=2100)
    symptom_epi_week_number: int | None = Field(default=None, ge=1, le=53)

    # Sintomas (1 = sim, 0 = não)
    fever: int | None = Field(default=None, ge=0, le=1)
    myalgia: int | None = Field(default=None, ge=0, le=1)
    headache: int | None = Field(default=None, ge=0, le=1)
    rash: int | None = Field(default=None, ge=0, le=1)
    vomiting: int | None = Field(default=None, ge=0, le=1)
    nausea: int | None = Field(default=None, ge=0, le=1)
    back_pain: int | None = Field(default=None, ge=0, le=1)
    conjunctivitis: int | None = Field(default=None, ge=0, le=1)
    arthritis: int | None = Field(default=None, ge=0, le=1)
    joint_pain: int | None = Field(default=None, ge=0, le=1)
    petechiae: int | None = Field(default=None, ge=0, le=1)
    retro_orbital_pain: int | None = Field(default=None, ge=0, le=1)

    # Hospitalização
    hospitalized: Literal[1, 2, 9] | None = None
    hospital_state: int | None = None

    @field_validator("residence_state", "hospital_state")
    @classmethod
    def validar_uf(cls, valor):
        ufs = {
            11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53,
        }
        if valor is not None and valor not in ufs:
            raise ValueError("use um código IBGE de UF válido")
        return valor

    @model_validator(mode="after")
    def validar_datas(self):
        if (
            self.notification_date
            and self.symptom_onset_date
            and self.notification_date < self.symptom_onset_date
        ):
            raise ValueError(
                "a notificação não pode ser anterior ao início dos sintomas"
            )
        return self

    @model_validator(mode="after")
    def derivar_semana_epi(self):
        # A semana da notificação alimenta o lookup de densidade/positividade
        # local. Se o cliente não a enviou mas mandou a data, deriva-a — senão
        # as duas features mais fortes do modelo ficariam NaN.
        if self.notification_epi_week is None and self.notification_date is not None:
            self.notification_epi_week = _semana_epidemiologica(
                self.notification_date
            )
        if self.notification_year is None and self.notification_date is not None:
            self.notification_year = self.notification_date.year
        return self


class SimulacaoRandomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Pré-processamento — replica o transformar_ml() do cleaner
# ---------------------------------------------------------------------------

SINTOMAS = list(SYMPTOM_COLUMNS)

# Código SINAN da escolaridade (0-10) -> ordinal usado no treino.
# Vem da composição EDUCATION_LABELS (código -> texto) com o map_escolaridade do
# cleaner (texto -> 0..5). Ex.: 0 = Analfabeto -> 1; 9/10 = Ignorado/NA -> 0.
SIMULATION_YEAR = 2021
SIMULATION_NOTIFICATION_MONTH_MIN = 1
SIMULATION_VALID_CLASSIFICATIONS = frozenset(DENGUE_CLASSIFICATION_LABELS)
MAX_SAMPLE_ATTEMPTS = 10

SIMULATION_SOURCE_COLUMNS = (
    "age_years",
    "sex",
    "pregnancy_status",
    "race",
    "education_level",
    "occupation_code",
    "residence_state",
    "residence_municipality",
    "residence_health_region",
    "notification_date",
    "notification_year",
    "notification_epi_week",
    "notif_municipality",
    "notif_health_region",
    "health_facility",
    "symptom_onset_date",
    "days_to_notification",
    "symptom_epi_year",
    "symptom_epi_week_number",
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
    "hospitalized",
    "hospital_state",
    "final_classification_code",
)
SIMULATION_DERIVED_COLUMNS = (
    "occupation_name",
    "final_classification_label",
)
# Features de contexto epidemiológico calculadas sobre 2021 e guardadas no pool,
# para a simulação usar o valor EXATO que o modelo viu no teste (não o lookup).
SIMULATION_CONTEXT_COLUMNS = (
    "local_density",
    "local_positivity",
)
SIMULATION_POOL_COLUMNS = (
    *SIMULATION_SOURCE_COLUMNS,
    *SIMULATION_DERIVED_COLUMNS,
    *SIMULATION_CONTEXT_COLUMNS,
)

SIMULATION_SYMPTOM_LABELS = {
    "fever": "Febre",
    "myalgia": "Mialgia",
    "headache": "Cefaleia",
    "rash": "Exantema",
    "vomiting": "Vomito",
    "nausea": "Nausea",
    "back_pain": "Dor nas costas",
    "conjunctivitis": "Conjuntivite",
    "arthritis": "Artrite",
    "joint_pain": "Dor nas articulacoes",
    "petechiae": "Petequias",
    "retro_orbital_pain": "Dor retro-orbital",
}

_simulation_pool: pd.DataFrame | None = None
_simulation_pool_lock = Lock()


def _carregar_lookup(path: Path, coluna: str) -> dict[tuple[int, int], float]:
    """Lê um artefato (município, semana-do-ano) -> valor típico.

    Features de contexto epidemiológico não são deriváveis de uma notificação
    isolada, então são consultadas aqui. Ausência do artefato ou da chave ->
    NaN, e os modelos de árvore lidam nativamente.
    """
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
        return {
            (int(row.residence_municipality), int(row.epi_week_of_year)): float(
                getattr(row, coluna)
            )
            for row in frame.itertuples(index=False)
        }
    except Exception:
        logger.exception("Não foi possível carregar %s", path)
        return {}


LOCAL_DENSITY_LOOKUP = _carregar_lookup(LOCAL_DENSITY_LOOKUP_PATH, "local_density")
LOCAL_POSITIVITY_LOOKUP = _carregar_lookup(
    LOCAL_POSITIVITY_LOOKUP_PATH, "local_positivity"
)


def _consultar_local(
    dados: DadosPaciente, lookup: dict[tuple[int, int], float]
) -> list[float] | None:
    municipio = dados.residence_municipality
    semana = dados.notification_epi_week
    if municipio is None or semana is None:
        return None
    valor = lookup.get((int(municipio), int(semana) % 100))
    return None if valor is None else [valor]


def construir_features(
    dados: DadosPaciente,
    local_density: list[float] | None = None,
    local_positivity: list[float] | None = None,
) -> pd.DataFrame:
    """Build one inference row with the shared training feature schema.

    Se ``local_density``/``local_positivity`` forem passados (caso histórico da
    simulação, com o valor exato de 2021), usa-os; senão consulta o lookup
    (paciente novo do /predict).
    """
    if local_density is None:
        local_density = _consultar_local(dados, LOCAL_DENSITY_LOOKUP)
    if local_positivity is None:
        local_positivity = _consultar_local(dados, LOCAL_POSITIVITY_LOOKUP)
    return build_model_features(
        pd.DataFrame([dados.model_dump()]),
        local_density=local_density,
        local_positivity=local_positivity,
    )


def _colunas_esperadas(modelo):
    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)
    nomes = getattr(modelo, "feature_names", None)
    return list(nomes) if nomes else None


def alinhar_colunas(df: pd.DataFrame, modelo):
    """Alinha as colunas com o que o modelo espera. Se faltar alguma coluna que
    o modelo precisa, devolve None + a lista de faltantes (em vez de preencher
    com 0 em silêncio), para o /predict pular esse modelo e avisar."""
    esperadas = _colunas_esperadas(modelo)
    if esperadas is None:
        return df, []
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        return None, faltantes
    return df[esperadas].astype("float32"), []


def _inferir_modelos(df: pd.DataFrame):
    """Executa a inferência em todos os modelos carregados para um vetor de
    features já construído."""
    resultados = []
    ignorados = []
    probabilidades = {}

    for nome, modelo in modelos.items():
        df_alinhado, faltantes = alinhar_colunas(df.copy(), modelo)
        if df_alinhado is None:
            ignorados.append(
                {
                    "name": nome,
                    "reason": "features ausentes",
                    "missing": faltantes,
                }
            )
            continue

        try:
            proba = np.asarray(modelo.predict_proba(df_alinhado))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                prob = float(proba[0, 1])
            elif proba.size:
                prob = float(proba.reshape(-1)[0])
            else:
                raise ValueError("predict_proba retornou um array vazio")
            if not np.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f"probabilidade inválida: {prob}")
            probabilidades[nome] = prob
            resultados.append(
                {
                    "name": nome,
                    "probability": round(prob * 100, 1),
                }
            )
        except Exception as exc:
            logger.exception("Falha ao executar o modelo %s", nome)
            ignorados.append(
                {
                    "name": nome,
                    "reason": str(exc),
                }
            )

    if not resultados:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Nenhum modelo conseguiu gerar uma predição",
                "ignored": ignorados,
            },
        )

    pesos_disponiveis = {
        name: ENSEMBLE_WEIGHTS[name]
        for name in probabilidades
        if name in ENSEMBLE_WEIGHTS
    }
    if set(pesos_disponiveis) != set(probabilidades):
        raise HTTPException(
            status_code=503,
            detail="A configuração do ensemble não corresponde aos modelos carregados.",
        )

    total_pesos = sum(pesos_disponiveis.values())
    pesos_normalizados = {
        name: weight / total_pesos
        for name, weight in pesos_disponiveis.items()
    }
    for resultado in resultados:
        resultado["weight"] = round(
            pesos_normalizados[resultado["name"]] * 100,
            1,
        )

    score_ponderado = float(
        sum(
            probabilidades[name] * pesos_normalizados[name]
            for name in probabilidades
        )
    )
    score_percentual = round(score_ponderado * 100, 1)
    threshold_percentual = round(float(ENSEMBLE_THRESHOLD) * 100, 1)
    return {
        "models": resultados,
        "average": score_percentual,
        "threshold": threshold_percentual,
        "weighting": "recall",
        "isDengue": score_ponderado >= float(ENSEMBLE_THRESHOLD),
        "ignored": ignorados,
    }


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _one_of(value: Any, allowed: set[int]) -> int | None:
    parsed = _to_int(value)
    if parsed in allowed:
        return parsed
    return None


def _parse_age_years(encoded_age: Any) -> float | None:
    age = _to_int(encoded_age)
    if age is None:
        return None

    # SINAN usa unidade + quantidade no formato UYYY (ex.: 4025 = 25 anos).
    text = str(age).zfill(4)
    unit = int(text[0])
    value = int(text[1:])

    if unit == 4:
        years = float(value)
    elif unit == 3:
        years = float(value) / 12
    elif unit == 2:
        years = float(value) / 365
    elif unit == 1:
        years = float(value) / 8760
    else:
        years = float(age) if 0 <= age <= 130 else None

    if years is None or years < 0 or years > 130:
        return None
    return years


def _to_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _to_occupation_code(value: Any) -> str | None:
    code = _to_int(value)
    if code is None or code < 0:
        return None
    code_text = str(code)
    return code_text if code == 0 or len(code_text) in (5, 6) else None


def _flag_from_sinan(value: Any) -> int | None:
    parsed = _to_int(value)
    if parsed == 1:
        return 1
    if parsed in {0, 2}:
        return 0
    return None


def _build_patient_from_sample(row: pd.Series) -> DadosPaciente:
    notification_date = _to_date(row.get("notification_date"))
    symptom_onset_date = _to_date(row.get("symptom_onset_date"))

    return DadosPaciente(
        age_years=(
            float(row.get("age_years"))
            if pd.notna(row.get("age_years"))
            else None
        ),
        sex=row.get("sex") if row.get("sex") in {"M", "F", "I"} else None,
        pregnancy_status=_one_of(row.get("pregnancy_status"), {1, 2, 3, 4, 5, 6, 9}),
        race=_one_of(row.get("race"), {1, 2, 3, 4, 5, 9}),
        education_level=_to_int(row.get("education_level")),
        occupation_code=_to_occupation_code(row.get("occupation_code")),
        residence_state=_to_int(row.get("residence_state")),
        residence_municipality=_to_int(row.get("residence_municipality")),
        residence_health_region=_to_int(row.get("residence_health_region")),
        notification_date=notification_date,
        notification_year=(notification_date.year if notification_date else None),
        notification_month=(notification_date.month if notification_date else None),
        notification_epi_week=_to_int(row.get("notification_epi_week")),
        notif_municipality=_to_int(row.get("notif_municipality")),
        notif_health_region=_to_int(row.get("notif_health_region")),
        health_facility=_to_int(row.get("health_facility")),
        symptom_onset_date=symptom_onset_date,
        days_to_notification=_to_int(row.get("days_to_notification")),
        symptom_epi_year=_to_int(row.get("symptom_epi_year")),
        symptom_epi_week_number=_to_int(
            row.get("symptom_epi_week_number")
        ),
        fever=_flag_from_sinan(row.get("fever")),
        myalgia=_flag_from_sinan(row.get("myalgia")),
        headache=_flag_from_sinan(row.get("headache")),
        rash=_flag_from_sinan(row.get("rash")),
        vomiting=_flag_from_sinan(row.get("vomiting")),
        nausea=_flag_from_sinan(row.get("nausea")),
        back_pain=_flag_from_sinan(row.get("back_pain")),
        conjunctivitis=_flag_from_sinan(row.get("conjunctivitis")),
        arthritis=_flag_from_sinan(row.get("arthritis")),
        joint_pain=_flag_from_sinan(row.get("joint_pain")),
        petechiae=_flag_from_sinan(row.get("petechiae")),
        retro_orbital_pain=_flag_from_sinan(row.get("retro_orbital_pain")),
        hospitalized=_one_of(row.get("hospitalized"), {1, 2, 9}),
        hospital_state=_to_int(row.get("hospital_state")),
    )


def _anonymized_case_from_sample(row: pd.Series, patient: DadosPaciente) -> dict[str, Any]:
    symptom_labels = [
        label
        for key, label in SIMULATION_SYMPTOM_LABELS.items()
        if getattr(patient, key, 0) == 1
    ]

    age = int(round(patient.age_years)) if patient.age_years is not None else None
    state_code = _to_int(row.get("residence_state"))
    municipality_code = _to_int(row.get("residence_municipality"))
    occupation_name = row.get("occupation_name")

    return {
        "age": age,
        "sex": SEX_LABELS.get(patient.sex),
        "race": RACE_LABELS.get(patient.race),
        "occupation": None if pd.isna(occupation_name) else str(occupation_name),
        "state": UF_ABBR_LABELS.get(state_code),
        "municipality": str(municipality_code) if municipality_code is not None else None,
        "symptoms": symptom_labels,
    }


def _simulation_pool_is_valid(pool: pd.DataFrame) -> bool:
    if pool.empty or not set(SIMULATION_POOL_COLUMNS).issubset(pool.columns):
        return False

    dates = pd.to_datetime(pool["notification_date"], errors="coerce")
    years = pd.to_numeric(pool["notification_year"], errors="coerce")
    classifications = pd.to_numeric(
        pool["final_classification_code"],
        errors="coerce",
    )

    return bool(
        dates.notna().all()
        and (years == SIMULATION_YEAR).all()
        and (dates.dt.month >= SIMULATION_NOTIFICATION_MONTH_MIN).all()
        and classifications.isin(SIMULATION_VALID_CLASSIFICATIONS).all()
        and pool["final_classification_label"].notna().all()
    )


def _build_simulation_pool() -> pd.DataFrame:
    """Build the public simulation pool from the untouched 2021 test set."""
    if not SIMULATION_SOURCE_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Base processada de 2021 não encontrada: "
                f"{SIMULATION_SOURCE_PARQUET}"
            ),
        )

    try:
        source = pd.read_parquet(
            SIMULATION_SOURCE_PARQUET,
            columns=list(
                SIMULATION_SOURCE_COLUMNS + SIMULATION_DERIVED_COLUMNS
            ),
        )
        # Contexto epidemiológico calculado sobre TODAS as notificações de 2021
        # (antes de filtrar), para bater exatamente com o dataset de treino/teste.
        source["local_density"] = compute_local_density(source).to_numpy()
        source["local_positivity"] = compute_local_positivity(source).to_numpy()
        dates = pd.to_datetime(source["notification_date"], errors="coerce")
        years = pd.to_numeric(
            source["notification_year"],
            errors="coerce",
        )
        classifications = pd.to_numeric(
            source["final_classification_code"],
            errors="coerce",
        )
        eligible = (
            years.eq(SIMULATION_YEAR)
            & dates.dt.month.ge(SIMULATION_NOTIFICATION_MONTH_MIN)
            & classifications.isin(SIMULATION_VALID_CLASSIFICATIONS)
        )
        filtered = source.loc[
            eligible,
            list(SIMULATION_POOL_COLUMNS),
        ].reset_index(drop=True)
        if not _simulation_pool_is_valid(filtered):
            raise ValueError("pool reduzido da simulação ficou inválido")

        SIMULATION_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(SIMULATION_POOL_PATH, index=False)
        return filtered
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Não foi possível gerar o pool da simulação")
        raise HTTPException(
            status_code=503,
            detail="Não foi possível preparar a base histórica da simulação",
        ) from exc


def _load_simulation_pool() -> pd.DataFrame:
    global _simulation_pool
    if _simulation_pool is not None:
        return _simulation_pool

    with _simulation_pool_lock:
        if _simulation_pool is not None:
            return _simulation_pool

        pool = None
        source_is_newer = (
            SIMULATION_POOL_PATH.exists()
            and SIMULATION_SOURCE_PARQUET.exists()
            and SIMULATION_SOURCE_PARQUET.stat().st_mtime
            > SIMULATION_POOL_PATH.stat().st_mtime
        )

        if SIMULATION_POOL_PATH.exists() and not source_is_newer:
            try:
                candidate = pd.read_parquet(
                    SIMULATION_POOL_PATH,
                    columns=list(SIMULATION_POOL_COLUMNS),
                )
                if _simulation_pool_is_valid(candidate):
                    pool = candidate
                else:
                    logger.warning(
                        "Pool da simulacao existente e invalido; regenerando"
                    )
            except Exception:
                logger.exception(
                    "Nao foi possivel carregar %s; regenerando",
                    SIMULATION_POOL_PATH,
                )

        if pool is None:
            pool = _build_simulation_pool()

        _simulation_pool = pool.reset_index(drop=True)
        return _simulation_pool


def escolher_caso_real_simulacao(seed: int | None = None) -> dict[str, Any]:
    pool = _load_simulation_pool()
    rng = np.random.default_rng(seed)

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        sampled_idx = int(rng.integers(0, len(pool)))
        row = pool.iloc[sampled_idx]

        try:
            classification = _to_int(
                row.get("final_classification_code")
            )
            observed = DENGUE_CLASSIFICATION_LABELS.get(classification)
            if observed is None:
                raise ValueError("classificacao observada invalida")

            patient = _build_patient_from_sample(row)
            anonymized_case = _anonymized_case_from_sample(row, patient)
        except (ValidationError, ValueError, TypeError, OverflowError) as exc:
            logger.warning(
                "Caso historico rejeitado no indice %s: %s",
                sampled_idx,
                exc,
            )
            continue

        return {
            "sampled_index": sampled_idx,
            "patient": patient,
            "case": anonymized_case,
            "observed_classification": observed,
            "local_density": row.get("local_density"),
            "local_positivity": row.get("local_positivity"),
        }

    raise HTTPException(
        status_code=503,
        detail="Nao foi possivel selecionar um caso historico valido",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    faltantes = [
        nome for nome in MODELOS_DISPONIVEIS if nome not in modelos
    ]
    ensemble_ready = (
        set(ENSEMBLE_WEIGHTS) == set(MODELOS_DISPONIVEIS)
        and 0 <= ENSEMBLE_THRESHOLD <= 1
    )
    return {
        "status": (
            "ok"
            if not faltantes and artifact_set_compatible and ensemble_ready
            else "degraded"
        ),
        "modelos_carregados": list(modelos.keys()),
        "modelos_ausentes": faltantes,
        "erros_carregamento": erros_carregamento,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "artefatos_compativeis": artifact_set_compatible,
        "periodos": model_manifest.get("periods", {}),
        "ensemble_threshold": ENSEMBLE_THRESHOLD,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }


@app.post("/predict")
def predict(dados: DadosPaciente):
    if not modelos:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo foi carregado",
        )
    df = construir_features(dados)

    return _inferir_modelos(df)


@app.post("/api/v1/simulations/random")
def simulation_random(payload: SimulacaoRandomRequest | None = None):
    if not modelos:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo foi carregado",
        )
    ausentes = [nome for nome in MODELOS_DISPONIVEIS if nome not in modelos]
    if ausentes:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Nem todos os modelos necessários foram carregados",
                "missing": ausentes,
            },
        )

    sample = escolher_caso_real_simulacao(seed=(payload.seed if payload else None))
    # Caso histórico: usa a densidade/positividade EXATAS de 2021 (as que o modelo
    # viu no teste), não a média sazonal do lookup.
    features = construir_features(
        sample["patient"],
        local_density=[float(sample["local_density"])],
        local_positivity=[float(sample["local_positivity"])],
    )
    prediction = _inferir_modelos(features)

    if prediction["ignored"]:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Nem todos os modelos conseguiram gerar predição",
                "ignored": prediction["ignored"],
            },
        )

    return {
        "case": sample["case"],
        "observedClassification": sample["observed_classification"],
        "prediction": {
            "models": prediction["models"],
            "average": prediction["average"],
            "threshold": prediction["threshold"],
            "weighting": prediction["weighting"],
            "isDengue": prediction["isDengue"],
        },
    }

# ===========================================================================
# NOVOS ENDPOINTS — Pessoa 2
# ===========================================================================

import unicodedata as _unicodedata
from fastapi import Query as _Query

# ---------------------------------------------------------------------------
# Dados de referência — municípios
# ---------------------------------------------------------------------------

def _norm(texto: str) -> str:
    return (
        _unicodedata.normalize("NFKD", texto.lower())
        .encode("ascii", "ignore")
        .decode()
    )

def _carregar_municipios_ref() -> list[dict]:
    import json
    caminho = Path(__file__).parent / "data" / "municipios.json"
    if not caminho.exists():
        return []
    raw = json.loads(caminho.read_text(encoding="utf-8"))
    resultado = []
    for m in raw:
        uf = ((m.get("microrregiao") or {}).get("mesorregiao") or {}).get("UF") or {}
        nome = m.get("nome", "")
        resultado.append({
            "code": m["id"],
            "name": nome,
            "stateCode": uf.get("id", 0),
            "state": uf.get("sigla", ""),
            "name_norm": _norm(nome),
        })
    return resultado

_MUNICIPIOS_REF = _carregar_municipios_ref()

# Regiões de saúde por município (carrega data/regioes_saude.json se existir)
def _carregar_regioes_ref() -> dict[int, list[dict]]:
    import json
    caminho = Path(__file__).parent / "data" / "regioes_saude.json"
    if not caminho.exists():
        return {}
    raw = json.loads(caminho.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}

_REGIOES_REF = _carregar_regioes_ref()

# Ocupações para busca
_OCUPACOES_REF = [
    {
        "code": str(v),
        "name": k.title(),
        "name_norm": _norm(k),
    }
    for k, v in CBO_MAP.items()
]

DENGUE_THRESHOLD = round(ENSEMBLE_THRESHOLD * 100, 1)

# ---------------------------------------------------------------------------
# GET /api/v1/triage/options
# ---------------------------------------------------------------------------

@app.get("/api/v1/triage/options")
def triage_options():
    """Retorna todas as listas de opções necessárias para o formulário de triagem."""
    ufs = [
        {"code": code, "sigla": sigla, "name": UF_LABELS[code]}
        for code, sigla in UF_ABBR_LABELS.items()
    ]
    return {
        "sexos": [
            {"code": k, "name": v} for k, v in SEX_LABELS.items()
        ],
        "racas": [
            {"code": k, "name": v} for k, v in RACE_LABELS.items()
        ],
        "escolaridades": [
            {"code": k, "name": v} for k, v in EDUCATION_LABELS.items()
        ],
        "situacoesGestacao": [
            {"code": k, "name": v} for k, v in PREGNANCY_LABELS.items()
        ],
        "sintomas": [
            {"id": "fever",              "label": "Febre"},
            {"id": "myalgia",            "label": "Mialgia / dor muscular"},
            {"id": "headache",           "label": "Cefaleia / dor de cabeça"},
            {"id": "rash",               "label": "Exantema / manchas na pele"},
            {"id": "vomiting",           "label": "Vômitos"},
            {"id": "nausea",             "label": "Náusea / enjoo"},
            {"id": "back_pain",          "label": "Dor nas costas"},
            {"id": "conjunctivitis",     "label": "Conjuntivite"},
            {"id": "arthritis",          "label": "Artrite"},
            {"id": "joint_pain",         "label": "Dor nas articulações"},
            {"id": "petechiae",          "label": "Petéquias / pontos vermelhos na pele"},
            {"id": "retro_orbital_pain", "label": "Dor atrás dos olhos"},
        ],
        "ufs": ufs,
        "modelosAtivos": list(modelos.keys()),
        "liamiarClassificacao": DENGUE_THRESHOLD,
        "pesosModelos": ENSEMBLE_WEIGHTS,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/references/occupations
# ---------------------------------------------------------------------------

@app.get("/api/v1/references/occupations")
def buscar_ocupacoes(
    query: str = _Query(..., min_length=2),
    limit: int = _Query(10, ge=1, le=50),
):
    """Busca ocupações CBO por nome. Prioriza matches que começam com o texto."""
    q = _norm(query)
    starts   = [o for o in _OCUPACOES_REF if o["name_norm"].startswith(q)]
    contains = [o for o in _OCUPACOES_REF if not o["name_norm"].startswith(q) and q in o["name_norm"]]
    resultado = (starts + contains)[:limit]
    return {"items": [{"code": o["code"], "name": o["name"]} for o in resultado]}


# ---------------------------------------------------------------------------
# GET /api/v1/references/municipalities
# ---------------------------------------------------------------------------

@app.get("/api/v1/references/municipalities")
def buscar_municipios(
    query: str = _Query(..., min_length=2),
    state: int | None = _Query(None),
    limit: int = _Query(20, ge=1, le=100),
):
    """Busca municípios por nome, com filtro opcional por código IBGE da UF."""
    if not _MUNICIPIOS_REF:
        raise HTTPException(
            status_code=503,
            detail="Base de municípios não encontrada. Adicione data/municipios.json.",
        )
    q = _norm(query)
    pool = [m for m in _MUNICIPIOS_REF if state is None or m["stateCode"] == state]
    starts   = [m for m in pool if m["name_norm"].startswith(q)]
    contains = [m for m in pool if not m["name_norm"].startswith(q) and q in m["name_norm"]]
    resultado = (starts + contains)[:limit]
    return {
        "items": [
            {
                "code": m["code"],
                "name": m["name"],
                "stateCode": m["stateCode"],
                "state": m["state"],
            }
            for m in resultado
        ]
    }


# ---------------------------------------------------------------------------
# GET /api/v1/references/health-regions
# ---------------------------------------------------------------------------

@app.get("/api/v1/references/health-regions")
def buscar_regioes_saude(municipality: int = _Query(...)):
    """Retorna as regiões de saúde associadas a um município (código IBGE)."""
    regioes = _REGIOES_REF.get(municipality, [])
    return {"items": regioes}
