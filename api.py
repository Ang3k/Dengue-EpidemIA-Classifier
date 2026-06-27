"""
API de predição de dengue.

Rode com:
    .venv\Scripts\python -m uvicorn api:app --reload

O pré-processamento aqui reproduz exatamente o transformar_ml do cleaner. Os
encoders ajustados no treino (ocupação e UF) são carregados de
artifacts/models/ml_preprocess.joblib, que é gerado pelo notebook de modelagem.
"""

from calendar import monthrange
from datetime import date
from itertools import combinations
import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carregar modelos e o pré-processamento salvos
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "artifacts" / "models"
PREPROCESS_PATH = MODELS_DIR / "ml_preprocess.joblib"

MODELOS_DISPONIVEIS = {
    "logistic_regression": "logistic_regression.joblib",
    "xgboost":             "xgboost.joblib",
    "lightgbm":            "lightgbm.joblib",
    "decision_tree":       "decision_tree.joblib",
}

modelos = {}
erros_carregamento = {}
for nome, arquivo in MODELOS_DISPONIVEIS.items():
    caminho = MODELS_DIR / arquivo
    if not caminho.exists():
        erros_carregamento[nome] = f"arquivo não encontrado: {caminho}"
        logger.warning("Modelo %s não encontrado em %s", nome, caminho)
        continue

    try:
        modelo = joblib.load(caminho)
        if nome == "xgboost":
            modelo_interno = getattr(modelo, "model", None)
            if hasattr(modelo_interno, "set_params"):
                modelo_interno.set_params(
                    device=os.getenv("XGBOOST_DEVICE", "cpu")
                )
        modelos[nome] = modelo
        logger.info("Modelo %s carregado", nome)
    except Exception as exc:
        erros_carregamento[nome] = str(exc)
        logger.exception("Não foi possível carregar o modelo %s", nome)

# Encoders ajustados no treino (mesmos objetos usados pelo cleaner).
preprocess = {}
if PREPROCESS_PATH.exists():
    try:
        preprocess = joblib.load(PREPROCESS_PATH)
        logger.info("Pré-processamento carregado: %s", sorted(preprocess))
    except Exception:
        logger.exception("Não foi possível carregar %s", PREPROCESS_PATH)
else:
    logger.warning(
        "Pré-processamento não encontrado em %s; rode o notebook de modelagem",
        PREPROCESS_PATH,
    )

OCCUPATION_ENCODER = preprocess.get("occupation_encoder")
RESIDENCE_STATE_ENCODER = preprocess.get("residence_state_encoder")
DAYS_TO_NOTIFICATION_MEDIAN = preprocess.get("days_to_notification_median", 0.0)

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
        pattern=r"^\d{5,6}$",
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
    fever: int = Field(default=0, ge=0, le=1)
    myalgia: int = Field(default=0, ge=0, le=1)
    headache: int = Field(default=0, ge=0, le=1)
    rash: int = Field(default=0, ge=0, le=1)
    vomiting: int = Field(default=0, ge=0, le=1)
    nausea: int = Field(default=0, ge=0, le=1)
    back_pain: int = Field(default=0, ge=0, le=1)
    conjunctivitis: int = Field(default=0, ge=0, le=1)
    arthritis: int = Field(default=0, ge=0, le=1)
    joint_pain: int = Field(default=0, ge=0, le=1)
    petechiae: int = Field(default=0, ge=0, le=1)
    retro_orbital_pain: int = Field(default=0, ge=0, le=1)
    tourniquet_test: int = Field(default=0, ge=0, le=1)

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


# ---------------------------------------------------------------------------
# Pré-processamento — replica o transformar_ml() do cleaner
# ---------------------------------------------------------------------------

SINTOMAS = [
    "fever", "myalgia", "headache", "rash", "vomiting", "nausea",
    "back_pain", "conjunctivitis", "arthritis", "joint_pain",
    "petechiae", "retro_orbital_pain",
]

# Código SINAN da escolaridade (0-10) -> ordinal usado no treino.
# Vem da composição EDUCATION_LABELS (código -> texto) com o map_escolaridade do
# cleaner (texto -> 0..5). Ex.: 0 = Analfabeto -> 1; 9/10 = Ignorado/NA -> 0.
MAP_ESCOLARIDADE = {0: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5, 9: 0, 10: 0}

MAP_SEXO_LABEL = {"M": "Masculino", "F": "Feminino", "I": "Ignorado"}

RACE_CODES = [1, 2, 3, 4, 5, 9]


def _encodar_ordinal(encoder, valor, coluna, como_int64: bool):
    """Aplica um OrdinalEncoder ajustado no treino a um único valor, do mesmo
    jeito que o cleaner faz: desconhecido/ausente vira 0."""
    if encoder is None:
        return 0
    serie = pd.DataFrame({coluna: [valor]})
    if como_int64:
        serie = serie.astype("Int64").astype("string")
    else:
        serie = serie.astype("string").replace("0", pd.NA)
    serie = serie.astype(object).where(serie.notna(), np.nan)
    encoded = encoder.transform(serie) + 1
    valor_enc = encoded[0][0]
    return 0 if pd.isna(valor_enc) else int(valor_enc)


def construir_features(dados: DadosPaciente) -> pd.DataFrame:
    """Transforma os dados do formulário no vetor de features do modelo,
    reproduzindo o transformar_ml()."""

    row: dict = {}

    # --- idade ---
    row["age_years"] = dados.age_years

    # --- sexo (one-hot) ---
    sexo_label = MAP_SEXO_LABEL.get(dados.sex or "", "Ignorado")
    row["sex_Female"] = int(sexo_label == "Feminino")
    row["sex_Male"] = int(sexo_label == "Masculino")
    row["sex_Ignored"] = int(sexo_label == "Ignorado")

    # --- raça (one-hot, inclui o código 9 = Ignorado) ---
    for cod in RACE_CODES:
        row[f"race_{cod}"] = int((dados.race or 0) == cod)

    # --- escolaridade (ordinal) ---
    row["education_level"] = MAP_ESCOLARIDADE.get(dados.education_level, 0)

    # --- ocupação e UF: encoders ajustados no treino ---
    row["occupation_code"] = _encodar_ordinal(
        OCCUPATION_ENCODER, dados.occupation_code, "occupation_code", como_int64=False
    )
    row["residence_state"] = _encodar_ordinal(
        RESIDENCE_STATE_ENCODER, dados.residence_state, "residence_state", como_int64=True
    )

    # --- demais geográficos / administrativos (crus, como no transformar_ml) ---
    row["residence_municipality"] = dados.residence_municipality or 0
    row["residence_health_region"] = dados.residence_health_region or 0
    ano_notif = (
        dados.notification_year
        or (dados.notification_date.year if dados.notification_date else None)
    )
    row["notification_year"] = ano_notif or 0
    row["notif_municipality"] = dados.notif_municipality or 0
    row["notif_health_region"] = dados.notif_health_region or 0
    row["health_facility"] = dados.health_facility or 0

    # --- sazonalidade cíclica ---
    mes_notif = (
        dados.notification_month
        or (dados.notification_date.month if dados.notification_date else None)
    )
    mes_notif_num = mes_notif if mes_notif is not None else np.nan
    row["notification_month"] = mes_notif_num
    row["notification_month_sin"] = np.sin(
        2 * np.pi * mes_notif_num / 12
    )
    row["notification_month_cos"] = np.cos(
        2 * np.pi * mes_notif_num / 12
    )

    semana = (
        dados.symptom_epi_week_number
        or (
            dados.symptom_onset_date.isocalendar().week
            if dados.symptom_onset_date
            else None
        )
    )
    semana_num = semana if semana is not None else np.nan
    row["symptom_epi_week_number"] = semana_num
    row["symptom_epi_week_number_sin"] = np.sin(
        2 * np.pi * semana_num / 53
    )
    row["symptom_epi_week_number_cos"] = np.cos(
        2 * np.pi * semana_num / 53
    )

    if dados.symptom_onset_date:
        mes_sint = dados.symptom_onset_date.month
        row["symptom_month"] = mes_sint
        row["symptom_day"] = dados.symptom_onset_date.day
        row["symptom_month_end"] = int(
            dados.symptom_onset_date.day
            == monthrange(
                dados.symptom_onset_date.year,
                dados.symptom_onset_date.month,
            )[1]
        )
    else:
        mes_sint = np.nan
        row["symptom_month"] = np.nan
        row["symptom_day"] = np.nan
        row["symptom_month_end"] = 0
    row["symptom_month_sin"] = np.sin(2 * np.pi * mes_sint / 12)
    row["symptom_month_cos"] = np.cos(2 * np.pi * mes_sint / 12)

    # --- dias até notificação: mesma regra do cleaner (mediana p/ ausente, clip 0-90) ---
    dias = dados.days_to_notification
    if (
        dias is None
        and dados.notification_date
        and dados.symptom_onset_date
    ):
        dias = (dados.notification_date - dados.symptom_onset_date).days
    if dias is None:
        dias = DAYS_TO_NOTIFICATION_MEDIAN
    row["days_to_notification"] = float(np.clip(dias, 0, 90))

    # --- sintomas (binários vindos do frontend) ---
    for s in SINTOMAS + ["tourniquet_test"]:
        row[s] = int(getattr(dados, s, 0) == 1)

    # --- interações entre sintomas ---
    for s_a, s_b in combinations(SINTOMAS, 2):
        row[f"{s_a}_and_{s_b}"] = row[s_a] * row[s_b]

    # --- agregados de sintomas ---
    row["number_of_symptoms"] = sum(row[s] for s in SINTOMAS)
    row["number_of_important_symptoms"] = row["rash"] + row["retro_orbital_pain"]

    # --- gravidez ---
    row["pregnancy"] = int(dados.pregnancy_status in [1, 2, 3, 4])
    row["pregnancy_informed"] = int(dados.pregnancy_status in [1, 2, 3, 4, 5])

    df = pd.DataFrame([row])
    return df.select_dtypes(include=["number"]).astype("float32")


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    faltantes = [
        nome for nome in MODELOS_DISPONIVEIS if nome not in modelos
    ]
    preprocess_pronto = all(
        [
            OCCUPATION_ENCODER is not None,
            RESIDENCE_STATE_ENCODER is not None,
            "days_to_notification_median" in preprocess,
        ]
    )
    return {
        "status": "ok" if not faltantes and preprocess_pronto else "degraded",
        "modelos_carregados": list(modelos.keys()),
        "modelos_ausentes": faltantes,
        "erros_carregamento": erros_carregamento,
        "preprocess_carregado": preprocess_pronto,
    }


@app.post("/predict")
def predict(dados: DadosPaciente):
    if not modelos:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo foi carregado",
        )
    if OCCUPATION_ENCODER is None or RESIDENCE_STATE_ENCODER is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pré-processamento indisponível. Gere "
                "artifacts/models/ml_preprocess.joblib no notebook de modelagem."
            ),
        )

    df = construir_features(dados)

    resultados = []
    ignorados = []
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

    media = round(sum(r["probability"] for r in resultados) / len(resultados), 1)

    return {
        "models": resultados,
        "average": media,
        "isDengue": media >= 40,
        "ignored": ignorados,
    }
