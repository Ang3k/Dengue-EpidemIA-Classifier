"""
API de predição de dengue.

Coloque este arquivo na raiz do projeto (mesma pasta que dengue_pipeline/).
Rode com:
    py -3.11 -m uvicorn api:app --reload

O pré-processamento aqui reproduz exatamente o transformar_ml do cleaner. Os
encoders ajustados no treino (ocupação e UF) são carregados de
artifacts/models/ml_preprocess.joblib, que é gerado pelo notebook de modelagem.
"""

from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
for nome, arquivo in MODELOS_DISPONIVEIS.items():
    caminho = MODELS_DIR / arquivo
    if caminho.exists():
        modelos[nome] = joblib.load(caminho)
        print(f"OK {nome} carregado")
    else:
        print(f"-- {nome} nao encontrado ({caminho})")

# Encoders ajustados no treino (mesmos objetos usados pelo cleaner).
preprocess = {}
if PREPROCESS_PATH.exists():
    preprocess = joblib.load(PREPROCESS_PATH)
    print(f"OK pre-processamento carregado ({sorted(preprocess)})")
else:
    print(
        f"-- ml_preprocess.joblib nao encontrado ({PREPROCESS_PATH}). "
        "occupation_code e residence_state ficarao em 0 (rode o notebook de "
        "modelagem para gerar o arquivo)."
    )

OCCUPATION_ENCODER = preprocess.get("occupation_encoder")
RESIDENCE_STATE_ENCODER = preprocess.get("residence_state_encoder")
DAYS_TO_NOTIFICATION_MEDIAN = preprocess.get("days_to_notification_median", 0.0)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="API Dengue")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção, troque por ["http://localhost:5173"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schema de entrada — campos que o frontend já envia
# ---------------------------------------------------------------------------

class DadosPaciente(BaseModel):
    # Paciente
    age_years: float | None = None
    sex: str | None = None                  # "M", "F", "I"
    pregnancy_status: int | None = None     # 1-4 grávida, 5 não, 6 N/A, 9 ignorado
    race: int | None = None                 # 1-5, 9
    education_level: int | None = None      # 0-10 (código SINAN)
    occupation_code: str | None = None      # código CBO

    # Residência
    residence_state: int | None = None      # código IBGE da UF (ex.: 35)
    residence_municipality: int | None = None
    residence_health_region: int | None = None

    # Notificação / datas
    notification_date: str | None = None    # "YYYY-MM-DD"
    notification_year: int | None = None
    notification_month: int | None = None
    notification_epi_week: int | None = None
    notif_municipality: int | None = None
    notif_health_region: int | None = None
    health_facility: int | None = None

    # Início dos sintomas
    symptom_onset_date: str | None = None   # "YYYY-MM-DD"
    days_to_notification: float | None = None
    symptom_epi_year: int | None = None
    symptom_epi_week_number: int | None = None

    # Sintomas (1 = sim, 0 = não)
    fever: int = 0
    myalgia: int = 0
    headache: int = 0
    rash: int = 0
    vomiting: int = 0
    nausea: int = 0
    back_pain: int = 0
    conjunctivitis: int = 0
    arthritis: int = 0
    joint_pain: int = 0
    petechiae: int = 0
    retro_orbital_pain: int = 0
    tourniquet_test: int = 0

    # Hospitalização
    hospitalized: int | None = None
    hospital_state: int | None = None


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
    row["notification_year"] = dados.notification_year or 0
    row["notif_municipality"] = dados.notif_municipality or 0
    row["notif_health_region"] = dados.notif_health_region or 0
    row["health_facility"] = dados.health_facility or 0

    # --- sazonalidade cíclica ---
    mes_notif = dados.notification_month or 1
    row["notification_month_sin"] = np.sin(2 * np.pi * mes_notif / 12)
    row["notification_month_cos"] = np.cos(2 * np.pi * mes_notif / 12)

    semana = dados.symptom_epi_week_number or 1
    row["symptom_epi_week_number_sin"] = np.sin(2 * np.pi * semana / 53)
    row["symptom_epi_week_number_cos"] = np.cos(2 * np.pi * semana / 53)

    if dados.symptom_onset_date:
        mes_sint = pd.to_datetime(dados.symptom_onset_date, errors="coerce").month or 1
    elif dados.notification_month:
        mes_sint = dados.notification_month
    else:
        mes_sint = 1
    row["symptom_month_sin"] = np.sin(2 * np.pi * mes_sint / 12)
    row["symptom_month_cos"] = np.cos(2 * np.pi * mes_sint / 12)

    # --- dias até notificação: mesma regra do cleaner (mediana p/ ausente, clip 0-90) ---
    dias = dados.days_to_notification
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
    return {
        "status": "ok",
        "modelos_carregados": list(modelos.keys()),
        "preprocess_carregado": bool(preprocess),
    }


@app.post("/predict")
def predict(dados: DadosPaciente):
    df = construir_features(dados)

    resultados = []
    ignorados = []
    for nome, modelo in modelos.items():
        df_alinhado, faltantes = alinhar_colunas(df.copy(), modelo)
        if df_alinhado is None:
            ignorados.append({"name": nome, "missing": faltantes[:10]})
            continue
        proba = np.array(modelo.predict_proba(df_alinhado))
        prob = float(proba[0][1]) if proba.ndim == 2 else float(proba[0])
        resultados.append({"name": nome, "probability": round(prob * 100, 1)})

    if not resultados:
        return {"error": "Nenhum modelo pode prever", "ignorados": ignorados}

    media = round(sum(r["probability"] for r in resultados) / len(resultados), 1)

    return {
        "models": resultados,
        "average": media,
        "isDengue": media >= 40,
        "ignorados": ignorados,
    }
