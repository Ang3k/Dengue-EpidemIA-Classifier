"""
API de predição de dengue.

Coloque este arquivo na raiz do projeto (mesma pasta que dengue_pipeline/).
Rode com:
    py -3.11 -m uvicorn api:app --reload
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Carregar modelos salvos
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "artifacts" / "models"

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
        print(f"✓ {nome} carregado")
    else:
        print(f"✗ {nome} não encontrado ({caminho})")

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
    education_level: int | None = None      # 0-10
    occupation_code: str | None = None

    # Residência
    residence_state: int | None = None      # código IBGE da UF
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
# Pré-processamento — replica o que o cleaner faz em transformar_ml()
# ---------------------------------------------------------------------------

SINTOMAS = [
    "fever", "myalgia", "headache", "rash", "vomiting", "nausea",
    "back_pain", "conjunctivitis", "arthritis", "joint_pain",
    "petechiae", "retro_orbital_pain",
]

MAP_ESCOLARIDADE = {
    0: 0, 9: 0, 10: 0,
    1: 1,
    2: 2, 3: 2,
    4: 3,
    5: 4, 6: 4,
    7: 5, 8: 5,
}

MAP_SEXO_LABEL = {"M": "Masculino", "F": "Feminino", "I": "Ignorado"}

# Ordem exata das colunas que o modelo espera (gerada pelo notebook)
FEATURE_COLUMNS = None   # preenchida na primeira chamada, abaixo


def construir_features(dados: DadosPaciente) -> pd.DataFrame:
    """Transforma os dados do formulário no vetor de features do modelo."""

    row: dict = {}

    # --- idade ---
    row["age_years"] = dados.age_years

    # --- sexo (one-hot) ---
    sexo_label = MAP_SEXO_LABEL.get(dados.sex or "", "Ignorado")
    row["sex_Female"]  = int(sexo_label == "Feminino")
    row["sex_Male"]    = int(sexo_label == "Masculino")
    row["sex_Ignored"] = int(sexo_label == "Ignorado")

    # --- raça (one-hot) ---
    for cod in [1, 2, 3, 4, 5]:
        row[f"race_{cod}"] = int((dados.race or 0) == cod)

    # --- escolaridade (ordinal) ---
    row["education_level"] = MAP_ESCOLARIDADE.get(dados.education_level or 0, 0)

    # --- ocupação (ordinal simples: 0 = desconhecido, 1+ = presente) ---
    row["occupation_code"] = 1 if dados.occupation_code else 0

    # --- residência ---
    row["residence_state"]         = dados.residence_state or 0
    row["residence_municipality"]  = dados.residence_municipality or 0
    row["residence_health_region"] = dados.residence_health_region or 0

    # --- notificação ---
    row["notification_year"]  = dados.notification_year or 0
    row["notif_municipality"] = dados.notif_municipality or 0
    row["notif_health_region"]= dados.notif_health_region or 0
    row["health_facility"]    = dados.health_facility or 0

    # --- sazonalidade cíclica (mês da notificação) ---
    mes_notif = dados.notification_month or 1
    row["notification_month_sin"] = np.sin(2 * np.pi * mes_notif / 12)
    row["notification_month_cos"] = np.cos(2 * np.pi * mes_notif / 12)

    # --- sazonalidade cíclica (semana epidemiológica dos sintomas) ---
    semana = dados.symptom_epi_week_number or 1
    row["symptom_epi_week_number_sin"] = np.sin(2 * np.pi * semana / 53)
    row["symptom_epi_week_number_cos"] = np.cos(2 * np.pi * semana / 53)

    # --- sazonalidade cíclica (mês de início dos sintomas) ---
    if dados.symptom_onset_date:
        mes_sint = pd.to_datetime(dados.symptom_onset_date, errors="coerce").month or 1
    elif dados.notification_month:
        mes_sint = dados.notification_month
    else:
        mes_sint = 1
    row["symptom_month_sin"] = np.sin(2 * np.pi * mes_sint / 12)
    row["symptom_month_cos"] = np.cos(2 * np.pi * mes_sint / 12)

    # --- dias até notificação ---
    row["days_to_notification"] = min(dados.days_to_notification or 0, 90)

    # --- sintomas (já binários vindos do frontend: 0/1) ---
    for s in SINTOMAS + ["tourniquet_test"]:
        row[s] = getattr(dados, s, 0)

    # --- interações entre sintomas ---
    from itertools import combinations
    for s_a, s_b in combinations(SINTOMAS, 2):
        row[f"{s_a}_and_{s_b}"] = row[s_a] * row[s_b]

    # --- agregados de sintomas ---
    row["number_of_symptoms"] = sum(row[s] for s in SINTOMAS)
    row["number_of_important_symptoms"] = row["rash"] + row["retro_orbital_pain"]

    # --- gravidez ---
    row["pregnancy"]          = int(dados.pregnancy_status in [1, 2, 3, 4])
    row["pregnancy_informed"] = int(dados.pregnancy_status in [1, 2, 3, 4, 5])

    df = pd.DataFrame([row])

    # Garante que as colunas numéricas estejam em float32 (igual ao treino)
    df = df.select_dtypes(include=["number"]).astype("float32")

    return df


def alinhar_colunas(df: pd.DataFrame, modelo) -> pd.DataFrame:
    """
    Alinha as colunas do DataFrame com as que o modelo espera.
    Colunas ausentes viram 0; colunas extras são descartadas.
    """
    try:
        # sklearn Pipeline / modelos com feature_names_in_
        esperadas = list(modelo.feature_names_in_)
    except AttributeError:
        try:
            # GradientBoostingDiseaseClassifier (wrapper XGB/LGB)
            esperadas = modelo.feature_names
        except AttributeError:
            return df  # modelo não informa colunas; usa o que temos

    for col in esperadas:
        if col not in df.columns:
            df[col] = 0.0
    return df[esperadas].astype("float32")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "modelos_carregados": list(modelos.keys())}


@app.post("/predict")
def predict(dados: DadosPaciente):
    df = construir_features(dados)

    resultados = []
    for nome, modelo in modelos.items():
        df_alinhado = alinhar_colunas(df.copy(), modelo)
        proba = modelo.predict_proba(df_alinhado)
        # predict_proba pode retornar (n_samples, n_classes) ou (n_samples,)
        proba = np.array(proba)
        if proba.ndim == 2:
            prob = float(proba[0][1])
        else:
            prob = float(proba[0])
        resultados.append({
            "name": nome,
            "probability": round(prob * 100, 1),
        })

    if not resultados:
        return {"error": "Nenhum modelo carregado"}

    media = round(sum(r["probability"] for r in resultados) / len(resultados), 1)

    return {
        "models": resultados,
        "average": media,
        "isDengue": media >= 40,
    }