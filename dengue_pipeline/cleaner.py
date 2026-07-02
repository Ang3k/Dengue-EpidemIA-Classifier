from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from .paths import (
    PROJECT_ROOT,
    RAW_CSV_DIR,
    RAW_DENGUE_PARQUETS,
    RAW_PARQUET_DIR,
)
from .sinan_mappings import add_sinan_cbo_labels, standardize_columns


FINAL_COLUMN_ORDER = [
    # Paciente
    "age_years",
    "sex_label",
    "pregnancy_status",
    "pregnancy_status_label",
    "race",
    "race_label",
    "education_level",
    "education_level_label",
    "occupation_code",
    "occupation_name",
    # Residência
    "residence_state",
    "residence_state_label",
    "residence_municipality",
    "residence_health_region",
    # Notificação
    "disease_code",
    "notification_date",
    "notification_year",
    "notification_month",
    "notification_day",
    "notification_epi_week",
    "notif_municipality",
    "notif_health_region",
    "health_facility",
    # Início dos sintomas
    "symptom_onset_date",
    "days_to_notification",
    "symptom_epi_year",
    "symptom_epi_week_number",
    # Sintomas e sinais auto-relatáveis/registrados
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
    # Atendimento, hospitalização e encerramento
    "hospitalized",
    "hospital_state",
    "hospital_state_label",
    # Alvo
    "final_classification",
    "final_classification_label",
]

ML_COLUMNS_TO_DROP = [
    "race_label",
    "education_level_label",
    "occupation_name",
    "residence_state_label",
    "hospital_state_label",
    "final_classification_label",
    "notification_date",
    "notification_day",
    "notification_epi_week",
    "symptom_onset_date",
    "symptom_epi_year",         # ano absoluto: não generaliza, fora das features
    "symptom_epi_week_number",  # crua: mantemos só a versão cíclica (sin/cos)
    "hospitalized",
    "hospital_state",
]


def ordenar_colunas_finais(df):
    colunas_ordenadas = [coluna for coluna in FINAL_COLUMN_ORDER if coluna in df.columns]
    outras_colunas = [coluna for coluna in df.columns if coluna not in colunas_ordenadas]
    return df[colunas_ordenadas + outras_colunas]


class DengueDataCleaner:
    def __init__(self, arquivos=None):
        self.occupation_encoder = None
        self.residence_state_encoder = None
        self.days_to_notification_median = None

        if arquivos is None:
            arquivos = RAW_DENGUE_PARQUETS
        elif isinstance(arquivos, (str, Path)):
            arquivos = [arquivos]

        self.arquivos = [Path(arquivo) for arquivo in arquivos]
        self.df = self.carregar()

    def carregar(self):
        dfs = []

        for arquivo in self.arquivos:
            fallback_dir = RAW_CSV_DIR if arquivo.suffix.lower() == ".csv" else RAW_PARQUET_DIR
            if not arquivo.exists() and (fallback_dir / arquivo.name).exists():
                arquivo = fallback_dir / arquivo.name

            if arquivo.suffix.lower() == ".csv":
                dfs.append(pd.read_csv(arquivo))
            else:
                dfs.append(pd.read_parquet(arquivo))

        return standardize_columns(pd.concat(dfs, ignore_index=True))

    def remover_colunas_iguais(self, df):
        for coluna in list(df.columns):
            if df[coluna].isnull().all():
                df = df.drop(coluna, axis=1)
            elif df[coluna].nunique() == 1:
                df = df.drop(coluna, axis=1)
        return df

    def limpar_angel(self):
        colunas = [
            "alarm_liver_enlarged", "infection_country", "severe_metrorrhagia",
            "infection_municipality", "autoimmune_disease",
            "petechiae_hemorrh", "severe_hypotension",
            "kidney_disease", "retro_orbital_pain", "chik_clinical_form",
            "duplicate_flag", "dengue_hemorrhagic_fever", "hemorrhagic_evidence",
            "joint_pain", "headache", "severe_tachycardia",
            "alarm_hematocrit_rise", "symptom_onset_date", "symptom_epi_week", "severe_hematemesis",
            "final_classification", "hematuria", "viral_isolation_result", "rash",
            "vomiting", "birth_date", "notification_date", "severe_weak_pulse", "race",
            "alarm_low_platelets", "alarm_signs_date", "severe_bleeding",
            "plasma_leakage", "petechiae", "pregnancy_status",
            "severe_ast_elevated", "severe_cap_refill", "severe_myocarditis",
            "severe_convulsions",
        ]

        df = self.df[colunas].copy()

        cols_leakage = [
            "confirmation_criteria", "case_closure_date", "alarm_hypotension",
            "alarm_low_platelets", "alarm_persistent_vomit", "alarm_bleeding",
            "alarm_hematocrit_rise", "alarm_abdominal_pain", "alarm_lethargy",
            "alarm_liver_enlarged", "alarm_fluid_accumul", "alarm_signs_date",
            "severe_weak_pulse", "severe_convulsions", "severe_cap_refill",
            "severe_resp_distress", "severe_tachycardia", "severe_cold_extremities",
            "severe_hypotension", "severe_hematemesis", "severe_melena",
            "severe_metrorrhagia", "severe_bleeding", "severe_ast_elevated",
            "severe_myocarditis", "severe_altered_consc", "severe_organ_damage",
            "severity_signs_date", "infection_country", "infection_municipality",
        ]
        df = df.drop(cols_leakage, axis=1, errors="ignore")
        df = self.remover_colunas_iguais(df)

        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
        df["birth_year_derived"] = df["birth_date"].dt.year
        df = df.drop(["birth_date", "birth_year_derived"], axis=1, errors="ignore")
        df["notification_date"] = pd.to_datetime(df["notification_date"], errors="coerce")
        df["symptom_onset_date"] = pd.to_datetime(df["symptom_onset_date"], errors="coerce")
        df["days_to_notification"] = (df["notification_date"] - df["symptom_onset_date"]).dt.days

        df = df.drop(["autoimmune_disease", "kidney_disease"], axis=1, errors="ignore")
        df = df.drop(
            ["chik_clinical_form", "viral_isolation_result"],
            axis=1,
            errors="ignore",
        )

        df["symptom_epi_year"] = df["symptom_epi_week"] // 100
        df["symptom_epi_week_number"] = df["symptom_epi_week"] % 100
        df = df.drop(columns=["symptom_epi_week"])

        df["final_classification"] = df["final_classification"].map({
            5: 0,
            10: 1,
            11: 1,
            12: 1,
        })
        df = df[df["final_classification"].notna()].copy()

        return df

    def limpar_pedro(self):
        colunas = [
            "residence_health_region", "notification_date", "disease_code",
            "fever", "notification_year", "sex", "autochthonous_case",
            "hospitalized", "health_facility", "occupation_code", "nausea",
            "notif_health_region", "residence_state", "age",
        ]

        df = self.df[colunas].copy()

        def parse_idade(valor):
            try:
                s = str(int(valor)).zfill(4)
                unidade = int(s[0])
                numero = int(s[1:])
                if unidade == 4:
                    return numero
                elif unidade == 3:
                    return numero / 12
                elif unidade == 2:
                    return numero / 365
                elif unidade == 1:
                    return numero / 8760
                else:
                    return None
            except:
                return None

        df["age_years"] = df["age"].apply(parse_idade)
        df = df.drop(columns=["age"])

        df["notification_date"] = pd.to_datetime(df["notification_date"], format="%Y-%m-%d")
        df["notification_month"] = df["notification_date"].dt.month
        df["notification_day"] = df["notification_date"].dt.day
        df = df.drop(columns=["notification_date"])

        df["occupation_code"] = df["occupation_code"].fillna(0)
        df = df.drop(columns=["autochthonous_case"])

        return df

    def limpar_ruan(self):
        minhas_colunas = [
            "pcr_date", "hospital_state", "death_date", "prnt_result",
            "hemorrhagic_manifest", "arthritis", "blood_disorder",
            "chik_test1_result", "infection_state", "serology_result", "myalgia",
            "prnt_date", "severe_organ_damage", "ns1_test_date",
            "investigation_date", "alarm_abdominal_pain", "notif_state",
            "immunohistochemistry", "serotype", "severe_resp_distress",
            "peptic_ulcer", "conjunctivitis", "nosebleed", "flow_received",
            "back_pain", "notification_type", "notif_municipality",
            "notification_epi_week", "hospitalization_date", "hypertension",
            "liver_disease", "residence_municipality", "alarm_fluid_accumul",
            "symptom_onset_date", "metrorrhagia", "system_type", "histopathology",
            "education_level", "severe_altered_consc",
        ]

        df = self.df[minhas_colunas].copy()

        col = [
            "serotype", "infection_state", "ns1_test_date", "pcr_date",
            "death_date", "prnt_date", "prnt_result", "chik_test1_result",
            "serology_result", "alarm_abdominal_pain", "alarm_fluid_accumul",
            "severe_organ_damage", "severe_altered_consc", "severe_resp_distress",
            "histopathology", "immunohistochemistry",
        ]
        df = df.drop(columns=col)
        df = self.remover_colunas_iguais(df)

        col = ["blood_disorder", "liver_disease", "hypertension", "peptic_ulcer"]
        df = df.drop(columns=col, errors="ignore")

        col = ["investigation_date", "hospitalization_date", "notif_state", "notification_type"]
        df = df.drop(columns=col)

        return df

    def juntar(self, df_angel, df_pedro, df_ruan):
        df = pd.concat([df_angel, df_pedro, df_ruan], axis=1, join="inner")
        df = df.loc[:, ~df.columns.duplicated()]
        df = add_sinan_cbo_labels(df.reset_index(drop=True))
        return ordenar_colunas_finais(df)

    def transformar_analise(self):
        df_angel = self.limpar_angel()
        df_pedro = self.limpar_pedro()
        df_ruan = self.limpar_ruan()
        return self.juntar(df_angel, df_pedro, df_ruan)

    def transformar_ml(self, df_tratado=None):
        if df_tratado is None:
            df_tratado = self.transformar_analise()

        # One Hot Encoding para sexo
        df_tratado = pd.get_dummies(df_tratado, columns=["sex_label"], dtype=int)
        df_tratado = df_tratado.drop(columns=["sex"], errors="ignore")
        df_tratado = df_tratado.rename(columns={
            "sex_label_Feminino": "sex_Female",
            "sex_label_Ignorado": "sex_Ignored",
            "sex_label_Masculino": "sex_Male",
        }).copy()

        # Raça permanece em one-hot.
        df_tratado["race"] = (
            df_tratado["race"].astype("Int64").astype("string")
        )
        df_tratado = pd.get_dummies(
            df_tratado,
            columns=["race"],
            dtype=int,
        )

        # Estado fica em uma única coluna ordinal para comparação com one-hot.
        # O valor 0 é reservado para ausente ou desconhecido.
        self.residence_state_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        residence_state = (
            df_tratado[["residence_state"]]
            .astype("Int64")
            .astype("string")
        )
        residence_state = residence_state.astype(object).where(
            residence_state.notna(),
            np.nan,
        )
        df_tratado[["residence_state"]] = (
            self.residence_state_encoder.fit_transform(residence_state) + 1
        )
        df_tratado["residence_state"] = (
            df_tratado["residence_state"].fillna(0).astype(int)
        )

        # Analise ciclica de meses do ano e semanas epidemiológicas
        # Aqui, temos os meses representados num circulo --> (cos, sen)
        df_tratado["notification_month_sin"] = np.sin(2 * np.pi * df_tratado["notification_month"] / 12)
        df_tratado["notification_month_cos"] = np.cos(2 * np.pi * df_tratado["notification_month"] / 12)

        # Max da semana epidemiológica é 53
        df_tratado["symptom_epi_week_number_sin"] = np.sin(2 * np.pi * df_tratado["symptom_epi_week_number"] / 53)
        df_tratado["symptom_epi_week_number_cos"] = np.cos(2 * np.pi * df_tratado["symptom_epi_week_number"] / 53)

        # Mês do início dos sintomas em forma cíclica (sin/cos), igual aos demais
        # sinais de sazonalidade. Não guardamos a versão crua/linear nem o dia.
        symptom_onset = pd.to_datetime(
            df_tratado["symptom_onset_date"],
            errors="coerce",
        )
        symptom_month = symptom_onset.dt.month
        df_tratado["symptom_month_sin"] = np.sin(2 * np.pi * symptom_month / 12)
        df_tratado["symptom_month_cos"] = np.cos(2 * np.pi * symptom_month / 12)

        # Criar uma coluna para quantidade total de sintomas.
        sintomas = [
            "fever", "myalgia", "headache", "rash", "vomiting", "nausea",
            "back_pain", "conjunctivitis", "arthritis", "joint_pain",
            "petechiae", "retro_orbital_pain",
        ]

        # SINAN: 1 = Sim, 2 = Não; ainda há ~58 NaN por coluna. Binarizamos com
        # == 1 para que 2/NaN (e qualquer código != 1) virem 0, deixando as
        # colunas prontas para o modelo (sem NaN).
        df_tratado[sintomas] = (df_tratado[sintomas] == 1).astype(int)

        interaction_columns = {
            f"{symptom_a}_and_{symptom_b}": (
                df_tratado[symptom_a] * df_tratado[symptom_b]
            ).astype("int8")
            for symptom_a, symptom_b in combinations(sintomas, 2)
        }
        df_tratado = pd.concat(
            [
                df_tratado,
                pd.DataFrame(interaction_columns, index=df_tratado.index),
            ],
            axis=1,
        )

        # Agora 1 = Tem ; 0 = Não tem
        df_tratado["number_of_symptoms"] = df_tratado[sintomas].sum(axis=1)

        # Rash e Retro Orbital Pain são os sintomas que apresentam maior gap relativo entre descartados x confirmados
        sintomas_importantes = ["rash", "retro_orbital_pain"]

        df_tratado["number_of_important_symptoms"] = df_tratado[sintomas_importantes].sum(axis=1)

        # Ordinal Encoding na escolaridade
        map_escolaridade = {
            "Ignorado": 0,
            None: 0,
            "Não se aplica": 0,

            "Analfabeto": 1,

            "1ª a 4ª série incompleta": 2,
            "4ª série completa": 2,
            "5ª à 8ª série incompleta": 2,

            "Ensino fundamental completo": 3,

            "Ensino médio incompleto": 4,
            "Ensino médio completo": 4,

            "Educação superior incompleta": 5,
            "Educação superior completa": 5,
        }

        df_tratado["education_level"] = df_tratado["education_level_label"].map(map_escolaridade)

        df_tratado = df_tratado.drop(columns=["disease_code"])  # Disease_code todos são A90.

        days_to_notification = pd.to_numeric(
            df_tratado["days_to_notification"],
            errors="coerce",
        )
        self.days_to_notification_median = days_to_notification.median()
        df_tratado["days_to_notification"] = (
            days_to_notification
            .fillna(self.days_to_notification_median)
            .clip(0, 90)
        )

        # Gravidez agora é binario, 1 para sim e 0 para não. Alem disso, temos agora uma coluna
        # binaria para dizer se a gravidez foi informada (exclui o caso dos homens e ignorado)
        
        df_tratado["pregnancy"] = df_tratado["pregnancy_status"].isin([1, 2, 3, 4]).astype(int)
        df_tratado["pregnancy_informed"] = df_tratado["pregnancy_status"].isin([1, 2, 3, 4, 5]).astype(int)
        df_tratado = df_tratado.drop(columns=["pregnancy_status_label", "pregnancy_status"], errors="ignore").copy()

        self.occupation_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        occupation = df_tratado[["occupation_code"]].astype("string")
        occupation = occupation.replace("0", pd.NA)
        occupation = occupation.astype(object).where(occupation.notna(), np.nan)
        df_tratado[["occupation_code"]] = (
            self.occupation_encoder.fit_transform(occupation) + 1
        )
        df_tratado["occupation_code"] = (
            df_tratado["occupation_code"].fillna(0).astype(int)
        )

        # A data de início já está representada pelo ano, semana epidemiológica
        # e pelas versões cíclicas da semana.
        df_tratado = df_tratado.drop(
            columns=ML_COLUMNS_TO_DROP,
            errors="ignore",
        )

        return df_tratado

    def salvar_df(self, df, caminho_saida):
        caminho_saida = Path(caminho_saida)
        if not caminho_saida.is_absolute():
            caminho_saida = PROJECT_ROOT / caminho_saida
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(caminho_saida, index=False)
