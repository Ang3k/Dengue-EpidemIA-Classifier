import pandas as pd

from .cbo_map import CBO_MAP


COLUMN_RENAME_MAP = {
    "TP_NOT": "notification_type",
    "ID_AGRAVO": "disease_code",
    "DT_NOTIFIC": "notification_date",
    "SEM_NOT": "notification_epi_week",
    "NU_ANO": "notification_year",
    "SG_UF_NOT": "notif_state",
    "ID_MUNICIP": "notif_municipality",
    "ID_REGIONA": "notif_health_region",
    "ID_UNIDADE": "health_facility",
    "DT_SIN_PRI": "symptom_onset_date",
    "SEM_PRI": "symptom_epi_week",
    "ANO_NASC": "birth_year",
    "DT_NASC": "birth_date",
    "NU_IDADE_N": "age",
    "CS_SEXO": "sex",
    "CS_GESTANT": "pregnancy_status",
    "CS_RACA": "race",
    "CS_ESCOL_N": "education_level",
    "ID_OCUPA_N": "occupation_code",
    "SG_UF": "residence_state",
    "ID_MN_RESI": "residence_municipality",
    "ID_RG_RESI": "residence_health_region",
    "ID_PAIS": "residence_country",
    "DT_INVEST": "investigation_date",
    "FEBRE": "fever",
    "MIALGIA": "myalgia",
    "CEFALEIA": "headache",
    "EXANTEMA": "rash",
    "VOMITO": "vomiting",
    "NAUSEA": "nausea",
    "DOR_COSTAS": "back_pain",
    "CONJUNTVIT": "conjunctivitis",
    "ARTRITE": "arthritis",
    "ARTRALGIA": "joint_pain",
    "PETEQUIA_N": "petechiae",
    "LEUCOPENIA": "leucopenia",
    "DOR_RETRO": "retro_orbital_pain",
    "DIABETES": "diabetes",
    "HEMATOLOG": "blood_disorder",
    "HEPATOPAT": "liver_disease",
    "RENAL": "kidney_disease",
    "HIPERTENSA": "hypertension",
    "ACIDO_PEPT": "peptic_ulcer",
    "AUTO_IMUNE": "autoimmune_disease",
    "DT_CHIK_S1": "chik_test1_date",
    "DT_CHIK_S2": "chik_test2_date",
    "RES_CHIKS1": "chik_test1_result",
    "RES_CHIKS2": "chik_test2_result",
    "DT_PRNT": "prnt_date",
    "RESUL_PRNT": "prnt_result",
    "DT_SORO": "serology_date",
    "RESUL_SORO": "serology_result",
    "DT_NS1": "ns1_test_date",
    "RESUL_NS1": "ns1_result",
    "DT_VIRAL": "viral_isolation_date",
    "RESUL_VI_N": "viral_isolation_result",
    "DT_PCR": "pcr_date",
    "RESUL_PCR_": "pcr_result",
    "SOROTIPO": "serotype",
    "HISTOPA_N": "histopathology",
    "IMUNOH_N": "immunohistochemistry",
    "HOSPITALIZ": "hospitalized",
    "DT_INTERNA": "hospitalization_date",
    "UF": "hospital_state",
    "MUNICIPIO": "hospital_municipality",
    "TPAUTOCTO": "autochthonous_case",
    "COUFINF": "infection_state",
    "COPAISINF": "infection_country",
    "COMUNINF": "infection_municipality",
    "CLASSI_FIN": "final_classification",
    "CRITERIO": "confirmation_criteria",
    "DOENCA_TRA": "work_related",
    "CLINC_CHIK": "chik_clinical_form",
    "EVOLUCAO": "case_outcome",
    "DT_OBITO": "death_date",
    "DT_ENCERRA": "case_closure_date",
    "ALRM_HIPOT": "alarm_hypotension",
    "ALRM_PLAQ": "alarm_low_platelets",
    "ALRM_VOM": "alarm_persistent_vomit",
    "ALRM_SANG": "alarm_bleeding",
    "ALRM_HEMAT": "alarm_hematocrit_rise",
    "ALRM_ABDOM": "alarm_abdominal_pain",
    "ALRM_LETAR": "alarm_lethargy",
    "ALRM_HEPAT": "alarm_liver_enlarged",
    "ALRM_LIQ": "alarm_fluid_accumul",
    "DT_ALRM": "alarm_signs_date",
    "GRAV_PULSO": "severe_weak_pulse",
    "GRAV_CONV": "severe_convulsions",
    "GRAV_ENCH": "severe_cap_refill",
    "GRAV_INSUF": "severe_resp_distress",
    "GRAV_TAQUI": "severe_tachycardia",
    "GRAV_EXTRE": "severe_cold_extremities",
    "GRAV_HIPOT": "severe_hypotension",
    "GRAV_HEMAT": "severe_hematemesis",
    "GRAV_MELEN": "severe_melena",
    "GRAV_METRO": "severe_metrorrhagia",
    "GRAV_SANG": "severe_bleeding",
    "GRAV_AST": "severe_ast_elevated",
    "GRAV_MIOC": "severe_myocarditis",
    "GRAV_CONSC": "severe_altered_consc",
    "GRAV_ORGAO": "severe_organ_damage",
    "DT_GRAV": "severity_signs_date",
    "MANI_HEMOR": "hemorrhagic_manifest",
    "EPISTAXE": "nosebleed",
    "GENGIVO": "gum_bleeding",
    "METRO": "metrorrhagia",
    "PETEQUIAS": "petechiae_hemorrh",
    "HEMATURA": "hematuria",
    "SANGRAM": "other_bleeding",
    "PLASMATICO": "plasma_leakage",
    "EVIDENCIA": "hemorrhagic_evidence",
    "PLAQ_MENOR": "platelets_below_100k",
    "CON_FHD": "dengue_hemorrhagic_fever",
    "COMPLICA": "complications",
    "TP_SISTEMA": "system_type",
    "NDUPLIC_N": "duplicate_flag",
    "DT_DIGITA": "data_entry_date",
    "CS_FLXRET": "return_flow_flag",
    "FLXRECEBI": "flow_received",
    "MIGRADO_W": "migrated_from_windows",
}


SEX_LABELS = {"M": "Masculino", "F": "Feminino", "I": "Ignorado"}

PREGNANCY_LABELS = {
    1: "1º Trimestre",
    2: "2º Trimestre",
    3: "3º Trimestre",
    4: "Idade gestacional ignorada",
    5: "Não",
    6: "Não se aplica",
    9: "Ignorado",
}

RACE_LABELS = {
    1: "Branca",
    2: "Preta",
    3: "Amarela",
    4: "Parda",
    5: "Indígena",
    9: "Ignorado",
}

EDUCATION_LABELS = {
    0: "Analfabeto",
    1: "1ª a 4ª série incompleta",
    2: "4ª série completa",
    3: "5ª à 8ª série incompleta",
    4: "Ensino fundamental completo",
    5: "Ensino médio incompleto",
    6: "Ensino médio completo",
    7: "Educação superior incompleta",
    8: "Educação superior completa",
    9: "Ignorado",
    10: "Não se aplica",
}

UF_LABELS = {
    11: "Rondônia",
    12: "Acre",
    13: "Amazonas",
    14: "Roraima",
    15: "Pará",
    16: "Amapá",
    17: "Tocantins",
    21: "Maranhão",
    22: "Piauí",
    23: "Ceará",
    24: "Rio Grande do Norte",
    25: "Paraíba",
    26: "Pernambuco",
    27: "Alagoas",
    28: "Sergipe",
    29: "Bahia",
    31: "Minas Gerais",
    32: "Espírito Santo",
    33: "Rio de Janeiro",
    35: "São Paulo",
    41: "Paraná",
    42: "Santa Catarina",
    43: "Rio Grande do Sul",
    50: "Mato Grosso do Sul",
    51: "Mato Grosso",
    52: "Goiás",
    53: "Distrito Federal",
}

UF_ABBR_LABELS = {
    11: "RO",
    12: "AC",
    13: "AM",
    14: "RR",
    15: "PA",
    16: "AP",
    17: "TO",
    21: "MA",
    22: "PI",
    23: "CE",
    24: "RN",
    25: "PB",
    26: "PE",
    27: "AL",
    28: "SE",
    29: "BA",
    31: "MG",
    32: "ES",
    33: "RJ",
    35: "SP",
    41: "PR",
    42: "SC",
    43: "RS",
    50: "MS",
    51: "MT",
    52: "GO",
    53: "DF",
}

DENGUE_CLASSIFICATION_LABELS = {
    5: "Descartado",
    10: "Dengue",
    11: "Dengue com sinais de alarme",
    12: "Dengue grave",
}

BINARY_CLASSIFICATION_LABELS = {
    0: "Descartado",
    1: "Confirmado",
}

REVERSE_CBO_MAP = {int(code): name for name, code in CBO_MAP.items()}
REVERSE_CBO_MAP[0] = "Ignorado"


def standardize_columns(df):
    return df.rename(columns=COLUMN_RENAME_MAP)


def _numeric_code(value):
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None


def _map_by_numeric_code(value, mapping):
    code = _numeric_code(value)
    if code is None:
        return None
    return mapping.get(code)


def map_cbo_code(value):
    code = _numeric_code(value)
    if code is None:
        return None
    return REVERSE_CBO_MAP.get(code)


def add_sinan_cbo_labels(df, inplace=False):
    if not inplace:
        df = df.copy()

    label_mappings = {
        "pregnancy_status": ("pregnancy_status_label", PREGNANCY_LABELS),
        "race": ("race_label", RACE_LABELS),
        "education_level": ("education_level_label", EDUCATION_LABELS),
        "residence_state": ("residence_state_label", UF_LABELS),
        "notif_state": ("notif_state_label", UF_LABELS),
        "hospital_state": ("hospital_state_label", UF_LABELS),
        "infection_state": ("infection_state_label", UF_LABELS),
    }

    if "sex" in df.columns:
        df["sex_label"] = df["sex"].map(SEX_LABELS)

    for column, (label_column, mapping) in label_mappings.items():
        if column in df.columns:
            df[label_column] = df[column].apply(
                lambda value: _map_by_numeric_code(value, mapping)
            )

    if "occupation_code" in df.columns:
        df["occupation_name"] = df["occupation_code"].apply(map_cbo_code)

    if "final_classification" in df.columns:
        values = set(df["final_classification"].dropna().astype(int).unique())
        mapping = (
            BINARY_CLASSIFICATION_LABELS
            if values.issubset(set(BINARY_CLASSIFICATION_LABELS))
            else DENGUE_CLASSIFICATION_LABELS
        )
        df["final_classification_label"] = df["final_classification"].apply(
            lambda value: _map_by_numeric_code(value, mapping)
        )

    return df
