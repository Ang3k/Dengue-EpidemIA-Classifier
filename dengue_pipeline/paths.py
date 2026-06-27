from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PARQUET_DIR = RAW_DATA_DIR / "parquet"
RAW_CSV_DIR = RAW_DATA_DIR / "csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_FIGURES_DIR = FIGURES_DIR / "modeling"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

RAW_DENGUE_PARQUETS = tuple(
    RAW_PARQUET_DIR / f"DENGBR{year}.parquet"
    for year in (17, 18, 19)
)
RAW_DENGUE_CSVS = tuple(
    RAW_CSV_DIR / f"DENGBR{year}.csv"
    for year in (17, 18, 19)
)

ML_DATASET_PATH = PROCESSED_DATA_DIR / "dengue_tratado_ml.parquet"

# Encoders ajustados no treino (OrdinalEncoder de ocupação e UF), usados pela API
# para reproduzir o mesmo pré-processamento do transformar_ml na hora da predição.
ML_PREPROCESS_PATH = MODELS_DIR / "ml_preprocess.joblib"
