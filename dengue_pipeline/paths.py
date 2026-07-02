from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DOWNLOAD_DIR = RAW_DATA_DIR / "downloads"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_ANALYSIS_DIR = PROCESSED_DATA_DIR / "analysis"
PROCESSED_ML_DIR = PROCESSED_DATA_DIR / "ml"

DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_FIGURES_DIR = FIGURES_DIR / "modeling"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

DENGUE_YEARS = tuple(range(2014, 2022))
TRAIN_YEARS = tuple(range(2014, 2020))
VALIDATION_YEARS = (2020,)
TEST_YEARS = (2021,)
EXPECTED_SPLIT_ROWS = {
    "train": 7_723_448,
    "validation": 1_331_664,
    "test": 940_304,
}

def analysis_dataset_path(year: int) -> Path:
    return PROCESSED_ANALYSIS_DIR / f"dengue_analysis_{year}.parquet"


def ml_dataset_path(year: int) -> Path:
    return PROCESSED_ML_DIR / f"dengue_ml_{year}.parquet"


SIMULATION_SOURCE_PARQUET = analysis_dataset_path(2021)
SIMULATION_POOL_PATH = PROCESSED_DATA_DIR / "dengue_simulation_pool.parquet"

MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
