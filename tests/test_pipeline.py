import json
from pathlib import Path
import unittest
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from dengue_pipeline.cleaner import (
    ANALYSIS_COLUMNS,
    REQUIRED_STANDARDIZED_COLUMNS,
    DengueDataCleaner,
    harmonize_final_classification,
)
from dengue_pipeline.features import (
    DATASET_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SYMPTOM_COLUMNS,
)
from dengue_pipeline.paths import (
    DENGUE_YEARS,
    RAW_DOWNLOAD_DIR,
    TEST_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
    analysis_dataset_path,
    ml_dataset_path,
)
from dengue_pipeline.sinan_mappings import COLUMN_RENAME_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "dengue_manifest.json"


def raw_row(year: int, classification: object = 10) -> dict:
    reverse = {
        standardized: raw
        for raw, standardized in COLUMN_RENAME_MAP.items()
    }
    values = {
        standardized: ""
        for standardized in REQUIRED_STANDARDIZED_COLUMNS
    }
    values.update(
        {
            "age": 4025,
            "sex": "F",
            "pregnancy_status": 5,
            "race": 4,
            "education_level": 6,
            "occupation_code": 225125,
            "residence_state": 33,
            "residence_municipality": 3304557,
            "residence_health_region": 33005,
            "disease_code": "A90",
            "notification_date": f"{year}-03-08",
            "notification_year": year,
            "notification_epi_week": int(f"{year}10"),
            "notif_municipality": 3304557,
            "notif_health_region": 33005,
            "health_facility": 1234567,
            "symptom_onset_date": f"{year}-03-05",
            "symptom_epi_week": int(f"{year}10"),
            "hospitalized": 2,
            "hospital_state": 33,
            "final_classification": classification,
        }
    )
    values.update({symptom: 2 for symptom in SYMPTOM_COLUMNS})
    return {
        reverse[standardized]: value
        for standardized, value in values.items()
    }


class ClassificationMappingTestCase(unittest.TestCase):
    def test_legacy_mapping_2014_to_2016(self):
        raw = pd.Series([1, 2, 3, 4, 10, 11, 12, 5, 0, 8, 9, None, 13])
        for year in (2014, 2015, 2016):
            with self.subTest(year=year):
                mapped = harmonize_final_classification(raw, year)
                self.assertEqual(mapped.iloc[:7].tolist(), [1] * 7)
                self.assertEqual(mapped.iloc[7], 0)
                self.assertTrue(mapped.iloc[8:].isna().all())

    def test_modern_mapping_2017_to_2021(self):
        raw = pd.Series([10, 11, 12, 5, 1, 2, 3, 4, 0, 8, 9, None, 13])
        for year in (2017, 2018, 2019, 2020, 2021):
            with self.subTest(year=year):
                mapped = harmonize_final_classification(raw, year)
                self.assertEqual(mapped.iloc[:3].tolist(), [1, 1, 1])
                self.assertEqual(mapped.iloc[3], 0)
                self.assertTrue(mapped.iloc[4:].isna().all())


class FeatureSchemaTestCase(unittest.TestCase):
    def test_years_and_chunks_produce_the_same_schema(self):
        schemas = []
        for year in (2014, 2019, 2020, 2021):
            raw = pd.DataFrame(
                [
                    raw_row(year, 10),
                    raw_row(year, 5),
                    raw_row(year, 8),
                ]
            )
            analysis = DengueDataCleaner.transformar_analise_chunk(raw, year)
            ml = DengueDataCleaner.transformar_ml(analysis)
            schemas.append(tuple(ml.columns))

            self.assertEqual(tuple(analysis.columns), ANALYSIS_COLUMNS)
            self.assertEqual(len(analysis), 2)
            self.assertEqual(set(analysis["final_classification"]), {0, 1})

        self.assertEqual(len(set(schemas)), 1)
        self.assertEqual(
            schemas[0],
            DATASET_METADATA_COLUMNS + MODEL_FEATURE_COLUMNS,
        )
        self.assertNotIn("tourniquet_test", schemas[0])

    def test_symptom_unknown_is_not_treated_as_no(self):
        row = raw_row(2021)
        row["FEBRE"] = 1
        row["MIALGIA"] = 2
        row["CEFALEIA"] = 9
        row["EXANTEMA"] = ""
        analysis = DengueDataCleaner.transformar_analise_chunk(
            pd.DataFrame([row]),
            2021,
        )
        features = DengueDataCleaner.transformar_ml(analysis)

        self.assertEqual(features.loc[0, "fever"], 1)
        self.assertEqual(features.loc[0, "myalgia"], 0)
        self.assertTrue(pd.isna(features.loc[0, "headache"]))
        self.assertTrue(pd.isna(features.loc[0, "rash"]))
        self.assertEqual(features.loc[0, "number_of_symptoms"], 1)
        self.assertEqual(
            features.loc[0, "number_of_reported_symptoms"],
            10,
        )

    def test_birth_date_is_not_required(self):
        reverse = {
            standardized: raw
            for raw, standardized in COLUMN_RENAME_MAP.items()
        }
        required_raw = {
            reverse[column]
            for column in REQUIRED_STANDARDIZED_COLUMNS
        }
        self.assertNotIn("DT_NASC", required_raw)
        self.assertNotIn("ANO_NASC", required_raw)


class TemporalAndManifestTestCase(unittest.TestCase):
    def test_temporal_periods_are_disjoint(self):
        train = set(TRAIN_YEARS)
        validation = set(VALIDATION_YEARS)
        test = set(TEST_YEARS)

        self.assertEqual(TRAIN_YEARS, (2017, 2018, 2019))
        self.assertEqual(VALIDATION_YEARS, (2020,))
        self.assertEqual(TEST_YEARS, (2021,))
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_2021_is_opened_only_after_calibration_is_frozen(self):
        training_source = (
            PROJECT_ROOT / "scripts" / "train_models.py"
        ).read_text(encoding="utf-8")
        evaluation_source = (
            PROJECT_ROOT / "scripts" / "evaluate_models.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "load_ml_years(TEST_YEARS)",
            training_source,
        )
        config_position = evaluation_source.index("ensemble_config = {")
        test_load_position = evaluation_source.index(
            "test_dataset = load_ml_years(TEST_YEARS)"
        )
        self.assertLess(config_position, test_load_position)

    def test_manifest_has_pinned_official_counts_and_schema(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            tuple(int(year) for year in manifest["years"]),
            DENGUE_YEARS,
        )
        self.assertTrue(
            all(
                len(resource["sha256"]) == 64
                for resource in manifest["years"].values()
            )
        )
        self.assertEqual(
            manifest["schema"]["model_feature_columns"],
            list(MODEL_FEATURE_COLUMNS),
        )
        self.assertNotIn(
            "tourniquet_test",
            manifest["schema"]["model_feature_columns"],
        )
        self.assertEqual(
            sum(item["raw_rows"] for item in manifest["years"].values()),
            11_441_770,
        )

        labeled_by_year = {}
        for raw_year, resource in manifest["years"].items():
            year = int(raw_year)
            positives = (
                (1, 2, 3, 4, 10, 11, 12)
                if year <= 2016
                else (10, 11, 12)
            )
            labeled_by_year[year] = resource["class_counts"]["5"] + sum(
                resource["class_counts"].get(str(code), 0)
                for code in positives
            )
        self.assertEqual(sum(labeled_by_year.values()), 9_995_416)
        self.assertEqual(
            sum(labeled_by_year[year] for year in TRAIN_YEARS),
            2_874_235,
        )
        self.assertEqual(labeled_by_year[2020], 1_331_664)
        self.assertEqual(labeled_by_year[2021], 940_304)

    def test_downloaded_headers_cover_2014_2019_2020_2021(self):
        paths = [
            RAW_DOWNLOAD_DIR / f"DENGBR{str(year)[-2:]}.csv.zip"
            for year in (2014, 2019, 2020, 2021)
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("official ZIP files are not available locally")

        for path in paths:
            with zipfile.ZipFile(path) as archive:
                member = archive.namelist()[0]
                with archive.open(member) as file:
                    columns = set(
                        pd.read_csv(
                            file,
                            nrows=0,
                            encoding="latin1",
                        ).columns
                    )
            self.assertIn("NU_IDADE_N", columns)
            self.assertIn("CLASSI_FIN", columns)

        with zipfile.ZipFile(paths[-1]) as archive:
            with archive.open(archive.namelist()[0]) as file:
                columns_2021 = set(
                    pd.read_csv(
                        file,
                        nrows=0,
                        encoding="latin1",
                    ).columns
                )
        self.assertNotIn("DT_NASC", columns_2021)

    def test_processed_partitions_have_exact_schema_and_both_classes(self):
        paths = [
            ml_dataset_path(year)
            for year in (2014, 2019, 2020, 2021)
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("processed partitions are not available locally")

        expected = list(DATASET_METADATA_COLUMNS + MODEL_FEATURE_COLUMNS)
        for year, path in zip((2014, 2019, 2020, 2021), paths):
            parquet = pq.ParquetFile(path)
            self.assertEqual(parquet.schema.names, expected)
            classes = set(
                pd.read_parquet(
                    path,
                    columns=["final_classification"],
                )["final_classification"]
            )
            self.assertEqual(classes, {0, 1}, year)
            self.assertTrue(analysis_dataset_path(year).exists())


if __name__ == "__main__":
    unittest.main()
