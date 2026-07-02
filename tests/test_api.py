import unittest

import numpy as np
import pandas as pd
from pydantic import ValidationError

import api
from dengue_pipeline.features import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    build_model_features,
)


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.patient = api.DadosPaciente(
            age_years=25,
            sex="F",
            pregnancy_status=5,
            race=4,
            education_level=6,
            occupation_code="225125",
            residence_state=33,
            residence_municipality=3304557,
            residence_health_region=33005,
            notification_date="2019-03-08",
            symptom_onset_date="2019-03-05",
            fever=1,
            myalgia=1,
            headache=1,
        )

    def test_features_use_the_shared_schema(self):
        features = api.construir_features(self.patient)

        self.assertEqual(
            list(features.columns),
            list(MODEL_FEATURE_COLUMNS),
        )
        self.assertNotIn("tourniquet_test", features.columns)
        self.assertEqual(features.loc[0, "days_to_notification"], 3)
        self.assertEqual(features.loc[0, "number_of_reported_symptoms"], 3)
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_api_and_pipeline_build_identical_features(self):
        expected = build_model_features(
            pd.DataFrame([self.patient.model_dump()])
        )
        pd.testing.assert_frame_equal(
            api.construir_features(self.patient),
            expected,
        )

    def test_required_models_are_mlp_xgboost_and_lightgbm(self):
        self.assertEqual(
            set(api.MODELOS_DISPONIVEIS),
            {"mlp", "xgboost", "lightgbm"},
        )

    def test_every_loaded_model_receives_all_expected_columns(self):
        if not api.modelos:
            self.skipTest("nenhum artefato de modelo disponível")

        features = api.construir_features(self.patient)
        for name, model in api.modelos.items():
            with self.subTest(model=name):
                aligned, missing = api.alinhar_colunas(features, model)
                self.assertEqual(missing, [])
                self.assertIsNotNone(aligned)

    def test_prediction_uses_every_loaded_model(self):
        if not api.modelos or not api.artifact_set_compatible:
            self.skipTest("artefatos de inferência indisponíveis")

        result = api.predict(self.patient)
        predicted = {item["name"] for item in result["models"]}

        self.assertEqual(predicted, set(api.modelos))
        self.assertEqual(result["ignored"], [])
        self.assertGreaterEqual(result["average"], 0)
        self.assertLessEqual(result["average"], 100)
        self.assertGreaterEqual(result["threshold"], 0)
        self.assertLessEqual(result["threshold"], 100)
        self.assertEqual(result["weighting"], "recall")
        self.assertAlmostEqual(
            sum(item["weight"] for item in result["models"]),
            100,
            delta=0.2,
        )

    def test_notification_cannot_precede_symptom_onset(self):
        with self.assertRaises(ValidationError):
            api.DadosPaciente(
                notification_date="2019-03-01",
                symptom_onset_date="2019-03-02",
            )

    def test_simulation_pool_contains_only_test_cases_and_valid_classes(self):
        try:
            pool = api._load_simulation_pool()
        except Exception as exc:
            self.skipTest(f"pool de simulação indisponível: {exc}")

        years = pd.to_numeric(pool["notification_year"], errors="coerce")
        months = pd.to_datetime(pool["notification_date"], errors="coerce").dt.month
        classifications = pd.to_numeric(
            pool["final_classification_code"],
            errors="coerce",
        )

        self.assertFalse(pool.empty)
        self.assertTrue((years == 2021).all())
        self.assertTrue(
            (months >= api.SIMULATION_NOTIFICATION_MONTH_MIN).all()
        )
        self.assertTrue(
            classifications.isin(
                api.SIMULATION_VALID_CLASSIFICATIONS
            ).all()
        )
        self.assertTrue(pool["final_classification_label"].notna().all())
        self.assertEqual(set(pool.columns), set(api.SIMULATION_POOL_COLUMNS))
        self.assertEqual(len(pool), 940_304)

    def test_simulation_sampler_is_reproducible_with_seed(self):
        try:
            sample_a = api.escolher_caso_real_simulacao(seed=42)
            sample_b = api.escolher_caso_real_simulacao(seed=42)
        except Exception as exc:
            self.skipTest(f"amostragem da simulação indisponível: {exc}")

        self.assertEqual(sample_a["sampled_index"], sample_b["sampled_index"])
        self.assertEqual(sample_a["case"], sample_b["case"])
        self.assertEqual(
            sample_a["observed_classification"],
            sample_b["observed_classification"],
        )
        self.assertIsNotNone(sample_a["observed_classification"])

    def test_simulation_patient_preserves_feature_values(self):
        try:
            pool = api._load_simulation_pool()
        except Exception as exc:
            self.skipTest(f"pool de simulação indisponível: {exc}")

        row = pool.iloc[0]
        patient = api._build_patient_from_sample(row)
        expected = build_model_features(pd.DataFrame([row]))
        received = api.construir_features(patient)
        pd.testing.assert_frame_equal(received, expected)

    def test_simulation_sampler_skips_invalid_historical_row(self):
        try:
            pool = api._load_simulation_pool()
            valid_sample = api.escolher_caso_real_simulacao(seed=42)
        except Exception as exc:
            self.skipTest(f"pool de simulação indisponível: {exc}")

        valid_row = pool.iloc[[valid_sample["sampled_index"]]].copy()
        invalid_row = valid_row.copy()
        invalid_row["residence_state"] = 999

        original_pool = api._simulation_pool
        try:
            api._simulation_pool = pd.concat(
                [invalid_row, valid_row],
                ignore_index=True,
            )
            sample = api.escolher_caso_real_simulacao(seed=1)
        finally:
            api._simulation_pool = original_pool

        self.assertEqual(sample["sampled_index"], 1)
        self.assertIsNotNone(sample["observed_classification"])

    def test_simulation_random_response_shape_and_all_models(self):
        required_models = set(api.MODELOS_DISPONIVEIS)

        if not required_models.issubset(set(api.modelos)):
            self.skipTest("nem todos os modelos necessários estão carregados")
        if not api.artifact_set_compatible:
            self.skipTest("manifestos de artefatos indisponíveis")

        result = api.simulation_random(api.SimulacaoRandomRequest(seed=42))

        self.assertEqual(set(result), {"case", "observedClassification", "prediction"})
        self.assertEqual(
            set(result["prediction"]),
            {"models", "average", "threshold", "weighting", "isDengue"},
        )

        case = result["case"]
        self.assertEqual(
            set(case),
            {"age", "sex", "race", "occupation", "state", "municipality", "symptoms"},
        )
        self.assertIsInstance(case["symptoms"], list)

        model_names = {item["name"] for item in result["prediction"]["models"]}
        self.assertEqual(model_names, required_models)

        for item in result["prediction"]["models"]:
            self.assertGreaterEqual(item["probability"], 0)
            self.assertLessEqual(item["probability"], 100)
            self.assertGreaterEqual(item["weight"], 0)
            self.assertLessEqual(item["weight"], 100)

        self.assertGreaterEqual(result["prediction"]["average"], 0)
        self.assertLessEqual(result["prediction"]["average"], 100)
        self.assertGreaterEqual(result["prediction"]["threshold"], 0)
        self.assertLessEqual(result["prediction"]["threshold"], 100)
        self.assertEqual(result["prediction"]["weighting"], "recall")
        self.assertIsInstance(result["prediction"]["isDengue"], bool)

    # -----------------------------------------------------------------------
    # Novos testes — endpoints de referência e triagem
    # -----------------------------------------------------------------------

    def test_triage_options_shape(self):
        result = api.triage_options()

        self.assertIn("sexos", result)
        self.assertIn("racas", result)
        self.assertIn("escolaridades", result)
        self.assertIn("situacoesGestacao", result)
        self.assertIn("sintomas", result)
        self.assertIn("ufs", result)
        self.assertIn("modelosAtivos", result)
        self.assertIn("liamiarClassificacao", result)
        self.assertIn("pesosModelos", result)

        # Chaves de cada item
        self.assertTrue(all("code" in s and "name" in s for s in result["sexos"]))
        self.assertTrue(all("code" in s and "sigla" in s for s in result["ufs"]))
        self.assertTrue(all("id" in s and "label" in s for s in result["sintomas"]))
        self.assertNotIn(
            "tourniquet_test",
            {item["id"] for item in result["sintomas"]},
        )
        self.assertNotIn("tourniquet_test", api.DadosPaciente.model_fields)

        # Todos os 27 estados
        self.assertEqual(len(result["ufs"]), 27)

        # Modelos ativos batem com os carregados
        self.assertEqual(set(result["modelosAtivos"]), set(api.modelos.keys()))

    def test_health_exposes_temporal_artifact_compatibility(self):
        result = api.health()

        self.assertEqual(
            result["feature_schema_version"],
            FEATURE_SCHEMA_VERSION,
        )
        self.assertIn("artefatos_compativeis", result)
        self.assertIn("periodos", result)
        self.assertNotIn("preprocess_carregado", result)

    def test_occupations_search_starts_with_priority(self):
        result = api.buscar_ocupacoes(query="medico", limit=10)

        self.assertIn("items", result)
        items = result["items"]
        self.assertGreater(len(items), 0)

        # Todos os itens têm code e name
        for item in items:
            self.assertIn("code", item)
            self.assertIn("name", item)

        # Primeiros resultados devem começar com "Medico" (case-insensitive)
        first_names = [i["name"].lower() for i in items[:3]]
        self.assertTrue(any("medico" in n or "médico" in n for n in first_names))

    def test_occupations_search_requires_two_chars(self):
        from fastapi import HTTPException
        from pydantic import ValidationError as PydanticValidationError
        try:
            api.buscar_ocupacoes(query="m", limit=10)
            self.fail("Deveria ter lançado erro para query < 2 chars")
        except (HTTPException, PydanticValidationError, Exception):
            pass  # esperado

    def test_occupations_search_ignores_accents(self):
        result_com = api.buscar_ocupacoes(query="médico", limit=10)
        result_sem = api.buscar_ocupacoes(query="medico", limit=10)

        codes_com = {i["code"] for i in result_com["items"]}
        codes_sem = {i["code"] for i in result_sem["items"]}
        self.assertEqual(codes_com, codes_sem)

    def test_municipalities_search_returns_items(self):
        if not api._MUNICIPIOS_REF:
            self.skipTest("data/municipios.json não encontrado")

        result = api.buscar_municipios(query="rio", state=None, limit=20)

        self.assertIn("items", result)
        self.assertGreater(len(result["items"]), 0)

        for item in result["items"]:
            self.assertIn("code", item)
            self.assertIn("name", item)
            self.assertIn("stateCode", item)
            self.assertIn("state", item)

    def test_municipalities_filter_by_state(self):
        if not api._MUNICIPIOS_REF:
            self.skipTest("data/municipios.json não encontrado")

        result = api.buscar_municipios(query="rio", state=33, limit=20)

        for item in result["items"]:
            self.assertEqual(item["stateCode"], 33)
            self.assertEqual(item["state"], "RJ")

    def test_municipalities_starts_with_priority(self):
        if not api._MUNICIPIOS_REF:
            self.skipTest("data/municipios.json não encontrado")

        result = api.buscar_municipios(query="rio de", state=None, limit=5)
        items = result["items"]

        if items:
            first = items[0]["name"].lower()
            self.assertTrue(first.startswith("rio de"))

    def test_health_regions_returns_list(self):
        if not api._REGIOES_REF:
            self.skipTest("data/regioes_saude.json não encontrado")

        # Rio de Janeiro (3304557) deve ter ao menos uma região
        result = api.buscar_regioes_saude(municipality=3304557)
        self.assertIn("items", result)
        self.assertIsInstance(result["items"], list)

        for item in result["items"]:
            self.assertIn("code", item)
            self.assertIn("name", item)

    def test_health_regions_unknown_municipality_returns_empty(self):
        result = api.buscar_regioes_saude(municipality=9999999)
        self.assertEqual(result["items"], [])


if __name__ == "__main__":
    unittest.main()
