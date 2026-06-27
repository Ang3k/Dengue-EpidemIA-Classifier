import unittest

import numpy as np
from pydantic import ValidationError

import api


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

    def test_features_include_current_and_legacy_date_columns(self):
        features = api.construir_features(self.patient)

        self.assertEqual(features.loc[0, "notification_month"], 3)
        self.assertEqual(features.loc[0, "symptom_month"], 3)
        self.assertEqual(features.loc[0, "symptom_day"], 5)
        self.assertEqual(features.loc[0, "days_to_notification"], 3)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

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
        if not api.modelos or not api.preprocess:
            self.skipTest("artefatos de inferência indisponíveis")

        result = api.predict(self.patient)
        predicted = {item["name"] for item in result["models"]}

        self.assertEqual(predicted, set(api.modelos))
        self.assertEqual(result["ignored"], [])
        self.assertGreaterEqual(result["average"], 0)
        self.assertLessEqual(result["average"], 100)

    def test_notification_cannot_precede_symptom_onset(self):
        with self.assertRaises(ValidationError):
            api.DadosPaciente(
                notification_date="2019-03-01",
                symptom_onset_date="2019-03-02",
            )


if __name__ == "__main__":
    unittest.main()
