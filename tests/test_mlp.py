import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dengue_pipeline.models import MLPDiseaseClassifier


class MLPDiseaseClassifierTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        total = 128
        cls.features = pd.DataFrame(
            {
                "education_level": rng.integers(0, 6, total),
                "occupation_code": rng.integers(0, 20, total),
                "residence_state": rng.integers(0, 5, total),
                "residence_municipality": rng.integers(100, 120, total),
                "residence_health_region": rng.integers(
                    1,
                    10,
                    total,
                ).astype(float),
                "age_years": rng.normal(35, 15, total),
                "fever": rng.integers(0, 2, total),
                "rash": rng.integers(0, 2, total),
            }
        )
        cls.features.loc[0, "residence_health_region"] = np.nan
        noise = rng.normal(0, 0.5, total)
        cls.target = (
            cls.features["fever"] + cls.features["rash"] + noise > 1
        ).astype(int)

        cls.model = MLPDiseaseClassifier(
            hidden_layers=(32, 16),
            batch_size=32,
            max_epochs=2,
            patience=2,
            device="cpu",
            verbose=False,
            random_state=42,
        )
        cls.model.fit(cls.features, cls.target)

    def test_predict_proba_has_sklearn_contract(self):
        probabilities = self.model.predict_proba(self.features.iloc[:8])

        self.assertEqual(probabilities.shape, (8, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(8),
            atol=1e-6,
        )

    def test_unknown_categories_are_supported(self):
        unknown = self.features.iloc[[0]].copy()
        unknown["occupation_code"] = 999_999
        unknown["residence_municipality"] = 9_999_999

        probabilities = self.model.predict_proba(unknown)

        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_joblib_round_trip_preserves_predictions(self):
        expected = self.model.predict_proba(self.features.iloc[:8])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlp.joblib"
            joblib.dump(self.model, path)
            loaded = joblib.load(path)
            received = loaded.predict_proba(self.features.iloc[:8])

        np.testing.assert_allclose(expected, received, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
