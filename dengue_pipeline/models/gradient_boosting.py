from __future__ import annotations

from typing import Literal

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ModelType = Literal["lgbm", "xgb"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GradientBoostingDiseaseClassifier:
    """Interface comum para LightGBM e XGBoost em classificação binária."""

    def __init__(
        self,
        model: ModelType,
        fast_train: bool = True,
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        device: str = "cpu",
        random_state: int = 42,
        tuning_metric: str = "average_precision",
        model_params: dict | None = None,
    ):
        if model not in {"lgbm", "xgb"}:
            raise ValueError("model deve ser 'lgbm' ou 'xgb'")

        self.model_type = model
        self.fast_train = fast_train
        self.device = device
        self.random_state = random_state
        self.tuning_metric = tuning_metric
        self.feature_names: list[str] | None = None
        self.best_params_: dict | None = None
        self.best_score_: float | None = None

        self.base_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        if model_params:
            self.base_params.update(model_params)

        self.model = self._build_model() if fast_train else None

    def _build_model(self, overrides: dict | None = None):
        params = self.base_params.copy()
        if overrides:
            params.update(overrides)

        if self.model_type == "lgbm":
            device_type = "gpu" if self.device in {"gpu", "cuda"} else "cpu"
            defaults = {
                "objective": "binary",
                "device_type": device_type,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            defaults.update(params)
            return lgb.LGBMClassifier(**defaults)

        device = "cuda" if self.device in {"gpu", "cuda"} else "cpu"
        defaults = {
            "objective": "binary:logistic",
            "device": device,
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        defaults.update(params)
        return XGBClassifier(**defaults)

    def _optimize(
        self,
        X,
        y,
        X_validation,
        y_validation,
        n_trials: int,
    ):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    500,
                    step=100,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.1,
                    log=True,
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-8,
                    1.0,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-8,
                    10.0,
                    log=True,
                ),
            }

            if self.model_type == "lgbm":
                params.update(
                    {
                        "num_leaves": trial.suggest_int(
                            "num_leaves",
                            31,
                            255,
                        ),
                        "min_child_samples": trial.suggest_int(
                            "min_child_samples",
                            20,
                            100,
                        ),
                    }
                )
            else:
                params["min_child_weight"] = trial.suggest_int(
                    "min_child_weight",
                    1,
                    10,
                )

            model = self._build_model(params)
            self._fit_with_validation(
                model,
                X,
                y,
                X_validation,
                y_validation,
            )
            probabilities = model.predict_proba(X_validation)[:, 1]
            return average_precision_score(y_validation, probabilities)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        print(f"Melhor {self.tuning_metric}: {study.best_value:.4f}")
        print(f"Melhores parâmetros: {study.best_params}")

        return self._build_model(study.best_params)

    def fit(
        self,
        X_train,
        y_train,
        X_validation=None,
        y_validation=None,
        n_trials: int = 10,
        tuning_sample_size: int | None = 100_000,
        tuning_validation_size: int | None = 200_000,
    ):
        X_train = self._prepare_features(X_train, fit=True)
        y_train = self._prepare_target(y_train)

        if not self.fast_train:
            if X_validation is None or y_validation is None:
                raise ValueError(
                    "Temporal validation data is required when "
                    "fast_train=False"
                )
            X_validation = self._prepare_features(X_validation)
            y_validation = self._prepare_target(y_validation)
            X_tune, y_tune = self._sample_for_tuning(
                X_train,
                y_train,
                tuning_sample_size,
            )
            X_validation_tune, y_validation_tune = self._sample_for_tuning(
                X_validation,
                y_validation,
                tuning_validation_size,
            )
            self.model = self._optimize(
                X_tune,
                y_tune,
                X_validation_tune,
                y_validation_tune,
                n_trials=n_trials,
            )

        if self.fast_train:
            self.model.fit(X_train, y_train)
        else:
            self._fit_with_validation(
                self.model,
                X_train,
                y_train,
                X_validation,
                y_validation,
            )
        return self

    def _fit_with_validation(
        self,
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) -> None:
        if self.model_type == "lgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_validation, y_validation)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            return

        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

    def predict(self, X):
        self._ensure_fitted()
        return self.model.predict(self._prepare_features(X))

    def predict_proba(self, X):
        self._ensure_fitted()
        return self.model.predict_proba(self._prepare_features(X))[:, 1]

    def evaluate(self, X_test, y_test, thresholds=None):
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        probabilities = self.predict_proba(X_test)
        y_true = self._prepare_target(y_test)
        rows = []

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            rows.append(
                {
                    "threshold": threshold,
                    "accuracy": (predictions == y_true).mean(),
                    "precision": precision_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                }
            )

        results = pd.DataFrame(rows)
        print(
            results.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
        return results

    def feature_importance(self, importance_type: str = "gain"):
        self._ensure_fitted()
        if self.model_type == "lgbm":
            values = self.model.booster_.feature_importance(
                importance_type=importance_type,
            )
            importances = pd.Series(values, index=self.feature_names)
        else:
            scores = self.model.get_booster().get_score(
                importance_type=importance_type,
            )
            importances = pd.Series(
                {
                    feature: scores.get(feature, 0.0)
                    for feature in self.feature_names
                }
            )

        return importances.sort_values(ascending=False)

    def plot_feature_importance(
        self,
        top_n: int = 30,
        importance_type: str = "gain",
    ):
        importances = self.feature_importance(importance_type)
        ax = importances.head(top_n).sort_values().plot(
            kind="barh",
            figsize=(10, 8),
        )
        ax.set_title(f"{self.model_type.upper()} - Top {top_n} features")
        ax.set_xlabel("Importância")
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        plt.show()
        return importances

    def _prepare_features(self, X, fit: bool = False):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = frame.columns.astype(str)
        else:
            values = np.asarray(X)
            columns = (
                self.feature_names
                if self.feature_names is not None
                else [f"feature_{index}" for index in range(values.shape[1])]
            )
            frame = pd.DataFrame(values, columns=columns)

        non_numeric = frame.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise TypeError(
                f"Todas as features devem ser numéricas: {non_numeric}"
            )

        frame = frame.replace([np.inf, -np.inf], np.nan)

        if fit:
            self.feature_names = frame.columns.tolist()
        elif self.feature_names is not None:
            missing = set(self.feature_names) - set(frame.columns)
            if missing:
                raise ValueError(f"Features ausentes: {sorted(missing)}")
            frame = frame[self.feature_names]

        return frame

    @staticmethod
    def _prepare_target(y):
        values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        return values.ravel()

    def _sample_for_tuning(self, X, y, sample_size):
        if sample_size is None or sample_size >= len(X):
            return X, y

        X_tune, _, y_tune, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=self.random_state,
        )
        print(
            f"Optuna usando amostra estratificada de {len(X_tune):,} registros."
        )
        return X_tune, y_tune

    def _ensure_fitted(self):
        if self.model is None or self.feature_names is None:
            raise RuntimeError("O modelo precisa ser treinado antes da previsão")
