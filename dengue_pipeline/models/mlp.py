from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_CATEGORICAL_COLUMNS = (
    "education_level",
    "occupation_code",
    "residence_state",
    "residence_municipality",
    "residence_health_region",
)


class ArbovirosesMLP(nn.Module):
    """MLP tabular do projeto original, com embeddings para categorias."""

    def __init__(
        self,
        numerical_size: int,
        embedding_sizes: Sequence[tuple[int, int]],
        hidden_layers: Sequence[int] = (1024, 512, 256, 128),
        embedding_dropout: float = 0.1,
        hidden_dropout: float = 0.2,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList(
            nn.Embedding(category_count, embedding_size)
            for category_count, embedding_size in embedding_sizes
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.numerical_normalization = nn.BatchNorm1d(numerical_size)

        input_size = numerical_size + sum(size for _, size in embedding_sizes)
        layers: list[nn.Module] = []
        for layer_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_size, layer_size),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(hidden_dropout),
                ]
            )
            input_size = layer_size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        categorical_features = self.embedding_dropout(
            torch.cat(embedded, dim=1)
        )
        numerical_features = self.numerical_normalization(numerical)
        combined = torch.cat([categorical_features, numerical_features], dim=1)
        return self.output_layer(self.layers(combined)).squeeze(1)


class MLPDiseaseClassifier(ClassifierMixin, BaseEstimator):
    """Wrapper sklearn-like para treino e inferência da MLP tabular.

    O wrapper mantém o formato de DataFrame usado pelos demais modelos, ajusta
    encoding e imputação apenas no subconjunto de treino interno e transfere
    somente cada batch para a GPU.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
        hidden_layers: Sequence[int] = (1024, 512, 256, 128),
        embedding_max_size: int = 50,
        embedding_dropout: float = 0.1,
        hidden_dropout: float = 0.2,
        batch_size: int = 16384,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 150,
        patience: int = 10,
        validation_size: float = 0.1,
        device: str = "auto",
        num_workers: int = 0,
        random_state: int = 42,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.categorical_columns = categorical_columns
        self.hidden_layers = hidden_layers
        self.embedding_max_size = embedding_max_size
        self.embedding_dropout = embedding_dropout
        self.hidden_dropout = hidden_dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_size = validation_size
        self.device = device
        self.num_workers = num_workers
        self.random_state = random_state
        self.threshold = threshold
        self.verbose = verbose

        self.model: ArbovirosesMLP | None = None
        self.categorical_encoder: OrdinalEncoder | None = None
        self.numerical_medians_: pd.Series | None = None
        self.categorical_columns_: list[str] | None = None
        self.numerical_columns_: list[str] | None = None
        self.embedding_sizes_: list[tuple[int, int]] | None = None
        self.feature_names: list[str] | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.classes_ = np.array([0, 1])
        self.best_epoch_: int | None = None
        self.history_: list[dict[str, float]] = []

    def fit(
        self,
        X,
        y,
        X_validation=None,
        y_validation=None,
    ):
        frame = self._prepare_frame(X, fit=True)
        target = self._prepare_target(y)
        self._validate_training_data(frame, target)
        self._seed_everything()

        if (X_validation is None) != (y_validation is None):
            raise ValueError(
                "X_validation and y_validation must be provided together"
            )

        if X_validation is None:
            indices = np.arange(len(frame))
            train_indices, validation_indices = train_test_split(
                indices,
                test_size=self.validation_size,
                stratify=target,
                random_state=self.random_state,
            )
            training_frame = frame.iloc[train_indices]
            training_target = target[train_indices]
            validation_frame = frame.iloc[validation_indices]
            validation_target = target[validation_indices]
        else:
            training_frame = frame
            training_target = target
            validation_frame = self._prepare_validation_frame(X_validation)
            validation_target = self._prepare_target(y_validation)
            self._validate_validation_data(
                validation_frame,
                validation_target,
            )

        self._fit_preprocessor(training_frame)

        train_categorical, train_numerical = self._transform_features(
            training_frame
        )
        validation_categorical, validation_numerical = self._transform_features(
            validation_frame
        )

        training_loader = self._make_loader(
            train_categorical,
            train_numerical,
            training_target,
            shuffle=True,
        )
        validation_loader = self._make_loader(
            validation_categorical,
            validation_numerical,
            validation_target,
            shuffle=False,
        )

        runtime_device = self._resolve_device()
        self.model = self._build_network(runtime_device)

        positive_count = int(training_target.sum())
        negative_count = len(training_target) - positive_count
        positive_weight = negative_count / max(positive_count, 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=runtime_device,
            )
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
        )

        best_state = None
        best_validation_score = float("-inf")
        epochs_without_improvement = 0
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            training_loss = self._train_epoch(
                training_loader,
                criterion,
                optimizer,
                runtime_device,
            )
            validation_loss, validation_pr_auc = self._validation_metrics(
                validation_loader,
                criterion,
                runtime_device,
            )
            scheduler.step(validation_pr_auc)

            learning_rate = optimizer.param_groups[0]["lr"]
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_pr_auc": validation_pr_auc,
                    "learning_rate": float(learning_rate),
                }
            )
            if self.verbose:
                print(
                    f"Época {epoch:03d} | treino: {training_loss:.5f} | "
                    f"val_loss: {validation_loss:.5f} | "
                    f"val_PR-AUC: {validation_pr_auc:.5f} | "
                    f"lr: {learning_rate:.2e}",
                    flush=True,
                )

            if validation_pr_auc > best_validation_score + 1e-6:
                best_validation_score = validation_pr_auc
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"Early stopping na época {epoch}.", flush=True)
                    break

        if best_state is None:
            raise RuntimeError("A MLP não produziu um estado de treinamento válido.")

        self.model.load_state_dict(best_state)
        self.model.to("cpu")
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._ensure_fitted()
        frame = self._prepare_frame(X)
        categorical, numerical = self._transform_features(frame)
        if len(frame) == 0:
            return np.empty((0, 2), dtype=np.float32)

        loader = self._make_loader(
            categorical,
            numerical,
            target=None,
            shuffle=False,
        )
        runtime_device = self._inference_device()
        self.model.to(runtime_device)
        self.model.eval()

        probabilities = []
        with torch.inference_mode():
            for categorical_batch, numerical_batch in loader:
                categorical_batch = categorical_batch.to(
                    runtime_device,
                    non_blocking=True,
                )
                numerical_batch = numerical_batch.to(
                    runtime_device,
                    non_blocking=True,
                )
                logits = self.model(categorical_batch, numerical_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        positive = np.concatenate(probabilities).astype(np.float32, copy=False)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int8)

    def evaluate(self, X, y, thresholds=None) -> pd.DataFrame:
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.4, 0.5, 0.6]

        target = self._prepare_target(y)
        probabilities = self.predict_proba(X)[:, 1]
        rows = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(np.int8)
            rows.append(
                {
                    "threshold": threshold,
                    "accuracy": float((predictions == target).mean()),
                    "precision": precision_score(
                        target,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        target,
                        predictions,
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        target,
                        predictions,
                        zero_division=0,
                    ),
                }
            )

        results = pd.DataFrame(rows)
        if self.verbose:
            print(
                results.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.4f}",
                )
            )
        return results

    def permutation_feature_importance(
        self,
        X,
        y,
        sample_size: int = 2_000,
        n_repeats: int = 5,
        scoring: str = "average_precision",
    ) -> pd.Series:
        frame = self._prepare_frame(X)
        target = self._prepare_target(y)
        if len(frame) > sample_size:
            rng = np.random.default_rng(self.random_state)
            selected = rng.choice(len(frame), size=sample_size, replace=False)
            frame = frame.iloc[selected]
            target = target[selected]

        result = permutation_importance(
            self,
            frame,
            target,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=1,
        )
        return pd.Series(
            result.importances_mean,
            index=frame.columns,
        ).sort_values(ascending=False)

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None and self.feature_names is not None

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model", None)
        state["_serialized_model_state"] = (
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if model is not None
            else None
        )
        state["model"] = None
        return state

    def __setstate__(self, state):
        serialized_model_state = state.pop("_serialized_model_state", None)
        self.__dict__.update(state)
        self.model = None
        if serialized_model_state is not None:
            self.model = self._build_network(torch.device("cpu"))
            self.model.load_state_dict(serialized_model_state)
            self.model.eval()

    def _prepare_frame(self, X, fit: bool = False) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            if fit:
                raise TypeError("O primeiro fit da MLP exige um pandas DataFrame.")
            frame = pd.DataFrame(X, columns=self.feature_names)

        if fit:
            self.feature_names = frame.columns.astype(str).tolist()
            self.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
            self.n_features_in_ = len(self.feature_names)
            frame.columns = self.feature_names
            return frame

        self._ensure_fitted()
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Features ausentes para a MLP: {missing}")
        return frame[self.feature_names]

    def _prepare_validation_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=self.feature_names)
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Validation features missing for the MLP: {missing}"
            )
        return frame[self.feature_names]

    @staticmethod
    def _prepare_target(y) -> np.ndarray:
        target = np.asarray(y).reshape(-1)
        if pd.isna(target).any():
            raise ValueError("O target da MLP contém valores ausentes.")
        return target.astype(np.int64, copy=False)

    def _validate_training_data(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
    ) -> None:
        if len(frame) != len(target):
            raise ValueError("X e y possuem quantidades diferentes de registros.")
        if len(frame) < 20:
            raise ValueError("A MLP exige pelo menos 20 registros para treinamento.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size deve estar entre 0 e 1.")
        classes = set(np.unique(target))
        if classes != {0, 1}:
            raise ValueError(f"O target deve conter apenas 0 e 1; recebido: {classes}")

        unavailable = sorted(
            set(self.categorical_columns) - set(frame.columns)
        )
        if unavailable:
            raise ValueError(
                f"Features categóricas ausentes para a MLP: {unavailable}"
            )

    def _validate_validation_data(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
    ) -> None:
        if len(frame) != len(target):
            raise ValueError(
                "Validation X and y have different row counts"
            )
        if len(frame) == 0:
            raise ValueError("Temporal validation data cannot be empty")
        classes = set(np.unique(target))
        if not classes.issubset({0, 1}):
            raise ValueError(
                f"Validation target must contain only 0 and 1: {classes}"
            )

    def _fit_preprocessor(self, frame: pd.DataFrame) -> None:
        self.categorical_columns_ = list(self.categorical_columns)
        self.numerical_columns_ = [
            column
            for column in self.feature_names
            if column not in self.categorical_columns_
        ]

        categorical = self._categorical_frame(frame)
        self.categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        self.categorical_encoder.fit(categorical)

        self.embedding_sizes_ = []
        for categories in self.categorical_encoder.categories_:
            category_count = len(categories) + 1
            embedding_size = min(
                self.embedding_max_size,
                (category_count // 2) + 1,
            )
            self.embedding_sizes_.append(
                (category_count, embedding_size)
            )

        numerical = self._numerical_frame(frame)
        self.numerical_medians_ = (
            numerical.replace([np.inf, -np.inf], np.nan)
            .median()
            .fillna(0.0)
        )

    def _transform_features(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.categorical_encoder is None or self.numerical_medians_ is None:
            raise RuntimeError("O pré-processamento da MLP ainda não foi ajustado.")

        categorical = (
            self.categorical_encoder.transform(
                self._categorical_frame(frame)
            )
            + 1
        ).astype(np.int64, copy=False)

        numerical = self._numerical_frame(frame)
        numerical = (
            numerical.replace([np.inf, -np.inf], np.nan)
            .fillna(self.numerical_medians_)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        return categorical, numerical

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[self.categorical_columns_ or self.categorical_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-1.0)
            .astype(np.float64)
        )

    def _numerical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self.numerical_columns_ or []
        return frame[columns].apply(pd.to_numeric, errors="coerce")

    def _build_network(self, device: torch.device) -> ArbovirosesMLP:
        if self.embedding_sizes_ is None or self.numerical_columns_ is None:
            raise RuntimeError("Metadados da arquitetura da MLP indisponíveis.")
        model = ArbovirosesMLP(
            numerical_size=len(self.numerical_columns_),
            embedding_sizes=self.embedding_sizes_,
            hidden_layers=self.hidden_layers,
            embedding_dropout=self.embedding_dropout,
            hidden_dropout=self.hidden_dropout,
        )
        return model.to(device)

    def _make_loader(
        self,
        categorical: np.ndarray,
        numerical: np.ndarray,
        target: np.ndarray | None,
        shuffle: bool,
    ) -> DataLoader:
        tensors = [
            torch.from_numpy(np.ascontiguousarray(categorical)),
            torch.from_numpy(np.ascontiguousarray(numerical)),
        ]
        if target is not None:
            tensors.append(
                torch.from_numpy(
                    np.ascontiguousarray(target, dtype=np.float32)
                )
            )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(categorical) > self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            generator=generator if shuffle else None,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for categorical, numerical, target in loader:
            categorical = categorical.to(device, non_blocking=True)
            numerical = numerical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(categorical, numerical)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        return total_loss / len(loader.dataset)

    def _validation_metrics(
        self,
        loader: DataLoader,
        criterion,
        device: torch.device,
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.inference_mode():
            for categorical, numerical, target in loader:
                categorical = categorical.to(device, non_blocking=True)
                numerical = numerical.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = self.model(categorical, numerical)
                total_loss += criterion(logits, target).item() * len(target)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(target.cpu().numpy())
        loss = total_loss / len(loader.dataset)
        pr_auc = float(
            average_precision_score(
                np.concatenate(targets),
                np.concatenate(probabilities),
            )
        )
        return loss, pr_auc

    def _resolve_device(self) -> torch.device:
        requested = self.device.lower()
        if requested in {"auto", "gpu"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' foi solicitado, mas o PyTorch não detectou CUDA."
            )
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device deve ser 'auto', 'cpu', 'cuda' ou 'gpu'.")
        return torch.device(requested)

    def _inference_device(self) -> torch.device:
        """Device para inferência: usa CUDA se houver, senão CPU.

        Diferente de ``_resolve_device``, nunca levanta erro — um modelo
        treinado na GPU precisa prever normalmente em máquinas sem CUDA
        (por exemplo, a API em produção).
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _seed_everything(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _ensure_fitted(self) -> None:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("A MLP ainda não foi treinada.")
