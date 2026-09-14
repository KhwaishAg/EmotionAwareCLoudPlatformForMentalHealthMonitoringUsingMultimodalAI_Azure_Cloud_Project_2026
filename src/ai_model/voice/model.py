"""
Voice Modality — Machine Learning Models: Logistic Regression & Random Forest

Implements classical machine learning baseline and tree-ensemble models for voice-derived
stress/risk assessment using fixed-length acoustic feature representations (41D default).

Components:
    - BaseVoiceClassifier: Common base class providing feature/target separation,
      rigorous input validation, standard interfaces, and probability utilities.
    - VoiceLogisticRegression: Scikit-learn based Logistic Regression baseline classifier
      with standard feature scaling, configurable random_state and max_iter,
      and support for binary and multiclass target labels.
    - VoiceRandomForest: Scikit-learn based Random Forest classifier with configurable
      n_estimators, max_depth, min_samples_split, class_weight, and random_state.
    - separate_features_and_target: Utility function to decouple acoustic feature columns
      from metadata identifiers (participant_id) and target labels.
    - Input validation: Column integrity, numeric dtype checks, NaN/Inf checks,
      and sample count verification.
    - Full fit / predict / predict_proba / predict_proba_dict interface.

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (ModelConfig, FeatureConfig)
    - Input: pandas DataFrame with 41 voice features (or numpy ndarray) and target
    - No hardcoded target labels or dataset names.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import MODEL_CONFIG, ModelConfig
from .features import COMBINED_FEATURE_NAMES
from .recording import AudioError


# Generic type for subclass method chaining
T = TypeVar("T", bound="BaseVoiceClassifier")


# ===========================================================================
# Exceptions
# ===========================================================================

class ModelError(AudioError):
    """Base exception for voice model errors."""
    pass


class ModelNotFittedError(ModelError):
    """Raised when predict or predict_proba is called before fitting."""
    pass


class ModelInputError(ModelError):
    """Raised when input features, columns, or target labels are invalid."""
    pass


class ModelSerializationError(ModelError):
    """Raised when a voice model cannot be serialized or saved to disk."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a specified model file does not exist on disk."""
    pass


class ModelCorruptError(ModelError):
    """Raised when a model file is corrupted, empty, or fails integrity checks."""
    pass



# ===========================================================================
# Helper Functions
# ===========================================================================

def separate_features_and_target(
    data: pd.DataFrame,
    target_column: str = MODEL_CONFIG.label_column,
    id_column: Optional[str] = MODEL_CONFIG.participant_id_column,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Decouple acoustic feature columns from metadata identifiers and target labels.

    Parameters:
        data: pandas DataFrame containing feature columns, optional participant ID, and target label.
        target_column: Column name of the prediction target (default: 'target').
        id_column: Column name of the participant identifier to exclude (default: 'participant_id').
        feature_columns: Optional explicit list of feature columns to retain.
                         If None, drops id_column and target_column from data.

    Returns:
        Tuple of (X, y) where:
            X: pd.DataFrame containing feature columns only.
            y: pd.Series containing target labels.

    Raises:
        ModelInputError: If data is empty, target_column is missing, or no feature columns remain.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data)}.")

    if data.empty:
        raise ModelInputError("Input DataFrame is empty (0 rows).")

    if target_column not in data.columns:
        raise ModelInputError(
            f"Target column '{target_column}' not found in DataFrame columns: {list(data.columns)}."
        )

    y = data[target_column].copy()

    if feature_columns is not None:
        missing_cols = [c for c in feature_columns if c not in data.columns]
        if missing_cols:
            raise ModelInputError(
                f"Specified feature columns not found in DataFrame: {missing_cols}."
            )
        X = data[feature_columns].copy()
    else:
        exclude_cols = [target_column]
        if id_column and id_column in data.columns:
            exclude_cols.append(id_column)
        X = data.drop(columns=exclude_cols).copy()

    if X.shape[1] == 0:
        raise ModelInputError("No feature columns remaining after removing target and ID columns.")

    return X, y


# ===========================================================================
# Base Voice Classifier
# ===========================================================================

class BaseVoiceClassifier:
    """
    Common base class for voice modality machine learning classifiers.

    Provides:
        - Common initialization and configuration handling
        - Rigorous validation of input matrices, feature columns, and target labels
        - Feature matrix standardization via optional StandardScaler
        - Unified fit, predict, predict_proba, and predict_proba_dict interfaces
    """

    def __init__(
        self,
        scale_features: bool = False,
        config: ModelConfig = MODEL_CONFIG,
        expected_feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize BaseVoiceClassifier.

        Parameters:
            scale_features: Whether to scale features using StandardScaler (default: False).
            config: Centralized ModelConfig settings.
            expected_feature_names: Optional explicit feature names expected during fit/predict.
        """
        self.scale_features = scale_features
        self.config = config
        self.expected_feature_names = (
            list(expected_feature_names) if expected_feature_names is not None else None
        )

        self.scaler_: Optional[StandardScaler] = StandardScaler() if scale_features else None
        self.classifier_: Any = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check if the classifier has been fitted."""
        return self.is_fitted_

    def _check_is_fitted(self) -> None:
        """Verify that the model has been fitted."""
        if not self.is_fitted_ or self.classes_ is None or self.classifier_ is None:
            raise ModelNotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with training data before using 'predict' or 'predict_proba'."
            )

    def _validate_features(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
        is_fit: bool = False,
    ) -> np.ndarray:
        """
        Validate feature inputs and convert to 2D float32 numpy array.

        Parameters:
            X: Input features (DataFrame, ndarray, dict, or list).
            is_fit: True if called during fit(), False if called during predict().

        Returns:
            2D numpy array of float32 features.

        Raises:
            ModelInputError: If dimensions, columns, or values are invalid.
        """
        # Handle dict input (e.g. CombinedFeatureDict or single row dict)
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ModelInputError("Input feature DataFrame is empty.")

            # If fitting, learn or verify feature columns
            if is_fit:
                if self.expected_feature_names is not None:
                    missing = [c for c in self.expected_feature_names if c not in X.columns]
                    if missing:
                        raise ModelInputError(
                            f"Input DataFrame is missing expected feature columns: {missing}."
                        )
                    self.feature_names_ = list(self.expected_feature_names)
                    X_ordered = X[self.feature_names_]
                else:
                    self.feature_names_ = list(X.columns)
                    X_ordered = X
            else:
                # During prediction, ensure columns match fitted feature names
                if self.feature_names_ is not None:
                    missing = [c for c in self.feature_names_ if c not in X.columns]
                    if missing:
                        raise ModelInputError(
                            f"Prediction input is missing fitted feature columns: {missing}."
                        )
                    X_ordered = X[self.feature_names_]
                else:
                    X_ordered = X

            # Check numeric types
            for col in X_ordered.columns:
                if not np.issubdtype(X_ordered[col].dtype, np.number):
                    raise ModelInputError(
                        f"Feature column '{col}' has non-numeric dtype '{X_ordered[col].dtype}'."
                    )

            # Check for NaN / Inf
            if X_ordered.isna().any().any():
                raise ModelInputError("Input features contain NaN values.")

            arr = X_ordered.to_numpy(dtype=np.float32)

        elif isinstance(X, (np.ndarray, list)):
            arr = np.asarray(X, dtype=np.float32)
            if arr.size == 0:
                raise ModelInputError("Input feature array is empty.")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ModelInputError(
                    f"Feature array must be 2D of shape (n_samples, n_features), got {arr.ndim}D array."
                )

            if not is_fit and self.n_features_in_ is not None and arr.shape[1] != self.n_features_in_:
                raise ModelInputError(
                    f"Number of features ({arr.shape[1]}) does not match fitted features ({self.n_features_in_})."
                )

            if is_fit and self.feature_names_ is None:
                if self.expected_feature_names is not None and len(self.expected_feature_names) == arr.shape[1]:
                    self.feature_names_ = list(self.expected_feature_names)
                elif arr.shape[1] == len(COMBINED_FEATURE_NAMES):
                    self.feature_names_ = list(COMBINED_FEATURE_NAMES)
                else:
                    self.feature_names_ = [f"feature_{i}" for i in range(arr.shape[1])]
        else:
            raise TypeError(
                f"Unsupported features type: {type(X)}. "
                "Expected pandas.DataFrame, numpy.ndarray, dict, or list."
            )

        if not np.all(np.isfinite(arr)):
            raise ModelInputError("Feature matrix contains non-finite (NaN or Inf) values.")

        return arr

    def _validate_target(
        self,
        y: Union[pd.Series, np.ndarray, list],
        n_samples: int,
    ) -> np.ndarray:
        """
        Validate target labels.

        Parameters:
            y: Target label vector.
            n_samples: Expected number of samples.

        Returns:
            1D numpy array of target labels.
        """
        if y is None:
            raise ModelInputError("Target labels 'y' cannot be None.")

        if isinstance(y, pd.Series):
            if y.isna().any():
                raise ModelInputError("Target vector contains null/NaN values.")
            arr_y = y.to_numpy()
        else:
            arr_y = np.asarray(y)

        if arr_y.size == 0:
            raise ModelInputError("Target labels 'y' is empty.")

        if arr_y.ndim != 1:
            arr_y = arr_y.ravel()

        if len(arr_y) != n_samples:
            raise ModelInputError(
                f"Sample count mismatch: features has {n_samples} samples but target has {len(arr_y)}."
            )

        # Ensure no null/empty strings
        for val in arr_y:
            if pd.isna(val) or val is None or (isinstance(val, str) and not val.strip()):
                raise ModelInputError(f"Target contains invalid or empty label: '{val}'.")

        distinct_classes = np.unique(arr_y)
        if len(distinct_classes) < 2:
            raise ModelInputError(
                f"Target must contain at least 2 distinct classes for classification, "
                f"got {len(distinct_classes)}: {list(distinct_classes)}."
            )

        return arr_y

    def fit(
        self: T,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, list]] = None,
        target_column: Optional[str] = None,
        id_column: Optional[str] = MODEL_CONFIG.participant_id_column,
    ) -> T:
        """
        Fit the classifier and optional feature scaler.

        Supports two usage styles:
            1. Separated features and labels: fit(X, y)
            2. Combined DataFrame: fit(df) with target_column specified or defaulted.

        Parameters:
            X: Feature matrix (DataFrame or 2D ndarray) or combined dataset DataFrame.
            y: Target labels (Series, 1D ndarray, or list). If None, extracted from X.
            target_column: Target column name if X is a combined DataFrame.
            id_column: Participant ID column to exclude if X is a combined DataFrame.

        Returns:
            self: The fitted classifier instance.
        """
        # If y is None and X is a DataFrame, attempt to separate target
        if y is None and isinstance(X, pd.DataFrame):
            lbl_col = target_column or self.config.label_column
            X_feats, y_target = separate_features_and_target(
                data=X,
                target_column=lbl_col,
                id_column=id_column,
                feature_columns=self.expected_feature_names,
            )
        else:
            X_feats = X
            y_target = y

        # Validate features and target
        X_arr = self._validate_features(X_feats, is_fit=True)
        y_arr = self._validate_target(y_target, n_samples=X_arr.shape[0])

        self.n_features_in_ = X_arr.shape[1]

        # Fit feature scaler if enabled
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_arr)
        else:
            self.scaler_ = None
            X_proc = X_arr

        # Fit scikit-learn estimator
        self.classifier_.fit(X_proc, y_arr)

        self.classes_ = self.classifier_.classes_
        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    ) -> np.ndarray:
        """
        Predict class labels for given acoustic features.

        Parameters:
            X: Features matrix of shape (n_samples, n_features) or single-sample dict/array.

        Returns:
            1D numpy array of predicted class labels.
        """
        self._check_is_fitted()
        X_arr = self._validate_features(X, is_fit=False)

        if self.scale_features and self.scaler_ is not None:
            X_proc = self.scaler_.transform(X_arr)
        else:
            X_proc = X_arr

        preds = self.classifier_.predict(X_proc)
        return preds

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    ) -> np.ndarray:
        """
        Compute class probabilities for given acoustic features.

        Parameters:
            X: Features matrix of shape (n_samples, n_features) or single-sample dict/array.

        Returns:
            2D numpy array of shape (n_samples, n_classes) where values sum to 1.0.
        """
        self._check_is_fitted()
        X_arr = self._validate_features(X, is_fit=False)

        if self.scale_features and self.scaler_ is not None:
            X_proc = self.scaler_.transform(X_arr)
        else:
            X_proc = X_arr

        proba = self.classifier_.predict_proba(X_proc)
        return proba

    def predict_proba_dict(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    ) -> List[Dict[Any, float]]:
        """
        Predict class probabilities and return as a list of dictionaries mapping class names to probabilities.

        Parameters:
            X: Features matrix or single sample.

        Returns:
            List of dicts, one per sample, e.g. [{'low': 0.85, 'high': 0.15}, ...]
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        classes = self.classes_

        results: List[Dict[Any, float]] = []
        for row in proba:
            results.append({cls: float(prob) for cls, prob in zip(classes, row)})
        return results

    def save(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Serialize and save this fitted voice classifier to disk using joblib.

        Parameters:
            filepath: Destination file path (e.g., 'model.joblib').
            metadata: Optional additional metadata dictionary to store in the bundle.

        Returns:
            Path to the saved model file.
        """
        return save_voice_model(self, filepath=filepath, metadata=metadata)

    @classmethod
    def load(
        cls: Type[T],
        filepath: Union[str, Path],
    ) -> T:
        """
        Load a serialized voice classifier from disk and ensure matching class type.

        Parameters:
            filepath: Path to the serialized model file.

        Returns:
            Fitted classifier instance of type cls.
        """
        model = load_voice_model(filepath=filepath, expected_type=cls)
        return model  # type: ignore[return-value]


# ===========================================================================
# Voice Logistic Regression Classifier
# ===========================================================================

class VoiceLogisticRegression(BaseVoiceClassifier):
    """
    Baseline Logistic Regression classifier for voice modality stress/risk assessment.

    Features:
        - Scikit-learn LogisticRegression with configurable random_state and max_iter.
        - Integrated StandardScaler for acoustic features (default: True).
        - Strict validation of input types, feature names, and non-null values.
        - Dynamic support for arbitrary user target labels (binary, multiclass, string, int).
        - Predict class labels and calibrated class probability distributions.
    """

    def __init__(
        self,
        random_state: Optional[int] = MODEL_CONFIG.random_seed,
        max_iter: int = 1000,
        C: float = 1.0,
        scale_features: bool = True,
        solver: str = "lbfgs",
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        config: ModelConfig = MODEL_CONFIG,
        expected_feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize VoiceLogisticRegression baseline model.

        Parameters:
            random_state: Random seed for solver reproducibility (default: 42).
            max_iter: Maximum iterations for solver convergence (default: 1000).
            C: Inverse regularization strength (default: 1.0).
            scale_features: Whether to apply StandardScaler to features before classification (default: True).
            solver: Optimization algorithm (default: 'lbfgs').
            class_weight: Weights associated with classes ('balanced' or dict).
            config: Centralized ModelConfig settings.
            expected_feature_names: Optional list of expected feature names. If None,
                                   learned during fit() or defaults to COMBINED_FEATURE_NAMES.
        """
        super().__init__(
            scale_features=scale_features,
            config=config,
            expected_feature_names=expected_feature_names,
        )
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.class_weight = class_weight

        # Internal estimator
        self.classifier_: LogisticRegression = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
            class_weight=self.class_weight,
        )

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients of the features in the decision function."""
        self._check_is_fitted()
        return self.classifier_.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept (bias) added to the decision function."""
        self._check_is_fitted()
        return self.classifier_.intercept_


# ===========================================================================
# Voice Random Forest Classifier
# ===========================================================================

class VoiceRandomForest(BaseVoiceClassifier):
    """
    Random Forest classifier for voice modality stress/risk assessment.

    Features:
        - Scikit-learn RandomForestClassifier with configurable hyperparameters:
          n_estimators, max_depth, min_samples_split, class_weight, random_state.
        - Reuses existing feature/target separation and input validation.
        - Provides fit, predict, predict_proba, and predict_proba_dict methods.
        - Supports binary and multiclass arbitrary labels dynamically.
        - Exposes feature_importances_ and feature_importances_dict for interpretability.
        - Optional feature scaling (defaults to False for tree ensembles).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        random_state: Optional[int] = MODEL_CONFIG.random_seed,
        scale_features: bool = False,
        config: ModelConfig = MODEL_CONFIG,
        expected_feature_names: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize VoiceRandomForest classifier.

        Parameters:
            n_estimators: Number of trees in the forest (default: 100).
            max_depth: Maximum depth of the tree (default: None).
            min_samples_split: Minimum number of samples required to split an internal node (default: 2).
            class_weight: Weights associated with classes ('balanced', 'balanced_subsample', or dict).
            random_state: Random seed for reproducibility (default: 42).
            scale_features: Whether to apply StandardScaler (default: False).
            config: Centralized ModelConfig settings.
            expected_feature_names: Optional list of expected feature names.
            **kwargs: Additional keyword arguments passed to sklearn RandomForestClassifier.
        """
        super().__init__(
            scale_features=scale_features,
            config=config,
            expected_feature_names=expected_feature_names,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.kwargs = kwargs

        # Internal estimator
        self.classifier_: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the fitted Random Forest classifier."""
        self._check_is_fitted()
        return self.classifier_.feature_importances_

    @property
    def feature_importances_dict(self) -> Dict[str, float]:
        """Map feature names to their corresponding importance scores."""
        self._check_is_fitted()
        importances = self.classifier_.feature_importances_
        names = self.feature_names_ or [f"feature_{i}" for i in range(len(importances))]
        return {name: float(imp) for name, imp in zip(names, importances)}


# Aliases for classifier naming flexibility
VoiceBaselineClassifier = VoiceLogisticRegression
VoiceLogisticRegressionClassifier = VoiceLogisticRegression
VoiceRandomForestClassifier = VoiceRandomForest


# ===========================================================================
# Model Serialization Utilities
# ===========================================================================

def save_voice_model(
    model: BaseVoiceClassifier,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serialize and save a trained voice model to disk using joblib.

    Parameters:
        model: Fitted BaseVoiceClassifier instance (VoiceLogisticRegression, VoiceRandomForest).
        filepath: Destination file path (e.g. 'models/voice_rf.joblib').
        metadata: Optional additional metadata dictionary to store with the model.

    Returns:
        Path to the saved model file.

    Raises:
        ModelNotFittedError: If model is not in a fitted state.
        ModelSerializationError: If saving fails due to filesystem or serialization errors.
    """
    if model is None:
        raise ModelSerializationError("Cannot save None as a voice model.")

    if not isinstance(model, BaseVoiceClassifier):
        raise TypeError(
            f"Expected an instance of BaseVoiceClassifier, got {type(model).__name__}."
        )

    if not model.is_fitted:
        raise ModelNotFittedError(
            f"Cannot save unfitted model '{model.__class__.__name__}'. "
            "Call 'fit' with training data before saving."
        )

    path = Path(filepath)
    if not path.name or path.is_dir():
        raise ModelSerializationError(f"Invalid model destination path: '{filepath}'.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle: Dict[str, Any] = {
            "format_version": "1.0",
            "model_type": model.__class__.__name__,
            "model": model,
            "feature_names": model.feature_names_,
            "classes": model.classes_,
            "n_features_in": model.n_features_in_,
            "scale_features": model.scale_features,
            "config": model.config,
            "metadata": metadata or {},
        }
        joblib.dump(bundle, path)
        return path
    except Exception as e:
        if isinstance(e, ModelError):
            raise
        raise ModelSerializationError(f"Failed to save voice model to '{filepath}': {e}") from e


def load_voice_model(
    filepath: Union[str, Path],
    expected_type: Optional[Type[BaseVoiceClassifier]] = None,
) -> BaseVoiceClassifier:
    """
    Load a serialized voice model from disk.

    Validates that the file exists, is non-empty, deserializes properly,
    and reconstructs a valid fitted voice classifier with intact feature names and classes.

    Parameters:
        filepath: Path to the serialized model file.
        expected_type: Optional subclass type check (e.g. VoiceRandomForest).

    Returns:
        Loaded, ready-to-use BaseVoiceClassifier instance.

    Raises:
        ModelNotFoundError: If filepath does not exist on disk.
        ModelCorruptError: If the file is corrupt, empty, or contains an invalid model.
    """
    path = Path(filepath)

    if not path.exists():
        raise ModelNotFoundError(f"Model file not found at: '{filepath}'.")

    if path.is_dir():
        raise ModelCorruptError(f"Specified path is a directory, not a file: '{filepath}'.")

    if path.stat().st_size == 0:
        raise ModelCorruptError(f"Model file is empty (0 bytes): '{filepath}'.")

    try:
        loaded = joblib.load(path)
    except Exception as e:
        raise ModelCorruptError(f"Failed to deserialize model file '{filepath}': {e}") from e

    # Extract model from bundle or handle direct instance
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    elif isinstance(loaded, BaseVoiceClassifier):
        model = loaded
    else:
        raise ModelCorruptError(
            f"Deserialized object from '{filepath}' is of invalid type '{type(loaded).__name__}'. "
            "Expected a serialized voice model bundle or BaseVoiceClassifier instance."
        )

    if not isinstance(model, BaseVoiceClassifier):
        raise ModelCorruptError(
            f"Model inside bundle is of invalid type '{type(model).__name__}'."
        )

    # Verify fitted state and essential attributes
    if not getattr(model, "is_fitted_", False) or getattr(model, "classes_", None) is None:
        raise ModelCorruptError(
            f"Loaded model '{model.__class__.__name__}' is not in a valid fitted state."
        )

    if getattr(model, "feature_names_", None) is None:
        raise ModelCorruptError(
            f"Loaded model '{model.__class__.__name__}' is missing feature names."
        )

    # Verify expected subclass type if requested
    if expected_type is not None and not isinstance(model, expected_type):
        raise ModelCorruptError(
            f"Expected model of type '{expected_type.__name__}', but loaded '{type(model).__name__}'."
        )

    return model

