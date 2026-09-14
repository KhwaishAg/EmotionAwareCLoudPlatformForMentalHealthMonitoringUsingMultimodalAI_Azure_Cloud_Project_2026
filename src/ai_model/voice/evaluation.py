"""
Voice Modality — Model Evaluation Metrics & Confusion Matrix

Computes classification performance metrics (Accuracy, Precision, Recall, F1, Macro F1)
and confusion matrices for voice modality stress/risk assessment classifiers.

Components:
    - VoiceModelMetrics: Structured dataclass encapsulating overall, macro, weighted,
      and per-class performance metrics with dictionary conversion and indexing support.
    - VoiceConfusionMatrix: Structured dataclass encapsulating raw integer counts,
      class labels, row-normalized proportions, and DataFrame conversion.
    - calculate_voice_metrics: Calculate metrics directly from ground-truth (y_true)
      and model predictions (y_pred).
    - compute_voice_confusion_matrix: Compute raw and normalized confusion matrices with
      strict preservation of custom label orderings.
    - evaluate_voice_model: End-to-end evaluation utility accepting a fitted voice classifier
      (VoiceLogisticRegression, VoiceRandomForest, or any standard estimator) and features/labels.
    - evaluate_voice_confusion_matrix: End-to-end confusion matrix evaluation helper for a
      fitted voice classifier.
    - EvaluationError, EvaluationInputError: Specific exception hierarchy for evaluation failures.

Design:
    - Strict validation: Empty inputs, mismatched lengths, null/NaN values are caught early.
    - Arbitrary label support: Dynamic discovery of unique labels; handles binary and multiclass
      string, integer, and categorical targets without hardcoded labels.
    - Explicit label ordering: When a custom class list is provided, row/column ordering is strictly
      preserved.
    - Zero-division safety: Guaranteed non-crashing metric calculations (zero_division=0.0)
      when classes have zero true or predicted samples.
    - Decoupled from training: No training or synthetic datasets are fabricated in this module.
    - No plotting dependencies: Plotting is reserved for subsequent visualization modules.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .config import MODEL_CONFIG
from .model import (
    ModelError,
    ModelInputError,
    ModelNotFittedError,
    separate_features_and_target,
)


# ===========================================================================
# Exceptions
# ===========================================================================

class EvaluationError(ModelError):
    """Base exception for voice evaluation errors."""
    pass


class EvaluationInputError(EvaluationError, ModelInputError):
    """Raised when ground truth, predictions, or evaluation datasets are invalid."""
    pass


# ===========================================================================
# Structured Confusion Matrix Result Container
# ===========================================================================

@dataclass
class VoiceConfusionMatrix:
    """
    Structured container for voice classifier confusion matrix results.

    Attributes:
        matrix: 2D integer numpy array of shape (n_classes, n_classes),
                where row i represents true labels and column j represents predicted labels.
        labels: Ordered list of unique class labels corresponding to matrix rows/columns.
        normalized_matrix: Optional 2D float numpy array with row-normalized proportions
                           (range: [0.0, 1.0]).
    """

    matrix: np.ndarray
    labels: List[Any]
    normalized_matrix: Optional[np.ndarray] = None

    @property
    def total_samples(self) -> int:
        """Total number of evaluated samples in the confusion matrix."""
        return int(np.sum(self.matrix))

    @property
    def correct_predictions(self) -> int:
        """Total number of correct predictions along the diagonal."""
        return int(np.trace(self.matrix))

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy derived from the matrix trace."""
        if self.total_samples == 0:
            return 0.0
        return float(self.correct_predictions / self.total_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert confusion matrix results to a standard Python dictionary."""
        return {
            "matrix": self.matrix.tolist(),
            "labels": list(self.labels),
            "normalized_matrix": (
                self.normalized_matrix.tolist() if self.normalized_matrix is not None else None
            ),
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
        }

    def to_dataframe(self, normalized: bool = False) -> pd.DataFrame:
        """
        Convert confusion matrix to a labeled pandas DataFrame.

        Parameters:
            normalized: If True and normalized_matrix is available, returns proportions.
                        Otherwise returns integer counts.

        Returns:
            pd.DataFrame with row index 'true_<label>' and column headers 'pred_<label>'.
        """
        data = self.normalized_matrix if (normalized and self.normalized_matrix is not None) else self.matrix
        row_labels = [f"true_{lbl}" for lbl in self.labels]
        col_labels = [f"pred_{lbl}" for lbl in self.labels]
        return pd.DataFrame(data, index=row_labels, columns=col_labels)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style attribute access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Attribute '{key}' not found in VoiceConfusionMatrix.")

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists in results."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with fallback default value."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        """Readable summary representation."""
        return (
            f"VoiceConfusionMatrix("
            f"classes={len(self.labels)}, "
            f"total_samples={self.total_samples}, "
            f"accuracy={self.accuracy:.4f}, "
            f"labels={self.labels})"
        )


# Aliases for naming flexibility
VoiceConfusionMatrixResult = VoiceConfusionMatrix
ConfusionMatrixResult = VoiceConfusionMatrix


# ===========================================================================
# Structured Metrics Result Container
# ===========================================================================

@dataclass
class VoiceModelMetrics:
    """
    Structured metrics result container for voice classifier evaluation.

    Attributes:
        accuracy: Overall classification accuracy (range: [0.0, 1.0]).
        precision: Precision score according to selected averaging strategy.
        recall: Recall score according to selected averaging strategy.
        f1: F1 score according to selected averaging strategy.
        macro_f1: Unweighted macro-averaged F1 across all classes.
        macro_precision: Unweighted macro-averaged precision across all classes.
        macro_recall: Unweighted macro-averaged recall across all classes.
        weighted_f1: Support-weighted F1 across all classes.
        weighted_precision: Support-weighted precision across all classes.
        weighted_recall: Support-weighted recall across all classes.
        support: Total number of evaluated samples.
        classes: List of evaluated unique classes.
        per_class: Detailed per-class precision, recall, f1, and support mapping.
        confusion_matrix: Optional VoiceConfusionMatrix container.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    weighted_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    support: int = 0
    classes: List[Any] = field(default_factory=list)
    per_class: Dict[Any, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[VoiceConfusionMatrix] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard Python dictionary."""
        d = asdict(self)
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix.to_dict()
        return d

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access (e.g., metrics['accuracy'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Metric '{key}' not found in VoiceModelMetrics.")

    def __contains__(self, key: str) -> bool:
        """Check if metric attribute exists in results."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric with fallback default value."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        """Readable summary representation of key metrics."""
        return (
            f"VoiceModelMetrics("
            f"accuracy={self.accuracy:.4f}, "
            f"precision={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"f1={self.f1:.4f}, "
            f"macro_f1={self.macro_f1:.4f}, "
            f"support={self.support})"
        )


# Aliases for naming flexibility
VoiceEvaluationMetrics = VoiceModelMetrics
EvaluationMetrics = VoiceModelMetrics


# ===========================================================================
# Metric Calculation Utilities
# ===========================================================================

def _validate_ground_truth_and_predictions(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate y_true and y_pred vectors for consistency."""
    if y_true is None:
        raise EvaluationInputError("Ground truth labels 'y_true' cannot be None.")
    if y_pred is None:
        raise EvaluationInputError("Predicted labels 'y_pred' cannot be None.")

    if isinstance(y_true, pd.Series):
        if y_true.isna().any():
            raise EvaluationInputError("y_true contains null/NaN values.")
        arr_true = y_true.to_numpy()
    else:
        arr_true = np.asarray(y_true)

    if isinstance(y_pred, pd.Series):
        if y_pred.isna().any():
            raise EvaluationInputError("y_pred contains null/NaN values.")
        arr_pred = y_pred.to_numpy()
    else:
        arr_pred = np.asarray(y_pred)

    if arr_true.size == 0 or arr_pred.size == 0:
        raise EvaluationInputError("Input labels cannot be empty (0 samples).")

    if arr_true.ndim != 1:
        arr_true = arr_true.ravel()
    if arr_pred.ndim != 1:
        arr_pred = arr_pred.ravel()

    if len(arr_true) != len(arr_pred):
        raise EvaluationInputError(
            f"Sample count mismatch: y_true has {len(arr_true)} samples but y_pred has {len(arr_pred)}."
        )

    # Check for NaN / None / empty string
    for val in arr_true:
        if pd.isna(val) or val is None or (isinstance(val, str) and not val.strip()):
            raise EvaluationInputError(f"y_true contains invalid label: '{val}'.")
    for val in arr_pred:
        if pd.isna(val) or val is None or (isinstance(val, str) and not val.strip()):
            raise EvaluationInputError(f"y_pred contains invalid label: '{val}'.")

    return arr_true, arr_pred


def calculate_voice_metrics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    average: str = "weighted",
    pos_label: Optional[Any] = None,
    labels: Optional[List[Any]] = None,
    include_confusion_matrix: bool = False,
) -> VoiceModelMetrics:
    """
    Calculate classification performance metrics between ground truth and predicted labels.

    Computes:
        - Accuracy
        - Precision, Recall, F1 (weighted, macro, or binary)
        - Macro F1 (unweighted class average)
        - Detailed per-class precision, recall, f1, and support
        - Optional confusion matrix

    Parameters:
        y_true: Ground truth target labels (Series, ndarray, or list).
        y_pred: Predicted target labels (Series, ndarray, or list).
        average: Averaging strategy for summary precision, recall, f1:
                 'weighted' (default), 'macro', or 'binary'.
        pos_label: Target positive class label if average='binary'.
        labels: Optional explicit list of class labels to evaluate. If None,
                inferred from unique values in y_true and y_pred.
        include_confusion_matrix: Whether to attach VoiceConfusionMatrix to results.

    Returns:
        VoiceModelMetrics structured results.

    Raises:
        EvaluationInputError: If inputs are empty, have mismatched lengths, or contain nulls.
    """
    arr_true, arr_pred = _validate_ground_truth_and_predictions(y_true, y_pred)

    # Determine unique classes (preserve explicit labels ordering if provided)
    if labels is not None:
        unique_classes = list(labels)
    else:
        # Preserve sorted unique classes
        unique_classes = list(np.unique(np.concatenate([arr_true, arr_pred])))

    # 1. Overall Accuracy
    accuracy = float(accuracy_score(arr_true, arr_pred))

    # 2. Macro-averaged metrics
    macro_p = float(precision_score(arr_true, arr_pred, average="macro", zero_division=0.0))
    macro_r = float(recall_score(arr_true, arr_pred, average="macro", zero_division=0.0))
    macro_f1 = float(f1_score(arr_true, arr_pred, average="macro", zero_division=0.0))

    # 3. Weighted-averaged metrics
    weighted_p = float(precision_score(arr_true, arr_pred, average="weighted", zero_division=0.0))
    weighted_r = float(recall_score(arr_true, arr_pred, average="weighted", zero_division=0.0))
    weighted_f1 = float(f1_score(arr_true, arr_pred, average="weighted", zero_division=0.0))

    # 4. Strategy-specific summary precision, recall, and f1
    if average == "macro":
        p_summary, r_summary, f1_summary = macro_p, macro_r, macro_f1
    elif average == "binary":
        if pos_label is not None:
            p_summary = float(
                precision_score(arr_true, arr_pred, average="binary", pos_label=pos_label, zero_division=0.0)
            )
            r_summary = float(
                recall_score(arr_true, arr_pred, average="binary", pos_label=pos_label, zero_division=0.0)
            )
            f1_summary = float(
                f1_score(arr_true, arr_pred, average="binary", pos_label=pos_label, zero_division=0.0)
            )
        else:
            # If no pos_label specified, fallback to second class if binary or weighted
            if len(unique_classes) == 2:
                p_summary = float(
                    precision_score(
                        arr_true, arr_pred, average="binary", pos_label=unique_classes[1], zero_division=0.0
                    )
                )
                r_summary = float(
                    recall_score(
                        arr_true, arr_pred, average="binary", pos_label=unique_classes[1], zero_division=0.0
                    )
                )
                f1_summary = float(
                    f1_score(
                        arr_true, arr_pred, average="binary", pos_label=unique_classes[1], zero_division=0.0
                    )
                )
            else:
                p_summary, r_summary, f1_summary = weighted_p, weighted_r, weighted_f1
    else:  # default 'weighted'
        p_summary, r_summary, f1_summary = weighted_p, weighted_r, weighted_f1

    # 5. Per-class metrics
    p_class, r_class, f_class, s_class = precision_recall_fscore_support(
        arr_true,
        arr_pred,
        labels=unique_classes,
        zero_division=0.0,
    )

    per_class_dict: Dict[Any, Dict[str, float]] = {}
    for cls_name, p, r, f, s in zip(unique_classes, p_class, r_class, f_class, s_class):
        per_class_dict[cls_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    # 6. Optional Confusion Matrix
    cm: Optional[VoiceConfusionMatrix] = None
    if include_confusion_matrix:
        cm = compute_voice_confusion_matrix(y_true=arr_true, y_pred=arr_pred, labels=unique_classes)

    return VoiceModelMetrics(
        accuracy=accuracy,
        precision=p_summary,
        recall=r_summary,
        f1=f1_summary,
        macro_f1=macro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        weighted_f1=weighted_f1,
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        support=len(arr_true),
        classes=unique_classes,
        per_class=per_class_dict,
        confusion_matrix=cm,
    )


# Aliases for metric calculation
compute_voice_metrics = calculate_voice_metrics
calculate_classification_metrics = calculate_voice_metrics


# ===========================================================================
# Confusion Matrix Calculation Utilities
# ===========================================================================

def compute_voice_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    labels: Optional[List[Any]] = None,
    normalize: Optional[str] = "true",
) -> VoiceConfusionMatrix:
    """
    Compute a classification confusion matrix for voice model predictions.

    Preserves explicit label order when 'labels' is provided.

    Parameters:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        labels: Optional explicit ordering of class labels to index the matrix.
                If None, sorted unique classes present in y_true and y_pred are used.
        normalize: Normalization mode for normalized_matrix:
                   'true' (row proportions / recall), 'pred' (column proportions / precision),
                   'all' (total proportion), or None.

    Returns:
        VoiceConfusionMatrix structured result.

    Raises:
        EvaluationInputError: If inputs are invalid, empty, or have mismatched lengths.
    """
    arr_true, arr_pred = _validate_ground_truth_and_predictions(y_true, y_pred)

    if labels is not None:
        target_labels = list(labels)
        if len(target_labels) == 0:
            raise EvaluationInputError("Explicit 'labels' list cannot be empty.")
    else:
        target_labels = list(np.unique(np.concatenate([arr_true, arr_pred])))

    # Compute raw integer confusion matrix
    cm_raw = confusion_matrix(
        y_true=arr_true,
        y_pred=arr_pred,
        labels=target_labels,
    )

    # Compute normalized confusion matrix if requested
    cm_norm: Optional[np.ndarray] = None
    if normalize in ("true", "pred", "all"):
        cm_norm = confusion_matrix(
            y_true=arr_true,
            y_pred=arr_pred,
            labels=target_labels,
            normalize=normalize,
        )
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    return VoiceConfusionMatrix(
        matrix=cm_raw,
        labels=target_labels,
        normalized_matrix=cm_norm,
    )


# Aliases for confusion matrix calculation
calculate_voice_confusion_matrix = compute_voice_confusion_matrix
compute_confusion_matrix = compute_voice_confusion_matrix


# ===========================================================================
# End-to-End Model Evaluation
# ===========================================================================

def evaluate_voice_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    y: Optional[Union[pd.Series, np.ndarray, list]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = MODEL_CONFIG.participant_id_column,
    average: str = "weighted",
    pos_label: Optional[Any] = None,
    include_confusion_matrix: bool = False,
) -> VoiceModelMetrics:
    """
    Evaluate a fitted voice classifier on a feature matrix or dataset DataFrame.

    Supports:
        - VoiceLogisticRegression and VoiceRandomForest models
        - Any standard fitted scikit-learn classifier with predict()
        - Separated features/labels: evaluate_voice_model(model, X, y)
        - Combined dataset DataFrame: evaluate_voice_model(model, dataset_df)

    Parameters:
        model: Fitted classifier instance.
        X: Feature matrix or combined dataset DataFrame.
        y: Ground truth labels. If None, extracted from X via target_column.
        target_column: Target label column name if X is a combined DataFrame.
        id_column: Participant ID column to exclude if X is a combined DataFrame.
        average: Averaging strategy ('weighted', 'macro', 'binary').
        pos_label: Positive class label if average='binary'.
        include_confusion_matrix: Whether to compute and attach confusion matrix.

    Returns:
        VoiceModelMetrics structured results.

    Raises:
        EvaluationError: If model is None or does not implement predict.
        ModelNotFittedError: If model has not been fitted yet.
        EvaluationInputError: If feature matrix or target labels are invalid.
    """
    if model is None:
        raise EvaluationError("Model cannot be None.")

    # Check fitted status
    if hasattr(model, "is_fitted") and not model.is_fitted:
        raise ModelNotFittedError(
            f"Model '{model.__class__.__name__}' is not fitted yet. "
            "Fit the model before evaluating."
        )

    if not hasattr(model, "predict"):
        raise EvaluationError(
            f"Model '{model.__class__.__name__}' does not implement 'predict' method."
        )

    # If y is None and X is a DataFrame, attempt to decouple target
    if y is None and isinstance(X, pd.DataFrame):
        lbl_col = target_column or MODEL_CONFIG.label_column
        X_feats, y_true = separate_features_and_target(
            data=X,
            target_column=lbl_col,
            id_column=id_column,
        )
    else:
        if y is None:
            raise EvaluationInputError(
                "Target labels 'y' must be provided when X is not a DataFrame."
            )
        X_feats = X
        y_true = y

    # Generate predictions
    y_pred = model.predict(X_feats)

    # Determine labels order to preserve if model has classes_
    model_labels = list(model.classes_) if hasattr(model, "classes_") and model.classes_ is not None else None

    # Calculate metrics
    return calculate_voice_metrics(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        pos_label=pos_label,
        labels=model_labels,
        include_confusion_matrix=include_confusion_matrix,
    )


def evaluate_voice_confusion_matrix(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    y: Optional[Union[pd.Series, np.ndarray, list]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = MODEL_CONFIG.participant_id_column,
    labels: Optional[List[Any]] = None,
    normalize: Optional[str] = "true",
) -> VoiceConfusionMatrix:
    """
    Evaluate confusion matrix for a fitted voice classifier on a test dataset.

    Parameters:
        model: Fitted classifier instance (VoiceLogisticRegression, VoiceRandomForest, etc.).
        X: Feature matrix or combined dataset DataFrame.
        y: Ground truth labels. If None and X is a DataFrame, extracted via target_column.
        target_column: Target label column name if X is a combined DataFrame.
        id_column: Participant ID column to exclude if X is a combined DataFrame.
        labels: Explicit class label order. If None and model has classes_, uses model.classes_.
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        VoiceConfusionMatrix structured result with counts, labels, and normalized proportions.

    Raises:
        EvaluationError: If model is None or does not implement predict.
        ModelNotFittedError: If model has not been fitted yet.
        EvaluationInputError: If feature matrix or target labels are invalid.
    """
    if model is None:
        raise EvaluationError("Model cannot be None.")

    # Check fitted status
    if hasattr(model, "is_fitted") and not model.is_fitted:
        raise ModelNotFittedError(
            f"Model '{model.__class__.__name__}' is not fitted yet. "
            "Fit the model before evaluating."
        )

    if not hasattr(model, "predict"):
        raise EvaluationError(
            f"Model '{model.__class__.__name__}' does not implement 'predict' method."
        )

    # If y is None and X is a DataFrame, attempt to decouple target
    if y is None and isinstance(X, pd.DataFrame):
        lbl_col = target_column or MODEL_CONFIG.label_column
        X_feats, y_true = separate_features_and_target(
            data=X,
            target_column=lbl_col,
            id_column=id_column,
        )
    else:
        if y is None:
            raise EvaluationInputError(
                "Target labels 'y' must be provided when X is not a DataFrame."
            )
        X_feats = X
        y_true = y

    # Determine labels order
    if labels is not None:
        matrix_labels = list(labels)
    elif hasattr(model, "classes_") and model.classes_ is not None:
        matrix_labels = list(model.classes_)
    else:
        matrix_labels = None

    # Predict class labels
    y_pred = model.predict(X_feats)

    return compute_voice_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=matrix_labels,
        normalize=normalize,
    )


# Alias for evaluate confusion matrix
evaluate_confusion_matrix = evaluate_voice_confusion_matrix
