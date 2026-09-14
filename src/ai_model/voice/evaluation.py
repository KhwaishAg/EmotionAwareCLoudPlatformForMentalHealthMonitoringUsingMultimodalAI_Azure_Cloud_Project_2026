"""
Voice Modality — Model Evaluation Metrics

Computes classification performance metrics (Accuracy, Precision, Recall, F1, Macro F1)
for voice modality stress/risk assessment classifiers.

Components:
    - VoiceModelMetrics: Structured dataclass encapsulating overall, macro, weighted,
      and per-class performance metrics with dictionary conversion and indexing support.
    - calculate_voice_metrics: Calculate metrics directly from ground-truth (y_true)
      and model predictions (y_pred).
    - evaluate_voice_model: End-to-end evaluation utility accepting a fitted voice classifier
      (VoiceLogisticRegression, VoiceRandomForest, or any standard estimator) and features/labels.
    - EvaluationError, EvaluationInputError: Specific exception hierarchy for evaluation failures.

Design:
    - Strict validation: Empty inputs, mismatched lengths, null/NaN values are caught early.
    - Arbitrary label support: Dynamic discovery of unique labels; handles binary and multiclass
      string, integer, and categorical targets without hardcoded labels.
    - Zero-division safety: Guaranteed non-crashing metric calculations (zero_division=0.0)
      when classes have zero true or predicted samples.
    - Decoupled from training: No training or synthetic datasets are fabricated in this module.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard Python dictionary."""
        return asdict(self)

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
) -> VoiceModelMetrics:
    """
    Calculate classification performance metrics between ground truth and predicted labels.

    Computes:
        - Accuracy
        - Precision, Recall, F1 (weighted, macro, or binary)
        - Macro F1 (unweighted class average)
        - Detailed per-class precision, recall, f1, and support

    Parameters:
        y_true: Ground truth target labels (Series, ndarray, or list).
        y_pred: Predicted target labels (Series, ndarray, or list).
        average: Averaging strategy for summary precision, recall, f1:
                 'weighted' (default), 'macro', or 'binary'.
        pos_label: Target positive class label if average='binary'.
        labels: Optional explicit list of class labels to evaluate. If None,
                inferred from unique values in y_true and y_pred.

    Returns:
        VoiceModelMetrics structured results.

    Raises:
        EvaluationInputError: If inputs are empty, have mismatched lengths, or contain nulls.
    """
    arr_true, arr_pred = _validate_ground_truth_and_predictions(y_true, y_pred)

    # Determine unique classes
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
    )


# Aliases for metric calculation
compute_voice_metrics = calculate_voice_metrics
calculate_classification_metrics = calculate_voice_metrics


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

    # Calculate metrics
    return calculate_voice_metrics(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        pos_label=pos_label,
    )
