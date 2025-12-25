"""
Model Evaluation Module

Provides metrics calculation and evaluation utilities.
Implements FR-44 to FR-52 from the requirements.

Metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- ROC-AUC (binary classification only)
- Confusion Matrix
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]  # Only for binary classification
    confusion_matrix: np.ndarray
    classification_report: str
    training_time: float
    is_binary: bool
    roc_data: Optional[Dict[str, Any]] = None  # For ROC curve plotting


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time: float = 0.0,
    class_names: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        training_time: Time taken to train
        class_names: Optional list of class names
        
    Returns:
        EvaluationResult object
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Determine if binary classification
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    target_names = class_names if class_names else [str(i) for i in range(n_classes)]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    
    # ROC-AUC (binary only)
    roc_auc = None
    roc_data = None
    
    if is_binary and hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Calculate ROC curve data
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc
            }
        except Exception:
            pass
    
    return EvaluationResult(
        model_name=model_name,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        roc_auc=round(roc_auc, 4) if roc_auc else None,
        confusion_matrix=cm,
        classification_report=report,
        training_time=round(training_time, 2),
        is_binary=is_binary,
        roc_data=roc_data
    )


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, EvaluationResult]:
    """
    Evaluate all trained models.
    
    Args:
        trained_models: Dict of model_name -> TrainedModel objects
        X_test: Test features
        y_test: Test labels
        class_names: Optional list of class names
        
    Returns:
        Dict of model_name -> EvaluationResult
    """
    results = {}
    
    for model_name, trained_model in trained_models.items():
        try:
            result = evaluate_model(
                model=trained_model.model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                training_time=trained_model.training_time,
                class_names=class_names
            )
            results[model_name] = result
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    return results


def create_comparison_dataframe(
    evaluation_results: Dict[str, EvaluationResult]
) -> pd.DataFrame:
    """
    Create a comparison DataFrame from evaluation results.
    
    Args:
        evaluation_results: Dict of model_name -> EvaluationResult
        
    Returns:
        DataFrame with all metrics for comparison
    """
    data = []
    
    for model_name, result in evaluation_results.items():
        row = {
            'Model': model_name,
            'Accuracy': result.accuracy,
            'Precision': result.precision,
            'Recall': result.recall,
            'F1-Score': result.f1,
            'Training Time (s)': result.training_time
        }
        
        if result.roc_auc is not None:
            row['ROC-AUC'] = result.roc_auc
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by F1-Score (primary ranking metric per FR-47)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df


def get_best_model(
    evaluation_results: Dict[str, EvaluationResult],
    metric: str = 'f1'
) -> Tuple[str, EvaluationResult]:
    """
    Get the best model based on a specific metric.
    
    Args:
        evaluation_results: Dict of model_name -> EvaluationResult
        metric: Metric to use for comparison ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
        
    Returns:
        Tuple of (model_name, EvaluationResult)
    """
    best_name = None
    best_result = None
    best_score = -1
    
    for model_name, result in evaluation_results.items():
        if metric == 'accuracy':
            score = result.accuracy
        elif metric == 'precision':
            score = result.precision
        elif metric == 'recall':
            score = result.recall
        elif metric == 'f1':
            score = result.f1
        elif metric == 'roc_auc':
            score = result.roc_auc if result.roc_auc else 0
        else:
            score = result.f1
        
        if score > best_score:
            best_score = score
            best_name = model_name
            best_result = result
    
    return best_name, best_result


def format_metrics_for_display(result: EvaluationResult) -> Dict[str, str]:
    """
    Format metrics as strings for display.
    
    Args:
        result: EvaluationResult object
        
    Returns:
        Dict with formatted metric strings
    """
    metrics = {
        'Accuracy': f"{result.accuracy * 100:.2f}%",
        'Precision': f"{result.precision * 100:.2f}%",
        'Recall': f"{result.recall * 100:.2f}%",
        'F1-Score': f"{result.f1 * 100:.2f}%",
        'Training Time': f"{result.training_time:.2f}s"
    }
    
    if result.roc_auc is not None:
        metrics['ROC-AUC'] = f"{result.roc_auc:.4f}"
    
    return metrics


def get_per_class_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get per-class precision, recall, and F1 scores.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        DataFrame with per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )
    
    n_classes = len(precision)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    return pd.DataFrame({
        'Class': class_names,
        'Precision': precision.round(4),
        'Recall': recall.round(4),
        'F1-Score': f1.round(4),
        'Support': support.astype(int)
    })


def calculate_multiclass_roc(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int
) -> Dict[str, Any]:
    """
    Calculate ROC curves for multiclass classification (one-vs-rest).
    
    Args:
        model: Trained model with predict_proba
        X_test: Test features
        y_test: Test labels
        n_classes: Number of classes
        
    Returns:
        Dict with ROC data for each class
    """
    if not hasattr(model, 'predict_proba'):
        return None
    
    try:
        y_proba = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        roc_data = {}
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_data[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        return roc_data
    except Exception:
        return None
