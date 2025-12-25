"""
Models Module

Provides the 7 required classification algorithms.
Implements FR-36 to FR-39 from the requirements.

Classifiers:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Naive Bayes (Gaussian)
5. Random Forest
6. Support Vector Machine
7. Rule-Based (Decision Tree with interpretability focus)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time


@dataclass
class ModelConfig:
    """Configuration for a classifier model."""
    name: str
    model_class: Any
    default_params: Dict[str, Any]
    tuning_params: Dict[str, List[Any]]
    description: str


# Model configurations with default parameters and tuning grids
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Logistic Regression": ModelConfig(
        name="Logistic Regression",
        model_class=LogisticRegression,
        default_params={
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
            "class_weight": "balanced"
        },
        tuning_params={
            "C": [0.001, 0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"]
        },
        description="Linear model for classification. Fast and interpretable. Works well for linearly separable data."
    ),
    
    "K-Nearest Neighbors": ModelConfig(
        name="K-Nearest Neighbors",
        model_class=KNeighborsClassifier,
        default_params={
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski"
        },
        tuning_params={
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        description="Instance-based learning. Simple and effective for small datasets. Sensitive to feature scaling."
    ),
    
    "Decision Tree": ModelConfig(
        name="Decision Tree",
        model_class=DecisionTreeClassifier,
        default_params={
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 2,
            "class_weight": "balanced"
        },
        tuning_params={
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        description="Tree-based model. Highly interpretable. Handles non-linear relationships. Prone to overfitting."
    ),
    
    "Naive Bayes": ModelConfig(
        name="Naive Bayes",
        model_class=GaussianNB,
        default_params={},
        tuning_params={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        description="Probabilistic classifier based on Bayes theorem. Fast training. Assumes feature independence."
    ),
    
    "Random Forest": ModelConfig(
        name="Random Forest",
        model_class=RandomForestClassifier,
        default_params={
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 10,
            "class_weight": "balanced",
            "n_jobs": -1
        },
        tuning_params={
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        description="Ensemble of decision trees. Robust to overfitting. Handles high-dimensional data well."
    ),
    
    "Support Vector Machine": ModelConfig(
        name="Support Vector Machine",
        model_class=SVC,
        default_params={
            "random_state": 42,
            "probability": True,  # Enable probability estimates for ROC-AUC
            "class_weight": "balanced",
            "max_iter": 5000
        },
        tuning_params={
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        },
        description="Powerful classifier using hyperplanes. Effective in high-dimensional spaces. Slower on large datasets."
    ),
    
    "Rule-Based": ModelConfig(
        name="Rule-Based",
        model_class=DecisionTreeClassifier,
        default_params={
            "random_state": 42,
            "max_depth": 5,  # Shallow for interpretability
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
        },
        tuning_params={
            "max_depth": [3, 4, 5, 6],
            "min_samples_leaf": [5, 10, 20]
        },
        description="Shallow decision tree optimized for interpretability. Generates human-readable rules."
    )
}


@dataclass
class TrainedModel:
    """Container for a trained model with metadata."""
    name: str
    model: Any
    training_time: float
    params: Dict[str, Any]
    is_tuned: bool = False
    cv_score: Optional[float] = None


def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of model names
    """
    return list(MODEL_CONFIGS.keys())


def get_model_description(model_name: str) -> str:
    """
    Get description for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model description string
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].description
    return "Unknown model"


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig object or None
    """
    return MODEL_CONFIGS.get(model_name)


def create_model(
    model_name: str,
    custom_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create an untrained model instance.
    
    Args:
        model_name: Name of the model
        custom_params: Optional custom parameters to override defaults
        
    Returns:
        sklearn classifier instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    params = config.default_params.copy()
    
    if custom_params:
        params.update(custom_params)
    
    return config.model_class(**params)


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    custom_params: Optional[Dict[str, Any]] = None
) -> TrainedModel:
    """
    Train a single model.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training labels
        custom_params: Optional custom parameters
        
    Returns:
        TrainedModel object with trained model and metadata
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    params = config.default_params.copy()
    
    if custom_params:
        params.update(custom_params)
    
    model = config.model_class(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return TrainedModel(
        name=model_name,
        model=model,
        training_time=training_time,
        params=params,
        is_tuned=False
    )


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_to_train: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, TrainedModel]:
    """
    Train all specified models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        models_to_train: List of model names to train (default: all)
        progress_callback: Optional callback function(model_name, i, total)
        
    Returns:
        Dict mapping model names to TrainedModel objects
    """
    if models_to_train is None:
        models_to_train = list(MODEL_CONFIGS.keys())
    
    trained_models = {}
    total = len(models_to_train)
    
    for i, model_name in enumerate(models_to_train):
        if progress_callback:
            progress_callback(model_name, i, total)
        
        try:
            trained_model = train_model(model_name, X_train, y_train)
            trained_models[model_name] = trained_model
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    return trained_models


def get_model_feature_importance(
    model: Any,
    feature_names: List[str]
) -> Optional[Dict[str, float]]:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        
    Returns:
        Dict mapping feature names to importance scores, or None if not available
    """
    importance = None
    
    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    # Logistic Regression (use coefficient magnitude)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim == 2:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
    
    if importance is not None:
        return dict(zip(feature_names, importance.tolist()))
    
    return None


def get_decision_rules(model: Any, feature_names: List[str], max_depth: int = 5) -> str:
    """
    Extract decision rules from a tree-based model.
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        max_depth: Maximum depth to traverse
        
    Returns:
        String representation of decision rules
    """
    if not hasattr(model, 'tree_'):
        return "Model does not support rule extraction."
    
    from sklearn.tree import export_text
    
    try:
        rules = export_text(model, feature_names=list(feature_names), max_depth=max_depth)
        return rules
    except Exception as e:
        return f"Could not extract rules: {e}"
