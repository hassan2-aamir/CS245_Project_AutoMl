"""
Hyperparameter Tuning Module

Provides GridSearchCV and RandomizedSearchCV with 5-fold stratified CV.
Implements FR-40 to FR-43 from the requirements.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import time
import warnings

from ml.models import MODEL_CONFIGS, TrainedModel, create_model

warnings.filterwarnings('ignore')


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    model_name: str
    best_model: Any
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, Any]
    tuning_time: float
    n_iterations: int


def get_tuning_grid(model_name: str) -> Dict[str, List[Any]]:
    """
    Get the hyperparameter tuning grid for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of parameter grids
    """
    if model_name not in MODEL_CONFIGS:
        return {}
    
    return MODEL_CONFIGS[model_name].tuning_params.copy()


def estimate_tuning_iterations(model_name: str, search_method: str = "grid") -> int:
    """
    Estimate the number of iterations for tuning.
    
    Args:
        model_name: Name of the model
        search_method: 'grid' or 'random'
        
    Returns:
        Estimated number of iterations
    """
    param_grid = get_tuning_grid(model_name)
    
    if search_method == "grid":
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        return n_combinations * 5  # 5-fold CV
    else:  # random
        return 10 * 5  # Default 10 iterations with 5-fold CV


def tune_model_grid_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1
) -> TuningResult:
    """
    Tune model using GridSearchCV with stratified K-fold.
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid (default: use predefined)
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        TuningResult object
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get base model and param grid
    config = MODEL_CONFIGS[model_name]
    base_model = create_model(model_name)
    
    if param_grid is None:
        param_grid = config.tuning_params.copy()
    
    # Create stratified K-fold
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=0
    )
    
    # Run tuning
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Calculate iterations
    n_iterations = len(grid_search.cv_results_['mean_test_score'])
    
    return TuningResult(
        model_name=model_name,
        best_model=grid_search.best_estimator_,
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        cv_results=grid_search.cv_results_,
        tuning_time=tuning_time,
        n_iterations=n_iterations
    )


def tune_model_random_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: Optional[Dict[str, List[Any]]] = None,
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1
) -> TuningResult:
    """
    Tune model using RandomizedSearchCV with stratified K-fold.
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        param_distributions: Parameter distributions (default: use predefined grid)
        n_iter: Number of random iterations
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        TuningResult object
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get base model and param grid
    config = MODEL_CONFIGS[model_name]
    base_model = create_model(model_name)
    
    if param_distributions is None:
        param_distributions = config.tuning_params.copy()
    
    # Create stratified K-fold
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
        random_state=42,
        verbose=0
    )
    
    # Run tuning
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    return TuningResult(
        model_name=model_name,
        best_model=random_search.best_estimator_,
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        cv_results=random_search.cv_results_,
        tuning_time=tuning_time,
        n_iterations=n_iter
    )


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_method: str = "grid",
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "f1_weighted"
) -> TrainedModel:
    """
    Tune a model and return a TrainedModel object.
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training labels
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        TrainedModel object with tuned model
    """
    if search_method == "grid":
        result = tune_model_grid_search(
            model_name, X_train, y_train, cv=cv, scoring=scoring
        )
    else:
        result = tune_model_random_search(
            model_name, X_train, y_train, n_iter=n_iter, cv=cv, scoring=scoring
        )
    
    return TrainedModel(
        name=model_name,
        model=result.best_model,
        training_time=result.tuning_time,
        params=result.best_params,
        is_tuned=True,
        cv_score=result.best_score
    )


def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_to_tune: Optional[List[str]] = None,
    search_method: str = "random",
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "f1_weighted",
    progress_callback: Optional[callable] = None
) -> Dict[str, TrainedModel]:
    """
    Tune all specified models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        models_to_tune: List of model names (default: all)
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        cv: Number of CV folds
        scoring: Scoring metric
        progress_callback: Optional callback(model_name, i, total)
        
    Returns:
        Dict mapping model names to TrainedModel objects
    """
    if models_to_tune is None:
        models_to_tune = list(MODEL_CONFIGS.keys())
    
    tuned_models = {}
    total = len(models_to_tune)
    
    for i, model_name in enumerate(models_to_tune):
        if progress_callback:
            progress_callback(model_name, i, total)
        
        try:
            tuned_model = tune_model(
                model_name, X_train, y_train,
                search_method=search_method,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring
            )
            tuned_models[model_name] = tuned_model
        except Exception as e:
            print(f"Error tuning {model_name}: {e}")
            continue
    
    return tuned_models


def format_tuning_results(result: TuningResult) -> Dict[str, Any]:
    """
    Format tuning results for display.
    
    Args:
        result: TuningResult object
        
    Returns:
        Dict with formatted results
    """
    cv_results = result.cv_results
    
    # Get top 5 parameter combinations
    indices = np.argsort(cv_results['mean_test_score'])[::-1][:5]
    
    top_results = []
    for idx in indices:
        top_results.append({
            'params': cv_results['params'][idx],
            'mean_score': round(cv_results['mean_test_score'][idx], 4),
            'std_score': round(cv_results['std_test_score'][idx], 4),
            'train_score': round(cv_results['mean_train_score'][idx], 4)
        })
    
    return {
        'model_name': result.model_name,
        'best_score': round(result.best_score, 4),
        'best_params': result.best_params,
        'tuning_time': round(result.tuning_time, 2),
        'n_iterations': result.n_iterations,
        'top_results': top_results
    }
