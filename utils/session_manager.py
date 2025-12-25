"""
Session State Manager

Provides helper functions for managing Streamlit session state
across multiple pages in the AutoML application.

This module standardizes session state access and provides utilities
for state persistence, logging user decisions, and workflow tracking.
"""

import streamlit as st
from typing import Any, Optional, List, Dict
from datetime import datetime


# Standard session state keys
SESSION_KEYS = {
    # Data storage
    "uploaded_df": None,
    "uploaded_filename": None,
    
    # EDA results (cached)
    "eda_results": None,
    "detected_issues": None,
    
    # User decisions for report
    "user_decisions": [],
    
    # Preprocessing configuration
    "target_column": None,
    "feature_columns": None,
    "preprocessing_config": None,
    
    # Train/test splits
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "preprocessing_pipeline": None,
    
    # Model training results
    "trained_models": {},
    "evaluation_results": {},
    "best_model_name": None,
    
    # Workflow state tracking
    "current_step": 1,
    "upload_complete": False,
    "eda_complete": False,
    "preprocessing_complete": False,
    "training_complete": False,
}


def get_state(key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from session state.
    
    Args:
        key: The session state key to retrieve
        default: Default value if key doesn't exist
        
    Returns:
        The stored value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: The session state key
        value: The value to store
    """
    st.session_state[key] = value


def has_state(key: str) -> bool:
    """
    Check if a key exists and has a non-None value in session state.
    
    Args:
        key: The session state key to check
        
    Returns:
        True if key exists and is not None
    """
    return key in st.session_state and st.session_state[key] is not None


def clear_state(key: str) -> None:
    """
    Remove a key from session state.
    
    Args:
        key: The session state key to remove
    """
    if key in st.session_state:
        del st.session_state[key]


def reset_all_state() -> None:
    """Reset all session state to default values."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def initialize_defaults() -> None:
    """Initialize all session state keys with default values."""
    for key, default_value in SESSION_KEYS.items():
        if key not in st.session_state:
            if isinstance(default_value, (list, dict)):
                # Create new instances for mutable types
                st.session_state[key] = type(default_value)()
            else:
                st.session_state[key] = default_value


def log_user_decision(
    decision_type: str,
    description: str,
    details: Optional[Dict] = None
) -> None:
    """
    Log a user decision for inclusion in the final report.
    
    Args:
        decision_type: Category of decision (e.g., "preprocessing", "fix_applied")
        description: Human-readable description of the decision
        details: Optional dictionary with additional details
    """
    if "user_decisions" not in st.session_state:
        st.session_state["user_decisions"] = []
    
    decision = {
        "timestamp": datetime.now().isoformat(),
        "type": decision_type,
        "description": description,
        "details": details or {}
    }
    
    st.session_state["user_decisions"].append(decision)


def get_user_decisions() -> List[Dict]:
    """
    Retrieve all logged user decisions.
    
    Returns:
        List of decision dictionaries
    """
    return st.session_state.get("user_decisions", [])


def update_workflow_step(step: int) -> None:
    """
    Update the current workflow step.
    
    Args:
        step: The step number (1-6)
    """
    st.session_state["current_step"] = step


def mark_step_complete(step_name: str) -> None:
    """
    Mark a workflow step as complete.
    
    Args:
        step_name: One of 'upload', 'eda', 'preprocessing', 'training'
    """
    key = f"{step_name}_complete"
    if key in SESSION_KEYS:
        st.session_state[key] = True


def is_step_complete(step_name: str) -> bool:
    """
    Check if a workflow step is complete.
    
    Args:
        step_name: One of 'upload', 'eda', 'preprocessing', 'training'
        
    Returns:
        True if step is marked complete
    """
    key = f"{step_name}_complete"
    return st.session_state.get(key, False)


def get_dataframe():
    """
    Get the uploaded DataFrame from session state.
    
    Returns:
        The uploaded DataFrame or None
    """
    return st.session_state.get("uploaded_df")


def set_dataframe(df, filename: str = None) -> None:
    """
    Store a DataFrame in session state.
    
    Args:
        df: The pandas DataFrame to store
        filename: Optional filename for reference
    """
    st.session_state["uploaded_df"] = df
    if filename:
        st.session_state["uploaded_filename"] = filename


def get_train_test_data():
    """
    Get the train/test split data from session state.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or (None, None, None, None)
    """
    return (
        st.session_state.get("X_train"),
        st.session_state.get("X_test"),
        st.session_state.get("y_train"),
        st.session_state.get("y_test")
    )


def set_train_test_data(X_train, X_test, y_train, y_test) -> None:
    """
    Store train/test split data in session state.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test


def get_trained_models() -> Dict:
    """
    Get all trained models from session state.
    
    Returns:
        Dictionary mapping model names to model objects
    """
    return st.session_state.get("trained_models", {})


def add_trained_model(name: str, model: Any, metrics: Dict) -> None:
    """
    Add a trained model and its metrics to session state.
    
    Args:
        name: Model name/identifier
        model: The trained model object
        metrics: Dictionary of evaluation metrics
    """
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}
    if "evaluation_results" not in st.session_state:
        st.session_state["evaluation_results"] = {}
    
    st.session_state["trained_models"][name] = model
    st.session_state["evaluation_results"][name] = metrics


def get_best_model():
    """
    Get the best performing model.
    
    Returns:
        Tuple of (model_name, model_object) or (None, None)
    """
    best_name = st.session_state.get("best_model_name")
    if best_name:
        models = st.session_state.get("trained_models", {})
        return best_name, models.get(best_name)
    return None, None


def set_best_model(model_name: str) -> None:
    """
    Set the best model by name.
    
    Args:
        model_name: Name of the best performing model
    """
    st.session_state["best_model_name"] = model_name
