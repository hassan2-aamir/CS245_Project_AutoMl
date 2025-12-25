"""
Input Validators

Provides validation utilities for user inputs, file uploads,
and data integrity checks throughout the AutoML application.

All validators return tuple of (is_valid: bool, error_message: str)
following a consistent pattern for error handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any
import os


# Constants
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv'}
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """
    Validate that the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename:
        return False, "No filename provided."
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"**Problem**: File format '{ext}' is not supported. "
            f"**Solution**: Please upload a CSV file (.csv extension)."
        )
    
    return True, ""


def validate_file_size(file_size_bytes: int) -> Tuple[bool, str]:
    """
    Validate that the uploaded file is within size limits.
    
    Args:
        file_size_bytes: Size of the file in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, (
            f"**Problem**: File size ({size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit. "
            f"**Solution**: Please reduce your dataset size or split into smaller files."
        )
    
    return True, ""


def validate_dataframe_not_empty(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that a DataFrame is not empty.
    
    Args:
        df: The pandas DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "**Problem**: No data loaded. **Solution**: Please upload a CSV file first."
    
    if df.empty:
        return False, (
            "**Problem**: The uploaded file contains no data. "
            "**Solution**: Please upload a CSV file with at least one row of data."
        )
    
    if len(df.columns) == 0:
        return False, (
            "**Problem**: The uploaded file contains no columns. "
            "**Solution**: Please ensure your CSV file has a header row with column names."
        )
    
    return True, ""


def validate_target_column(
    df: pd.DataFrame, 
    target_column: str
) -> Tuple[bool, str]:
    """
    Validate the selected target column for classification.
    
    Args:
        df: The pandas DataFrame
        target_column: Name of the selected target column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not target_column:
        return False, "**Problem**: No target column selected. **Solution**: Please select a target column."
    
    if target_column not in df.columns:
        return False, (
            f"**Problem**: Column '{target_column}' not found in dataset. "
            f"**Solution**: Please select a valid column from the dropdown."
        )
    
    # Check for null values in target
    null_count = df[target_column].isnull().sum()
    if null_count > 0:
        return False, (
            f"**Problem**: Target column '{target_column}' has {null_count} missing values. "
            f"**Solution**: Please handle missing values in the target column or select a different target."
        )
    
    # Check number of unique classes
    n_classes = df[target_column].nunique()
    if n_classes < 2:
        return False, (
            f"**Problem**: Target column '{target_column}' has only {n_classes} unique value(s). "
            f"**Solution**: Classification requires at least 2 classes. Please select a different target column."
        )
    
    return True, ""


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[bool, str]:
    """
    Validate the selected feature columns.
    
    Args:
        df: The pandas DataFrame
        feature_columns: List of selected feature column names
        target_column: Name of the target column (to ensure exclusion)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not feature_columns or len(feature_columns) == 0:
        return False, (
            "**Problem**: No feature columns selected. "
            "**Solution**: Please select at least one feature column for training."
        )
    
    # Check if target is accidentally included
    if target_column in feature_columns:
        return False, (
            f"**Problem**: Target column '{target_column}' is included in features. "
            f"**Solution**: The target column will be automatically excluded from features."
        )
    
    # Check all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return False, (
            f"**Problem**: Columns not found: {missing_cols}. "
            f"**Solution**: Please select only valid columns from the dataset."
        )
    
    return True, ""


def validate_split_ratio(ratio: float) -> Tuple[bool, str]:
    """
    Validate the train/test split ratio.
    
    Args:
        ratio: The train split ratio (0.0 to 1.0)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if ratio < 0.6 or ratio > 0.9:
        return False, (
            f"**Problem**: Split ratio {ratio:.0%} is outside the allowed range. "
            f"**Solution**: Please set the training ratio between 60% and 90%."
        )
    
    return True, ""


def validate_numeric_input(
    value: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    field_name: str = "Value"
) -> Tuple[bool, str]:
    """
    Validate a numeric input value.
    
    Args:
        value: The value to validate
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value
        field_name: Name of the field for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        return False, f"**Problem**: {field_name} must be a number. **Solution**: Please enter a valid number."
    
    if min_val is not None and num_value < min_val:
        return False, (
            f"**Problem**: {field_name} ({num_value}) is below minimum ({min_val}). "
            f"**Solution**: Please enter a value >= {min_val}."
        )
    
    if max_val is not None and num_value > max_val:
        return False, (
            f"**Problem**: {field_name} ({num_value}) exceeds maximum ({max_val}). "
            f"**Solution**: Please enter a value <= {max_val}."
        )
    
    return True, ""


def validate_prediction_input(
    input_data: dict,
    expected_features: List[str],
    feature_types: dict
) -> Tuple[bool, str]:
    """
    Validate input data for prediction.
    
    Args:
        input_data: Dictionary of feature name to value
        expected_features: List of expected feature names
        feature_types: Dictionary mapping feature names to expected dtypes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check all required features are present
    missing_features = [f for f in expected_features if f not in input_data]
    if missing_features:
        return False, (
            f"**Problem**: Missing required features: {missing_features}. "
            f"**Solution**: Please fill in all required fields."
        )
    
    # Check for empty values
    empty_features = [f for f, v in input_data.items() if v is None or v == ""]
    if empty_features:
        return False, (
            f"**Problem**: Empty values for: {empty_features}. "
            f"**Solution**: Please provide values for all fields."
        )
    
    return True, ""


def validate_csv_content(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the content of a loaded CSV file.
    
    Args:
        df: The loaded pandas DataFrame
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        return False, (
            f"**Problem**: Duplicate column names found: {duplicate_cols}. "
            f"**Solution**: Please ensure all column names are unique in your CSV file."
        )
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        # Warning, not error - we'll flag this as an issue
        pass  # Handled by issue detector
    
    # Check minimum requirements
    if len(df) < 10:
        return False, (
            f"**Problem**: Dataset has only {len(df)} rows. "
            f"**Solution**: Please upload a dataset with at least 10 rows for meaningful analysis."
        )
    
    if len(df.columns) < 2:
        return False, (
            f"**Problem**: Dataset has only {len(df.columns)} column(s). "
            f"**Solution**: Classification requires at least one feature column and one target column."
        )
    
    return True, ""


def validate_model_training_ready(
    X_train,
    y_train,
    X_test,
    y_test
) -> Tuple[bool, str]:
    """
    Validate that data is ready for model training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if X_train is None or y_train is None:
        return False, (
            "**Problem**: Training data not prepared. "
            "**Solution**: Please complete the preprocessing step first."
        )
    
    if X_test is None or y_test is None:
        return False, (
            "**Problem**: Test data not prepared. "
            "**Solution**: Please complete the preprocessing step first."
        )
    
    if len(X_train) == 0:
        return False, (
            "**Problem**: Training set is empty. "
            "**Solution**: Please check your preprocessing configuration and try again."
        )
    
    if len(X_test) == 0:
        return False, (
            "**Problem**: Test set is empty. "
            "**Solution**: Please adjust your train/test split ratio."
        )
    
    # Check for sufficient samples per class
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return False, (
            "**Problem**: Training data has fewer than 2 classes. "
            "**Solution**: Ensure your dataset has multiple classes for classification."
        )
    
    return True, ""
