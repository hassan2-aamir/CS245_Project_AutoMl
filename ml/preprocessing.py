"""
Preprocessing Module

Handles data preprocessing including:
- Missing value imputation
- Outlier treatment
- Feature scaling
- Categorical encoding
- Train/test split

Implements FR-29 to FR-35 from the requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    target_column: str
    feature_columns: List[str]
    
    # Missing value handling
    numerical_imputer: str = "median"  # mean, median, most_frequent, constant
    categorical_imputer: str = "most_frequent"  # most_frequent, constant
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, none
    
    # Encoding
    encoding_method: str = "onehot"  # onehot, ordinal
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class PreprocessingResult:
    """Results from preprocessing."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_classes: List[str]
    pipeline: Any  # sklearn Pipeline
    label_encoder: LabelEncoder
    config: PreprocessingConfig


def identify_column_types(
    df: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical columns.
    
    Args:
        df: The DataFrame
        feature_columns: List of feature column names
        
    Returns:
        Tuple of (numerical_columns, categorical_columns)
    """
    numerical_cols = []
    categorical_cols = []
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols


def create_preprocessing_pipeline(
    config: PreprocessingConfig,
    numerical_columns: List[str],
    categorical_columns: List[str]
) -> ColumnTransformer:
    """
    Create a sklearn preprocessing pipeline.
    
    Args:
        config: PreprocessingConfig object
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        
    Returns:
        ColumnTransformer pipeline
    """
    transformers = []
    
    # Numerical pipeline
    if numerical_columns:
        num_steps = []
        
        # Imputation
        if config.numerical_imputer == "mean":
            num_steps.append(('imputer', SimpleImputer(strategy='mean')))
        elif config.numerical_imputer == "median":
            num_steps.append(('imputer', SimpleImputer(strategy='median')))
        elif config.numerical_imputer == "most_frequent":
            num_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        elif config.numerical_imputer == "constant":
            num_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        
        # Scaling
        if config.scaling_method == "standard":
            num_steps.append(('scaler', StandardScaler()))
        elif config.scaling_method == "minmax":
            num_steps.append(('scaler', MinMaxScaler()))
        # else: no scaling
        
        if num_steps:
            transformers.append(('numerical', Pipeline(num_steps), numerical_columns))
        else:
            transformers.append(('numerical', 'passthrough', numerical_columns))
    
    # Categorical pipeline
    if categorical_columns:
        cat_steps = []
        
        # Imputation
        cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        # Encoding
        if config.encoding_method == "onehot":
            cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:  # ordinal
            from sklearn.preprocessing import OrdinalEncoder
            cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        
        transformers.append(('categorical', Pipeline(cat_steps), categorical_columns))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_columns: List[str],
    categorical_columns: List[str],
    df: pd.DataFrame
) -> List[str]:
    """
    Get feature names after preprocessing (handles one-hot encoding).
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_columns: Original numerical column names
        categorical_columns: Original categorical column names
        df: Original DataFrame
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Add numerical column names (unchanged)
    feature_names.extend(numerical_columns)
    
    # Add categorical column names (may be expanded by one-hot)
    try:
        # Try to get feature names from the transformer
        if hasattr(preprocessor, 'get_feature_names_out'):
            all_names = preprocessor.get_feature_names_out()
            return list(all_names)
    except:
        pass
    
    # Fallback: construct names manually
    for col in categorical_columns:
        unique_values = df[col].dropna().unique()
        for val in unique_values:
            feature_names.append(f"{col}_{val}")
    
    return feature_names


def preprocess_data(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> PreprocessingResult:
    """
    Execute the full preprocessing pipeline.
    
    Args:
        df: The pandas DataFrame
        config: PreprocessingConfig object
        
    Returns:
        PreprocessingResult with transformed data
    """
    # Validate columns
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in dataset")
    
    missing_features = [col for col in config.feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")
    
    # Separate features and target
    X = df[config.feature_columns].copy()
    y = df[config.target_column].copy()
    
    # Handle target column
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    target_classes = list(label_encoder.classes_)
    
    # Identify column types
    numerical_cols, categorical_cols = identify_column_types(X, config.feature_columns)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(config, numerical_cols, categorical_cols)
    
    # Split data BEFORE fitting preprocessor (to prevent data leakage)
    stratify_param = y_encoded if config.stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_param
    )
    
    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numerical_cols, categorical_cols, X
    )
    
    return PreprocessingResult(
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_classes=target_classes,
        pipeline=preprocessor,
        label_encoder=label_encoder,
        config=config
    )


def apply_preprocessing_to_new_data(
    df: pd.DataFrame,
    preprocessing_result: PreprocessingResult
) -> np.ndarray:
    """
    Apply fitted preprocessing pipeline to new data.
    
    Args:
        df: New data DataFrame
        preprocessing_result: Previously fitted PreprocessingResult
        
    Returns:
        Transformed feature array
    """
    X = df[preprocessing_result.config.feature_columns].copy()
    return preprocessing_result.pipeline.transform(X)


def get_preprocessing_summary(config: PreprocessingConfig) -> Dict[str, str]:
    """
    Get a human-readable summary of preprocessing configuration.
    
    Args:
        config: PreprocessingConfig object
        
    Returns:
        Dictionary with configuration summary
    """
    imputer_names = {
        "mean": "Mean Imputation",
        "median": "Median Imputation",
        "most_frequent": "Mode Imputation",
        "constant": "Constant Value (0)"
    }
    
    scaler_names = {
        "standard": "StandardScaler (Z-score)",
        "minmax": "MinMaxScaler (0-1)",
        "none": "No Scaling"
    }
    
    encoder_names = {
        "onehot": "One-Hot Encoding",
        "ordinal": "Ordinal Encoding"
    }
    
    return {
        "Target Column": config.target_column,
        "Feature Count": str(len(config.feature_columns)),
        "Numerical Imputation": imputer_names.get(config.numerical_imputer, config.numerical_imputer),
        "Categorical Imputation": imputer_names.get(config.categorical_imputer, config.categorical_imputer),
        "Scaling Method": scaler_names.get(config.scaling_method, config.scaling_method),
        "Encoding Method": encoder_names.get(config.encoding_method, config.encoding_method),
        "Test Size": f"{config.test_size * 100:.0f}%",
        "Stratified Split": "Yes" if config.stratify else "No",
        "Random State": str(config.random_state)
    }


def validate_preprocessing_config(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[bool, str]:
    """
    Validate preprocessing configuration.
    
    Args:
        df: The DataFrame
        config: PreprocessingConfig to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check target column
    if not config.target_column:
        return False, "Target column not specified."
    
    if config.target_column not in df.columns:
        return False, f"Target column '{config.target_column}' not found in dataset."
    
    # Check target has no missing values
    if df[config.target_column].isnull().any():
        return False, f"Target column '{config.target_column}' contains missing values."
    
    # Check target has at least 2 classes
    n_classes = df[config.target_column].nunique()
    if n_classes < 2:
        return False, f"Target column '{config.target_column}' has only {n_classes} unique value(s). Classification requires at least 2 classes."
    
    # Check feature columns
    if not config.feature_columns or len(config.feature_columns) == 0:
        return False, "No feature columns specified."
    
    missing_cols = [col for col in config.feature_columns if col not in df.columns]
    if missing_cols:
        return False, f"Feature columns not found: {missing_cols}"
    
    # Check target not in features
    if config.target_column in config.feature_columns:
        return False, "Target column should not be included in feature columns."
    
    # Check test size
    if config.test_size < 0.1 or config.test_size > 0.4:
        return False, f"Test size {config.test_size} is outside valid range (0.1 to 0.4)."
    
    # Check stratification is possible
    if config.stratify:
        class_counts = df[config.target_column].value_counts()
        min_count = class_counts.min()
        n_splits = int(1 / config.test_size) + 1
        if min_count < n_splits:
            return False, f"Minority class has only {min_count} samples. Cannot perform stratified split. Consider disabling stratification or using more training data."
    
    return True, ""
