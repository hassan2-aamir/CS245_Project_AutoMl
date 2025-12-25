"""
Exploratory Data Analysis Module

Provides automated EDA functions including:
- Missing value analysis
- Outlier detection (IQR and Z-score methods)
- Correlation analysis
- Distribution analysis

Implements FR-10 to FR-17 from the requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EDAResults:
    """Container for all EDA results."""
    missing_value_analysis: pd.DataFrame
    global_missing_pct: float
    outlier_analysis_iqr: pd.DataFrame
    outlier_analysis_zscore: pd.DataFrame
    correlation_matrix: pd.DataFrame
    numerical_columns: List[str]
    categorical_columns: List[str]
    column_statistics: Dict[str, Dict]


def analyze_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Analyze missing values in the dataset.
    
    Implements FR-10 and FR-11.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Tuple of (per-column DataFrame, global missing percentage)
    """
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df) * 100).round(2)
    
    analysis = pd.DataFrame({
        'feature': df.columns,
        'missing_count': missing_counts.values,
        'missing_pct': missing_pcts.values,
        'dtype': df.dtypes.values.astype(str)
    })
    
    analysis = analysis.sort_values('missing_pct', ascending=False)
    
    # Calculate global missing percentage
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    global_missing_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0
    
    return analysis, global_missing_pct


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Implements FR-12.
    
    Outliers are defined as values:
    - Below Q1 - 1.5 * IQR
    - Above Q3 + 1.5 * IQR
    
    Args:
        df: The pandas DataFrame
        columns: Optional list of columns to analyze (default: all numeric)
        
    Returns:
        DataFrame with outlier statistics per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_low = (col_data < lower_bound).sum()
        outliers_high = (col_data > upper_bound).sum()
        total_outliers = outliers_low + outliers_high
        
        results.append({
            'feature': col,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'outliers_low': outliers_low,
            'outliers_high': outliers_high,
            'total_outliers': total_outliers,
            'outlier_pct': round(total_outliers / len(col_data) * 100, 2)
        })
    
    return pd.DataFrame(results)


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers using the Z-score method.
    
    Implements FR-13.
    
    Outliers are defined as values with |Z-score| > threshold.
    
    Args:
        df: The pandas DataFrame
        columns: Optional list of columns to analyze (default: all numeric)
        threshold: Z-score threshold for outlier detection (default: 3.0)
        
    Returns:
        DataFrame with outlier statistics per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        mean = col_data.mean()
        std = col_data.std()
        
        if std == 0:
            # No variation, no outliers possible
            results.append({
                'feature': col,
                'mean': round(mean, 2),
                'std': 0,
                'threshold': threshold,
                'outliers_count': 0,
                'outlier_pct': 0
            })
            continue
        
        z_scores = np.abs((col_data - mean) / std)
        outliers_count = (z_scores > threshold).sum()
        
        results.append({
            'feature': col,
            'mean': round(mean, 2),
            'std': round(std, 2),
            'threshold': threshold,
            'outliers_count': outliers_count,
            'outlier_pct': round(outliers_count / len(col_data) * 100, 2)
        })
    
    return pd.DataFrame(results)


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for numerical features.
    
    Implements FR-14.
    
    Args:
        df: The pandas DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix DataFrame
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method).round(3)


def get_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of features with correlation above threshold.
    
    Args:
        corr_matrix: The correlation matrix
        threshold: Correlation threshold (default: 0.95)
        
    Returns:
        List of tuples (feature1, feature2, correlation)
    """
    if corr_matrix.empty:
        return []
    
    pairs = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                pairs.append((cols[i], cols[j], round(corr_matrix.iloc[i, j], 3)))
    
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def analyze_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze the distribution of each column.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        Dictionary with distribution statistics per column
    """
    distributions = {}
    
    for col in df.columns:
        col_data = df[col]
        stats = {
            'dtype': str(col_data.dtype),
            'count': len(col_data),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique(),
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) > 0:
                stats.update({
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'skewness': float(non_null.skew()) if len(non_null) > 2 else 0,
                    'kurtosis': float(non_null.kurtosis()) if len(non_null) > 3 else 0,
                })
        else:
            # Categorical column
            value_counts = col_data.value_counts()
            if len(value_counts) > 0:
                stats.update({
                    'top_value': str(value_counts.index[0]),
                    'top_count': int(value_counts.iloc[0]),
                    'top_pct': round(value_counts.iloc[0] / len(col_data) * 100, 2),
                })
        
        distributions[col] = stats
    
    return distributions


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numerical column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_train_test_preview(df: pd.DataFrame, train_ratio: float) -> Dict:
    """
    Get a preview of train/test split sizes.
    
    Args:
        df: The DataFrame
        train_ratio: Proportion of data for training (0-1)
        
    Returns:
        Dict with train_rows, test_rows, train_pct, test_pct
    """
    n = len(df)
    train_rows = int(n * train_ratio)
    test_rows = n - train_rows
    
    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pct": round(train_ratio * 100, 1),
        "test_pct": round((1 - train_ratio) * 100, 1)
    }


def run_full_eda(df: pd.DataFrame) -> EDAResults:
    """
    Run complete exploratory data analysis on the dataset.
    
    This is the main entry point for EDA, running all analyses
    and returning consolidated results.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        EDAResults object containing all analysis results
    """
    # Get column types
    numerical_cols = get_numerical_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Missing value analysis
    missing_analysis, global_missing_pct = analyze_missing_values(df)
    
    # Outlier detection
    outliers_iqr = detect_outliers_iqr(df, numerical_cols)
    outliers_zscore = detect_outliers_zscore(df, numerical_cols)
    
    # Correlation matrix
    corr_matrix = compute_correlation_matrix(df)
    
    # Distribution statistics
    column_stats = analyze_distributions(df)
    
    return EDAResults(
        missing_value_analysis=missing_analysis,
        global_missing_pct=global_missing_pct,
        outlier_analysis_iqr=outliers_iqr,
        outlier_analysis_zscore=outliers_zscore,
        correlation_matrix=corr_matrix,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        column_statistics=column_stats
    )


def get_train_test_preview(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Dict[str, int]:
    """
    Preview train/test split dimensions.
    
    Implements FR-17.
    
    Args:
        df: The pandas DataFrame
        train_ratio: Training set ratio (0.6 to 0.9)
        
    Returns:
        Dictionary with train_rows, test_rows, train_pct, test_pct
    """
    total_rows = len(df)
    train_rows = int(total_rows * train_ratio)
    test_rows = total_rows - train_rows
    
    return {
        'total_rows': total_rows,
        'train_rows': train_rows,
        'test_rows': test_rows,
        'train_pct': round(train_ratio * 100, 1),
        'test_pct': round((1 - train_ratio) * 100, 1)
    }
