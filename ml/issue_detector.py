"""
Issue Detector Module

Automatically detects data quality issues in the dataset including:
- Missing values
- Outliers
- Class imbalance
- High cardinality features
- Constant/near-constant features
- Multicollinearity

Implements FR-18 to FR-23 from the requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class DataIssue:
    """Represents a single detected data quality issue."""
    issue_type: str
    severity: IssueSeverity
    affected_columns: List[str]
    description: str
    suggestion: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueDetectionResults:
    """Container for all detected issues."""
    issues: List[DataIssue]
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    
    @classmethod
    def from_issues(cls, issues: List[DataIssue]) -> 'IssueDetectionResults':
        """Create results from a list of issues."""
        return cls(
            issues=issues,
            total_issues=len(issues),
            critical_count=sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            high_count=sum(1 for i in issues if i.severity == IssueSeverity.HIGH),
            medium_count=sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM),
            low_count=sum(1 for i in issues if i.severity == IssueSeverity.LOW),
        )


# Detection thresholds (from requirements)
MISSING_VALUE_HIGH_THRESHOLD = 50  # >50% missing = HIGH severity
MISSING_VALUE_MEDIUM_THRESHOLD = 20  # >20% missing = MEDIUM severity
CLASS_IMBALANCE_RATIO = 3  # Majority:Minority ratio > 3:1 = flag (FR-20)
HIGH_CARDINALITY_THRESHOLD = 50  # >50 unique values = flag (FR-21)
CONSTANT_FEATURE_THRESHOLD = 1  # <=1 unique value = flag (FR-22)
MULTICOLLINEARITY_THRESHOLD = 0.95  # |correlation| > 0.95 = flag (FR-23)
OUTLIER_HIGH_THRESHOLD = 10  # >10% outliers = HIGH severity
OUTLIER_MEDIUM_THRESHOLD = 5  # >5% outliers = MEDIUM severity


def detect_missing_values(df: pd.DataFrame) -> List[DataIssue]:
    """
    Detect columns with missing values.
    
    Implements FR-18.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        List of DataIssue objects for missing value problems
    """
    issues = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_pct > 0:
            # Determine severity
            if missing_pct > MISSING_VALUE_HIGH_THRESHOLD:
                severity = IssueSeverity.HIGH
                suggestion = f"Consider dropping column '{col}' (>{MISSING_VALUE_HIGH_THRESHOLD}% missing) or use advanced imputation."
            elif missing_pct > MISSING_VALUE_MEDIUM_THRESHOLD:
                severity = IssueSeverity.MEDIUM
                suggestion = f"Impute missing values in '{col}' using median (numeric) or mode (categorical)."
            else:
                severity = IssueSeverity.LOW
                suggestion = f"Impute missing values in '{col}' using mean, median, or mode."
            
            issues.append(DataIssue(
                issue_type="Missing Values",
                severity=severity,
                affected_columns=[col],
                description=f"Column '{col}' has {missing_count:,} missing values ({missing_pct:.1f}%).",
                suggestion=suggestion,
                details={
                    'missing_count': missing_count,
                    'missing_pct': round(missing_pct, 2),
                    'dtype': str(df[col].dtype)
                }
            ))
    
    return issues


def detect_outliers(
    df: pd.DataFrame,
    method: str = 'iqr'
) -> List[DataIssue]:
    """
    Detect columns with outliers.
    
    Implements FR-19.
    
    Args:
        df: The pandas DataFrame
        method: Detection method ('iqr' or 'zscore')
        
    Returns:
        List of DataIssue objects for outlier problems
    """
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        else:  # zscore
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue
            z_scores = np.abs((col_data - mean) / std)
            outlier_count = (z_scores > 3).sum()
        
        outlier_pct = (outlier_count / len(col_data)) * 100
        
        if outlier_count > 0:
            # Determine severity
            if outlier_pct > OUTLIER_HIGH_THRESHOLD:
                severity = IssueSeverity.HIGH
            elif outlier_pct > OUTLIER_MEDIUM_THRESHOLD:
                severity = IssueSeverity.MEDIUM
            else:
                severity = IssueSeverity.LOW
            
            issues.append(DataIssue(
                issue_type="Outliers",
                severity=severity,
                affected_columns=[col],
                description=f"Column '{col}' has {outlier_count:,} outliers ({outlier_pct:.1f}%) detected using {method.upper()} method.",
                suggestion=f"Consider capping (Winsorize), removing, or keeping outliers in '{col}' based on domain knowledge.",
                details={
                    'outlier_count': outlier_count,
                    'outlier_pct': round(outlier_pct, 2),
                    'method': method
                }
            ))
    
    return issues


def detect_class_imbalance(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> List[DataIssue]:
    """
    Detect class imbalance in the target variable.
    
    Implements FR-20.
    
    Args:
        df: The pandas DataFrame
        target_column: Name of the target column (if None, tries to detect)
        
    Returns:
        List of DataIssue objects for class imbalance
    """
    issues = []
    
    if target_column is None:
        return issues
    
    if target_column not in df.columns:
        return issues
    
    class_counts = df[target_column].value_counts()
    
    if len(class_counts) < 2:
        return issues
    
    majority_count = class_counts.iloc[0]
    minority_count = class_counts.iloc[-1]
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
    
    if imbalance_ratio > CLASS_IMBALANCE_RATIO:
        severity = IssueSeverity.HIGH if imbalance_ratio > 10 else IssueSeverity.MEDIUM
        
        issues.append(DataIssue(
            issue_type="Class Imbalance",
            severity=severity,
            affected_columns=[target_column],
            description=f"Target column '{target_column}' has class imbalance with ratio {imbalance_ratio:.1f}:1 (threshold: {CLASS_IMBALANCE_RATIO}:1).",
            suggestion="Consider using class weights, SMOTE oversampling, or stratified sampling to handle imbalance.",
            details={
                'class_distribution': class_counts.to_dict(),
                'imbalance_ratio': round(imbalance_ratio, 2),
                'majority_class': class_counts.index[0],
                'minority_class': class_counts.index[-1]
            }
        ))
    
    return issues


def detect_high_cardinality(df: pd.DataFrame) -> List[DataIssue]:
    """
    Detect categorical features with high cardinality.
    
    Implements FR-21.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        List of DataIssue objects for high cardinality problems
    """
    issues = []
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if unique_count > HIGH_CARDINALITY_THRESHOLD:
            # Determine if it might be an ID column
            is_likely_id = unique_count == len(df) or 'id' in col.lower()
            
            if is_likely_id:
                suggestion = f"Column '{col}' appears to be an identifier. Consider excluding it from features."
                severity = IssueSeverity.MEDIUM
            else:
                suggestion = f"Column '{col}' has high cardinality. Consider grouping rare categories, using target encoding, or limiting to top {HIGH_CARDINALITY_THRESHOLD} categories."
                severity = IssueSeverity.MEDIUM
            
            issues.append(DataIssue(
                issue_type="High Cardinality",
                severity=severity,
                affected_columns=[col],
                description=f"Column '{col}' has {unique_count:,} unique values (threshold: {HIGH_CARDINALITY_THRESHOLD}).",
                suggestion=suggestion,
                details={
                    'unique_count': unique_count,
                    'threshold': HIGH_CARDINALITY_THRESHOLD,
                    'is_likely_id': is_likely_id
                }
            ))
    
    return issues


def detect_constant_features(df: pd.DataFrame) -> List[DataIssue]:
    """
    Detect constant or near-constant features.
    
    Implements FR-22.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        List of DataIssue objects for constant features
    """
    issues = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        if unique_count <= CONSTANT_FEATURE_THRESHOLD:
            if unique_count == 0:
                description = f"Column '{col}' is completely empty (all null values)."
            elif unique_count == 1:
                single_value = df[col].dropna().iloc[0] if df[col].notna().any() else "NULL"
                description = f"Column '{col}' is constant (only value: '{single_value}')."
            else:
                description = f"Column '{col}' has only {unique_count} unique value(s)."
            
            issues.append(DataIssue(
                issue_type="Constant Feature",
                severity=IssueSeverity.HIGH,
                affected_columns=[col],
                description=description,
                suggestion=f"Consider dropping column '{col}' as it provides no discriminative information.",
                details={
                    'unique_count': unique_count,
                    'sample_values': df[col].dropna().unique()[:5].tolist()
                }
            ))
    
    return issues


def detect_multicollinearity(
    df: pd.DataFrame,
    threshold: float = MULTICOLLINEARITY_THRESHOLD
) -> List[DataIssue]:
    """
    Detect highly correlated feature pairs.
    
    Implements FR-23.
    
    Args:
        df: The pandas DataFrame
        threshold: Correlation threshold (default: 0.95)
        
    Returns:
        List of DataIssue objects for multicollinearity
    """
    issues = []
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return issues
    
    corr_matrix = numeric_df.corr().abs()
    
    # Find pairs above threshold
    high_corr_pairs = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= threshold:
                high_corr_pairs.append((cols[i], cols[j], round(corr_value, 3)))
    
    if high_corr_pairs:
        # Group all pairs into one issue
        affected_cols = list(set(
            col for pair in high_corr_pairs for col in [pair[0], pair[1]]
        ))
        
        pair_descriptions = [f"'{p[0]}' and '{p[1]}' (r={p[2]})" for p in high_corr_pairs]
        
        issues.append(DataIssue(
            issue_type="Multicollinearity",
            severity=IssueSeverity.MEDIUM,
            affected_columns=affected_cols,
            description=f"Found {len(high_corr_pairs)} highly correlated feature pair(s) with |correlation| > {threshold}: {', '.join(pair_descriptions[:3])}{'...' if len(pair_descriptions) > 3 else ''}",
            suggestion="Consider removing one feature from each correlated pair to reduce multicollinearity.",
            details={
                'correlated_pairs': high_corr_pairs,
                'threshold': threshold
            }
        ))
    
    return issues


def run_issue_detection(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> IssueDetectionResults:
    """
    Run all issue detection checks on the dataset.
    
    This is the main entry point for issue detection.
    
    Args:
        df: The pandas DataFrame to analyze
        target_column: Optional name of the target column
        
    Returns:
        IssueDetectionResults containing all detected issues
    """
    all_issues = []
    
    # Run all detection methods
    all_issues.extend(detect_missing_values(df))
    all_issues.extend(detect_outliers(df, method='iqr'))
    all_issues.extend(detect_constant_features(df))
    all_issues.extend(detect_high_cardinality(df))
    all_issues.extend(detect_multicollinearity(df))
    
    if target_column:
        all_issues.extend(detect_class_imbalance(df, target_column))
    
    # Sort by severity (Critical > High > Medium > Low)
    severity_order = {
        IssueSeverity.CRITICAL: 0,
        IssueSeverity.HIGH: 1,
        IssueSeverity.MEDIUM: 2,
        IssueSeverity.LOW: 3
    }
    all_issues.sort(key=lambda x: severity_order[x.severity])
    
    return IssueDetectionResults.from_issues(all_issues)


def get_issue_summary(results: IssueDetectionResults) -> Dict[str, Any]:
    """
    Get a summary of detected issues.
    
    Args:
        results: The IssueDetectionResults object
        
    Returns:
        Dictionary with issue summary
    """
    issue_types = {}
    for issue in results.issues:
        if issue.issue_type not in issue_types:
            issue_types[issue.issue_type] = 0
        issue_types[issue.issue_type] += 1
    
    return {
        'total_issues': results.total_issues,
        'by_severity': {
            'Critical': results.critical_count,
            'High': results.high_count,
            'Medium': results.medium_count,
            'Low': results.low_count
        },
        'by_type': issue_types,
        'affected_columns': list(set(
            col for issue in results.issues for col in issue.affected_columns
        ))
    }
