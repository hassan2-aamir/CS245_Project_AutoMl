"""
ML Package Initialization

This package contains machine learning modules for the AutoML application:
- data_loader: CSV loading and validation
- eda: Exploratory data analysis
- issue_detector: Data quality issue detection
- preprocessing: Data preprocessing pipeline
- models: Classification models
- tuning: Hyperparameter optimization
- evaluation: Model evaluation metrics
- report_generator: Report generation
"""

from .data_loader import (
    load_csv,
    detect_encoding,
    get_column_info,
    get_summary_statistics,
    get_categorical_summary,
    infer_column_types,
)

from .eda import (
    run_full_eda,
    analyze_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    compute_correlation_matrix,
    get_train_test_preview,
    EDAResults,
)

from .issue_detector import (
    run_issue_detection,
    detect_missing_values,
    detect_class_imbalance,
    DataIssue,
    IssueSeverity,
    IssueDetectionResults,
)

from .preprocessing import (
    preprocess_data,
    PreprocessingConfig,
    PreprocessingResult,
    apply_preprocessing_to_new_data,
    validate_preprocessing_config,
)

from .models import (
    get_available_models,
    get_model_description,
    train_model,
    train_all_models,
    TrainedModel,
    MODEL_CONFIGS,
)

from .tuning import (
    tune_model,
    tune_all_models,
    tune_model_grid_search,
    tune_model_random_search,
    TuningResult,
)

from .evaluation import (
    evaluate_model,
    evaluate_all_models,
    create_comparison_dataframe,
    get_best_model,
    EvaluationResult,
)

from .report_generator import (
    generate_html_report,
    generate_markdown_report,
    save_report_to_file,
)

__all__ = [
    # Data loader
    'load_csv',
    'detect_encoding',
    'get_column_info',
    'get_summary_statistics',
    'get_categorical_summary',
    'infer_column_types',
    # EDA
    'run_full_eda',
    'analyze_missing_values',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'compute_correlation_matrix',
    'get_train_test_preview',
    'EDAResults',
    # Issue detector
    'run_issue_detection',
    'detect_missing_values',
    'detect_class_imbalance',
    'DataIssue',
    'IssueSeverity',
    # Preprocessing
    'preprocess_data',
    'PreprocessingConfig',
    'PreprocessingResult',
    'apply_preprocessing_to_new_data',
    'validate_preprocessing_config',
    # Models
    'get_available_models',
    'get_model_description',
    'train_model',
    'train_all_models',
    'TrainedModel',
    'MODEL_CONFIGS',
    # Tuning
    'tune_model',
    'tune_all_models',
    'tune_model_grid_search',
    'tune_model_random_search',
    'TuningResult',
    # Evaluation
    'evaluate_model',
    'evaluate_all_models',
    'create_comparison_dataframe',
    'get_best_model',
    'EvaluationResult',
    # Report generator
    'generate_html_report',
    'generate_markdown_report',
    'save_report_to_file',
]
