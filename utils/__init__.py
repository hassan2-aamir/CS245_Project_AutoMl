"""
Utils Package Initialization

This package contains utility modules for the AutoML application:
- session_manager: Session state management helpers
- validators: Input validation utilities  
- visualizations: Matplotlib/Seaborn plot wrappers
"""

from .session_manager import (
    get_state,
    set_state,
    has_state,
    clear_state,
    reset_all_state,
    initialize_defaults,
    log_user_decision,
    get_user_decisions,
    mark_step_complete,
    is_step_complete,
    get_dataframe,
    set_dataframe,
    get_train_test_data,
    set_train_test_data,
    get_trained_models,
    add_trained_model,
    get_best_model,
    set_best_model,
)

from .validators import (
    validate_file_extension,
    validate_file_size,
    validate_dataframe_not_empty,
    validate_target_column,
    validate_feature_columns,
    validate_split_ratio,
    validate_numeric_input,
    validate_prediction_input,
    validate_csv_content,
    validate_model_training_ready,
)

from .visualizations import (
    create_histogram,
    create_bar_plot,
    create_correlation_heatmap,
    create_confusion_matrix,
    create_roc_curves,
    create_metrics_comparison_bar,
    create_class_distribution_plot,
    create_missing_values_plot,
    fig_to_base64,
    close_all_figures,
)

__all__ = [
    # Session manager
    'get_state',
    'set_state', 
    'has_state',
    'clear_state',
    'reset_all_state',
    'initialize_defaults',
    'log_user_decision',
    'get_user_decisions',
    'mark_step_complete',
    'is_step_complete',
    'get_dataframe',
    'set_dataframe',
    'get_train_test_data',
    'set_train_test_data',
    'get_trained_models',
    'add_trained_model',
    'get_best_model',
    'set_best_model',
    
    # Validators
    'validate_file_extension',
    'validate_file_size',
    'validate_dataframe_not_empty',
    'validate_target_column',
    'validate_feature_columns',
    'validate_split_ratio',
    'validate_numeric_input',
    'validate_prediction_input',
    'validate_csv_content',
    'validate_model_training_ready',
    
    # Visualizations
    'create_histogram',
    'create_bar_plot',
    'create_correlation_heatmap',
    'create_confusion_matrix',
    'create_roc_curves',
    'create_metrics_comparison_bar',
    'create_class_distribution_plot',
    'create_missing_values_plot',
    'fig_to_base64',
    'close_all_figures',
]
