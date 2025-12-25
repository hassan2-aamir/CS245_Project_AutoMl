"""
Preprocess and Split Page

Allows users to configure preprocessing options and train/test split.
Implements FR-29 to FR-35 from the requirements.

This page allows users to:
1. Select target and feature columns
2. Configure imputation, scaling, and encoding methods
3. Set train/test split ratio
4. Preview and apply preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.preprocessing import (
    PreprocessingConfig,
    preprocess_data,
    validate_preprocessing_config,
    get_preprocessing_summary,
    identify_column_types,
)
from ml.eda import get_train_test_preview
from utils.session_manager import (
    get_dataframe,
    get_state,
    set_state,
    mark_step_complete,
    log_user_decision,
    set_train_test_data,
)
from utils.validators import validate_target_column, validate_feature_columns

# Page configuration
st.set_page_config(
    page_title="Preprocess & Split - AutoML",
    layout="wide",
)

st.title("Preprocessing & Train/Test Split")

# Check if data is loaded
df = get_dataframe()

if df is None:
    st.warning("No dataset loaded. Please go to **Upload & Info** first.")
    st.stop()

st.markdown(f"""
Configure preprocessing for your dataset with **{len(df):,} rows** and **{len(df.columns)} columns**.
""")

st.markdown("---")

# Step 1: Target and Feature Selection
st.subheader("1. Select Target and Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Target Column (y)**")
    st.markdown("Select the column you want to predict.")
    
    target_column = st.selectbox(
        "Target column",
        options=df.columns.tolist(),
        key="target_select",
        help="The column containing the class labels to predict"
    )
    
    # Validate and show target info
    if target_column:
        is_valid, error = validate_target_column(df, target_column)
        if not is_valid:
            st.error(error)
        else:
            n_classes = df[target_column].nunique()
            class_dist = df[target_column].value_counts()
            st.success(f"Valid target with {n_classes} classes")
            
            with st.expander("View class distribution"):
                st.dataframe(
                    pd.DataFrame({
                        'Class': class_dist.index,
                        'Count': class_dist.values,
                        '%': (class_dist.values / len(df) * 100).round(2)
                    }),
                    use_container_width=True,
                    hide_index=True
                )

with col2:
    st.markdown("**Feature Columns (X)**")
    st.markdown("Select the columns to use as features for training.")
    
    # Get all columns except target
    available_features = [col for col in df.columns if col != target_column]
    
    # Default: select all non-target columns
    default_features = available_features
    
    feature_columns = st.multiselect(
        "Feature columns",
        options=available_features,
        default=default_features,
        key="feature_select",
        help="Select one or more feature columns"
    )
    
    if feature_columns:
        numerical_cols, categorical_cols = identify_column_types(df, feature_columns)
        st.info(f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical features selected")

st.markdown("---")

# Step 2: Preprocessing Options
st.subheader("2. Configure Preprocessing")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Missing Value Handling**")
    
    numerical_imputer = st.selectbox(
        "Numerical columns",
        options=["median", "mean", "most_frequent", "constant"],
        index=0,
        key="num_imputer",
        help="Method to fill missing values in numerical columns"
    )
    
    with st.expander("Imputation methods"):
        st.markdown("""
        - **Median**: Robust to outliers, good default
        - **Mean**: Sensitive to outliers
        - **Mode (most_frequent)**: Use the most common value
        - **Constant**: Fill with 0
        """)

with col2:
    st.markdown("**Feature Scaling**")
    
    scaling_method = st.selectbox(
        "Scaling method",
        options=["standard", "minmax", "none"],
        index=0,
        key="scaler",
        help="How to normalize numerical features"
    )
    
    with st.expander("Scaling methods"):
        st.markdown("""
        - **StandardScaler**: Mean=0, Std=1. Good for SVM, Logistic Regression
        - **MinMaxScaler**: Range [0,1]. Good for neural networks
        - **None**: Keep original values. Good for tree-based models
        """)

with col3:
    st.markdown("**Categorical Encoding**")
    
    encoding_method = st.selectbox(
        "Encoding method",
        options=["onehot", "ordinal"],
        index=0,
        key="encoder",
        help="How to convert categorical features to numbers"
    )
    
    with st.expander("Encoding methods"):
        st.markdown("""
        - **One-Hot**: Creates binary columns for each category. Best for nominal categories.
        - **Ordinal**: Assigns integer values. Use when categories have natural order.
        """)

st.markdown("---")

# Step 3: Train/Test Split
st.subheader("3. Configure Train/Test Split")

col1, col2 = st.columns([2, 1])

with col1:
    train_ratio = st.slider(
        "Training set size (%)",
        min_value=60,
        max_value=90,
        value=80,
        step=5,
        key="train_ratio",
        help="Percentage of data to use for training (remaining goes to test set)"
    )
    
    test_size = (100 - train_ratio) / 100
    
    # Preview split
    preview = get_train_test_preview(df, train_ratio / 100)
    
    st.markdown(f"""
    | Set | Rows | Percentage |
    |-----|------|------------|
    | **Training** | {preview['train_rows']:,} | {preview['train_pct']}% |
    | **Test** | {preview['test_rows']:,} | {preview['test_pct']}% |
    """)

with col2:
    st.markdown("**Split Options**")
    
    stratify = st.checkbox(
        "Stratified split",
        value=True,
        help="Maintain class proportions in train/test sets (recommended for imbalanced data)"
    )
    
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducibility. Use 42 for consistent results."
    )

st.markdown("---")

# Step 4: Summary and Apply
st.subheader("4. Review and Apply")

if target_column and feature_columns:
    # Create configuration
    config = PreprocessingConfig(
        target_column=target_column,
        feature_columns=feature_columns,
        numerical_imputer=numerical_imputer,
        categorical_imputer="most_frequent",
        scaling_method=scaling_method,
        encoding_method=encoding_method,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Validate configuration
    is_valid, error_msg = validate_preprocessing_config(df, config)
    
    if not is_valid:
        st.error(error_msg)
    else:
        # Show configuration summary
        summary = get_preprocessing_summary(config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuration Summary:**")
            for key, value in summary.items():
                st.markdown(f"- **{key}**: {value}")
        
        with col2:
            # Warning for high cardinality
            for col in feature_columns:
                if df[col].dtype == 'object' and df[col].nunique() > 50:
                    st.warning(f"Column '{col}' has {df[col].nunique()} unique values. One-Hot encoding may create many features.")
        
        # Apply preprocessing button
        st.markdown("---")
        
        if st.button("Apply Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Applying preprocessing..."):
                try:
                    result = preprocess_data(df, config)
                    
                    # Store results in session state
                    set_train_test_data(
                        result.X_train,
                        result.X_test,
                        result.y_train,
                        result.y_test
                    )
                    set_state("preprocessing_result", result)
                    set_state("preprocessing_config", config)
                    set_state("preprocessing_pipeline", result.pipeline)
                    set_state("label_encoder", result.label_encoder)
                    set_state("target_classes", result.target_classes)
                    set_state("feature_names", result.feature_names)
                    
                    # Log decision
                    log_user_decision(
                        decision_type="preprocessing",
                        description="Applied preprocessing pipeline",
                        details=summary
                    )
                    
                    mark_step_complete("preprocessing")
                    set_state("preprocessing_complete", True)
                    
                    st.success("Preprocessing complete!")
                    
                    # Show results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Training Samples", f"{len(result.X_train):,}")
                    with col2:
                        st.metric("Test Samples", f"{len(result.X_test):,}")
                    with col3:
                        st.metric("Features (after encoding)", len(result.feature_names))
                    with col4:
                        st.metric("Target Classes", len(result.target_classes))
                    
                    # Show class distribution in train/test
                    st.markdown("**Class Distribution:**")
                    train_classes, train_counts = np.unique(result.y_train, return_counts=True)
                    test_classes, test_counts = np.unique(result.y_test, return_counts=True)
                    
                    dist_df = pd.DataFrame({
                        'Class': [result.target_classes[i] for i in train_classes],
                        'Train Count': train_counts,
                        'Train %': (train_counts / len(result.y_train) * 100).round(2),
                        'Test Count': test_counts,
                        'Test %': (test_counts / len(result.y_test) * 100).round(2)
                    })
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"**Problem**: Preprocessing failed - {str(e)}. **Solution**: Check your configuration and try again.")

else:
    st.info("Please select target and feature columns above to continue.")

# Show existing preprocessing if available
if get_state("preprocessing_complete"):
    st.markdown("---")
    st.success("""
    Preprocessing Already Applied!
    
    Next Step: Go to Train & Tune in the sidebar to train classification models.
    
    You can modify the configuration above and re-apply if needed.
    """)
