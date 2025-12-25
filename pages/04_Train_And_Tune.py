"""
Train and Tune Page

Allows users to train and optionally tune all 7 classifiers.
Implements FR-36 to FR-45 from the requirements.

This page allows users to:
1. Select which models to train
2. Choose training mode (default or tuned)
3. Monitor training progress
4. View preliminary results
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models import (
    get_available_models,
    get_model_description,
    train_all_models,
    TrainedModel
)
from ml.tuning import tune_all_models, get_tuning_grid
from ml.evaluation import evaluate_all_models, create_comparison_dataframe, get_best_model
from utils.session_manager import (
    get_state,
    set_state,
    mark_step_complete,
    log_user_decision,
    get_train_test_data,
)

# Page configuration
st.set_page_config(
    page_title="Train & Tune - AutoML",
    layout="wide",
)

st.title("Model Training & Tuning")

# Check prerequisites
X_train, X_test, y_train, y_test = get_train_test_data()
preprocessing_result = get_state("preprocessing_result")

if X_train is None or preprocessing_result is None:
    st.warning("Preprocessing not complete. Please go to Preprocess & Split first.")
    st.stop()

# Get metadata
target_classes = get_state("target_classes")
n_samples, n_features = X_train.shape

st.markdown(f"""
Train classification models on your preprocessed data:
- **Training samples**: {n_samples:,}
- **Features**: {n_features}
- **Target classes**: {len(target_classes)} ({', '.join(str(c) for c in target_classes)})
""")

st.markdown("---")

# Step 1: Model Selection
st.subheader("1. Select Models to Train")

available_models = get_available_models()

col1, col2 = st.columns([3, 1])

with col1:
    selected_models = st.multiselect(
        "Select models to train",
        options=available_models,
        default=available_models,
        help="Choose which classification algorithms to train"
    )

with col2:
    if st.button("Select All"):
        selected_models = available_models
        st.rerun()

# Show model descriptions
if selected_models:
    with st.expander("View model descriptions"):
        for model_name in selected_models:
            st.markdown(f"**{model_name}**: {get_model_description(model_name)}")

st.markdown("---")

# Step 2: Training Mode
st.subheader("2. Choose Training Mode")

col1, col2 = st.columns(2)

with col1:
    training_mode = st.radio(
        "Training mode",
        options=["Quick Training", "With Hyperparameter Tuning"],
        help="Quick training uses default parameters. Tuning finds optimal parameters using cross-validation."
    )

with col2:
    if training_mode == "With Hyperparameter Tuning":
        st.info("""
        Hyperparameter Tuning will:
        - Use 5-fold stratified cross-validation
        - Search for optimal parameters
        - Take longer but may improve performance
        """)
        
        tuning_method = st.selectbox(
            "Tuning method",
            options=["Random Search", "Grid Search"],
            help="Random Search is faster; Grid Search is exhaustive"
        )
        
        if tuning_method == "Random Search":
            n_iter = st.slider("Iterations per model", 5, 20, 10)
        else:
            n_iter = None
            st.warning("Grid Search may take longer on complex models.")

st.markdown("---")

# Step 3: Train Models
st.subheader("3. Train Models")

if not selected_models:
    st.warning("Please select at least one model to train.")
    st.stop()

# Training button
if st.button("Start Training", type="primary", use_container_width=True):
    trained_models = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(model_name, i, total):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Training {model_name}... ({i + 1}/{total})")
    
    try:
        if training_mode == "Quick Training":
            # Train with default parameters
            with st.spinner("Training models with default parameters..."):
                trained_models = train_all_models(
                    X_train, y_train,
                    models_to_train=selected_models,
                    progress_callback=progress_callback
                )
        else:
            # Train with hyperparameter tuning
            search_method = "random" if tuning_method == "Random Search" else "grid"
            
            with st.spinner("Training models with hyperparameter tuning..."):
                trained_models = tune_all_models(
                    X_train, y_train,
                    models_to_tune=selected_models,
                    search_method=search_method,
                    n_iter=n_iter if n_iter else 10,
                    progress_callback=progress_callback
                )
        
        progress_bar.progress(1.0)
        status_text.text("Evaluating models...")
        
        # Evaluate all models
        evaluation_results = evaluate_all_models(
            trained_models, X_test, y_test,
            class_names=target_classes
        )
        
        # Store results in session state
        set_state("trained_models", trained_models)
        set_state("evaluation_results", evaluation_results)
        
        # Log decision
        log_user_decision(
            decision_type="training",
            description=f"Trained {len(trained_models)} models",
            details={
                "models": list(trained_models.keys()),
                "mode": training_mode,
                "tuning_method": tuning_method if training_mode == "With Hyperparameter Tuning" else "N/A"
            }
        )
        
        mark_step_complete("training")
        set_state("training_complete", True)
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"Successfully trained {len(trained_models)} models!")
        
        # Show quick results
        st.markdown("### Quick Results")
        
        comparison_df = create_comparison_dataframe(evaluation_results)
        
        # Highlight best model
        best_name, best_result = get_best_model(evaluation_results, metric='f1')
        st.info(f"Best Model: {best_name} (F1-Score: {best_result.f1:.4f})")
        
        # Display comparison table
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Training time summary
        total_time = sum(m.training_time for m in trained_models.values())
        st.markdown(f"**Total training time**: {total_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"**Problem**: Training failed - {str(e)}. **Solution**: Check your data and try again with fewer models.")
        raise e

# Show existing results if available
if get_state("training_complete"):
    st.markdown("---")
    
    trained_models = get_state("trained_models")
    evaluation_results = get_state("evaluation_results")
    
    if trained_models and evaluation_results:
        st.subheader("Training Results")
        
        # Comparison table
        comparison_df = create_comparison_dataframe(evaluation_results)
        
        # Best model highlight
        best_name, best_result = get_best_model(evaluation_results, metric='f1')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", best_name)
        with col2:
            st.metric("F1-Score", f"{best_result.f1:.4f}")
        with col3:
            st.metric("Accuracy", f"{best_result.accuracy:.4f}")
        
        # Full comparison table
        st.markdown("#### Model Comparison (sorted by F1-Score)")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                color='#1f77b4; color: white'
            ).highlight_min(
                subset=['Training Time (s)'],
                color='#17a2b8; color: white'
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Model details expander
        st.markdown("#### Model Details")
        
        for model_name in trained_models.keys():
            trained_model = trained_models[model_name]
            eval_result = evaluation_results.get(model_name)
            
            with st.expander(f"{model_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Parameters:**")
                    st.json(trained_model.params)
                    
                    if trained_model.is_tuned:
                        st.success("Hyperparameters tuned")
                        if trained_model.cv_score:
                            st.markdown(f"**CV Score**: {trained_model.cv_score:.4f}")
                
                with col2:
                    if eval_result:
                        st.markdown("**Classification Report:**")
                        st.code(eval_result.classification_report)
        
        st.markdown("---")
        st.success("""
        Training Complete!
        
        Next Step: Go to Compare & Report in the sidebar to:
        - View detailed comparison visualizations
        - Generate and download reports
        - Save the best model
        """)

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### Training Info")
    st.markdown("""
    **7 Classifiers Available:**
    1. Logistic Regression
    2. K-Nearest Neighbors
    3. Decision Tree
    4. Naive Bayes
    5. Random Forest
    6. Support Vector Machine
    7. Rule-Based
    
    **Primary Metric:** F1-Score (weighted)
    """)
