"""
Compare, Report & Deploy Page

Provides detailed model comparison, report generation, and model saving.
Implements FR-46 to FR-61 from the requirements.

This page allows users to:
1. View detailed comparison visualizations
2. Generate and download reports (HTML/Markdown)
3. Save the best model for deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.evaluation import (
    create_comparison_dataframe,
    get_best_model,
    format_metrics_for_display,
    get_per_class_metrics
)
from ml.report_generator import (
    generate_html_report,
    generate_markdown_report,
    save_report_to_file
)
from utils.visualizations import (
    create_confusion_matrix,
    create_roc_curves,
    create_metrics_comparison_bar
)
from utils.session_manager import (
    get_state,
    set_state,
    mark_step_complete,
    log_user_decision,
    get_train_test_data,
)

# Page configuration
st.set_page_config(
    page_title="Compare & Report - AutoML",
    layout="wide",
)

st.title("Model Comparison & Reports")

# Check prerequisites
trained_models = get_state("trained_models")
evaluation_results = get_state("evaluation_results")

if not trained_models or not evaluation_results:
    st.warning("No trained models found. Please go to Train & Tune first.")
    st.stop()

# Get additional data
target_classes = get_state("target_classes")
X_train, X_test, y_train, y_test = get_train_test_data()
preprocessing_config = get_state("preprocessing_config")

st.markdown(f"Comparing **{len(trained_models)}** trained classification models.")
st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance Comparison",
    "Detailed Analysis",
    "Generate Report",
    "Save Model"
])

# Tab 1: Performance Comparison
with tab1:
    st.subheader("Model Performance Comparison")
    
    # Get best model
    best_name, best_result = get_best_model(evaluation_results, metric='f1')
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_name)
    with col2:
        st.metric("F1-Score", f"{best_result.f1:.4f}")
    with col3:
        st.metric("Accuracy", f"{best_result.accuracy:.4f}")
    with col4:
        if best_result.roc_auc:
            st.metric("ROC-AUC", f"{best_result.roc_auc:.4f}")
        else:
            st.metric("Precision", f"{best_result.precision:.4f}")
    
    st.markdown("---")
    
    # Comparison table
    st.markdown("### Comparison Table (sorted by F1-Score)")
    comparison_df = create_comparison_dataframe(evaluation_results)
    
    # Style the dataframe with better contrast
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
    
    # Metrics comparison bar chart
    st.markdown("### Metrics Comparison Chart")
    metrics_data = {
        k: {
            'Accuracy': v.accuracy,
            'Precision': v.precision,
            'Recall': v.recall,
            'F1-Score': v.f1
        }
        for k, v in evaluation_results.items()
    }
    fig = create_metrics_comparison_bar(metrics_data)
    st.pyplot(fig)

# Tab 2: Detailed Analysis
with tab2:
    st.subheader("Detailed Model Analysis")
    
    # Model selector
    selected_model = st.selectbox(
        "Select a model to analyze",
        options=list(evaluation_results.keys()),
        index=list(evaluation_results.keys()).index(best_name)
    )
    
    result = evaluation_results[selected_model]
    model_obj = trained_models[selected_model]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {selected_model}")
        
        # Metrics
        metrics = format_metrics_for_display(result)
        for metric, value in metrics.items():
            st.markdown(f"- **{metric}**: {value}")
        
        # Model parameters
        st.markdown("### Parameters")
        st.json(model_obj.params)
        
        if model_obj.is_tuned:
            st.success("Hyperparameters tuned")
    
    with col2:
        # Confusion matrix
        st.markdown("### Confusion Matrix")
        fig = create_confusion_matrix(
            result.confusion_matrix,
            class_names=target_classes,
            title=selected_model
        )
        st.pyplot(fig)
    
    # Per-class metrics
    st.markdown("### Per-Class Metrics")
    y_pred = model_obj.model.predict(X_test)
    per_class_df = get_per_class_metrics(y_test, y_pred, class_names=target_classes)
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)
    
    # ROC Curves (binary only)
    binary_results = {k: v for k, v in evaluation_results.items() if v.is_binary and v.roc_data}
    
    if binary_results:
        st.markdown("### ROC Curves (All Models)")
        roc_data = {k: v.roc_data for k, v in binary_results.items()}
        fig = create_roc_curves(roc_data)
        st.pyplot(fig)
    
    # Classification Report
    st.markdown("### Classification Report")
    st.code(result.classification_report)

# Tab 3: Generate Report
with tab3:
    st.subheader("Generate Report")
    
    st.markdown("""
    Generate a comprehensive report including:
    - Dataset overview
    - EDA summary
    - Detected issues
    - Preprocessing decisions
    - Model comparison
    - Best model details
    - Visualizations
    """)
    
    # Collect report data
    df = get_state("uploaded_df")
    eda_results = get_state("eda_results")
    detected_issues = get_state("detected_issues") or []
    print('detected_issues in report tab',detected_issues)
    user_decisions = get_state("user_decisions") or []
    
    dataset_info = {
        "filename": get_state("uploaded_filename") or "Unknown",
        "rows": len(df) if df is not None else 0,
        "columns": len(df.columns) if df is not None else 0,
        "target": preprocessing_config.target_column if preprocessing_config else "Unknown"
    }
    
    eda_summary = {
        "missing_values": eda_results.missing_value_analysis['missing_count'].sum() if eda_results  else 0,
        "outliers": len(eda_results.outlier_analysis_iqr) if eda_results and eda_results.outlier_analysis_iqr is not None else 0,
        "high_correlations": len(eda_results.correlation_matrix) if eda_results else 0
    }
    
    # Convert issues to dict format
    issues_dicts = []
    if detected_issues:
        for issue in detected_issues.issues:
            issues_dicts.append({
                "type": issue.issue_type,
                "column": issue.affected_columns,
                "severity": issue.severity.value,
                "description": issue.description
            })
    
    preprocessing_decisions = {}
    if preprocessing_config:
        from ml.preprocessing import get_preprocessing_summary
        preprocessing_decisions = get_preprocessing_summary(preprocessing_config)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### HTML Report")
        st.markdown("Self-contained report with embedded visualizations.")
        
        if st.button("Generate HTML Report", key="gen_html"):
            with st.spinner("Generating HTML report..."):
                html_report = generate_html_report(
                    dataset_info=dataset_info,
                    eda_summary=eda_summary,
                    detected_issues=issues_dicts,
                    preprocessing_decisions=preprocessing_decisions,
                    trained_models=trained_models,
                    evaluation_results=evaluation_results,
                    target_classes=target_classes,
                    best_model_name=best_name,
                    include_visualizations=True
                )
                
                set_state("html_report", html_report)
                st.success("HTML report generated")
        
        if get_state("html_report"):
            st.download_button(
                label="Download HTML Report",
                data=get_state("html_report"),
                file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
    
    with col2:
        st.markdown("### Markdown Report")
        st.markdown("Plain text report for documentation.")
        
        if st.button("Generate Markdown Report", key="gen_md"):
            with st.spinner("Generating Markdown report..."):
                md_report = generate_markdown_report(
                    dataset_info=dataset_info,
                    eda_summary=eda_summary,
                    detected_issues=issues_dicts,
                    preprocessing_decisions=preprocessing_decisions,
                    trained_models=trained_models,
                    evaluation_results=evaluation_results,
                    target_classes=target_classes,
                    best_model_name=best_name
                )
                
                set_state("md_report", md_report)
                st.success("Markdown report generated")
        
        if get_state("md_report"):
            st.download_button(
                label="Download Markdown Report",
                data=get_state("md_report"),
                file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

# Tab 4: Save Model
with tab4:
    st.subheader("Save Model for Deployment")
    
    st.markdown("""
    Save the trained model and preprocessing pipeline for future predictions.
    
    The saved bundle includes:
    - Trained model
    - Preprocessing pipeline
    - Label encoder
    - Feature names
    - Model metadata
    """)
    
    # Model selector
    model_to_save = st.selectbox(
        "Select model to save",
        options=list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_name),
        key="save_model_select"
    )
    
    # Show model info
    selected_result = evaluation_results[model_to_save]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{selected_result.accuracy:.4f}")
    with col2:
        st.metric("F1-Score", f"{selected_result.f1:.4f}")
    with col3:
        st.metric("Precision", f"{selected_result.precision:.4f}")
    with col4:
        st.metric("Recall", f"{selected_result.recall:.4f}")
    
    st.markdown("---")
    
    if st.button("Prepare Model for Download", type="primary"):
        with st.spinner("Preparing model bundle..."):
            try:
                # Create model bundle
                preprocessing_result = get_state("preprocessing_result")
                
                model_bundle = {
                    "model": trained_models[model_to_save].model,
                    "preprocessing_pipeline": preprocessing_result.pipeline if preprocessing_result else None,
                    "label_encoder": get_state("label_encoder"),
                    "feature_names": get_state("feature_names"),
                    "target_classes": target_classes,
                    "preprocessing_config": preprocessing_config,
                    "model_name": model_to_save,
                    "metrics": {
                        "accuracy": selected_result.accuracy,
                        "precision": selected_result.precision,
                        "recall": selected_result.recall,
                        "f1": selected_result.f1,
                        "roc_auc": selected_result.roc_auc
                    },
                    "training_time": trained_models[model_to_save].training_time,
                    "is_tuned": trained_models[model_to_save].is_tuned,
                    "params": trained_models[model_to_save].params,
                    "created_at": datetime.now().isoformat()
                }
                
                # Serialize to bytes
                import io
                buffer = io.BytesIO()
                joblib.dump(model_bundle, buffer)
                model_bytes = buffer.getvalue()
                
                set_state("model_bundle_bytes", model_bytes)
                set_state("saved_model_name", model_to_save)
                
                mark_step_complete("model_saved")
                
                log_user_decision(
                    decision_type="model_save",
                    description=f"Saved model: {model_to_save}",
                    details={"model": model_to_save, "f1": selected_result.f1}
                )
                
                st.success(f"Model '{model_to_save}' prepared for download")
                
            except Exception as e:
                st.error(f"**Problem**: Could not prepare model - {str(e)}")
    
    if get_state("model_bundle_bytes"):
        saved_name = get_state("saved_model_name")
        st.download_button(
            label=f"Download {saved_name} Model Bundle (.joblib)",
            data=get_state("model_bundle_bytes"),
            file_name=f"automl_model_{saved_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.joblib",
            mime="application/octet-stream"
        )
        
        st.info("""
        To use the saved model:
        ```python
        import joblib
        bundle = joblib.load('your_model.joblib')
        
        # Preprocess new data
        X_new = bundle['preprocessing_pipeline'].transform(new_df)
        
        # Make predictions
        predictions = bundle['model'].predict(X_new)
        
        # Decode labels
        class_names = bundle['label_encoder'].inverse_transform(predictions)
        ```
        """)

st.markdown("---")
st.success("""
Comparison and Reporting Complete!

Next Step: Go to Prediction in the sidebar to make predictions on new data.
""")
