"""
Prediction Page

Allows users to make predictions on new data using trained models.
Implements FR-62 to FR-67 from the requirements.

This page allows users to:
1. Select a trained model
2. Input new data (manual entry or CSV upload)
3. Get predictions with confidence scores
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.preprocessing import apply_preprocessing_to_new_data
from ml.data_loader import load_csv
from utils.session_manager import get_state, set_state, log_user_decision
from utils.validators import validate_file_extension, validate_csv_content

# Page configuration
st.set_page_config(
    page_title="Prediction - AutoML",
    layout="wide",
)

st.title("Make Predictions")

# Check prerequisites
trained_models = get_state("trained_models")
preprocessing_result = get_state("preprocessing_result")

if not trained_models:
    st.warning("No trained models found. Please complete the training workflow first.")
    st.stop()

if not preprocessing_result:
    st.warning("Preprocessing pipeline not found. Please complete the preprocessing step first.")
    st.stop()

# Get model info
target_classes = get_state("target_classes")
feature_columns = preprocessing_result.config.feature_columns
label_encoder = get_state("label_encoder")

st.markdown(f"""
Make predictions using your trained classification models.

**Available Models**: {len(trained_models)}  
**Required Features**: {len(feature_columns)}  
**Target Classes**: {', '.join(str(c) for c in target_classes)}
""")

st.markdown("---")

# Model selection
st.subheader("1. Select Model")

from ml.evaluation import get_best_model
evaluation_results = get_state("evaluation_results")

if evaluation_results:
    best_name, _ = get_best_model(evaluation_results, metric='f1')
    default_index = list(trained_models.keys()).index(best_name)
else:
    default_index = 0

selected_model = st.selectbox(
    "Choose a model for prediction",
    options=list(trained_models.keys()),
    index=default_index,
    help="Select which trained model to use for predictions"
)

# Show model metrics
if evaluation_results and selected_model in evaluation_results:
    result = evaluation_results[selected_model]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{result.accuracy:.2%}")
    with col2:
        st.metric("F1-Score", f"{result.f1:.2%}")
    with col3:
        st.metric("Precision", f"{result.precision:.2%}")
    with col4:
        st.metric("Recall", f"{result.recall:.2%}")

st.markdown("---")

# Input method tabs
st.subheader("2. Input Data")

input_method = st.radio(
    "Choose input method",
    options=["Manual Entry", "Upload CSV"],
    horizontal=True
)

# Initialize prediction_df from session state if available
prediction_df = get_state("prediction_input_df")

if input_method == "Manual Entry":
    st.markdown("Enter values for each feature:")
    
    # Get original dataframe for reference
    original_df = get_state("uploaded_df")
    
    input_data = {}
    
    # Create input fields in columns
    n_cols = 3
    cols = st.columns(n_cols)
    
    for i, col_name in enumerate(feature_columns):
        with cols[i % n_cols]:
            if original_df is not None and col_name in original_df.columns:
                col_data = original_df[col_name]
                
                if pd.api.types.is_numeric_dtype(col_data):
                    # Numerical input
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    
                    input_data[col_name] = st.number_input(
                        col_name,
                        min_value=min_val - (max_val - min_val) * 0.5,
                        max_value=max_val + (max_val - min_val) * 0.5,
                        value=mean_val,
                        help=f"Range in training data: {min_val:.2f} - {max_val:.2f}"
                    )
                else:
                    # Categorical input
                    unique_values = col_data.dropna().unique().tolist()
                    input_data[col_name] = st.selectbox(
                        col_name,
                        options=unique_values,
                        help=f"Categories: {len(unique_values)} unique values"
                    )
            else:
                # Fallback if column not found
                input_data[col_name] = st.text_input(col_name)
    
    if st.button("Create Input Row"):
        prediction_df = pd.DataFrame([input_data])
        set_state("prediction_input_df", prediction_df)
        st.success("Input row created!")
        st.rerun()

else:  # Upload CSV
    st.markdown(f"""
    Upload a CSV file with the following columns:
    
    ```
    {', '.join(feature_columns)}
    ```
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV for prediction",
        type=['csv'],
        help="CSV file with the same feature columns as training data"
    )
    
    if uploaded_file is not None:
        # Validate file extension
        is_valid, error = validate_file_extension(uploaded_file.name)
        if not is_valid:
            st.error(error)
        else:
            try:
                # Load CSV using our data loader
                loaded_df, metadata = load_csv(uploaded_file)
                
                if loaded_df is None:
                    st.error(metadata.get('error', 'Failed to load CSV file'))
                else:
                    # Validate content
                    is_valid, error = validate_csv_content(loaded_df)
                    if not is_valid:
                        st.error(error)
                    else:
                        # Check for required columns
                        missing_cols = [col for col in feature_columns if col not in loaded_df.columns]
                        
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            prediction_df = loaded_df
                            set_state("prediction_input_df", prediction_df)
                            st.success(f"Loaded {len(prediction_df)} rows for prediction")
                            
                            with st.expander("Preview uploaded data"):
                                st.dataframe(prediction_df[feature_columns].head(10))
            
            except Exception as e:
                st.error(f"**Problem**: Could not load file - {str(e)}")

st.markdown("---")

# Make predictions
st.subheader("3. Get Predictions")

# Get prediction_df from session state again (in case it was updated)
prediction_df = get_state("prediction_input_df")

if prediction_df is not None and len(prediction_df) > 0:
    st.markdown("**Input Data Preview:**")
    st.dataframe(prediction_df[feature_columns].head(10), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        predict_button = st.button("Predict", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Input", use_container_width=True):
            set_state("prediction_input_df", None)
            set_state("prediction_results", None)
            st.rerun()
    
    if predict_button:
        with st.spinner("Making predictions..."):
            try:
                # Get the model
                model = trained_models[selected_model].model
                
                # Preprocess the input data
                X_new = apply_preprocessing_to_new_data(prediction_df, preprocessing_result)
                
                # Make predictions
                predictions = model.predict(X_new)
                
                # Decode labels
                predicted_classes = label_encoder.inverse_transform(predictions)
                
                # Get probabilities if available
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X_new)
                    except:
                        pass
                
                # Create results dataframe
                results_df = prediction_df[feature_columns].copy()
                results_df['Predicted Class'] = predicted_classes
                
                if probabilities is not None:
                    # Add confidence (max probability)
                    results_df['Confidence'] = np.max(probabilities, axis=1)
                    
                    # Add probability for each class
                    for i, class_name in enumerate(target_classes):
                        results_df[f'P({class_name})'] = probabilities[:, i]
                
                # Store results
                set_state("prediction_results", results_df)
                
                log_user_decision(
                    decision_type="prediction",
                    description=f"Made predictions using {selected_model}",
                    details={"n_samples": len(prediction_df), "model": selected_model}
                )
                
                st.success(f"Predictions complete for {len(prediction_df)} samples!")
                
            except Exception as e:
                st.error(f"**Problem**: Prediction failed - {str(e)}. **Solution**: Ensure your input data matches the training data format.")

# Display results
prediction_results = get_state("prediction_results")

if prediction_results is not None:
    st.markdown("### Prediction Results")
    
    # Summary statistics
    class_counts = prediction_results['Predicted Class'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Class Distribution:**")
        for cls, count in class_counts.items():
            pct = count / len(prediction_results) * 100
            st.markdown(f"- **{cls}**: {count} ({pct:.1f}%)")
    
    with col2:
        if 'Confidence' in prediction_results.columns:
            avg_conf = prediction_results['Confidence'].mean()
            min_conf = prediction_results['Confidence'].min()
            max_conf = prediction_results['Confidence'].max()
            
            st.markdown("**Confidence Statistics:**")
            st.markdown(f"- **Average**: {avg_conf:.2%}")
            st.markdown(f"- **Min**: {min_conf:.2%}")
            st.markdown(f"- **Max**: {max_conf:.2%}")
    
    st.markdown("---")
    
    # Results table
    st.markdown("### Full Results")
    
    # Highlight prediction column
    st.dataframe(
        prediction_results.style.background_gradient(
            subset=['Confidence'] if 'Confidence' in prediction_results.columns else [],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Download results
    csv_results = prediction_results.to_csv(index=False)
    
    st.download_button(
        label="Download Predictions (CSV)",
        data=csv_results,
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### Prediction Info")
    
    st.markdown(f"""
    **Selected Model**: {selected_model}
    
    **Required Features ({len(feature_columns)}):**
    """)
    
    with st.expander("View feature list"):
        for col in feature_columns:
            st.markdown(f"- `{col}`")
    
    st.markdown(f"""
    **Target Classes ({len(target_classes)}):**
    """)
    for cls in target_classes:
        st.markdown(f"- {cls}")
