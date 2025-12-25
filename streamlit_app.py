"""
AutoML Classification System - Main Entry Point

A Streamlit-based AutoML web application for end-to-end machine learning
workflows including data upload, EDA, preprocessing, model training,
comparison, and deployment.

Author: CS-245 Machine Learning Project Team
Version: 1.0
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="AutoML Classification System",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        # Data storage
        "uploaded_df": None,
        "uploaded_filename": None,
        
        # EDA results (cached)
        "eda_results": None,
        "detected_issues": None,
        
        # User decisions for report
        "user_decisions": [],
        
        # Preprocessing configuration
        "target_column": None,
        "feature_columns": None,
        "preprocessing_config": None,
        
        # Train/test splits
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "preprocessing_pipeline": None,
        
        # Model training results
        "trained_models": {},
        "evaluation_results": {},
        "best_model_name": None,
        
        # Workflow state tracking
        "current_step": 1,
        "upload_complete": False,
        "eda_complete": False,
        "preprocessing_complete": False,
        "training_complete": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation info
    st.sidebar.title("AutoML System")
    st.sidebar.markdown("---")
    
    # Workflow progress indicator
    st.sidebar.markdown("### Workflow Progress")
    
    steps = [
        ("1. Upload Data", st.session_state.upload_complete),
        ("2. EDA & Issues", st.session_state.eda_complete),
        ("3. Preprocessing", st.session_state.preprocessing_complete),
        ("4. Train Models", st.session_state.training_complete),
        ("5. Compare & Report", st.session_state.training_complete),
        ("6. Prediction", st.session_state.get("best_model_name") is not None),
    ]
    
    for step_name, completed in steps:
        if completed:
            st.sidebar.markdown(f"✅ {step_name}")
        else:
            st.sidebar.checkbox(step_name, value=False, disabled=True)
    
    st.sidebar.markdown("---")
    
    # Reset button
    if st.sidebar.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Main content - Landing page
    st.title("AutoML Classification System")
    st.markdown("""
    Welcome to the **AutoML Classification System** - an end-to-end machine learning 
    platform that automates the complete ML workflow for classification tasks.
    
    ### Getting Started
    
    Use the **sidebar navigation** to access different pages of the workflow:
    
    1. **Upload & Info** - Upload your CSV dataset and view metadata
    2. **EDA & Issues** - Explore data and detect quality issues  
    3. **Preprocess & Split** - Configure preprocessing and train/test split
    4. **Train & Tune** - Train 7 classifiers with hyperparameter optimization
    5. **Compare & Report** - Compare models and generate reports
    6. **Prediction** - Make real-time predictions with the best model
    
    ### Key Features
    
    - **CSV Upload** with automatic encoding detection
    - **Automated EDA** with visualizations
    - **Data Quality Detection** (missing values, outliers, imbalance)
    - **User-Controlled Preprocessing** (you approve all fixes)
    - **7 Classification Algorithms** trained automatically
    - **Comprehensive Comparison** with multiple metrics
    - **Detailed Reports** in HTML/Markdown format
    - **Real-Time Prediction** with trained models
    
    ---
    
    Select a page from the sidebar to begin!
    """)
    
    # Quick stats if data is loaded
    if st.session_state.uploaded_df is not None:
        st.markdown("---")
        st.markdown("### Current Session")
        
        col1, col2, col3, col4 = st.columns(4)
        
        df = st.session_state.uploaded_df
        with col1:
            st.metric("Dataset", st.session_state.uploaded_filename or "Loaded")
        with col2:
            st.metric("Rows", f"{len(df):,}")
        with col3:
            st.metric("Columns", f"{len(df.columns):,}")
        with col4:
            if st.session_state.target_column:
                st.metric("Target", st.session_state.target_column)
            else:
                st.metric("Target", "Not selected")


if __name__ == "__main__":
    main()
