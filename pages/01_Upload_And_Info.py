"""
Upload and Info Page

Handles dataset upload, validation, and metadata display.
Implements FR-1 to FR-9 from the requirements.

This is the first page in the AutoML workflow where users:
1. Upload their CSV dataset
2. View basic metadata (rows, columns, types)
3. See summary statistics
4. Preview the data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_loader import (
    load_csv,
    get_column_info,
    get_summary_statistics,
    get_categorical_summary,
    infer_column_types,
)
from utils.session_manager import (
    set_dataframe,
    get_dataframe,
    mark_step_complete,
    log_user_decision,
    set_state,
    get_state,
)
from utils.validators import validate_file_extension

# Page configuration
st.set_page_config(
    page_title="Upload & Info - AutoML",
    layout="wide",
)

st.title("Upload Dataset & View Information")

st.markdown("""
Upload your CSV dataset to begin the AutoML workflow. The system will automatically:
- Detect file encoding (UTF-8, Latin-1, etc.)
- Validate the file format
- Extract metadata and summary statistics
""")

# File upload section
st.markdown("---")
st.subheader("Upload CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Maximum file size: 200 MB. Only CSV format is supported.",
    key="csv_uploader"
)

if uploaded_file is not None:
    # Validate file extension
    is_valid, error_msg = validate_file_extension(uploaded_file.name)
    
    if not is_valid:
        st.error(error_msg)
    else:
        # Load the CSV file
        with st.spinner("Loading dataset..."):
            df, metadata = load_csv(uploaded_file)
        
        if not metadata['success']:
            st.error(metadata['error'])
        else:
            # Store in session state
            set_dataframe(df, metadata['filename'])
            set_state("file_metadata", metadata)
            mark_step_complete("upload")
            
            # Log the upload decision
            log_user_decision(
                decision_type="upload",
                description=f"Uploaded dataset: {metadata['filename']}",
                details=metadata
            )
            
            # Success message
            st.success(f"""
             **Dataset uploaded successfully!**
            - **File**: {metadata['filename']}
            - **Size**: {metadata['file_size_mb']} MB
            - **Encoding**: {metadata['encoding']}
            - **Rows**: {metadata['rows']:,}
            - **Columns**: {metadata['columns']}
            """)

# Display data info if loaded
df = get_dataframe()

if df is not None:
    st.markdown("---")
    
    # Dataset overview metrics
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns)}")
    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        st.metric("Categorical Columns", categorical_cols)
    with col5:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    
    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")
    
    preview_rows = st.slider("Number of rows to display", 5, 50, 10)
    
    tab1, tab2 = st.tabs(["First Rows", "Last Rows"])
    
    with tab1:
        st.dataframe(df.head(preview_rows), use_container_width=True)
    
    with tab2:
        st.dataframe(df.tail(preview_rows), use_container_width=True)
    
    # Column information
    st.markdown("---")
    st.subheader("Column Information")
    
    column_info = get_column_info(df)
    st.dataframe(column_info, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("Summary Statistics (Numerical Columns)")
    
    summary_stats = get_summary_statistics(df)
    
    if not summary_stats.empty:
        st.dataframe(summary_stats, use_container_width=True)
    else:
        st.info("No numerical columns found in the dataset.")
    
    # Categorical summaries
    st.markdown("---")
    st.subheader("Categorical Column Summary")
    
    categorical_summaries = get_categorical_summary(df)
    
    if categorical_summaries:
        # Let user select which categorical column to view
        selected_cat_col = st.selectbox(
            "Select categorical column to view",
            options=list(categorical_summaries.keys())
        )
        
        if selected_cat_col:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown(f"**Value counts for `{selected_cat_col}`:**")
                counts = categorical_summaries[selected_cat_col]
                counts_df = pd.DataFrame({
                    'Value': counts.index,
                    'Count': counts.values,
                    'Percentage': (counts.values / len(df) * 100).round(2)
                })
                st.dataframe(counts_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Distribution:**")
                st.bar_chart(counts)
    else:
        st.info("No categorical columns found in the dataset.")
    
    # Inferred column types
    st.markdown("---")
    st.subheader("Inferred Column Types")
    
    inferred_types = infer_column_types(df)
    
    type_df = pd.DataFrame([
        {'Column': col, 'Pandas Type': str(df[col].dtype), 'Inferred Type': inferred_types.get(col, 'unknown')}
        for col in df.columns
    ])
    
    st.dataframe(type_df, use_container_width=True, hide_index=True)
    
    # Help text
    with st.expander("Understanding Inferred Types"):
        st.markdown("""
        | Type | Description |
        |------|-------------|
        | `numeric_continuous` | Continuous numerical data (e.g., price, temperature) |
        | `numeric_discrete` | Integer-like data with few unique values (e.g., rating 1-5) |
        | `categorical` | Text or category data |
        | `binary` | Columns with exactly 2 unique values |
        | `datetime` | Date/time columns |
        | `id` | Likely identifier columns (high uniqueness) |
        """)
    
    # Next step guidance
    st.markdown("---")
    st.success("""
    **Data loaded successfully!** 
    
    **Next Step**: Go to **EDA & Issues** in the sidebar to explore your data 
    and detect quality issues.
    """)
    
else:
    # No data loaded - show instructions
    st.info("""
    **No dataset loaded yet.**
    
    Please upload a CSV file above to begin the AutoML workflow.
    
    **Supported Features:**
    - CSV files up to 200 MB
    - Automatic encoding detection (UTF-8, Latin-1)
    - Both binary and multiclass classification
    """)
    

