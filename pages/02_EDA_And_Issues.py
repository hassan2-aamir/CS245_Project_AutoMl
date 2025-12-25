"""
EDA and Issues Page

Provides automated exploratory data analysis and data quality issue detection.
Implements FR-10 to FR-28 from the requirements.

This page allows users to:
1. View EDA visualizations (distributions, correlations)
2. See detected data quality issues
3. Approve fixes for identified problems
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.eda import (
    run_full_eda,
    get_train_test_preview,
)
from ml.issue_detector import (
    run_issue_detection,
    get_issue_summary,
    IssueSeverity,
)
from utils.session_manager import (
    get_dataframe,
    get_state,
    set_state,
    mark_step_complete,
    log_user_decision,
)
from utils.visualizations import (
    create_histogram,
    create_bar_plot,
    create_correlation_heatmap,
    create_missing_values_plot,
    create_class_distribution_plot,
    close_all_figures,
)

# Page configuration
st.set_page_config(
    layout="wide",
)

st.title("Exploratory Data Analysis & Issue Detection")


def display_severity_badge(severity: IssueSeverity) -> str:
    """Return HTML badge for severity level."""
    colors = {
        IssueSeverity.CRITICAL: "#dc3545",
        IssueSeverity.HIGH: "#fd7e14",
        IssueSeverity.MEDIUM: "#ffc107",
        IssueSeverity.LOW: "#28a745",
    }
    return f'<span style="background-color: {colors[severity]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{severity.value}</span>'


# Check if data is loaded
df = get_dataframe()

if df is None:
    st.warning("No dataset loaded. Please go to **Upload & Info** first.")
    st.stop()

st.markdown(f"""
Analyzing dataset with **{len(df):,} rows** and **{len(df.columns)} columns**.
""")

# Run EDA (with caching in session state)
@st.cache_data(show_spinner=False)
def cached_eda(df_hash):
    """Cache EDA results based on dataframe hash."""
    df = get_dataframe()
    return run_full_eda(df)

# Use a simple hash of the dataframe
df_hash = hash(tuple(pd.util.hash_pandas_object(df)))

with st.spinner("Running exploratory data analysis..."):
    eda_results = cached_eda(df_hash)
    set_state("eda_results", eda_results)

# EDA Tabs
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Missing Values",
    "Distributions",
    "Correlations",
    "Issues",
    "Fix Issues"
])

# Tab 1: Missing Values
with tab1:
    st.subheader("Missing Value Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Global Missing %", f"{eda_results.global_missing_pct}%")
        
        # Summary
        missing_cols = len(eda_results.missing_value_analysis[
            eda_results.missing_value_analysis['missing_count'] > 0
        ])
        st.metric("Columns with Missing", missing_cols)
    
    with col2:
        # Missing values table
        missing_df = eda_results.missing_value_analysis[
            eda_results.missing_value_analysis['missing_count'] > 0
        ].copy()
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values found in the dataset.")
    
    # Missing values plot
    if len(missing_df) > 0:
        fig = create_missing_values_plot(eda_results.missing_value_analysis)
        st.pyplot(fig)
        close_all_figures()

# Tab 2: Distributions
with tab2:
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Features**")
        if eda_results.numerical_columns:
            selected_num_col = st.selectbox(
                "Select numerical column",
                options=eda_results.numerical_columns,
                key="num_col_select"
            )
            
            if selected_num_col:
                fig = create_histogram(
                    df[selected_num_col].dropna(),
                    title=f"Distribution of {selected_num_col}",
                    xlabel=selected_num_col
                )
                st.pyplot(fig)
                close_all_figures()
                
                # Show statistics
                stats = eda_results.column_statistics.get(selected_num_col, {})
                if stats:
                    st.markdown("**Statistics:**")
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Mean", f"{stats.get('mean', 0):.2f}")
                    with stat_col2:
                        st.metric("Median", f"{stats.get('median', 0):.2f}")
                    with stat_col3:
                        st.metric("Std", f"{stats.get('std', 0):.2f}")
                    with stat_col4:
                        st.metric("Skewness", f"{stats.get('skewness', 0):.2f}")
        else:
            st.info("No numerical columns found.")
    
    with col2:
        st.markdown("**Categorical Features**")
        if eda_results.categorical_columns:
            selected_cat_col = st.selectbox(
                "Select categorical column",
                options=eda_results.categorical_columns,
                key="cat_col_select"
            )
            
            if selected_cat_col:
                fig = create_bar_plot(
                    df[selected_cat_col].dropna(),
                    title=f"Distribution of {selected_cat_col}",
                    xlabel=selected_cat_col
                )
                st.pyplot(fig)
                close_all_figures()
                
                # Show value counts
                st.markdown("**Top Values:**")
                value_counts = df[selected_cat_col].value_counts().head(10)
                st.dataframe(
                    pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        '%': (value_counts.values / len(df) * 100).round(2)
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No categorical columns found.")

# Tab 3: Correlations
with tab3:
    st.subheader("Correlation Analysis")
    
    if not eda_results.correlation_matrix.empty:
        # Correlation heatmap
        fig = create_correlation_heatmap(
            df,
            title="Pearson Correlation Matrix"
        )
        st.pyplot(fig)
        close_all_figures()
        
        # High correlations
        from ml.eda import get_highly_correlated_pairs
        high_corr_pairs = get_highly_correlated_pairs(eda_results.correlation_matrix, threshold=0.7)
        
        if high_corr_pairs:
            st.markdown("**Highly Correlated Pairs (|r| > 0.7):**")
            corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
        else:
            st.info("No highly correlated feature pairs found (threshold: 0.7).")
    else:
        st.info("Not enough numerical columns for correlation analysis.")

# Tab 4: Issue Detection
with tab4:
    st.subheader("Detected Data Quality Issues")
    
    # Target column selection for class imbalance detection
    target_col = st.selectbox(
        "Select target column (for class imbalance check)",
        options=["(None)"] + list(df.columns),
        key="target_for_issues"
    )
    
    target_for_detection = target_col if target_col != "(None)" else None
    
    # Run issue detection
    with st.spinner("Detecting issues..."):
        issue_results = run_issue_detection(df, target_for_detection)
        set_state("detected_issues", issue_results)
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Issues", issue_results.total_issues)
    with col2:
        st.metric("Critical", issue_results.critical_count)
    with col3:
        st.metric("High", issue_results.high_count)
    with col4:
        st.metric("Medium", issue_results.medium_count)
    with col5:
        st.metric("Low", issue_results.low_count)
    
    st.markdown("---")
    
    # Display each issue
    if issue_results.total_issues == 0:
        st.success("No data quality issues detected. Your dataset looks clean.")
    else:
        for i, issue in enumerate(issue_results.issues):
            badge_html = display_severity_badge(issue.severity)
            with st.expander(
                f"{issue.issue_type}: {', '.join(issue.affected_columns[:3])}{'...' if len(issue.affected_columns) > 3 else ''}",
                expanded=(issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH])
            ):
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown(f"**Description:** {issue.description}")
                st.markdown(f"**Suggestion:** {issue.suggestion}")
                st.markdown(f"**Affected Columns:** `{', '.join(issue.affected_columns)}`")
                
                if issue.details:
                    with st.expander("Show Details"):
                        st.json(issue.details)

# Tab 5: Fix Issues
with tab5:
    st.subheader("Approve and Apply Fixes")
    
    st.markdown("""
    Review the detected issues and select which fixes to apply. 
    **All fixes require your approval before being applied.**
    """)
    
    issue_results = get_state("detected_issues")
    
    if issue_results is None or issue_results.total_issues == 0:
        st.info("No issues to fix. Run issue detection in the 'Issues' tab first.")
    else:
        # Group issues by type for easier management
        fix_decisions = {}
        
        st.markdown("### Missing Values")
        missing_issues = [i for i in issue_results.issues if i.issue_type == "Missing Values"]
        
        if missing_issues:
            for issue in missing_issues:
                col = issue.affected_columns[0]
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**{col}** ({issue.details.get('missing_pct', 0)}% missing)")
                
                with col2:
                    method = st.selectbox(
                        f"Fix method for {col}",
                        options=["Keep as is", "Mean", "Median", "Mode", "Drop rows", "Drop column"],
                        key=f"fix_missing_{col}",
                        label_visibility="collapsed"
                    )
                
                with col3:
                    apply = st.checkbox("Apply", key=f"apply_missing_{col}")
                
                if apply and method != "Keep as is":
                    fix_decisions[f"missing_{col}"] = {
                        'column': col,
                        'method': method,
                        'issue_type': 'missing'
                    }
        else:
            st.info("No missing value issues detected.")
        
        st.markdown("---")
        st.markdown("### Outliers")
        outlier_issues = [i for i in issue_results.issues if i.issue_type == "Outliers"]
        
        if outlier_issues:
            for issue in outlier_issues:
                col = issue.affected_columns[0]
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**{col}** ({issue.details.get('outlier_pct', 0)}% outliers)")
                
                with col2:
                    method = st.selectbox(
                        f"Fix method for {col}",
                        options=["Keep as is", "Remove", "Cap (Winsorize)"],
                        key=f"fix_outlier_{col}",
                        label_visibility="collapsed"
                    )
                
                with col3:
                    apply = st.checkbox("Apply", key=f"apply_outlier_{col}")
                
                if apply and method != "Keep as is":
                    fix_decisions[f"outlier_{col}"] = {
                        'column': col,
                        'method': method,
                        'issue_type': 'outlier'
                    }
        else:
            st.info("No outlier issues detected.")
        
        st.markdown("---")
        
        # Apply fixes button
        if st.button("Apply Selected Fixes", type="primary", disabled=len(fix_decisions) == 0):
            with st.spinner("Applying fixes..."):
                df_fixed = df.copy()
                fixes_applied = []
                
                for fix_key, fix_info in fix_decisions.items():
                    col = fix_info['column']
                    method = fix_info['method']
                    
                    if fix_info['issue_type'] == 'missing':
                        if method == "Mean":
                            df_fixed[col].fillna(df_fixed[col].mean(), inplace=True)
                            fixes_applied.append(f"Imputed {col} with mean")
                        elif method == "Median":
                            df_fixed[col].fillna(df_fixed[col].median(), inplace=True)
                            fixes_applied.append(f"Imputed {col} with median")
                        elif method == "Mode":
                            df_fixed[col].fillna(df_fixed[col].mode()[0], inplace=True)
                            fixes_applied.append(f"Imputed {col} with mode")
                        elif method == "Drop rows":
                            df_fixed = df_fixed.dropna(subset=[col])
                            fixes_applied.append(f"Dropped rows with missing {col}")
                        elif method == "Drop column":
                            df_fixed = df_fixed.drop(columns=[col])
                            fixes_applied.append(f"Dropped column {col}")
                    
                    elif fix_info['issue_type'] == 'outlier':
                        if method == "Remove":
                            Q1 = df_fixed[col].quantile(0.25)
                            Q3 = df_fixed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_fixed = df_fixed[
                                (df_fixed[col] >= Q1 - 1.5 * IQR) & 
                                (df_fixed[col] <= Q3 + 1.5 * IQR)
                            ]
                            fixes_applied.append(f"Removed outliers from {col}")
                        elif method == "Cap (Winsorize)":
                            Q1 = df_fixed[col].quantile(0.25)
                            Q3 = df_fixed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            df_fixed[col] = df_fixed[col].clip(lower=lower, upper=upper)
                            fixes_applied.append(f"Capped outliers in {col}")
                
                # Update session state
                set_state("uploaded_df", df_fixed)
                
                # Log decisions
                for fix in fixes_applied:
                    log_user_decision(
                        decision_type="fix_applied",
                        description=fix,
                        details=fix_decisions
                    )
                
                st.success(f"Applied {len(fixes_applied)} fixes.")
                for fix in fixes_applied:
                    st.write(f"  - {fix}")
                
                st.info(f"Dataset now has {len(df_fixed):,} rows and {len(df_fixed.columns)} columns.")
                st.rerun()

# Mark step complete
mark_step_complete("eda")
set_state("eda_complete", True)

# Next step guidance
st.markdown("---")
st.success("""
EDA Complete!

Next Step: Go to Preprocess & Split in the sidebar to configure 
preprocessing and train/test split.
""")
