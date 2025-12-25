"""
Report Generator Module

Generates HTML and Markdown reports with embedded visualizations.
Implements FR-53 to FR-58 from the requirements.

Report sections:
1. Dataset Overview
2. EDA Summary
3. Detected Issues
4. Preprocessing Decisions
5. Model Configurations
6. Performance Comparison
7. Best Model Summary
8. Visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64
import io

from utils.visualizations import (
    create_confusion_matrix,
    create_roc_curves,
    create_metrics_comparison_bar,
    fig_to_base64
)


def generate_html_report(
    dataset_info: Dict[str, Any],
    eda_summary: Dict[str, Any],
    detected_issues: List[Dict[str, Any]],
    preprocessing_decisions: Dict[str, Any],
    trained_models: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    target_classes: List[str],
    best_model_name: str,
    include_visualizations: bool = True
) -> str:
    """
    Generate a self-contained HTML report.
    
    Args:
        dataset_info: Info about the dataset
        eda_summary: EDA analysis summary
        detected_issues: List of detected issues
        preprocessing_decisions: User's preprocessing choices
        trained_models: Dict of trained model objects
        evaluation_results: Dict of evaluation results
        target_classes: List of class names
        best_model_name: Name of the best performing model
        include_visualizations: Whether to embed visualizations
        
    Returns:
        Complete HTML report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate visualizations as base64
    viz_html = ""
    
    if include_visualizations and evaluation_results:
        # Confusion matrices
        viz_html += "<h3>Confusion Matrices</h3>"
        for model_name, result in evaluation_results.items():
            fig = create_confusion_matrix(
                result.confusion_matrix,
                class_names=target_classes,
                title=model_name
            )
            img_base64 = fig_to_base64(fig)
            viz_html += f'<div style="display:inline-block;margin:10px;"><img src="data:image/png;base64,{img_base64}" width="400"/></div>'
        
        # ROC curves (binary only)
        binary_results = {k: v for k, v in evaluation_results.items() if v.is_binary and v.roc_data}
        if binary_results:
            viz_html += "<h3>ROC Curves</h3>"
            roc_data = {k: v.roc_data for k, v in binary_results.items()}
            fig = create_roc_curves(roc_data)
            img_base64 = fig_to_base64(fig)
            viz_html += f'<img src="data:image/png;base64,{img_base64}" width="600"/>'
        
        # Metrics comparison
        viz_html += "<h3>Metrics Comparison</h3>"
        metrics_data = {
            k: {'Accuracy': v.accuracy, 'Precision': v.precision, 'Recall': v.recall, 'F1-Score': v.f1}
            for k, v in evaluation_results.items()
        }
        fig = create_metrics_comparison_bar(metrics_data)
        img_base64 = fig_to_base64(fig)
        viz_html += f'<img src="data:image/png;base64,{img_base64}" width="800"/>'
    
    # Build issues table
    issues_table = ""
    if detected_issues:
        issues_table = """
        <table>
            <tr><th>Issue Type</th><th>Column</th><th>Severity</th><th>Description</th></tr>
        """
        for issue in detected_issues:
            issues_table += f"""
            <tr>
                <td>{issue.get('type', 'Unknown')}</td>
                <td>{issue.get('column', 'N/A')}</td>
                <td>{issue.get('severity', 'Medium')}</td>
                <td>{issue.get('description', '')}</td>
            </tr>
            """
        issues_table += "</table>"
    else:
        issues_table = "<p>No data quality issues detected.</p>"
    
    # Build model comparison table
    model_table = """
    <table>
        <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Training Time</th></tr>
    """
    
    sorted_results = sorted(
        evaluation_results.items(),
        key=lambda x: x[1].f1,
        reverse=True
    )
    
    for model_name, result in sorted_results:
        highlight = ' style="background-color:#90EE90;"' if model_name == best_model_name else ''
        model_table += f"""
        <tr{highlight}>
            <td>{model_name}</td>
            <td>{result.accuracy:.4f}</td>
            <td>{result.precision:.4f}</td>
            <td>{result.recall:.4f}</td>
            <td>{result.f1:.4f}</td>
            <td>{result.training_time:.2f}s</td>
        </tr>
        """
    model_table += "</table>"
    
    # Build preprocessing summary
    preproc_html = "<ul>"
    for key, value in preprocessing_decisions.items():
        preproc_html += f"<li><strong>{key}:</strong> {value}</li>"
    preproc_html += "</ul>"
    
    # Get best model info
    best_result = evaluation_results.get(best_model_name)
    best_model = trained_models.get(best_model_name)
    best_params_html = ""
    if best_model:
        best_params_html = "<ul>"
        for k, v in best_model.params.items():
            best_params_html += f"<li><code>{k}</code>: {v}</li>"
        best_params_html += "</ul>"
    
    # Complete HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Classification Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
            border-left: 4px solid #1f77b4;
            padding-left: 10px;
        }}
        h3 {{
            color: #555;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #1f77b4;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight {{
            background-color: #90EE90 !important;
        }}
        .metric-card {{
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background: linear-gradient(135deg, #1f77b4, #4393c3);
            color: white;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-card .label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #888;
            font-size: 12px;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        ul li {{
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AutoML Classification Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <h2>1. Dataset Overview</h2>
        <div class="metric-card">
            <div class="value">{dataset_info.get('rows', 'N/A'):,}</div>
            <div class="label">Rows</div>
        </div>
        <div class="metric-card">
            <div class="value">{dataset_info.get('columns', 'N/A')}</div>
            <div class="label">Columns</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(target_classes)}</div>
            <div class="label">Classes</div>
        </div>
        <p><strong>Filename:</strong> {dataset_info.get('filename', 'N/A')}</p>
        <p><strong>Target Column:</strong> {dataset_info.get('target', 'N/A')}</p>
        <p><strong>Classes:</strong> {', '.join(str(c) for c in target_classes)}</p>
        
        <h2>2. EDA Summary</h2>
        <ul>
            <li><strong>Missing Values:</strong> {eda_summary.get('missing_values', 0)} total</li>
            <li><strong>Outliers Detected:</strong> {eda_summary.get('outliers', 0)} columns with outliers</li>
            <li><strong>High Correlations:</strong> {eda_summary.get('high_correlations', 0)} pairs</li>
        </ul>
        
        <h2>3. Detected Issues</h2>
        {issues_table}
        
        <h2>4. Preprocessing Decisions</h2>
        {preproc_html}
        
        <h2>5. Model Configurations</h2>
        <p>Trained <strong>{len(trained_models)}</strong> classification models.</p>
        
        <h2>6. Performance Comparison</h2>
        {model_table}
        
        <h2>7. Best Model Summary</h2>
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px;margin:15px 0;">
            <h3 style="margin-top:0;">{best_model_name}</h3>
            <div class="metric-card">
                <div class="value">{best_result.f1 if best_result else 0:.4f}</div>
                <div class="label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="value">{best_result.accuracy if best_result else 0:.4f}</div>
                <div class="label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="value">{best_result.precision if best_result else 0:.4f}</div>
                <div class="label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="value">{best_result.recall if best_result else 0:.4f}</div>
                <div class="label">Recall</div>
            </div>
            <h4>Best Model Parameters:</h4>
            {best_params_html}
        </div>
        
        <h2>8. Visualizations</h2>
        {viz_html if viz_html else '<p>Visualizations not included.</p>'}
        
        <div class="footer">
            <p>Generated by AutoML Classification System</p>
            <p>Report is self-contained with embedded images.</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


def generate_markdown_report(
    dataset_info: Dict[str, Any],
    eda_summary: Dict[str, Any],
    detected_issues: List[Dict[str, Any]],
    preprocessing_decisions: Dict[str, Any],
    trained_models: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    target_classes: List[str],
    best_model_name: str
) -> str:
    """
    Generate a Markdown report.
    
    Args:
        Same as generate_html_report
        
    Returns:
        Markdown report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Issues section
    issues_md = ""
    if detected_issues:
        issues_md = "| Issue Type | Column | Severity | Description |\n"
        issues_md += "|------------|--------|----------|-------------|\n"
        for issue in detected_issues:
            issues_md += f"| {issue.get('type', 'Unknown')} | {issue.get('column', 'N/A')} | {issue.get('severity', 'Medium')} | {issue.get('description', '')} |\n"
    else:
        issues_md = "No data quality issues detected."
    
    # Model comparison table
    model_md = "| Model | Accuracy | Precision | Recall | F1-Score | Training Time |\n"
    model_md += "|-------|----------|-----------|--------|----------|---------------|\n"
    
    sorted_results = sorted(
        evaluation_results.items(),
        key=lambda x: x[1].f1,
        reverse=True
    )
    
    for model_name, result in sorted_results:
        best_marker = ""
        model_md += f"| {model_name}{best_marker} | {result.accuracy:.4f} | {result.precision:.4f} | {result.recall:.4f} | {result.f1:.4f} | {result.training_time:.2f}s |\n"
    
    # Preprocessing section
    preproc_md = ""
    for key, value in preprocessing_decisions.items():
        preproc_md += f"- **{key}:** {value}\n"
    
    # Best model parameters
    best_model = trained_models.get(best_model_name)
    best_result = evaluation_results.get(best_model_name)
    best_params_md = ""
    if best_model:
        for k, v in best_model.params.items():
            best_params_md += f"- `{k}`: {v}\n"
    
    markdown = f"""# AutoML Classification Report

**Generated:** {timestamp}

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Rows** | {dataset_info.get('rows', 'N/A'):,} |
| **Columns** | {dataset_info.get('columns', 'N/A')} |
| **Filename** | {dataset_info.get('filename', 'N/A')} |
| **Target Column** | {dataset_info.get('target', 'N/A')} |
| **Classes** | {len(target_classes)} ({', '.join(str(c) for c in target_classes)}) |

---

## 2. EDA Summary

- **Missing Values:** {eda_summary.get('missing_values', 0)} total
- **Outliers Detected:** {eda_summary.get('outliers', 0)} columns with outliers
- **High Correlations:** {eda_summary.get('high_correlations', 0)} pairs

---

## 3. Detected Issues

{issues_md}

---

## 4. Preprocessing Decisions

{preproc_md}

---

## 5. Model Configurations

Trained **{len(trained_models)}** classification models.

---

## 6. Performance Comparison

{model_md}

---

## 7. Best Model Summary

### {best_model_name}

| Metric | Value |
|--------|-------|
| **F1-Score** | {best_result.f1 if best_result else 0:.4f} |
| **Accuracy** | {best_result.accuracy if best_result else 0:.4f} |
| **Precision** | {best_result.precision if best_result else 0:.4f} |
| **Recall** | {best_result.recall if best_result else 0:.4f} |
| **Training Time** | {best_result.training_time if best_result else 0:.2f}s |

### Parameters

{best_params_md}

---

## 8. Notes

- Primary ranking metric: **F1-Score (weighted)**
- All models trained with `random_state=42` for reproducibility
- Cross-validation: 5-fold stratified

---

*Generated by AutoML Classification System*
"""
    
    return markdown


def save_report_to_file(
    report_content: str,
    filename: str,
    format: str = "html"
) -> bytes:
    """
    Convert report to bytes for download.
    
    Args:
        report_content: Report string content
        filename: Filename for download
        format: 'html' or 'md'
        
    Returns:
        Bytes content for download
    """
    return report_content.encode('utf-8')
