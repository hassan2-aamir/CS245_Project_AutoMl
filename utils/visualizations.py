"""
Visualization Utilities

Provides wrapper functions for creating consistent visualizations
using Matplotlib and Seaborn throughout the AutoML application.

All plot functions return matplotlib Figure objects for embedding
in Streamlit via st.pyplot(fig).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import io
import base64


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def create_histogram(
    data: pd.Series,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 30,
    kde: bool = True,
    color: str = "#3498db"
) -> plt.Figure:
    """
    Create a histogram with optional KDE curve.
    
    Args:
        data: The data series to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        kde: Whether to show KDE curve
        color: Bar color
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(data=data, bins=bins, kde=kde, color=color, ax=ax)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel or data.name, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    plt.tight_layout()
    return fig


def create_bar_plot(
    data: pd.Series,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    color: str = "#2ecc71",
    max_categories: int = 20,
    horizontal: bool = False
) -> plt.Figure:
    """
    Create a bar plot for categorical data.
    
    Args:
        data: The data series to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        max_categories: Maximum number of categories to show
        horizontal: Whether to create horizontal bars
        
    Returns:
        Matplotlib Figure object
    """
    value_counts = data.value_counts()
    
    # Limit categories if needed
    if len(value_counts) > max_categories:
        value_counts = value_counts.head(max_categories)
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(value_counts) * 0.3)))
    
    if horizontal:
        value_counts.plot(kind='barh', color=color, ax=ax)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_ylabel(xlabel or data.name, fontsize=10)
    else:
        value_counts.plot(kind='bar', color=color, ax=ax)
        ax.set_xlabel(xlabel or data.name, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    cmap: str = "RdBu_r",
    annot: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create a correlation matrix heatmap.
    
    Args:
        df: DataFrame with numerical columns
        title: Plot title
        cmap: Color map
        annot: Whether to show correlation values
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation matrix",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust annotation size based on number of features
    annot_size = max(6, 12 - len(corr_matrix.columns) // 3)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot and len(corr_matrix.columns) <= 15,
        fmt='.2f',
        annot_kws={'size': annot_size},
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig


def create_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    normalize: bool = False
) -> plt.Figure:
    """
    Create a confusion matrix heatmap.
    
    Args:
        cm: The confusion matrix array
        class_names: List of class names
        title: Plot title
        cmap: Color map
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib Figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves Comparison"
) -> plt.Figure:
    """
    Create overlaid ROC curves for multiple models.
    
    Args:
        roc_data: Dictionary mapping model name to (fpr, tpr, auc) tuples
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    
    for (model_name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {auc:.3f})')
    
    # Diagonal line for random classifier
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_metrics_comparison_bar(
    metrics_df: pd.DataFrame,
    metric_columns: List[str] = None,
    title: str = "Model Performance Comparison"
) -> plt.Figure:
    """
    Create grouped bar chart comparing metrics across models.
    
    Args:
        metrics_df: DataFrame with model names as index, metrics as columns
        metric_columns: List of metric columns to plot (default: all numeric)
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(metrics_df, dict):
        metrics_df = pd.DataFrame(metrics_df)
    
    if metric_columns is None:
        metric_columns = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.8 / len(metric_columns)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metric_columns)))
    
    for i, (metric, color) in enumerate(zip(metric_columns, colors)):
        offset = (i - len(metric_columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, metrics_df[metric], width, label=metric, color=color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def create_class_distribution_plot(
    series: pd.Series,
    title: str = "Class Distribution"
) -> plt.Figure:
    """
    Create a pie chart and bar chart for class distribution.
    
    Args:
        series: The target variable series
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    value_counts = series.value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(value_counts)))
    ax1.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title(f'{title} (Pie)', fontsize=11, fontweight='bold')
    
    # Bar chart
    value_counts.plot(kind='bar', color=colors, ax=ax2)
    ax2.set_title(f'{title} (Bar)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, (idx, val) in enumerate(value_counts.items()):
        ax2.annotate(f'{val}', xy=(i, val), xytext=(0, 3),
                    textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_missing_values_plot(
    missing_data: pd.DataFrame,
    title: str = "Missing Values by Feature"
) -> plt.Figure:
    """
    Create a bar plot showing missing values percentage.
    
    Args:
        missing_data: DataFrame with 'feature' and 'missing_pct' columns
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    # Filter to only columns with missing values
    missing_data = missing_data[missing_data['missing_pct'] > 0].sort_values(
        'missing_pct', ascending=True
    )
    
    if len(missing_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values found! ✓",
                ha='center', va='center', fontsize=14, color='green')
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_data) * 0.3)))
    
    colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 20 else '#27ae60' 
              for x in missing_data['missing_pct']]
    
    bars = ax.barh(missing_data['feature'], missing_data['missing_pct'], color=colors)
    
    ax.set_xlabel('Missing Percentage (%)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, pct in zip(bars, missing_data['missing_pct']):
        ax.annotate(f'{pct:.1f}%', xy=(bar.get_width() + 1, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to base64 encoded string for embedding in HTML.
    
    Args:
        fig: Matplotlib Figure object
        
    Returns:
        Base64 encoded string of the image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def close_all_figures():
    """Close all open matplotlib figures to free memory."""
    plt.close('all')
