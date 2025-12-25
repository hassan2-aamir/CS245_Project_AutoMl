"""
Data Loader Module

Handles CSV file loading, encoding detection, and initial validation
for the AutoML application.

Implements FR-1 to FR-4:
- FR-1: CSV file upload
- FR-2: File format validation
- FR-3: Encoding detection
- FR-4: Large file handling (200 MB limit)
"""

import pandas as pd
import numpy as np
import chardet
import io
from typing import Tuple, Optional, Dict, Any
import streamlit as st


# Constants
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SAMPLE_SIZE_FOR_ENCODING = 10000  # Bytes to sample for encoding detection


def detect_encoding(file_content: bytes) -> str:
    """
    Detect the encoding of a file using chardet.
    
    Args:
        file_content: Raw bytes of the file
        
    Returns:
        Detected encoding string (e.g., 'utf-8', 'latin-1')
    """
    # Sample the first portion for faster detection
    sample = file_content[:SAMPLE_SIZE_FOR_ENCODING]
    result = chardet.detect(sample)
    
    encoding = result.get('encoding', 'utf-8')
    confidence = result.get('confidence', 0)
    
    # Fall back to common encodings if confidence is low
    if confidence < 0.7:
        # Try common encodings in order
        for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                file_content.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
    
    # Normalize encoding names
    encoding_map = {
        'ascii': 'utf-8',
        'ISO-8859-1': 'latin-1',
        'Windows-1252': 'cp1252',
    }
    
    return encoding_map.get(encoding, encoding) or 'utf-8'


def validate_csv_format(file_content: bytes, encoding: str) -> Tuple[bool, str]:
    """
    Validate that the file content is valid CSV format.
    
    Args:
        file_content: Raw bytes of the file
        encoding: Detected encoding
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to decode and parse as CSV
        text_content = file_content.decode(encoding)
        
        # Quick check for CSV-like structure (has delimiters)
        first_lines = text_content.split('\n')[:5]
        if not first_lines:
            return False, "File appears to be empty."
        
        # Check if it looks like CSV (has commas or other delimiters)
        first_line = first_lines[0]
        if ',' not in first_line and '\t' not in first_line and ';' not in first_line:
            # Could still be valid if it's a single-column CSV
            pass
        
        # Try to actually parse with pandas
        df = pd.read_csv(io.StringIO(text_content), nrows=5)
        if df.empty and len(df.columns) == 0:
            return False, "Unable to parse file as CSV. No data found."
        
        return True, ""
        
    except UnicodeDecodeError:
        return False, f"Unable to decode file with {encoding} encoding."
    except pd.errors.EmptyDataError:
        return False, "File is empty or contains no data."
    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error reading file: {str(e)}"


def load_csv(
    uploaded_file,
    encoding: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load a CSV file from an uploaded file object.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        encoding: Optional encoding override
        
    Returns:
        Tuple of (DataFrame or None, metadata dict)
        
    Metadata dict contains:
        - success: bool
        - error: str (if failed)
        - filename: str
        - file_size_mb: float
        - encoding: str
        - rows: int
        - columns: int
        - memory_mb: float
    """
    metadata = {
        'success': False,
        'error': None,
        'filename': uploaded_file.name,
        'file_size_mb': 0,
        'encoding': None,
        'rows': 0,
        'columns': 0,
        'memory_mb': 0,
    }
    
    # Check file size
    file_content = uploaded_file.getvalue()
    file_size_bytes = len(file_content)
    file_size_mb = file_size_bytes / (1024 * 1024)
    metadata['file_size_mb'] = round(file_size_mb, 2)
    
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        metadata['error'] = (
            f"**Problem**: File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit. "
            f"**Solution**: Please reduce your dataset size or split into smaller files."
        )
        return None, metadata
    
    # Detect encoding if not provided
    if encoding is None:
        encoding = detect_encoding(file_content)
    metadata['encoding'] = encoding
    
    # Validate CSV format
    is_valid, error_msg = validate_csv_format(file_content, encoding)
    if not is_valid:
        metadata['error'] = f"**Problem**: {error_msg} **Solution**: Please ensure you're uploading a valid CSV file."
        return None, metadata
    
    # Load the CSV
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Try to read with detected encoding
        df = pd.read_csv(
            uploaded_file,
            encoding=encoding,
            low_memory=False  # For consistent dtype inference
        )
        
        # Basic validation
        if df.empty:
            metadata['error'] = (
                "**Problem**: The file was parsed but contains no data. "
                "**Solution**: Please upload a CSV file with data rows."
            )
            return None, metadata
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
            # Make column names unique
            cols = []
            seen = {}
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    cols.append(col)
            df.columns = cols
        
        # Update metadata
        metadata['success'] = True
        metadata['rows'] = len(df)
        metadata['columns'] = len(df.columns)
        metadata['memory_mb'] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        
        return df, metadata
        
    except UnicodeDecodeError:
        # Try alternative encodings
        for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']:
            if alt_encoding == encoding:
                continue
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=alt_encoding, low_memory=False)
                metadata['success'] = True
                metadata['encoding'] = alt_encoding
                metadata['rows'] = len(df)
                metadata['columns'] = len(df.columns)
                metadata['memory_mb'] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
                return df, metadata
            except:
                continue
        
        metadata['error'] = (
            "**Problem**: Unable to decode file with any common encoding. "
            "**Solution**: Please ensure your CSV file is saved with UTF-8 or Latin-1 encoding."
        )
        return None, metadata
        
    except pd.errors.EmptyDataError:
        metadata['error'] = (
            "**Problem**: The file is empty or contains only whitespace. "
            "**Solution**: Please upload a CSV file with actual data."
        )
        return None, metadata
        
    except pd.errors.ParserError as e:
        metadata['error'] = (
            f"**Problem**: Error parsing CSV file - {str(e)}. "
            "**Solution**: Please check that your file is a valid CSV with consistent columns."
        )
        return None, metadata
        
    except Exception as e:
        metadata['error'] = (
            f"**Problem**: Unexpected error loading file - {str(e)}. "
            "**Solution**: Please try re-saving your CSV file and uploading again."
        )
        return None, metadata


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about each column in the DataFrame.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        DataFrame with column information
    """
    info = []
    
    for col in df.columns:
        col_data = df[col]
        
        col_info = {
            'Column': col,
            'Type': str(col_data.dtype),
            'Non-Null Count': col_data.notna().sum(),
            'Null Count': col_data.isna().sum(),
            'Null %': round(col_data.isna().sum() / len(df) * 100, 2),
            'Unique Values': col_data.nunique(),
            'Sample Values': str(col_data.dropna().head(3).tolist())[:50]
        }
        
        # Add numeric-specific info
        if pd.api.types.is_numeric_dtype(col_data):
            col_info['Mean'] = round(col_data.mean(), 2) if col_data.notna().any() else None
            col_info['Std'] = round(col_data.std(), 2) if col_data.notna().any() else None
            col_info['Min'] = col_data.min() if col_data.notna().any() else None
            col_info['Max'] = col_data.max() if col_data.notna().any() else None
        
        info.append(col_info)
    
    return pd.DataFrame(info)


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for numerical columns.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        DataFrame with summary statistics (transposed describe)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    summary = numeric_df.describe().T
    summary = summary.round(2)
    summary.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    return summary


def get_categorical_summary(df: pd.DataFrame, max_categories: int = 10) -> Dict[str, pd.Series]:
    """
    Get value counts for categorical columns.
    
    Args:
        df: The pandas DataFrame
        max_categories: Maximum number of categories to show per column
        
    Returns:
        Dictionary mapping column names to value counts
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    summaries = {}
    for col in categorical_cols:
        counts = df[col].value_counts().head(max_categories)
        summaries[col] = counts
    
    return summaries


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic column types for better preprocessing decisions.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        Dictionary mapping column names to inferred types:
        - 'numeric_continuous': Continuous numeric data
        - 'numeric_discrete': Integer-like data with few unique values
        - 'categorical': Categorical/text data
        - 'binary': Boolean or 2-value columns
        - 'datetime': Date/time columns
        - 'id': Likely identifier columns
    """
    type_mapping = {}
    
    for col in df.columns:
        col_data = df[col]
        n_unique = col_data.nunique()
        n_total = len(col_data)
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            type_mapping[col] = 'datetime'
            continue
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(col_data):
            # Check if it's likely an ID column
            if n_unique == n_total or (n_unique / n_total > 0.95 and 'id' in col.lower()):
                type_mapping[col] = 'id'
            # Check if it's binary
            elif n_unique == 2:
                type_mapping[col] = 'binary'
            # Check if it's discrete (few unique values relative to total)
            elif n_unique <= 20 or (pd.api.types.is_integer_dtype(col_data) and n_unique / n_total < 0.05):
                type_mapping[col] = 'numeric_discrete'
            else:
                type_mapping[col] = 'numeric_continuous'
        else:
            # Text/categorical
            if n_unique == 2:
                type_mapping[col] = 'binary'
            elif n_unique == n_total:
                type_mapping[col] = 'id'
            else:
                type_mapping[col] = 'categorical'
    
    return type_mapping
