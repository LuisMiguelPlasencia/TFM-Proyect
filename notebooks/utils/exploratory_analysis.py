"""
Utilities for exploratory data analysis using Polars.
"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path


def load_dataset(file_path: Path) -> pl.DataFrame:
    """
    Loads a dataset in CSV or Excel format using Polars.
    
    Args:
        file_path: Path to the file
    Returns:
        Polars DataFrame
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return (
                pl.scan_csv(
                    file_path,
                    sep=";",                    # use semicolons
                    infer_schema_length=10_000, # scan more rows before typing
                    ignore_errors=False,        # or True to skip bad rows
                    # dtype_overrides={         # uncomment to force text type
                    #     "Total": pl.Utf8
                    # },
                )
                .collect()
            )
        elif suffix in [".xls", ".xlsx"]:
            import pandas as pd
            df_pd = pd.read_excel(file_path)
            return pl.from_pandas(df_pd)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_basic_stats(df: pl.DataFrame) -> dict:
    """
    Gets basic statistics from the DataFrame.
    """
    stats = {
        "rows": df.height,
        "columns": df.width,
        "dtypes": df.dtypes,
        "memory_usage": df.estimated_size(),
        "null_counts": df.null_count().to_dict(),
        "duplicate_rows": df.is_duplicated().sum()
    }
    
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if pl.datatypes.is_numeric(dtype)]
    
    if numeric_cols:
        stats["numeric_summary"] = df.select(numeric_cols).describe()
    
    return stats

def plot_numeric_distributions(df: pl.DataFrame, 
                             cols: Optional[List[str]] = None,
                             max_cols: int = 20) -> go.Figure:
    """
    Generates distribution plots for numeric variables using Plotly.
    """
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if pl.datatypes.is_numeric(dtype)]
    
    if cols:
        numeric_cols = [col for col in cols if col in numeric_cols]
    
    if len(numeric_cols) > max_cols:
        print(f"Warning: Limiting visualization to {max_cols} columns")
        numeric_cols = numeric_cols[:max_cols]
    
    n_cols = len(numeric_cols)
    if n_cols == 0:
        return None
    
    fig = make_subplots(rows=n_cols, cols=2,
                       subplot_titles=[f"Distribution of {col}" for col in numeric_cols]*2)
    
    for idx, col in enumerate(numeric_cols, 1):
        # Histogram
        hist_data = df.select(col).to_numpy().flatten()
        hist_data = hist_data[~np.isnan(hist_data)]  # Remove NaN values
        fig.add_trace(
            go.Histogram(x=hist_data, name=f"{col} (hist)"),
            row=idx, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=hist_data, name=f"{col} (box)"),
            row=idx, col=2
        )
    
    fig.update_layout(height=300*n_cols, width=1200, showlegend=False)
    return fig

def plot_categorical_distributions(df: pl.DataFrame,
                                 cols: Optional[List[str]] = None,
                                 max_categories: int = 20) -> List[go.Figure]:
    """
    Generates bar plots for categorical variables.
    """
    categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                       if dtype in [pl.Utf8, pl.Categorical]]
    
    if cols:
        categorical_cols = [col for col in cols if col in categorical_cols]
    
    figures = []
    for col in categorical_cols:
        value_counts = (df.select(col)
                       .groupby(col)
                       .count()
                       .sort("count", reverse=True)
                       .limit(max_categories))
        
        fig = px.bar(value_counts.to_pandas(), 
                    x=col, 
                    y="count",
                    title=f"Distribution of {col}")
        figures.append(fig)
    
    return figures

def plot_correlation_matrix(df: pl.DataFrame) -> go.Figure:
    """
    Generates correlation matrix for numeric variables.
    """
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if pl.datatypes.is_numeric(dtype)]
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df.select(numeric_cols).corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Variable", y="Variable", color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols,
                    aspect="auto")
    
    fig.update_layout(title="Correlation Matrix")
    return fig

def analyze_dataset(df: pl.DataFrame, name: str) -> None:
    """
    Performs a complete analysis of the dataset.
    """
    print(f"\n{'='*50}")
    print(f"Dataset Analysis: {name}")
    print(f"{'='*50}")
    
    # Basic statistics
    stats = get_basic_stats(df)
    print("\nBasic Statistics:")
    print(f"- Rows: {stats['rows']}")
    print(f"- Columns: {stats['columns']}")
    print(f"- Memory Usage: {stats['memory_usage']/1024/1024:.2f} MB")
    print("\nData Types:")
    for col, dtype in zip(df.columns, stats['dtypes']):
        print(f"- {col}: {dtype}")
    
    print("\nNull Values:")
    for col, count in stats['null_counts'].items():
        if count > 0:
            print(f"- {col}: {count}")
    
    print(f"\nDuplicate Rows: {stats['duplicate_rows']}")
    
    if 'numeric_summary' in stats:
        print("\nNumeric Summary Statistics:")
        print(stats['numeric_summary'])
    
    return stats 