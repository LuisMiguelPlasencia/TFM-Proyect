"""
Utilities for exploratory data analysis using modern data science libraries.

This module provides a comprehensive set of tools for analyzing real estate data,
with a focus on performance, reliability, and maintainability.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots


class DataLoader:
    """Efficient data loading with support for multiple formats."""
    
    @staticmethod
    def load_dataset(file_path: Union[str, Path]) -> Optional[pl.DataFrame]:
        """
        Load a dataset in CSV or Excel format using Polars for optimal performance.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Polars DataFrame or None if loading fails
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                # Use scan_csv for lazy loading, then collect for immediate use
                return pl.scan_csv(
                    file_path, 
                    infer_schema_length=10000,
                    ignore_errors=True
                ).collect()
                
            elif suffix in ['.xls', '.xlsx']:
                # For Excel files, we still need pandas as a bridge
                import pandas as pd
                logger.warning(
                    f"Loading Excel file {file_path.name} via pandas bridge. "
                    "Consider converting to CSV for better performance."
                )
                df_pd = pd.read_excel(file_path)
                return pl.from_pandas(df_pd)
                
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            return None


class DataAnalyzer:
    """Comprehensive data analysis functionality."""
    
    @staticmethod
    def get_basic_stats(df: pl.DataFrame) -> Dict:
        """
        Generate comprehensive basic statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "shape": {"rows": df.height, "columns": df.width},
            "memory_usage_mb": df.estimated_size() / (1024 * 1024),
            "dtypes": dict(zip(df.columns, df.dtypes)),
            "null_counts": df.null_count().to_dict(),
            "duplicate_rows": df.is_duplicated().sum(),
        }
        
        # Get numeric and categorical columns
        numeric_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if pl.datatypes.is_numeric(dtype)
        ]
        categorical_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if dtype in [pl.Utf8, pl.Categorical]
        ]
        
        stats["column_types"] = {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "other": [col for col in df.columns 
                     if col not in numeric_cols + categorical_cols]
        }
        
        # Numeric summary
        if numeric_cols:
            stats["numeric_summary"] = df.select(numeric_cols).describe()
            
        # Categorical summary
        if categorical_cols:
            categorical_stats = {}
            for col in categorical_cols:
                unique_count = df.select(pl.col(col).n_unique()).item()
                categorical_stats[col] = {
                    "unique_values": unique_count,
                    "most_frequent": df.select(pl.col(col).mode().first()).item()
                }
            stats["categorical_summary"] = categorical_stats
            
        return stats
    
    @staticmethod
    def analyze_dataset(df: pl.DataFrame, name: str) -> Dict:
        """
        Perform comprehensive dataset analysis with detailed reporting.
        
        Args:
            df: DataFrame to analyze
            name: Name of the dataset for reporting
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting analysis of dataset: {name}")
        
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS: {name.upper()}")
        print(f"{'='*60}")
        
        try:
            stats = DataAnalyzer.get_basic_stats(df)
            
            # Basic information
            print(f"\n📊 BASIC INFORMATION")
            print(f"├── Rows: {stats['shape']['rows']:,}")
            print(f"├── Columns: {stats['shape']['columns']}")
            print(f"└── Memory Usage: {stats['memory_usage_mb']:.2f} MB")
            
            # Data types
            print(f"\n🔍 DATA TYPES")
            print(f"├── Numeric: {len(stats['column_types']['numeric'])}")
            print(f"├── Categorical: {len(stats['column_types']['categorical'])}")
            print(f"└── Other: {len(stats['column_types']['other'])}")
            
            # Data quality
            print(f"\n🚨 DATA QUALITY")
            total_nulls = sum(stats['null_counts'].values())
            print(f"├── Total Null Values: {total_nulls:,}")
            print(f"├── Duplicate Rows: {stats['duplicate_rows']:,}")
            
            if total_nulls > 0:
                print("├── Columns with Nulls:")
                for col, count in stats['null_counts'].items():
                    if count > 0:
                        pct = (count / stats['shape']['rows']) * 100
                        print(f"│   ├── {col}: {count:,} ({pct:.2f}%)")
            
            # Numeric variables summary
            if stats['column_types']['numeric']:
                print(f"\n📈 NUMERIC VARIABLES ({len(stats['column_types']['numeric'])})")
                for col in stats['column_types']['numeric']:
                    col_stats = df.select(
                        pl.col(col).min().alias('min'),
                        pl.col(col).max().alias('max'),
                        pl.col(col).mean().alias('mean'),
                        pl.col(col).std().alias('std')
                    ).row(0)
                    print(f"├── {col}:")
                    print(f"│   ├── Range: [{col_stats[0]:.2f}, {col_stats[1]:.2f}]")
                    print(f"│   └── Mean±Std: {col_stats[2]:.2f}±{col_stats[3]:.2f}")
            
            # Categorical variables summary
            if stats['column_types']['categorical']:
                print(f"\n📋 CATEGORICAL VARIABLES ({len(stats['column_types']['categorical'])})")
                for col in stats['column_types']['categorical']:
                    unique_count = df.select(pl.col(col).n_unique()).item()
                    print(f"├── {col}: {unique_count} unique values")
            
            logger.success(f"Analysis completed for dataset: {name}")
            return stats
            
        except Exception as e:
            logger.error(f"Analysis failed for dataset {name}: {str(e)}")
            raise


class DataVisualizer:
    """High-quality data visualization using Plotly."""
    
    @staticmethod
    def plot_numeric_distributions(
        df: pl.DataFrame, 
        cols: Optional[List[str]] = None,
        max_cols: int = 15,
        title_prefix: str = ""
    ) -> Optional[go.Figure]:
        """
        Create comprehensive distribution plots for numeric variables.
        
        Args:
            df: Input DataFrame
            cols: Specific columns to plot (if None, all numeric)
            max_cols: Maximum number of columns to visualize
            title_prefix: Prefix for plot titles
            
        Returns:
            Plotly Figure or None if no numeric columns
        """
        numeric_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if pl.datatypes.is_numeric(dtype)
        ]
        
        if cols:
            numeric_cols = [col for col in cols if col in numeric_cols]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for distribution plotting")
            return None
            
        if len(numeric_cols) > max_cols:
            logger.warning(f"Limiting visualization to {max_cols} columns")
            numeric_cols = numeric_cols[:max_cols]
        
        n_cols = len(numeric_cols)
        fig = make_subplots(
            rows=n_cols, 
            cols=2,
            subplot_titles=[f"{title_prefix}Distribution: {col}" for col in numeric_cols] * 2,
            horizontal_spacing=0.1,
            vertical_spacing=0.05
        )
        
        for idx, col in enumerate(numeric_cols, 1):
            try:
                # Get clean data (remove nulls)
                clean_data = df.select(col).drop_nulls().to_numpy().flatten()
                
                if len(clean_data) == 0:
                    logger.warning(f"No valid data for column {col}")
                    continue
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=clean_data, 
                        name=f"{col}_hist",
                        nbinsx=min(50, max(10, len(clean_data) // 100)),
                        showlegend=False
                    ),
                    row=idx, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(
                        y=clean_data, 
                        name=f"{col}_box",
                        showlegend=False,
                        boxpoints='outliers'
                    ),
                    row=idx, col=2
                )
                
            except Exception as e:
                logger.error(f"Failed to plot {col}: {str(e)}")
                continue
        
        fig.update_layout(
            height=300 * n_cols,
            width=1200,
            title_text=f"{title_prefix}Numeric Variables Distribution Analysis",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_categorical_distributions(
        df: pl.DataFrame,
        cols: Optional[List[str]] = None,
        max_categories: int = 20,
        title_prefix: str = ""
    ) -> List[go.Figure]:
        """
        Create bar plots for categorical variables.
        
        Args:
            df: Input DataFrame
            cols: Specific columns to plot
            max_categories: Maximum categories to show per variable
            title_prefix: Prefix for plot titles
            
        Returns:
            List of Plotly figures
        """
        categorical_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if dtype in [pl.Utf8, pl.Categorical]
        ]
        
        if cols:
            categorical_cols = [col for col in cols if col in categorical_cols]
        
        if not categorical_cols:
            logger.warning("No categorical columns found")
            return []
        
        figures = []
        for col in categorical_cols:
            try:
                # Get value counts
                value_counts = (
                    df.select(col)
                    .group_by(col)
                    .count()
                    .sort("count", descending=True)
                    .limit(max_categories)
                )
                
                if value_counts.height == 0:
                    continue
                
                # Convert to pandas for plotly
                df_plot = value_counts.to_pandas()
                
                fig = px.bar(
                    df_plot,
                    x=col,
                    y="count",
                    title=f"{title_prefix}Distribution of {col}",
                    labels={"count": "Frequency", col: col}
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=False
                )
                
                figures.append(fig)
                
            except Exception as e:
                logger.error(f"Failed to plot categorical variable {col}: {str(e)}")
                continue
        
        return figures
    
    @staticmethod
    def plot_correlation_matrix(
        df: pl.DataFrame, 
        title_prefix: str = ""
    ) -> Optional[go.Figure]:
        """
        Generate an interactive correlation heatmap for numeric variables.
        
        Args:
            df: Input DataFrame
            title_prefix: Prefix for plot title
            
        Returns:
            Plotly Figure or None if insufficient numeric columns
        """
        numeric_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if pl.datatypes.is_numeric(dtype)
        ]
        
        if len(numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for correlation matrix")
            return None
        
        try:
            # Calculate correlation matrix
            corr_matrix = df.select(numeric_cols).corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix.to_numpy(),
                labels=dict(x="Variables", y="Variables", color="Correlation"),
                x=numeric_cols,
                y=numeric_cols,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title=f"{title_prefix}Correlation Matrix"
            )
            
            fig.update_layout(
                width=max(600, len(numeric_cols) * 50),
                height=max(600, len(numeric_cols) * 50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create correlation matrix: {str(e)}")
            return None


# Convenience functions for backward compatibility
def load_dataset(file_path: Union[str, Path]) -> Optional[pl.DataFrame]:
    """Load a dataset using the DataLoader class."""
    return DataLoader.load_dataset(file_path)


def get_basic_stats(df: pl.DataFrame) -> Dict:
    """Get basic statistics using the DataAnalyzer class."""
    return DataAnalyzer.get_basic_stats(df)


def analyze_dataset(df: pl.DataFrame, name: str) -> Dict:
    """Analyze dataset using the DataAnalyzer class."""
    return DataAnalyzer.analyze_dataset(df, name)


def plot_numeric_distributions(
    df: pl.DataFrame, 
    cols: Optional[List[str]] = None,
    max_cols: int = 15
) -> Optional[go.Figure]:
    """Plot numeric distributions using the DataVisualizer class."""
    return DataVisualizer.plot_numeric_distributions(df, cols, max_cols)


def plot_categorical_distributions(
    df: pl.DataFrame,
    cols: Optional[List[str]] = None,
    max_categories: int = 20
) -> List[go.Figure]:
    """Plot categorical distributions using the DataVisualizer class."""
    return DataVisualizer.plot_categorical_distributions(df, cols, max_categories)


def plot_correlation_matrix(df: pl.DataFrame) -> Optional[go.Figure]:
    """Plot correlation matrix using the DataVisualizer class."""
    return DataVisualizer.plot_correlation_matrix(df) 