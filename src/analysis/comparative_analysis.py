"""
Comparative Analysis Utilities

This module provides utilities for comparing real estate sales and rental markets,
following project best practices for reproducible analysis.

Dependencies: pandas, numpy, plotly, matplotlib, seaborn
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
    # Set Plotly defaults
    px.defaults.template = "plotly_white" 
    px.defaults.width = 1200
    px.defaults.height = 700
except ImportError:
    HAS_PLOTLY = False

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_style("whitegrid")


def compare_market_columns(df_sales: pd.DataFrame, df_rental: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify column structure for market comparison.
    
    Parameters
    ----------
    df_sales : pd.DataFrame
        Sales data
    df_rental : pd.DataFrame
        Rental data
        
    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Common columns, price columns, surface columns
    """
    common_columns = sorted(list(set(df_sales.columns) & set(df_rental.columns)))
    
    price_columns = [col for col in common_columns if 'price' in col.lower()]
    surface_columns = [col for col in common_columns if any(x in col.lower() for x in ['surface', 'm2'])]
    
    return common_columns, price_columns, surface_columns


def calculate_market_metrics(df: pd.DataFrame, 
                           price_col: str, 
                           surface_col: str, 
                           market_type: str = 'sales') -> Dict[str, float]:
    """
    Calculate standardized market metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Property data
    price_col : str
        Name of price column
    surface_col : str
        Name of surface area column
    market_type : str, default 'sales'
        Type of market ('sales' or 'rental')
        
    Returns
    -------
    Dict[str, float]
        Dictionary with calculated metrics
    """
    df_metrics = df.copy()
    df_metrics['price_per_sqm'] = df[price_col] / df[surface_col]
    
    metrics = {
        'count': len(df_metrics),
        'avg_price': df_metrics[price_col].mean(),
        'median_price': df_metrics[price_col].median(),
        'avg_surface': df_metrics[surface_col].mean(),
        'avg_price_per_sqm': df_metrics['price_per_sqm'].mean()
    }
    
    if market_type == 'rental':
        df_metrics['annual_price'] = df[price_col] * 12
        metrics['avg_annual_price'] = df_metrics['annual_price'].mean()
    
    return metrics


def display_comparative_metrics(sales_metrics: Dict[str, float], 
                              rental_metrics: Dict[str, float]) -> None:
    """
    Display formatted comparison between sales and rental metrics.
    
    Parameters
    ----------
    sales_metrics : Dict[str, float]
        Sales market metrics
    rental_metrics : Dict[str, float]
        Rental market metrics
    """
    print("=== COMPARATIVE METRICS: SALES vs. RENTALS ===\n")
    print(f"Sales listings: {sales_metrics['count']:,}")
    print(f"Rental listings: {rental_metrics['count']:,}\n")
    
    print(f"Average sales price: €{sales_metrics['avg_price']:,.2f}")
    print(f"Average monthly rental: €{rental_metrics['avg_price']:,.2f}")
    if 'avg_annual_price' in rental_metrics:
        print(f"Average annual rental: €{rental_metrics['avg_annual_price']:,.2f}\n")
    
    print(f"Average property size (sales): {sales_metrics['avg_surface']:,.2f} m²")
    print(f"Average property size (rentals): {rental_metrics['avg_surface']:,.2f} m²\n")
    
    print(f"Average price per m² (sales): €{sales_metrics['avg_price_per_sqm']:,.2f}")
    print(f"Average price per m² (rentals): €{rental_metrics['avg_price_per_sqm']:,.2f} /month")
    
    # Calculate yield if annual price available
    if 'avg_annual_price' in rental_metrics:
        avg_yield = rental_metrics['avg_annual_price'] / sales_metrics['avg_price'] * 100
        print(f"\nEstimated average rental yield: {avg_yield:.2f}%")


def create_distribution_comparison(df_sales: pd.DataFrame, 
                                 df_rental: pd.DataFrame,
                                 column: str, 
                                 title: str,
                                 xlabel: str = None,
                                 use_plotly: bool = None) -> None:
    """
    Create distribution comparison visualizations.
    
    Parameters
    ----------
    df_sales : pd.DataFrame
        Sales data
    df_rental : pd.DataFrame
        Rental data
    column : str
        Column to compare
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    use_plotly : bool, optional
        Force plotly usage. If None, auto-detect
    """
    if use_plotly is None:
        use_plotly = HAS_PLOTLY
    
    xlabel = xlabel or column.replace('_', ' ').title()
    
    def create_matplotlib_plot():
        """Fallback matplotlib visualization."""
        plt.figure(figsize=(12, 6))
        plt.hist(
            [df_sales[column].dropna().values, df_rental[column].dropna().values],
            bins=30, alpha=0.7, label=['Sales', 'Rentals']
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    if use_plotly:
        try:
            fig = go.Figure()
            
            # Add sales data
            fig.add_trace(go.Histogram(
                x=df_sales[column].dropna().values,
                name='Sales',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Add rental data
            fig.add_trace(go.Histogram(
                x=df_rental[column].dropna().values,
                name='Rentals',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig.update_layout(
                title_text=title,
                xaxis_title_text=xlabel,
                yaxis_title_text='Count',
                bargap=0.2,
                bargroupgap=0.1
            )
            
            fig.show()
        except Exception as e:
            print(f"Error with Plotly visualization: {e}")
            print("Falling back to matplotlib...")
            create_matplotlib_plot()
    else:
        create_matplotlib_plot()


def safe_compare_markets(df_sales: pd.DataFrame, 
                        df_rental: pd.DataFrame,
                        price_col: str = None,
                        surface_col: str = None) -> bool:
    """
    Safely compare markets with automatic column detection.
    
    Parameters
    ----------
    df_sales : pd.DataFrame
        Sales data
    df_rental : pd.DataFrame
        Rental data
    price_col : str, optional
        Price column name. Auto-detected if None
    surface_col : str, optional
        Surface column name. Auto-detected if None
        
    Returns
    -------
    bool
        True if comparison was successful
    """
    try:
        print("\n=== EXAMINING COLUMNS FOR COMPARISON ===")
        
        # Auto-detect columns if not provided
        common_columns, price_columns, surface_columns = compare_market_columns(df_sales, df_rental)
        
        if price_col is None:
            price_col = price_columns[0] if price_columns else 'price'
        if surface_col is None:
            surface_col = surface_columns[0] if surface_columns else None
            
        print(f"Price column: {price_col}")
        print(f"Surface columns available: {surface_columns}")
        print(f"Selected surface column: {surface_col}")
        
        # Check column availability
        sales_has_required = all(col in df_sales.columns for col in [price_col, surface_col] if col)
        rental_has_required = all(col in df_rental.columns for col in [price_col, surface_col] if col)
        
        print(f"\nSales data has required columns: {sales_has_required}")
        print(f"Rental data has required columns: {rental_has_required}")
        
        if not (sales_has_required and rental_has_required):
            print(f"\n❌ Missing required columns for comparison")
            return False
            
        # Calculate metrics
        sales_metrics = calculate_market_metrics(df_sales, price_col, surface_col, 'sales')
        rental_metrics = calculate_market_metrics(df_rental, price_col, surface_col, 'rental')
        
        # Display comparison
        display_comparative_metrics(sales_metrics, rental_metrics)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Surface area distribution
        create_distribution_comparison(
            df_sales, df_rental, surface_col,
            'Property Size Distribution: Sales vs. Rentals',
            'Surface Area (m²)'
        )
        
        # Price per sqm distribution
        df_sales_viz = df_sales.copy()
        df_rental_viz = df_rental.copy()
        df_sales_viz['price_per_sqm'] = df_sales[price_col] / df_sales[surface_col]
        df_rental_viz['price_per_sqm'] = df_rental[price_col] / df_rental[surface_col]
        
        create_distribution_comparison(
            df_sales_viz, df_rental_viz, 'price_per_sqm',
            'Price per m² Distribution: Sales vs. Rentals',
            'Price per m²'
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Error in market comparison: {e}")
        return False


def examine_data_structure(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Examine and display data structure information.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to examine
    name : str, default "Dataset"
        Name for display purposes
    """
    print(f"\n=== {name.upper()} STRUCTURE ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    print(f"\nColumns ({len(df.columns)}):")
    for col, dtype in zip(df.columns, df.dtypes):
        non_null_count = df[col].count()
        null_count = df[col].isnull().sum()
        print(f"  - {col}: {dtype} ({non_null_count} non-null, {null_count} null)")
    
    print(f"\nSample data:")
    print(df.head(3).to_string()) 