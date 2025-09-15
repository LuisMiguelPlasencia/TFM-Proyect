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
        df_sales_viz['price_per_sqm'] = pd.to_numeric(df_sales[price_col], errors='coerce') / pd.to_numeric(df_sales[surface_col], errors='coerce')
        df_rental_viz['price_per_sqm'] = pd.to_numeric(df_rental[price_col], errors='coerce') / pd.to_numeric(df_rental[surface_col], errors='coerce')
        #df_sales_viz['price_per_sqm'] = df_sales[price_col] / df_sales[surface_col]
        #df_rental_viz['price_per_sqm'] = df_rental[price_col] / df_rental[surface_col]
        
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


def compare_cities_markets(all_city_data: Dict[str, Dict], 
                          price_col: str = 'price',
                          surface_col: str = 'm2_real') -> Optional[pd.DataFrame]:
    """
    Compare real estate markets across multiple cities.
    
    Parameters
    ----------
    all_city_data : Dict[str, Dict]
        Dictionary with city data from process_all_cities
    price_col : str
        Name of the price column
    surface_col : str  
        Name of the surface area column
        
    Returns
    -------
    Optional[pd.DataFrame]
        Comparison DataFrame or None if error
    """
    print("\n🔄 COMPARING MARKETS ACROSS ALL CITIES")
    print("=" * 50)
    
    comparison_data = []
    
    for city, data in all_city_data.items():
        sales_df = data['sales']
        rental_df = data['rental']
        
        city_stats = {
            'City': city.title(),
            'Sales_Count': data['sales_count'],
            'Rental_Count': data['rental_count'],
            'Total_Properties': data['sales_count'] + data['rental_count']
        }
        
        # Sales market statistics
        if sales_df is not None and len(sales_df) > 0 and price_col in sales_df.columns:
            city_stats.update({
                'Avg_Sales_Price': sales_df[price_col].mean(),
                'Median_Sales_Price': sales_df[price_col].median(),
                'Sales_Price_Std': sales_df[price_col].std(),
                'Min_Sales_Price': sales_df[price_col].min(),
                'Max_Sales_Price': sales_df[price_col].max()
            })
            
            # Surface area statistics for sales
            if surface_col in sales_df.columns:
                city_stats.update({
                    'Avg_Sales_Surface': sales_df[surface_col].mean(),
                    'Median_Sales_Surface': sales_df[surface_col].median()
                })
                
                # Price per m2 for sales
                if sales_df[surface_col].mean() > 0:
                    city_stats['Avg_Sales_Price_per_m2'] = city_stats['Avg_Sales_Price'] / city_stats['Avg_Sales_Surface']
        
        # Rental market statistics  
        if rental_df is not None and len(rental_df) > 0 and price_col in rental_df.columns:
            city_stats.update({
                'Avg_Rental_Price': rental_df[price_col].mean(),
                'Median_Rental_Price': rental_df[price_col].median(),
                'Rental_Price_Std': rental_df[price_col].std(),
                'Min_Rental_Price': rental_df[price_col].min(),
                'Max_Rental_Price': rental_df[price_col].max()
            })
            
            # Surface area statistics for rentals
            if surface_col in rental_df.columns:
                city_stats.update({
                    'Avg_Rental_Surface': rental_df[surface_col].mean(),
                    'Median_Rental_Surface': rental_df[surface_col].median()
                })
                
                # Price per m2 for rentals
                if rental_df[surface_col].mean() > 0:
                    city_stats['Avg_Rental_Price_per_m2'] = city_stats['Avg_Rental_Price'] / city_stats['Avg_Rental_Surface']
            
            # Calculate rental yield if both markets exist
            if 'Avg_Sales_Price' in city_stats and city_stats['Avg_Sales_Price'] > 0:
                annual_rental = city_stats['Avg_Rental_Price'] * 12
                city_stats['Estimated_Rental_Yield'] = (annual_rental / city_stats['Avg_Sales_Price']) * 100
        
        comparison_data.append(city_stats)
    
    if not comparison_data:
        print("❌ No data available for comparison")
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by total properties (descending)
    comparison_df = comparison_df.sort_values('Total_Properties', ascending=False)
    
    print(f"📊 Market comparison completed for {len(comparison_df)} cities")
    
    return comparison_df


def create_cities_visualization(comparison_df: pd.DataFrame, 
                               output_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Create visualizations comparing cities markets.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison data from compare_cities_markets
    output_dir : Optional[Union[str, Path]]
        Directory to save plots
        
    Returns
    -------
    bool
        Success status
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spanish Real Estate Markets - City Comparison', fontsize=16, fontweight='bold')
        
        # 1. Property counts by city
        if 'Total_Properties' in comparison_df.columns:
            top_cities = comparison_df.head(10)  # Top 10 cities by property count
            ax1.bar(range(len(top_cities)), top_cities['Total_Properties'], 
                   color='skyblue', alpha=0.7)
            ax1.set_title('Total Properties by City (Top 10)')
            ax1.set_xlabel('Cities')
            ax1.set_ylabel('Number of Properties')
            ax1.set_xticks(range(len(top_cities)))
            ax1.set_xticklabels(top_cities['City'], rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(top_cities['Total_Properties']):
                ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. Average sales prices by city
        if 'Avg_Sales_Price' in comparison_df.columns:
            sales_data = comparison_df.dropna(subset=['Avg_Sales_Price']).head(10)
            ax2.bar(range(len(sales_data)), sales_data['Avg_Sales_Price'], 
                   color='lightgreen', alpha=0.7)
            ax2.set_title('Average Sales Price by City (Top 10)')
            ax2.set_xlabel('Cities')
            ax2.set_ylabel('Average Sales Price (€)')
            ax2.set_xticks(range(len(sales_data)))
            ax2.set_xticklabels(sales_data['City'], rotation=45, ha='right')
            
            # Format y-axis as currency
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
        
        # 3. Average rental prices by city
        if 'Avg_Rental_Price' in comparison_df.columns:
            rental_data = comparison_df.dropna(subset=['Avg_Rental_Price']).head(10)
            ax3.bar(range(len(rental_data)), rental_data['Avg_Rental_Price'], 
                   color='orange', alpha=0.7)
            ax3.set_title('Average Monthly Rental Price by City (Top 10)')
            ax3.set_xlabel('Cities')
            ax3.set_ylabel('Average Monthly Rental (€)')
            ax3.set_xticks(range(len(rental_data)))
            ax3.set_xticklabels(rental_data['City'], rotation=45, ha='right')
            
            # Format y-axis as currency
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
        
        # 4. Rental yields by city
        if 'Estimated_Rental_Yield' in comparison_df.columns:
            yield_data = comparison_df.dropna(subset=['Estimated_Rental_Yield']).head(10)
            ax4.bar(range(len(yield_data)), yield_data['Estimated_Rental_Yield'], 
                   color='coral', alpha=0.7)
            ax4.set_title('Estimated Rental Yield by City (Top 10)')
            ax4.set_xlabel('Cities')
            ax4.set_ylabel('Rental Yield (%)')
            ax4.set_xticks(range(len(yield_data)))
            ax4.set_xticklabels(yield_data['City'], rotation=45, ha='right')
            
            # Add percentage labels
            for i, v in enumerate(yield_data['Estimated_Rental_Yield']):
                ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            plot_file = output_path / "cities_market_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        print("Falling back to text summary...")
        return False


def print_cities_summary(comparison_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of cities comparison.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison data from compare_cities_markets
    """
    print("\n📊 SPANISH CITIES REAL ESTATE MARKET SUMMARY")
    print("=" * 60)
    
    total_cities = len(comparison_df)
    total_properties = comparison_df['Total_Properties'].sum()
    total_sales = comparison_df['Sales_Count'].sum()
    total_rentals = comparison_df['Rental_Count'].sum()
    
    print(f"🏙️ Total cities analyzed: {total_cities}")
    print(f"🏠 Total properties: {total_properties:,}")
    print(f"💰 Total sales listings: {total_sales:,}")
    print(f"🏡 Total rental listings: {total_rentals:,}")
    
    # Top cities by different metrics
    print(f"\n🏆 TOP PERFORMING CITIES")
    print("-" * 30)
    
    # By total properties
    if 'Total_Properties' in comparison_df.columns:
        top_by_count = comparison_df.nlargest(5, 'Total_Properties')
        print(f"\n📊 Most properties:")
        for i, (_, row) in enumerate(top_by_count.iterrows(), 1):
            print(f"   {i}. {row['City']}: {row['Total_Properties']:,} properties")
    
    # By average sales price
    if 'Avg_Sales_Price' in comparison_df.columns:
        top_by_sales = comparison_df.dropna(subset=['Avg_Sales_Price']).nlargest(5, 'Avg_Sales_Price')
        print(f"\n💎 Highest average sales prices:")
        for i, (_, row) in enumerate(top_by_sales.iterrows(), 1):
            print(f"   {i}. {row['City']}: €{row['Avg_Sales_Price']:,.0f}")
    
    # By rental yield
    if 'Estimated_Rental_Yield' in comparison_df.columns:
        top_by_yield = comparison_df.dropna(subset=['Estimated_Rental_Yield']).nlargest(5, 'Estimated_Rental_Yield')
        print(f"\n📈 Best rental yields:")
        for i, (_, row) in enumerate(top_by_yield.iterrows(), 1):
            print(f"   {i}. {row['City']}: {row['Estimated_Rental_Yield']:.2f}%") 