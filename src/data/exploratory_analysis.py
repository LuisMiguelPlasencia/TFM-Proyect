"""
Exploratory Data Analysis Utilities

This module provides standardized functions for loading, cleaning, and exploring
real estate data following project best practices.

Dependencies: pandas, numpy, pathlib
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def load_data(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file into a pandas DataFrame with informative output.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the CSV file

    Returns
    -------
    Optional[pd.DataFrame]
        Loaded data or None if loading fails
    """
    try:
        file_path = Path(file_path)
        data = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully from {file_path}")
        print(f"   Shape: {data.shape}")
        print(f"   Memory usage: {data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        return data
    except Exception as e:
        print(f"❌ Error loading data from {file_path}: {e}")
        return None


def clean_data(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, int]]]:
    """
    Clean and preprocess real estate data following standardized procedures.

    This function applies the cleaning steps used across the project:
    - Cleans bathroom and room number columns
    - Converts garage to binary
    - Creates house type mapping
    - Drops unnecessary columns
    - Handles missing values
    - Applies one-hot encoding to categorical columns

    Parameters
    ----------
    data : pd.DataFrame
        Raw data to be cleaned

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Dict[str, int]]]
        Cleaned data and house type mapping, or (None, None) if cleaning fails
    """
    try:
        print("🧹 Cleaning and preprocessing data...")
        
        # Create a copy to avoid modifying the original
        data_copy = data.copy()
        
        # Clean bath_num column
        if 'bath_num' in data_copy.columns:
            data_copy['bath_num'] = data_copy['bath_num'].replace('sin baños', '0').astype(float)
        
        # Clean room_num column
        if 'room_num' in data_copy.columns:
            data_copy['room_num'] = data_copy['room_num'].replace('sin habitación', '0').astype(float)

        # Convert garage column to binary
        if 'garage' in data_copy.columns:
            data_copy['garage'] = data_copy['garage'].notna().astype(int)

        # Create house type mapping
        house_type_mapping = {}
        if 'house_type' in data_copy.columns:
            house_type_values = data_copy['house_type'].unique()
            house_type_mapping = {value: idx for idx, value in enumerate(house_type_values)}
            data_copy['house_type'] = data_copy['house_type'].map(house_type_mapping)

        # Drop unnecessary columns
        columns_to_drop = ['ground_size', 'kitchen', 'unfurnished', 'loc_street', 'ad_description']
        columns_to_drop = [col for col in columns_to_drop if col in data_copy.columns]
        if columns_to_drop:
            data_copy = data_copy.drop(columns=columns_to_drop)
            print(f"   Dropped columns: {columns_to_drop}")

        # Handle missing values for specific numeric columns
        numeric_cols_to_fill = ['construct_date', 'm2_useful', 'lift']
        for col in numeric_cols_to_fill:
            if col in data_copy.columns and data_copy[col].dtype in [np.number]:
                before_fill = data_copy[col].isnull().sum()
                data_copy[col].fillna(data_copy[col].median(), inplace=True)
                if before_fill > 0:
                    print(f"   Filled {before_fill} missing values in {col} with median")

        # One-hot encoding for categorical columns
        categorical_columns = [col for col in ['condition', 'heating', 'orientation'] 
                              if col in data_copy.columns]
        if categorical_columns:
            data_copy = pd.get_dummies(data_copy, columns=categorical_columns)
            print(f"   Applied one-hot encoding to: {categorical_columns}")

        print(f"✅ Data cleaned successfully. Final shape: {data_copy.shape}")
        return data_copy, house_type_mapping
        
    except Exception as e:
        print(f"❌ Error cleaning data: {e}")
        return None, None


def split_data(data_cleaned: pd.DataFrame, 
               house_type_mapping: Dict[str, int]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split cleaned data into rental and sales datasets based on house_type.

    Parameters
    ----------
    data_cleaned : pd.DataFrame
        Cleaned data
    house_type_mapping : Dict[str, int]
        Mapping of house types to integer values

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
        Rental data and sales data, or (None, None) if splitting fails
    """
    try:
        # Identify codes corresponding to rental types
        alquiler_codes = [
            code for key, code in house_type_mapping.items()
            if 'alquiler' in str(key).lower()
        ]

        if alquiler_codes:
            rental_data = data_cleaned[data_cleaned['house_type'].isin(alquiler_codes)]
            sales_data = data_cleaned[~data_cleaned['house_type'].isin(alquiler_codes)]
            
            print(f"📊 Data split successfully:")
            print(f"   Rental data: {rental_data.shape[0]} records")
            print(f"   Sales data: {sales_data.shape[0]} records")
            
            return rental_data, sales_data
        else:
            print("⚠️ No rental types found in house_type mapping")
            return None, data_cleaned
            
    except Exception as e:
        print(f"❌ Error splitting data: {e}")
        return None, None


def save_data(data: pd.DataFrame, output_file_path: Union[str, Path]) -> bool:
    """
    Save data to a CSV file with error handling.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be saved
    output_file_path : Union[str, Path]
        Path to save the CSV file

    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    try:
        output_file_path = Path(output_file_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_file_path, index=False)
        print(f"💾 Data saved successfully to {output_file_path}")
        print(f"   Shape: {data.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving data to {output_file_path}: {e}")
        return False


def create_mapping_dataframe(dictionary: Dict[str, int], 
                           df_name: str = 'Mapping') -> pd.DataFrame:
    """
    Convert a dictionary to a pandas DataFrame for saving mappings.

    Parameters
    ----------
    dictionary : Dict[str, int]
        Dictionary to convert
    df_name : str, default 'Mapping'
        Name for the index of the DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame representation of the dictionary
    """
    df = pd.DataFrame(list(dictionary.items()), columns=['House_Type', 'Code'])
    df.index.name = df_name
    return df


def generate_basic_profile(df: pd.DataFrame, title: str) -> Dict:
    """
    Generate a basic profile report for a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to profile
    title : str
        Title for the report
    
    Returns
    -------
    Dict
        Dictionary with basic statistics and profile information
    """
    print(f"📈 Generating basic profile for: {title}")
    print(f"   DataFrame shape: {df.shape}")
    
    # Basic statistics
    profile = {
        "title": title,
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing_values": df.isna().sum(),
        "missing_percentage": (df.isna().sum() / len(df) * 100).round(2),
        "duplicates": df.duplicated().sum(),
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        profile["numeric_stats"] = df[numeric_cols].describe()
    
    # Categorical statistics
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        profile["categorical_stats"] = {
            col: {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            } for col in cat_cols
        }
    
    # Print summary
    print(f"\n   === PROFILE SUMMARY ===")
    print(f"   Rows: {profile['shape'][0]:,}")
    print(f"   Columns: {profile['shape'][1]}")
    print(f"   Duplicate rows: {profile['duplicates']}")
    
    print(f"\n   Missing values:")
    missing = profile['missing_values'][profile['missing_values'] > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"     - {col}: {count} ({profile['missing_percentage'][col]}%)")
    else:
        print(f"     No missing values")
    
    return profile


def harmonize_datasets(df_sales: pd.DataFrame, 
                      df_rental: pd.DataFrame,
                      output_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize sales and rental datasets for comparative analysis.
    
    Parameters
    ----------
    df_sales : pd.DataFrame
        Sales data
    df_rental : pd.DataFrame
        Rental data
    output_dir : Union[str, Path]
        Directory to save harmonized datasets
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Harmonized sales and rental datasets
    """
    print("🔄 Harmonizing datasets for export...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify common columns
    common_columns = sorted(list(set(df_sales.columns) & set(df_rental.columns)))
    print(f"   Common columns: {len(common_columns)}")
    
    # Select important columns
    rental_describe_columns = df_rental.describe().columns.tolist()
    
    # Key columns for analysis
    key_columns = ['price', 'bath_num', 'room_num', 'house_type', 'house_id', 
                  'm2_real', 'm2_useful', 'loc_city', 'loc_zone', 'construct_date']
    
    # Build selected columns list
    selected_columns = [col for col in key_columns if col in common_columns]
    
    # Add numeric columns from describe
    for col in rental_describe_columns:
        if col not in selected_columns and col in common_columns:
            selected_columns.append(col)
    
    # Add important categorical columns
    categorical_columns = []
    
    for col in categorical_columns:
        if col in common_columns and col not in selected_columns:
            selected_columns.append(col)
    
    print(f"   Selected {len(selected_columns)} columns for export")
    
    # Create harmonized datasets
    df_sales_export = df_sales[selected_columns].copy()
    df_rental_export = df_rental[selected_columns].copy()
    
    # Generate and save descriptions
    sales_describe = df_sales_export.describe(include='all')
    rental_describe = df_rental_export.describe(include='all')
    
    # Save all files
    save_data(df_sales_export, output_dir / 'alava_sales_final.csv')
    save_data(df_rental_export, output_dir / 'alava_rental_final.csv')
    
    sales_describe.to_csv(output_dir / 'alava_sales_describe.csv')
    rental_describe.to_csv(output_dir / 'alava_rental_describe.csv')
    
    print(f"✅ Harmonized datasets saved to: {output_dir}")
    print(f"   Sales: {df_sales_export.shape[0]} records, {df_sales_export.shape[1]} columns")
    print(f"   Rental: {df_rental_export.shape[0]} records, {df_rental_export.shape[1]} columns")
    
    return df_sales_export, df_rental_export 


def discover_city_files(data_dir: Union[str, Path]) -> List[str]:
    """
    Discover all available city files in the format houses_*.csv.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing the data files
        
    Returns
    -------
    List[str]
        List of city names found
    """
    data_path = Path(data_dir)
    city_files = list(data_path.glob("houses_*.csv"))
    cities = []
    
    for file in city_files:
        # Extract city name from filename houses_cityname.csv
        city_name = file.stem.replace("houses_", "")
        cities.append(city_name)
    
    print(f"🔍 Discovered {len(cities)} cities with housing data:")
    for i, city in enumerate(sorted(cities), 1):
        print(f"   {i:2d}. {city.title()}")
    
    return sorted(cities)


def process_city_data(city: str, data_dir: Union[str, Path], 
                     output_dir: Union[str, Path]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """
    Process housing data for a specific city.
    
    Parameters
    ----------
    city : str
        City name
    data_dir : Union[str, Path]
        Directory containing raw data files
    output_dir : Union[str, Path]
        Directory for processed output
        
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]
        Tuple of (sales_data, rental_data, house_type_mapping)
    """
    print(f"\n📍 Processing {city.title()} housing data...")
    
    # Construct file path
    data_path = Path(data_dir)
    city_file = data_path / f"houses_{city}.csv"
    
    if not city_file.exists():
        print(f"❌ File not found: {city_file}")
        return None, None, {}
    
    # Load raw data
    raw_data = load_data(city_file)
    if raw_data is None:
        return None, None, {}
    
    # Clean data
    cleaned_data, house_type_mapping = clean_data(raw_data)
    if cleaned_data is None:
        return None, None, {}
    
    # Split into rental and sales
    rental_data, sales_data = split_data(cleaned_data, house_type_mapping)
    
    # Create output directory for city
    city_output_dir = Path(output_dir) / city
    city_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed datasets
    if rental_data is not None and len(rental_data) > 0:
        save_data(rental_data, city_output_dir / f"{city}_rental_processed.csv")
    
    if sales_data is not None and len(sales_data) > 0:
        save_data(sales_data, city_output_dir / f"{city}_sales_processed.csv")
    
    # Save house type mapping
    if house_type_mapping:
        mapping_df = create_mapping_dataframe(house_type_mapping, f'{city.title()}_House_Type_Mapping')
        save_data(mapping_df, city_output_dir / f"{city}_type_mapping.csv")
    
    return sales_data, rental_data, house_type_mapping


def process_all_cities(data_dir: Union[str, Path], 
                      output_dir: Union[str, Path]) -> Dict[str, Dict]:
    """
    Process housing data for all available cities.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing raw data files
    output_dir : Union[str, Path]
        Directory for processed output
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary with city data: {city: {'sales': df, 'rental': df, 'mapping': dict}}
    """
    print("🏙️ PROCESSING ALL AVAILABLE CITIES")
    print("=" * 50)
    
    # Discover all available cities
    cities = discover_city_files(data_dir)
    
    if not cities:
        print("❌ No city files found!")
        return {}
    
    # Process each city
    all_city_data = {}
    successful_cities = []
    failed_cities = []
    
    for city in cities:
        try:
            sales_data, rental_data, mapping = process_city_data(city, data_dir, output_dir)
            
            all_city_data[city] = {
                'sales': sales_data,
                'rental': rental_data,
                'mapping': mapping,
                'sales_count': len(sales_data) if sales_data is not None else 0,
                'rental_count': len(rental_data) if rental_data is not None else 0
            }
            
            successful_cities.append(city)
            
        except Exception as e:
            print(f"❌ Error processing {city}: {str(e)}")
            failed_cities.append(city)
    
    # Summary report
    print(f"\n📊 PROCESSING SUMMARY")
    print("=" * 30)
    print(f"✅ Successfully processed: {len(successful_cities)} cities")
    print(f"❌ Failed to process: {len(failed_cities)} cities")
    
    if successful_cities:
        print(f"\n🏆 Successful cities:")
        for city in successful_cities:
            data = all_city_data[city]
            print(f"   📍 {city.title()}: {data['sales_count']:,} sales, {data['rental_count']:,} rentals")
    
    if failed_cities:
        print(f"\n💥 Failed cities: {', '.join(failed_cities)}")
    
    return all_city_data


def create_city_summary_report(all_city_data: Dict[str, Dict], 
                              output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Create a summary report of all processed cities.
    
    Parameters
    ----------
    all_city_data : Dict[str, Dict]
        Dictionary with processed city data
    output_dir : Union[str, Path]
        Directory for output files
        
    Returns
    -------
    pd.DataFrame
        Summary report DataFrame
    """
    print("\n📋 Creating city summary report...")
    
    summary_data = []
    for city, data in all_city_data.items():
        sales_df = data['sales']
        rental_df = data['rental']
        
        row = {
            'city': city.title(),
            'sales_count': data['sales_count'],
            'rental_count': data['rental_count'],
            'total_properties': data['sales_count'] + data['rental_count']
        }
        
        # Add price statistics if available
        if sales_df is not None and 'price' in sales_df.columns and len(sales_df) > 0:
            row.update({
                'avg_sales_price': sales_df['price'].mean(),
                'median_sales_price': sales_df['price'].median(),
                'min_sales_price': sales_df['price'].min(),
                'max_sales_price': sales_df['price'].max()
            })
        
        if rental_df is not None and 'price' in rental_df.columns and len(rental_df) > 0:
            row.update({
                'avg_rental_price': rental_df['price'].mean(),
                'median_rental_price': rental_df['price'].median(),
                'min_rental_price': rental_df['price'].min(),
                'max_rental_price': rental_df['price'].max()
            })
            
            # Calculate estimated rental yield if both prices available
            if 'avg_sales_price' in row and row['avg_sales_price'] > 0:
                annual_rental = row['avg_rental_price'] * 12
                row['estimated_rental_yield'] = (annual_rental / row['avg_sales_price']) * 100
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary report
    output_path = Path(output_dir)
    summary_file = output_path / "all_cities_summary.csv"
    save_data(summary_df, summary_file)
    
    print(f"💾 Summary report saved to: {summary_file}")
    
    return summary_df 