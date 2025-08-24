"""
Utilities for notebooks - redirects to main src package.
This maintains backward compatibility while using the proper source structure.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import from the main package
from src.data.exploratory_analysis import *  # noqa

__all__ = [
    "analyze_dataset",
    "get_basic_stats", 
    "load_dataset",
    "plot_categorical_distributions",
    "plot_correlation_matrix",
    "plot_numeric_distributions",
    "DataLoader",
    "DataAnalyzer", 
    "DataVisualizer",
] 