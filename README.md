# 🏘️ Real Estate Market Forecasting in Spanish Cities

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Forecasting del Mercado Inmobiliario en Ciudades Españolas con Big Data: Aplicación de Algoritmos de Machine Learning y Modelos de Series Temporales**

A comprehensive data science project that applies machine learning algorithms and time series models to forecast real estate markets in Spanish cities using Big Data techniques.

## 🎯 Project Overview

This project aims to develop predictive models for real estate market trends across Spanish cities by combining multiple data sources and applying state-of-the-art machine learning and time series forecasting techniques.

### Key Features

- **Multi-source Data Integration**: INE, MITMA, and market data from Kaggle
- **Advanced Analytics**: Polars for efficient data processing
- **Interactive Visualizations**: Plotly-based exploratory analysis
- **Machine Learning Models**: Comprehensive forecasting models
- **Time Series Analysis**: Specialized techniques for temporal data

## 🗂️ Project Structure

```
real-estate-forecasting-spain/
├── 📁 config/                 # Configuration files
│   └── environment.yml        # Conda environment specification
├── 📁 data/                   # Data directory (DVC tracked)
│   ├── raw/                   # Original data sources
│   ├── processed/             # Cleaned and processed data
│   └── final/                 # Model-ready datasets
├── 📁 notebooks/              # Jupyter notebooks for analysis
│   ├── 01a_ine_exploration.ipynb
│   ├── 01b_mitma_exploration.ipynb
│   ├── 01c_kaggle_exploration.ipynb
│   └── utils/                 # Notebook utilities
├── 📁 reports/                # Generated reports and figures
│   └── figures/               # Visualization outputs
├── 📁 src/                    # Source code package
│   ├── data/                  # Data processing modules
│   ├── features/              # Feature engineering
│   ├── models/                # ML models and training
│   └── visualization/         # Plotting utilities
├── 📁 tests/                  # Unit tests
├── 📄 pyproject.toml          # Project configuration
├── 📄 Makefile                # Automation commands
└── 📄 README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git with DVC support

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/real-estate-forecasting-spain.git
   cd real-estate-forecasting-spain
   ```

2. **Set up the development environment**
   ```bash
   make setup-env
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   ```

4. **Download data** (if available via DVC)
   ```bash
   make data-download
   ```

### Usage

#### Exploratory Data Analysis

Run the comprehensive data exploration notebooks:

```bash
# Start Jupyter Lab
jupyter lab

# Or execute specific notebooks
jupyter nbconvert --execute notebooks/01a_ine_exploration.ipynb
```

#### Data Processing Pipeline

```bash
# Process raw data
make data-process
```

#### Model Training

```bash
# Train models with MLflow tracking
make train

# Evaluate model performance
make evaluate

# View experiments in MLflow UI
make mlflow-ui
```

## 📊 Data Sources

### INE (Instituto Nacional de Estadística)
- Official Spanish statistical data
- Demographics and economic indicators
- Regional and municipal level data

### MITMA (Ministerio de Transportes, Movilidad y Agenda Urbana)
- Transportation and urban mobility data
- Infrastructure development indicators
- Regional connectivity metrics

### Kaggle Housing Data
- Real estate listings and prices
- Property characteristics and features
- Historical market trends

## 🔬 Methodology

### 1. Data Exploration and Quality Assessment
- Comprehensive EDA using Polars for performance
- Interactive visualizations with Plotly
- Data quality profiling and validation

### 2. Feature Engineering
- Domain-specific feature creation
- Temporal feature extraction
- Geospatial feature engineering

### 3. Model Development
- **Classical ML**: Random Forest, XGBoost, LightGBM
- **Time Series**: ARIMA, Prophet, Neural Networks
- **Deep Learning**: LSTM, Transformer models

### 4. Model Evaluation and Selection
- Cross-validation strategies
- Multiple evaluation metrics
- Model interpretability analysis

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

## 📊 Data Sources

### INE (Instituto Nacional de Estadística)
- Official Spanish statistical data
- Demographics and economic indicators
- Regional and municipal level data

### MITMA (Ministerio de Transportes, Movilidad y Agenda Urbana)
- Transportation and urban mobility data
- Infrastructure development indicators
- Regional connectivity metrics

### Kaggle Housing Data
- Real estate listings and prices
- Property characteristics and features
- Historical market trends

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: INE, MITMA, Kaggle community
- **Institutions**: Master en Big Data y Data Science
- **Open Source Community**: All the amazing libraries that make this possible

