# Concrete Strength Prediction

This project implements machine learning models to predict concrete compressive strength based on the concrete mixture components.

## Project Overview

The project uses regression models to predict concrete compressive strength, which is crucial for construction and civil engineering applications. The models help determine optimal concrete mixtures for desired strength characteristics.

## Dataset

The concrete dataset contains information about concrete mixtures and their resulting compressive strength:

- **Cement (kg/m³)**: Amount of cement in the mixture
- **Blast Furnace Slag (kg/m³)**: Amount of blast furnace slag
- **Fly Ash (kg/m³)**: Amount of fly ash
- **Water (kg/m³)**: Amount of water in the mixture
- **Superplasticizer (kg/m³)**: Amount of superplasticizer
- **Coarse Aggregate (kg/m³)**: Amount of coarse aggregate
- **Fine Aggregate (kg/m³)**: Amount of fine aggregate
- **Age (days)**: Age of the concrete when tested
- **Concrete Compressive Strength (MPa)**: Target variable - compressive strength

## Models

The project implements and compares:
- **Linear Regression**: Basic linear relationship modeling
- **Random Forest Regressor**: Ensemble method for improved accuracy

## Files Structure

- `Concrete_Model.py`: Main script for model training and evaluation
- `data/Concrete_Data.xlsx`: Concrete mixture dataset

## Key Features

- Data preprocessing with StandardScaler normalization
- Model comparison between Linear Regression and Random Forest
- Performance evaluation using R² score and MSE
- Visualization of predictions vs actual values
- Model performance comparison plots

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, matplotlib, openpyxl

### Installation

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib openpyxl
   ```

2. Run the model:
   ```
   python Concrete_Model.py
   ```

## Results

The script provides:
- R² scores for both models
- Mean Squared Error comparisons
- Visualization comparing Linear Regression vs Random Forest predictions
- Performance metrics to determine the best model

## Usage

1. Run the Python script to train both models
2. View performance metrics in the console output
3. Analyze the generated plots showing prediction accuracy
4. Use the trained models to predict concrete strength for new mixtures

## Applications

This model can be used for:
- Optimizing concrete mixture designs
- Quality control in concrete production
- Predicting concrete strength for construction planning
- Research and development of new concrete formulations
