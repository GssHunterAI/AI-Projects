# Credit Score and Default Risk Prediction

This project implements machine learning models to predict customer credit scores and loan default risk based on financial transaction data and current financial standing.

## Project Overview

The system uses multiple regression models to predict numerical credit scores and classification models to assess the probability of loan default, helping financial institutions make better lending decisions.

### Key Features

- Credit score prediction (regression model)
- Default risk assessment (classification model)
- Web interface for easy data input and visualization
- Comprehensive model evaluation with visualizations
- API endpoint for integration with other systems

## Dataset

The dataset comprises information on 1000 customers with 84 features derived from their financial transactions and current financial standing, including:

- Income, savings, and debt information
- Transaction history across 11 spending categories
- Financial ratios (savings-to-income, debt-to-income, etc.)
- Categorical features (gambling category, credit card ownership, etc.)

## Models

The project evaluates and compares multiple model types:

### Regression Models (Credit Score Prediction)
- Linear Regression
- Ridge Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

### Classification Models (Default Risk Prediction)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

## Files Structure

- `iscore_model.py`: Main model training, evaluation and visualization script
- `web_app/`: Flask web application for making predictions
- `models/`: Saved trained models and metadata
- `data/`: Dataset files
- `visualization/`: Generated plots and charts

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages: pandas, numpy, scikit-learn, flask, matplotlib, seaborn

### Installation

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn flask matplotlib seaborn
   ```

2. Train and save models:
   ```
   python iscore_model.py
   ```

3. Run the web application:
   ```
   cd web_app
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Enter customer financial information in the web form
2. The application will automatically calculate financial ratios
3. Submit the form to get predictions
4. View the predicted credit score and default risk assessment
5. Review loan eligibility recommendations
