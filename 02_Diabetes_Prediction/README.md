# Diabetes Prediction Project

This project implements machine learning models to predict diabetes diagnosis and diabetes pedigree function using patient health data.

## Project Overview

The system uses both classification models for binary diabetes prediction and regression models for predicting diabetes pedigree function values.

### Key Features

- Binary diabetes classification (diabetic/non-diabetic)
- Diabetes pedigree function prediction (regression)
- Web interface for patient data input
- Model comparison and evaluation
- Visualization of model performance

## Dataset

The diabetes dataset contains health information for patients including:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1) indicating diabetes diagnosis

## Models

### Classification Models (Diabetes Diagnosis)
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

### Regression Models (Diabetes Pedigree Prediction)
- Linear Regression
- Random Forest Regressor
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor

## Files Structure

- `diabetes_ml_model.py`: Main model training and evaluation script
- `web_app/`: Flask web application for predictions
- `models/`: Saved trained models, scaler, and metadata
- `data/`: Dataset files
- `visualization/`: Performance comparison plots

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, flask, matplotlib

### Installation

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn flask matplotlib
   ```

2. Train models:
   ```
   python diabetes_ml_model.py
   ```

3. Run web application:
   ```
   cd web_app
   python app.py
   ```

## Usage

1. Enter patient health data in the web form
2. Get diabetes classification prediction
3. View diabetes pedigree function prediction
4. Review model confidence and recommendations
