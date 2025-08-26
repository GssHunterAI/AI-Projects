# Heart Disease Classification

This project implements a machine learning classification model to predict heart disease diagnosis based on patient medical data.

## Project Overview

Using various patient health metrics, this project builds and evaluates classification models to predict the presence of heart disease, helping in early diagnosis and medical decision-making.

## Dataset

The heart disease dataset contains the following features:

- Age: Age of the patient
- Sex: Gender of the patient (1 = male; 0 = female)
- CP: Chest pain type (4 values)
- Trestbps: Resting blood pressure
- Chol: Serum cholesterol in mg/dl
- FBS: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- Restecg: Resting electrocardiographic results (values 0,1,2)
- Thalach: Maximum heart rate achieved
- Exang: Exercise induced angina (1 = yes; 0 = no)
- Oldpeak: ST depression induced by exercise relative to rest
- Slope: The slope of the peak exercise ST segment
- CA: Number of major vessels (0-3) colored by flourosopy
- Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
- Target: Diagnosis of heart disease (angiographic disease status)

## Files Structure

- `Heart_disease_classification.ipynb`: Jupyter notebook with complete analysis
- `data/heart.csv`: Heart disease dataset

## Analysis Includes

- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Feature correlation analysis
- Multiple classification algorithms comparison
- Model evaluation and performance metrics
- Visualization of results

## Classification Models

The notebook compares various classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- And more...

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Launch Jupyter Notebook:
   ```
   jupyter notebook Heart_disease_classification.ipynb
   ```

## Usage

1. Open the Jupyter notebook
2. Run cells sequentially to see the complete analysis
3. Explore different classification models and their performance
4. Analyze feature importance and correlations
5. Review confusion matrices and classification reports
