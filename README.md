# AI Projects Collection

This repository contains a comprehensive collection of machine learning and artificial intelligence projects, each organized in separate folders for easy management and individual GitHub repository creation.

## Project Overview

Each project is self-contained with its own dataset, code, models, and documentation, making them ready for individual GitHub repositories.

## Project List

### 1. Credit Score and Default Risk Prediction
**Folder:** `01_Credit_Score_Prediction`
- **Type:** Complete ML Project with Web App
- **Algorithms:** Regression & Classification
- **Features:** Flask web interface, model comparison, API endpoints
- **Dataset:** Financial transaction data (1000 customers, 84 features)

### 2. Diabetes Prediction
**Folder:** `02_Diabetes_Prediction`
- **Type:** Complete ML Project with Web App
- **Algorithms:** Classification & Regression
- **Features:** Health data analysis, web interface, dual prediction models
- **Dataset:** Patient health metrics and diabetes indicators

### 3. Heart Disease Classification
**Folder:** `03_Heart_Disease_Classification`
- **Type:** Medical Data Analysis
- **Algorithms:** Classification
- **Features:** Jupyter notebook analysis, medical data insights
- **Dataset:** Heart disease patient data

### 4. California Housing EDA
**Folder:** `04_California_Housing_EDA`
- **Type:** Exploratory Data Analysis
- **Algorithms:** Statistical Analysis & Visualization
- **Features:** Comprehensive EDA, geographical analysis, feature correlation
- **Dataset:** California housing market data

### 5. Concrete Strength Prediction
**Folder:** `05_Concrete_Strength_Prediction`
- **Type:** Engineering ML Application
- **Algorithms:** Regression
- **Features:** Construction industry application, material optimization
- **Dataset:** Concrete mixture components and strength measurements

### 6. Titanic Data Visualization
**Folder:** `06_Titanic_Visualization`
- **Type:** Data Visualization Project
- **Algorithms:** Statistical Analysis
- **Features:** Historical data analysis, survival pattern visualization
- **Dataset:** Titanic passenger data

### 7. Computer Vision: Cup and Pen Classifier
**Folder:** `07_Computer_Vision_Cup_Pen_Classifier`
- **Type:** Deep Learning Computer Vision
- **Algorithms:** Convolutional Neural Networks
- **Features:** Image classification, CNN implementation
- **Dataset:** Custom image dataset of cups and pens

### 8. ML Classifiers Examples
**Folder:** `08_ML_Classifiers_Examples`
- **Type:** Educational ML Examples
- **Algorithms:** Multiple Classification Algorithms
- **Features:** Algorithm comparison, hyperparameter tuning, educational content
- **Dataset:** Social media advertising data

### 9. Linear Regression Examples
**Folder:** `09_Linear_Regression_Examples`
- **Type:** Educational ML Fundamentals
- **Algorithms:** Linear Regression (Custom & Library)
- **Features:** Mathematical foundations, comparison implementations
- **Dataset:** Housing price data

## Repository Structure

Each project folder contains:
```
Project_Name/
├── README.md              # Project-specific documentation
├── main_script.py         # Primary implementation file
├── data/                  # Dataset files
├── models/                # Trained models and weights
├── web_app/              # Web interface (if applicable)
├── visualization/        # Generated plots and charts
└── requirements.txt      # Project dependencies (when applicable)
```

## Technologies Used

### Machine Learning
- **Libraries:** scikit-learn, pandas, numpy
- **Algorithms:** Linear/Logistic Regression, Random Forest, SVM, Gradient Boosting
- **Evaluation:** Cross-validation, Grid Search, Multiple metrics

### Deep Learning
- **Framework:** TensorFlow/Keras
- **Applications:** Computer Vision, Image Classification
- **Architecture:** Convolutional Neural Networks

### Web Development
- **Framework:** Flask
- **Features:** Interactive forms, API endpoints, real-time predictions
- **Frontend:** HTML, CSS, Bootstrap

### Data Visualization
- **Libraries:** Matplotlib, Seaborn, Plotly
- **Types:** Statistical plots, geographical maps, correlation matrices
- **Applications:** EDA, model evaluation, result presentation

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook (for .ipynb files)
- Required packages vary by project (see individual README files)

### General Installation
```bash
# Clone specific project or entire collection
git clone <repository-url>

# Navigate to desired project
cd Project_Name

# Install dependencies
pip install -r requirements.txt  # if available
# OR install common packages:
pip install pandas numpy scikit-learn matplotlib seaborn flask tensorflow
```

## Usage Recommendations

### For Learning
1. Start with **Linear Regression Examples** for ML fundamentals
2. Progress to **ML Classifiers Examples** for algorithm comparison
3. Explore **EDA projects** for data analysis skills
4. Advance to **complete projects** with web interfaces

### For Portfolio
- **Credit Score Prediction**: Comprehensive business application
- **Diabetes Prediction**: Healthcare ML application
- **Computer Vision**: Deep learning demonstration
- **Heart Disease Classification**: Medical data analysis

### For Practice
- Each project includes educational components
- Modify datasets and parameters to experiment
- Extend functionality and add new features
- Compare different algorithms and approaches

## Project Complexity Levels

### Beginner
- Linear Regression Examples
- ML Classifiers Examples
- Titanic Visualization

### Intermediate
- Heart Disease Classification
- California Housing EDA
- Concrete Strength Prediction

### Advanced
- Credit Score Prediction (Full Stack)
- Diabetes Prediction (Full Stack)
- Computer Vision Classifier

## Contributing

Each project can be extended and improved:
- Add new algorithms and models
- Improve web interfaces
- Enhance visualizations
- Add more comprehensive documentation
- Include additional datasets

## Future Enhancements

Potential additions to projects:
- API documentation
- Docker containerization
- Cloud deployment guides
- More advanced deep learning models
- Real-time data processing
- Interactive dashboards

## License

Each project can be licensed independently when moved to separate repositories.

---

**Note:** This collection is designed for educational purposes, portfolio development, and practical ML application learning. Each project demonstrates different aspects of the machine learning pipeline from data preprocessing to model deployment.
