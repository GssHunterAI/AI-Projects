import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
import os

# Create directories for saving models and visualizations if they don't exist
os.makedirs('d:/AI Projects/models/diabetes_model', exist_ok=True)
os.makedirs('d:/AI Projects/visualization/diabetes', exist_ok=True)

# Load the diabetes dataset
data = pd.read_csv('D:/AI Projects/data/diabetes.csv')

# Preprocess the data
X = data.drop('Diabetic', axis=1)
y_classification = data['Diabetic']
y_regression = data['DiabetesPedigree']

# Split the data
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
with open('d:/AI Projects/models/diabetes_model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Classification models
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

best_class_model = None
best_class_score = 0

class_results = {}
for name, model in classifiers.items():
    model.fit(X_train, y_train_class)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test_class, predictions)
    class_results[name] = score
    print(f'{name} Accuracy: {score}')
    if score > best_class_score:
        best_class_score = score
        best_class_model = model

# Plot classification results
plt.figure(figsize=(10, 6))
plt.bar(class_results.keys(), class_results.values(), color='skyblue')
plt.title('Classification Model Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('d:/AI Projects/visualization/diabetes/classification_models_comparison.png')
plt.close()

# Save the best classification model
with open('d:/AI Projects/models/diabetes_model/best_class_model.pkl', 'wb') as f:
    pickle.dump(best_class_model, f)

# Regression models
regressors = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Gradient Boosting': GradientBoostingRegressor()
}

best_reg_model = None
best_reg_score = float('inf')

reg_results = {}
for name, model in regressors.items():
    model.fit(X_train, y_train_reg)
    predictions = model.predict(X_test)
    score = root_mean_squared_error(y_test_reg, predictions)
    reg_results[name] = score
    print(f'{name} RMSE: {score}')
    if score < best_reg_score:
        best_reg_score = score
        best_reg_model = model

# Plot regression results
plt.figure(figsize=(10, 6))
plt.bar(reg_results.keys(), reg_results.values(), color='lightcoral')
plt.title('Regression Model Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('d:/AI Projects/visualization/diabetes/regression_models_comparison.png')
plt.close()

# Save the best regression model
with open('d:/AI Projects/models/diabetes_model/best_reg_model.pkl', 'wb') as f:
    pickle.dump(best_reg_model, f)

# Save model metadata
metadata = {
    'classification': {
        'best_model': type(best_class_model).__name__,
        'accuracy': best_class_score,
        'all_models': class_results
    },
    'regression': {
        'best_model': type(best_reg_model).__name__,
        'rmse': best_reg_score,
        'all_models': reg_results
    },
    'features': list(X.columns),
    'date_created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('d:/AI Projects/models/diabetes_model/model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print('Best classification model saved with accuracy:', best_class_score)
print('Best regression model saved with RMSE:', best_reg_score)
print(f"Models saved in 'd:/AI Projects/models/diabetes_model/'")
print(f"Visualization plots saved in 'd:/AI Projects/visualization/diabetes/'")