import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score

#load data
df = pd.read_csv("D:\AI Projects\data\credit_score.csv")

# For regression (credit score prediction)
X = df.drop(['CREDIT_SCORE', 'DEFAULT', 'CUST_ID'], axis=1, errors='ignore')
y_score = df['CREDIT_SCORE']
y_default = df['DEFAULT']  # Use actual DEFAULT column for loan eligibility

# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data (use same split for both regression and classification)
X_train, X_test, y_score_train, y_score_test, y_default_train, y_default_test = train_test_split(
    X, y_score, y_default, test_size=0.2, random_state=42, stratify=y_default)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionaries to store model performances
regression_performances = {}
classification_performances = {}

# Function to evaluate regression models
def evaluate_regression_model(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    regression_performances[name] = {'RMSE': rmse, 'R2': r2}
    return model, rmse, r2

# Function to evaluate classification models
def evaluate_classification_model(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_performances[name] = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1': report['weighted avg']['f1-score']
    }
    return model, accuracy, auc

# Evaluate regression models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

trained_reg_models = {}
for name, model in reg_models.items():
    if name in ['Random Forest', 'Gradient Boosting']:
        trained_model, _, _ = evaluate_regression_model(model, name, X_train, X_test, y_score_train, y_score_test)
    else:
        trained_model, _, _ = evaluate_regression_model(model, name, X_train_scaled, X_test_scaled, y_score_train, y_score_test)
    trained_reg_models[name] = trained_model

# Evaluate classification models for DEFAULT prediction
class_models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

trained_class_models = {}
for name, model in class_models.items():
    if name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
        trained_model, _, _ = evaluate_classification_model(model, name, X_train, X_test, y_default_train, y_default_test)
    else:
        trained_model, _, _ = evaluate_classification_model(model, name, X_train_scaled, X_test_scaled, y_default_train, y_default_test)
    trained_class_models[name] = trained_model

# Display results
print("Credit Score Prediction Models (Regression):")
reg_df = pd.DataFrame(regression_performances).T.sort_values('RMSE')
print(reg_df)

print("\nLoan Default Prediction Models (Classification):")
class_df = pd.DataFrame(classification_performances).T.sort_values('AUC', ascending=False)
print(class_df)

# Show confusion matrix for best classification model
best_class_name = class_df.index[0]
best_class_model = trained_class_models[best_class_name]
if best_class_name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
    y_pred = best_class_model.predict(X_test)
else:
    y_pred = best_class_model.predict(X_test_scaled)

print(f"\nConfusion Matrix for {best_class_name}:")
cm = confusion_matrix(y_default_test, y_pred)
print(cm)

# Function to make combined predictions (credit score and loan eligibility)
def predict_credit_and_default_risk(features, regression_model, classification_model, reg_needs_scaling=False, class_needs_scaling=False, scaler=None):
 
    features_scaled = scaler.transform(features) if scaler is not None else None
    
    # Credit score prediction
    if reg_needs_scaling and features_scaled is not None:
        credit_score = regression_model.predict(features_scaled)
    else:
        credit_score = regression_model.predict(features)
    
    # Default risk prediction
    if class_needs_scaling and features_scaled is not None:
        default_risk = classification_model.predict(features_scaled)
        default_prob = classification_model.predict_proba(features_scaled)[:, 1] if hasattr(classification_model, 'predict_proba') else None
    else:
        default_risk = classification_model.predict(features)
        default_prob = classification_model.predict_proba(features)[:, 1] if hasattr(classification_model, 'predict_proba') else None
    
    return credit_score, default_risk, default_prob

# Example usage:
# Find best models from results
best_reg_model_name = reg_df.index[0]
best_class_model_name = class_df.index[0]

best_reg_model = trained_reg_models[best_reg_model_name]
best_class_model = trained_class_models[best_class_model_name]

# Check if models need scaling
reg_needs_scaling = best_reg_model_name not in ['Random Forest', 'Gradient Boosting']
class_needs_scaling = best_class_model_name not in ['Random Forest Classifier', 'Gradient Boosting Classifier']

# Add after your existing functions and before the result display:

# --------- VISUALIZATION FUNCTIONS ---------

def plot_regression_results(regression_performances, X_test, y_score_test, trained_reg_models, scaler=None):
    """Plot regression model comparison and predictions vs actual."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Performance Metrics Comparison
    plt.subplot(2, 2, 1)
    reg_df = pd.DataFrame(regression_performances).T
    reg_df['RMSE'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('RMSE by Model (Lower is Better)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.subplot(2, 2, 2)
    reg_df['R2'].sort_values(ascending=False).plot(kind='barh', color='lightgreen')
    plt.title('R² Score by Model (Higher is Better)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Plot 2: Actual vs Predicted for Best Model
    best_model_name = reg_df.sort_values('RMSE').index[0]
    best_model = trained_reg_models[best_model_name]
    
    plt.subplot(2, 2, 3)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        y_pred = best_model.predict(X_test)
    else:
        y_pred = best_model.predict(X_test_scaled)
    
    plt.scatter(y_score_test, y_pred, alpha=0.5)
    plt.plot([y_score_test.min(), y_score_test.max()], 
             [y_score_test.min(), y_score_test.max()], 
             'r--')
    plt.xlabel('Actual Credit Score')
    plt.ylabel('Predicted Credit Score')
    plt.title(f'Actual vs Predicted: {best_model_name}')
    
    # Plot 3: Residuals
    plt.subplot(2, 2, 4)
    residuals = y_score_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Credit Score')
    plt.ylabel('Residuals')
    plt.title('Residual Plot - (Errors)')
    
    plt.tight_layout()
    plt.savefig('regression_models_comparison.png', dpi=300)
    plt.show()
    
    # Plot 4: Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        plt.figure(figsize=(10, 8))
        feature_importances = best_model.feature_importances_
        feature_names = X_test.columns
        importances = pd.Series(feature_importances, index=feature_names)
        importances = importances.sort_values(ascending=False)
        importances[:15].plot(kind='barh', color='lightgreen')  # Top 15 features
        plt.title(f'Feature Importance: {best_model_name}')
        plt.tight_layout()
        plt.savefig('regression_feature_importance.png', dpi=300)
        plt.show()

def plot_classification_results(classification_performances, X_test, y_default_test, trained_class_models, scaler=None):
    """Plot classification model comparison, ROC curves and confusion matrix."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Performance Metrics Comparison
    plt.subplot(2, 2, 1)
    class_df = pd.DataFrame(classification_performances).T
    metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
    ax = class_df[metrics].plot(kind='bar', figsize=(15, 6), ax=plt.gca())
    plt.title('Classification Metrics by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot 2: ROC Curves for All Models
    plt.subplot(2, 2, 2)
    for name, model in trained_class_models.items():
        if name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_default_test, y_prob)
        auc = roc_auc_score(y_default_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    # Plot 3: Confusion Matrix for Best Model
    plt.subplot(2, 2, 3)
    best_model_name = class_df.sort_values('AUC', ascending=False).index[0]
    best_model = trained_class_models[best_model_name]
    
    if best_model_name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
        y_pred = best_model.predict(X_test)
    else:
        y_pred = best_model.predict(X_test_scaled)
        
    cm = confusion_matrix(y_default_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {best_model_name}')
    
    # Plot 4: Precision-Recall Curve
    plt.subplot(2, 2, 4)
    for name, model in trained_class_models.items():
        if name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_default_test, y_prob)
        plt.plot(recall, precision, lw=2, label=f'{name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig('classification_models_comparison.png', dpi=300)
    plt.show()
    
    # Plot 5: Feature importance for tree-based models
    if best_model_name in ['Random Forest Classifier', 'Gradient Boosting Classifier']:
        plt.figure(figsize=(10, 8))
        feature_importances = best_model.feature_importances_
        feature_names = X_test.columns
        importances = pd.Series(feature_importances, index=feature_names)
        importances = importances.sort_values(ascending=False)
        importances[:15].plot(kind='barh', color='skyblue')  # Top 15 features
        plt.title(f'Feature Importance: {best_model_name}')
        plt.tight_layout()
        plt.savefig('classification_feature_importance.png', dpi=300)
        plt.show()

# Also add a combined visualization for the default probability threshold
def plot_default_threshold_optimization(classification_model, X_test, y_default_test, needs_scaling=False, scaler=None):
    """Plot the effect of different probability thresholds on precision, recall and F1."""
    if needs_scaling:
        y_prob = classification_model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        y_prob = classification_model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions, recalls, f1_scores, accuracies = [], [], [], []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precisions.append(precision_score(y_default_test, y_pred))
        recalls.append(recall_score(y_default_test, y_pred))
        f1_scores.append(f1_score(y_default_test, y_pred))
        accuracies.append(accuracy_score(y_default_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, accuracies, 'y-', label='Accuracy')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics at Different Probability Thresholds')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_optimization.png', dpi=300)
    plt.show()

# Add this after the other visualizations
print("\nAnalyzing probability threshold impact...")
best_class_needs_scaling = best_class_model_name not in ['Random Forest Classifier', 'Gradient Boosting Classifier']
plot_default_threshold_optimization(best_class_model, X_test, y_default_test, best_class_needs_scaling, scaler)

# Save the best models and scaler for later use with our web application
print("\nSaving the best models and necessary components...")
import joblib
import os

# Create a directory to store the models if it doesn't exist
os.makedirs('models/credit_score', exist_ok=True)

# Save the best regression model
joblib.dump(best_reg_model, 'models/credit_score/best_reg_model.pkl')

# Save the best classification model
joblib.dump(best_class_model, 'models/credit_score/best_class_model.pkl')

# Save the scaler
joblib.dump(scaler, 'models/credit_score/scaler.pkl')

# Save model metadata (for model loading)
model_metadata = {
    'best_reg_model_name': best_reg_model_name,
    'best_class_model_name': best_class_model_name,
    'reg_needs_scaling': reg_needs_scaling,
    'class_needs_scaling': class_needs_scaling,
    'feature_names': list(X.columns),
}
joblib.dump(model_metadata, 'models/credit_score/model_metadata.pkl')

print(f"Models successfully saved to 'models/credit_score/' directory")
print(f"Best regression model: {best_reg_model_name}")
print(f"Best classification model: {best_class_model_name}")
