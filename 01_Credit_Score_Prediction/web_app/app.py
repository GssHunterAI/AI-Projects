import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the saved models and components
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'credit_score')

# Load the model metadata
metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
feature_names = metadata['feature_names']
reg_model_name = metadata['best_reg_model_name']
class_model_name = metadata['best_class_model_name']
reg_needs_scaling = metadata['reg_needs_scaling']
class_needs_scaling = metadata['class_needs_scaling']

# Load the regression model (for credit score prediction)
reg_model = joblib.load(os.path.join(MODEL_DIR, 'best_reg_model.pkl'))

# Load the classification model (for default prediction)
class_model = joblib.load(os.path.join(MODEL_DIR, 'best_class_model.pkl'))

# Load the scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# Define key features that will be collected from the web form
# Note: These should be a subset of the most important features for prediction
key_features = [
    'INCOME', 'SAVINGS', 'DEBT', 'R_SAVINGS_INCOME', 'R_DEBT_INCOME', 'R_DEBT_SAVINGS',
    'T_GROCERIES_12', 'T_HOUSING_12', 'T_ENTERTAINMENT_12', 'T_GAMBLING_12', 
    'CAT_GAMBLING_low', 'CAT_GAMBLING_none', 'CAT_DEBT', 'CAT_CREDIT_CARD', 
    'CAT_MORTGAGE', 'CAT_SAVINGS_ACCOUNT', 'CAT_DEPENDENTS'
]

def predict_credit_and_default(input_data):

    df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Update the DataFrame with the provided features
    for feature in input_data:
        if feature in df.columns:
            df[feature] = input_data[feature]
    
    # Apply scaling if needed
    df_scaled = scaler.transform(df) if (reg_needs_scaling or class_needs_scaling) else None
    
    # Credit score prediction
    if reg_needs_scaling:
        credit_score = reg_model.predict(df_scaled)[0]
    else:
        credit_score = reg_model.predict(df)[0]
    
    # Default risk prediction
    if class_needs_scaling:
        default_risk = class_model.predict(df_scaled)[0]
        default_prob = class_model.predict_proba(df_scaled)[0][1] if hasattr(class_model, 'predict_proba') else None
    else:
        default_risk = class_model.predict(df)[0]
        default_prob = class_model.predict_proba(df)[0][1] if hasattr(class_model, 'predict_proba') else None
    
    return int(round(credit_score)), int(default_risk), default_prob

@app.route('/')
def home():
    return render_template('index.html', key_features=key_features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {}
    
    # Extract numerical features
    numerical_features = [
        'INCOME', 'SAVINGS', 'DEBT', 'R_SAVINGS_INCOME', 'R_DEBT_INCOME', 'R_DEBT_SAVINGS',
        'T_GROCERIES_12', 'T_HOUSING_12', 'T_ENTERTAINMENT_12', 'T_GAMBLING_12'
    ]
    for feature in numerical_features:
        if feature in request.form:
            input_data[feature] = float(request.form.get(feature, 0))
    
    # Extract categorical features
    cat_features = {
        'CAT_GAMBLING': request.form.get('CAT_GAMBLING', 'none'),
        'CAT_DEBT': request.form.get('CAT_DEBT', '0'),
        'CAT_CREDIT_CARD': request.form.get('CAT_CREDIT_CARD', '0'),
        'CAT_MORTGAGE': request.form.get('CAT_MORTGAGE', '0'),
        'CAT_SAVINGS_ACCOUNT': request.form.get('CAT_SAVINGS_ACCOUNT', '0'),
        'CAT_DEPENDENTS': request.form.get('CAT_DEPENDENTS', '0')
    }
    
    # Convert categorical features to one-hot encoding
    if cat_features['CAT_GAMBLING'] == 'low':
        input_data['CAT_GAMBLING_low'] = 1
        input_data['CAT_GAMBLING_none'] = 0
    elif cat_features['CAT_GAMBLING'] == 'none':
        input_data['CAT_GAMBLING_low'] = 0
        input_data['CAT_GAMBLING_none'] = 1
    else:  # high
        input_data['CAT_GAMBLING_low'] = 0
        input_data['CAT_GAMBLING_none'] = 0
    
    # Other binary categorical features
    for feature, value in cat_features.items():
        if feature != 'CAT_GAMBLING':
            input_data[feature] = int(value)
    
    # Make prediction
    credit_score, default_risk, default_prob = predict_credit_and_default(input_data)
    
    # Return results
    return render_template('result.html', 
                          credit_score=credit_score, 
                          default_risk=default_risk, 
                          default_prob=default_prob)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    # Get JSON data
    input_data = request.json
    
    # Make prediction
    credit_score, default_risk, default_prob = predict_credit_and_default(input_data)
    
    # Return results as JSON
    return jsonify({
        'credit_score': credit_score,
        'default_risk': int(default_risk),
        'default_probability': float(default_prob) if default_prob is not None else None
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
    
    app.run(debug=True)