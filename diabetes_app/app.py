from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the models
models_path = 'd:/AI Projects/models/diabetes_model/'
classification_model = pickle.load(open(os.path.join(models_path, 'best_class_model.pkl'), 'rb'))
regression_model = pickle.load(open(os.path.join(models_path, 'best_reg_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(models_path, 'scaler.pkl'), 'rb'))
metadata = pickle.load(open(os.path.join(models_path, 'model_metadata.pkl'), 'rb'))

# Get feature names from metadata
features = metadata['features']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    input_data = []
    for feature in features:
        value = request.form.get(feature)
        input_data.append(float(value) if value else 0.0)
    
    # Convert to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input
    scaled_input = scaler.transform(input_array)
    
    # Make predictions
    classification_prediction = classification_model.predict(scaled_input)[0]
    regression_prediction = regression_model.predict(scaled_input)[0]
    
    # Interpret classification result
    diabetes_status = "Diabetic" if classification_prediction == 1 else "Non-Diabetic"
    
    # Create result dictionary
    result = {
        'classification': diabetes_status,
        'regression': round(regression_prediction, 3),
        'class_model': metadata['classification']['best_model'],
        'reg_model': metadata['regression']['best_model']
    }
    
    return render_template('result.html', result=result, input_features=dict(zip(features, input_data)))

if __name__ == '__main__':
    app.run(debug=True)