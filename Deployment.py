"""# Model Deployment"""

"""# Creating the Flask API"""
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("final_house_price_model.pkl")

# Define preprocessing function
def preprocess_input(input_data):
    df = pd.DataFrame(input_data, index=[0])

    # Ensure categorical encoding is applied properly
    Type_map = {'Builder_Floor': 0, 'Apartment': 1, }
    df['Type'] = df['Type'].map(Type_map)
    
    if df['Type'].isnull().any():
       raise ValueError("Invalid value in 'Type' Column")

    return df.values

# Flask App
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()  # Get input data
        input_arr = preprocess_input(input_data)  # Preprocess input
        prediction = model.predict(input_arr)  # Make Prediction
        return jsonify({"Predicted House Price": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)