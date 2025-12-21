"""Model Deployment"""

from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load trained model
model = joblib.load('final_house_price_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle form submission
        try:
            area = float(request.form['area'])
            bathroom = float(request.form['bathroom'])
            per_sqft = float(request.form['per_sqft'])
            type_encoded = int(request.form['type'])
            furnishing = int(request.form['furnishing'])
            locality_encoded = float(request.form['locality_encoded'])
            
            input_data = {
                'Area': area,
                'Bathroom': bathroom,
                'Per_Sqft': per_sqft,
                'Type': type_encoded,
                'Furnishing': furnishing,
                'Locality_Encoded': locality_encoded
            }
            input_arr = preprocess_input(input_data)  # Note: preprocess expects list of dicts? Wait, adjust
            prediction = model.predict(input_arr)
            result = f"Predicted House Price: ₹{round(float(prediction[0]), 2)}"
        except Exception as e:
            result = f"Error: {str(e)}"
        
        return f'''
        <h1>House Price Prediction</h1>
        <p>{result}</p>
        <a href="/">Predict Again</a>
        '''
    
    return '''
    <h1>House Price Prediction</h1>
    <form method="post">
        Area (sqft): <input type="number" name="area" required><br>
        Bathroom: <input type="number" name="bathroom" required><br>
        Per Sqft: <input type="number" name="per_sqft" required><br>
        Type (0=Apartment, 1=Builder_Floor): <input type="number" name="type" min="0" max="1" required><br>
        Furnishing (0=Unfurnished, 1=Semi-Furnished, 2=Furnished): <input type="number" name="furnishing" min="0" max="2" required><br>
        Locality Encoded (mean price of locality): <input type="number" name="locality_encoded" required><br>
        <input type="submit" value="Predict">
    </form>
    <p>Note: Locality Encoded is the average price for the area. For demo, use 5000000.</p>
    '''

# define Prerocessing function
def preprocess_input(input_data):
    keys = ['Area', 'Bathroom', 'Per_Sqft', 'Type', 'Furnishing', 'Locality_Encoded']
    input_arr = np.array([input_data[key] for key in keys]).reshape(1, -1)
    return input_arr

# Define Prediction route
@app.route('/predict', methods = ['POST'])
def predict():
  try:
    input_data = request.get_json()    # Get input data as json
    input_arr = preprocess_input(input_data)   # preprocess the input
    prediction = model.predict(input_arr)  #    Get Prediction
    return jsonify({"Predicted House Price": round(float(prediction[0]), 2)})
  except Exception as e:
    return jsonify({"Error":str(e)})

# run Flask App
if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000, debug=True)