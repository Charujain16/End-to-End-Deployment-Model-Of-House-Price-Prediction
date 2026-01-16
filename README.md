# India House Price Prediction

Predict house prices across Indian cities using machine learning and Streamlit.

## Project Overview
This project uses a Random Forest model and various preprocessing techniques to predict house prices based on property features, location, and amenities. The app is built with Streamlit for an interactive user interface.

## Features
- Interactive UI for inputting property details
- Location selection (State, City, Locality)
- Amenities and property features
- Real-time price prediction
- Data preprocessing (encoding, scaling)
- Model comparison (Baseline, Linear Regression, Random Forest)

## File Structure
- `app.py` — Streamlit web app for prediction
- `house_price_prediction_model.py` — Model training and preprocessing (notebook)
- `india_housing_prices.csv` — Dataset
- `Indian_house_price_prediction_model.ipynb` — Full EDA, feature engineering, model training
- `requirements.txt` — Python dependencies

## Setup Instructions
1. Clone the repository and navigate to the project folder.
                        git clone https://github.com/your-username/house-price-prediction.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser.
2. Select location, property details, and amenities.
3. Click "Predict Price" to get the estimated house price.

## Model & Preprocessing
- Random Forest model trained on Indian housing data
- Ordinal encoding for low-cardinality categorical features
- Target mean encoding for high-cardinality categorical features
- Robust scaling for numerical features
- Model and preprocessing objects saved as `.pkl` files

## Results
Model performance is compared using MAE, RMSE, and R² metrics. Random Forest generally outperforms baseline and linear models.

## Visualization
- EDA plots for feature distributions and correlations
- Residual analysis for model validation

## Contributing
Pull requests and suggestions are welcome!

## License
This project is for educational purposes.
# End-to-End-Deployment-Model-Of-House-Price-Prediction
### House Price Prediction Model 🏡  

