# End-to-End-Deployment-Model-Of-House-Price-Prediction
### House Price Prediction Model 🏡  

This project aims to predict house prices in **Delhi** using **machine learning techniques**. The dataset includes various features like **locality, furnishing status, number of bathrooms, per square foot price, and transaction type**.  

#### 🔹 Key Features:  
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature engineering.  
- **Feature Selection**: Correlation heatmap and mutual information analysis.  
- **Model Training**: Using **Linear Regression, Random Forest, and XGBoost** to find the best-performing model.  
- **Hyperparameter Tuning**: Optimizing Random Forest using **GridSearchCV**.  
- **Deployment**: Flask-based API for real-time predictions.  

#### 📊 Visualization:  
- **Heatmaps**, **bar charts**, and **learning curves** to analyze data and model performance.  

🔗 **Check out the detailed report for in-depth insights!** 🚀


**Instruction to running the project**
#### 🔹Repository Structure:  

├── data/                        # Dataset files
│   ├── Delhi_house_price.csv
├── models/                      # Saved trained models
│   ├── final_house_price_model.pkl
├── src/                   # # Source code file for exploration & training
│   ├── House_Price_Prediction.py
├── app.py                        # Flask API script
├── requirements.txt              # Required dependencies
├── README.md                     # Project documentation (this file)

#### 🔹Setup Instructions:
1. **Clone the Repository & Install Dependencies**
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   pip install -r requirements.txt
   
2. **Run the Flask API**
   API runs at http://127.0.0.1:5000/
   If Flask API not starting?--> Ensure port 5000 is available.

4. **Make a Prediction**
  Send a POST request to /predict with JSON input:


