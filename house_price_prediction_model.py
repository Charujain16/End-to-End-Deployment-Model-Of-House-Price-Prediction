import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""# Part 1:- Data Preprocessing"""

df = pd.read_csv('C:\\Users\\DELL\\End-to-End-Deployment-Model-Of-House-Price-Prediction\\Delhi house data.csv')
df.head()

df.shape

df.info()

"""# Handling Missing Values"""

df.isnull().sum()

df.drop(columns=['Status'],inplace=True)  # drop the column 'Status' as it is not contibute in predictiin of house price

df.duplicated().sum()   # return count of duplicate values

df.drop_duplicates(inplace = True)  # drop the duplicate values


df[['Bathroom', 'Per_Sqft']] = df[['Bathroom', 'Per_Sqft']].fillna(df[['Bathroom', 'Per_Sqft']].mean())

df.fillna({'Parking': 0}, inplace=True)   # Replace null values with 0 as we consider as no parking facility available

df['Furnishing'].value_counts()   # Checking count of unique values in 'Furnishing' column

df.fillna({'Furnishing':'Unfurnished'}, inplace=True)

df['Type'].value_counts()

df.fillna({'Type': 'Apartment'}, inplace=True)

df.isnull().sum()

"""We are skiping Outlier Handling because every house has its own features for considering price so handling outlier may miss interpret the data.

# Feature Engineering
"""

cat_col = df.select_dtypes(include='object').columns   # Filter out Categorical columns
cat_col

df.value_counts('Transaction')

df.value_counts('Type')

"""** Performing One-Hot Encoding**
1.   'Type' :-
*   Builder_Floor  --> 1
*   Apartment --> 0

2.  'Transaction' :-
*   Resale --> 1
*   New_Property --> 0
"""
# Perform one-hot Encoding
df[['Transaction', 'Type']] = pd.get_dummies(df[['Transaction', 'Type']], drop_first=True, dtype=int)


"""** Performing Lable Encoding on Furnishing Column**
'Furnishing' :-
*   Semi-Furnished --> 0
*   Unfurnished --> 1
*   Furnished --> 2
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Furnishing'] = le.fit_transform(df['Furnishing'])


df.value_counts('Locality')
# Extract the first 100 characters from the 'Category' column
import re
df['Short_Loc'] = df['Locality'].str.slice(0, 37)

def extract_locality(text):
    # Remove unwanted keywords
    text = re.sub(r'\b(carpet area|status|transaction|Contact Owner|View Phone No\.|Share Feedback|Owner)\b', '', text, flags=re.IGNORECASE)

    # Regex pattern to capture localities (assumes locality names contain letters & numbers)
    match = re.search(r'([A-Za-z\s]+?(?:Sector \d+|Enclave|Nagar|Apartment|Colony|Phase \d+|Block \w+))', text, re.IGNORECASE)

    if match:
        return match.group(1).strip()  # Extract matched locality name

    return text.strip()  # Return original text if no match found

# Apply function to dataset
df['Cleaned_Locality'] = df['Locality'].apply(extract_locality)

# Display sample results
df[['Locality', 'Cleaned_Locality']].head()

# df.drop(columns=['Locality'], inplace=True)  # Moved to later

df.head()

# Standard Deviation Approach
locality_price_stats = df.groupby('Cleaned_Locality')['Price'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print(locality_price_stats)

# Filter Localities with Enough Listings
df_filtered = df[df['Cleaned_Locality'].map(df['Cleaned_Locality'].value_counts()) >= 1]    # Keep localities with ≥5 listings

df['Cleaned_Locality'].isnull().sum()

from sklearn.feature_selection import mutual_info_regression

locality_encoded = df['Cleaned_Locality'].astype('category').cat.codes  # Convert to numerical
mi_score = mutual_info_regression(locality_encoded.values.reshape(-1, 1), df['Price'])

print(f"Mutual Information Score for Locality: {mi_score[0]}")

print(df['Cleaned_Locality'].isnull().sum())  # Check missing values in original locality column

# df.drop(columns=['Locality'], inplace=True)  # Drop Locality now

# Now as Locality has major impact on House Price so we Replace each locality with the mean price of that locality
# locality_price_map = df.groupby('Cleaned_Locality')['Price'].mean()     # Compute mean price per Locality
# df['Locality_Encoded'] = df['Cleaned_Locality'].map(locality_price_map)      # Replace with mean Price

# df.drop(columns = ['Cleaned_Locality'], inplace = True)

df.head()

df.drop(columns='Short_Loc', inplace=True)

df.drop(columns=['Locality'], inplace=True)  # Drop Locality here after all uses

df.isnull().sum()

"""# Feature Selection

Visualize correlations between features and the target variable.
"""

# Checking relationship between variables by using Correlation heatmap
figure = plt.figure(figsize=(15,10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()

"""**As 'Parking' and 'Transaction' doesn't play a crucial role with PRedicting Target Variable, so dropping them.**"""

df.drop(columns = ['Parking', 'Furnishing', 'Transaction'], inplace=True)

"""# Part 2:-  Model Training & Evaluation"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define the Models
model_selection = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

# Split the data into feature and target
X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = []

# Target encoding using TRAIN ONLY
locality_price_map = X_train.join(y_train).groupby('Cleaned_Locality')['Price'].mean()

X_train['Locality_Encoded'] = X_train['Cleaned_Locality'].map(locality_price_map)
X_test['Locality_Encoded'] = X_test['Cleaned_Locality'].map(locality_price_map)

# Handle unseen localities
global_mean = y_train.mean()
X_test['Locality_Encoded'].fillna(global_mean, inplace=True)

X_train.drop(columns=['Cleaned_Locality'], inplace=True)
X_test.drop(columns=['Cleaned_Locality'], inplace=True)

# Train, predict, and evaluate each model in a loop
for name, model in model_selection.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Make prediction

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Append results to the list
    results.append([name, mae, rmse, r2])

# Convert results to a DataFrame
df_results = pd.DataFrame(results, columns=["Model", "Mean Absolute Error", "Root Mean Squared Error", "R² Score"])
print(df_results.to_string(index=False)) # Print the DataFrame in a readable format

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [5, 10, None],      # Depth of trees
    "min_samples_split": [2, 5, 10], # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4]    # Minimum samples in leaf
}

# Perform hyperparameter tuning for Random Forest
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=1, verbose=2, scoring='r2', refit=True)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters for Random Forest
best_rf = grid_search.best_estimator_
print("Best Parameters for Random Forest:", grid_search.best_params_)

# Train the optimized Random Forest model
rfg = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_depth=10,
                       min_samples_split=5)
rfg.fit(X_train, y_train)

# Plot Training vs Validation Score
train_scores = []
val_scores = []
estimators = [10, 50, 100, 200, 500]

for n in estimators:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)

    # Training Score
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_scores.append(train_r2)

    # Validation Score
    y_val_pred = model.predict(X_test)
    val_r2 = r2_score(y_test, y_val_pred)
    val_scores.append(val_r2)

# Plot the learning curve
plt.figure(figsize=(7,5))

plt.plot(estimators, train_scores, label="Train Score", color="blue")
plt.plot(estimators, val_scores, label="Validation Score", color="orange")
plt.xlabel("Number of Estimators")
plt.ylabel("R² Score")
plt.legend()
plt.show()

# Save the trained model using Joblib
import joblib
joblib.dump(model, "final_house_price_model.pkl")
# joblib.dump(scaler, "scaler.pkl")

print("Model finalized and saved successfully!")