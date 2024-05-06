import csv
import sqlite3

# Load data from CSV
with open('train.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Assuming the first row contains column names
    data = [row for row in reader]

# Create SQLite database and table
conn = sqlite3.connect('house_prices.db')
cursor = conn.cursor()

# Create table
create_table_query = """
CREATE TABLE IF NOT EXISTS house_prices (
    Id INTEGER PRIMARY KEY,
    MSSubClass INTEGER,
    MSZoning TEXT,
    LotFrontage REAL,
    LotArea REAL,
    -- Add other columns here
    SalePrice REAL
);
"""
cursor.execute(create_table_query)

# Insert data into the table
insert_query = "INSERT INTO house_prices VALUES ({})".format(','.join(['?']*len(header)))
cursor.executemany(insert_query, data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database setup complete.")

# SQL statements to fetch data from the database into a Pandas DataFrame using joins.

import pandas as pd
import sqlite3

# Connect to the database
conn = sqlite3.connect('house_prices.db')

# Write SQL query with joins to fetch data
query = """
SELECT *
FROM house_prices
-- Add JOIN clauses if necessary
"""

# Fetch data into a Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close connection
conn.close()

# Display the DataFrame
print(df.head())

# data preprocessing, exploratory data analysis, model training, and deployment.

# Data processing

from sklearn.model_selection import train_test_split

# Perform train/test split
X = df.drop(columns=['SalePrice'])  # Features
y = df['SalePrice']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)  # Adjust test_size and random_state as needed

# Exploratory Data analysis:

import seaborn as sns
import matplotlib.pyplot as plt

# Categorize data into categorical and numerical values
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

# Identify null and missing values
null_values = X_train.isnull().sum()

# Plot feature correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Display violin plots
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=col, y='SalePrice', data=pd.concat([X_train[col], y_train], axis=1))
    plt.title(f'{col} vs SalePrice')
    plt.show()

# Examine attribute distributions
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(X_train[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.show()

# 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self):
        self.num_features = numerical_cols
        self.cat_features = categorical_cols
        
        # Preprocessing steps for numerical features
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing steps for categorical features
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine numerical and categorical preprocessing steps
        self.preprocessor = ColumnTransformer([
            ('num', self.num_pipeline, self.num_features),
            ('cat', self.cat_pipeline, self.cat_features)
        ])
    
    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

# use this preprocessor to transform our data:

preprocessor = Preprocessor()
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Model Training and optimization:

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Register experiments and models in MLFlow
mlflow.set_experiment("House_Prices_Prediction")

# Baseline model - Linear Regression
with mlflow.start_run(run_name="Baseline_Linear_Regression"):
    lr = LinearRegression()
    lr.fit(X_train_preprocessed, y_train)
    
    y_pred_train_lr = lr.predict(X_train_preprocessed)
    y_pred_test_lr = lr.predict(X_test_preprocessed)
    
    mae_lr = mean_absolute_error(y_test, y_pred_test_lr)
    mse_lr = mean_squared_error(y_test, y_pred_test_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_test_lr, squared=False)
    
    mlflow.log_metric("MAE", mae_lr)
    mlflow.log_metric("MSE", mse_lr)
    mlflow.log_metric("RMSE", rmse_lr)
    
    mlflow.sklearn.log_model(lr, "model")

# Top models - RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
with mlflow.start_run(run_name="RandomForestRegressor"):
    grid_search.fit(X_train_preprocessed, y_train)
    
    best_rf = grid_search.best_estimator_
    
    y_pred_train_rf = best_rf.predict(X_train_preprocessed)
    y_pred_test_rf = best_rf.predict(X_test_preprocessed)
    
    mae_rf = mean_absolute_error(y_test, y_pred_test_rf)
    mse_rf = mean_squared_error(y_test, y_pred_test_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_test_rf, squared=False)
    
    mlflow.log_metric("MAE", mae_rf)
    mlflow.log_metric("MSE", mse_rf)
    mlflow.log_metric("RMSE", rmse_rf)
    
    mlflow.sklearn.log_model(best_rf, "model")

# hyperparameter tuning, feature selection, and feature engineering

from sklearn.feature_selection import SelectFromModel

# Feature selection using RandomForestRegressor
feature_importances = best_rf.feature_importances_
sfm = SelectFromModel(best_rf, threshold='median')
X_train_selected = sfm.fit_transform(X_train_preprocessed, y_train)
X_test_selected = sfm.transform(X_test_preprocessed)

# Feature engineering (if needed)
# For example, you can create new features by combining existing ones or use PCA

# Hyperparameter Tuning:

# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
with mlflow.start_run(run_name="RandomForestRegressor_Hyperparameter_Tuning"):
    grid_search.fit(X_train_selected, y_train)
    
    best_rf_tuned = grid_search.best_estimator_
    
    y_pred_train_rf_tuned = best_rf_tuned.predict(X_train_selected)
    y_pred_test_rf_tuned = best_rf_tuned.predict(X_test_selected)
    
    mae_rf_tuned = mean_absolute_error(y_test, y_pred_test_rf_tuned)
    mse_rf_tuned = mean_squared_error(y_test, y_pred_test_rf_tuned)
    rmse_rf_tuned = mean_squared_error(y_test, y_pred_test_rf_tuned, squared=False)
    
    mlflow.log_metric("MAE", mae_rf_tuned)
    mlflow.log_metric("MSE", mse_rf_tuned)
    mlflow.log_metric("RMSE", rmse_rf_tuned)
    
    mlflow.sklearn.log_model(best_rf_tuned, "model")

# Model Deployment:

# Choose the best model for deployment
# For example, let's say best_rf_tuned is the best model

# Export the model as a pickle file
import joblib
joblib.dump(best_rf_tuned, 'best_model.pkl')

# Create a Docker image for deployment
# You can use Dockerfile to create an image with necessary dependencies and the model file

# Deploy the Docker image on Digital Ocean or alternative services
# Follow the deployment process for the chosen service

# # Develop a Streamlit app that interfaces with the deployed model
# You can create a new Python script containing the Streamlit app code
# Use Streamlit to create a user-friendly interface where users can input data and get predictions from the deployed model

# Deploy the Streamlit app on the cloud
# Follow the deployment process for Streamlit apps, which typically involves hosting on platforms like Heroku or Streamlit Sharing





