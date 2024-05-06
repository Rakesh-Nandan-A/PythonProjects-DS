Here's a summary of what we've accomplished:

Data Loading and Preprocessing: Loaded the dataset into a SQLite database, performed data preprocessing, and split the data into training and testing sets.
Exploratory Data Analysis (EDA): Conducted EDA by categorizing data into categorical and numerical values, identifying null and missing values, plotting correlation heatmaps, violin plots, and examining attribute distributions.
Class-based Preprocessor: Implemented a class-based preprocessor to handle null and missing values, non-normal distributions, one-hot encoding, and scaling of the data.
Model Training and Optimization: Trained and optimized models using MLFlow, including a baseline Linear Regression model and a tuned RandomForestRegressor model.
Deployment: Created a Docker image for deployment and deployed the Streamlit app on a cloud service (e.g., Heroku) for accessibility.