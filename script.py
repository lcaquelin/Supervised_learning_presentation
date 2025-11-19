# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import mlflow

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# Data pre-processing and main info
## Loading the data and general info

# Define the path to the CSV file. Make sure this path is correct if your file is in a different location.
file_path = 'C:\\Documents\\Pro\\10-Supervised_learning\\Final_project\\Supervised_learning_presentation\\KAG_conversion_data.csv'

# Read the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(file_path)

    # Display the first 5 rows of the DataFrame to get a quick look at the data structure and content.
    print("First 5 rows of the DataFrame:")
    print(df.head())

    # Display information about the DataFrame, including the index dtype and column dtypes, non-null values and memory usage.
    print("\nDataFrame information:")
    print(df.info())

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

## Converting text data to categorical and description

columns_to_convert = ['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'age', 'gender', 'interest']
for col in columns_to_convert:
    df[col] = df[col].astype('category')

# Display the data types to confirm the conversion
print("Data types after conversion:")
print(df.dtypes)


## Dropping unique id

# Drop the 'ad_id' column
df = df.drop('ad_id', axis=1)

# Display the columns of the DataFrame to confirm the column has been dropped
print("Columns of the DataFrame after dropping 'ad_id':")
print(df.columns)

# Target: ROI or conversion ? The zero-value cases
# Condition 1: spent = 0 and approved_conversion = 0
print("Row(s) where Spent = 0 and Approved_Conversion = 0:")
condition1_df = df[(df['Spent'] == 0) & (df['Approved_Conversion'] == 0)]
if not condition1_df.empty:
    print(condition1_df.head(1))
else:
    print("No rows found for this condition.")

print("\n" + "="*30 + "\n")

# Condition 2: spent = 0 and approved_conversion > 0
print("Row(s) where Spent = 0 and Approved_Conversion > 0:")
condition2_df = df[(df['Spent'] == 0) & (df['Approved_Conversion'] > 0)]
if not condition2_df.empty:
    print(condition2_df.head(1))
else:
    print("No rows found for this condition.")

print("\n" + "="*30 + "\n")

# Condition 3: spent > 0 and approved_conversion = 0
print("Row(s) where Spent > 0 and Approved_Conversion = 0:")
condition3_df = df[(df['Spent'] > 0) & (df['Approved_Conversion'] == 0)]
if not condition3_df.empty:
    print(condition3_df.head(1))
else:
    print("No rows found for this condition.")

print("\n" + "="*30 + "\n")

# Condition 4: spent > 0 and approved_conversion > 0
print("Row(s) where Spent > 0 and Approved_Conversion > 0:")
condition4_df = df[(df['Spent'] > 0) & (df['Approved_Conversion'] > 0)]
if not condition4_df.empty:
    print(condition4_df.head(1))
else:
    print("No rows found for this condition.")


# Calculate the percentage of rows where 'Spent' is 0
spent_zero_percentage = (df['Spent'] == 0).mean() * 100

# Calculate the percentage of rows where 'Approved_Conversion' is 0
approved_conversion_zero_percentage = (df['Approved_Conversion'] == 0).mean() * 100

print(f"Percentage of rows where Spent = 0: {spent_zero_percentage:.0f}%")
print(f"Percentage of rows where Approved_Conversion = 0: {approved_conversion_zero_percentage:.0f}%")

# Add a new column 'Spent_Conversion_Ratio' to df
# Calculate the ratio Spent/Approved_conversion, handle division by zero by setting the ratio to 0
df['Spent_on_Conversion_Ratio'] = df.apply(lambda row: row['Spent'] / row['Approved_Conversion'] if row['Approved_Conversion'] != 0 else 0, axis=1)

# Display the first few rows with the new column
print("DataFrame with 'Spent_Conversion_Ratio':")
print(df.head())




# Splitting the data set and defining the targets

# Define features (X) and targets (y)
features = df[['age', 'gender', 'interest', 'Spent']]
target_approved = df['Approved_Conversion']
target_ratio = df['Spent_on_Conversion_Ratio']

# Reset the index of the features DataFrame
features = features.reset_index(drop=True)

# Define categorical and numerical features for the ColumnTransformer
categorical_features = ['age', 'gender', 'interest']
numerical_features = ['Spent']

# Create a ColumnTransformer to apply OneHotEncoder to categorical features
# Use sparse_output=False to get a dense array output
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' # Keep numerical features as they are
)

# Apply the preprocessor to the features
features_processed = preprocessor.fit_transform(features)

# Get the feature names after one-hot encoding and from the passthrough
encoded_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)
passthrough_feature_names = numerical_features # 'Spent' is passed through

# Combine all feature names
all_feature_names = list(encoded_feature_names) + passthrough_feature_names

# Convert the processed features to a DataFrame
features_processed_df = pd.DataFrame(features_processed, columns=all_feature_names)

print("Final features (first 5 rows):")
print(features_processed_df.head())

# Split the data for the 'Approved_Conversion' target
X_train_approved, X_test_approved, y_train_approved, y_test_approved = train_test_split(
    features_processed, target_approved, test_size=0.2, random_state=42
)

# Split the data for the 'Spent_on_Conversion_Ratio' target
X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(
    features_processed, target_ratio, test_size=0.2, random_state=42
)

print("\nShapes of the split data:")
print(f"X_train_approved: {X_train_approved.shape}")
print(f"X_test_approved: {X_test_approved.shape}")
print(f"y_train_approved: {y_train_approved.shape}")
print(f"y_test_approved: {y_test_approved.shape}")
print(f"X_train_ratio: {X_train_ratio.shape}")
print(f"X_test_ratio: {X_test_ratio.shape}")
print(f"y_train_ratio: {y_train_ratio.shape}")
print(f"y_test_ratio: {y_test_ratio.shape}")

# Task
""" Generate Python code to train and evaluate a `MultiOutputRegressor` with a `DummyRegressor` 
and a `LinearRegression` model on the provided data. Log the training process and results for 
both models using MLflow, including parameters and evaluation metrics (MAE, MSE, R²) 
for both targets (`target_approved` and `target_ratio`). The primary tuning metric for 
future steps will be the R² of `spent_on_conversion_ratio`.
"""

## Setup mlflow

### Subtask:
# Install and configure MLflow for experiment tracking.

import mlflow
mlflow.set_tracking_uri("mlruns")

## 1. Dummy regressor

### Subtask:
# Train and evaluate a `MultiOutputRegressor` with a `DummyRegressor` as the baseline model, 
# logging results with MLflow.

# Start an MLflow run for the Dummy Regressor
with mlflow.start_run(run_name="Dummy Regressor Baseline"):
    # Log the model parameters
    mlflow.log_param("strategy", "mean")

    # Initialize the MultiOutputRegressor with a DummyRegressor
    dummy_regressor = MultiOutputRegressor(DummyRegressor(strategy="mean"))

    # Combine the target variables for training
    y_train = np.column_stack((y_train_approved, y_train_ratio))
    y_test = np.column_stack((y_test_approved, y_test_ratio))

    # Train the model
    dummy_regressor.fit(X_train_approved, y_train)

    # Make predictions
    y_pred = dummy_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics:")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics:")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")


    # Create the 'out' directory if it doesn't exist already
    # 'exist_ok=True' prevents Python from crashing if the folder is already there
    os.makedirs('out', exist_ok=True)

    # Write to the 'score.txt' file
    with open('out/score.txt', 'w') as f:
        f.write(f"dummy_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n") 
        f.write(" " * 30 + "\n")   
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been created successfully.")


    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Dummy Regressor: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Dummy Regressor: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()


# The MLflow run is automatically ended when exiting the 'with' block

## 2. Linear regression

### Subtask:
# Train and evaluate a `LinearRegression` model, logging results with MLflow.

# Start an MLflow run for the Linear Regression model
with mlflow.start_run(run_name="Linear Regression"):
    # Initialize the MultiOutputRegressor with a LinearRegression model
    linear_regressor = MultiOutputRegressor(LinearRegression())

    # Train the model using the combined target training data
    linear_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = linear_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Linear Regression):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Linear Regression):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Create the 'out' directory if it doesn't exist already
    # 'exist_ok=True' prevents Python from crashing if the folder is already there
    os.makedirs('out', exist_ok=True)

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"linear_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n") 
        f.write(" " * 30 + "\n")   
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")


    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Linear Regression: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Linear Regression: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

## 2.2. Linear regression - Lasso

# Start an MLflow run for the Lasso Regression model
with mlflow.start_run(run_name="Lasso Regression"):
    # Define and log model parameters (you can tune these later)
    alpha_param = 1.0  # Default alpha for Lasso
    mlflow.log_param("alpha", alpha_param)

    # Initialize the MultiOutputRegressor with a Lasso model
    lasso_regressor = MultiOutputRegressor(Lasso(alpha=alpha_param, random_state=42)) # Use MultiOutputRegressor for Lasso

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    lasso_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = lasso_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Lasso Regression):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Lasso Regression):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"lasso_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n") 
        f.write(" " * 30 + "\n")   
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")

    # Display the learned weights for each target
    print("\nLearned Weights for Approved_Conversion:")
    print(lasso_regressor.estimators_[0].coef_)
    print("\nLearned Weights for Spent_on_Conversion_Ratio:")
    print(lasso_regressor.estimators_[1].coef_)


    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Lasso Regression: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Lasso Regression: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block


## 3. Polynomial regression

# Start an MLflow run for the Polynomial Regression model
with mlflow.start_run(run_name="Polynomial Regression (Degree 2)"):
    # Log the model parameter
    mlflow.log_param("degree", 2)

    # Create a pipeline with PolynomialFeatures and LinearRegression
    polynomial_regressor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    polynomial_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = polynomial_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Polynomial Regression Degree 2):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Polynomial Regression Degree 2):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    
    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"polynomial_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n")   
        f.write(" " * 30 + "\n") 
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")
    
    print("File out/score.txt has been uploaded successfully.")

    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Polynomial Regression (Degree 2): Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Polynomial Regression (Degree 2): Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block


## 4. Polynomial regression - Lasso

# Start an MLflow run for the Polynomial Regression with Lasso model
with mlflow.start_run(run_name="Polynomial Regression (Degree 2) with Lasso"):
    # Define and log model parameters (you can tune these later)
    alpha_param = 1.0  # Default alpha for Lasso
    mlflow.log_param("degree", 2)
    mlflow.log_param("alpha", alpha_param)

    # Create a pipeline with PolynomialFeatures and Lasso
    lasso_polynomial_regressor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('lasso', MultiOutputRegressor(Lasso(alpha=alpha_param, random_state=42))) # Use MultiOutputRegressor for Lasso
    ])

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    lasso_polynomial_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = lasso_polynomial_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Polynomial Regression Degree 2 with Lasso):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Polynomial Regression Degree 2 with Lasso):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Create the 'out' directory if it doesn't exist already
    # 'exist_ok=True' prevents Python from crashing if the folder is already there
    os.makedirs('out', exist_ok=True)

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"lasso_polynomial_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n")    
        f.write(" " * 30 + "\n")
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")

    # Display the learned weights for each target
    print("\nLearned Weights for Approved_Conversion:")
    print(lasso_polynomial_regressor.named_steps['lasso'].estimators_[0].coef_)
    print("\nLearned Weights for Spent_on_Conversion_Ratio:")
    print(lasso_polynomial_regressor.named_steps['lasso'].estimators_[1].coef_)

    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Polynomial Regression (Degree 2) with Lasso: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Polynomial Regression (Degree 2) with Lasso: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block


## 5. Decision Tree

# Start an MLflow run for the Decision Tree Regressor model
with mlflow.start_run(run_name="Decision Tree Regressor"):
    # Initialize the MultiOutputRegressor with a DecisionTreeRegressor
    # You can add parameters like max_depth, min_samples_split, etc. here for tuning later
    decision_tree_regressor = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    decision_tree_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = decision_tree_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Decision Tree Regressor):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Decision Tree Regressor):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")


    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"decision_tree_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n")  
        f.write(" " * 30 + "\n")  
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")

    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Decision Tree Regressor: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Decision Tree Regressor: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block


## 6. Random forest

# Start an MLflow run for the Random Forest Regressor model
with mlflow.start_run(run_name="Random Forest Regressor"):
    # Initialize the MultiOutputRegressor with a RandomForestRegressor
    # You can add parameters like n_estimators, max_depth, etc. here for tuning later
    random_forest_regressor = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    random_forest_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = random_forest_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Random Forest Regressor):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Random Forest Regressor):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"random_forest_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n")  
        f.write(" " * 30 + "\n")  
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")

    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Random Forest Regressor: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Random Forest Regressor: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block


## 7. Gradient Boosting Tree

# Start an MLflow run for the Gradient Boosting Regressor model
with mlflow.start_run(run_name="Gradient Boosting Regressor"):
    # Initialize the MultiOutputRegressor with a GradientBoostingRegressor
    # You can add parameters like n_estimators, learning_rate, max_depth, etc. here for tuning later
    gradient_boosting_regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    gradient_boosting_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = gradient_boosting_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (Gradient Boosting Regressor):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (Gradient Boosting Regressor):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"gradient_boosting_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n") 
        f.write(" " * 30 + "\n")   
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")


    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("Gradient Boosting Regressor: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("Gradient Boosting Regressor: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block

## 8. kNeighbor

# Start an MLflow run for the KNeighbors Regressor model
with mlflow.start_run(run_name="KNeighbors Regressor"):
    # Initialize the MultiOutputRegressor with a KNeighborsRegressor
    # You can add parameters like n_neighbors, weights, algorithm, etc. here for tuning later
    kneighbors_regressor = MultiOutputRegressor(KNeighborsRegressor())

    # Train the model using the combined target training data
    # Note: We use X_train_approved here as the features are the same for both targets
    kneighbors_regressor.fit(X_train_approved, y_train)

    # Make predictions on the test data
    y_pred = kneighbors_regressor.predict(X_test_approved)

    # Calculate and log metrics for Approved_Conversion
    mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

    mlflow.log_metric("approved_conversion_mae", mae_approved)
    mlflow.log_metric("approved_conversion_mse", mse_approved)
    mlflow.log_metric("approved_conversion_r2", r2_approved)

    print(f"Approved_Conversion Metrics (KNeighbors Regressor):")
    print(f"  MAE: {mae_approved:.4f}")
    print(f"  MSE: {mse_approved:.4f}")
    print(f"  R2: {r2_approved:.4f}")

    # Calculate and log metrics for Spent_on_Conversion_Ratio
    mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

    mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

    print(f"Spent_on_Conversion_Ratio Metrics (KNeighbors Regressor):")
    print(f"  MAE: {mae_ratio:.4f}")
    print(f"  MSE: {mse_ratio:.4f}")
    print(f"  R2: {r2_ratio:.4f}")

    # Write to the 'score.txt' file
    with open('out/score.txt', 'a') as f:
        f.write(" " * 30 + "\n")
        f.write(f"kneighbors_regressor\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {mae_approved}\n")
        f.write(f"MSE: {mse_approved}\n")
        f.write(f"R2: {r2_approved}\n")    
        f.write(" " * 30 + "\n")
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {mae_ratio}\n")
        f.write(f"MSE: {mse_ratio}\n")
        f.write(f"R2: {r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("File out/score.txt has been uploaded successfully.")


    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title("KNeighbors Regressor: Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2) # Diagonal line for perfect predictions
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title("KNeighbors Regressor: Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

# The MLflow run is automatically ended when exiting the 'with' block

## 9. kNeighbor with Hyperopt
# Set the MLflow tracking URI
mlflow.set_tracking_uri("mlruns")
print("Libraries imported and MLflow tracking URI set to 'mlruns'.")

# Create an objective function that trains and evaluates for minimization of MAE.
def objective(params):
    n_neighbors = int(params['n_neighbors'])

    with mlflow.start_run(run_name=f"KNeighbors Regressor Hyperopt (n_neighbors={n_neighbors})", nested=True) as run:
        # Log hyperparameters
        mlflow.log_param("n_neighbors", n_neighbors)

        # Initialize the MultiOutputRegressor with KNeighborsRegressor
        knn_regressor = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=n_neighbors))

        # Train the model
        knn_regressor.fit(X_train_approved, y_train)

        # Make predictions
        y_pred = knn_regressor.predict(X_test_approved)

        # Calculate metrics for Approved_Conversion
        mae_approved = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mse_approved = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        r2_approved = r2_score(y_test[:, 0], y_pred[:, 0])

        # Log metrics for Approved_Conversion
        mlflow.log_metric("approved_conversion_mae", mae_approved)
        mlflow.log_metric("approved_conversion_mse", mse_approved)
        mlflow.log_metric("approved_conversion_r2", r2_approved)

        # Calculate metrics for Spent_on_Conversion_Ratio
        mae_ratio = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        mse_ratio = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        r2_ratio = r2_score(y_test[:, 1], y_pred[:, 1])

        # Log metrics for Spent_on_Conversion_Ratio
        mlflow.log_metric("spent_on_conversion_ratio_mae", mae_ratio)
        mlflow.log_metric("spent_on_conversion_ratio_mse", mse_ratio)
        mlflow.log_metric("spent_on_conversion_ratio_r2", r2_ratio)

        # Hyperopt minimizes the objective, so we return MAE for 'spent_on_conversion_ratio'
        return {'loss': mae_ratio,
                'status': STATUS_OK,
                'run_id': run.info.run_id,
                'n_neighbors': n_neighbors,
                'mae_approved': mae_approved,
                'mse_approved': mse_approved,
                'r2_approved': r2_approved,
                'mae_ratio': mae_ratio,
                'mse_ratio': mse_ratio,
                'r2_ratio': r2_ratio}

print("Objective function 'objective' has been defined.")


# Define the scope of research
search_space = {
    'n_neighbors': hp.quniform('n_neighbors', 1, 20, 1) # Integers from 1 to 20
}
print("Search space for 'n_neighbors' defined.")

# Find the best parameter n_neighbors
trials = Trials()
best_run = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20, # Number of different n_neighbors values to try
    trials=trials
)

print("\nHyperopt optimization complete.")
print(f"Best parameters found: {best_run}")

# Retrieve the result of the search and launch a new trial
best_trial = trials.best_trial['result']
best_n_neighbors = int(best_trial['n_neighbors'])
best_r2_ratio = best_trial['r2_ratio']
best_mae_ratio = best_trial['mae_ratio']
best_mse_ratio = best_trial['mse_ratio']
best_r2_approved = best_trial['r2_approved']
best_mae_approved = best_trial['mae_approved']
best_mse_approved = best_trial['mse_approved']
best_run_id = best_trial['run_id'] # Store the run_id for the best trial

print(f"\nBest n_neighbors: {best_n_neighbors}")
print(f"Best Spent_on_Conversion_Ratio MAE: {best_mae_ratio:.4f}")
print(f"Best Approved_Conversion R2: {best_r2_approved:.4f}")

# Train and evaluate the final model with the best parameters
with mlflow.start_run(run_name=f"KNeighbors Regressor Best Model (n_neighbors={best_n_neighbors})") as final_run:
    mlflow.log_param("n_neighbors", best_n_neighbors)

    final_knn_regressor = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=best_n_neighbors))
    final_knn_regressor.fit(X_train_approved, y_train)
    y_pred_best = final_knn_regressor.predict(X_test_approved)

    # Log metrics for the best model
    mlflow.log_metric("approved_conversion_mae", best_mae_approved)
    mlflow.log_metric("approved_conversion_mse", best_mse_approved)
    mlflow.log_metric("approved_conversion_r2", best_r2_approved)

    mlflow.log_metric("spent_on_conversion_ratio_mae", best_mae_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_mse", best_mse_ratio)
    mlflow.log_metric("spent_on_conversion_ratio_r2", best_r2_ratio)

    print(f"\nFinal Best KNeighbors Regressor Metrics (n_neighbors={best_n_neighbors}):")
    print(f"  Approved_Conversion MAE: {best_mae_approved:.4f}")
    print(f"  Approved_Conversion MSE: {best_mse_approved:.4f}")
    print(f"  Approved_Conversion R2: {best_r2_approved:.4f}")
    print(f"  Spent_on_Conversion_Ratio MAE: {best_mae_ratio:.4f}")
    print(f"  Spent_on_Conversion_Ratio MSE: {best_mse_ratio:.4f}")
    print(f"  Spent_on_Conversion_Ratio R2: {best_r2_ratio:.4f}")

    # Update score.txt with best model metrics
    with open('out/score.txt', 'a') as f:
        f.write(f"kneighbors_regressor_tuned (n_neighbors={best_n_neighbors})\n")
        f.write(" " * 30 + "\n")
        f.write(f"approved_conversion_mae\n")
        f.write(f"MAE: {best_mae_approved}\n")
        f.write(f"MSE: {best_mse_approved}\n")
        f.write(f"R2: {best_r2_approved}\n")
        f.write(" " * 30 + "\n")
        f.write(f"spent_on_conversion_ratio_mae\n")
        f.write(f"MAE: {best_mae_ratio}\n")
        f.write(f"MSE: {best_mse_ratio}\n")
        f.write(f"R2: {best_r2_ratio}\n")
        f.write(" " * 30 + "\n")
        f.write("-" * 30 + "\n")

    print("\nFile out/score.txt updated with best KNeighbors Regressor metrics.")

    # Visualize predictions vs. actual values for Approved_Conversion
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 0], y_pred_best[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
    plt.xlabel("Actual Approved Conversion")
    plt.ylabel("Predicted Approved Conversion")
    plt.title(f"Best KNeighbors Regressor (n_neighbors={best_n_neighbors}): Actual vs. Predicted Approved Conversion")
    plt.grid(True)
    plt.show()

    # Visualize predictions vs. actual values for Spent_on_Conversion_Ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, 1], y_pred_best[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
    plt.xlabel("Actual Spent on Conversion Ratio")
    plt.ylabel("Predicted Spent on Conversion Ratio")
    plt.title(f"Best KNeighbors Regressor (n_neighbors={best_n_neighbors}): Actual vs. Predicted Spent on Conversion Ratio")
    plt.grid(True)
    plt.show()

print(f"\nSummary of Hyperparameter Tuning for KNeighborsRegressor using Hyperopt:")
print(f"Hyperopt was used to tune the 'n_neighbors' parameter for the KNeighborsRegressor model.")
print(f"The search space for 'n_neighbors' was explored from 1 to 20.")
print(f"The optimization aimed to minimize the MAE for 'Spent_on_Conversion_Ratio'.")
print(f"The optimal 'n_neighbors' found was: {best_n_neighbors}")
print(f"The final evaluation metrics for this best model are:")
print(f"  Approved_Conversion:")
print(f"    MAE: {best_mae_approved:.4f}")
print(f"    MSE: {best_mse_approved:.4f}")
print(f"    R2: {best_r2_approved:.4f}")
print(f"  Spent_on_Conversion_Ratio:")
print(f"    MAE: {best_mae_ratio:.4f}")
print(f"    MSE: {best_mse_ratio:.4f}")
print(f"    R2: {best_r2_ratio:.4f}")



