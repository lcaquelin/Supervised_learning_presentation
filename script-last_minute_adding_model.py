"""
Implement LightGBM
"""

# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# Display descriptive statistics of the DataFrame, including count, mean, standard deviation, min, max, and quartiles.
print("\nDataFrame descriptive statistics:")
print(df.describe().round(2))

# Display descriptive statistics of the DataFrame for categorical data
print("\nDataFrame descriptive statistics for categorical data:")
print(df[['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'age', 'gender', 'interest']].describe())

## Dropping unique id

# Drop the 'ad_id' column
df = df.drop('ad_id', axis=1)

# Display the columns of the DataFrame to confirm the column has been dropped
print("Columns of the DataFrame after dropping 'ad_id':")
print(df.columns)

# EDA - Visualisations

# Dropping the unused correlated columns


# Drop the 'Impressions', 'Clicks', and 'Total_Conversion' columns
df = df.drop(['Impressions', 'Clicks', 'Total_Conversion'], axis=1)

# Display the columns of the DataFrame to confirm the columns have been dropped
print("Columns of the DataFrame after dropping 'Impressions', 'Clicks', and 'Total_Conversion':")
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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

# Define features (X) and targets (y)
# Keeping all specified features: 'age', 'gender', 'interest', and 'Spent'
features = df[['age', 'gender', 'interest', 'Spent']]
target_approved = df['Approved_Conversion']
target_ratio = df['Spent_on_Conversion_Ratio']

# Reset the index of the features DataFrame
features = features.reset_index(drop=True)

# Define categorical and numerical features
categorical_features = ['age', 'gender', 'interest']
numerical_features = ['Spent']

# Create a ColumnTransformer.
# To remove one-hot encoding, we will pass categorical features through without transforming them.
# Numerical features will also be passed through.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features) # Pass numerical features as they are
    ],
    remainder='passthrough' # This will pass 'age', 'gender', 'interest' as they are (categorical dtypes)
)

# Apply the preprocessor to the features
features_processed = preprocessor.fit_transform(features)

# Get the feature names after processing. The order will be numerical_features, then categorical_features.
# Note: With 'remainder="passthrough"' and no explicit transformer for categorical features,
# the output order will be numerical_features first, then the remaining (categorical) features in their original order.
# The names will correspond to the original column names.
all_feature_names = numerical_features + categorical_features

# Convert the processed features to a DataFrame
# Ensure features_processed is 2D and has the correct number of columns
# If features_processed comes as a single column from preprocessor, it might be 1D, so reshape if needed.
if features_processed.ndim == 1:
    features_processed = features_processed.reshape(-1, 1)

# Handle the case where remainder='passthrough' includes non-numeric columns in features_processed
# and `pd.DataFrame` might try to infer types, potentially leading to issues with models expecting all numeric.
# For now, we proceed assuming `features_processed` contains suitable types for conversion based on the original dtypes
# of 'age', 'gender', 'interest', and 'Spent' being passed directly.
features_processed_df = pd.DataFrame(features_processed, columns=all_feature_names)

print("Final features (first 5 rows) after removing one-hot encoding:")
print(features_processed_df.head())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import numpy as np

# Split the data for the 'Approved_Conversion' target
X_train_approved, X_test_approved, y_train_approved, y_test_approved = train_test_split(
    features_processed_df, target_approved, test_size=0.2, random_state=42
)

# Split the data for the 'Spent_on_Conversion_Ratio' target
X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(
    features_processed_df, target_ratio, test_size=0.2, random_state=42
)

# Combine y_train_approved and y_train_ratio into a single target array for multi-output models
y_train = np.column_stack((y_train_approved, y_train_ratio))
y_test = np.column_stack((y_test_approved, y_test_ratio))

print("\nShapes of the split data:")
print(f"X_train_approved: {X_train_approved.shape}")
print(f"X_test_approved: {X_test_approved.shape}")
print(f"y_train_approved: {y_train_approved.shape}")
print(f"y_test_approved: {y_test_approved.shape}")
print(f"X_train_ratio: {X_train_ratio.shape}")
print(f"X_test_ratio: {X_test_ratio.shape}")
print(f"y_train_ratio: {y_train_ratio.shape}")
print(f"y_test_ratio: {y_test_ratio.shape}")
print(f"y_train (combined): {y_train.shape}")
print(f"y_test (combined): {y_test.shape}")

## Setup mlflow

### Subtask:
# Install and configure MLflow for experiment tracking.

# Install the mlflow library using pip.


# Commented out IPython magic to ensure Python compatibility.
# %pip install mlflow

# Import the mlflow library and set the tracking URI to a local directory.


import mlflow
mlflow.set_tracking_uri("mlruns")

## Implement LightGBM


import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd

# Ensure categorical features are of 'category' dtype for LightGBM to handle them directly
# This needs to be done on the split data as well
categorical_cols = ['age', 'gender', 'interest']
for col in categorical_cols:
    X_train_approved[col] = X_train_approved[col].astype('category')
    X_test_approved[col] = X_test_approved[col].astype('category')
    X_train_ratio[col] = X_train_ratio[col].astype('category')
    X_test_ratio[col] = X_test_ratio[col].astype('category')

# Convert 'Spent' column to numeric type, handling potential errors
for df_set in [X_train_approved, X_test_approved, X_train_ratio, X_test_ratio]:
    df_set['Spent'] = pd.to_numeric(df_set['Spent'], errors='coerce')
    # Fill any NaN values that might result from 'coerce' if necessary, e.g., with 0 or mean
    df_set['Spent'] = df_set['Spent'].fillna(0) # Filling with 0 as a simple strategy


# Define model parameters
# These can be tuned for better performance
lgbm_params = {
    'objective': 'regression_l1', # MAE objective, good for robustness to outliers
    'metric': 'mae', # Evaluation metric
    'n_estimators': 100, # Number of boosting rounds
    'learning_rate': 0.1,
    'feature_fraction': 0.8, # Fraction of features to consider at each split
    'bagging_fraction': 0.8, # Fraction of data to sample for each tree
    'bagging_freq': 1,
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1, # Use all available cores
    'random_state': 42
}

# Create a LightGBM Regressor instance
lgbm = lgb.LGBMRegressor(**lgbm_params)

# Wrap the LightGBM regressor for multi-output prediction
# This creates two separate models, one for each target
multi_output_lgbm = MultiOutputRegressor(estimator=lgbm)

# Train the model
# LightGBM will automatically handle 'category' dtypes in DataFrames
print("Training LightGBM MultiOutputRegressor...")
multi_output_lgbm.fit(
    X_train_approved, y_train # X_train_approved (which is a DataFrame with 'category' dtypes) is used here
)
print("Training complete.")

# Make predictions on the test set
y_pred = multi_output_lgbm.predict(X_test_approved)

# Separate predictions for each target
y_pred_approved = y_pred[:, 0]
y_pred_ratio = y_pred[:, 1]

# Separate actual values for each target
y_test_approved_actual = y_test[:, 0]
y_test_ratio_actual = y_test[:, 1]

# Evaluate the model
mae_approved = mean_absolute_error(y_test_approved_actual, y_pred_approved)
rmse_approved = np.sqrt(mean_squared_error(y_test_approved_actual, y_pred_approved))

mae_ratio = mean_absolute_error(y_test_ratio_actual, y_pred_ratio)
rmse_ratio = np.sqrt(mean_squared_error(y_test_ratio_actual, y_pred_ratio))

print(f"\nMetrics for Approved_Conversion:")
print(f"  MAE: {mae_approved:.4f}")
print(f"  RMSE: {rmse_approved:.4f}")

print(f"\nMetrics for Spent_on_Conversion_Ratio:")
print(f"  MAE: {mae_ratio:.4f}")
print(f"  RMSE: {rmse_ratio:.4f}")

# MLflow Logging
# You might want to create a new experiment or use an existing one
# mlflow.set_experiment("LightGBM Multi-Output Regression") # Uncomment to set a specific experiment name

with mlflow.start_run():
    mlflow.log_params(lgbm_params)

    mlflow.log_metric("mae_approved", mae_approved)
    mlflow.log_metric("rmse_approved", rmse_approved)

    mlflow.log_metric("mae_ratio", mae_ratio)
    mlflow.log_metric("rmse_ratio", rmse_ratio)

    mlflow.lightgbm.log_model(multi_output_lgbm, "lightgbm_model")

    print("\nMLflow run completed. Check 'mlruns' directory for logs.")

# Visualization
plt.figure(figsize=(14, 6))

# Plot for Approved_Conversion
plt.subplot(1, 2, 1)
plt.scatter(y_test_approved_actual, y_pred_approved, alpha=0.7)
plt.plot([min(y_test_approved_actual), max(y_test_approved_actual)], [min(y_test_approved_actual), max(y_test_approved_actual)], 'r--')
plt.title('Approved_Conversion: Actual vs. Predicted')
plt.xlabel('Actual Approved_Conversion')
plt.ylabel('Predicted Approved_Conversion')
plt.grid(True)

# Plot for Spent_on_Conversion_Ratio
plt.subplot(1, 2, 2)
plt.scatter(y_test_ratio_actual, y_pred_ratio, alpha=0.7)
plt.plot([min(y_test_ratio_actual), max(y_test_ratio_actual)], [min(y_test_ratio_actual), max(y_test_ratio_actual)], 'r--')
plt.title('Spent_on_Conversion_Ratio: Actual vs. Predicted')
plt.xlabel('Actual Spent_on_Conversion_Ratio')
plt.ylabel('Predicted Spent_on_Conversion_Ratio')
plt.grid(True)

plt.tight_layout()
plt.show()