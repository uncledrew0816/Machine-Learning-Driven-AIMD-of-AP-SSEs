import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. Load data and split into training and testing sets
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'  # Actual file path
df = pd.read_excel(file_path)

# Features and target variables
X = df.drop(columns=['Diffusion Coefficient (cm2/s)', 'Ionic Conductivity (mS/cm)', 'Defect'])
y_diffusion = df['Diffusion Coefficient (cm2/s)']
y_conductivity = df['Ionic Conductivity (mS/cm)']

# Split dataset into training and testing sets
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X, y_diffusion, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_conductivity, test_size=0.2, random_state=42)

# 2. Data standardization
scaler = StandardScaler()
X_train_diff_scaled = scaler.fit_transform(X_train_diff)
X_test_diff_scaled = scaler.transform(X_test_diff)
X_train_cond_scaled = scaler.fit_transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

# Convert the scaled data back to DataFrame, preserving feature names
X_train_diff_scaled = pd.DataFrame(X_train_diff_scaled, columns=X.columns)
X_test_diff_scaled = pd.DataFrame(X_test_diff_scaled, columns=X.columns)
X_train_cond_scaled = pd.DataFrame(X_train_cond_scaled, columns=X.columns)
X_test_cond_scaled = pd.DataFrame(X_test_cond_scaled, columns=X.columns)

# 3. Define and train models
# Define models
best_diffusion_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
best_conductivity_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

# Train Gradient Boosting model for diffusivity
best_diffusion_model.fit(X_train_diff_scaled, y_train_diff)

# Train XGBoost model for conductivity
best_conductivity_model.fit(X_train_cond_scaled, y_train_cond)

# 4. Explain models using SHAP
# Use SHAP for Gradient Boosting model
explainer_diff = shap.Explainer(best_diffusion_model, X_train_diff_scaled)
shap_values_diff = explainer_diff(X_test_diff_scaled)

# Use SHAP for XGBoost model
explainer_cond = shap.Explainer(best_conductivity_model, X_train_cond_scaled)
shap_values_cond = explainer_cond(X_test_cond_scaled)

# 5. Plot SHAP summary plots
# SHAP summary plot for diffusion coefficient model
shap.summary_plot(shap_values_diff, X_test_diff_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot - Diffusivity")

# SHAP summary plot for conductivity model
shap.summary_plot(shap_values_cond, X_test_cond_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot - Conductivity")

# 6. Plot SHAP force plot for a single sample
# Randomly choose a sample
sample_index = 0  # Replace with the sample index you want to visualize

# SHAP force plot for diffusion coefficient model
shap.force_plot(explainer_diff.expected_value, shap_values_diff[sample_index].values, X_test_diff_scaled.iloc[sample_index, :], feature_names=X.columns)

# SHAP force plot for conductivity model
shap.force_plot(explainer_cond.expected_value, shap_values_cond[sample_index].values, X_test_cond_scaled.iloc[sample_index, :], feature_names=X.columns)