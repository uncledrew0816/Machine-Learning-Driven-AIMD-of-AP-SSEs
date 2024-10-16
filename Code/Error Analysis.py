import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

# 1. Load data
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# 2. Drop rows with missing values
df.dropna(subset=['Jump_Diffusivity', 'Avg_E_Act', 'Attempt_Freq', 
                  'Std_Attempt_Freq', 'Vibration_Amp', 'Particle_Density'], inplace=True)

# 3. Features and target variables
X = df.drop(columns=['Diffusion Coefficient (cm2/s)', 'Ionic Conductivity (mS/cm)', 'Defect', 'Jump_Diffusivity'])
y_diffusion = df['Diffusion Coefficient (cm2/s)']
y_conductivity = df['Ionic Conductivity (mS/cm)']

# 4. Split dataset into training and testing sets
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X, y_diffusion, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_conductivity, test_size=0.2, random_state=42)

# 5. Data standardization
scaler = StandardScaler()
X_train_diff_scaled = scaler.fit_transform(X_train_diff)
X_test_diff_scaled = scaler.transform(X_test_diff)
X_train_cond_scaled = scaler.fit_transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

# Error analysis and residual distribution function
def error_analysis_with_residuals(y_train, y_train_pred, y_test, y_test_pred, model_name, target_name):
    # Print array length information for debugging
    print(f"Length of actual training values: {len(y_train)}, Length of predicted training values: {len(y_train_pred)}")
    print(f"Length of actual testing values: {len(y_test)}, Length of predicted testing values: {len(y_test_pred)}")
    
    # Calculate MSE and R² for training and testing sets
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Print error results
    print(f"{model_name} - {target_name} Error Analysis:")
    print(f"Training Set MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
    print(f"Testing Set MSE: {mse_test:.4f}, R²: {r2_test:.4f}")
    
    # Calculate residuals
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    # Save training set residuals as CSV for use in Origin
    residuals_train_df = pd.DataFrame({
        'Actual_Training_Set': y_train,
        'Predicted_Training_Set': y_train_pred,
        'Residuals_Training_Set': residuals_train
    })
    output_file_residuals_train = f'C:/Users/Focalors/Desktop/Test/{model_name}_{target_name}_residuals_train.csv'
    residuals_train_df.to_csv(output_file_residuals_train, index=False)
    print(f"Training set residuals for {model_name} saved to {output_file_residuals_train}")
    
    # Save testing set residuals as CSV for use in Origin
    residuals_test_df = pd.DataFrame({
        'Actual_Testing_Set': y_test,
        'Predicted_Testing_Set': y_test_pred,
        'Residuals_Testing_Set': residuals_test
    })
    output_file_residuals_test = f'C:/Users/Focalors/Desktop/Test/{model_name}_{target_name}_residuals_test.csv'
    residuals_test_df.to_csv(output_file_residuals_test, index=False)
    print(f"Testing set residuals for {model_name} saved to {output_file_residuals_test}")

# Optimize and predict with Extra Trees and XGBoost models
def optimize_and_predict(models, param_grids, X_train, X_test, y_train, y_test, target_name, model_name):
    if model_name not in models or model_name not in param_grids:
        raise ValueError(f"{model_name} is not defined in models and parameter grids.")

    print(f"Optimizing hyperparameters for {model_name}...")
    
    # Perform grid search
    grid = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"Best parameters for {model_name}: {best_params}")
    
    # Predictions on training and testing sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Error analysis and residual distribution analysis
    error_analysis_with_residuals(y_train, y_train_pred, y_test, y_test_pred, model_name, target_name)

# 7. Define models
models = {
    "Extra Trees": ExtraTreesRegressor(),
    "XGBoost": XGBRegressor()
}

# Parameter grid for Extra Trees
param_grids_et = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Parameter grid for XGBoost
param_grids_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9]
}

# 8. Use Extra Trees model for Diffusion Coefficient and perform error analysis and residual distribution
optimize_and_predict(models, {'Extra Trees': param_grids_et}, X_train_diff_scaled, X_test_diff_scaled, y_train_diff, y_test_diff, 'Diffusion Coefficient', 'Extra Trees')

# 9. Use XGBoost model for Ionic Conductivity and perform error analysis and residual distribution
optimize_and_predict(models, {'XGBoost': param_grids_xgb}, X_train_cond_scaled, X_test_cond_scaled, y_train_cond, y_test_cond, 'Ionic Conductivity', 'XGBoost')