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

# 6. Optimize and predict with Extra Trees and XGBoost models
def optimize_and_plot(models, param_grids, X_train, X_test, y_train, y_test, target_name, model_name):
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
    
    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    
    # Training set scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training Set')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Ideal Line')
    plt.title(f'{model_name} - {target_name} (Training Set)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    
    # Testing set scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Testing Set')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
    plt.title(f'{model_name} - {target_name} (Testing Set)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save training and testing predictions as CSV, ensuring lengths match
    train_results_df = pd.DataFrame({
        'Actual_Training_Set': y_train,
        'Predicted_Training_Set': y_train_pred
    })
    test_results_df = pd.DataFrame({
        'Actual_Testing_Set': y_test,
        'Predicted_Testing_Set': y_test_pred
    })
    
    output_file_train = f'C:/Users/Focalors/Desktop/Test/{model_name}_{target_name}_train_predictions.csv'
    output_file_test = f'C:/Users/Focalors/Desktop/Test/{model_name}_{target_name}_test_predictions.csv'
    
    train_results_df.to_csv(output_file_train, index=False)
    test_results_df.to_csv(output_file_test, index=False)
    
    print(f"Training set predictions for {model_name} saved to {output_file_train}")
    print(f"Testing set predictions for {model_name} saved to {output_file_test}")

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

# 8. Use Extra Trees model for Diffusion Coefficient
optimize_and_plot(models, {'Extra Trees': param_grids_et}, X_train_diff_scaled, X_test_diff_scaled, y_train_diff, y_test_diff, 'Diffusion Coefficient', 'Extra Trees')

# 9. Use XGBoost model for Ionic Conductivity
optimize_and_plot(models, {'XGBoost': param_grids_xgb}, X_train_cond_scaled, X_test_cond_scaled, y_train_cond, y_test_cond, 'Ionic Conductivity', 'XGBoost')