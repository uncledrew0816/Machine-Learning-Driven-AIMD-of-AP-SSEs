import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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

# 4. Data standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train Extra Trees model (for Diffusion Coefficient)
extra_trees = ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
extra_trees.fit(X_scaled, y_diffusion)

# 6. Train XGBoost model (for Ionic Conductivity)
xgboost = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgboost.fit(X_scaled, y_conductivity)

# 7. Extract feature importance
def plot_feature_importance(model, feature_names, target_name, model_name):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(feature_importance_df['Importance'], labels=feature_importance_df['Feature'], autopct='%1.1f%%', startangle=90)
    plt.title(f'{model_name} - Feature Importance for {target_name}')
    plt.tight_layout()
    plt.show()
    
    # Save feature importance data
    output_file = f'C:/Users/Focalors/Desktop/Test/{model_name}_{target_name}_feature_importances.csv'
    feature_importance_df.to_csv(output_file, index=False)
    print(f'Feature importance for {model_name} saved to {output_file}')
    
# 8. Plot and save feature importance for Extra Trees (for Diffusion Coefficient)
plot_feature_importance(extra_trees, X.columns, 'Diffusion Coefficient', 'Extra Trees')

# 9. Plot and save feature importance for XGBoost (for Ionic Conductivity)
plot_feature_importance(xgboost, X.columns, 'Ionic Conductivity', 'XGBoost')