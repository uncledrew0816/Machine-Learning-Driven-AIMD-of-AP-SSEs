import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# Drop rows with missing values
df.dropna(subset=['Jump_Diffusivity', 'Avg_E_Act', 'Attempt_Freq', 
                  'Std_Attempt_Freq', 'Vibration_Amp', 'Particle_Density'], inplace=True)

# Features and target variables
X = df.drop(columns=['Diffusion Coefficient (cm2/s)', 'Ionic Conductivity (mS/cm)', 'Defect', 'Jump_Diffusivity'])
y_diffusion = df['Diffusion Coefficient (cm2/s)']
y_conductivity = df['Ionic Conductivity (mS/cm)']

# Split the dataset into training and testing sets
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X, y_diffusion, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_conductivity, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler()
X_train_diff_scaled = scaler.fit_transform(X_train_diff)
X_test_diff_scaled = scaler.transform(X_test_diff)
X_train_cond_scaled = scaler.fit_transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

# Define models and hyperparameter search space
param_grids = {
    "Lasso Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Ridge Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Elastic Net": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    "Partial Least Squares": {
        'n_components': [1, 2, 3, 4, 5]
    },
    "Support Vector Regression": {
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "Adaptive Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "Extra Trees": {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "K-Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Light Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_data_in_leaf': [10, 20, 50],  
        'max_depth': [3, 6, 9],  
    },
}

# Optimize feedforward neural network architecture
def create_feedforward_nn(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),  # Add Dropout to prevent overfitting
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Optimize RBF neural network architecture
def create_rbf_nn(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='tanh', input_shape=input_shape),
        layers.Dropout(0.3),  # Use Dropout layer
        layers.Dense(64, activation='tanh'),
        layers.Dense(32, activation='tanh'),
        layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "Elastic Net": ElasticNet(),
    "Partial Least Squares": PLSRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "Adaptive Boosting": AdaBoostRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "Support Vector Regression": SVR(),
    "K-Neighbors": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Light Gradient Boosting": LGBMRegressor(),
    "Feedforward Neural Network": create_feedforward_nn((X_train_diff.shape[1],)),  
    "Radial Basis Function Neural Network": create_rbf_nn((X_train_diff.shape[1],))  
}

# Evaluation function, using cross-validation and grid search optimization
def evaluate_models(models, param_grids, X_train, X_test, y_train, y_test):
    results = []
    best_params_results = []  # New list to store the best parameters
    for name, model in models.items():
        print(f"Training {name}...")
        best_params = None  # To store the best parameters
        if "Neural Network" in name:  # Handle neural network models separately
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

            model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=0, validation_split=0.2,
                      callbacks=[early_stopping, reduce_lr])
            predictions = model.predict(X_test)
            predictions = predictions.flatten()  # Convert 2D array to 1D
        else:
            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_  
                print(f"Best parameters for {name}: {best_params}")
            else:
                best_model = model  
                best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results.append([name, mse, r2])

        # Store model name and best parameters in best_params_results
        best_params_results.append([name, best_params if best_params else "N/A"])
    
    return pd.DataFrame(results, columns=['Model', 'MSE', 'R²']), pd.DataFrame(best_params_results, columns=['Model', 'Best Parameters'])

# Evaluate models for diffusion coefficient and ionic conductivity
diffusion_results, diffusion_best_params = evaluate_models(models, param_grids, X_train_diff_scaled, X_test_diff_scaled, y_train_diff, y_test_diff)
conductivity_results, conductivity_best_params = evaluate_models(models, param_grids, X_train_cond_scaled, X_test_cond_scaled, y_train_cond, y_test_cond)

# Combine results and save as CSV
final_results = pd.concat([diffusion_results.assign(Target='Diffusion Coefficient'),
                           conductivity_results.assign(Target='Ionic Conductivity')])

best_params_results = pd.concat([diffusion_best_params.assign(Target='Diffusion Coefficient'),
                                 conductivity_best_params.assign(Target='Ionic Conductivity')])

# Save model performance results and best parameters results
output_file_results = 'C:/Users/Focalors/Desktop/Test/model_performance_results_with_cv.csv'
output_file_params = 'C:/Users/Focalors/Desktop/Test/model_best_params_with_cv.csv'

final_results.to_csv(output_file_results, index=False)
best_params_results.to_csv(output_file_params, index=False)