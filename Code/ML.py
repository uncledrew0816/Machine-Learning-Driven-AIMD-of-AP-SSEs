import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# 加载数据
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# 特征和目标变量
X = df.drop(columns=['Diffusion Coefficient (cm2/s)', 'Ionic Conductivity (mS/cm)', 'Defect'])
y_diffusion = df['Diffusion Coefficient (cm2/s)']
y_conductivity = df['Ionic Conductivity (mS/cm)']

# 拆分数据集为训练集和测试集
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X, y_diffusion, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_conductivity, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_diff_scaled = scaler.fit_transform(X_train_diff)
X_test_diff_scaled = scaler.transform(X_test_diff)
X_train_cond_scaled = scaler.fit_transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

# 定义模型和超参数搜索范围
param_grids = {
    "Ridge Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Lasso Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Support Vector Regressor": {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 1]
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
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "K-Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
}

# 定义模型
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Support Vector Regressor": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "K-Neighbors": KNeighborsRegressor(),
    "XGBoost": XGBRegressor(),
}

# 定义评估函数，使用交叉验证和网格搜索优化
def evaluate_models(models, param_grids, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        print(f"正在进行 {name} 的网格搜索...")
        if name in param_grids:
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"{name} 的最佳参数: {grid.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        # 用最优模型进行预测
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # 使用交叉验证来评估模型
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        
        results.append([name, mse, r2, cv_mse])
    
    return pd.DataFrame(results, columns=['Model', 'MSE', 'R²', 'CV MSE'])

# 评估扩散系数和离子导电率的模型
diffusion_results = evaluate_models(models, param_grids, X_train_diff_scaled, X_test_diff_scaled, y_train_diff, y_test_diff)
conductivity_results = evaluate_models(models, param_grids, X_train_cond_scaled, X_test_cond_scaled, y_train_cond, y_test_cond)

# 合并结果并保存为CSV
final_results = pd.concat([diffusion_results.assign(Target='Diffusion Coefficient'),
                           conductivity_results.assign(Target='Ionic Conductivity')])

output_file = 'C:/Users/Focalors/Desktop/Test/model_performance_results_with_cv.csv'
final_results.to_csv(output_file, index=False)
