import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 加载数据并拆分训练集和测试集
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'  # 替换为实际文件路径
df = pd.read_excel(file_path)

# 特征和目标变量
X = df.drop(columns=['Diffusion Coefficient (cm2/s)', 'Ionic Conductivity (mS/cm)', 'Defect'])
y_diffusion = df['Diffusion Coefficient (cm2/s)']
y_conductivity = df['Ionic Conductivity (mS/cm)']

# 拆分数据集为训练集和测试集
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X, y_diffusion, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_conductivity, test_size=0.2, random_state=42)

# 2. 数据标准化
scaler = StandardScaler()
X_train_diff_scaled = scaler.fit_transform(X_train_diff)
X_test_diff_scaled = scaler.transform(X_test_diff)
X_train_cond_scaled = scaler.fit_transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

# 3. 定义和训练模型
# 定义模型
best_diffusion_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
best_conductivity_model = XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=3)

# 训练 Diffusivity 的 Gradient Boosting 模型
best_diffusion_model.fit(X_train_diff_scaled, y_train_diff)
train_diff_pred = best_diffusion_model.predict(X_train_diff_scaled)
test_diff_pred = best_diffusion_model.predict(X_test_diff_scaled)

# 训练 Conductivity 的 XGBoost 模型
best_conductivity_model.fit(X_train_cond_scaled, y_train_cond)
train_cond_pred = best_conductivity_model.predict(X_train_cond_scaled)
test_cond_pred = best_conductivity_model.predict(X_test_cond_scaled)

# 4. 提取特征重要性
# Gradient Boosting 特征重要性
diff_feature_importance = best_diffusion_model.feature_importances_
cond_feature_importance = best_conductivity_model.feature_importances_

# 获取特征名
feature_names = X.columns

# 5. 将特征重要性结果保存为CSV文件
# 扩散系数 (Diffusivity) 的特征重要性
diff_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': diff_feature_importance
})
diff_importance_df = diff_importance_df.sort_values(by='Importance', ascending=False)
diff_importance_df.to_csv('diffusivity_feature_importance.csv', index=False)

# 离子导电率 (Conductivity) 的特征重要性
cond_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': cond_feature_importance
})
cond_importance_df = cond_importance_df.sort_values(by='Importance', ascending=False)
cond_importance_df.to_csv('conductivity_feature_importance.csv', index=False)

# 输出文件名
print("Feature importance for Diffusivity saved to 'diffusivity_feature_importance.csv'")
print("Feature importance for Conductivity saved to 'conductivity_feature_importance.csv'")
