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
best_conductivity_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

# 训练 Diffusivity 的 Gradient Boosting 模型
best_diffusion_model.fit(X_train_diff_scaled, y_train_diff)
train_diff_pred = best_diffusion_model.predict(X_train_diff_scaled)
test_diff_pred = best_diffusion_model.predict(X_test_diff_scaled)

# 训练 Conductivity 的 XGBoost 模型
best_conductivity_model.fit(X_train_cond_scaled, y_train_cond)
train_cond_pred = best_conductivity_model.predict(X_train_cond_scaled)
test_cond_pred = best_conductivity_model.predict(X_test_cond_scaled)

# 4. 绘制预测值与实际值比较图
# 扩散系数 (Diffusivity)
# 绘制 Diffusivity 的训练集实际值与预测值比较图
plt.figure(figsize=(7, 6))
plt.scatter(y_train_diff, train_diff_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_train_diff.min(), y_train_diff.max()], [y_train_diff.min(), y_train_diff.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title('Gradient Boosting - Diffusivity (Training Set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制 Diffusivity 的测试集实际值与预测值比较图
plt.figure(figsize=(7, 6))
plt.scatter(y_test_diff, test_diff_pred, color='green', alpha=0.6, label='Predictions')
plt.plot([y_test_diff.min(), y_test_diff.max()], [y_test_diff.min(), y_test_diff.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title('Gradient Boosting - Diffusivity (Test Set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 离子导电率 (Conductivity)
# 绘制 Conductivity 的训练集实际值与预测值比较图
plt.figure(figsize=(7, 6))
plt.scatter(y_train_cond, train_cond_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_train_cond.min(), y_train_cond.max()], [y_train_cond.min(), y_train_cond.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title('XGBoost - Conductivity (Training Set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制 Conductivity 的测试集实际值与预测值比较图
plt.figure(figsize=(7, 6))
plt.scatter(y_test_cond, test_cond_pred, color='green', alpha=0.6, label='Predictions')
plt.plot([y_test_cond.min(), y_test_cond.max()], [y_test_cond.min(), y_test_cond.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title('XGBoost - Conductivity (Test Set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 保存实际值和预测值为 CSV 文件
# 将训练集和测试集的实际值和预测值保存为CSV文件
train_diff_df = pd.DataFrame({'Actual': y_train_diff, 'Predicted': train_diff_pred})
test_diff_df = pd.DataFrame({'Actual': y_test_diff, 'Predicted': test_diff_pred})
train_cond_df = pd.DataFrame({'Actual': y_train_cond, 'Predicted': train_cond_pred})
test_cond_df = pd.DataFrame({'Actual': y_test_cond, 'Predicted': test_cond_pred})

# 保存CSV文件
train_diff_df.to_csv('diffusivity_training_results.csv', index=False)
test_diff_df.to_csv('diffusivity_testing_results.csv', index=False)
train_cond_df.to_csv('conductivity_training_results.csv', index=False)
test_cond_df.to_csv('conductivity_testing_results.csv', index=False)
