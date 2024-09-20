import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. 加载数据并拆分训练集和测试集
file_path = 'C:/Users/Focalors/Desktop/Test/ML_Data.xlsx'  # 实际文件路径
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

# 4. 误差分析：计算残差
# 扩散系数模型的残差
diff_residuals_train = y_train_diff - train_diff_pred
diff_residuals_test = y_test_diff - test_diff_pred

# 导电率模型的残差
cond_residuals_train = y_train_cond - train_cond_pred
cond_residuals_test = y_test_cond - test_cond_pred

# 5. 绘制误差分布图
# 扩散系数模型的残差分布图
plt.figure(figsize=(10, 6))
plt.hist(diff_residuals_test, bins=20, edgecolor='black', alpha=0.7)
plt.title('Residual Distribution for Diffusivity (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 导电率模型的残差分布图
plt.figure(figsize=(10, 6))
plt.hist(cond_residuals_test, bins=20, edgecolor='black', alpha=0.7)
plt.title('Residual Distribution for Conductivity (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 6. 计算残差的标准差
diff_residual_std = np.std(diff_residuals_test)
cond_residual_std = np.std(cond_residuals_test)

print(f"Diffusivity Model Residual Standard Deviation (Test Set): {diff_residual_std}")
print(f"Conductivity Model Residual Standard Deviation (Test Set): {cond_residual_std}")

# 7. 绘制残差 vs. 预测值的散点图
# 扩散系数模型的残差散点图
plt.figure(figsize=(10, 6))
plt.scatter(test_diff_pred, diff_residuals_test, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted for Diffusivity (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# 导电率模型的残差散点图
plt.figure(figsize=(10, 6))
plt.scatter(test_cond_pred, cond_residuals_test, color='green', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted for Conductivity (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
