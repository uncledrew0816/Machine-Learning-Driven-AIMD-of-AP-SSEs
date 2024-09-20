import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

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

# 将标准化后的数据转换回 DataFrame，并保留特征名称
X_train_diff_scaled = pd.DataFrame(X_train_diff_scaled, columns=X.columns)
X_test_diff_scaled = pd.DataFrame(X_test_diff_scaled, columns=X.columns)
X_train_cond_scaled = pd.DataFrame(X_train_cond_scaled, columns=X.columns)
X_test_cond_scaled = pd.DataFrame(X_test_cond_scaled, columns=X.columns)

# 3. 定义和训练模型
# 定义模型
best_diffusion_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
best_conductivity_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

# 训练 Diffusivity 的 Gradient Boosting 模型
best_diffusion_model.fit(X_train_diff_scaled, y_train_diff)

# 训练 Conductivity 的 XGBoost 模型
best_conductivity_model.fit(X_train_cond_scaled, y_train_cond)

# 4. 使用 SHAP 进行解释
# 对 Gradient Boosting 模型使用 SHAP
explainer_diff = shap.Explainer(best_diffusion_model, X_train_diff_scaled)
shap_values_diff = explainer_diff(X_test_diff_scaled)

# 对 XGBoost 模型使用 SHAP
explainer_cond = shap.Explainer(best_conductivity_model, X_train_cond_scaled)
shap_values_cond = explainer_cond(X_test_cond_scaled)

# 5. 绘制 SHAP summary plot
# 扩散系数模型的 SHAP 总结图
shap.summary_plot(shap_values_diff, X_test_diff_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot - Diffusivity")

# 导电率模型的 SHAP 总结图
shap.summary_plot(shap_values_cond, X_test_cond_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot - Conductivity")

# 6. 绘制单个样本的 SHAP force plot
# 随机选择一个样本
sample_index = 0  # 替换为你想要查看的样本索引

# 扩散系数模型的 SHAP force plot
shap.force_plot(explainer_diff.expected_value, shap_values_diff[sample_index].values, X_test_diff_scaled.iloc[sample_index,:], feature_names=X.columns)

# 导电率模型的 SHAP force plot
shap.force_plot(explainer_cond.expected_value, shap_values_cond[sample_index].values, X_test_cond_scaled.iloc[sample_index,:], feature_names=X.columns)
