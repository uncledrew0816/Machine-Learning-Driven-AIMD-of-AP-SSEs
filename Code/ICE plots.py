import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from pycebox.ice import ice, ice_plot

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

# 4. 绘制 Individual Conditional Expectation (ICE) 图
# 选择特征 'Li atom count' 进行ICE分析
feature_to_plot = 'Li atom count'  # 确保这个特征在数据中存在

# 检查特征是否存在于数据集中
if feature_to_plot not in X_train_diff_scaled.columns:
    raise ValueError(f"'{feature_to_plot}' 不在特征集中，请检查数据")

# 对扩散系数的Gradient Boosting 模型绘制ICE图
ice_df_diff = ice(best_diffusion_model, X_train_diff_scaled, feature_to_plot)
ice_plot(ice_df_diff, frac_to_plot=0.1)  # 绘制10%的数据子集以提高可视化性能
plt.title('Gradient Boosting - Diffusivity ICE Plot for {}'.format(feature_to_plot))
plt.show()

# 对导电率的XGBoost 模型绘制ICE图
ice_df_cond = ice(best_conductivity_model, X_train_cond_scaled, feature_to_plot)
ice_plot(ice_df_cond, frac_to_plot=0.1)
plt.title('XGBoost - Conductivity ICE Plot for {}'.format(feature_to_plot))
plt.show()
