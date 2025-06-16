import os.path

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
# matplotlib.use('tkagg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

fig_dir = 'output/Fig3'

os.makedirs(fig_dir, exist_ok=True)

# 读取数据
# data = pd.read_excel('data20240102.xlsx', sheet_name=2)[['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
data = pd.read_excel('data20240401.xlsx', sheet_name=0)[['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
# data = pd.read_excel('/home/liyuan/Pycharm/pythonProject/3DChemical/3D/data-0819.xlsx', skiprows=[0])[['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
# data = pd.read_excel('/home/liyuan/Pycharm/pythonProject/3DChemical/3D/data0705.xlsx', skiprows=[0])[['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
# data = pd.read_excel('/home/liyuan/Pycharm/pythonProject/3DChemical/3D/data0705.xlsx', skiprows=[0])[['Ueff', 'θ1', 'd1', 'θ2', 'd2']]
# all_data = pd.read_excel('/home/liyuan/Pycharm/pythonProject/3DChemical/3D/513.xlsx')[['Ueff']].dropna().values
# 删除自变量或因变量为空的样本
data.dropna(subset=['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å'], inplace=True)

# 查看前几行数据
print(data.head())

# 查看基本统计信息
print(data.describe())

# 绘制单变量直方图
sns.histplot(data['Ueff/K'], binwidth=50)
plt.xlabel('Ueff/K')
plt.ylabel('Frequency')
plt.title('Histogram of Ueff/K')
plt.savefig(os.path.join(fig_dir, 'Histogram-2.png'), dpi=900, bbox_inches='tight')
plt.show()

# 绘制单变量直方图
sns.histplot(np.log(data['Ueff/K']))
plt.xlabel('Ueff_log/K')
plt.ylabel('Frequency')
plt.title('Histogram of Ueff_log/K')
plt.show()

# # 绘制单变量直方图
# sns.histplot(all_data)
# plt.xlabel('Ueff/K')
# plt.ylabel('Frequency')
# plt.title('Histogram of all Ueff/K')
# plt.show()

# 绘制单变量密度图
sns.kdeplot(data['Ueff/K'])
plt.xlabel('Ueff/K')
plt.ylabel('Density')
plt.title('Density Plot of Ueff/K')
plt.show()

# 提取自变量和因变量
features = data[['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
target = data['Ueff/K']

# 绘制箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=features)
plt.title('Boxplot of Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.savefig(os.path.join(fig_dir, 'Boxplot-2.png'), dpi=900, bbox_inches='tight')
plt.show()

# 绘制箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=target)
plt.title('Boxplot of Ueff')
plt.xlabel('Ueff')
plt.ylabel('Values')
plt.show()

# # 绘制箱型图
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=all_data)
# plt.title('Boxplot of all Ueff')
# plt.xlabel('Ueff')
# plt.ylabel('Values')
# plt.show()

# 绘制箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=np.log(target))
plt.title('Boxplot of log Ueff')
plt.xlabel('Ueff')
plt.ylabel('Values')
plt.show()

# # 绘制箱型图
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=np.log(all_data))
# plt.title('Boxplot of all Ueff')
# plt.xlabel('Ueff')
# plt.ylabel('Values')
# plt.show()

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制相关性热图
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(fig_dir, 'Correlation-2.png'), dpi=900, bbox_inches='tight')
plt.show()

# 绘制自变量和因变量之间的散点图
sns.pairplot(data, x_vars=['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å'], y_vars=['Ueff/K'], kind='scatter')
plt.show()

