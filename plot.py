import matplotlib.pyplot as plt

# 模型名称
models = ['SVR', 'ExtraTrees', 'CNN', 'AdaBoost', 'Bagging', 'RandomForest', 'KRR', 'Catboost', 'XGBoost']

# 对应的R^2值
r_squared_values = [0.7801, 0.8856, 0.9013, 0.8342, 0.8942, 0.8406, 0.8993, 0.8997, 0.8842]

# 调整图形大小
plt.figure(figsize=(8, 6))

# 创建柱状图
plt.bar(models, r_squared_values, color='#86bf91', edgecolor='white', width=0.5)

# 添加标题和标签
plt.title('$R^2$ Values for Different Models')  # 使用LaTeX语法添加上标
plt.xlabel('Model Name')
plt.ylabel('$R^2$')

# 调整横轴标签的倾斜角度
plt.xticks(rotation=25, ha='right')

# 调整纵轴坐标范围
plt.ylim(0.5, 1)

# 添加横线
plt.yticks([i/10 for i in range(5, 11)])  # 在纵轴上每隔0.1添加一条横线

# 隐藏边框
plt.box(on=None)

# 显示网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 增加横轴标签的间距
plt.tight_layout()

# 显示图形
plt.show()
