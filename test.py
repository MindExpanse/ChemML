# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris

# 1. 加载数据集
# 使用sklearn自带的Iris数据集
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 显示前几行数据
# print(df.head())

print(df.describe())

print(df['target'].value_counts())

# 计算相关性矩阵
correlation_matrix = df.drop('target', axis=1).corr()
# 绘制相关性矩阵
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
# plt.xlabel("Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=600)
plt.show()


# 2. 数据预处理
# 我们对数据进行标准化处理，因为部分机器学习模型对数据的尺度比较敏感
X = df.drop('target', axis=1)  # 特征
y = df['target']  # 标签

# 数据分割：训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 探索性数据分析（EDA）
# 3.1 特征分布可视化
sns.pairplot(df, hue='target')
# plt.title("Pairplot of Iris Dataset")
plt.savefig("pairplot.png", dpi=600)
plt.show()


# 3.2 箱型图
plt.figure(figsize=(12, 6))
sns.boxplot(x='target', y='sepal length (cm)', data=df)
plt.title("Boxplot of Sepal Length by Target Class")
plt.savefig("boxplot.png", dpi=600)
plt.show()


# 4. 模型训练与评估

# 4.1 使用随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4.2 交叉验证
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10)
print(f"Cross-validated Accuracy: {cv_scores.mean()} ± {cv_scores.std()}")

# 5. 模型调优
# 5.1 设置超参数调优的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# 5.2 执行网格搜索
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
# 输出最佳超参数和交叉验证分数
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-validation Score: ", grid_search.best_score_)
# 使用最佳参数重新训练模型
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test_scaled)
# 输出基于最佳超参数的分类报告
print("Classification Report (with best parameters):")
print(classification_report(y_test, y_pred_best))

# 6. 可视化模型结果（特征重要性）
feature_importances = best_rf_model.feature_importances_
# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=df.columns[:-1], y=feature_importances)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig("feature_importance.png", dpi=600)
plt.show()

