# iris_classification.py
# 更新了一下，尝试....
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 转换为 DataFrame，方便处理
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# 数据探索：查看前几行
print("数据集前5行：")
print(df.head())

# 可视化：特征两两散点图
sns.pairplot(df, hue='species')
plt.savefig('iris_pairplot.png')  # 保存可视化结果
plt.close()

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练模型：随机森林
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率：{accuracy:.2f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=target_names))

# 保存特征重要性图
importance = model.feature_importances_
plt.bar(feature_names, importance)
plt.title('特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.savefig('feature_importance.png')
plt.close()

print("可视化结果已保存为 'iris_pairplot.png' 和 'feature_importance.png'")