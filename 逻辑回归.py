import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1.读取数据
data = pd.read_csv("data.csv")
print(data.head())
print("shape:",data.shape)

# 2.建立逻辑回归模型
#（1）构建 X（特征向量）和 y（标签列）
feature_cols = ["x1","x2"]
X = data[feature_cols]
y = data["label"]

#（2）构建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)  # 25% 测试

#（3）构建逻辑回归模型并训练
model = LogisticRegression().fit(X_train,y_train)

#（4）输出模型结果
print("截距：",model.intercept_)
print("回归系数：",model.coef_)

# 3.预测
y_pred = model.predict([[0.5564,-1.5543]])
print("预测类别：",y_pred)

# 4.评价模型准确率
model.score(X_test,y_test)
