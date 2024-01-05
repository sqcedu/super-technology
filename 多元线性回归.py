import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score,\
mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

# 1.读取数据
data = pd.read_csv("Advertising.csv")
print(data.head())
print("shape:",data.shape)


# 2.分析数据
sns.pairplot(data, x_vars=["TV","radio","newspaper"], y_vars="sales",height=5,aspect=0.8,kind="reg")
plt.show()


# 3.建立线性回归模型

# （1）使用 pandas 构建 X（特征向量）和 y（标签列）
feature_cols = ["TV","radio","newspaper"]
X = data[feature_cols]
y = data["sales"]

# （2）构建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)  # 25% 测试

# （3）构建线性回归模型并训练
model = LinearRegression().fit(X_train,y_train)

# （4）输出模型结果
print("截距：",model.intercept_)
coef = zip(feature_cols, model.coef_)
print("回归系数：",list(coef))


# 4. 预测
y_pred = model.predict(X_test)


# 5. 评价
# 这个是自己写函数计算
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test.values[i])**2
sum_erro = np.sqrt(sum_mean/len(y_test))
print("均方根误差（RMSE）：",sum_erro)

# 这个是调用已有函数,以后就直接用
print("平均绝对误差（MAE）：",mean_absolute_error(y_test,y_pred))
print("均方误差（MSE）：",mean_squared_error(y_test,y_pred))
print("中值绝对误差：",median_absolute_error(y_test,y_pred))
print("可解释方差：",explained_variance_score(y_test,y_pred))
print("R方值：",r2_score(y_test,y_pred))

# 绘制 ROC 曲线
plt.plot(range(len(y_pred)),y_pred,"b",label="predict")
plt.plot(range(len(y_pred)),y_test,"r",label="test")
plt.xlabel("number of sales")
plt.ylabel("value of sales")
plt.legend(loc="upper right")
plt.show()
