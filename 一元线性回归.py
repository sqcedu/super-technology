import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# 读取数据的函数
def get_data(file_name):
    data = pd.read_csv(file_name)
    X = []
    Y = []
    for square_feet, price in zip(data["square_feet"],data["price"]):
        X.append([square_feet])
        Y.append(price)
    return X,Y

# 建立线性模型，并进行预测
def get_linear_model(X, Y, predict_value):
    model = linear_model.LinearRegression().fit(X,Y)
    pre = model.predict(predict_value)
    predictions = {}
    predictions["intercept"] = model.intercept_  # 截距值   
    predictions["coefficient"] = model.coef_     # 回归系数（斜率）
    predictions["predictted_value"] = pre
    return predictions

# 显示线性拟合模型结果
def show_linear_line(X,Y):
    model = linear_model.LinearRegression().fit(X,Y)
    plt.scatter(X,Y)
    plt.plot(X,model.predict(X),color="red")
    plt.title("Prediction of House")
    plt.xlabel("square feet")
    plt.ylabel("price")
    plt.show()  

# 定义主函数
def main():
    X, Y = get_data("input_data.csv")
    print("X:",X)
    print("Y:",Y)
    predictions = get_linear_model(X,Y,[[700]])
    print(predictions)
    show_linear_line(X,Y)
    
main()
