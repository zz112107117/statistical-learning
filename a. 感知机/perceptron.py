import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# 加载数据
df = pd.DataFrame(iris.data, columns = feature_names)
df['label'] = iris.target

# 二分类，线性可分
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# 数据集
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])    # 适应loss

class Perceptron:
    def __init__(self, n):
        # 初值
        self.w = np.ones(n, dtype = np.float32)
        self.b = 0
        self.l_rate = 0.1

    def affine(self, x, w, b):
        return np.dot(x, w) + b

    # SGD
    def fit(self, X_train, y_train):
        wrong = True
        while wrong:
            wrong_count = 0
            # 训练集中选取样本
            for X, y in zip(X_train, y_train):
                # 更新参数
                if y * self.affine(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            # 直至训练集中没有误分类点
            if wrong_count == 0:
                wrong = False

# 训练模型
perceptron = Perceptron(len(data[0]) - 1)
perceptron.fit(X, y)

# 数据集
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
# 决策边界（w1x+w2y+b=0->y=-(w1x+b)/w2）
x_ = np.linspace(4, 7, 5)
y_ = -(perceptron.w[0] * x_ + perceptron.b) / perceptron.w[1]
plt.plot(x_, y_, color = 'black')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


from sklearn.linear_model import Perceptron

p = Perceptron(fit_intercept = True, max_iter = 100, shuffle = True)
p.fit(X, y)

# 权重
print(p.coef_)
# 偏置
print(p.intercept_)

# 数据集
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
# 决策边界（w1x+w2y+b=0->y=-(w1x+b)/w2）
x_ = np.linspace(4, 7, 5)
y_ = -(p.coef_[0][0] * x_ + p.intercept_) / p.coef_[0][1]
plt.plot(x_, y_, color = 'black')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

