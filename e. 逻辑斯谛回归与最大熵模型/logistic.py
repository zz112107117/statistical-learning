import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = np.array(iris.data)[:100, 0:2], np.array(iris.target)[:100]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

class logisticRegression:
    def __init__(self, epoch, learning_rate):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = None

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # 增广（w.x+b->~w.~x）
    def data_matrix(self, X):
        X = X.tolist()
        data_mat = []
        # 增广一维
        for each in X:
            data_mat.append(each + [1.0])
        return data_mat

    def fit(self, X_train, y_train):
        data_mat = self.data_matrix(X_train)
        # 权重初始化
        self.weights = np.zeros((len(data_mat[0]), 1), dtype = np.float32)

        for _ in range(self.epoch):
            for X, y in zip(data_mat, y_train):
                # 梯度下降
                pred = self.sigmoid(np.dot(X, self.weights))
                self.weights -= self.learning_rate * (pred - y) * np.transpose([X])

    def acc(self, X_test, y_test):
        right_count = 0
        X_test = self.data_matrix(X_test)
        for X, y in zip(X_test, y_test):
            pred = np.dot(X, self.weights)
            if (pred > 0 and y == 1) or (pred < 0 and y == 0):
                right_count += 1
        return right_count / len(X_test)

lr = logisticRegression(100, 0.01)
lr.fit(X_train, y_train)
print(lr.acc(X_test, y_test))

x_ponits = np.arange(4, 8)
y_ = -(lr.weights[0] * x_ponits + lr.weights[2])/lr.weights[1]
plt.plot(x_ponits, y_)

plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
plt.show()