import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

class KNN:
    def __init__(self, X_train, y_train, k, p):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def predict(self, test_point):
        # 计算范数（距离）
        distances = {
                     '{}'.format(X):
                     (y, np.linalg.norm(test_point - X, ord = self.p))
                     for X, y in zip(self.X_train, self.y_train)
                    }
        # 按照距离升序排序
        distances = sorted(distances.items(), key = lambda x:x[1][1])

        count = 0    # 收录的数量
        knn_list = []
        for each in distances:
            if count == self.k:
                break
            count += 1
            knn_list.append((each[1][0], each[1][1]))

        # 统计每类个数
        kclass = [k[0] for k in knn_list]
        # 类别-数量
        count_pairs = Counter(kclass)
        # 数量最多的类
        max_count = sorted(count_pairs.items(), key = lambda x:x[1])[-1][0]
        return max_count

    def acc(self, X_test, y_test):
        right_count = 0

        for X, y in zip(X_test, y_test):
            pred = self.predict(X)
            if pred == y:
                right_count += 1

        return right_count / len(X_test)

knn = KNN(X_train, y_train, 3, 2)
print('\nAccuracy on test set: {}'.format(knn.acc(X_test, y_test)))
# 测试样本点
X, y = X_test[0], y_test[0]
print('Test X: {}, y: {}, pred: {}'.format(X, y, knn.predict(X)))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
# 测试样本点的坐标
plt.scatter(X[0], X[1], c = 'black', marker = 'x', label = 'test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()