import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    # 加载数据
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    df = pd.DataFrame(iris.data, columns = feature_names)
    df['label'] = iris.target
    # 保证线性可分
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]

X, y = create_data()    # 数据集


class maxEntropy:
    def __init__(self, maxIter = 100):
        self.maxIter = maxIter

    def init_args(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.m, self.n = X_train.shape    # 训练集样本数，特征数

        self.xy_num, self.fixy, self.ixy_num = self.calcFixy()
        self.xy2id, self.id2xy = self.createSearchDict()    # i(x,y)->id；id->i(x,y)

        self.w = [0] * self.xy_num    # Pw(y|x)中的w

    # fi(x,y)
    def calcFixy(self):
        fixy = [defaultdict(int) for _ in range(self.n)]
        ixy_num = [defaultdict(int) for _ in range(self.n)]
        # 遍历特征
        for i in range(self.n):
            # 遍历训练集
            for j in range(self.m):
                fixy[i][(self.X_train[j][i], self.y_train[j])] = 1    # 记录fi(x,y)
                ixy_num[i][(self.X_train[j][i], self.y_train[j])] += 1    # 统计i(x,y)数目
        # 统计(x,y)总数
        total = 0
        for i in fixy:
            total += len(i)
        return total, fixy, ixy_num

    # 查询字典
    def createSearchDict(self):
        xy2id = [{} for _ in range(self.n)]    # i(x,y)->id
        id2xy = {}    # id->i(x,y)

        index = 0    # id
        # 遍历特征
        for i in range(self.n):
            # 遍历fi(x,y)
            for (x, y) in self.fixy[i]:
                xy2id[i][(x, y)] = index
                id2xy[index] = (x, y)
                index += 1    # 更新id
        return xy2id, id2xy

    # 式6.22：计算Pw(y|x)
    def calcPwy_x(self, X, y):
        numerator = 0    # 分子
        Z = 0    # 分母
        # 遍历特征
        for i in range(self.n):
            # 训练集中出现i(X,y)
            if (X[i], y) in self.xy2id[i]:
                index = self.xy2id[i][(X[i], y)]    # fi(X,y)对应的wi
                numerator += self.w[index]    # wi*fi(X,y)，其中fi(X,y)=1
            # 另一种标签
            if (X[i], 1 - y) in self.xy2id[i]:
                index = self.xy2id[i][(X[i], 1 - y)]    # fi(X,1-y)对应的wi
                Z += self.w[index]    # wi*fi(X,1-y)，其中fi(X,1-y)=1
        numerator = np.exp(numerator)    # 分子
        Z = np.exp(Z) + numerator    # 分母
        return numerator / Z    # Pw(y|x)

    # Ep(fi(x,y))
    def calcEpixy(self):
        Epixy = [0] * self.xy_num
        # 遍历训练集
        for j in range(self.m):
            Pwxy = [0] * 2
            Pwxy[0] = self.calcPwy_x(self.X_train[j], 0)    # Pw(y=0|x)
            Pwxy[1] = self.calcPwy_x(self.X_train[j], 1)    # Pw(y=1|x)
            # 遍历特征
            for i in range(self.n):
                # 遍历标签（0，1）
                for y in range(2):
                    # fi(x,y)=1
                    if (self.X_train[j][i], y) in self.fixy[i]:
                        id = self.xy2id[i][(self.X_train[j][i], y)]    # i(x,y)->id
                        Epixy[id] += (1 / self.m) * Pwxy[y]    # P_(x)*P(y|x)*fi(x,y)
        return Epixy

    # Ep_(fi(x,y))
    def calcEp_ixy(self):
        Ep_ixy = [0] * self.xy_num
        # 遍历特征
        for i in range(self.n):
            # 遍历特征i中的(x,y)
            for (x, y) in self.fixy[i]:
                id = self.xy2id[i][(x, y)]    # i(x,y)->id
                Ep_ixy[id] += self.ixy_num[i][(x, y)] / self.m    # P_(x,y)*fi(x,y)
        return Ep_ixy

    # IIS算法
    def fit(self, X_train, y_train):
        self.init_args(X_train, y_train)

        Epixy = self.calcEpixy()
        Ep_ixy = self.calcEp_ixy()

        for _ in range(self.maxIter):
            sigmaList = [0] * self.xy_num    # 参数更新量
            for j in range(self.xy_num):
                M = 0    # 式6.34中的M
                for i in range(self.n):
                    M += self.fixy[i][self.id2xy[j]]
                sigmaList[j] = (1 / M) * np.log(Ep_ixy[j] / Epixy[j])    # 式6.34

            self.w = [self.w[i] + sigmaList[i] for i in range(self.xy_num)]    # 更新参数

    # 预测函数
    def predict(self, X):
        pred = [0] * 2    # 标签（0，1）
        for i in range(2):
            pred[i] = self.calcPwy_x(X, i)    # Pw(y|x)
        return pred.index(max(pred))    # 使Pw(y|x)达到最大的y

    # 测试集accuracy
    def score(self, X_test, y_test):
        correct = 0    # 分类正确的样本数
        # 遍历测试集
        for i in range(len(X_test)):
            pred = self.predict(X_test[i])    # 预测值
            # 分类正确
            if pred == y_test[i]:
                correct += 1
        return correct / len(X_test)    # accuracy


result = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)    # 划分训练集和测试集

    max_entropy = maxEntropy()
    max_entropy.fit(X_train, y_train)    # 训练模型
    result.append(max_entropy.score(X_test, y_test))    # 测试

print('Accuracy on test set: {:.3f}'.format(sum(result) / 100))