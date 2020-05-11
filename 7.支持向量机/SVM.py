import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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
    # 修改label
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]

X, y = create_data()    # 数据集

plt.scatter(X[:50, 0], X[:50, 1], label = '0')
plt.scatter(X[50:, 0], X[50:, 1], label = '1')
plt.legend()
# plt.show()


class SVM:
    def __init__(self, maxIter = 100, C = 200, kernel = "linear"):
        self.maxIter = maxIter
        self.C = C
        self.kernel = kernel

    def init_args(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.m, self.n = X_train.shape

        self.K = self.calcKernel()

        self.alpha = [0] * self.X_train.shape[0]
        self.b = 0
        self.E = [self.calcEi(i) for i in range(self.m)]

        self.supportVecIndex = []

    # 核的初始化
    def calcKernel(self, sigma = 10):
        K = [[0 for _ in range(self.m)] for _ in range(self.m)]    # mxm矩阵
        # 遍历训练集
        for i in range(self.m):
            X = self.X_train[i]
            # K(X,Z)==k(Z,X)，计算一半即可
            # 循环从i开始
            for j in range(i, self.m):
                Z = self.X_train[j]
                # 线性
                if self.kernel == "linear":
                    result = sum([X[k] * Z[k] for k in range(self.n)])
                # 高斯核
                elif self.kernel == "gaussian":
                    X_Z = X - Z    # X-Z
                    result = sum([X_Z[k] * X_Z[k] for k in range(self.n)])     # ||X-Z||^2
                    result = np.exp(-1 * result / (2 * (sigma ** 2)))    # 高斯核
                else:
                    result = None
                    exit('wrong kernel type')
                # 存储结果
                K[i][j] = result
                K[j][i] = result
        return K    # 核矩阵

    # 式7.104：计算g(xi)
    def calc_gxi(self, i):
        gxi = 0    # 初始化
        index = [i for i, aplha in enumerate(self.alpha) if aplha != 0]    # α>0对应的下标
        # 遍历α>0，j为对应的下标
        for j in index:
            gxi += self.alpha[j] * self.y_train[j] * self.K[j][i]
        gxi += self.b    # 偏置b
        return gxi

    # 检查第(Xi,yi)是否满足KKT条件
    def isSatisKKT(self, i):
        gxi = self.calc_gxi(i)
        yi = self.y_train[i]
        yg = gxi * yi
        # 式7.111
        if self.alpha[i] == 0:
            return yg >= 1
        # 式7.112
        elif 0 < self.alpha[i] < self.C:
            return yg == 1
        # 式7.113
        elif self.alpha[i] == self.C:
            return yg <= 1

    # 式7.105：计算Ei
    def calcEi(self, i):
        gxi = self.calc_gxi(i)    # 计算g(xi)
        return gxi - self.y_train[i]    # Ei=g(xi)-yi

    # SMO中的第2个变量（E1是第1个变量的E1，i是第1个变量的下标）
    def getAlphaJ(self, E1, i):
        # 初始化
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        # 遍历训练样本
        for j in range(self.m):
            E2_tmp = self.calcEi(j)
            # |E1-E2|>maxNow
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)    # 更新最大值
                E2 = E2_tmp    # 更新E2
                maxIndex = j    # 更新下标
        return E2, maxIndex

    # SMO算法
    def fit(self, X_train, y_train):
        self.init_args(X_train, y_train)
        iterStep = 0    # 迭代次数
        parameterChanged = 1    # 单轮迭代中有参数改变则置为1
        # （1）没达到限制的迭代次数
        # （2）上轮迭代中有参数改变
        # ——>继续迭代
        # parameterChanged==0——>上轮迭代无参数改变——>收敛状态
        while (iterStep < self.maxIter) and (parameterChanged > 0):
            iterStep += 1
            parameterChanged = 0    # 新一轮将参数改变标志置为0
            # 外层循环（第1个变量的选择）
            for i in range(self.m):
                # 遍历训练样本，检查是否满足KKT条件，不满足则作为SMO中的第1个变量进行优化
                if self.isSatisKKT(i) == False:
                    # (Xi,yi)不满足KKT条件
                    E1 = self.calcEi(i)
                    E2, j = self.getAlphaJ(E1, i)    # 选择第2个变量，使|E1-E2|最大

                    y1 = self.y_train[i]
                    y2 = self.y_train[j]

                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]

                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 无法优化
                    if L == H:
                        continue

                    k11 = self.K[i][i]
                    k22 = self.K[j][j]
                    k21 = self.K[j][i]
                    k12 = self.K[i][j]

                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k12 - 2 * k12)    # 式7.106
                    # 剪辑
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H

                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)    # 式7.109

                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b    # 式7.115
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b    # 式7.116
                    # 确定bNew
                    if 0 < alphaNew_1 < self.C:
                        bNew = b1New
                    elif 0 < alphaNew_2 < self.C:
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2
                    # 更新参数
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew
                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)
                    # α2改变量过小，则认为参数未改变
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged = 1
        # 记录支持向量的下标
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    # 核
    def calcSingKernel(self, X1, X2, sigma = 10):
        # 线性
        if self.kernel == "linear":
            result = sum([X1[k] * X2[k] for k in range(self.n)])
        # 高斯核
        elif self.kernel == "gaussian":
            X_Z = X1 - X2    # X-Z
            result = sum([X_Z[k] * X_Z[k] for k in range(self.n)])    # ||X-Z||^2
            result = np.exp(-1 * result / (2 * sigma ** 2))    # 高斯核
        else:
            result = None
            exit('wrong kernel type')
        return result

    # 预测函数
    def predict(self, X):
        pred = 0    # 求和函数的值
        # 遍历支持向量
        for i in self.supportVecIndex:
            tmp = self.calcSingKernel(self.X_train[i], X)    # K(Xi,X)
            pred += self.alpha[i] * self.y_train[i] * tmp
        pred += self.b    # 偏置b
        return np.sign(pred)    # 决策函数

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

    svm = SVM(kernel = "linear")
    svm.fit(X_train, y_train)    # 训练模型
    result.append(svm.score(X_test, y_test))    # 测试

print('Accuracy on test set: {:.3f}'.format(sum(result) / 100))