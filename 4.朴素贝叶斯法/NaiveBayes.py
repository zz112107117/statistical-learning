import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# 加载数据
df = pd.DataFrame(iris.data, columns = feature_names)
df['label'] = iris.target

# 数据集
data = np.array(df.iloc[:100, :])
X, y = data[:, :-1], data[:, -1]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# 连续变量->高斯朴素贝叶斯
class naiveBayes:
    def __init__(self):
        self.labels = None    # 类别
        self.prior = None    # 先验
        self.mean = None    # P(Xi|Y=yj)对应的均值
        self.stdev = None    # P(Xi|Y=yj)对应的标准差

    # P(X^i|Y=yj)均值
    def cal_mean(self, train_data):
        means = []
        for data in zip(*train_data):
            value = sum(data) / float(len(data))
            means.append(value)
        return means

    # P(X^i|Y=yj)标准差
    def cal_stdev(self, train_data):
        stdevs = []
        for data in zip(*train_data):
            mean = sum(data) / float(len(data))
            value = math.sqrt(sum([pow(each - mean, 2) for each in data]) / float(len(data)))
            stdevs.append(value)
        return stdevs

    # P(X^i=x^i|Y=yj)
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 模型参数
    def fit(self, X, y):
        self.labels = list(set(y))
        # 把数据按类别分离
        # 统计先验概率
        data = {label:[] for label in self.labels}
        self.prior = {label:0 for label in self.labels}
        for value, label in zip(X, y):
            data[label].append(value)
            self.prior[label] += (1 / len(X))

        self.mean = {label:self.cal_mean(value) for label, value in data.items()}
        self.stdev = {label:self.cal_stdev(value) for label, value in data.items()}

    # 后验概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        # 每类
        for label in self.labels:
            probabilities[label] = self.prior[label]
            # P(X^1=x^1|Y=label)P(X^2=x^2|Y=label)...P(X^i=x^i|Y=label)
            for i in range(len(input_data)):
                mean, stdev = self.mean[label][i], self.stdev[label][i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 分类（后验概率最大）
    def predict(self, X):
        return sorted(self.calculate_probabilities(X).items(), key = lambda x:x[1])[-1][0]

    def acc(self, X_test, y_test):
        right_count = 0    # 正确分类
        for X, y in zip(X_test, y_test):
            pred = self.predict(X)
            # 预测正确
            if pred == y:
                right_count += 1
        return right_count / float(len(X_test))

# 训练模型
nb = naiveBayes()
nb.fit(X_train, y_train)

print('Accuracy on test set: {}'.format(nb.acc(X_test, y_test)))
# 测试样本点
X, y = X_test[0], y_test[0]
print('Test X: {}, y: {}, pred: {}'.format(X, y, nb.predict(X)))


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
# 训练模型
nb.fit(X_train, y_train)

print('\nAccuracy on test set: {}'.format(nb.score(X_test, y_test)))
# 测试样本点
X, y = X_test[0], y_test[0]
print('Test X: {}, y: {}, pred: {}'.format(X, y, nb.predict([X])))