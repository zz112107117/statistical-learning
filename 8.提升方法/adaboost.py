import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split


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


class AdaBoost:
    def __init__(self, wc_num = 50, interval = 0.1):
        self.wc_num = wc_num    # 弱分类器数目
        self.interval = interval    # 阈值搜索间隔

    def init_args(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.m, self.n = X_train.shape

        self.weights = [1.0 / self.m] * self.m    # 初始化训练数据集的权值分布

        self.alphas = []    # 弱分类器系数
        self.wc_sets = []    # 弱分类器集合

    # 弱分类器（feature：某特征的全部值）
    def calcG(self, feature):
        direct = None    # 分类方向
        compare_array = None    # 分类结果
        error = float("inf")    # 误差初始化为无穷大
        best_v = 0.0    # 最佳阈值

        feature_min = min(feature)    # 特征中的最小值
        feature_max = max(feature)    # 特征中的最大值
        n_step = (feature_max - feature_min) // self.interval    # 迭代次数

        # 搜索最佳阈值
        for i in range(1, int(n_step)):
            v = feature_min + self.interval * i    # 阈值
            # 阈值不等于特征的值
            if v not in feature:
                # 大于v：1，小于v：-1
                compare_positive = [1 if feature[i] > v else -1 for i in range(self.m)]
                weight_error_positive = sum([self.weights[i] for i in range(self.m) if compare_positive[i] != self.y_train[i]])    # 加权误差
                # 大于v：-1，小于v：1
                compare_nagetive = [-1 if feature[i] > v else 1 for i in range(self.m)]
                weight_error_nagetive = sum([self.weights[i] for i in range(self.m) if compare_nagetive[i] != self.y_train[i]])    # 加权误差
                # positive和negative两者中误差较小的
                if weight_error_positive < weight_error_nagetive:
                    tmp_weight_error = weight_error_positive
                    tmp_compare_array = compare_positive
                    tmp_direct = "positive"
                else:
                    tmp_weight_error = weight_error_nagetive
                    tmp_compare_array = compare_nagetive
                    tmp_direct = "nagetive"

                if tmp_weight_error < error:
                    direct = tmp_direct
                    compare_array = tmp_compare_array
                    error = tmp_weight_error
                    best_v = v

        return direct, compare_array, error, best_v

    # 式8.2：弱分类器的系数
    def calcAlpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 式8.5：规范化因子
    def calcZ(self, a, wc_result):
        return sum([self.weights[i] * np.exp(-1 * a * self.y_train[i] * wc_result[i]) for i in range(self.m)])

    # 式8.4：更新训练数据集的权值分布
    def calcW(self, a, wc_result, Z):
        for i in range(self.m):
            self.weights[i] = self.weights[i] / Z * np.exp(-1 * a * self.y_train[i] * wc_result[i])

    # 训练模型
    def fit(self, X_train, y_train):
        self.init_args(X_train, y_train)
        # 所有弱分类器
        for epoch in range(self.wc_num):
            best_direct = None
            wc_result = None
            best_wc_error = float("inf")
            best_v = 0.0
            feature_index = 0

            # 遍历特征，选择对应误差最小的
            for j in range(self.n):
                # 分类方向，分类结果，误差，阈值
                direct, compare_array, error, v = self.calcG(self.X_train[:, j])

                if error < best_wc_error:
                    feature_index = j    # 更新特征索引
                    best_direct = direct    # 更新分类方向
                    best_wc_error = error    # 更新误差
                    best_v = v    # 更新阈值
                    wc_result = compare_array    # 更新分类结果


            # 弱分类器的系数
            a = self.calcAlpha((best_wc_error))
            self.alphas.append(a)

            self.wc_sets.append((feature_index, best_v, best_direct))    # 记录弱分类器

            Z = self.calcZ(a, wc_result)    # 规范化因子
            self.calcW(a, wc_result, Z)    # 更新训练数据集的权值分布

    # 弱分类器
    def G(self, feature, v, direct):
        if direct == 'positive':
            return 1 if feature > v else -1
        else:
            return -1 if feature > v else 1

    # 预测函数
    def predict(self, X):
        result = 0.0
        # 遍历弱分类器
        for i in range(len(self.wc_sets)):
            feature_index, v, direct = self.wc_sets[i]
            result += self.alphas[i] * self.G(X[feature_index], v, direct)    # 弱分类器线性组合
        return 1 if result > 0 else -1

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

    ab = AdaBoost()
    ab.fit(X_train, y_train)    # 训练模型
    result.append(ab.score(X_test, y_test))    # 测试

print('Accuracy on test set: {:.3f}'.format(sum(result) / 100))