import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.feature_names)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# 加载数据
df = pd.DataFrame(iris.data, columns = feature_names)
df['label'] = iris.target

# 数据集
data = np.array(df.iloc[:100, [0, 1, -1]])
# 划分训练集和测试集
train, test = train_test_split(data, test_size = 0.3)
x0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
x1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])

# 树节点
class Node:
    def __init__(self, data, depth, lchild = None, rchild = None):
        self.data = data
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild

class KDTree:
    def __init__(self, m):
        self.k = 0    # k维空间
        self.m = m    # m近邻
        self.root = None    # kd数根节点
        self.nearest = None

    # 构造平衡kd树
    def create(self, dataset, depth = 0):
        # 两个子区域没有实例存在时停止
        if len(dataset) <= 0:
            return None
        n, k = np.shape(dataset)    # n是样本数，k是维度空间
        self.k = k - 1    # 删去label
        axis = depth % self.k    # 作为坐标轴的维
        dataset = sorted(dataset, key = lambda x: x[axis])    # 排序
        # 中位数作为切分点，构造根节点
        mid = int(n / 2)
        node = Node(dataset[mid], depth)
        # kd树根节点
        if depth == 0:
            self.root = node
        # 生成深度为depth+1的左右子节点
        node.lchild = self.create(dataset[:mid], depth + 1)    # 左闭右开
        node.rchild = self.create(dataset[mid + 1:], depth + 1)
        return node

    # 前序遍历
    def preOrder(self, node):
        if node is None:
            return
        print(node.depth, node.data)
        self.preOrder(node.lchild)
        self.preOrder(node.rchild)

    # 搜索kd树
    def search(self, x):
        nearest = []
        # m近邻初始化
        for i in range(self.m):
            nearest.append([-1, None])    # [距离，节点]
        self.nearest = np.array(nearest)

        def recurve(node):
            # 到达叶节点
            if node is None:
                return
            axis = node.depth % self.k    # 当前维
            dis = x[axis] - node.data[axis]
            # 目标点x当前维坐标小于切分点的坐标，移动到左子节点
            if dis < 0:
                recurve(node.lchild)
            # 否则移动到右子节点
            else:
                recurve(node.rchild)

            # 当前最近点的距离
            dist = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(x, node.data)))
            # 更新self.nearest
            for i, d in enumerate(self.nearest):
                if d[0] < 0 or dist < d[0]:
                    # 满足条件的点插入到指定位置
                    self.nearest = np.insert(self.nearest, i, [dist, node], axis = 0)
                    # 去除无效点
                    self.nearest = self.nearest[:-1]
                    break

            num = list(self.nearest[:, 0]).count(-1)    # 统计-1个数
            # 更新self.nearest
            if self.nearest[-num - 1, 0] > abs(dis):
                if dis < 0:
                    recurve(node.rchild)
                else:
                    recurve(node.lchild)

        recurve(self.root)

        res = [each.data[-1] for each in self.nearest[:, 1]]    # 距离最近的k个节点的类别
        return max(res, key = res.count)

    def acc(self, test):
        right_count = 0

        for X in test:
            pred = self.search(X[:-1])
            if pred == X[-1]:
                right_count += 1

        return right_count / len(test)

kdtree = KDTree(8)
kdtree.create(train)
kdtree.preOrder(kdtree.root)

print('\nAccuracy on test set: {}'.format(kdtree.acc(test)))
# 测试样本点
instance, y = test[0], test[0][-1]
print('Test X: {}, y: {}, pred: {}'.format(instance[:-1], y, kdtree.search(instance)))