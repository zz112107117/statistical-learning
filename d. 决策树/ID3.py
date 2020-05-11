import pandas as pd
import numpy as np
import math


feature_names = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
data = [['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否']]
df = pd.DataFrame(data, columns = feature_names)    # 加载数据


# 节点
class Node:
    def __init__(self, isSingle = True, label = None, feature_index = None, feature = None):
        self.isSingle = isSingle
        self.label = label

        self.feature_index = feature_index
        self.feature = feature

        self.subTree = {}

        self.result = {
            'label': self.label,
            'feature': self.feature,
            'subTree': self.subTree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def addSubTree(self, feature_value, sub_tree):
        self.subTree[feature_value] = sub_tree


# 决策树
class DTree:
    def __init__(self, eta = 0.1):
        self.eta = eta
        self.root = {}

    # 标签的熵
    def entropy(self, data):
        count = len(data)    # 实例数
        label_count = {}    # 各标签对应的实例数
        # 遍历数据集
        for i in range(count):
            label = data[i][-1]    # 实例i对应的标签
            # 统计
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        return -sum([(value / count) * math.log(value / count, 2) for value in label_count.values()])    # 计算熵

    # 经验条件熵
    def condEntropy(self, data, feature_index):
        count = len(data)    # 实例数
        feature_count = {}    # 特征[feature_index]的各个值对应的全部实例
        # 遍历数据集
        for i in range(count):
            feature_value = data[i][feature_index]
            # 统计
            if feature_value not in feature_count:
                feature_count[feature_value] = []
            feature_count[feature_value].append(data[i])
        return sum([(len(d) / count) * self.entropy(d) for d in feature_count.values()])    # 计算经验条件熵

    # 信息增益
    def infoGain(self, ent, cond_ent):
        return ent - cond_ent

    # 最大信息增益
    def maxInfoGain(self, data):
        count = len(data[0]) - 1    # 特征数
        ent = self.entropy(data)    # 标签的熵
        features_gain = []    # 各个特征对应的信息增益
        # 遍历所有特征
        for i in range(count):
            i_info_gain = self.infoGain(ent, self.condEntropy(data, feature_index = i))    # 特征i的信息增益
            features_gain.append((i, i_info_gain))

            print("特征：{:10s}信息增益：{:.3f}".format(feature_names[i], i_info_gain))

        max_info_gain = max(features_gain, key = lambda x: x[-1])    # 最大信息增益

        print("\n*** '{}'有最大信息增益，作为节点特征 ***\n\n".format(feature_names[max_info_gain[0]]))

        return max_info_gain

    # ID3算法
    def ID3(self, data):
        features, y = data.columns[: -1], data.iloc[:, -1]

        # 数据集中所有实例属于同一类Ck，T为单节点树，将类Ck作为该结点的类标记，返回T
        if len(y.value_counts()) == 1:
            return Node(isSingle = True, label = y.iloc[0])

        # 特征为空，T为单节点树，将数据集中实例数最大的类Ck作为该结点的类标记，返回T
        if len(features) == 0:
            return Node(isSingle = True, label = y.value_counts().sort_values(ascending = False).index[0])

        # 计算features中各特征对数据集的信息增益，选择信息增益最大的特征最大的特征Ag
        Ag_index, max_info_gain = self.maxInfoGain(np.array(data))
        Ag = features[Ag_index]

        # Ag的信息增益小于阈值，T为单节点树，将数据集中实例数最大的类Ck作为该结点的类标记，返回T
        if max_info_gain < self.eta:
            return Node(isSingle = True, label = y.value_counts().sort_values(ascending = False).index[0])

        # Ag为根节点的子树
        Ag_tree = Node(isSingle = False, feature_index = Ag_index, feature = Ag)
        # Ag的所有值
        Ag_values = data[Ag].value_counts().index
        # 根据Ag=ai划分数据集
        for value in Ag_values:
            sub_data = data.loc[data[Ag] == value].drop([Ag], axis = 1)    # A-{Ag}
            # 递归得到子树
            sub_tree = self.ID3(sub_data)
            Ag_tree.addSubTree(value, sub_tree)

        return Ag_tree

    # 生成决策树
    def fit(self, data):
        self.root = self.ID3(data)

    # 预测
    def predict(self, node, features):
        # 递归出口
        if node.isSingle is True:
            return node.label
        return self.predict(node.subTree[features[node.feature_index]], features)    # 进入子树搜索


dt = DTree()
dt.fit(df)    # 生成决策树
print("决策树结构：\n", dt.root, '\n')

for each in data:
    print("年龄：{:10s}有工作：{:10s}有自己的房子：{:10s}信贷情况：{:10s}类别：{:10s}预测：{:10s}"
          .format(each[0], each[1], each[2], each[3], each[4], dt.predict(dt.root, each[:-1])))
