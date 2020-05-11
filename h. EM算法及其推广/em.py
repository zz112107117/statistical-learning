import numpy as np
import random
import math


# 数据集
def createData(alpha0, mu0, sigma0, alpha1, mu1, sigma1):
    length = 1000    # 数据集长度
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))    # 高斯分布1
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))    # 高斯分布2

    data = []    # list
    data.extend(data0)
    data.extend(data1)
    random.shuffle(data)    # 打乱数据集

    return np.array(data)    # 混合两个高斯分布的数据


# 高斯混合模型参数估计的EM算法
class GMMEM:
    def __init__(self, alpha0, mu0, sigma0, alpha1, mu1, sigma1, maxIter = 500):
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.alpha1 = alpha1
        self.mu1 = mu1
        self.sigma1 = sigma1

        self.maxIter = maxIter

    # 根据高斯分布密度计算值
    def calcGauss(self, data, mu, sigma):
        part1 = 1 / (math.sqrt(2 * math.pi) * sigma)
        part2 = np.exp(-1 * (data - mu) * (data - mu) / (2 * (sigma ** 2)))
        return part1 * part2    # 向量形式

    # E步，计算分模型k对观测数据yj的响应度
    def Estep(self, data):
        gamma0 = self.alpha0 * self.calcGauss(data, self.mu0, self.sigma0)    # 模型0对观测数据yj的响应度的分子
        gamma1 = self.alpha1 * self.calcGauss(data, self.mu1, self.sigma1)    # 模型1对观测数据yj的响应度的分子
        sum = gamma0 + gamma1    # 分母
        gamma0 = gamma0 / sum    # 模型0对观测数据yj的响应度
        gamma1 = gamma1 / sum    # 模型1对观测数据yj的响应度
        return gamma0, gamma1    # 向量形式

    # M步，计算新一轮迭代的模型参数
    def Mstep(self, data, gamma0, gamma1):
        # 更新mu
        self.mu0 = np.dot(gamma0, data) / np.sum(gamma0)
        self.mu1 = np.dot(gamma1, data) / np.sum(gamma1)
        # 更新sigma
        self.sigma0 = math.sqrt(np.dot(gamma0, (data - self.mu0) ** 2) / np.sum(gamma0))
        self.sigma1 = math.sqrt(np.dot(gamma1, (data - self.mu1) ** 2) / np.sum(gamma1))
        # 更新alpha
        self.alpha0 = np.sum(gamma0) / len(gamma0)
        self.alpha1 = np.sum(gamma1) / len(gamma1)

    # 参数估计
    def fit(self, data):
        step = 0
        while step < self.maxIter:
            step += 1
            # E步
            gamma0, gamma1 = self.Estep(data)
            # M步
            self.Mstep(data, gamma0, gamma1)
        print('\npredict:')
        print('alpha0: %.1f, mu0: %.1f, sigmod0: %.1f;    alpha1: %.1f, mu1: %.1f, sigmod1: %.1f'
              % (self.alpha0, self.mu0, self.sigma0, self.alpha1, self.mu1, self.sigma1))


# 生成数据集
alpha0 = 0.3
mu0 = -2
sigma0 = 0.5
alpha1 = 0.7
mu1 = 0.5
sigma1 = 1
print('set:')
print('alpha0: %.1f, mu0: %.1f, sigmod0: %.1f;    alpha1: %.1f, mu1: %.1f, sigmod1: %.1f'
      % (alpha0, mu0, sigma0, alpha1, mu1, sigma1))
data = createData(alpha0, mu0, sigma0, alpha1, mu1, sigma1)

# 参数的初始值
alpha0 = 0.5
mu0 = 0
sigma0 = 1
alpha1 = 0.5
mu1 = 1
sigma1 = 1

gmmem = GMMEM(alpha0, mu0, sigma0, alpha1, mu1, sigma1)
gmmem.fit(data)    # 参数估计