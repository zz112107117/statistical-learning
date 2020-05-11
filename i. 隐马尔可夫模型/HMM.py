import numpy as np


# 加载文本
def loadText(file_name, train = True):
    text = []
    # 打开文件
    with open(file_name, encoding = "utf-8") as f:
        # 按行读取
        for line in f.readlines():
            # 训练文本按行分词
            if train == True:
                line = line.strip().split()
            # 测试文本
            else:
                line = line.strip()
            text.append(line)
    return text


class HMM:
    def __init__(self, statu_dict):
        self.statu_dict = statu_dict    # 状态集合
        self.statu_num = len(self.statu_dict)    # 状态数

        self.PI = np.zeros(self.statu_num)    # 初始状态概率向量
        self.A = np.zeros((self.statu_num, self.statu_num))    # 状态转移概率矩阵
        self.B = np.zeros((self.statu_num, 65536))    # 观测概率矩阵

    # HMM参数估计
    def fit(self, train_text):
        # 按行读取训练文本
        for line in train_text:
            statu_seq = []    # 每行的状态序列
            # 一行中的每个词语
            for i in range(len(line)):
                # 单字词语
                if len(line[i]) == 1:
                    statu_word = 'S'    # 每个词语的状态
                # 'B'开头，'E'结尾
                else:
                    statu_word = 'B' + 'M' * (len(line[i]) - 2) + 'E'    # 每个词语的状态
                # 初始状态
                if i == 0:
                    self.PI[self.statu_dict[statu_word[0]]] += 1
                # 统计观测频度
                for j in range(len(statu_word)):
                    self.B[self.statu_dict[statu_word[j]]][ord(line[i][j])] += 1
                statu_seq.extend(statu_word)
            # 统计状态转移频度
            for i in range(1, len(statu_seq)):
                self.A[self.statu_dict[statu_seq[i - 1]]][self.statu_dict[statu_seq[i]]] += 1
        # 监督学习：极大似然
        # 估计初始状态概率向量
        sum = np.sum(self.PI)
        for i in range(len(self.PI)):
            if self.PI[i] == 0:
                self.PI[i] = -3.14e+100
            else:
                self.PI[i] = np.log(self.PI[i] / sum)
        # 估计状态转移概率矩阵
        for i in range(len(self.A)):
            sum = np.sum(self.A[i])
            for j in range(len(self.A[i])):
                if self.A[i][j] == 0:
                    self.A[i][j] = -3.14e+100
                else:
                    self.A[i][j] = np.log(self.A[i][j] / sum)
        # 估计观测概率矩阵
        for i in range(len(self.B)):
            sum = np.sum(self.B[i])
            for j in range(len(self.B[i])):
                if self.B[i][j] == 0:
                    self.B[i][j] = -3.14e+100
                else:
                    self.B[i][j] = np.log(self.B[i][j] / sum)

    # 预测
    def viterbi(self, test_text):
        text = []    # 分词后的文本
        # 按行读取
        for line in test_text:
            # 初始化
            delta = [[0 for _ in range(4)] for _ in range(len(line))]    # δ
            psi = [[0 for _ in range(4)] for _ in range(len(line))]    # ψ
            for i in range(self.statu_num):
                delta[0][i] = self.PI[i] + self.B[i][ord(line[0])]    # 取对数：乘法->加法
            # 递推
            for t in range(1, len(line)):
                for i in range(self.statu_num):
                    tmp_delta = [0] * self.statu_num
                    for j in range(self.statu_num):
                        tmp_delta[j] = delta[t - 1][j] + self.A[j][i]    # 取对数：乘法->加法
                    max_delta = max(tmp_delta)    # tmp_delta中的最大值
                    max_delta_index = tmp_delta.index(max_delta)    # tmp_delta中最大值对应的索引

                    delta[t][i] = max_delta + self.B[i][ord(line[t])]    # 取对数：乘法->加法
                    psi[t][i] = max_delta_index
            # 终止
            i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
            sequence = []    # 状态序列
            sequence.append(i_opt)
            # 最优路径回溯
            for t in range(len(line) - 1, 0, -1):
                i_opt = psi[t][i_opt]
                sequence.append(i_opt)
            sequence.reverse()    # 反转->从前到后

            res_text = ""
            # 一行中到每个字（不是词语）
            for i in range(len(line)):
                res_text += line[i]    # 添加字
                # 字不是行的最后一个字
                # 字对应的状态是2（E->词语结尾）或3（S->单字词语）
                # 在字的后面添加分隔符
                if (i != len(line) - 1) and (sequence[i] == 2 or sequence[i] == 3):
                    res_text += " / "
            text.append(res_text)
        # 打印分词后的文本
        for line in text:
            print(line)


# 状态集合
statu_dict = {'B': 0,    # 词语开头
              'M': 1,    # 词语中间
              'E': 2,    # 词语结尾
              'S': 3}    # 单字词语
# 加载数据集
train_text = loadText('TrainText.txt', train = True)
test_text = loadText('TestText.txt', train = False)

hmm = HMM(statu_dict)
hmm.fit(train_text)    # 训练模型
hmm.viterbi(test_text)    # 打印分词结果