import math

# 计算距离
def lp(x, y, p):
    sum = 0
    for i in range(len(x)):
        sum += math.pow(abs(x[i] - y[i]), p)
    return math.pow(sum, 1.0 / p)

x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

for i in range(1, 5):
    l = {
         'l{}([1, 1], {})'.format(i, each):
         lp(x1, each, i) for each in [x2, x3]
        }
    print(min(l.items(), key = lambda x:x[1]))


from sklearn import datasets, neighbors

iris = datasets.load_iris()
X = iris.data
y = iris.target

knn = neighbors.KNeighborsClassifier()
knn.fit(X, y)
predict = knn.predict(X)

accuracy = (y == predict).astype(int).mean()
print(accuracy)