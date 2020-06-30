from numpy import *

def loadDataSet():
    dataMat = [];labelMat=[]
    fr = open("logRegtestSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(intX):
    return 1.0/(1 + exp(-intX))
# 梯度上升算法用来求函数的最大值，而梯度下降算法用来求函数的最小值。
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1)) #Theta θ
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # m * 3  3 * 1 [1 1 1]^
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
dataMat,labelMat = loadDataSet()
# print(gradAscent(dataMat,labelMat))
weights = gradAscent(dataMat,labelMat)
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s = 30,c = 'red',marker='s')
    ax.scatter(xcord2,ycord2, s = 30,c = 'green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
print(weights)
plotBestFit(weights)


# 随机梯度上升算法
def stocGradAscent0(dataMatIn,classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatIn[i] * weights)  # m * 3  3 * 1 [1 1 1]^
        error = (classLabels[i] - h)