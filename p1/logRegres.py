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
#weights = gradAscent(dataMat,labelMat)
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    #weights = wei.getA()
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
#print(weights)
#plotBestFit(weights)


# 随机梯度上升算法
def stocGradAscent0(dataMatIn,classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatIn[i] * weights)  # m * 3  3 * 1 [1 1 1]^
        error = (classLabels[i] - h)
        weights = weights + alpha * error * dataMatIn[i]
    return weights
#weights = stocGradAscent0(array(dataMat),labelMat)
#print(weights)
#plotBestFit(array(weights))

# 改进的随机梯度下降算法
def stocGradAscent1(dataMatIn,classLabels,numIter=150):
    m, n = shape(dataMatIn)
    
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex] * weights))  # m * 3  3 * 1 [1 1 1]^ 随机选取样本来更新回归系数 减少周期性的波动
            #print(dataMatIn[randIndex])
            #print(weights)
            #print(dataMatIn[randIndex] * weights)
            error = (classLabels[randIndex] - h)
            weights = weights + alpha * error * dataMatIn[randIndex]
            #del(dataMatIn[randIndex])
            delete(dataMatIn, randIndex)
    return weights
#weights = stocGradAscent1(array(dataMat),labelMat)
#print(weights)
#plotBestFit(array(weights))

# logistic回归分类函数
def classifyVector(intx,weights):
    prob = sigmoid(sum(intx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
        

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0.0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(trainingSet), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is :%f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is :%f" % (numTests,errorSum/float(numTests)))
multiTest()