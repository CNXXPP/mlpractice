from math import log
import operator
#计算数据集的香农熵

def calcShannonEnt(dataset):
    m = len(dataset)
    labelCnt = {}
    for featVec in dataset:
        currLabel = featVec[-1]
        if currLabel not in labelCnt:
            labelCnt[currLabel] = 0
        labelCnt[currLabel] += 1
    shannonEnt = 0.0
    for key in labelCnt:
        prob = float(labelCnt[key]/m)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataset():
    dateSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    label = ['no surfacing', 'flippers']
    return  dateSet, label

dateSet, label = createDataset()
#dateSet[0][-1] = 'else' 增加分类 熵增加 代表混乱度
#print(calcShannonEnt(dateSet))

#按照给定特征划分数据集
#三个输入参数:待划分的数据集、划分数据集的特征、需要返回 的特征的值
def splitDataSet(datasSet, axis, value):
    retDataSet = []
    for featVec in dateSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return  retDataSet

#print(splitDataSet(dateSet,0,1))

#选择最好的数据集划分方式

def chooseBestFeatToSplit(dataset):
    featIdx = len(dateSet[0]) - 1
    baseEntropy = calcShannonEnt(dateSet)
    bestInfoGain = 0.0; bestFeat = -1
    for i in range(featIdx):
        featList = [example[i] for example in dateSet] # 数据集第i列数据
        #print(featList)
        uqVals = set(featList)
        newEntropy = 0.0
        for val in uqVals:
            subDataset = splitDataSet(dateSet, i, val)
            prob = len(subDataset) / float(len(dateSet))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat

print(chooseBestFeatToSplit(dateSet))

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatToSplit(dataSet)
    if len(labels) == 0:
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

mytree = createTree(dateSet,label)
print(mytree)
